"""DSA indexer profile generator.

The compressed (C4) paged decode ladder is raced: every candidate launch
config is planned through the production policy override, checked against the
paged logits reference on the stack's own inputs, and timed in place inside
the DSV4 attention chain (indexer -> compressed sparse MLA -> WO projection)
with the runtime's all-auto config as the baseline. The remaining production
shapes are single-candidate qualification cases timed on their benchmark
harnesses, so the profile keeps covering them without changing their route.
"""

from __future__ import annotations

import gc
import io
import json
import math
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, redirect_stdout

from b12x.attention.dsa_indexer._policy import (
    INDEXER_FUSED_CTAS_WAVES,
    DsaIndexerConfig,
    DsaIndexerQuery,
)
from b12x.policy.components import DSA_INDEXER
from b12x.policy.generation.contracts import GenerationContext
from b12x.policy.generation.crash_guard import (
    CRASHED_ERROR,
    clear_inflight,
    load_crashed,
    mark_inflight,
)
from b12x.policy.generation.dsa_indexer_corpus import (
    DSA_INDEXER_RACED_GEOMETRIES,
    dsa_indexer_cases,
    qualification_dsa_indexer_cases,
    raced_dsa_indexer_cases,
)
from b12x.policy.generation.sweep import (
    DiscreteSweepGenerator,
    SweepCandidate,
    SweepCase,
    SweepMeasurement,
)

from .gpu_workers import _l2_flush_fn
from .layer_stack import (
    CONTEXT_LAYERS,
    CONTEXT_MARGIN,
    _confirm_winner,
    _context_race,
    _keep_race,
)

INDEXER_QUERY_FIELDS = (
    "source_layout",
    "mode",
    "dtype",
    "kv_dtype",
    "num_q_heads",
    "num_idx_heads",
    "max_q_rows",
    "max_k_rows",
    "max_page_table_width",
    "top_k",
    "page_size",
    "score_mode",
    "shared_page_table",
)
INDEXER_RANGE_FIELDS = frozenset({"max_q_rows", "max_k_rows", "max_page_table_width"})
INDEXER_WIDTH_BOUNDS = (0, 1 << 20)
INDEXER_ROWS_BOUNDS = (1, 1 << 16)
AUTO_CONFIG: dict[str, object] = DsaIndexerConfig().to_dict()
# Tiled supertile widths in K rows; 0 is the capacity-aware default.
SUPERTILE_ROWS = (0, 8_192, 16_384, 65_536)
FORCE_LAST_CTA_MERGE = 1 << 30
# Sorted top-k logits of the selected slots must match the reference top-k
# within this fraction of the row's largest logit (fp32 reference versus the
# kernel's accumulation order; ties at the tail are value-equal either way).
SCORE_TOLERANCE = 0.02


def _tiled_config(supertile_k: int, fold: str) -> dict[str, object]:
    return {
        **AUTO_CONFIG,
        "route": "paged_tiled",
        "supertile_k": int(supertile_k),
        "two_level_fold": fold,
    }


def _fused_config(ctas: int, merge: int) -> dict[str, object]:
    return {
        **AUTO_CONFIG,
        "route": "paged_fused",
        "fused_ctas_per_group": int(ctas),
        "fused_merge_threshold": int(merge),
    }


def candidate_configs(
    query: Mapping[str, object],
    *,
    sm_count: int,
    compute_capability: tuple[int, int] | None,
) -> tuple[dict[str, object], ...]:
    """Launch configs raced for one paged decode query on one device."""
    from b12x.attention.dsa_indexer.fused_indexer import resolve_fused_indexer_path

    rows = max(1, int(query["max_q_rows"]))
    width_rows = int(query["max_page_table_width"]) * int(query["page_size"])
    configs: list[dict[str, object]] = [dict(AUTO_CONFIG)]
    fused_ok = resolve_fused_indexer_path(
        topk=int(query["top_k"]),
        num_rows=rows,
        width=width_rows,
        num_heads=int(query["num_q_heads"]),
        compute_capability=compute_capability,
    )
    if fused_ok and str(query["mode"]) == "decode" and not query["shared_page_table"]:
        wave = max(1, int(sm_count) // rows)
        budget = (INDEXER_FUSED_CTAS_WAVES * int(sm_count)) // rows
        for ctas in sorted({0, max(1, wave // 2), min(2 * wave, budget)}):
            for merge in (-1, 0, FORCE_LAST_CTA_MERGE):
                configs.append(_fused_config(ctas, merge))
    aligned_width = -(-width_rows // 512) * 512
    for supertile in SUPERTILE_ROWS:
        if supertile > aligned_width:
            continue
        for fold in ("auto", "off"):
            configs.append(_tiled_config(supertile, fold))
    unique: list[dict[str, object]] = []
    seen: set[tuple[tuple[str, object], ...]] = set()
    for config in configs:
        key = tuple(sorted(config.items()))
        if key not in seen:
            seen.add(key)
            unique.append(config)
    return tuple(unique)


def _score_error(reference, selected) -> float:
    """Largest sorted-top-k logit gap, relative to each row's peak logit."""
    import torch

    peak = reference[:, :1].abs().clamp_min(1e-6)
    gap = (reference - selected).abs()
    gap = torch.where(torch.isfinite(gap), gap, torch.full_like(gap, math.inf))
    return float((gap / peak).max().item())


def _heuristic_chunk_cap(*, heads: int, rows: int, device) -> int:
    from b12x.attention.compressed_sparse_mla._policy import SparseMlaQuery, _heuristic
    from b12x.policy import detect_device

    query = SparseMlaQuery(
        layout="compressed_dsv4",
        mode="decode",
        q_dtype="bfloat16",
        kv_dtype="float8_e4m3fn",
        num_q_heads=heads,
        qk_head_dim=512,
        v_head_dim=448,
        swa_width=128,
        swa_page_size=64,
        indexed_width=512,
        indexed_page_size=64,
        query_rows=rows,
    )
    return int(
        _heuristic(query, detect_device(str(device)).identity).max_chunks_per_row
    )


class _DsaIndexerSession(AbstractContextManager["_DsaIndexerSession"]):
    def __init__(self, context: GenerationContext) -> None:
        self._context = context
        self._candidate_cache: dict[str, tuple[SweepCandidate, ...]] = {}

    def __enter__(self) -> "_DsaIndexerSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        try:
            import torch

            gc.collect()
            torch.cuda.synchronize(self._context.device_ordinal)
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - cleanup must preserve the root error
            pass
        return None

    def candidates(self, case: SweepCase) -> tuple[SweepCandidate, ...]:
        cached = self._candidate_cache.get(case.case_id)
        if cached is not None:
            return cached
        if not bool(case.metadata.get("raced", False)):
            result = (SweepCandidate.create(AUTO_CONFIG),)
        else:
            import torch

            props = torch.cuda.get_device_properties(self._context.device_ordinal)
            result = tuple(
                SweepCandidate.create(config)
                for config in candidate_configs(
                    case.query,
                    sm_count=int(props.multi_processor_count),
                    compute_capability=(int(props.major), int(props.minor)),
                )
            )
        self._candidate_cache[case.case_id] = result
        return result

    def measure(
        self,
        case: SweepCase,
        candidates: tuple[SweepCandidate, ...],
    ) -> tuple[SweepMeasurement, ...]:
        if not bool(case.metadata.get("raced", False)):
            if len(candidates) != 1:
                raise ValueError("qualification cases carry exactly one candidate")
            return (_measure_qualification(case, candidates[0], self._context),)
        return self._measure_raced(case, candidates)

    def _measure_raced(
        self,
        case: SweepCase,
        candidates: tuple[SweepCandidate, ...],
    ) -> tuple[SweepMeasurement, ...]:
        import torch

        from .dsv4_layer_stack import _DsvLayerContext

        context = self._context
        settings = context.settings
        work_dir = context.work_dir
        device = torch.device("cuda", context.device_ordinal)
        query = case.query
        rows = int(query["max_q_rows"])
        heads = int(query["num_q_heads"])

        def crashed_all() -> tuple[SweepMeasurement, ...]:
            return tuple(
                SweepMeasurement(
                    candidate=candidate,
                    latency_us=None,
                    correct=False,
                    error=CRASHED_ERROR,
                )
                for candidate in candidates
            )

        crashed = load_crashed(work_dir)
        if (case.case_id, "*") in crashed:
            return crashed_all()
        mark_inflight(work_dir, case.case_id, "*")
        with torch.cuda.device(context.device_ordinal):
            case_seed = settings.seed + int(case.case_id[-8:], 16) % 1_000_003
            stack = _DsvLayerContext(
                heads=heads,
                rows=rows,
                swa_width=128,
                indexed_width=512,
                context_tokens=int(case.metadata["context_tokens"]),
                page_table_width=int(query["max_page_table_width"]),
                device=device,
                generator=torch.Generator(device=device).manual_seed(case_seed + 7919),
                seed=case_seed,
                layers=CONTEXT_LAYERS,
                slot="indexer",
            )
            stack.prepare_attention(
                _heuristic_chunk_cap(heads=heads, rows=rows, device=device)
            )
            torch.cuda.synchronize(device)
            flush = _l2_flush_fn(device, enabled=settings.cold_l2)
            races: dict = {}
            measurements: list[SweepMeasurement] = []
            for candidate in candidates:
                if (case.case_id, candidate.candidate_id) in crashed:
                    measurements.append(
                        SweepMeasurement(
                            candidate=candidate,
                            latency_us=None,
                            correct=False,
                            error=CRASHED_ERROR,
                        )
                    )
                    continue
                mark_inflight(work_dir, case.case_id, candidate.candidate_id)
                try:
                    route = stack.prepare_indexer(
                        candidate.candidate_id, candidate.config.to_dict()
                    )
                    slot = ("indexer", candidate.candidate_id)
                    # One eager pass writes this run's inputs and the
                    # candidate's selection; the reference reads the same
                    # inputs back before the timed replays.
                    stack.run(slot)
                    torch.cuda.synchronize(device)
                    logits, reference = stack.indexer_reference_values(0)
                    selected, in_range = stack.indexer_selected_values(0, logits)
                    error = _score_error(reference, selected)
                    correct = in_range and error <= SCORE_TOLERANCE
                    del logits, reference, selected
                    race = _context_race(
                        context=stack,
                        slot=slot,
                        settings=settings,
                        device=device,
                        flush=flush,
                    )
                    correct = correct and race.finite
                    if correct:
                        _keep_race(races, candidate, race, AUTO_CONFIG)
                    measurements.append(
                        SweepMeasurement(
                            candidate=candidate,
                            latency_us=race.op_us,
                            correct=correct,
                            metrics={
                                "route": route,
                                "score_error": error,
                                "in_place": True,
                                "stack_us": race.stack_us,
                                "context_layers": stack.layers,
                                "context_repetitions": race.repetitions,
                                "context_tokens": int(case.metadata["context_tokens"]),
                            },
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - failed plans survive
                    measurements.append(
                        SweepMeasurement(
                            candidate=candidate,
                            latency_us=None,
                            correct=False,
                            error=f"{type(exc).__name__}: {exc}"[:400],
                        )
                    )
            confirmed = _confirm_winner(
                measurements,
                races,
                baseline_config=AUTO_CONFIG,
                on_sample=lambda candidate_id: mark_inflight(
                    work_dir, case.case_id, candidate_id
                ),
            )
            clear_inflight(work_dir)
            del stack, races
            gc.collect()
            torch.cuda.empty_cache()
            return tuple(confirmed)


def _measure_qualification(
    case: SweepCase,
    candidate: SweepCandidate,
    context: GenerationContext,
) -> SweepMeasurement:
    """Time one production shape on its benchmark harness (pass/fail gated)."""
    import torch

    probe = dict(case.metadata["probe"])
    kind = str(probe["kind"])
    device = torch.device("cuda", context.device_ordinal)
    settings = context.settings
    flush = _l2_flush_fn(device, enabled=settings.cold_l2)
    seed = settings.seed + int(case.case_id[-8:], 16) % 1_000_003
    try:
        with torch.cuda.device(context.device_ordinal):
            if kind == "dsa":
                latency, metrics = _dsa_probe(probe, settings, device, flush, seed)
                correct = True
            elif kind == "msa":
                latency, correct, metrics = _msa_probe(
                    probe, context, device, flush, seed
                )
            else:
                raise ValueError(f"unknown indexer probe kind {kind!r}")
    except Exception as exc:  # noqa: BLE001 - a broken harness fails the case
        return SweepMeasurement(
            candidate=candidate,
            latency_us=None,
            correct=False,
            error=f"{type(exc).__name__}: {exc}"[:400],
        )
    finally:
        gc.collect()
        torch.cuda.empty_cache()
    return SweepMeasurement(
        candidate=candidate,
        latency_us=float(latency),
        correct=bool(correct),
        metrics={"qualification": True, **metrics},
    )


def _dsa_probe(probe, settings, device, flush, seed) -> tuple[float, dict]:
    from benchmarks.benchmark_dsa_indexer import (
        GLMNSAConfig,
        _run_decode_case,
        _run_extend_case,
    )

    mode = str(probe["mode"])
    rows = int(probe["rows"])
    cache_len = int(probe["cache_len"])
    cfg = GLMNSAConfig(num_heads=int(probe["heads"]))
    replays = settings.groups * settings.repetitions
    captured = io.StringIO()
    with redirect_stdout(captured):
        if mode == "decode":
            _run_decode_case(
                cfg=cfg,
                q_rows=rows,
                cache_len=cache_len,
                width=cache_len,
                topk=int(probe["top_k"]),
                warmup=settings.warmup,
                replays=replays,
                seed=seed,
                device=device,
                pool_factor=2,
                l2_flush=flush,
            )
        else:
            _run_extend_case(
                cfg=cfg,
                batch=rows,
                q_len=128,
                cache_len=cache_len,
                width=cache_len,
                topk=int(probe["top_k"]),
                warmup=settings.warmup,
                replays=replays,
                seed=seed,
                device=device,
                pool_factor=2,
                l2_flush=flush,
            )
    records = [
        json.loads(line)
        for line in captured.getvalue().splitlines()
        if line.strip().startswith("{")
    ]
    if len(records) != 1:
        raise RuntimeError("DSA benchmark did not emit one timing record")
    record = records[0]
    latency = record["replay_median_us"] if mode == "decode" else record["median_us"]
    return float(latency), {
        "mode": mode,
        "query_rows": rows if mode == "decode" else rows * 128,
        "num_heads": int(probe["heads"]),
        "top_k": int(probe["top_k"]),
    }


def _msa_probe(probe, context, device, flush, seed) -> tuple[float, bool, dict]:
    import torch
    from benchmarks.benchmark_msa_indexer import _make_decode_case, _make_prefill_case

    from b12x.attention.dsa_indexer._impl import (
        msa_q2k_indices_decode,
        msa_q2k_indices_prefill,
    )
    from b12x.attention.dsa_indexer.msa_reference import (
        MSA_BLOCK_TOKENS,
        MSA_TOPK_BLOCKS,
        msa_q2k_indices_reference,
    )

    from .qualification import _timed_exact_graph_measurement

    mode = str(probe["mode"])
    rows = int(probe["rows"])
    heads = int(probe["heads"])
    width = int(probe["width"])
    output = torch.empty(
        (heads, rows, MSA_TOPK_BLOCKS), dtype=torch.int32, device=device
    )
    if mode == "decode":
        q_fp8, q_scale, index_k_cache, metadata = _make_decode_case(
            rows=rows, heads=heads, ctx_tokens=width, seed=seed, device=device
        )
        expected = msa_q2k_indices_reference(
            q_fp8=q_fp8,
            q_scale=q_scale,
            index_k_cache=index_k_cache,
            real_page_table=metadata.real_page_table,
            cache_seqlens_int32=metadata.cache_seqlens_int32,
            query_positions=metadata.cache_seqlens_int32 - 1,
        )

        def run() -> None:
            msa_q2k_indices_decode(
                q_fp8=q_fp8,
                q_scale=q_scale,
                index_k_cache=index_k_cache,
                metadata=metadata,
                out_indices=output,
            )

    else:
        q_fp8, q_scale, kv_fp8, metadata = _make_prefill_case(
            rows=rows, heads=heads, k_rows=width, seed=seed, device=device
        )
        expected = msa_q2k_indices_reference(
            q_fp8=q_fp8,
            q_scale=q_scale,
            kv_fp8=kv_fp8,
            k_start=metadata.k_start,
            k_end=metadata.k_end,
            query_positions=metadata.k_end - 1,
            block_base=torch.div(
                metadata.k_start, MSA_BLOCK_TOKENS, rounding_mode="floor"
            ),
        )

        def run() -> None:
            msa_q2k_indices_prefill(
                q_fp8=q_fp8,
                q_scale=q_scale,
                kv_fp8=kv_fp8,
                metadata=metadata,
                out_indices=output,
            )

    measured = _timed_exact_graph_measurement(
        context=context,
        label=str(probe.get("label", "msa")),
        run=run,
        output=output,
        expected=expected,
        flush=flush,
    )
    return (
        float(measured.latency_us),
        bool(measured.correct),
        {"mode": mode, "query_rows": rows, **dict(measured.metrics)},
    )


class DsaIndexerBenchmarkFactory:
    """Race paged indexer launch configs inside the DSV4 attention chain."""

    def __call__(self, group_id, cases, context):
        del group_id, cases
        return _DsaIndexerSession(context)


class DsaIndexerGenerator(DiscreteSweepGenerator):
    """Generate the DSA indexer component profile."""

    def __init__(
        self,
        *,
        benchmark_factory=None,
        cases: Sequence[SweepCase] | None = None,
    ) -> None:
        super().__init__(
            component_id=DSA_INDEXER,
            query_schema_version=2,
            config_schema_version=2,
            query_fields=INDEXER_QUERY_FIELDS,
            range_fields=INDEXER_RANGE_FIELDS,
            cases=dsa_indexer_cases() if cases is None else cases,
            benchmark_factory=benchmark_factory or DsaIndexerBenchmarkFactory(),
            coverage={
                "raced_geometries": len(DSA_INDEXER_RACED_GEOMETRIES),
                "raced_cases": len(raced_dsa_indexer_cases()),
                "qualification_cases": len(qualification_dsa_indexer_cases()),
            },
            # Serving widths follow max_model_len and decode rows follow the
            # batch, so the nearest measured anchor covers the whole domain
            # instead of falling back to the heuristic between anchors.
            nearest_range_bounds={
                "max_page_table_width": INDEXER_WIDTH_BOUNDS,
                "max_q_rows": INDEXER_ROWS_BOUNDS,
            },
            baseline_margin=CONTEXT_MARGIN,
            candidate_contract_version=1,
        )

    def baseline_config(self, case, context):
        del case, context
        return dict(AUTO_CONFIG)

    def reviewed_queries(self) -> tuple[DsaIndexerQuery, ...]:
        seen: dict[tuple[object, ...], DsaIndexerQuery] = {}
        for case in self._cases:
            query = DsaIndexerQuery(**dict(case.query))
            seen.setdefault(tuple(sorted(case.query.items())), query)
        return tuple(seen.values())


__all__ = [
    "AUTO_CONFIG",
    "INDEXER_ROWS_BOUNDS",
    "INDEXER_WIDTH_BOUNDS",
    "DsaIndexerBenchmarkFactory",
    "DsaIndexerGenerator",
    "candidate_configs",
]
