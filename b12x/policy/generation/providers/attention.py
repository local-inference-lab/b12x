"""Built-in attention component generators and reviewed corpora."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from b12x.policy.components import (
    COMPRESSED_SPARSE_MLA_ATTENTION,
    GDN_ATTENTION,
    GQA_ATTENTION,
    MLA_ATTENTION,
)
from b12x.policy.generation.attention_corpus import (
    COMMON_PREFILL_TOKEN_CAPACITIES,
    COMMON_SEQUENCE_CAPACITIES,
    GDN_GEOMETRIES,
    GDN_STATE_INDEX_COLUMNS,
    GQA_GEOMETRIES,
    MLA_GEOMETRIES,
    SPARSE_MLA_GEOMETRIES,
    gdn_cases,
    gqa_cases,
    mla_cases,
    qsa_cases,
    sparse_mla_cases,
)
from b12x.policy.generation.contracts import (
    GenerationContext,
)
from b12x.policy.generation.measured import (
    GpuProbeMeasurement,
    MeasuredPolicyGenerator,
)
from b12x.policy.generation.sweep import (
    DiscreteSweepGenerator,
    SweepBenchmarkFactory,
    SweepCase,
)

from .layer_stack import CONTEXT_MARGIN
from .gpu_workers import (
    GdnBenchmarkFactory,
    GqaBenchmarkFactory,
    MlaBenchmarkFactory,
    SparseMlaBenchmarkFactory,
)


class _MissingAttentionBenchmarkFactory:
    def __init__(self, component_id: str) -> None:
        self._component_id = component_id

    def __call__(self, group_id, cases, context):
        del group_id, cases, context
        raise RuntimeError(
            f"{self._component_id} has a reviewed corpus and reducer, but its "
            "production GPU measurement worker is not registered"
        )


class _AttentionGenerator(DiscreteSweepGenerator):
    def __init__(
        self,
        *,
        component_id: str,
        query_fields: tuple[str, ...],
        range_fields: frozenset[str],
        cases: Sequence[SweepCase],
        corpus_name: str,
        geometry_count: int,
        benchmark_factory: SweepBenchmarkFactory | None,
        query_schema_version: int = 1,
        config_schema_version: int = 1,
        nearest_range_bounds: Mapping[str, tuple[int, int]] | None = None,
        baseline_margin: float = 0.0,
        candidate_contract_version: int = 1,
    ) -> None:
        del corpus_name
        super().__init__(
            component_id=component_id,
            query_schema_version=query_schema_version,
            config_schema_version=config_schema_version,
            query_fields=query_fields,
            range_fields=range_fields,
            cases=cases,
            benchmark_factory=(
                benchmark_factory
                if benchmark_factory is not None
                else _MissingAttentionBenchmarkFactory(component_id)
            ),
            coverage={
                "model_geometries": geometry_count,
            },
            nearest_range_bounds=nearest_range_bounds,
            baseline_margin=baseline_margin,
            candidate_contract_version=candidate_contract_version,
        )


class GdnAttentionGenerator(_AttentionGenerator):
    """Generate the recurrent Qwen GDN attention component profile."""

    def __init__(
        self,
        *,
        benchmark_factory: SweepBenchmarkFactory | None = None,
        cases: Sequence[SweepCase] | None = None,
    ) -> None:
        super().__init__(
            component_id=GDN_ATTENTION,
            query_fields=(
                "gate_activation",
                "qk_l2norm",
                "state_dtype",
                "key_heads",
                "value_heads",
                "max_seqs",
                "max_tokens",
                "state_index_columns",
            ),
            range_fields=frozenset(
                {
                    "max_seqs",
                    "max_tokens",
                    "state_index_columns",
                }
            ),
            cases=gdn_cases() if cases is None else cases,
            corpus_name="gdn",
            geometry_count=len(GDN_GEOMETRIES),
            benchmark_factory=benchmark_factory or GdnBenchmarkFactory(),
            config_schema_version=2,
            nearest_range_bounds={
                "max_seqs": (1, max(COMMON_SEQUENCE_CAPACITIES)),
                "max_tokens": (
                    1,
                    max(COMMON_SEQUENCE_CAPACITIES) * max(GDN_STATE_INDEX_COLUMNS),
                ),
                "state_index_columns": (1, max(GDN_STATE_INDEX_COLUMNS)),
            },
        )


class GqaAttentionGenerator(_AttentionGenerator):
    """Generate the paged GQA attention component profile."""

    def __init__(
        self,
        *,
        benchmark_factory: SweepBenchmarkFactory | None = None,
        cases: Sequence[SweepCase] | None = None,
    ) -> None:
        super().__init__(
            component_id=GQA_ATTENTION,
            query_fields=(
                "mode",
                "q_dtype",
                "kv_dtype",
                "q_heads",
                "kv_heads",
                "head_dim_qk",
                "head_dim_vo",
                "page_size",
                "kv_cache_layout",
                "batch_size",
                "query_len",
                "cache_tokens",
                "window_left",
                "requested_graph_ctas_per_sm",
                "force_split_kv",
            ),
            range_fields=frozenset({"batch_size", "query_len", "cache_tokens"}),
            cases=gqa_cases() if cases is None else cases,
            corpus_name="gqa",
            geometry_count=len(GQA_GEOMETRIES),
            benchmark_factory=benchmark_factory or GqaBenchmarkFactory(),
            query_schema_version=2,
            config_schema_version=2,
        )


class _QsaProbe:
    _CASES = (
        ("tp1", 1, 2_048, "bf16", "throughput"),
        ("tp1", 4, 8_192, "fp8_e4m3", "throughput"),
        ("tp2", 1, 8_192, "fp8_e4m3", "throughput"),
        ("tp2", 4, 2_048, "bf16", "throughput"),
        ("tp4", 1, 2_048, "fp8_e4m3", "throughput"),
        ("tp4", 4, 8_192, "bf16", "throughput"),
        *(
            (profile, rows, 8_192, kv_dtype, "prefill")
            for profile in ("tp1", "tp2", "tp4")
            for rows in COMMON_PREFILL_TOKEN_CAPACITIES
            for kv_dtype in ("bf16", "fp8_e4m3")
        ),
    )

    @property
    def case_count(self) -> int:
        return len(self._CASES)

    @property
    def case_ids(self) -> tuple[str, ...]:
        from benchmarks.benchmark_qsa import BenchmarkCase, PROFILES

        return tuple(
            (
                f"{BenchmarkCase(PROFILES[profile], rows, sequence, kind=kind).name}"
                f"-{kv_dtype}"
            )
            for profile, rows, sequence, kv_dtype, kind in self._CASES
        )

    @property
    def description(self) -> str:
        return "production QSA graph qualification across TP and KV dtypes"

    def __call__(
        self,
        context: GenerationContext,
    ) -> tuple[GpuProbeMeasurement, ...]:
        import argparse
        from collections.abc import Mapping

        import torch

        from benchmarks.benchmark_qsa import BenchmarkCase, PROFILES, _run_case
        from b12x.policy import PolicyContext, PolicyMode

        from .gpu_workers import _l2_flush_fn

        device = torch.device("cuda", context.device_ordinal)
        settings = context.settings
        flush = _l2_flush_fn(device, enabled=settings.cold_l2)
        policy = PolicyContext.for_device(device, mode=PolicyMode.HEURISTIC_ONLY)
        measurements = []
        for index, (profile, rows, sequence, kv_dtype, kind) in enumerate(self._CASES):
            case = BenchmarkCase(PROFILES[profile], rows, sequence, kind=kind)
            label = f"{case.name}-{kv_dtype}"
            args = argparse.Namespace(
                seed=settings.seed,
                main_cache_layout="interleaved",
                kv_cache_dtype=kv_dtype,
                warmup=settings.warmup,
                eager_replays=1,
                graph_replays=max(1, settings.groups * settings.repetitions),
            )
            result = _run_case(
                case,
                args=args,
                device=device,
                l2_flush=flush,
                case_index=index,
                policy=policy,
            )
            timing = result["timing"]
            if not isinstance(timing, Mapping):
                raise TypeError("QSA benchmark timing must be an object")
            graph_timing = timing["cuda_graph"]
            if not isinstance(graph_timing, Mapping):
                raise TypeError("QSA graph timing must be an object")
            summary = graph_timing["replay_summary"]
            if not isinstance(summary, Mapping):
                raise TypeError("QSA replay summary must be an object")
            correctness = result["correctness"]
            if not isinstance(correctness, Mapping):
                raise TypeError("QSA correctness must be an object")
            measurements.append(
                GpuProbeMeasurement(
                    label=label,
                    latency_us=float(summary["median_us"]),
                    correct=bool(
                        correctness["graph_finite"]
                        and correctness["graph_nonzero_elements"]
                        and correctness["eager_graph_exact"]
                    ),
                    metrics={
                        "kind": kind,
                        "kv_dtype": kv_dtype,
                        "rows": rows,
                        "tp_size": PROFILES[profile].tensor_parallel_size,
                    },
                )
            )
        return tuple(measurements)


class QsaAttentionGenerator(MeasuredPolicyGenerator):
    """Measure the only production QSA backend over its serving envelope."""

    def __init__(self) -> None:
        from b12x.attention.qsa._policy import QSA_POLICY, QsaQuery

        super().__init__(
            policy=QSA_POLICY,
            queries=tuple(QsaQuery(**case.query.to_dict()) for case in qsa_cases()),
            encode_config=lambda config: {"backend": config.backend},
            probe=_QsaProbe(),
        )


class MlaAttentionGenerator(_AttentionGenerator):
    """Generate the dense MLA attention component profile."""

    def __init__(
        self,
        *,
        benchmark_factory: SweepBenchmarkFactory | None = None,
        cases: Sequence[SweepCase] | None = None,
    ) -> None:
        super().__init__(
            component_id=MLA_ATTENTION,
            query_fields=(
                "mode",
                "q_dtype",
                "kv_dtype",
                "num_q_heads",
                "qk_head_dim",
                "v_head_dim",
                "page_size",
                "query_rows",
                "max_batch",
                "cache_tokens",
                "physical_record_width",
                "window_size",
                "use_cuda_graph",
            ),
            range_fields=frozenset({"query_rows", "cache_tokens"}),
            cases=mla_cases() if cases is None else cases,
            corpus_name="mla",
            geometry_count=len(MLA_GEOMETRIES),
            benchmark_factory=benchmark_factory or MlaBenchmarkFactory(),
            query_schema_version=2,
        )


class CompressedSparseMlaAttentionGenerator(_AttentionGenerator):
    """Generate the compressed sparse-MLA component profile."""

    def __init__(
        self,
        *,
        benchmark_factory: SweepBenchmarkFactory | None = None,
        cases: Sequence[SweepCase] | None = None,
    ) -> None:
        super().__init__(
            component_id=COMPRESSED_SPARSE_MLA_ATTENTION,
            query_fields=(
                "layout",
                "mode",
                "q_dtype",
                "kv_dtype",
                "num_q_heads",
                "qk_head_dim",
                "v_head_dim",
                "swa_width",
                "swa_page_size",
                "indexed_width",
                "indexed_page_size",
                "query_rows",
            ),
            range_fields=frozenset({"swa_width", "indexed_width", "query_rows"}),
            cases=sparse_mla_cases() if cases is None else cases,
            corpus_name="sparse_mla",
            geometry_count=len(SPARSE_MLA_GEOMETRIES),
            benchmark_factory=benchmark_factory or SparseMlaBenchmarkFactory(),
            baseline_margin=CONTEXT_MARGIN,
            candidate_contract_version=2,
        )

    def baseline_config(self, case, context):
        from b12x.attention.compressed_sparse_mla._policy import (
            SparseMlaQuery,
            _heuristic,
        )

        config = _heuristic(SparseMlaQuery(**dict(case.query)), context.device)
        return {"max_chunks_per_row": int(config.max_chunks_per_row)}


__all__ = [
    "GdnAttentionGenerator",
    "GqaAttentionGenerator",
    "MlaAttentionGenerator",
    "QsaAttentionGenerator",
    "CompressedSparseMlaAttentionGenerator",
]
