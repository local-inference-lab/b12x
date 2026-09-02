"""Measured launch-policy providers for norm and sequence fusions."""

from __future__ import annotations

import gc
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field

from b12x.policy.components import HYPERCONNECTION, MHC, MTP_FEEDBACK
from b12x.policy.generation.attention_corpus import (
    COMMON_PREFILL_TOKEN_CAPACITIES,
    COMMON_SEQUENCE_CAPACITIES,
)
from b12x.policy.generation.sweep import (
    DiscreteSweepGenerator,
    SweepCandidate,
    SweepCase,
    SweepMeasurement,
)

from .gpu_workers import (
    _bounded_repetitions,
    _l2_flush_fn,
    _median_of_group_medians,
)
from .mhc_layer_stack import MhcLayerStack

_NORM_SEQUENCE_TOKEN_CAPACITIES = (
    *COMMON_SEQUENCE_CAPACITIES,
    512,
    *COMMON_PREFILL_TOKEN_CAPACITIES,
)
_ALLOCATOR_COUNTER_KEYS = (
    "allocation.all.allocated",
    "allocation.all.freed",
    "segment.all.allocated",
    "segment.all.freed",
    "num_alloc_retries",
    "num_ooms",
)


def _allocator_counters(device: object) -> dict[str, int]:
    import torch

    stats = torch.cuda.memory_stats(device)
    return {key: int(stats[key]) for key in _ALLOCATOR_COUNTER_KEYS}


def _hyperconnection_cases() -> tuple[SweepCase, ...]:
    return tuple(
        SweepCase.create(
            group_id=f"qwen-flash-next-m{tokens}",
            query={
                "dtype": "bfloat16",
                "max_tokens": tokens,
                "hidden_size": 2_560,
                "streams": 4,
                "lowrank": 320,
            },
        )
        for tokens in _NORM_SEQUENCE_TOKEN_CAPACITIES
    )


def _mtp_feedback_cases() -> tuple[SweepCase, ...]:
    return tuple(
        SweepCase.create(
            group_id=f"qwen-flash-next-m{tokens}",
            query={
                "dtype": "bfloat16",
                "max_tokens": tokens,
                "hidden_size": 2_560,
                "streams": 4,
            },
        )
        for tokens in _NORM_SEQUENCE_TOKEN_CAPACITIES
    )


def _mhc_cases() -> tuple[SweepCase, ...]:
    capacities = tuple(
        sorted(
            {
                *COMMON_SEQUENCE_CAPACITIES,
                24,
                48,
                96,
                192,
                320,
                384,
                512,
                *COMMON_PREFILL_TOKEN_CAPACITIES,
                2_304,
                3_072,
                3_584,
            }
        )
    )
    return tuple(
        SweepCase.create(
            group_id=f"mhc-h{hidden_size}-m{tokens}",
            query={
                "dtype": "bfloat16",
                "max_tokens": tokens,
                "hidden_size": hidden_size,
                "split_k": split_k,
            },
        )
        for hidden_size, split_k in ((4_096, 64), (7_168, 112))
        for tokens in capacities
    )


def _mhc_config(
    *,
    backend: str,
    native_post_pre_backend: str = "decode",
    decode_source_splits: int = 0,
    decode_tile_n: int = 0,
    decode_bf16x2: bool = False,
    decode_partials_per_cta: int = 4,
    decode_finalize_threads: int = 0,
    decode_finalize_ctas: int = 1,
    prefill_block_m: int = 0,
    prefill_tile_n: int = 0,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    stages: int,
    m_warps: int,
    n_warps: int,
    k_splits: int,
) -> dict[str, object]:
    return {
        "backend": backend,
        "native_post_pre_backend": native_post_pre_backend,
        "decode_source_splits": decode_source_splits,
        "decode_tile_n": decode_tile_n,
        "decode_bf16x2": decode_bf16x2,
        "decode_partials_per_cta": decode_partials_per_cta,
        "decode_finalize_threads": decode_finalize_threads,
        "decode_finalize_ctas": decode_finalize_ctas,
        "prefill_block_m": prefill_block_m,
        "prefill_tile_n": prefill_tile_n,
        "projection_tile_m": tile_m,
        "projection_tile_n": tile_n,
        "projection_tile_k": tile_k,
        "projection_num_stages": stages,
        "projection_num_m_warps": m_warps,
        "projection_num_n_warps": n_warps,
        "projection_k_splits": k_splits,
    }


def _mhc_decode_candidate(
    *,
    source_splits: int = 0,
    bf16x2: bool = False,
    partials_per_cta: int = 4,
    finalize_threads: int = 0,
    finalize_ctas: int = 1,
) -> SweepCandidate:
    return SweepCandidate.create(
        _mhc_config(
            backend="native",
            decode_source_splits=source_splits,
            decode_tile_n=6 if source_splits else 0,
            decode_bf16x2=bf16x2,
            decode_partials_per_cta=partials_per_cta,
            decode_finalize_threads=finalize_threads,
            decode_finalize_ctas=finalize_ctas,
            tile_m=16,
            tile_n=8,
            tile_k=256,
            stages=1,
            m_warps=1,
            n_warps=1,
            k_splits=1,
        )
    )


_MHC_BLOCK_M_CANDIDATES = {
    hidden_size: SweepCandidate.create(
        _mhc_config(
            backend="native",
            native_post_pre_backend="prefill_block_m",
            prefill_block_m=2,
            prefill_tile_n=tile_n,
            tile_m=16,
            tile_n=8,
            tile_k=256,
            stages=1,
            m_warps=1,
            n_warps=1,
            k_splits=1,
        )
    )
    for hidden_size, tile_n in ((4_096, 24), (7_168, 12))
}
_MHC_TF32_CANDIDATES = tuple(
    SweepCandidate.create(
        _mhc_config(
            backend="tf32_tma",
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            stages=stages,
            m_warps=m_warps,
            n_warps=n_warps,
            k_splits=k_splits,
        )
    )
    for tile_m, tile_n, tile_k, stages, m_warps, n_warps, k_splits in (
        (16, 8, 256, 1, 1, 1, 1),
        (32, 8, 256, 1, 2, 1, 1),
        (64, 24, 64, 3, 4, 1, 8),
        (64, 24, 64, 2, 4, 1, 8),
        (128, 24, 64, 2, 8, 1, 4),
        (192, 24, 64, 2, 12, 1, 8),
    )
)


class _GpuSession(AbstractContextManager):
    def __init__(self, context) -> None:
        self._context = context

    def __enter__(self):
        return self

    def __exit__(self, *_exc: object) -> None:
        import torch

        gc.collect()
        torch.cuda.synchronize(self._context.device_ordinal)
        torch.cuda.empty_cache()
        return None


class _HyperConnectionSession(_GpuSession):
    _CANDIDATES = tuple(
        SweepCandidate.create(
            {
                "backend": "cutedsl",
                "reduction_block_h": 4_096,
                "pointwise_block": pointwise_block,
                "reduction_num_warps": num_warps,
            }
        )
        for pointwise_block in (128, 256, 512)
        for num_warps in (4, 8)
    )

    def candidates(self, case: SweepCase) -> tuple[SweepCandidate, ...]:
        del case
        return self._CANDIDATES

    def measure(
        self,
        case: SweepCase,
        candidates: tuple[SweepCandidate, ...],
    ) -> tuple[SweepMeasurement, ...]:
        import torch

        from b12x.norm.hyperconnection._policy import HyperConnectionConfig
        from b12x.policy import PolicyContext, PolicyMode
        from benchmarks.benchmark_hyperconnection import (
            Profile,
            _graph_samples_us,
            _make_case,
        )

        device = torch.device("cuda", self._context.device_ordinal)
        settings = self._context.settings
        profile = Profile(tokens=int(case.query["max_tokens"]))
        base_policy = PolicyContext.for_device(
            device,
            mode=PolicyMode.HEURISTIC_ONLY,
        )
        flush = _l2_flush_fn(device, enabled=settings.cold_l2)
        measurements = []
        for candidate in candidates:
            try:
                config = HyperConnectionConfig.from_profile(candidate.config)
                policy = base_policy.with_override(HYPERCONNECTION, config)
                active = _make_case(
                    profile,
                    seed=settings.seed + int(candidate.candidate_id[-8:], 16),
                    device=device,
                    policy=policy,
                )
                samples, graph_contract, correctness = _graph_samples_us(
                    active,
                    "full_chain",
                    warmup=settings.warmup,
                    samples=settings.groups * settings.repetitions,
                    l2_flush=flush,
                )
                measurements.append(
                    SweepMeasurement(
                        candidate=candidate,
                        latency_us=_median_of_group_medians(
                            tuple(samples),
                            groups=settings.groups,
                            repetitions=settings.repetitions,
                        ),
                        correct=(
                            correctness.get("status") == "passed"
                            and graph_contract.get(
                                "replay_allocation_delta_bytes"
                            )
                            == 0
                        ),
                        metrics={
                            "operator": "full_chain",
                            "replay_allocation_bytes": graph_contract[
                                "replay_allocation_delta_bytes"
                            ],
                        },
                    )
                )
            except Exception as exc:  # noqa: BLE001 - failed configs survive
                measurements.append(
                    SweepMeasurement(
                        candidate=candidate,
                        latency_us=None,
                        correct=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        return tuple(measurements)


class _MtpFeedbackSession(_GpuSession):
    _CANDIDATES = tuple(
        SweepCandidate.create(
            {
                "backend": "cutedsl",
                "norm_block_h": 4_096,
                "norm_block_s": 4,
                "norm_num_warps": num_warps,
            }
        )
        for num_warps in (4, 8)
    )

    def candidates(self, case: SweepCase) -> tuple[SweepCandidate, ...]:
        del case
        return self._CANDIDATES

    def measure(
        self,
        case: SweepCase,
        candidates: tuple[SweepCandidate, ...],
    ) -> tuple[SweepMeasurement, ...]:
        import torch

        from b12x.policy import PolicyContext, PolicyMode
        from b12x.sequence.mtp_feedback._policy import MtpFeedbackConfig
        from benchmarks.benchmark_mtp_feedback import Profile, _benchmark_profile

        device = torch.device("cuda", self._context.device_ordinal)
        settings = self._context.settings
        tokens = int(case.query["max_tokens"])
        profile = Profile(name=f"profile-m{tokens}", phase="mixed", tokens=tokens)
        base_policy = PolicyContext.for_device(
            device,
            mode=PolicyMode.HEURISTIC_ONLY,
        )
        flush = _l2_flush_fn(device, enabled=settings.cold_l2)
        measurements = []
        for candidate in candidates:
            try:
                config = MtpFeedbackConfig.from_profile(candidate.config)
                policy = base_policy.with_override(MTP_FEEDBACK, config)
                result = _benchmark_profile(
                    profile,
                    seed=settings.seed + int(candidate.candidate_id[-8:], 16),
                    device=device,
                    eps=1.0e-6,
                    warmup=settings.warmup,
                    samples=settings.groups * settings.repetitions,
                    l2_flush=flush,
                    capacity_tokens=tokens,
                    policy=policy,
                )
                timings = result["timings"]
                correctness = result["correctness"]
                storage = result["storage"]
                measurements.append(
                    SweepMeasurement(
                        candidate=candidate,
                        latency_us=float(
                            timings["cuda_graph_replay"]["median_us"]
                        ),
                        correct=bool(
                            correctness["passed"]
                            and storage["graph_replay_allocation_delta_bytes"] == 0
                        ),
                        metrics={
                            "cosine": correctness[
                                "graph_replay_after_output_poison"
                            ]["cosine"],
                            "replay_allocation_bytes": storage[
                                "graph_replay_allocation_delta_bytes"
                            ],
                        },
                    )
                )
            except Exception as exc:  # noqa: BLE001 - failed configs survive
                measurements.append(
                    SweepMeasurement(
                        candidate=candidate,
                        latency_us=None,
                        correct=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        return tuple(measurements)


class _MhcSession(_GpuSession):
    def candidates(self, case: SweepCase) -> tuple[SweepCandidate, ...]:
        from b12x.norm.mhc._policy import MHC_POLICY, MhcConfig, MhcQuery

        query = MhcQuery(**case.query.to_dict())
        finalize_schedules = [(0, 1)]
        if query.hidden_size == 4_096 and 8 <= query.max_tokens <= 16:
            finalize_schedules.extend(((128, 1), (128, 8), (512, 1)))
        partial_groups = (4,)
        if query.hidden_size == 4_096 and 4 <= query.max_tokens <= 16:
            partial_groups = (4, 9, 25)
        candidates = [
            _mhc_decode_candidate(
                partials_per_cta=partials_per_cta,
                finalize_threads=finalize_threads,
                finalize_ctas=finalize_ctas,
            )
            for partials_per_cta in partial_groups
            for finalize_threads, finalize_ctas in finalize_schedules
        ]
        if query.hidden_size == 4_096 and query.max_tokens >= 8:
            bf16x2_modes = (False, True) if query.max_tokens <= 16 else (False,)
            candidates.extend(
                _mhc_decode_candidate(
                    source_splits=source_splits,
                    bf16x2=bf16x2,
                    finalize_threads=finalize_threads,
                    finalize_ctas=finalize_ctas,
                )
                for source_splits in (4, 8)
                for bf16x2 in bf16x2_modes
                for finalize_threads, finalize_ctas in finalize_schedules
            )
        block_m_candidate = _MHC_BLOCK_M_CANDIDATES.get(query.hidden_size)
        if query.max_tokens >= 32 and block_m_candidate is not None:
            candidates.append(block_m_candidate)
        if query.max_tokens >= 384:
            candidates.extend(_MHC_TF32_CANDIDATES)
        valid = []
        for candidate in candidates:
            config = MhcConfig.from_profile(candidate.config)
            try:
                MHC_POLICY.validate_config(query, config, self._context.device)
            except ValueError:
                continue
            valid.append(candidate)
        return tuple(valid)

    def measure(
        self,
        case: SweepCase,
        candidates: tuple[SweepCandidate, ...],
    ) -> tuple[SweepMeasurement, ...]:
        import torch
        import torch.nn.functional as torch_functional

        from b12x.norm import mhc
        from b12x.norm.mhc._policy import MhcConfig
        from b12x.policy import PolicyContext, PolicyMode
        from benchmarks.benchmark_residual import (
            _make_inputs,
            _mhc_pre_reference,
            _post_pre_reference,
        )

        device = torch.device("cuda", self._context.device_ordinal)
        settings = self._context.settings
        tokens = int(case.query["max_tokens"])
        hidden_size = int(case.query["hidden_size"])
        split_k = int(case.query["split_k"])
        residual, x, fn, scale, bias = _make_inputs(
            tokens=tokens,
            hidden_size=hidden_size,
            seed=settings.seed,
            device=device,
        )
        initial_y, prev_post, prev_comb = _mhc_pre_reference(
            residual,
            fn,
            scale,
            bias,
            rms_eps=1.0e-6,
            hc_eps=1.0e-6,
            sinkhorn_iters=20,
        )
        prev_post = prev_post.contiguous()
        prev_comb = prev_comb.contiguous()
        generator = torch.Generator(device="cpu").manual_seed(settings.seed + 17)
        norm_weight = (
            torch.randn(
                (hidden_size,),
                generator=generator,
                dtype=torch.float32,
            )
            .to(device=device, dtype=torch.bfloat16)
            .contiguous()
        )
        expected = _post_pre_reference(
            x,
            residual,
            prev_post,
            prev_comb,
            fn,
            scale,
            bias,
            rms_eps=1.0e-6,
            hc_eps=1.0e-6,
            sinkhorn_iters=20,
            norm_weight=norm_weight,
            norm_eps=1.0e-6,
        )
        base_policy = PolicyContext.for_device(
            device,
            mode=PolicyMode.HEURISTIC_ONLY,
        )
        context = MhcLayerStack(
            mhc=mhc,
            initial_y=initial_y.contiguous(),
            initial_residual=residual,
            initial_post=prev_post,
            initial_comb=prev_comb,
            tokens=tokens,
            hidden_size=hidden_size,
            device=device,
            generator=torch.Generator(device=device).manual_seed(
                settings.seed + int(case.case_id[-8:], 16) + 7_919
            ),
        )
        prepared: dict[str, _PreparedMhcCandidate] = {}
        failures: dict[str, SweepMeasurement] = {}
        for candidate in candidates:
            try:
                config = MhcConfig.from_profile(candidate.config)
                policy = base_policy.with_override(MHC, config)
                plan = mhc.plan(
                    mhc.Caps(
                        device=device,
                        max_tokens=tokens,
                        hidden_size=hidden_size,
                        split_k=split_k,
                    ),
                    policy=policy,
                )
                scratch = tuple(
                    torch.empty(shape, dtype=dtype, device=device)
                    for shape, dtype in plan.shapes_and_dtypes()
                )
                output = torch.empty(
                    (tokens, 4, hidden_size),
                    dtype=torch.bfloat16,
                    device=device,
                )
                y = torch.empty(
                    (tokens, hidden_size),
                    dtype=torch.bfloat16,
                    device=device,
                )
                post = torch.empty(
                    (tokens, 4),
                    dtype=torch.float32,
                    device=device,
                )
                comb = torch.empty(
                    (tokens, 4, 4),
                    dtype=torch.float32,
                    device=device,
                )
                binding = mhc.bind(
                    plan,
                    scratch=scratch,
                    tokens=tokens,
                    y=y,
                    post=post,
                    comb=comb,
                    out=output,
                )

                def run() -> None:
                    mhc.run_post_pre(
                        x,
                        residual,
                        prev_post,
                        prev_comb,
                        fn,
                        scale,
                        bias,
                        rms_eps=1.0e-6,
                        hc_eps=1.0e-6,
                        sinkhorn_iters=20,
                        norm_weight=norm_weight,
                        norm_eps=1.0e-6,
                        binding=binding,
                    )

                for _ in range(settings.warmup):
                    run()
                torch.cuda.synchronize(device)
                for actual in (output, y, post, comb):
                    actual.fill_(float("nan"))
                run()
                torch.cuda.synchronize(device)
                cosines = tuple(
                    float(
                        torch_functional.cosine_similarity(
                            actual.float().reshape(1, -1),
                            reference.float().reshape(1, -1),
                        ).item()
                    )
                    for actual, reference in zip(
                        (output, y, post, comb),
                        expected,
                        strict=True,
                    )
                )
                finite = all(
                    bool(torch.isfinite(actual).all().item())
                    for actual in (output, y, post, comb)
                )
                nonzero = all(
                    bool(torch.count_nonzero(actual).item())
                    for actual in (output, y, post, comb)
                )
                oracle_correct = (
                    finite
                    and nonzero
                    and min(cosines) >= settings.minimum_cosine
                )
                if not oracle_correct:
                    failures[candidate.candidate_id] = SweepMeasurement(
                        candidate=candidate,
                        latency_us=None,
                        correct=False,
                        metrics={
                            "output_cosine": cosines[0],
                            "y_cosine": cosines[1],
                            "post_cosine": cosines[2],
                            "comb_cosine": cosines[3],
                            "finite": finite,
                            "nonzero": nonzero,
                        },
                    )
                    continue

                context.prepare_candidate(candidate.candidate_id, plan)
                for _ in range(settings.warmup):
                    context.run(candidate.candidate_id)
                torch.cuda.synchronize(device)
                events = tuple(
                    (
                        torch.cuda.Event(enable_timing=True, external=True),
                        torch.cuda.Event(enable_timing=True, external=True),
                    )
                    for _ in range(context.boundaries)
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    context_output = context.run(candidate.candidate_id, events)
                context_outputs = (*context.mhc_outputs(), context_output)
                graph.replay()
                torch.cuda.synchronize(device)
                captured_outputs = tuple(actual.clone() for actual in context_outputs)
                for actual in context_outputs:
                    actual.fill_(float("nan"))
                caller_owned = (
                    *context.caller_owned_tensors(candidate.candidate_id),
                    context_output,
                )
                pointers_before = tuple(tensor.data_ptr() for tensor in caller_owned)
                allocator_before = _allocator_counters(device)
                graph.replay()
                torch.cuda.synchronize(device)
                allocator_after = _allocator_counters(device)
                allocator_deltas = {
                    key: allocator_after[key] - allocator_before[key]
                    for key in _ALLOCATOR_COUNTER_KEYS
                }
                stable_addresses = pointers_before == tuple(
                    tensor.data_ptr() for tensor in caller_owned
                )
                context_finite = all(
                    bool(torch.isfinite(actual).all().item())
                    for actual in context_outputs
                )
                context_nonzero = all(
                    bool(torch.count_nonzero(actual).item())
                    for actual in context_outputs
                )
                graph_replay_exact_outputs = tuple(
                    bool(torch.equal(actual, captured))
                    for actual, captured in zip(
                        context_outputs,
                        captured_outputs,
                        strict=True,
                    )
                )
                graph_replay_exact = all(graph_replay_exact_outputs)
                graph_replay_cosines = tuple(
                    float(
                        torch_functional.cosine_similarity(
                            actual.flatten().float(),
                            captured.flatten().float(),
                            dim=0,
                        ).item()
                    )
                    for actual, captured in zip(
                        context_outputs,
                        captured_outputs,
                        strict=True,
                    )
                )
                graph_replay_max_abs_errors = tuple(
                    float(
                        (actual.float() - captured.float()).abs().max().item()
                    )
                    for actual, captured in zip(
                        context_outputs,
                        captured_outputs,
                        strict=True,
                    )
                )
                graph_replay_mhc_cosine = min(
                    graph_replay_cosines[: len(context.mhc_outputs())]
                )
                graph_replay_consumer_cosine = graph_replay_cosines[-1]
                del captured_outputs

                def sample(
                    *,
                    graph=graph,
                    events=events,
                    boundaries=context.boundaries,
                ) -> tuple[float, float]:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    mhc_us = (
                        sum(begin.elapsed_time(finish) for begin, finish in events)
                        * 1_000.0
                        / boundaries
                    )
                    return mhc_us, float(start.elapsed_time(end)) * 1_000.0

                pilot_mhc_us, pilot_stack_us = sample()
                repetitions = _bounded_repetitions(
                    settings,
                    pilot_us=pilot_stack_us,
                )
                prepared[candidate.candidate_id] = _PreparedMhcCandidate(
                    candidate=candidate,
                    graph=graph,
                    retained=(
                        scratch,
                        output,
                        y,
                        post,
                        comb,
                        binding,
                        context_output,
                        context,
                    ),
                    correct=(
                        context_finite
                        and context_nonzero
                        and graph_replay_mhc_cosine >= settings.minimum_cosine
                        and stable_addresses
                        and not any(allocator_deltas.values())
                    ),
                    metrics={
                        "output_cosine": cosines[0],
                        "y_cosine": cosines[1],
                        "post_cosine": cosines[2],
                        "comb_cosine": cosines[3],
                        "finite": context_finite,
                        "nonzero": context_nonzero,
                        "graph_replay_exact": graph_replay_exact,
                        "graph_replay_exact_outputs": graph_replay_exact_outputs,
                        "graph_replay_cosines": graph_replay_cosines,
                        "graph_replay_max_abs_errors": graph_replay_max_abs_errors,
                        "graph_replay_mhc_cosine": graph_replay_mhc_cosine,
                        "graph_replay_consumer_cosine": (
                            graph_replay_consumer_cosine
                        ),
                        "stable_addresses": stable_addresses,
                        "replay_allocator_counter_deltas": allocator_deltas,
                        "timing_context": "two_decoder_layers",
                        "context_transformer_layers": context.transformer_layers,
                        "context_mhc_boundaries": context.boundaries,
                        "context_projection_pairs": context.boundaries + 1,
                        "context_fn_bf16_boundaries": sum(
                            parameters.fn_bf16 is not None
                            for parameters in context.parameters
                        ),
                        "context_pilot_mhc_us": pilot_mhc_us,
                        "context_pilot_stack_us": pilot_stack_us,
                        "context_repetitions": repetitions,
                    },
                    sample=sample,
                    repetitions=repetitions,
                )
            except Exception as exc:  # noqa: BLE001 - failed configs survive
                failures[candidate.candidate_id] = SweepMeasurement(
                    candidate=candidate,
                    latency_us=None,
                    correct=False,
                    error=f"{type(exc).__name__}: {exc}",
                )

        active = list(prepared.values())
        if not active:
            return tuple(failures[candidate.candidate_id] for candidate in candidates)
        for group in range(settings.groups):
            offset = group % len(active)
            ordered = active[offset:] + active[:offset]
            if group % 2:
                ordered.reverse()
            for item in ordered:
                for _ in range(item.repetitions):
                    mhc_us, stack_us = item.sample()
                    item.samples.append(mhc_us)
                    item.stack_samples.append(stack_us)

        measurements = []
        for candidate in candidates:
            failed = failures.get(candidate.candidate_id)
            if failed is not None:
                measurements.append(failed)
                continue
            item = prepared[candidate.candidate_id]
            measurements.append(
                SweepMeasurement(
                    candidate=candidate,
                    latency_us=_median_of_group_medians(
                        tuple(item.samples),
                        groups=settings.groups,
                        repetitions=item.repetitions,
                    ),
                    correct=item.correct,
                    metrics={
                        **item.metrics,
                        "context_stack_us": _median_of_group_medians(
                            tuple(item.stack_samples),
                            groups=settings.groups,
                            repetitions=item.repetitions,
                        ),
                        "context_mhc_samples_us": tuple(item.samples),
                        "context_stack_samples_us": tuple(item.stack_samples),
                    },
                )
            )
        return tuple(measurements)


@dataclass
class _PreparedMhcCandidate:
    candidate: SweepCandidate
    graph: object
    retained: tuple[object, ...]
    correct: bool
    metrics: dict[str, object]
    sample: Callable[[], tuple[float, float]]
    repetitions: int
    samples: list[float] = field(default_factory=list)
    stack_samples: list[float] = field(default_factory=list)


class _OneCaseFactory:
    def __init__(self, session_type) -> None:
        self._session_type = session_type

    def __call__(self, group_id, cases, context):
        del group_id
        if len(cases) != 1:
            raise ValueError("norm/sequence allocation groups contain one case")
        return self._session_type(context)


class HyperConnectionGenerator(DiscreteSweepGenerator):
    """Race production HyperConnection launch geometry."""

    def __init__(self, *, cases: Sequence[SweepCase] | None = None) -> None:
        super().__init__(
            component_id=HYPERCONNECTION,
            query_schema_version=1,
            config_schema_version=1,
            query_fields=(
                "dtype",
                "max_tokens",
                "hidden_size",
                "streams",
                "lowrank",
            ),
            range_fields=frozenset({"max_tokens"}),
            cases=_hyperconnection_cases() if cases is None else cases,
            benchmark_factory=_OneCaseFactory(_HyperConnectionSession),
            coverage={
                "token_capacities": list(_NORM_SEQUENCE_TOKEN_CAPACITIES),
            },
            nearest_range_bounds={"max_tokens": (1, 8_192)},
        )


class MtpFeedbackGenerator(DiscreteSweepGenerator):
    """Race production MTP feedback normalization launch geometry."""

    def __init__(self, *, cases: Sequence[SweepCase] | None = None) -> None:
        super().__init__(
            component_id=MTP_FEEDBACK,
            query_schema_version=1,
            config_schema_version=1,
            query_fields=("dtype", "max_tokens", "hidden_size", "streams"),
            range_fields=frozenset({"max_tokens"}),
            cases=_mtp_feedback_cases() if cases is None else cases,
            benchmark_factory=_OneCaseFactory(_MtpFeedbackSession),
            coverage={
                "token_capacities": list(_NORM_SEQUENCE_TOKEN_CAPACITIES),
            },
            nearest_range_bounds={"max_tokens": (1, 8_192)},
        )


class MhcGenerator(DiscreteSweepGenerator):
    """Race production mHC decode, block-M, and TF32 post/pre schedules."""

    def __init__(self, *, cases: Sequence[SweepCase] | None = None) -> None:
        super().__init__(
            component_id=MHC,
            query_schema_version=1,
            config_schema_version=4,
            query_fields=("dtype", "max_tokens", "hidden_size", "split_k"),
            range_fields=frozenset({"max_tokens"}),
            cases=_mhc_cases() if cases is None else cases,
            benchmark_factory=_OneCaseFactory(_MhcSession),
            coverage={
                "hidden_sizes": [4_096, 7_168],
                "decode_crossover_capacities": [24, 32, 48, 64, 96, 128, 192],
                "prefill_capacities": list(COMMON_PREFILL_TOKEN_CAPACITIES),
                "medium_prefill_anchors": [2_304, 3_072, 3_584],
            },
            candidate_contract_version=5,
            nearest_range_bounds={"max_tokens": (1, 8_192)},
            baseline_margin=0.02,
        )

    def baseline_config(self, case, context):
        from b12x.norm.mhc._policy import MHC_POLICY, MhcQuery

        query = MhcQuery(**case.query.to_dict())
        return MHC_POLICY.heuristic(query, context.device).to_dict()

    def reviewed_queries(self):
        from b12x.norm.mhc._policy import MhcQuery

        return tuple(MhcQuery(**case.query.to_dict()) for case in self._cases)


__all__ = ["HyperConnectionGenerator", "MhcGenerator", "MtpFeedbackGenerator"]
