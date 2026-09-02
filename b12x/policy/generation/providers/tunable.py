"""Measured sweep providers for tunable component launch policies."""

from __future__ import annotations

import gc
from collections.abc import Sequence
from contextlib import AbstractContextManager

from b12x.policy.components import NVFP4_QUANTIZATION, VARLEN_ATTENTION
from b12x.policy.generation.sweep import (
    DiscreteSweepGenerator,
    SweepCandidate,
    SweepCase,
    SweepMeasurement,
)

from .gpu_workers import (
    _bounded_repetitions,
    _cuda_event_samples_us,
    _l2_flush_fn,
    _median_of_group_medians,
)


def _nvfp4_cases() -> tuple[SweepCase, ...]:
    return tuple(
        SweepCase.create(
            group_id=f"m{rows}-k{columns}",
            query={
                "dtype": "bfloat16",
                "rows": rows,
                "columns": columns,
            },
        )
        for rows in (128, 512, 2_048)
        for columns in (2_560, 4_096, 5_120, 7_168, 10_240)
    )


def _varlen_attention_cases() -> tuple[SweepCase, ...]:
    return tuple(
        SweepCase.create(
            group_id=(f"{variant}-d{head_dim}-c{int(causal)}-b4-q128-k1024"),
            query={
                "variant": variant,
                "dtype": "bfloat16",
                "causal": causal,
                "batch_size": 4,
                "q_heads": 16,
                "kv_heads": 4,
                "q_head_dim": head_dim,
                "v_head_dim": head_dim,
                "query_rows": 512,
                "kv_rows": 4_096,
                "max_seqlen_q": 128,
                "max_seqlen_k": 1_024,
            },
        )
        for variant in ("batched", "varlen")
        for head_dim in (64, 128, 256)
        for causal in (False, True)
    )


class _Nvfp4Session(AbstractContextManager["_Nvfp4Session"]):
    _CANDIDATES = tuple(
        SweepCandidate.create(
            {
                "backend": "cutedsl",
                "liveness_strategy": strategy,
            }
        )
        for strategy in ("retain", "packed")
    )

    def __init__(self, context) -> None:
        self._context = context

    def __enter__(self) -> "_Nvfp4Session":
        return self

    def __exit__(self, *_exc: object) -> None:
        import torch

        gc.collect()
        torch.cuda.synchronize(self._context.device_ordinal)
        torch.cuda.empty_cache()
        return None

    def candidates(self, case: SweepCase) -> tuple[SweepCandidate, ...]:
        del case
        return self._CANDIDATES

    def measure(
        self,
        case: SweepCase,
        candidates: tuple[SweepCandidate, ...],
    ) -> tuple[SweepMeasurement, ...]:
        import torch

        from b12x._lib.intrinsics import quantize_grouped_nvfp4_torch
        from b12x.quantization import nvfp4
        from b12x.quantization.nvfp4._policy import Nvfp4QuantizationConfig

        if candidates != self._CANDIDATES:
            raise ValueError("NVFP4 worker received an unknown candidate set")
        settings = self._context.settings
        device = torch.device("cuda", self._context.device_ordinal)
        rows = int(case.query["rows"])
        columns = int(case.query["columns"])
        generator = torch.Generator(device=device).manual_seed(
            settings.seed + int(case.case_id[-8:], 16)
        )
        with torch.cuda.device(self._context.device_ordinal):
            source = torch.randn(
                (rows, columns),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            ).mul_(0.25)
            global_scale = torch.tensor([0.5], dtype=torch.float32, device=device)
            row_counts = torch.tensor([rows], dtype=torch.int32, device=device)
            packed_ref, scale_view_ref = quantize_grouped_nvfp4_torch(
                source.unsqueeze(0),
                row_counts,
                global_scale,
            )
            scale_ref = (
                scale_view_ref.permute(5, 2, 4, 0, 1, 3)
                .contiguous()
                .view(torch.uint8)
                .reshape(-1)
            )
            measurements = []
            flush = _l2_flush_fn(device, enabled=settings.cold_l2)
            for candidate in candidates:
                try:
                    config = Nvfp4QuantizationConfig.from_profile(candidate.config)
                    policy = self._context_policy(device).with_override(
                        NVFP4_QUANTIZATION,
                        config,
                    )
                    plan = nvfp4.plan(rows, columns, policy=policy)
                    outputs = nvfp4.allocate_outputs(plan, device=device)

                    def run() -> None:
                        nvfp4.run(
                            plan=plan,
                            x=source,
                            global_scale=global_scale,
                            outputs=outputs,
                        )

                    for _ in range(settings.warmup):
                        run()
                    torch.cuda.synchronize(device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        run()
                    graph.replay()
                    torch.cuda.synchronize(device)
                    packed_actual = outputs.packed_a_storage.permute(1, 2, 0)
                    packed_exact = bool(torch.equal(packed_actual, packed_ref))
                    scales_exact = bool(torch.equal(outputs.scale_flat, scale_ref))
                    nonzero = bool(
                        torch.count_nonzero(packed_actual).item()
                        and torch.count_nonzero(outputs.scale_flat).item()
                    )
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    repetitions = _bounded_repetitions(
                        settings,
                        pilot_us=float(start.elapsed_time(end)) * 1_000.0,
                    )
                    allocated_before = torch.cuda.memory_allocated(device)
                    samples = _cuda_event_samples_us(
                        graph.replay,
                        count=settings.groups * repetitions,
                        device=device,
                        flush=flush,
                    )
                    allocated_after = torch.cuda.memory_allocated(device)
                    latency = _median_of_group_medians(
                        samples,
                        groups=settings.groups,
                        repetitions=repetitions,
                    )
                    measurements.append(
                        SweepMeasurement(
                            candidate=candidate,
                            latency_us=latency,
                            correct=(
                                packed_exact
                                and scales_exact
                                and nonzero
                                and allocated_after <= allocated_before
                            ),
                            metrics={
                                "packed_exact": packed_exact,
                                "scales_exact": scales_exact,
                                "nonzero": nonzero,
                                "replay_allocation_bytes": (
                                    allocated_after - allocated_before
                                ),
                            },
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - failed candidates survive
                    measurements.append(
                        SweepMeasurement(
                            candidate=candidate,
                            latency_us=None,
                            correct=False,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
            return tuple(measurements)

    @staticmethod
    def _context_policy(device):
        from b12x.policy import PolicyContext, PolicyMode

        return PolicyContext.for_device(device, mode=PolicyMode.HEURISTIC_ONLY)


class _Nvfp4BenchmarkFactory:
    def __call__(self, group_id, cases, context):
        del group_id
        if len(cases) != 1:
            raise ValueError("NVFP4 allocation groups contain one case")
        return _Nvfp4Session(context)


class Nvfp4QuantizationGenerator(DiscreteSweepGenerator):
    """Race both real NVFP4 register-liveness schedules."""

    def __init__(self, *, cases: Sequence[SweepCase] | None = None) -> None:
        super().__init__(
            component_id=NVFP4_QUANTIZATION,
            query_schema_version=1,
            config_schema_version=2,
            query_fields=("dtype", "rows", "columns"),
            range_fields=frozenset({"rows", "columns"}),
            cases=_nvfp4_cases() if cases is None else cases,
            benchmark_factory=_Nvfp4BenchmarkFactory(),
            coverage={},
        )


def _attention_candidates(head_dim: int) -> tuple[SweepCandidate, ...]:
    if head_dim <= 128:
        tiles = ((64, 64), (64, 128), (128, 64), (128, 128))
    elif head_dim == 256:
        tiles = ((32, 32), (64, 32), (64, 48), (64, 64), (128, 32))
    else:
        raise ValueError(f"unsupported contiguous attention head dim {head_dim}")
    return tuple(
        SweepCandidate.create({"tile_m": tile_m, "tile_n": tile_n})
        for tile_m, tile_n in tiles
    )


def _attention_reference(case: SweepCase, q, k, v, cu_q, cu_k):
    import torch

    from b12x.attention.paged.reference import attention_reference

    causal = bool(case.query["causal"])
    if str(case.query["variant"]) == "batched":
        return attention_reference(q, k, v, causal=causal)
    outputs = []
    lses = []
    batch_size = int(case.query["batch_size"])
    for batch_idx in range(batch_size):
        q_start = int(cu_q[batch_idx].item())
        q_end = int(cu_q[batch_idx + 1].item())
        k_start = int(cu_k[batch_idx].item())
        k_end = int(cu_k[batch_idx + 1].item())
        output, lse = attention_reference(
            q[q_start:q_end],
            k[k_start:k_end],
            v[k_start:k_end],
            causal=causal,
        )
        outputs.append(output)
        lses.append(lse)
    return torch.cat(outputs, dim=0), torch.cat(lses, dim=1)


class _VarlenAttentionSession(AbstractContextManager["_VarlenAttentionSession"]):
    def __init__(self, context) -> None:
        self._context = context

    def __enter__(self) -> "_VarlenAttentionSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        import torch

        gc.collect()
        torch.cuda.synchronize(self._context.device_ordinal)
        torch.cuda.empty_cache()
        return None

    def candidates(self, case: SweepCase) -> tuple[SweepCandidate, ...]:
        return _attention_candidates(int(case.query["q_head_dim"]))

    def measure(
        self,
        case: SweepCase,
        candidates: tuple[SweepCandidate, ...],
    ) -> tuple[SweepMeasurement, ...]:
        import torch
        import torch.nn.functional as torch_functional

        from b12x.attention import varlen
        from b12x.attention.varlen._policy import VarlenAttentionConfig
        from b12x.policy import PolicyContext, PolicyMode

        settings = self._context.settings
        device = torch.device("cuda", self._context.device_ordinal)
        variant = str(case.query["variant"])
        batch_size = int(case.query["batch_size"])
        q_heads = int(case.query["q_heads"])
        kv_heads = int(case.query["kv_heads"])
        q_head_dim = int(case.query["q_head_dim"])
        v_head_dim = int(case.query["v_head_dim"])
        max_q = int(case.query["max_seqlen_q"])
        max_k = int(case.query["max_seqlen_k"])
        causal = bool(case.query["causal"])
        generator = torch.Generator(device=device).manual_seed(
            settings.seed + int(case.case_id[-8:], 16)
        )
        with torch.cuda.device(self._context.device_ordinal):
            if variant == "batched":
                q_shape = (batch_size, max_q, q_heads, q_head_dim)
                k_shape = (batch_size, max_k, kv_heads, q_head_dim)
                v_shape = (batch_size, max_k, kv_heads, v_head_dim)
                cu_q = None
                cu_k = None
            elif variant == "varlen":
                q_shape = (batch_size * max_q, q_heads, q_head_dim)
                k_shape = (batch_size * max_k, kv_heads, q_head_dim)
                v_shape = (batch_size * max_k, kv_heads, v_head_dim)
                cu_q = torch.arange(
                    batch_size + 1,
                    dtype=torch.int32,
                    device=device,
                ).mul_(max_q)
                cu_k = torch.arange(
                    batch_size + 1,
                    dtype=torch.int32,
                    device=device,
                ).mul_(max_k)
            else:
                raise ValueError(f"unsupported attention variant {variant!r}")
            q = torch.randn(
                q_shape,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            ).mul_(0.25)
            k = torch.randn(
                k_shape,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            ).mul_(0.25)
            v = torch.randn(
                v_shape,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            ).mul_(0.25)
            expected, _expected_lse = _attention_reference(case, q, k, v, cu_q, cu_k)
            flush = _l2_flush_fn(device, enabled=settings.cold_l2)
            base_policy = PolicyContext.for_device(
                device,
                mode=PolicyMode.HEURISTIC_ONLY,
            )
            measurements = []
            for candidate in candidates:
                try:
                    config = VarlenAttentionConfig.from_profile(candidate.config)
                    policy = base_policy.with_override(VARLEN_ATTENTION, config)
                    if variant == "batched":
                        plan = varlen.create_plan_batched(
                            q,
                            k,
                            v,
                            causal=causal,
                            policy=policy,
                        )
                        scratch_plan = varlen.plan_batched(plan)
                    else:
                        plan = varlen.create_plan(
                            q,
                            k,
                            v,
                            cu_q,
                            cu_k,
                            max_seqlen_q=max_q,
                            max_seqlen_k=max_k,
                            causal=causal,
                            policy=policy,
                        )
                        scratch_plan = varlen.plan(plan)
                    (scratch_spec,) = scratch_plan.scratch_specs()
                    scratch = torch.empty(
                        scratch_spec.shape,
                        dtype=scratch_spec.dtype,
                        device=scratch_spec.device,
                    )
                    if variant == "batched":
                        binding = scratch_plan.bind(
                            scratch=scratch,
                            q=q,
                            k=k,
                            v=v,
                        )

                        def run() -> None:
                            varlen.run_batched(binding=binding)

                    else:
                        binding = scratch_plan.bind(
                            scratch=scratch,
                            q=q,
                            k=k,
                            v=v,
                            cu_seqlens_q=cu_q,
                            cu_seqlens_k=cu_k,
                            max_seqlen_q=max_q,
                            max_seqlen_k=max_k,
                            causal=causal,
                        )

                        def run() -> None:
                            varlen.run(binding=binding)

                    for _ in range(settings.warmup):
                        run()
                    torch.cuda.synchronize(device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        run()
                    binding.output.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize(device)
                    actual = binding.output
                    finite = bool(torch.isfinite(actual).all().item())
                    cosine = float(
                        torch_functional.cosine_similarity(
                            actual.float().reshape(1, -1),
                            expected.float().reshape(1, -1),
                        ).item()
                    )
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    repetitions = _bounded_repetitions(
                        settings,
                        pilot_us=float(start.elapsed_time(end)) * 1_000.0,
                    )
                    allocated_before = torch.cuda.memory_allocated(device)
                    samples = _cuda_event_samples_us(
                        graph.replay,
                        count=settings.groups * repetitions,
                        device=device,
                        flush=flush,
                    )
                    allocated_after = torch.cuda.memory_allocated(device)
                    measurements.append(
                        SweepMeasurement(
                            candidate=candidate,
                            latency_us=_median_of_group_medians(
                                samples,
                                groups=settings.groups,
                                repetitions=repetitions,
                            ),
                            correct=(
                                finite
                                and cosine >= 0.999
                                and allocated_after <= allocated_before
                            ),
                            metrics={
                                "cosine": cosine,
                                "finite": finite,
                                "replay_allocation_bytes": (
                                    allocated_after - allocated_before
                                ),
                            },
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - failed tiles survive
                    measurements.append(
                        SweepMeasurement(
                            candidate=candidate,
                            latency_us=None,
                            correct=False,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
            return tuple(measurements)


class _VarlenAttentionBenchmarkFactory:
    def __call__(self, group_id, cases, context):
        del group_id
        if len(cases) != 1:
            raise ValueError("attention allocation groups contain one case")
        return _VarlenAttentionSession(context)


class VarlenAttentionGenerator(DiscreteSweepGenerator):
    """Race production contiguous-attention tiles on reviewed shapes."""

    def __init__(self, *, cases: Sequence[SweepCase] | None = None) -> None:
        super().__init__(
            component_id=VARLEN_ATTENTION,
            query_schema_version=1,
            config_schema_version=1,
            query_fields=(
                "variant",
                "dtype",
                "causal",
                "batch_size",
                "q_heads",
                "kv_heads",
                "q_head_dim",
                "v_head_dim",
                "query_rows",
                "kv_rows",
                "max_seqlen_q",
                "max_seqlen_k",
            ),
            range_fields=frozenset(
                {
                    "batch_size",
                    "query_rows",
                    "kv_rows",
                    "max_seqlen_q",
                    "max_seqlen_k",
                }
            ),
            cases=_varlen_attention_cases() if cases is None else cases,
            benchmark_factory=_VarlenAttentionBenchmarkFactory(),
            coverage={},
        )


__all__ = ["Nvfp4QuantizationGenerator", "VarlenAttentionGenerator"]
