"""Race public inverse-RoPE output projections over the bounded decode domain."""

from __future__ import annotations

import gc
from contextlib import AbstractContextManager

from b12x.policy import WO_PROJECTION
from b12x.policy.generation.sweep import (
    DiscreteSweepGenerator,
    SweepCandidate,
    SweepCase,
    SweepMeasurement,
)

from .gpu_workers import _cuda_event_samples_us, _l2_flush_fn, _median_of_group_medians


def _cases():
    result = []
    for tokens, tp in ((1, 1), (4, 2), (32, 4), (32, 8)):
        result.append(
            SweepCase.create(
                group_id=f"wo-tp{tp}-m{tokens}",
                query=dict(
                    dtype="bfloat16",
                    max_tokens=tokens,
                    groups=24 // tp,
                    group_width=512,
                    rank=512,
                    hidden=2560,
                ),
                metadata=dict(tokens=tokens, block_size=128),
            )
        )
    # One plan must remain correct over every live row count and both scale
    # layouts: weight-scale granularity is not a runtime policy query field.
    for block in (32, 128):
        for tokens in range(1, 9):
            result.append(
                SweepCase.create(
                    group_id="wo-ds41-tp4-decode",
                    query=dict(
                        dtype="bfloat16",
                        max_tokens=8,
                        groups=2,
                        group_width=4096,
                        rank=1024,
                        hidden=5120,
                    ),
                    scenario=f"m{tokens}-block{block}",
                    metadata=dict(tokens=tokens, block_size=block),
                )
            )
    return tuple(result)


class _Session(AbstractContextManager):
    def __init__(self, context):
        self.context = context

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        import torch

        gc.collect()
        torch.cuda.synchronize(self.context.device_ordinal)
        torch.cuda.empty_cache()

    def candidates(self, case):
        tiles = (0, 64, 128) if case.query["group_width"] == 4096 else (0,)
        return tuple(
            SweepCandidate.create(dict(backend="mxfp8", decode_tile_n=n)) for n in tiles
        )

    def measure(self, case, candidates):
        import torch
        from b12x.gemm import wo_projection as wo
        from b12x.gemm.wo_projection._policy import WoProjectionConfig
        from b12x.policy import PolicyContext, PolicyMode

        context = self.context
        settings = context.settings
        device = torch.device("cuda", context.device_ordinal)
        q = case.query
        tokens = int(case.metadata["tokens"])
        block = int(case.metadata["block_size"])
        geometry = {
            key: int(q[key]) for key in ("groups", "group_width", "rank", "hidden")
        }
        groups, width, rank, hidden = (geometry[k] for k in geometry)
        policy = PolicyContext.for_device(device, mode=PolicyMode.HEURISTIC_ONLY)
        flush = _l2_flush_fn(device, enabled=settings.cold_l2)
        results = []
        with torch.cuda.device(device):
            torch.manual_seed(settings.seed)

            def operand(n, k):
                values = torch.randn(n, k, device=device).to(torch.float8_e4m3fn)
                scales = (
                    torch.randint(118, 124, (n // block, k // block), device=device)
                    .byte()
                    .view(torch.float8_e8m0fnu)
                )
                return values, scales

            a, sa = operand(groups * rank, width)
            b, sb = operand(hidden, groups * rank)
            weights = wo.pack_weights(
                a, sa, b, sb, **geometry, block_size=(block, block), policy=policy
            )
            source = (
                torch.randn(tokens, groups * width // 512, 512, device=device) / 4
            ).bfloat16()
            positions = torch.arange(tokens, device=device, dtype=torch.int64)
            angles = torch.randn(64, 32, device=device)
            cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1)

            def binding_for(tile):
                selected = policy.with_override(
                    WO_PROJECTION, WoProjectionConfig(decode_tile_n=tile)
                )
                plan = wo.plan(
                    wo.Caps(device=device, max_tokens=int(q["max_tokens"]), **geometry),
                    policy=selected,
                )
                scratch = tuple(
                    torch.empty(shape, dtype=dtype, device=device)
                    for shape, dtype in plan.shapes_and_dtypes()
                )
                return plan.bind_inv_rope(
                    scratch=scratch,
                    o=source,
                    positions=positions,
                    cos_sin_cache=cos_sin,
                    weights=weights,
                    heads_per_group=width // 512,
                    nope_dim=448,
                    rope_dim=64,
                    expected_m=tokens,
                )

            control = binding_for(0)
            expected = wo.run_inv_rope(binding=control).clone()
            torch.cuda.synchronize(device)
            for candidate in candidates:
                try:
                    binding = binding_for(int(candidate.config["decode_tile_n"]))
                    for _ in range(max(2, settings.warmup)):
                        wo.run_inv_rope(binding=binding)
                    torch.cuda.synchronize(device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = wo.run_inv_rope(binding=binding)
                    output.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize(device)
                    exact = bool(torch.equal(output, expected))
                    finite = bool(torch.isfinite(output).all())
                    nonzero = bool(torch.count_nonzero(output))
                    before = torch.cuda.memory_allocated(device)
                    samples = _cuda_event_samples_us(
                        graph.replay,
                        count=settings.groups * settings.repetitions,
                        device=device,
                        flush=flush,
                    )
                    growth = torch.cuda.memory_allocated(device) - before
                    results.append(
                        SweepMeasurement(
                            candidate=candidate,
                            latency_us=_median_of_group_medians(
                                samples,
                                groups=settings.groups,
                                repetitions=settings.repetitions,
                            ),
                            correct=exact and finite and nonzero and growth <= 0,
                            metrics=dict(
                                bitwise_control_parity=exact,
                                finite=finite,
                                nonzero=nonzero,
                                replay_allocation_bytes=growth,
                            ),
                        )
                    )
                except Exception as exc:
                    results.append(
                        SweepMeasurement(
                            candidate=candidate,
                            latency_us=None,
                            correct=False,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
        return tuple(results)


class _Factory:
    def __call__(self, group_id, cases, context):
        return _Session(context)


class WoProjectionGenerator(DiscreteSweepGenerator):
    """Select tiles using complete public composites, not a WO-B proxy."""

    def __init__(self, *, cases=None):
        super().__init__(
            component_id=WO_PROJECTION,
            query_schema_version=1,
            config_schema_version=2,
            candidate_contract_version=1,
            query_fields=(
                "dtype",
                "max_tokens",
                "groups",
                "group_width",
                "rank",
                "hidden",
            ),
            range_fields=frozenset(),
            cases=_cases() if cases is None else cases,
            benchmark_factory=_Factory(),
            coverage={},
        )
