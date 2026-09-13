"""gemm.wo_projection: planned lifecycle parity (plan -> bind -> run) against
a quantized torch reference, plus the fused inverse-RoPE variant replaying
under CUDA-graph capture with poisoned scale padding.

Curated from b12x tests/test_gemm_wo_projection.py (1.2k lines upstream);
split-GEMM leaves, packing round-trips, and byte-identity policy tests stay
in the b12x repo.
"""

from __future__ import annotations

import torch
import pytest

from b12x.gemm import wo_projection as wo
from b12x.gemm._shared.wo_mxfp8 import (
    dequantize_mxfp8_rows_torch,
    quantize_wo_projection_weights_mxfp8_torch,
)

from ..conftest import require_b12x


@pytest.mark.parametrize("block_size", (32, 128))
def test_planned_decode_tile_replays_live_rows_without_kernel_resolution(block_size):
    """Capacity-eight plans retain fresh inputs across every smaller live batch."""
    from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
    from b12x.gemm.wo_projection._policy import WoProjectionConfig
    from b12x.policy import WO_PROJECTION, PolicyContext

    require_b12x()
    torch.manual_seed(41764)
    policy = PolicyContext.for_device("cuda")

    def operand(n, k):
        value = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
        scales = (
            torch.randint(118, 124, (n // block_size, k // block_size), device="cuda")
            .byte()
            .view(torch.float8_e8m0fnu)
        )
        return value, scales

    a, sa = operand(2048, 4096)
    b, sb = operand(5120, 2048)
    weights = [
        wo.pack_weights(
            a,
            sa,
            b,
            sb,
            groups=2,
            group_width=4096,
            rank=1024,
            hidden=5120,
            block_size=(block_size, block_size),
            policy=policy.with_override(
                WO_PROJECTION, WoProjectionConfig(decode_tile_n=tile)
            ),
        )
        for tile in (0, 64)
    ]
    assert [w.decode_tile_n for w in weights] == [0, 64]
    source = (torch.randn(8, 16, 512, device="cuda") / 4).bfloat16()
    positions = torch.arange(8, device="cuda", dtype=torch.int64)
    angles = torch.randn(64, 32, device="cuda")
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1)

    def run(rows, weight):
        return wo.run_inv_rope(
            source[:rows],
            positions[:rows],
            cos_sin,
            weight,
            heads_per_group=8,
            nope_dim=448,
            rope_dim=64,
            expected_m=8,
        )

    for weight in weights:
        run(8, weight)
    torch.cuda.synchronize()
    freeze_kernel_resolution("WO capacity-eight live-row graph qualification")
    try:
        for rows in range(1, 9):
            graphs, outputs = [], []
            for weight in weights:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = run(rows, weight)
                graphs.append(graph)
                outputs.append(output)
            for _ in range(3):
                source.copy_((torch.randn_like(source.float()) / 4).bfloat16())
                positions.copy_(torch.randint(0, 64, (8,), device="cuda"))
                replacement = torch.randn_like(weights[0].wo_b.values.float())
                for weight in weights:
                    weight.wo_b.values.copy_(replacement)
                for output in outputs:
                    output.fill_(float("nan"))
                for graph in graphs:
                    graph.replay()
                torch.cuda.synchronize()
                assert bool(torch.isfinite(outputs[0]).all())
                assert bool(torch.count_nonzero(outputs[0]))
                torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
                torch.testing.assert_close(
                    outputs[1], run(rows, weights[1]), atol=0, rtol=0
                )
    finally:
        unfreeze_kernel_resolution()


def test_plan_bind_run_singleton_group_matches_quantized_reference() -> None:
    """TP8 collapses DSV4's eight output groups to one local WO group."""
    require_b12x()
    torch.manual_seed(31005)

    tokens, groups, group_width, rank, hidden = 3, 1, 512, 128, 128
    x_tgd = (
        torch.randn((tokens, groups, group_width), device="cuda", dtype=torch.bfloat16)
        / 4
    )
    wo_a_grd = (
        torch.randn((groups, rank, group_width), device="cuda", dtype=torch.bfloat16)
        / group_width**0.5
    )
    wo_b_hgr = (
        torch.randn((hidden, groups * rank), device="cuda", dtype=torch.bfloat16)
        / (groups * rank) ** 0.5
    )

    weights = quantize_wo_projection_weights_mxfp8_torch(wo_a_grd, wo_b_hgr)
    plan = wo.plan(
        wo.Caps(
            device=x_tgd.device,
            max_tokens=tokens,
            groups=groups,
            group_width=group_width,
            rank=rank,
            hidden=hidden,
            dtype=x_tgd.dtype,
        )
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=x_tgd.device)
        for shape, dtype in plan.shapes_and_dtypes()
    )
    binding = wo.bind(
        plan, scratch=scratch, source_tgd=x_tgd, weights=weights, expected_m=tokens
    )
    actual = wo.run(binding=binding)
    torch.cuda.synchronize()

    x_q = wo.quantize_input(x_tgd)
    x_deq = dequantize_mxfp8_rows_torch(x_q.values, x_q.scale_rows)
    wo_a_deq = dequantize_mxfp8_rows_torch(weights.wo_a.values, weights.wo_a.scale_rows)
    tmp = (x_deq @ wo_a_deq.T).to(torch.bfloat16).unsqueeze(-1)
    tmp_q = wo.quantize_input_b(tmp)
    tmp_deq = dequantize_mxfp8_rows_torch(tmp_q.values, tmp_q.scale_rows)
    wo_b_deq = dequantize_mxfp8_rows_torch(weights.wo_b.values, weights.wo_b.scale_rows)
    expected = tmp_deq @ wo_b_deq.T

    torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=0, atol=0)


def test_run_inv_rope_replays_under_graph_with_uninitialized_scale_padding() -> None:
    require_b12x()
    torch.manual_seed(31007)

    tokens = 1
    groups = 2
    heads_per_group = 4
    nope_dim = 96
    rope_dim = 32
    head_dim = nope_dim + rope_dim
    group_width = heads_per_group * head_dim
    rank, hidden = 64, 128
    o = (
        torch.randn(
            (tokens, groups * heads_per_group, head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 4
    ).contiguous()
    positions = torch.zeros((tokens,), device="cuda", dtype=torch.long)
    cos_sin_cache = torch.zeros((4, rope_dim), device="cuda", dtype=torch.float32)
    cos_sin_cache[:, : rope_dim // 2] = 1
    wo_a = (
        torch.randn((groups, rank, group_width), device="cuda", dtype=torch.bfloat16)
        / group_width**0.5
    )
    wo_b = (
        torch.randn((hidden, groups * rank), device="cuda", dtype=torch.bfloat16)
        / (groups * rank) ** 0.5
    )
    weights = quantize_wo_projection_weights_mxfp8_torch(wo_a, wo_b)

    def run_once() -> torch.Tensor:
        return wo.run_inv_rope(
            o,
            positions,
            cos_sin_cache,
            weights,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            expected_m=1,
        )

    # Warm all compilation/allocation before capture, then verify replay
    # observes changed inputs without allocating or depending on stale scale
    # padding.
    run_once()
    run_once()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run_once()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    o.copy_(torch.randn_like(o) / 4)
    graph.replay()
    torch.cuda.synchronize()
    replayed = captured.clone()
    expected = run_once().clone()
    torch.cuda.synchronize()

    assert bool(torch.isfinite(replayed).all().item())
    assert bool((replayed != 0).any().item())
    torch.testing.assert_close(replayed, expected, rtol=0, atol=0)
