"""Compiled residency bounds preserve MXFP4 MoE numerics and graph replay."""

from __future__ import annotations

import pytest
import torch

from benchmarks.benchmark_ds4_moe import make_synthetic_mxfp4_moe


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("tile_m", [16, 32])
@pytest.mark.parametrize("capacity", [8, 16])
def test_v41_compact_grid_reuses_residency_for_live_counts(
    monkeypatch, record_property, tile_m, capacity
):
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x._lib import cooperative
    from b12x.moe._shared.kernels.reference import moe_reference_w4a8_mx
    from b12x.moe.fused_moe import _impl

    if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        pytest.skip("The repacked MXFP4 path requires SM120 or SM121")
    # Request two CTAs per SM so a one-resident-block specialization must
    # apply the compiled resource bound even on a GPU with many SMs.
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    _impl.clear_tp_moe_caches()
    experts_count, hidden, intermediate, topk = 32, 5120, 2304, 6
    device = torch.device("cuda", torch.cuda.current_device())
    weights = make_synthetic_mxfp4_moe(
        experts_count, hidden, intermediate, seed=41, device=device
    )
    gen = torch.Generator(device=device).manual_seed(42)
    x = torch.randn(capacity, hidden, generator=gen, device=device).to(torch.bfloat16)
    logits = torch.randn(capacity, experts_count, generator=gen, device=device)
    scores, ids = logits.topk(topk, dim=-1)
    ids = ids.to(torch.int32)
    scores = scores.softmax(dim=-1)
    changed_x = (x.float() * -0.75).to(torch.bfloat16)
    changed_ids = (ids + 1).remainder(experts_count)
    references = []
    # Weight preparation transfers the checkpoint allocations into the
    # repacked runtime layout, so build both logical oracles first.
    for inputs, routing in ((x, ids), (changed_x, changed_ids)):
        references.append(
            moe_reference_w4a8_mx(
                inputs.float(),
                weights["w13_fp4"],
                weights["w13_mx"],
                None,
                weights["alphas"],
                weights["w2_fp4"],
                weights["w2_mx"],
                None,
                weights["alphas"],
                routing,
                scores,
                experts_count,
                hidden,
                intermediate,
                activation="silu",
                swiglu_limit=10.0,
            )
        )
    weight_plan = _impl.plan_b12x_fp4_moe_weights(
        quant_modes="w4a8_mx",
        source_format="fp4_e8m0_k32",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=experts_count,
        hidden_size=hidden,
        intermediate_size=intermediate,
    )
    experts = _impl.prepare_b12x_fp4_moe_weights(
        plan=weight_plan,
        w1_fp4=weights["w13_fp4"],
        w1_blockscale=weights["w13_mx"],
        w1_global_scale=weights["alphas"],
        a1_gscale=weights["input_scale"],
        w2_fp4=weights["w2_fp4"],
        w2_blockscale=weights["w2_mx"],
        w2_global_scale=weights["alphas"],
        a2_gscale=weights["input_scale"],
        params_dtype=torch.bfloat16,
    )
    plan = _impl.plan_tp_moe_scratch(
        _impl.TPMoEScratchCaps(
            max_tokens=capacity,
            num_topk=topk,
            device=device,
            weight_plan=weight_plan,
            quant_mode="w4a8_mx",
            core_token_counts=(capacity,),
            route_num_experts=0,
            swiglu_limit=10.0,
            decode_config=_impl.MoeDecodeConfig(
                backend="dynamic",
                route_planner="internal",
                max_active_clusters=2 * sm_count,
                dynamic_tile_m=tile_m,
                dynamic_route_mode="grouped",
            ),
        )
    )
    assert plan.launch_plan.implementation == "dynamic"
    assert plan.launch_plan.execution.tile_m == tile_m
    scratch = tuple(
        torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        for spec in plan.scratch_specs()
    )
    output = torch.empty_like(x)

    def bind(rows):
        return plan.bind(
            scratch=scratch,
            a=x[:rows],
            experts=experts,
            topk_ids=ids[:rows],
            topk_weights=scores[:rows],
            output=output[:rows],
            input_scales_static=True,
            fast_math=True,
        )

    def check(rows, reference):
        actual = output[:rows]
        assert actual.isfinite().all() and actual.abs().sum().item() > 0
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(),
            reference[:rows].float().flatten(),
            dim=0,
        ).item()
        assert cosine > 0.998, (tile_m, rows, cosine)
        norm_ratio = actual.float().norm() / reference[:rows].float().norm()
        assert 0.95 < norm_ratio < 1.05, (tile_m, rows, norm_ratio)

    bind(capacity).run()
    torch.cuda.synchronize()
    check(capacity, references[0])
    compiled = tuple(_impl._DYNAMIC_KERNEL_CACHE.values())
    assert len(compiled) == 1
    limits = dict(compiled[0]._b12x_cooperative_grid_limits)
    assert limits and max(limits.values()) <= 2 * sm_count
    assert compiled[0]._b12x_launch_metadata["status"] == "exact"
    record_property("cooperative_grid_limits", repr(limits))
    record_property("block_threads", compiled[0]._b12x_block_threads)
    record_property("launch_metadata", repr(compiled[0]._b12x_launch_metadata))
    storage = (*scratch, x, ids, scores, output)
    addresses = tuple(t.data_ptr() for t in storage)

    def unexpected_query(*args):
        raise AssertionError("Warmed launch repeated the CUDA occupancy query")

    monkeypatch.setattr(cooperative, "_resident_blocks_per_sm", unexpected_query)
    with kernel_resolution_guard("V4.1 compact MoE capacity and residency are warmed"):
        x.copy_(changed_x)
        ids.copy_(changed_ids)
        for rows in (1, 2, capacity - 1, capacity):
            binding = bind(rows)
            binding.run()
            torch.cuda.synchronize()
            check(rows, references[1])
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.cuda.graph(graph):
                binding.run()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            for sentinel in (float("nan"), 997.0):
                output.fill_(sentinel)
                allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize()
                assert (
                    torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
                )
                assert tuple(t.data_ptr() for t in storage) == addresses
                check(rows, references[1])
        assert tuple(_impl._DYNAMIC_KERNEL_CACHE.values()) == compiled
        assert compiled[0]._b12x_cooperative_grid_limits == limits
