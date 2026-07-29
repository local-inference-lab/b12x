"""Kimi K3 TP16 single-token hybrid MoE latency harness.

Compares the production serial MXFP4+NF3 decode path with SparkInfer's
one-grid heterogeneous kernel at the real K3 shard geometry.  No checkpoint
or vLLM engine is loaded.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from sparkinfer.moe._shared.kernels.w4a16.host import packed_gemm_scratch_elements
from sparkinfer.moe._shared.kernels.w4a16.kernel import (
    build_w4a16_tier_local_map,
    compile_w4a16_fused_moe,
    run_w4a16_moe,
    run_w4a16_moe_hybrid,
)
from sparkinfer.moe._shared.kernels.w4a16.prepare import (
    make_w4a16_packed_buffers,
    prepare_nf3_moe_weights,
    prepare_w4a16_e8m0_native_weights,
)

HIDDEN = 3584
INTERMEDIATE = 192
TOPK = 16
EXPERTS = 896
DEFAULT_TILES = (64, 128, 64, 128)


def _e8m0(shape: tuple[int, ...], offset: int = 0) -> torch.Tensor:
    values = torch.arange(torch.tensor(shape).prod().item(), device="cuda")
    values = ((values + offset) % 4 + 119).to(torch.uint8).reshape(shape)
    return values.view(torch.float8_e8m0fnu)


def _e4m3(shape: tuple[int, ...]) -> torch.Tensor:
    value = torch.full(shape, 0.015625, dtype=torch.float32, device="cuda")
    return value.to(torch.float8_e4m3fn)


def _prepare_mxfp4(experts: int):
    rows = 2 * INTERMEDIATE
    w13 = torch.randint(
        0, 256, (experts, rows, HIDDEN // 2), dtype=torch.uint8, device="cuda"
    )
    w2 = torch.randint(
        0, 256, (experts, HIDDEN, INTERMEDIATE // 2), dtype=torch.uint8, device="cuda"
    )
    return prepare_w4a16_e8m0_native_weights(
        w13,
        _e8m0((experts, rows, HIDDEN // 32)),
        torch.ones(experts, dtype=torch.float32, device="cuda"),
        w2,
        _e8m0((experts, HIDDEN, INTERMEDIATE // 32), offset=1),
        torch.ones(experts, dtype=torch.float32, device="cuda"),
        activation="situ",
        params_dtype=torch.bfloat16,
        w13_layout="w31",
    )


def _prepare_nf3(experts: int, tiles: tuple[int, int, int, int]):
    rows = 2 * INTERMEDIATE
    w13 = torch.randint(0, 8, (experts, rows, HIDDEN), dtype=torch.int32, device="cuda")
    w2 = torch.randint(
        0, 8, (experts, HIDDEN, INTERMEDIATE), dtype=torch.int32, device="cuda"
    )
    return prepare_nf3_moe_weights(
        w13,
        _e4m3((experts, rows, HIDDEN // 32)),
        w2,
        _e4m3((experts, HIDDEN, INTERMEDIATE // 32)),
        activation="situ",
        fc1_tile_n=tiles[1],
        fc2_tile_n=tiles[3],
        params_dtype=torch.bfloat16,
    )


def _micro_scratch() -> torch.Tensor:
    fc2_n_chunks = ((INTERMEDIATE // 2) + 127) // 128
    return torch.empty(
        2 * fc2_n_chunks * 128 * TOPK,
        dtype=torch.bfloat16,
        device="cuda",
    )


def _latencies_ms(fn, iterations: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return values


def _graph_latencies_ms(fn, iterations: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    return _latencies_ms(graph.replay, iterations, warmup)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kept", type=int, default=717)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--fc1-tile-k", type=int, default=DEFAULT_TILES[0])
    parser.add_argument("--fc1-tile-n", type=int, default=DEFAULT_TILES[1])
    parser.add_argument("--fc2-tile-k", type=int, default=DEFAULT_TILES[2])
    parser.add_argument("--fc2-tile-n", type=int, default=DEFAULT_TILES[3])
    parser.add_argument(
        "--schedule-whole-tiles",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if not 1 <= args.kept < EXPERTS:
        raise ValueError("--kept must leave both tiers non-empty")
    tiles = (
        args.fc1_tile_k,
        args.fc1_tile_n,
        args.fc2_tile_k,
        args.fc2_tile_n,
    )
    nf3 = EXPERTS - args.kept
    torch.manual_seed(20260729)

    t0 = _prepare_mxfp4(args.kept)
    t1 = _prepare_nf3(nf3, tiles)
    x = (torch.randn(1, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.1).contiguous()
    # Exercise both tiers while keeping all TOPK routes in range for extreme
    # real K3 splits such as 895/1 and 2/894.
    tier0_routes = min(TOPK // 2, args.kept)
    tier1_routes = min(TOPK - tier0_routes, nf3)
    tier0_routes = TOPK - tier1_routes
    ids = torch.tensor(
        [
            list(range(tier0_routes))
            + list(range(args.kept, args.kept + tier1_routes))
        ],
        dtype=torch.int32,
        device="cuda",
    )
    weights = torch.softmax(torch.randn(1, TOPK, device="cuda"), dim=-1).float()
    t0_ids = torch.where(ids < args.kept, ids, torch.zeros_like(ids)).contiguous()
    t1_ids = torch.where(
        ids >= args.kept, ids - args.kept, torch.zeros_like(ids)
    ).contiguous()
    t0_weights = weights.masked_fill(ids >= args.kept, 0.0).contiguous()
    t1_weights = weights.masked_fill(ids < args.kept, 0.0).contiguous()

    b0 = make_w4a16_packed_buffers(
        t0, m=1, topk=TOPK, dtype=torch.bfloat16, device=torch.device("cuda")
    )
    b1 = make_w4a16_packed_buffers(
        t1, m=1, topk=TOPK, dtype=torch.bfloat16, device=torch.device("cuda")
    )
    micro0, micro1 = _micro_scratch(), _micro_scratch()
    serial_out = torch.empty((1, HIDDEN), dtype=torch.bfloat16, device="cuda")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    sms = int(props.multi_processor_count)
    max_shared_mem = int(
        getattr(props, "shared_memory_per_block_optin", 101_376)
    )

    def plan_tier(prepared):
        # Native MXFP4 takes SparkInfer's dedicated small-M microkernel before
        # the fused launch is consulted. Only NF3 needs an explicitly pinned
        # serial launch so its packed N geometry matches this tuning case.
        if prepared.weight_layout == "modelopt":
            return None
        return compile_w4a16_fused_moe(
            size_m=1,
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
            num_experts=int(prepared.num_experts),
            top_k=TOPK,
            activation="situ",
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            moe_block_size=8,
            max_m_blocks=TOPK,
            element_dtype="bf16",
            fast_math=True,
            sms=sms,
            max_shared_mem=max_shared_mem,
            weight_layout=prepared.weight_layout,
            scale_format=prepared.scale_format,
            w13_layout=prepared.w13_layout,
            direct_topk_routes=True,
            tc_decode_fused_sum=True,
            force_tile_config=tiles,
        )

    launch0, launch1 = plan_tier(t0), plan_tier(t1)
    serial_scratch_elements = packed_gemm_scratch_elements(
        size_n=max(2 * INTERMEDIATE, HIDDEN),
        # The serial direct-topk path budgets one 8-row block per route;
        # the hybrid one-grid path below needs only one slot per route.
        route_slots=TOPK * 8,
        moe_block_size=8,
        sms=sms,
    )
    serial_fc1_tmp = (
        torch.empty(serial_scratch_elements, dtype=torch.float32, device="cuda"),
        torch.empty(serial_scratch_elements, dtype=torch.float32, device="cuda"),
    )
    serial_fc2_tmp = (
        torch.empty(serial_scratch_elements, dtype=torch.float32, device="cuda"),
        torch.empty(serial_scratch_elements, dtype=torch.float32, device="cuda"),
    )

    def run_tier(
        prepared,
        route_weights,
        route_ids,
        buffers,
        micro,
        launch,
        scratch_index,
        input_x=x,
    ):
        return run_w4a16_moe(
            input_x,
            prepared,
            route_weights,
            route_ids,
            activation="situ",
            fast_math=True,
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=micro,
            output=buffers.output,
            fc1_c_tmp=serial_fc1_tmp[scratch_index],
            fc2_c_tmp=serial_fc2_tmp[scratch_index],
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
            fused_launch=launch,
        )

    def serial():
        torch.add(
            run_tier(t0, t0_weights, t0_ids, b0, micro0, launch0, 0),
            run_tier(t1, t1_weights, t1_ids, b1, micro1, launch1, 1),
            out=serial_out,
        )

    tier_map = build_w4a16_tier_local_map(
        range(args.kept),
        range(args.kept, EXPERTS),
        map_slots=EXPERTS,
        device=torch.device("cuda"),
    )
    hybrid_fc1 = torch.empty(
        TOPK * 2 * INTERMEDIATE, dtype=torch.bfloat16, device="cuda"
    )
    hybrid_act = torch.empty(TOPK * INTERMEDIATE, dtype=torch.bfloat16, device="cuda")
    hybrid_out = torch.empty((1, HIDDEN), dtype=torch.bfloat16, device="cuda")
    scratch_elements = packed_gemm_scratch_elements(
        size_n=max(2 * INTERMEDIATE, HIDDEN),
        route_slots=TOPK,
        moe_block_size=8,
        sms=sms,
    )
    hybrid_fc1_tmp = torch.empty(scratch_elements, dtype=torch.float32, device="cuda")
    hybrid_fc2_tmp = torch.empty(scratch_elements, dtype=torch.float32, device="cuda")

    def hybrid_into(input_x, route_weights, route_ids, output):
        run_w4a16_moe_hybrid(
            input_x,
            t0,
            t1,
            route_weights,
            route_ids,
            tier_map,
            activation="situ",
            intermediate_cache13=hybrid_fc1,
            intermediate_cache2=hybrid_act,
            output=output,
            force_tile_config=tiles,
            size_m=1,
            fc1_c_tmp=hybrid_fc1_tmp,
            fc2_c_tmp=hybrid_fc2_tmp,
            fast_math=True,
            schedule_whole_tiles=args.schedule_whole_tiles,
        )

    def hybrid():
        hybrid_into(x, weights, ids, hybrid_out)

    serial()
    hybrid()
    torch.cuda.synchronize()
    ref, got = serial_out.float(), hybrid_out.float()
    cosine = torch.nn.functional.cosine_similarity(ref, got).item()
    max_abs = (ref - got).abs().max().item()
    ref_abs = ref.abs().max().item()
    rel_max = max_abs / max(ref_abs, 1e-12)
    if not torch.isfinite(got).all() or cosine < 0.998 or rel_max >= 0.02:
        raise RuntimeError(
            "hybrid correctness failure: "
            f"cosine={cosine}, max_abs={max_abs}, rel_max={rel_max}"
        )

    # CUDA-graph correctness must cover changing inputs/routes and consecutive
    # layers sharing the production scratch buffers. A latency-only replay of
    # one frozen route table cannot detect stale captured pointers or scratch
    # lifetime bugs.
    graph_x = x.clone()
    graph_weights0 = weights.clone()
    graph_ids0 = ids.clone()
    graph_weights1 = weights.flip(-1).contiguous()
    graph_ids1 = ids.flip(-1).contiguous()
    graph_mid = torch.empty_like(hybrid_out)
    graph_final = torch.empty_like(hybrid_out)

    def graph_chain():
        hybrid_into(graph_x, graph_weights0, graph_ids0, graph_mid)
        hidden = graph_mid + graph_x
        hybrid_into(hidden, graph_weights1, graph_ids1, graph_final)

    graph_chain()
    torch.cuda.synchronize()
    correctness_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(correctness_graph):
        graph_chain()
    torch.cuda.synchronize()

    eager_mid = torch.empty_like(hybrid_out)
    eager_final = torch.empty_like(hybrid_out)
    graph_step_max_abs = []
    graph_step_cosine = []
    for step_index in range(8):
        step_x = (
            torch.randn_like(graph_x) * (0.05 + 0.01 * step_index)
        ).contiguous()
        # Rotate a fixed valid set so both tier-map lookup and route order vary.
        step_ids0 = ids.roll(step_index + 1, dims=-1).contiguous()
        step_ids1 = ids.flip(-1).roll(step_index + 2, dims=-1).contiguous()
        step_weights0 = torch.softmax(torch.randn_like(weights), dim=-1).float()
        step_weights1 = torch.softmax(torch.randn_like(weights), dim=-1).float()
        graph_x.copy_(step_x)
        graph_ids0.copy_(step_ids0)
        graph_ids1.copy_(step_ids1)
        graph_weights0.copy_(step_weights0)
        graph_weights1.copy_(step_weights1)
        correctness_graph.replay()
        torch.cuda.synchronize()
        graph_value = graph_final.clone()

        hybrid_into(step_x, step_weights0, step_ids0, eager_mid)
        hybrid_into(eager_mid + step_x, step_weights1, step_ids1, eager_final)
        torch.cuda.synchronize()
        graph_f32, eager_f32 = graph_value.float(), eager_final.float()
        graph_step_max_abs.append(float((graph_f32 - eager_f32).abs().max().item()))
        graph_step_cosine.append(
            float(torch.nn.functional.cosine_similarity(graph_f32, eager_f32).item())
        )

    if min(graph_step_cosine) < 0.999 or max(graph_step_max_abs) > 5e-4:
        raise RuntimeError(
            "dynamic two-layer CUDA graph mismatch: "
            f"cosine={graph_step_cosine}, max_abs={graph_step_max_abs}"
        )

    # The checkpoint also contains all-MXFP4 layers. They bypass the hybrid
    # launch and use the native small-M tier path, so validate dynamic graph
    # replay for that path independently.
    native_graph_x = x.clone()
    native_graph_ids = torch.arange(TOPK, dtype=torch.int32, device="cuda").view(1, -1)
    native_graph_weights = torch.softmax(torch.randn_like(weights), dim=-1).float()

    def native_graph_call():
        run_tier(
            t0,
            native_graph_weights,
            native_graph_ids,
            b0,
            micro0,
            launch0,
            0,
            native_graph_x,
        )

    native_graph_call()
    torch.cuda.synchronize()
    native_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(native_graph):
        native_graph_call()
    torch.cuda.synchronize()
    native_step_max_abs = []
    native_step_cosine = []
    for step_index in range(8):
        native_graph_x.copy_(torch.randn_like(native_graph_x) * 0.1)
        native_graph_ids.copy_(
            torch.arange(TOPK, dtype=torch.int32, device="cuda")
            .roll(step_index + 1)
            .view(1, -1)
        )
        native_graph_weights.copy_(
            torch.softmax(torch.randn_like(native_graph_weights), dim=-1)
        )
        native_graph.replay()
        torch.cuda.synchronize()
        graph_value = b0.output.clone()
        native_graph_call()
        torch.cuda.synchronize()
        graph_f32, eager_f32 = graph_value.float(), b0.output.float()
        native_step_max_abs.append(float((graph_f32 - eager_f32).abs().max().item()))
        native_step_cosine.append(
            float(torch.nn.functional.cosine_similarity(graph_f32, eager_f32).item())
        )

    if min(native_step_cosine) < 0.999 or max(native_step_max_abs) > 5e-4:
        raise RuntimeError(
            "dynamic native-MXFP4 CUDA graph mismatch: "
            f"cosine={native_step_cosine}, max_abs={native_step_max_abs}"
        )

    results = {}
    for name, fn in (("serial", serial), ("hybrid", hybrid)):
        eager = _latencies_ms(fn, args.iterations, args.warmup)
        graph = _graph_latencies_ms(fn, args.iterations, args.warmup)
        results[name] = {
            "eager_median_us": statistics.median(eager) * 1000,
            "graph_median_us": statistics.median(graph) * 1000,
        }
    results["shape"] = {
        "kept": args.kept,
        "nf3": nf3,
        "hidden": HIDDEN,
        "intermediate": INTERMEDIATE,
        "topk": TOPK,
        "tiles": tiles,
        "schedule_whole_tiles": args.schedule_whole_tiles,
    }
    results["correctness"] = {
        "cosine": cosine,
        "max_abs": max_abs,
        "reference_abs_max": ref_abs,
        "relative_max": rel_max,
        "dynamic_two_layer_graph_max_abs": graph_step_max_abs,
        "dynamic_two_layer_graph_cosine": graph_step_cosine,
        "dynamic_native_graph_max_abs": native_step_max_abs,
        "dynamic_native_graph_cosine": native_step_cosine,
    }
    results["hybrid_speedup_eager"] = (
        results["serial"]["eager_median_us"] / results["hybrid"]["eager_median_us"]
    )
    results["hybrid_speedup_graph"] = (
        results["serial"]["graph_median_us"] / results["hybrid"]["graph_median_us"]
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
