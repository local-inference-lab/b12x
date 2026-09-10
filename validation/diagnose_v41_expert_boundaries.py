"""Localize V4.1 FC1/midquant/FC2/reduction differences without changing tolerances."""

import argparse
import importlib.util
import json

import torch

from b12x.moe import fused_moe
from tests.moe.test_deepseek_v41_experts import _decode, _mxfp8, _oracle, _setup


def report(label, actual, expected, limit=8):
    difference = (actual.float() - expected.float()).abs()
    bad = difference != 0
    coordinates = bad.nonzero()[:limit]
    examples = []
    for coordinate in coordinates.tolist():
        index = tuple(coordinate)
        examples.append({"index": coordinate, "actual": actual[index].item(), "expected": expected[index].item()})
    print(json.dumps({"stage": label, "mismatches": bad.sum().item(), "max_abs": difference.max().item() if difference.numel() else 0, "examples": examples}), flush=True)


def quantized(x):
    blocks = x.float().reshape(x.shape[0], -1, 32)
    scales = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(-1, keepdim=True).clamp_min(1.e-4) / 448.)))
    codes = (blocks / scales).to(torch.float8_e4m3fn)
    return codes.reshape(x.shape), (torch.log2(scales).squeeze(-1) + 127).to(torch.uint8)


def activation(projected, route_weights):
    gate, up = projected.bfloat16().float().chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate.clamp(max=10.)) * up.clamp(-10., 10.) * route_weights[:, None]).bfloat16()


def reference_gemm(module, codes, scales, packed, weight_scales):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        return module.fp4_gemm(
            codes.contiguous(), scales.contiguous().view(torch.float8_e8m0fnu),
            packed.contiguous().view(torch.float4_e2m1fn_x2),
            weight_scales.contiguous().view(torch.float8_e8m0fnu),
            scale_dtype=torch.float8_e8m0fnu, act_block_size=32,
        )
    finally:
        torch.set_default_dtype(previous)


def rounding_stability(label, actual, a, b):
    exact = a.double() @ b.double().T
    # Diagnostic only: standard FP32 dot/reduction forward-error envelope;
    # no assertion tolerance is changed and no native result is substituted.
    steps = a.shape[1] // 32 - 1 + 31
    gamma = steps * 2.0**-24 / (1 - steps * 2.0**-24)
    bound = gamma * (a.double().abs() @ b.double().abs().T)
    lower = (exact - bound).bfloat16().double()
    upper = (exact + bound).bfloat16().double()
    outside = (actual.double() < lower) | (actual.double() > upper)
    rounded = exact.bfloat16().double()
    mismatched = actual.double() != rounded
    midpoint = (actual.double() + rounded) * .5
    examples = []
    for coord in mismatched.nonzero()[:12].tolist():
        index = tuple(coord)
        examples.append({
            "index": coord, "native": actual[index].item(),
            "exact": exact[index].item(), "exact_bf16": rounded[index].item(),
            "rounding_midpoint": midpoint[index].item(),
            "distance_to_midpoint": (exact[index] - midpoint[index]).abs().item(),
            "fp32_forward_bound": bound[index].item(),
        })
    print(json.dumps({"stage": label, "outside_fp32_rounding_envelope": outside.sum().item(), "examples": examples}), flush=True)


def snapshot(binding, tokens, topk, hidden, intermediate):
    rows = binding.token_map.numel()
    raw = binding.materialized_intermediate.view(torch.uint8).flatten()
    payload = raw[:rows * intermediate].reshape(rows, intermediate)
    scales = raw[rows * intermediate:rows * intermediate + rows * intermediate // 32].reshape(intermediate // 128, rows, 4).permute(1, 0, 2).reshape(rows, intermediate // 32)
    packed = torch.zeros((tokens * topk, intermediate), device=raw.device, dtype=torch.uint8)
    packed_scales = torch.zeros((tokens * topk, intermediate // 32), device=raw.device, dtype=torch.uint8)
    tail = int(binding.expert_tile_base[-1].item())
    for tile in range(tail):
        valid = int(binding.task_valid_rows[tile * (intermediate // 128)].item())
        physical = torch.arange(tile * 64, tile * 64 + valid, device=raw.device)
        pair = binding.token_map[physical].long()
        packed[pair] = payload[physical]
        packed_scales[pair] = scales[physical]
    return {
        "mid_payload_by_route": packed, "mid_scales_by_route": packed_scales,
        "route_output": binding.route_output.reshape(-1, hidden)[:tokens * topk].clone(),
        "input_payload": binding.packed_a_flat.view(torch.uint8).flatten()[:tokens * hidden].clone(),
        "input_scales": binding.scale_flat.view(torch.uint8).flatten()[:tokens * hidden // 32].clone(),
        "row_counts": binding.row_counts.clone(),
        "expert_tile_base": binding.expert_tile_base.clone(),
        "input_gs": binding.input_gs.clone(), "down_input_scale": binding.down_input_scale.clone(),
        "barrier_count": binding.barrier_count.clone(), "barrier_epoch": binding.barrier_epoch.clone(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=48)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--tokens", type=int, default=65)
    parser.add_argument("--reference-kernel", help="Pinned inference/kernel.py for actual published GPU GEMMs")
    parser.add_argument("--graph-mutation", action="store_true")
    args = parser.parse_args()
    reference_module = None
    if args.reference_kernel:
        spec = importlib.util.spec_from_file_location("deepseek_v41_reference_kernel", args.reference_kernel)
        reference_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference_module)
    torch.backends.cuda.matmul.allow_tf32 = False
    hidden, intermediate = 5120, 2304
    plan, experts, scratch, checkpoint, x, ids, weights, output = _setup(args.experts, hidden, intermediate, args.topk, args.tokens)
    binding = fused_moe.bind(plan, scratch=scratch, experts=experts, a=x, topk_ids=ids, topk_weights=weights, output=output, input_scales_static=True)
    fused_moe.run(binding=binding)
    if args.graph_mutation:
        baseline = output.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fused_moe.run(binding=binding)
        graph.replay()
        report("graph_unchanged_vs_eager", output, baseline)
        ids.fill_(-1)
        ids[::2, 0] = args.experts - 1
        x.mul_(.5)
        weights.mul_(.8)
        graph.replay()
        replay_output = output.clone()
        replay_state = snapshot(binding, args.tokens, args.topk, hidden, intermediate)
        expected_mutated = _oracle(x, ids, weights, checkpoint)
        report("graph_mutated_vs_oracle", replay_output, expected_mutated)
        for repeat in range(3):
            fused_moe.run(binding=binding)
            report(f"eager_mutated_{repeat}_vs_graph", output, replay_output)
            report(f"eager_mutated_{repeat}_vs_oracle", output, expected_mutated)
            eager_state = snapshot(binding, args.tokens, args.topk, hidden, intermediate)
            for name in replay_state:
                report(f"graph_vs_eager_{repeat}/{name}", replay_state[name], eager_state[name], limit=4)
            bad = ~torch.isclose(output, expected_mutated, rtol=.01, atol=.08)
            print(json.dumps({"stage": f"eager_mutated_{repeat}/bad_columns_per_token", "counts": bad.sum(1).tolist()}), flush=True)
        # The remaining boundary diagnostic now examines the same mutated
        # eager binding, distinguishing graph lifetime from high-expert math.
    expected = _oracle(x, ids, weights, checkpoint)
    report("final_native_vs_k32_fp32_oracle", output, expected)
    rejected = ~torch.isclose(output, expected, rtol=.01, atol=.08)
    print(json.dumps({"stage": "original_tolerance_rejections", "coordinates": rejected.nonzero().tolist()}), flush=True)

    rows_capacity = binding.token_map.numel()
    raw = binding.materialized_intermediate.view(torch.uint8).flatten()
    native_mid_codes = raw[:rows_capacity * intermediate].reshape(rows_capacity, intermediate).view(torch.float8_e4m3fn)
    native_mid_scales = raw[rows_capacity * intermediate:rows_capacity * intermediate + rows_capacity * intermediate // 32].reshape(intermediate // 128, rows_capacity, 4).permute(1, 0, 2).reshape(rows_capacity, intermediate // 32)
    native_route = binding.route_output.reshape(-1, hidden)[:args.tokens * args.topk].clone()
    report("final_native_vs_native_route_sum", output, native_route.reshape(args.tokens, args.topk, hidden).float().sum(1))
    x8 = _mxfp8(x)
    expected_input_codes, expected_input_scales = quantized(x)
    native_input_codes = binding.packed_a_flat.view(torch.uint8).flatten()[:x.numel()].reshape(x.shape).view(torch.float8_e4m3fn)
    native_input_scales = binding.scale_flat.view(torch.uint8).flatten()[:x.numel() // 32].reshape(args.tokens, hidden // 32)
    report("input_payload", native_input_codes.float(), expected_input_codes.float())
    report("input_scales", native_input_scales, expected_input_scales)
    sf13, sf2 = checkpoint[1], checkpoint[3]
    bases = binding.expert_tile_base.cpu().tolist()
    exact_routes = torch.zeros_like(native_route, dtype=torch.float32)
    fp32_native_mid_routes = torch.zeros_like(exact_routes)
    fp64_native_mid_routes = torch.zeros_like(exact_routes)
    ntiles = intermediate // 128
    for expert in range(args.experts):
        physical = []
        for tile in range(bases[expert], bases[expert + 1]):
            valid = int(binding.task_valid_rows[tile * ntiles].item())
            physical.extend(range(tile * 64, tile * 64 + valid))
        if not physical:
            continue
        physical = torch.tensor(physical, device=x.device)
        pair = binding.token_map[physical].long()
        tokens = pair // args.topk
        route_weights = weights.flatten()[pair]
        w13 = _decode(checkpoint[0][expert], sf13[expert])
        w2 = _decode(checkpoint[2][expert], sf2[expert])
        projection32 = x8[tokens] @ w13.T
        projection64 = (x8[tokens].double() @ w13.double().T).float()
        # Published reference accumulates independently scaled K32 partials.
        projection_k32 = torch.zeros_like(projection32)
        for k in range(0, hidden, 32):
            projection_k32.add_(x8[tokens, k:k + 32] @ w13[:, k:k + 32].T)
        report(f"e{expert}/fc1_fullk_fp32_vs_exact_bf16", projection32.bfloat16(), projection64.bfloat16())
        report(f"e{expert}/fc1_k32_fp32_vs_exact_bf16", projection_k32.bfloat16(), projection64.bfloat16())
        mid32 = activation(projection32, route_weights)
        mid64 = activation(projection64, route_weights)
        mid_k32 = activation(projection_k32, route_weights)
        actual_codes = native_mid_codes[physical]
        actual_scales = native_mid_scales[physical]
        for name, mid in (("fullk_fp32", mid32), ("exact", mid64), ("k32_fp32", mid_k32)):
            codes, scales = quantized(mid)
            report(f"e{expert}/mid_payload_vs_{name}", actual_codes.float(), codes.float())
            report(f"e{expert}/mid_scales_vs_{name}", actual_scales, scales)
        actual_mid = actual_codes.float() * torch.exp2(actual_scales.float() - 127).repeat_interleave(32, -1)
        down32 = (actual_mid @ w2.T).bfloat16().float()
        down64 = (actual_mid.double() @ w2.double().T).bfloat16().float()
        down_k32 = torch.zeros_like(down32)
        for k in range(0, intermediate, 32):
            down_k32.add_(actual_mid[:, k:k + 32] @ w2[:, k:k + 32].T)
        down_k32 = down_k32.bfloat16().float()
        report(f"e{expert}/fc2_native_vs_fullk_fp32_using_native_mid", native_route[pair], down32)
        report(f"e{expert}/fc2_native_vs_exact_using_native_mid", native_route[pair], down64)
        report(f"e{expert}/fc2_native_vs_k32_fp32_using_native_mid", native_route[pair], down_k32)
        rounding_stability(f"e{expert}/fc2_bf16_stability", native_route[pair], actual_mid, w2)
        if reference_module is not None:
            published_fc1 = reference_gemm(
                reference_module, expected_input_codes[tokens],
                expected_input_scales[tokens], checkpoint[0][expert], sf13[expert],
            )
            published_mid = activation(published_fc1, route_weights)
            published_codes, published_scales = quantized(published_mid)
            report(f"e{expert}/mid_payload_vs_published_gpu", actual_codes.float(), published_codes.float())
            report(f"e{expert}/mid_scales_vs_published_gpu", actual_scales, published_scales)
            published_fc2 = reference_gemm(
                reference_module, actual_codes, actual_scales,
                checkpoint[2][expert], sf2[expert],
            )
            report(f"e{expert}/fc2_native_vs_published_gpu_using_native_mid", native_route[pair], published_fc2)
        exact_mid = _mxfp8(mid64)
        exact_routes[pair] = (exact_mid.double() @ w2.double().T).bfloat16().float()
        fp32_native_mid_routes[pair] = down32
        fp64_native_mid_routes[pair] = down64
        del w13, w2, projection32, projection64, projection_k32
    for name, routes in (("exact_fc1_exact_fc2", exact_routes), ("native_mid_fullk_fp32_fc2", fp32_native_mid_routes), ("native_mid_exact_fc2", fp64_native_mid_routes)):
        reference = routes.reshape(args.tokens, args.topk, hidden).sum(1)
        report(f"final_native_vs_{name}", output, reference)
        print(json.dumps({"stage": f"original_tolerance_rejections/{name}", "coordinates": (~torch.isclose(output, reference, rtol=.01, atol=.08)).nonzero().tolist()}), flush=True)


if __name__ == "__main__":
    main()
