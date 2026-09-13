"""Qualification oracle for materialized NVFP4 with BF16 stage boundaries.

FC1 is rounded before SiLU, SiLU's output is rounded before requantization,
and FC2 is rounded before FP32 router weighting and accumulation. The final
output is BF16. This oracle is used only by tests and offline qualification;
production bind/run paths do not import it.
"""

import torch
import torch.nn.functional as F


def unswizzle(scales, rows, k):
    e = scales.shape[0]
    return (
        scales.view(torch.float8_e4m3fn)
        .reshape(e, rows // 128, k // 64, 32, 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .reshape(e, rows, k // 16)
        .float()
    )


def unpack(packed, scales):
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=packed.device,
    )
    codes = packed.view(torch.uint8)
    codes = torch.stack((codes & 15, codes >> 4), dim=-1).flatten(-2)
    return lut[codes.long()] * scales.repeat_interleave(16, dim=-1)


def quantize_dequantize(x, reciprocal_global):
    """Nearest-even E2M1 with E4M3 block scales and BF16 input boundaries."""
    from b12x._lib.intrinsics import fp4_quantize_values_torch

    x = x.to(torch.bfloat16).float()
    groups = x.reshape(-1, 16)
    sf = (
        (reciprocal_global * (groups.abs().amax(-1, keepdim=True) / 6))
        .clamp(max=448.0)
        .to(torch.float8_e4m3fn)
        .float()
    )
    multiplier = torch.where(sf > 0, reciprocal_global / sf, torch.zeros_like(sf))
    return (fp4_quantize_values_torch(groups * multiplier) * sf).reshape_as(x)


def reference(a, experts, ids, weights, *, limit=None):
    k, n = experts.hidden_size, experts.intermediate_size
    w1 = unpack(experts.w1_fp4, unswizzle(experts.w1_blockscale, 2 * n, k))
    w2 = unpack(experts.w2_fp4, unswizzle(experts.w2_blockscale, k, n))
    output = torch.zeros_like(a, dtype=torch.float32)
    for token in range(a.shape[0]):
        for rank in range(ids.shape[1]):
            expert = int(ids[token, rank])
            if not 0 <= expert < experts.num_experts:
                continue
            s1 = experts.a1_gscale.flatten()[
                0 if experts.a1_gscale.numel() == 1 else expert
            ]
            s2 = experts.a2_gscale.flatten()[
                0 if experts.a2_gscale.numel() == 1 else expert
            ]
            x = quantize_dequantize(a[token], s1)
            fc1 = (
                (F.linear(x, w1[expert]) * experts.w1_alphas[expert])
                .to(torch.bfloat16)
                .float()
            )
            first, second = fc1.chunk(2)
            gate, up = (
                (first, second) if experts.w13_layout == "w31" else (second, first)
            )
            if limit is not None:
                gate = gate.clamp(max=limit)
                up = up.clamp(-limit, limit)
            middle = quantize_dequantize(F.silu(gate) * up, s2)
            fc2 = (
                (F.linear(middle, w2[expert]) * experts.w2_alphas[expert])
                .to(torch.bfloat16)
                .float()
            )
            output[token] += fc2 * weights[token, rank]
    return output.to(torch.bfloat16)
