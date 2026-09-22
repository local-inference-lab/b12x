"""Independent GEMM/activation/reduction; uses the native activation quantizer."""

import torch
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    is_global_sf_supported_for_nvfp4_backend,
)


def reference(raw, x, ids, weights, backend):
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=x.device,
    )

    def dequant(packed, scales, global_scale):
        packed = packed.to(x.device)
        scales = scales.to(x.device).float()
        values = torch.stack(
            (lut[(packed & 15).long()], lut[(packed >> 4).long()]), -1
        ).flatten(-2)
        return values * scales.repeat_interleave(16, -1) * global_scale

    def quantized(a, scale):
        gs = torch.tensor(1 / scale, device=x.device, dtype=torch.float32)
        packed, sf = ops.scaled_fp4_quant(
            a.contiguous(), gs, is_sf_swizzled_layout=False
        )
        return dequant(packed, sf[: a.shape[0], : a.shape[1] // 16], scale)

    global_input = is_global_sf_supported_for_nvfp4_backend(backend)
    a1 = raw["w13_input_scale"].max().item() if global_input else None
    a2 = raw["w2_input_scale"].max().item() if global_input else None
    result = torch.empty((*ids.shape, x.shape[1]), device=x.device, dtype=torch.float32)
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for expert in ids.unique().cpu().tolist():
            rows, routes = (ids == expert).nonzero(as_tuple=True)
            qx = quantized(
                x[rows],
                a1 if a1 is not None else raw["w13_input_scale"][expert].max().item(),
            )
            w13 = dequant(
                raw["w13_weight"][expert],
                raw["w13_weight_scale"][expert],
                raw["w13_weight_scale_2"][expert, 0].item(),
            )
            gate, up = (qx @ w13.T).bfloat16().float().chunk(2, -1)
            h = (torch.nn.functional.silu(gate) * up).bfloat16()
            qh = quantized(
                h, a2 if a2 is not None else raw["w2_input_scale"][expert].item()
            )
            w2 = dequant(
                raw["w2_weight"][expert],
                raw["w2_weight_scale"][expert],
                raw["w2_weight_scale_2"][expert].item(),
            )
            result[rows, routes] = qh @ w2.T
        return (result * weights.unsqueeze(-1)).sum(1).bfloat16()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old
