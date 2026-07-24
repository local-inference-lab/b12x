#!/usr/bin/env python
"""Per-hop activation-quantization ablation for one routed-MoE layer.

Answers: of the ~0.009 KLD the runtime W6A6 activation path adds on MoE,
how much comes from each quantized activation hop, and what does each format
buy? The two hops are:

  FC1: the expert input (post-layernorm hidden state) quantized before
       gate_proj/up_proj
  FC2: the post-SwiGLU intermediate re-quantized inside the kernel before
       down_proj

The simulation loads REAL expert weights (one layer of the BF16 source
checkpoint), routes with the layer's real router gate, and runs a pure-torch
expert forward in fp32 where each hop is independently quantize-dequantized
per 32-element UE8M0 block (exactly the kernel's scale rule: amax-ceil
power-of-2). Weights are QDQ'd with the production E2M3 rule, so the
"weights only" row reproduces the dequant-BF16 checkpoint condition and every
activation variant shows its *additional* error on top.

Formats ablated per hop: e3m2 (current default), e2m3 (config-flip tested),
e4m3 (MXFP8 — the W6A8 candidate; same UE8M0 per-32 scaling, FP8 codes).

Inputs are unit-RMS gaussian hidden states; absolute numbers are a proxy, but
the FC1-vs-FC2 split and the format ranking are what scope the kernel work.

Example:
    python scripts/ablate_moe_act_quant.py \
        --model /media/fmodels/Qwen/Qwen3.6-35B-A3B --layer 20 --tokens 512
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from sparkinfer._lib.fp6 import _decode_fp6_e2m3, _decode_fp6_e3m2
from sparkinfer.quantization.mxfp6.fp6_safetensors_export import _layer_expert_matrices
from sparkinfer.quantization.mxfp6.model_fp6 import SafetensorsModel, discover_moe_experts

_FMT_MAX = {"e3m2": 28.0, "e2m3": 7.5, "e4m3": 448.0}
_BLOCK = 32


def _fp6_value_lut(fmt: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Sorted unique representable FP6 values + bucket boundaries (midpoints)."""
    decode = _decode_fp6_e3m2 if fmt == "e3m2" else _decode_fp6_e2m3
    vals = sorted({decode(c) for c in range(64)})
    lut = torch.tensor(vals, dtype=torch.float32, device=device)
    bounds = (lut[1:] + lut[:-1]) / 2
    return lut, bounds


_LUT_CACHE: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}


def _round_to_format(s: torch.Tensor, fmt: str) -> torch.Tensor:
    """Round scaled values to the nearest representable code value."""
    if fmt == "e4m3":
        return s.clamp(-_FMT_MAX[fmt], _FMT_MAX[fmt]).to(torch.float8_e4m3fn).float()
    key = (fmt, str(s.device))
    if key not in _LUT_CACHE:
        _LUT_CACHE[key] = _fp6_value_lut(fmt, s.device)
    lut, bounds = _LUT_CACHE[key]
    idx = torch.bucketize(s.clamp(-_FMT_MAX[fmt], _FMT_MAX[fmt]).contiguous(), bounds)
    return lut[idx]


def qdq_mx(x: torch.Tensor, fmt: str | None, rule: str = "ceil") -> torch.Tensor:
    """Quantize-dequantize per 32-element block with UE8M0 power-of-2 scales.

    rule="ceil": ``2^ceil(log2(amax/fmt_max))`` (amax-fit, never clips) —
    matches ``_ue8m0_scale_from_block_max`` + the in-kernel quantizers.
    rule="mse": per block, also try the one-smaller exponent (2x finer steps,
    clips the block max to fmt_max) and keep whichever reconstruction has the
    lower block MSE — the Phase 2 candidate, same one-byte scale storage.
    """
    if fmt is None:
        return x.float()
    v = x.float().reshape(-1, _BLOCK)
    amax = v.abs().amax(dim=1, keepdim=True)
    e = torch.ceil(torch.log2((amax / _FMT_MAX[fmt]).clamp(min=1e-38)))
    scale = torch.exp2(e.clamp(-127.0, 128.0))
    q = _round_to_format(v / scale, fmt) * scale
    if rule == "mse":
        scale_lo = scale * 0.5
        q_lo = _round_to_format(v / scale_lo, fmt) * scale_lo
        better = ((q_lo - v) ** 2).sum(dim=1, keepdim=True) < (
            (q - v) ** 2
        ).sum(dim=1, keepdim=True)
        q = torch.where(better, q_lo, q)
    q = torch.where(amax > 0, q, torch.zeros_like(q))
    return q.reshape(x.shape)


def moe_forward(
    x: torch.Tensor,        # (M, K) bf16 hidden states
    gate_w: torch.Tensor,   # (E, N, K) fp32 (already weight-QDQ'd or not)
    up_w: torch.Tensor,     # (E, N, K)
    down_w: torch.Tensor,   # (E, K, N)
    topk_ids: torch.Tensor,     # (M, topk)
    topk_wts: torch.Tensor,     # (M, topk) fp32
    *,
    fc1_fmt: str | None,
    fc2_fmt: str | None,
    rule: str = "ceil",
) -> torch.Tensor:
    """Routed-expert forward with per-hop activation QDQ (fp32 math)."""
    m, k = x.shape
    y = torch.zeros(m, k, dtype=torch.float32, device=x.device)
    x_q = qdq_mx(x, fc1_fmt, rule)  # one quant per token, shared across its experts
    for e in topk_ids.unique().tolist():
        tok, slot = (topk_ids == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        xe = x_q[tok]
        inter = F.silu(xe @ gate_w[e].T) * (xe @ up_w[e].T)
        inter = qdq_mx(inter, fc2_fmt, rule)
        out = inter @ down_w[e].T
        y.index_add_(0, tok, out * topk_wts[tok, slot].unsqueeze(1))
    return y


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="BF16 source model directory")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weight-fmt", default="e2m3", choices=["e2m3", "e3m2"])
    parser.add_argument(
        "--gate-up-order",
        default="gate_up",
        choices=["gate_up", "up_gate"],
        help="row order inside fused gate_up expert tensors",
    )
    parser.add_argument(
        "--x-from",
        default=None,
        help="*.safetensors file with real (tokens, K) hidden states "
        "(from scripts/capture_hidden_states.py; pickle rejected); "
        "default: unit-RMS gaussian",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model = SafetensorsModel(args.model)
    scheme = discover_moe_experts(model)
    if scheme is None:
        raise SystemExit(f"no MoE experts discovered under {args.model}")
    if args.layer not in scheme.layers:
        raise SystemExit(
            f"layer {args.layer} has no experts; MoE layers: "
            f"{scheme.layers[0]}..{scheme.layers[-1]} ({len(scheme.layers)} total)"
        )
    gate_bf, up_bf, down_bf = _layer_expert_matrices(
        model, scheme, args.layer, str(device), True, args.gate_up_order
    )
    e, n, k = gate_bf.shape
    print(
        f"[ablate] layer={args.layer} experts={e} N={n} K={k} "
        f"tokens={args.tokens} topk={args.topk} device={device.type}"
    )

    # Hidden states: real captured ones if provided, else unit-RMS gaussian.
    if args.x_from:
        from sparkinfer.quantization.mxfp6.calib_io import load_hidden_states

        x = load_hidden_states(args.x_from, device=device)
        if x.shape[-1] != k:
            raise SystemExit(f"--x-from K={x.shape[-1]} but experts expect K={k}")
        x = x.reshape(-1, k)
        if x.shape[0] > args.tokens:
            x = x[torch.randperm(x.shape[0], device=device)[: args.tokens]]
        x = x.to(torch.bfloat16)
        print(f"[ablate] using {x.shape[0]} real hidden states from {args.x_from}")
    else:
        x = torch.randn(args.tokens, k, device=device).to(torch.bfloat16)
        print("[ablate] using gaussian hidden states (no --x-from)")
    prefix = scheme.prefix_template.format(L=args.layer)
    try:
        router = model.get_tensor(f"{prefix}.gate.weight").to(device, torch.float32)
        logits = x.float() @ router.T
        print(f"[ablate] routing with real gate ({prefix}.gate.weight)")
    except Exception:
        logits = torch.randn(args.tokens, e, device=device)
        print("[ablate] router gate not found; using random logits")
    probs = logits.softmax(dim=-1)
    topk_wts, topk_ids = probs.topk(args.topk, dim=-1)
    topk_wts = (topk_wts / topk_wts.sum(dim=-1, keepdim=True)).float()

    # Oracle: BF16 weights, no activation quant.
    gate_f, up_f, down_f = gate_bf.float(), up_bf.float(), down_bf.float()
    y_ref = moe_forward(
        x, gate_f, up_f, down_f, topk_ids, topk_wts, fc1_fmt=None, fc2_fmt=None
    )

    # Production weight QDQ (per-32 UE8M0, same rule as the export quantizer).
    wf = args.weight_fmt
    gate_q, up_q = qdq_mx(gate_f, wf), qdq_mx(up_f, wf)
    down_q = qdq_mx(down_f, wf)
    del gate_f, up_f, down_f

    def run(
        fc1: str | None, fc2: str | None, rule: str = "ceil"
    ) -> tuple[float, float]:
        y = moe_forward(
            x, gate_q, up_q, down_q, topk_ids, topk_wts,
            fc1_fmt=fc1, fc2_fmt=fc2, rule=rule,
        )
        rel = ((y - y_ref).norm() / y_ref.norm()).item()
        cos = F.cosine_similarity(y.reshape(-1), y_ref.reshape(-1), dim=0).item()
        return rel, cos

    base_rel, base_cos = run(None, None)
    print(
        f"\n[ablate] {'variant':<26}{'rel-RMSE':>10}{'cos':>10}{'act-added':>11}"
    )
    print(f"[ablate] {'weights only (' + wf + ')':<26}{base_rel:>10.5f}{base_cos:>10.6f}{'-':>11}")
    for fmt in ("e3m2", "e2m3", "e4m3"):
        for label, fc1, fc2 in (
            (f"FC1 {fmt}", fmt, None),
            (f"FC2 {fmt}", None, fmt),
            (f"FC1+FC2 {fmt}", fmt, fmt),
        ):
            rel, cos = run(fc1, fc2)
            added = (max(rel**2 - base_rel**2, 0.0)) ** 0.5
            print(
                f"[ablate] {label:<26}{rel:>10.5f}{cos:>10.6f}{added:>11.5f}"
            )
    # The W6A8 candidate pairing: FP8 only where it matters most.
    for label, fc1, fc2 in (
        ("FC1 e4m3 + FC2 e2m3", "e4m3", "e2m3"),
        ("FC1 e2m3 + FC2 e4m3", "e2m3", "e4m3"),
    ):
        rel, cos = run(fc1, fc2)
        added = (max(rel**2 - base_rel**2, 0.0)) ** 0.5
        print(f"[ablate] {label:<26}{rel:>10.5f}{cos:>10.6f}{added:>11.5f}")
    # Phase 2 candidate: MSE-chosen scale exponent (ceil vs ceil-1) per block,
    # applied to the runtime activation quantizers. Same one-byte scale storage.
    for fmt in ("e2m3", "e4m3"):
        rel, cos = run(fmt, fmt, rule="mse")
        added = (max(rel**2 - base_rel**2, 0.0)) ** 0.5
        label = f"FC1+FC2 {fmt} mse-scale"
        print(f"[ablate] {label:<26}{rel:>10.5f}{cos:>10.6f}{added:>11.5f}")
    # Weight-side preview of the same MSE rule (Phase 2 on the checkpoint).
    gate_m, up_m = qdq_mx(gate_bf.float(), wf, "mse"), qdq_mx(up_bf.float(), wf, "mse")
    down_m = qdq_mx(down_bf.float(), wf, "mse")
    y = moe_forward(
        x, gate_m, up_m, down_m, topk_ids, topk_wts, fc1_fmt=None, fc2_fmt=None
    )
    rel = ((y - y_ref).norm() / y_ref.norm()).item()
    cos = F.cosine_similarity(y.reshape(-1), y_ref.reshape(-1), dim=0).item()
    label = f"weights only ({wf}) mse"
    print(f"[ablate] {label:<26}{rel:>10.5f}{cos:>10.6f}{'-':>11}")


if __name__ == "__main__":
    main()
