"""Probe dense-MXFP8 plans for narrow output projections on SM120.

The default shape matches a tensor-parallel projection with 7,168 input
features and 132 output features per rank.  Every plan is checked against a
dequantized reference after poisoning the output buffer, then timed only when
the result is finite and all output rows were written.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import torch

from b12x._lib.dense_gemm import dense_gemm
from b12x.gemm._shared.block_fp8 import (
    quantize_block_fp8_linear_input_mxfp8,
)
from b12x.gemm._shared.wo_mxfp8 import dequantize_mxfp8_rows_torch
from b12x.gemm.mxfp8_linear._kernel import pack_mxfp8_linear_weight


@dataclass(frozen=True)
class Plan:
    name: str
    tile: tuple[int, int]
    load_path: str
    swap_ab: bool


PLANS = (
    Plan("tma-16x64-unswapped", (16, 64), "tma", False),
    Plan("tma-32x64-unswapped", (32, 64), "tma", False),
    Plan("tma-64x64-unswapped", (64, 64), "tma", False),
    Plan("tma-16x128-unswapped", (16, 128), "tma", False),
    Plan("tma-32x128-unswapped", (32, 128), "tma", False),
    Plan("tma-64x128-unswapped", (64, 128), "tma", False),
    Plan("tma-64x32-swapped", (64, 32), "tma", True),
    Plan("tma-64x64-swapped", (64, 64), "tma", True),
    Plan("cpasync-64x64-unswapped", (64, 64), "cpasync", False),
)


def _quantize_modelopt_rows(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, width = map(int, source.shape)
    blocked = source.float().reshape(rows, width // 32, 32)
    max_abs = blocked.abs().amax(dim=-1)
    scale_base = torch.where(max_abs > 0, max_abs / 448.0, torch.ones_like(max_abs))
    scale_exp = torch.ceil(torch.log2(scale_base)).clamp(-127, 127)
    scale_u8 = (scale_exp + 127).to(torch.uint8)
    scale = scale_u8.view(torch.float8_e8m0fnu).float()
    values = (
        (blocked / scale[..., None])
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .reshape(rows, width)
        .contiguous()
    )
    return values, scale_u8.contiguous()


def _launch(
    plan: Plan,
    x_q,
    packed_weight,
    output: torch.Tensor,
    *,
    tokens: int,
    in_features: int,
) -> torch.Tensor:
    return dense_gemm(
        (
            x_q.values.reshape(tokens, in_features, 1),
            x_q.scale_mma,
        ),
        (
            packed_weight.weight.values.reshape(
                packed_weight.out_features,
                packed_weight.padded_in_features,
                1,
            ),
            packed_weight.weight.scale_mma,
        ),
        out=output,
        ab_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype="bfloat16",
        sf_vec_size=32,
        expected_m=tokens,
        mma_tiler_mn=plan.tile,
        load_path=plan.load_path,
        swap_ab=plan.swap_ab,
    )[:, :, 0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=3)
    parser.add_argument("--in-features", type=int, default=7168)
    parser.add_argument("--out-features", type=int, default=132)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument(
        "--plan",
        action="append",
        choices=tuple(plan.name for plan in PLANS),
        help="Run only the named plan; repeat the option to select several plans.",
    )
    args = parser.parse_args()

    torch.manual_seed(20260814)
    source = (
        torch.randn(
            (args.tokens, args.in_features),
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 4
    ).contiguous()
    weight_bf16 = (
        torch.randn(
            (args.out_features, args.in_features),
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 8
    ).contiguous()
    weight, weight_scale = _quantize_modelopt_rows(weight_bf16)
    packed_weight = pack_mxfp8_linear_weight(weight, weight_scale)
    source_q = quantize_block_fp8_linear_input_mxfp8(source)
    source_dequantized = dequantize_mxfp8_rows_torch(
        source_q.values, source_q.scale_rows
    )
    weight_dequantized = dequantize_mxfp8_rows_torch(
        packed_weight.weight.values,
        packed_weight.weight.scale_rows,
    )
    reference = (source_dequantized @ weight_dequantized.T).to(torch.bfloat16)

    results: list[dict[str, object]] = []
    selected_plans = (
        tuple(plan for plan in PLANS if plan.name in args.plan) if args.plan else PLANS
    )
    for plan in selected_plans:
        output = torch.full(
            (args.tokens, args.out_features, 1),
            float("nan"),
            device="cuda",
            dtype=torch.bfloat16,
        )
        try:
            actual = _launch(
                plan,
                source_q,
                packed_weight,
                output,
                tokens=args.tokens,
                in_features=packed_weight.padded_in_features,
            )
            torch.cuda.synchronize()
            finite = bool(torch.isfinite(actual).all().item())
            mismatches = int((actual != reference).sum().item())
            max_abs_error = float(
                torch.nan_to_num(
                    (actual.float() - reference.float()).abs(),
                    nan=float("inf"),
                    posinf=float("inf"),
                    neginf=float("inf"),
                ).max()
            )
            row_mismatches = [
                int((actual[row] != reference[row]).sum().item())
                for row in range(args.tokens)
            ]

            for _ in range(args.warmup):
                _launch(
                    plan,
                    source_q,
                    packed_weight,
                    output,
                    tokens=args.tokens,
                    in_features=packed_weight.padded_in_features,
                )
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.repeats):
                _launch(
                    plan,
                    source_q,
                    packed_weight,
                    output,
                    tokens=args.tokens,
                    in_features=packed_weight.padded_in_features,
                )
            stop.record()
            stop.synchronize()
            milliseconds = float(start.elapsed_time(stop) / args.repeats)
            results.append(
                {
                    "plan": plan.name,
                    "finite": finite,
                    "mismatches": mismatches,
                    "max_abs_error": max_abs_error,
                    "row_mismatches": row_mismatches,
                    "milliseconds": milliseconds,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "plan": plan.name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    print(
        json.dumps(
            {
                "shape": [args.tokens, args.out_features, args.in_features],
                "warmup": args.warmup,
                "repeats": args.repeats,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
