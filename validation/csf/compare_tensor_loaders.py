"""Compare prepared GPU tensors against a file-based CSF checkpoint reader.

Requires a matching vLLM CSF integration and one CUDA device. The reference
module must export read_mxfp4_csf_layer or read_nvfp4_csf_layer. Both arms use
the same weight plan and prepare_weights. Compare FP4 bytes, compressed scale
batches, calibration and layout after preparation. Scratch contents are excluded;
both arms must retain the caller-owned storage. No MoE inference is run.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch


def compare(first, second, path="weights"):
    """Check tensor bytes, metadata and scratch ownership recursively."""
    if isinstance(first, torch.Tensor):
        assert first.dtype == second.dtype and first.shape == second.shape, path
        a = first.reshape(-1).view(torch.uint8)
        b = second.reshape(-1).view(torch.uint8)
        assert torch.equal(a, b), path
        return [
            {
                "field": path,
                "shape": list(first.shape),
                "dtype": str(first.dtype),
                "bytes": a.numel(),
                "sha256": hashlib.sha256(a.cpu().numpy()).hexdigest(),
            }
        ]
    if dataclasses.is_dataclass(first):
        result = []
        for field in dataclasses.fields(first):
            a, b = getattr(first, field.name), getattr(second, field.name)
            if field.name in (
                "w13_scale_scratch",
                "w2_scale_scratch",
                "w13_block_scales",
                "w2_block_scales",
            ):
                assert a is b, field.name
            else:
                result.extend(compare(a, b, path + "." + field.name))
        return result
    if isinstance(first, dict):
        assert first.keys() == second.keys(), path
        return [
            entry
            for key in first
            for entry in compare(first[key], second[key], path + "." + key)
        ]
    assert first == second, path
    return []


def prepared_payload(prepared, scratch):
    """Select persistent numeric state and validate shared expansion storage."""
    impl = prepared._impl
    for actual, supplied in zip(
        (impl.w1_blockscale, impl.w2_blockscale), scratch, strict=True
    ):
        assert actual.data_ptr() == supplied.data_ptr()
        assert actual.numel() == supplied.numel()
    result = {
        name: getattr(impl, name)
        for name in (
            "a1_gscale",
            "a2_gscale",
            "w1_fp4",
            "w2_fp4",
            "w1_alphas",
            "w2_alphas",
        )
    }
    result["packing"] = prepared.plan.prepared_format
    result["activation"] = prepared.plan.activation
    if impl.nvfp4_csf is not None:
        result["scales13"] = impl.nvfp4_csf.first
        result["scales2"] = impl.nvfp4_csf.second
    else:
        result["scales13"] = impl.representation.value.x4t_w13_scale
        result["scales2"] = impl.representation.value.x4t_w2_scale
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-reader", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-id", required=True, help="Repository and immutable revision"
    )
    parser.add_argument("--codec", choices=("mxfp4", "nvfp4"), required=True)
    parser.add_argument("--nonlinearity", choices=("silu", "situ"), default="silu")
    parser.add_argument("--swiglu-limit", type=float)
    parser.add_argument("--activation", choices=("a4", "a16"))
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--num-experts", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--intermediate-size", type=int, required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not __debug__:
        parser.error("Byte-parity assertions require Python without -O")
    if args.output.exists():
        parser.error("Use an output path that does not already exist")
    if (
        args.tp <= 0
        or args.intermediate_size % args.tp
        or any(not 0 <= rank < args.tp for rank in args.ranks)
    ):
        parser.error("TP must divide the intermediate size and contain every rank")
    spec = importlib.util.spec_from_file_location(
        "reference_csf", args.reference_reader
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    before = getattr(reference, f"read_{args.codec}_csf_layer")
    if args.codec == "mxfp4":
        from vllm.model_executor.model_loader.mxfp4_csf_loader import (
            read_mxfp4_csf_layer as load,
        )
    else:
        from vllm.model_executor.model_loader.nvfp4_csf_loader import (
            read_nvfp4_csf_layer as load,
        )
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    e, h, n = args.num_experts, args.hidden_size, args.intermediate_size
    local = n // args.tp
    mx = args.codec == "mxfp4"
    shapes = (
        ((e, h // 32, 2 * local), (e, local // 32, h))
        if mx
        else ((e, 2 * local, h // 16), (e, h, local // 16))
    )
    dtype = torch.uint8 if mx else torch.float8_e4m3fn
    scratch13, scratch2 = [torch.empty(s, device="cuda", dtype=dtype) for s in shapes]
    result = {
        "checkpoint": args.checkpoint_id,
        "layer": args.layer,
        "tp": args.tp,
        "command": sys.argv,
        "reference_reader_sha256": hashlib.sha256(
            args.reference_reader.read_bytes()
        ).hexdigest(),
        "manifest_sha256": hashlib.sha256(
            (args.checkpoint / "manifest.json").read_bytes()
        ).hexdigest(),
        "default_dtype": str(torch.get_default_dtype()),
        "ranks": [],
    }
    from b12x.moe import fused_moe

    plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(
            format="fp4_e8m0_k32" if mx else "modelopt_nvfp4",
            w13_layout="w31" if mx else "w13",
        ),
        activation=fused_moe.ActivationSpec(
            mode=args.activation or ("a16" if mx else "a4"),
            nonlinearity=args.nonlinearity,
            swiglu_limit=args.swiglu_limit,
            io_dtype=torch.bfloat16,
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=e, hidden_size=h, intermediate_size=local
        ),
    )
    result["preparation"] = {
        "activation": plan.activation.mode.value,
        "nonlinearity": args.nonlinearity,
        "swiglu_limit": args.swiglu_limit,
        "packing": plan.prepared_format.packing.value,
        "scope": "persistent prepared tensors; shared scratch ownership; no inference",
    }
    for rank in args.ranks:
        kwargs = dict(
            num_experts=e,
            hidden_size=h,
            intermediate_size=n,
            tp_rank=rank,
            tp_size=args.tp,
            device="cuda",
            w13_scale_scratch=scratch13,
            w2_scale_scratch=scratch2,
        )
        a = before(args.checkpoint, args.layer, **kwargs)
        b = load(args.checkpoint, args.layer, **kwargs)
        pa = fused_moe.prepare_weights(plan=plan, weights=a)
        pb = fused_moe.prepare_weights(plan=plan, weights=b)
        tensors = compare(
            prepared_payload(pa, (scratch13, scratch2)),
            prepared_payload(pb, (scratch13, scratch2)),
            path="prepared",
        )
        result["ranks"].append({"rank": rank, "tensors": tensors, "equal": True})
        print(
            json.dumps({"rank": rank, "equal": True, "tensors": len(tensors)}),
            flush=True,
        )
        del a, b, pa, pb
        gc.collect()
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
