"""Qualify and time unscaled Trellis reconstruction at V4.1 projection dimensions.

This benchmark measures the quantizer-basis tile decoder. It does not measure
expert rotations, a matrix multiplication, or complete MoE execution.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from benchmarks.benchmark_roce_oneshot import _source_state
from benchmarks.benchmark_sm103_moe import timed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, choices=(2, 3, 4), default=3)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--intermediate", type=int, default=2304)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.hidden <= 0
        or args.hidden % 16
        or args.intermediate <= 0
        or args.intermediate % 16
    ):
        parser.error("projection dimensions must be positive multiples of 16")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        parser.error("this benchmark requires physical SM103 hardware")
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from b12x._lib.quant.sqg_e4m3 import sqg_xor_cheb_t12_direct_lut_cpu
    from b12x.moe._shared.kernels.sm103.launch import pointer
    from b12x.moe._shared.kernels.sm103.trellis import ReconstructTrellisTiles
    from tests._reference.trellis_moe import build_trellis_weight

    device = torch.device("cuda", torch.cuda.current_device())
    packed, expected = build_trellis_weight(
        torch.Generator().manual_seed(103),
        1,
        args.hidden,
        args.intermediate,
        args.bits,
        device,
    )
    capacity = packed.numel() // (16 * args.bits)
    out = torch.empty((capacity, 16, 16), device=device, dtype=torch.bfloat16)
    lut = sqg_xor_cheb_t12_direct_lut_cpu().to(device)
    params = [
        pointer(t, x)
        for t, x in (
            (cutlass.Uint32, packed),
            (cutlass.Uint8, lut),
            (cutlass.BFloat16, out),
        )
    ]
    params += [
        cutlass.Int32(capacity),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    ]
    compiled = cute.compile(
        ReconstructTrellisTiles(args.bits, capacity),
        *params,
        options="--gpu-arch=sm_103a",
    )
    compiled(*params)
    dense = (
        out.reshape(1, args.intermediate // 16, args.hidden // 16, 16, 16)
        .permute(0, 2, 3, 1, 4)
        .reshape_as(expected)
    )
    torch.testing.assert_close(dense.float(), expected, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(*params)
    out.fill_(float("nan"))
    before = torch.cuda.memory_allocated()
    graph.replay()
    torch.cuda.synchronize()
    assert before == torch.cuda.memory_allocated()
    dense = (
        out.reshape(1, args.intermediate // 16, args.hidden // 16, 16, 16)
        .permute(0, 2, 3, 1, 4)
        .reshape_as(expected)
    )
    torch.testing.assert_close(dense.float(), expected, atol=0, rtol=0)
    samples = timed(graph.replay, 20, 10)
    graph.reset()
    receipt = {
        **_source_state(),
        "command": sys.argv,
        "status": "exact reconstruction passed",
        "stage": "quantizer-basis tiles; full expert unsupported",
        "bits": args.bits,
        "hidden": args.hidden,
        "intermediate": args.intermediate,
        "experts": 1,
        "gpu": str(torch.cuda.get_device_properties(device)),
        "gpu_uuid": str(torch.cuda.get_device_properties(device).uuid),
        "graph_us": samples,
        "median_graph_us": statistics.median(samples),
        "compressed_bytes": packed.numel() * packed.element_size(),
        "output_bytes": out.numel() * out.element_size(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
