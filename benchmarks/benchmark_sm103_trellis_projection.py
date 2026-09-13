"""Compare inline Trellis projection with per-route Torch FP16 projections.

Both arms consume FP16 activations in the quantizer basis. The native arm
reads compressed weights; the Torch baseline retains decoded FP16 weights.
This measures one projection, excluding expert rotations, activation, and
weighted reduction. It is not a full MoE or serving benchmark.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from benchmarks.benchmark_blockscaled_precision import _capture, _check, _clock_checks, _paired, _snapshot
from benchmarks.common import make_l2_flush_fn
from scripts._sm103_source import package_source_sha256, source_identity
from tests._reference.trellis_decode import codebook_tensor, native_weight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codebook", choices=("mcg", "sqg_e4m3", "sqg_fp16"), default="mcg")
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 4, 8, 128])
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--n", type=int, default=2304)
    parser.add_argument("--k", type=int, default=5120)
    parser.add_argument("--experts", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--device-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rows) <= 0 or max(args.rows) > args.capacity or args.warmup < 1 or args.samples < 10:
        parser.error("positive live rows must fit capacity; require warmup and at least ten samples")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        parser.error("this deferred benchmark requires physical SM103")
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from b12x.moe._shared.kernels.sm103.launch import pointer
    from b12x.moe._shared.kernels.sm103.trellis_gemm import RoutedTrellisGemm

    kernel = RoutedTrellisGemm(args.n, args.k, args.experts, args.capacity, bits=args.bits, codebook=args.codebook)
    initial = _snapshot()
    identity = dict(zip(initial["fields"], initial["values"], strict=True))
    if identity["uuid"] != args.device_uuid:
        parser.error("selected GPU does not match --device-uuid")
    record = dict(
        status="running", command=sys.argv, **source_identity(ROOT),
        package_python_sha256=package_source_sha256(ROOT), initial_snapshot=initial,
        toolchain={name: importlib.metadata.version(name) for name in ("torch", "nvidia-cutlass-dsl")},
        gpu_mode=subprocess.check_output(["nvidia-smi", f"--id={args.device_uuid}",
            "--query-gpu=driver_version,compute_mode", "--format=csv,noheader"], text=True).strip(),
        geometry={key: getattr(args, key) for key in ("n", "k", "experts", "capacity", "bits", "codebook")},
        stage="quantizer-basis projection; excludes full expert execution",
        ratio_direction="inline Trellis projection / per-route Torch FP16 projection; lower favors Trellis",
        correctness=[], measurements=[],
    )
    cpu = torch.randint(-32768, 32768,
                        (args.experts, args.k // 16, args.n // 16, 16 * args.bits),
                        dtype=torch.int16, generator=torch.Generator().manual_seed(103))
    packed = cpu.cuda()
    decoded = native_weight(cpu, args.bits, args.codebook).cuda()
    lut = codebook_tensor(args.codebook, "cuda")
    torch.manual_seed(103)
    source = torch.randn((args.capacity, args.k), device="cuda", dtype=torch.float16) * 0.05
    expert_rows = [i % args.experts for i in range(args.capacity)]
    ids = torch.tensor(expert_rows, dtype=torch.int64, device="cuda")
    output = torch.empty((args.capacity, args.n), dtype=torch.float16, device="cuda")
    baseline = torch.empty_like(output)
    params = [pointer(t, v) for t, v in (
        (cutlass.Float16, source), (cutlass.Uint32, packed), (cutlass.Uint8, lut),
        (cutlass.Int64, ids), (cutlass.Float16, output),
    )]
    strides = (cutlass.Int64(args.k), cutlass.Int64(args.n))
    artifacts = args.output.parent / (args.output.stem + "-artifacts")
    artifacts.mkdir(parents=True, exist_ok=False)
    compiled = cute.compile(kernel, *params, cutlass.Int32(1), *strides, cuda.CUstream(0),
                            options=f"--gpu-arch=sm_103a --keep-ptx --keep-cubin --dump-dir={artifacts}")
    record["artifacts"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in artifacts.iterdir() if path.suffix in {".ptx", ".cubin"}}
    expected = torch.empty_like(output)
    for row, expert in enumerate(expert_rows):
        expected[row] = source[row].float() @ decoded[expert].float().T
    graphs = []
    flush = make_l2_flush_fn(enabled=True)
    for rows in args.rows:
        def native(rows=rows):
            compiled(*params, cutlass.Int32(rows), *strides, cuda.CUstream(torch.cuda.current_stream().cuda_stream))

        def expanded(rows=rows):
            for row in range(rows):
                torch.mm(source[row:row+1], decoded[expert_rows[row]].T, out=baseline[row:row+1])

        native()
        expanded()
        checks = _check(output[:rows], expected[:rows], f"Trellis M={rows}")
        _check(baseline[:rows], expected[:rows], f"Torch M={rows}")
        arms = {"trellis": _capture(native), "torch_fp16": _capture(expanded)}
        output.fill_(float("nan"))
        allocated = torch.cuda.memory_allocated()
        addresses = tuple(t.data_ptr() for t in (source, packed, lut, ids, output))
        arms["trellis"].replay()
        torch.cuda.synchronize()
        if allocated != torch.cuda.memory_allocated() or addresses != tuple(t.data_ptr() for t in (source, packed, lut, ids, output)):
            raise RuntimeError("Trellis graph replay changed allocation or addresses")
        _check(output[:rows], expected[:rows], f"Trellis replay M={rows}")
        record["correctness"].append(dict(rows=rows, **checks, graph_replay=True, stable_addresses=True, replay_allocation=False))
        graphs.append((rows, arms))
    for _, arms in graphs:
        for _ in range(args.warmup):
            for graph in arms.values():
                graph.replay()
    torch.cuda.synchronize()
    record["before"] = _snapshot()
    for rows, arms in graphs:
        for mode, flush_fn in (("warm", None), ("cold", flush)):
            samples = _paired(arms, args.warmup, args.samples, flush_fn)
            ratio = statistics.median(row["trellis"] for row in samples) / statistics.median(row["torch_fp16"] for row in samples)
            record["measurements"].append(dict(rows=rows, mode=mode, raw_microseconds=samples, median_ratio=ratio))
    record["after"] = _snapshot()
    record["clock_checks"] = _clock_checks(record["before"], record["after"])
    record["artifacts_unchanged"] = all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
                                        for path, digest in record["artifacts"].items())
    record["status"] = "measured" if record["clock_checks"]["valid"] and record["artifacts_unchanged"] else "invalid_evidence"
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    if record["status"] != "measured":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
