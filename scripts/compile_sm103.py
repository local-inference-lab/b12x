#!/usr/bin/env python3
"""Cross-compile SM103 kernels and record artifact identities without a GPU."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def compile_roce(out):
    """Trace the existing TP2 peer protocol without allocating host/NIC regions."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from b12x.comm.roce._oneshot_cute import _RoceOneshotLaunch, _dummy
    from b12x.comm.roce._allgather_cute import _RoceAllGatherLaunch

    launches = {}
    for rank in (0, 1):
        for dtype in ("bfloat16", "float16", "float32", "gather"):
            name = f"roce_{dtype}_rank{rank}"
            directory = out / name
            directory.mkdir()
            args = [_dummy(cutlass.Uint32, 16), _dummy(cutlass.Uint32, 16), 1, 16]
            if dtype == "gather":
                launch = _RoceAllGatherLaunch(2, rank, 512, 2, 128, 1)
                args.append(1)
            else:
                launch = _RoceOneshotLaunch(dtype, 2, rank, 512, 2, 128, 1)
            args += [16, 16, 16, 16, 4096, 16, 16, 16, 16, 1, 1, cuda.CUstream(0)]
            compiled = cute.compile(
                launch,
                *args,
                no_jit_engine=True,
                options=f"--gpu-arch=sm_103a --keep-ptx --keep-cubin --dump-dir={directory}",
            )
            (directory / (name + ".mlir")).write_text(str(compiled.ir_module))
            launches[name] = compiled
    return launches


def compile_trellis(out):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from b12x.moe._shared.kernels.sm103.launch import pointer
    from b12x.moe._shared.kernels.sm103.trellis import ReconstructTrellisTiles

    launches = {}
    # One V4.1 FC2 family in the checkpoint's K16/N16 tile ordering.
    capacity = 384 * (5120 // 16) * (2304 // 16)
    for bits in (2, 3, 4):
        name = f"trellis_k{bits}"
        directory = out / name
        directory.mkdir()
        kernel = ReconstructTrellisTiles(bits, capacity)
        args = [pointer(t) for t in (cutlass.Uint32, cutlass.Uint8, cutlass.BFloat16)]
        args += [cutlass.Int32(1), cuda.CUstream(0)]
        compiled = cute.compile(
            kernel,
            *args,
            no_jit_engine=True,
            options=f"--gpu-arch=sm_103a --keep-ptx --keep-cubin --dump-dir={directory}",
        )
        (directory / (name + ".mlir")).write_text(str(compiled.ir_module))
        launches[name] = compiled
    return launches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=288)
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--gate-first", action="store_true")
    parser.add_argument("--nvdisasm", type=Path)
    parser.add_argument("--cuobjdump", type=Path)
    parser.add_argument(
        "--component", choices=("moe", "roce", "trellis", "all"), default="moe"
    )
    args = parser.parse_args()
    # CuTe must have an architecture even when the CUDA driver is unavailable.
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_103a")
    from b12x.moe._shared.kernels.sm103.launch import compile_launches
    from b12x.moe.fused_moe._sm103 import heuristic
    from b12x.moe.fused_moe._policy import MoeDecodeQuery

    heuristic(
        MoeDecodeQuery(
            "nvfp4",
            "modelopt_nvfp4",
            "silu",
            args.experts,
            args.hidden,
            args.intermediate,
            args.top_k,
            args.capacity,
            args.top_k * args.capacity,
        )
    )
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(
            "--output-dir must be empty to keep artifact provenance unambiguous"
        )
    if any(c.isspace() for c in str(out)):
        parser.error("the CuTe dump directory must not contain whitespace")
    out.mkdir(parents=True, exist_ok=True)
    caps = SimpleNamespace(
        k=args.hidden,
        n=args.intermediate,
        weight_E=args.experts,
        max_tokens=args.capacity,
        num_topk=args.top_k,
        w13_layout="w31" if args.gate_first else "w13",
        swiglu_limit=None,
    )
    manifest = {
        "status": "compiling",
        "runtime_qualified": False,
        "target": "sm_103a",
        "component": args.component,
        "command": sys.argv,
        "trellis_geometry": {
            "experts": 384,
            "hidden": 5120,
            "intermediate": 2304,
            "stage": "unscaled quantizer-basis tiles",
        },
        "roce_geometry": {
            "world_size": 2,
            "threads": 512,
            "slots": 2,
            "flag_stride": 128,
            "hca_count": 1,
        },
        "geometry": vars(caps),
        "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted(Path("b12x").rglob("*.py")))
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).splitlines(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "cuda-python",
                "nvidia-cutlass-dsl",
                "nvidia-cutlass-dsl-libs-base",
                "nvidia-cutlass-dsl-libs-core",
                "nvidia-cutlass-dsl-libs-cu13",
                "triton",
            )
        },
    }
    for name in ("nvdisasm", "cuobjdump"):
        tool = getattr(args, name)
        if tool:
            manifest[name] = subprocess.check_output(
                [str(tool.resolve()), "--version"], text=True
            )
    try:
        launches = {}
        if args.component in ("moe", "all"):
            launches.update(compile_launches(caps, offline=True, artifact_dir=out))
        if args.component in ("roce", "all"):
            launches.update(compile_roce(out))
        if args.component in ("trellis", "all"):
            launches.update(compile_trellis(out))
        artifacts = []
        for key in launches:
            name = key if isinstance(key, str) else key[0] + "_" + key[1].__name__
            directory = out / name
            ptxs = list(directory.glob("*.ptx"))
            cubins = list(directory.glob("*.cubin"))
            if len(ptxs) != 1 or len(cubins) != 1:
                raise RuntimeError(
                    f"{name}: expected exactly one PTX and cubin; use an empty output directory"
                )
            text = ptxs[0].read_text()
            if ".target sm_103a" not in text:
                raise RuntimeError(f"{name}: wrong PTX target")
            if name.startswith("fc") and "tcgen05.mma" not in text:
                raise RuntimeError(f"{name}: missing native tcgen05 MMA")
            inspected = []
            if args.nvdisasm:
                sass = subprocess.check_output(
                    [str(args.nvdisasm.resolve()), str(cubins[0])],
                    text=True,
                    timeout=60,
                )
                if name.startswith("fc") and "UTCOMMA.BLOCK16" not in sass:
                    raise RuntimeError(f"{name}: missing block-scaled UMMA in SASS")
                path = directory / (name + ".sass")
                path.write_text(sass)
                inspected.append(path)
            if args.cuobjdump:
                resources = subprocess.check_output(
                    [str(args.cuobjdump.resolve()), "-res-usage", str(cubins[0])],
                    text=True,
                    timeout=60,
                )
                path = directory / (name + ".resources.txt")
                path.write_text(resources)
                inspected.append(path)
            artifacts.append(
                {
                    "name": name,
                    "native_mma": "tcgen05.mma" in text,
                    "files": {
                        p.name: {
                            "bytes": p.stat().st_size,
                            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                        }
                        for p in [*ptxs, *cubins, *directory.glob("*.mlir"), *inspected]
                    },
                }
            )
        manifest.update(
            status="cross-compiled", callables=len(launches), artifacts=artifacts
        )
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "artifacts"}, indent=2))


if __name__ == "__main__":
    main()
