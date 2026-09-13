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
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def compile_sequence(out):
    """Compile production recurrent launch factories using pointer prototypes.

    Binding descriptors supply static geometry and dtypes only. No device
    allocation or kernel execution occurs, and these objects are not runtime
    qualification evidence.
    """
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    import torch
    from b12x.sequence._shared.delta_prefill import _cute_kernels as prefill
    from b12x.sequence.gdn_decode import _cute_kernels as qwen, _cute_kda as kda
    from b12x.sequence.kda_prefill import _impl as kp
    from b12x.sequence.gdn_prefill import _impl as gp

    launches = {}
    case = ""

    def capture(kernel, *args, compile_spec):
        name = case + "_" + compile_spec.kernel_id.rsplit(".", 1)[-1]
        directory = out / name
        directory.mkdir()
        compiled = cute.compile(
            kernel,
            *args,
            no_jit_engine=True,
            options=f"--gpu-arch=sm_103a --keep-ptx --keep-cubin --dump-dir={directory}",
        )
        (directory / (name + ".mlir")).write_text(str(compiled.ir_module))
        launches[name] = compiled
        return compiled

    def descriptor(dtype):
        return SimpleNamespace(
            dtype=dtype,
            device=torch.device("cuda:0"),
            stride=lambda: (12 * 128 * 128, 128 * 128, 128, 1),
        )

    for index_type in (torch.int32, torch.int64):
        suffix = str(index_type).removeprefix("torch.")
        for recipe, impl in (("kda", kp), ("gdn", gp)):
            for checkpoint in (False, True):
                case = f"{recipe}_prefill_{suffix}_checkpoint{int(checkpoint)}"
                geometry = (
                    dict(heads=16)
                    if recipe == "kda"
                    else dict(key_heads=4, value_heads=12)
                )
                caps = impl.Caps(
                    device="cuda:0",
                    max_tokens=4096,
                    max_seqs=16,
                    max_state_slots=129,
                    checkpoint_export=checkpoint,
                    null_state_index=0,
                    **geometry,
                )
                plan = impl._materialize_plan(
                    caps,
                    v_split=64,
                    k_split=1,
                    stages=3,
                    window_tiles=128,
                    policy_resolution=None,
                )
                binding = SimpleNamespace(
                    plan=plan,
                    output=descriptor(torch.bfloat16),
                    initial_state_indices=descriptor(index_type),
                    A_log=descriptor(torch.float32),
                    dt_bias=descriptor(torch.float32),
                )
                prefill.clear_caches()
                with (
                    patch.object(prefill, "b12x_compile", capture),
                    patch.object(
                        prefill, "current_cuda_stream", lambda: cuda.CUstream(0)
                    ),
                ):
                    prefill._compile_prologue(binding)
                    prefill._compile_prepare(binding)
                    prefill._compile_recurrence(binding)
        for state_type in (torch.bfloat16, torch.float32):
            case = f"kda_decode_{suffix}_{str(state_type).removeprefix('torch.')}"
            key = (
                0,
                128,
                32,
                129,
                4,
                16,
                32,
                True,
                0,
                True,
                state_type,
                index_type,
                torch.float32,
                torch.float32,
                torch.float32,
            )
            with (
                patch.object(kda, "b12x_compile", capture),
                patch.object(kda, "current_cuda_stream", lambda: cuda.CUstream(0)),
            ):
                kda.compile_kernels(key)
            case = f"qwen_decode_{suffix}_{str(state_type).removeprefix('torch.')}"
            caps = SimpleNamespace(
                max_seqs=32,
                state_index_columns=4,
                key_heads=4,
                value_heads=12,
                key_head_dim=128,
                value_head_dim=128,
                qk_l2norm=True,
                null_state_index=0,
            )
            binding = qwen.Binding.__new__(qwen.Binding)
            fields = dict(
                plan=SimpleNamespace(caps=caps),
                output=descriptor(torch.bfloat16),
                recurrent_state=descriptor(state_type),
                state_indices=descriptor(index_type),
                A_log=descriptor(torch.float32),
                dt_bias=descriptor(torch.float32),
            )
            for field, value in fields.items():
                object.__setattr__(binding, field, value)
            qwen._KERNEL_CACHE.clear()
            with (
                patch.object(qwen, "b12x_compile", capture),
                patch.object(qwen, "current_cuda_stream", lambda: cuda.CUstream(0)),
            ):
                qwen._compile(binding)
    prefill.clear_caches()
    qwen._KERNEL_CACHE.clear()
    kda._CACHE.clear()
    return launches


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


def compile_dense_mla(out):
    """Compile dense MLA math, split reduction, and query quantization."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_compact_tensor as tensor
    from b12x.attention.dense_mla._forward import DenseMlaForwardKernel
    from b12x.attention.dense_mla._merge import DenseMlaMergeKernel
    from b12x.attention.dense_mla._layout import make_smem_layout

    launches = {}

    def compile_case(name, kernel, args):
        directory = out / name
        directory.mkdir()
        compiled = cute.compile(
            kernel,
            *args,
            no_jit_engine=True,
            options=f"--gpu-arch=sm_103a --keep-ptx --keep-cubin --dump-dir={directory}",
        )
        (directory / (name + ".mlir")).write_text(str(compiled.ir_module))
        launches[name] = compiled

    for fp8 in (False, True):
        for qk_dim, value_dim in ((576, 512), (1088, 1024)):
            for query_tile in (1, 2, 4) if fp8 else (1, 2):
                for window in (None, 128):
                    if window is not None and query_tile != 1:
                        continue
                    name = f"dense_mla_{'fp8' if fp8 else 'bf16'}_d{qk_dim}_q{query_tile}_w{window}"
                    kernel = DenseMlaForwardKernel(
                        layout=make_smem_layout(
                            query_tile=query_tile, fp8=fp8, qk_dim=qk_dim
                        ),
                        page_size=64,
                        num_heads=16,
                        num_splits=4,
                        chunks_per_split=16,
                        query_tile=query_tile,
                        fp8=fp8,
                        qk_dim=qk_dim,
                        value_dim=value_dim,
                        window_size=window,
                    )
                    args = [
                        tensor(cutlass.Uint8, (cute.sym_int(),), assumed_align=16)
                        for _ in range(2)
                    ]
                    args += [
                        tensor(cutlass.Int32, (cute.sym_int(),), assumed_align=4)
                        for _ in range(3)
                    ]
                    args += [
                        tensor(
                            cutlass.BFloat16,
                            (cute.sym_int(), 16, value_dim),
                            assumed_align=16,
                        ),
                        tensor(cutlass.Float32, (128, 16), assumed_align=4),
                        tensor(
                            cutlass.BFloat16, (128, 16, 4, value_dim), assumed_align=16
                        ),
                        tensor(cutlass.Float32, (128, 16, 4), assumed_align=4),
                    ]
                    args += [
                        tensor(cutlass.Float32, (1,), assumed_align=4) for _ in range(2)
                    ]
                    args += [
                        cutlass.Float32(1.0),
                        *([cutlass.Int64(1)] * 5),
                        *([cutlass.Int32(1)] * 3),
                        cuda.CUstream(0),
                    ]
                    compile_case(name, kernel, args)
    for value_dim in (512, 1024):
        compile_case(
            f"dense_mla_merge_v{value_dim}",
            DenseMlaMergeKernel(4, value_dim),
            [
                tensor(cutlass.BFloat16, (128, 16, 4, value_dim), assumed_align=16),
                tensor(cutlass.Float32, (128, 16, 4), assumed_align=4),
                tensor(
                    cutlass.BFloat16, (cute.sym_int(), 16, value_dim), assumed_align=16
                ),
                tensor(cutlass.Float32, (128, 16), assumed_align=4),
                cutlass.Int32(1),
                cuda.CUstream(0),
            ],
        )
    from b12x.attention._shared.static_fp8_quant import _StaticFp8QuantKernel

    compile_case(
        "dense_mla_query_quant",
        _StaticFp8QuantKernel(128 * 16 * 1088),
        [
            tensor(cutlass.Uint8, (cute.sym_int(),), assumed_align=16),
            tensor(cutlass.Uint8, (cute.sym_int(),), assumed_align=16),
            tensor(cutlass.Float32, (cute.sym_int(),), assumed_align=4),
            cutlass.Int32(4),
            cuda.CUstream(0),
        ],
    )
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


def compile_bf16_projection(out):
    """Compile SIMT and warp-MMA unquantized projection entry points."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_compact_tensor as tensor
    from b12x.gemm.bf16_gemv._kernel import ProjectionKernel
    from b12x.gemm.bf16_gemv._prefill import Bf16PrefillKernel
    from b12x.moe._shared.kernels.sm103.launch import pointer

    launches = {}

    def compile_case(name, kernel, args):
        directory = out / name
        directory.mkdir()
        compiled = cute.compile(
            kernel,
            *args,
            no_jit_engine=True,
            options=f"--gpu-arch=sm_103a --keep-ptx --keep-cubin --dump-dir={directory}",
        )
        (directory / (name + ".mlir")).write_text(str(compiled.ir_module))
        launches[name] = compiled

    for x_type in (cutlass.BFloat16, cutlass.Float32):
        for w_type in (cutlass.BFloat16, cutlass.Float32):
            for out_type in (cutlass.BFloat16, cutlass.Float32):
                name = f"projection_{x_type.__name__}_{w_type.__name__}_{out_type.__name__}"
                kernel = ProjectionKernel(
                    512, 4096, x_type == w_type == cutlass.BFloat16, True
                )
                args = [pointer(t) for t in (x_type, w_type, cutlass.Float32, out_type)]
                args += [
                    cutlass.Int32(1),
                    *([cutlass.Int64(1)] * 5),
                    cutlass.Int32(0),
                    cutlass.Int32(0),
                    cuda.CUstream(0),
                ]
                compile_case(name, kernel, args)
    for out_type in (cutlass.BFloat16, cutlass.Float32):
        compile_case(
            f"projection_prefill_{out_type.__name__}",
            Bf16PrefillKernel(512, 5120),
            [
                tensor(cutlass.BFloat16, (cute.sym_int(), 5120), assumed_align=16),
                tensor(cutlass.BFloat16, (512, 5120), assumed_align=16),
                tensor(out_type, (cute.sym_int(), 512), assumed_align=16),
                cutlass.Int32(1),
                cuda.CUstream(0),
            ],
        )
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
        "--component",
        choices=(
            "moe",
            "roce",
            "trellis",
            "sequence",
            "dense_mla",
            "projection",
            "all",
        ),
        default="moe",
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
        if args.component in ("sequence", "all"):
            launches.update(compile_sequence(out))
        if args.component in ("dense_mla", "all"):
            launches.update(compile_dense_mla(out))
        if args.component in ("projection", "all"):
            launches.update(compile_bf16_projection(out))
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
