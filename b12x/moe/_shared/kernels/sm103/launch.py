"""Compile and retain capacity-specialized SM103 launches before binding."""

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr
import torch

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.architecture import architecture_for, UnsupportedArchitectureError
from .nvfp4_gemm import RoutedNvfp4Gemm
from .pointwise import RoutedQuantize, TopKSum


def pointer(dtype, tensor=None):
    if tensor is not None and tensor.data_ptr() % 16:
        raise ValueError("SM103 launch pointers must be 16-byte aligned")
    return make_ptr(
        dtype,
        0 if tensor is None else tensor.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )


def compile_launches(caps, *, offline=False, artifact_dir=None):
    """Offline mode exports actual kernels without creating a CUDA context."""
    if not offline:
        capability = torch.cuda.get_device_capability(caps.device)
        arch = architecture_for(capability)
        if arch is None or arch.name != "sm103":
            raise UnsupportedArchitectureError(
                "SM103 launches require a physical SM103 device"
            )
        props = torch.cuda.get_device_properties(caps.device)
        if props.shared_memory_per_block_optin < 128 * 1024:
            raise UnsupportedArchitectureError(
                "SM103 launch requires at least 128 KiB opt-in SMEM"
            )
    routes = caps.max_tokens * caps.num_topk
    launches = {}
    for ids_dtype in (cutlass.Int32, cutlass.Int64):
        for name, n, k in (("fc1", 2 * caps.n, caps.k), ("fc2", caps.k, caps.n)):
            kernel = RoutedNvfp4Gemm(n, k, caps.weight_E, routes)
            args = [
                pointer(t)
                for t in (
                    cutlass.Float4E2M1FN,
                    cutlass.Float4E2M1FN,
                    cutlass.Float8E4M3FN,
                    cutlass.Float8E4M3FN,
                    cutlass.BFloat16,
                    ids_dtype,
                    cutlass.Float32,
                )
            ] + [cutlass.Int32(1), cuda.CUstream(0)]
            launches[name, ids_dtype] = _compile(
                name, kernel, args, ids_dtype, caps, offline, artifact_dir
            )
        for name, width, activation in (("q1", caps.k, False), ("q2", caps.n, True)):
            kernel = RoutedQuantize(
                width,
                caps.num_topk,
                caps.weight_E,
                activation=activation,
                gate_first=caps.w13_layout == "w31",
                limit=caps.swiglu_limit,
            )
            args = [
                pointer(t)
                for t in (
                    cutlass.BFloat16,
                    ids_dtype,
                    cutlass.Float32,
                    cutlass.Uint8,
                    cutlass.Float8E4M3FN,
                )
            ]
            args += [cutlass.Int32(1), cutlass.Int32(1), cuda.CUstream(0)]
            launches[name, ids_dtype] = _compile(
                name, kernel, args, ids_dtype, caps, offline, artifact_dir
            )
    kernel = TopKSum(caps.k, caps.num_topk)
    args = [pointer(t) for t in (cutlass.BFloat16, cutlass.Float32, cutlass.BFloat16)]
    args += [cutlass.Int32(1), cuda.CUstream(0)]
    launches["sum"] = _compile("sum", kernel, args, None, caps, offline, artifact_dir)
    return launches


def _compile(name, kernel, args, ids_dtype, caps, offline, artifact_dir):
    options = "--gpu-arch=sm_103a"
    if artifact_dir is not None:
        from pathlib import Path

        label = name + ("" if ids_dtype is None else "_" + ids_dtype.__name__)
        artifact_dir = Path(artifact_dir, label)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        options += f" --keep-ptx --keep-cubin --dump-dir={artifact_dir}"
    if offline:
        compiled = cute.compile(kernel, *args, options=options, no_jit_engine=True)
        if artifact_dir is not None:
            label = name + ("" if ids_dtype is None else "_" + ids_dtype.__name__)
            from pathlib import Path

            Path(artifact_dir, label + ".mlir").write_text(str(compiled.ir_module))
        return compiled
    spec = KernelCompileSpec.from_facts(
        "moe.sm103." + name,
        2,
        ("hidden", caps.k),
        ("intermediate", caps.n),
        ("experts", caps.weight_E),
        ("capacity", caps.max_tokens),
        ("top_k", caps.num_topk),
        ("id_dtype", None if ids_dtype is None else ids_dtype.__name__),
        ("w13_layout", caps.w13_layout),
        ("swiglu_limit", caps.swiglu_limit),
        ("target", "sm_103a"),
    )
    with torch.cuda.device(caps.device):
        return b12x_compile(kernel, *args, options=options, compile_spec=spec)


@dataclass(frozen=True)
class BoundLaunches:
    calls: tuple
    output: torch.Tensor
    owners: tuple
    device: torch.device

    def run(self):
        stream = cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream)
        for fn, args in self.calls:
            fn(*args, stream)
        return self.output


def bind_launches(launches, caps, a, experts, ids, weights, views, output):
    id_type = cutlass.Int32 if ids.dtype == torch.int32 else cutlass.Int64
    expected = (
        (
            experts.w1_fp4,
            (caps.weight_E, 2 * caps.n, caps.k // 2),
            (torch.uint8, torch.float4_e2m1fn_x2),
        ),
        (
            experts.w2_fp4,
            (caps.weight_E, caps.k, caps.n // 2),
            (torch.uint8, torch.float4_e2m1fn_x2),
        ),
        (
            experts.w1_blockscale,
            (caps.weight_E, 2 * caps.n, caps.k // 16),
            (torch.uint8, torch.float8_e4m3fn),
        ),
        (
            experts.w2_blockscale,
            (caps.weight_E, caps.k, caps.n // 16),
            (torch.uint8, torch.float8_e4m3fn),
        ),
        (experts.w1_alphas, (caps.weight_E,), (torch.float32,)),
        (experts.w2_alphas, (caps.weight_E,), (torch.float32,)),
    )
    for tensor, shape, dtypes in expected:
        if (
            tensor.shape != shape
            or tensor.dtype not in dtypes
            or tensor.device != caps.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"SM103 prepared weights require contiguous {shape} {dtypes} on {caps.device}"
            )
        if tensor.data_ptr() % 16:
            raise ValueError("SM103 TMA pointers must be 16-byte aligned")
    for scale in (experts.a1_gscale, experts.a2_gscale):
        if (
            scale.dtype != torch.float32
            or scale.numel() not in (1, caps.weight_E)
            or scale.device != caps.device
            or not scale.is_contiguous()
        ):
            raise ValueError(
                "SM103 activation global scales must be scalar or per-expert contiguous FP32"
            )
    pids = pointer(id_type, ids)
    live = cutlass.Int32(a.shape[0] * caps.num_topk)
    calls = []
    for stage, x, gs, w, sfw, alpha, c in (
        (
            1,
            a,
            experts.a1_gscale,
            experts.w1_fp4,
            experts.w1_blockscale,
            experts.w1_alphas,
            views["fc1"],
        ),
        (
            2,
            views["fc1"],
            experts.a2_gscale,
            experts.w2_fp4,
            experts.w2_blockscale,
            experts.w2_alphas,
            views["fc2"],
        ),
    ):
        q, sf = views[f"q{stage}"], views[f"s{stage}"]
        calls.append(
            (
                launches[f"q{stage}", id_type],
                (
                    pointer(cutlass.BFloat16, x),
                    pids,
                    pointer(cutlass.Float32, gs),
                    pointer(cutlass.Uint8, q),
                    pointer(cutlass.Float8E4M3FN, sf),
                    live,
                    cutlass.Int32(0 if gs.numel() == 1 else 1),
                ),
            )
        )
        calls.append(
            (
                launches[f"fc{stage}", id_type],
                (
                    pointer(cutlass.Float4E2M1FN, q),
                    pointer(cutlass.Float4E2M1FN, w),
                    pointer(cutlass.Float8E4M3FN, sf),
                    pointer(cutlass.Float8E4M3FN, sfw),
                    pointer(cutlass.BFloat16, c),
                    pids,
                    pointer(cutlass.Float32, alpha),
                    live,
                ),
            )
        )
    calls.append(
        (
            launches["sum"],
            (
                pointer(cutlass.BFloat16, views["fc2"]),
                pointer(cutlass.Float32, weights),
                pointer(cutlass.BFloat16, output),
                cutlass.Int32(a.shape[0]),
            ),
        )
    )
    return BoundLaunches(
        tuple(calls), output, (a, experts, ids, weights, views), caps.device
    )
