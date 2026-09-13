"""Dense SM103 NVFP4, MXFP4, and MXFP8 projections through tcgen05.

The callable specializes immutable weight geometry and dtype. Row counts and
batch strides are launch arguments. Prewarming retains the callable; callers
provide output and quantization workspace for allocation-stable graph replay.
"""

from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr
import torch

from b12x._lib.architecture import UnsupportedArchitectureError
from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import cuda_stream_to_int
from b12x.gemm._shared.sm103_blockscaled import BlockscaledGemm


class DenseBlockscaledGemm(BlockscaledGemm):
    @cute.jit
    def __call__(
        self, a: cute.Pointer, b: cute.Pointer, sfa: cute.Pointer,
        sfb: cute.Pointer, c: cute.Pointer, alpha: cute.Pointer,
        rows: cutlass.Int32, a_group_stride: cutlass.Int64,
        c_group_stride: cutlass.Int64, alpha_stride: cutlass.Int64,
        stream: cuda.CUstream,
    ):
        self._launch(a, b, sfa, sfb, c, None, alpha, rows,
                     a_group_stride, c_group_stride, alpha_stride, stream)


RECIPES = {
    ("float4_e2m1fn", "float8_e4m3fn", 16): "nvfp4",
    ("float4_e2m1fn", "float8_e8m0fnu", 32): "mxfp4",
    ("float8_e4m3fn", "float8_e8m0fnu", 32): "mxfp8",
}
OUTPUT_TYPES = {
    "bfloat16": (torch.bfloat16, cutlass.BFloat16),
    "float16": (torch.float16, cutlass.Float16),
    "float32": (torch.float32, cutlass.Float32),
}


def pointer(dtype, tensor=None):
    return make_ptr(dtype, 0 if tensor is None else tensor.data_ptr(),
                    cute.AddressSpace.gmem, assumed_align=16)


@lru_cache(maxsize=1024)
def compile_kernel(n, k, groups, recipe, c_dtype, device_ordinal, alpha_is_one=False):
    raise_if_kernel_resolution_frozen("SM103 blockscaled GEMM")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("SM103 blockscaled GEMM must be prewarmed before graph capture")
    kernel = DenseBlockscaledGemm(n, k, groups, recipe=recipe,
                                  c_dtype=OUTPUT_TYPES[c_dtype][1], alpha_is_one=alpha_is_one)
    types = (kernel.ab_dtype, kernel.ab_dtype, kernel.sf_dtype,
             kernel.sf_dtype, kernel.c_dtype, cutlass.Float32)
    spec = KernelCompileSpec.from_facts(
        "gemm.blockscaled.sm103", 1, ("n", n), ("k", k), ("groups", groups),
        ("recipe", recipe), ("c_dtype", c_dtype), ("device", device_ordinal),
        ("alpha_is_one", alpha_is_one),
    )
    return b12x_compile(
        kernel, *(pointer(t) for t in types), cutlass.Int32(1),
        cutlass.Int64(k), cutlass.Int64(n), cutlass.Int64(1), cuda.CUstream(0),
        options="--gpu-arch=sm_103a", compile_spec=spec,
    )


def _scale_storage(scale, rows, k, groups, vector, device, dtype):
    mt, kt = (rows + 127) // 128, (k // vector + 3) // 4
    if scale.device != device or scale.dtype not in (torch.uint8, dtype):
        raise ValueError("SM103 block scales must match the recipe and operand device")
    if scale.ndim == 6:
        if tuple(scale.shape) != (32, 4, mt, 4, kt, groups):
            raise ValueError("SM103 block scales have an invalid F8_128x4 shape")
        scale = scale.permute(5, 2, 4, 0, 1, 3)
    if not scale.is_contiguous() or scale.numel() != groups * mt * kt * 512:
        raise ValueError("SM103 block scales require contiguous F8_128x4 storage")
    if scale.numel() and scale.data_ptr() % 16:
        raise ValueError("SM103 block scales require 16-byte alignment")
    return scale


def _grouped_layout(tensor, rows, width, groups):
    if tensor.stride(1) != 1 or (rows > 1 and tensor.stride(0) != width):
        raise ValueError("SM103 GEMM requires contiguous rows within each group")
    if groups > 1 and tensor.stride(2) < rows * width:
        raise ValueError("SM103 GEMM groups must not overlap")
    if tensor.numel() and tensor.data_ptr() % 16:
        raise ValueError("SM103 GEMM operands require 16-byte alignment")


def _span(tensor):
    size = 0 if not tensor.numel() else 1 + sum(
        (extent - 1) * stride for extent, stride in zip(tensor.shape, tensor.stride(), strict=True)
    )
    return tensor.data_ptr(), tensor.data_ptr() + size * tensor.element_size()


def execute(lhs, rhs, out, *, ab_dtype, sf_dtype, c_dtype, sf_vec_size,
            alpha=None, stream=None):
    recipe = RECIPES.get((ab_dtype, sf_dtype, sf_vec_size))
    if recipe is None:
        raise UnsupportedArchitectureError("SM103 dense GEMM implements NVFP4, MXFP4, and MXFP8")
    if c_dtype not in OUTPUT_TYPES:
        raise ValueError("SM103 dense GEMM output must be BF16, FP16, or FP32")
    a, sfa = lhs
    b, sfb = rhs
    if a.device.type != "cuda" or b.device != a.device or a.ndim != 3 or b.ndim != 3:
        raise ValueError("SM103 GEMM operands must have shape [rows,K,groups] on one CUDA device")
    m, storage_k, groups = a.shape
    n = b.shape[0]
    k = storage_k if recipe == "mxfp8" else 2 * storage_k
    if min(n, k, groups) <= 0 or m >= 2**31 or groups > 65535:
        raise ValueError("SM103 GEMM requires positive geometry and a valid CUDA grid")
    if tuple(b.shape[1:]) != (storage_k, groups) or k % 128 or n % 8:
        raise ValueError("SM103 GEMM requires matching K/groups, K divisible by 128, and N by 8")
    value_types = (torch.float8_e4m3fn,) if recipe == "mxfp8" else (torch.uint8, torch.float4_e2m1fn_x2)
    if a.dtype not in value_types or b.dtype not in value_types:
        raise ValueError("SM103 GEMM value storage does not match the quantization recipe")
    _grouped_layout(a, m, storage_k, groups)
    _grouped_layout(b, n, storage_k, groups)
    if groups > 1 and a.stride(2) % 16:
        raise ValueError("SM103 GEMM activation group strides must be 16-byte aligned for TMA")
    if groups > 1 and b.stride(2) != n * storage_k:
        raise ValueError("SM103 GEMM weight groups must be contiguous")
    scale_dtype = torch.float8_e4m3fn if recipe == "nvfp4" else torch.float8_e8m0fnu
    sfa = _scale_storage(sfa, m, k, groups, sf_vec_size, a.device, scale_dtype)
    sfb = _scale_storage(sfb, n, k, groups, sf_vec_size, a.device, scale_dtype)
    if out is None:
        from b12x._lib.dense_gemm import _empty_dense_gemm_output
        out = _empty_dense_gemm_output(m, n, groups, dtype=OUTPUT_TYPES[c_dtype][0], device=a.device)
    if out.device != a.device or out.dtype != OUTPUT_TYPES[c_dtype][0] or tuple(out.shape) != (m, n, groups):
        raise ValueError("SM103 GEMM output must match operand device, shape, and selected dtype")
    _grouped_layout(out, m, n, groups)
    if alpha is not None and (alpha.device != a.device or alpha.dtype != torch.float32
            or alpha.numel() not in (1, groups) or not alpha.is_contiguous()
            or alpha.data_ptr() % 16):
        raise ValueError("SM103 GEMM alpha must be aligned contiguous FP32 scalar or one value per group")
    lo, hi = _span(out)
    reads = (a, b, sfa, sfb) + ((alpha,) if alpha is not None else ())
    if any(lo < _span(t)[1] and _span(t)[0] < hi for t in reads):
        raise ValueError("SM103 GEMM output must not overlap inputs")
    if m:
        _execute(a, b, sfa, sfb, out, alpha, recipe, c_dtype, cuda_stream_to_int(stream))
    return out


@torch.library.custom_op("b12x::sm103_blockscaled", mutates_args=("out",))
def _execute(a: torch.Tensor, b: torch.Tensor, sfa: torch.Tensor,
             sfb: torch.Tensor, out: torch.Tensor, alpha: torch.Tensor | None,
             recipe: str, c_dtype: str, stream_int: int | None) -> None:
    m, storage_k, groups = a.shape
    k = storage_k if recipe == "mxfp8" else 2 * storage_k
    with torch.cuda.device(a.device):
        fn = compile_kernel(b.shape[0], k, groups, recipe, c_dtype, a.device.index, alpha is None)
        ab_type = cutlass.Float8E4M3FN if recipe == "mxfp8" else cutlass.Float4E2M1FN
        sf_type = cutlass.Float8E4M3FN if recipe == "nvfp4" else cutlass.Float8E8M0FNU
        types = (ab_type, ab_type, sf_type, sf_type, OUTPUT_TYPES[c_dtype][1], cutlass.Float32)
        tensors = (a, b, sfa, sfb, out, alpha)
        fn(*(pointer(t, x) for t, x in zip(types, tensors, strict=True)), cutlass.Int32(m),
           cutlass.Int64(m * k if groups == 1 else a.stride(2) * (1 if recipe == "mxfp8" else 2)),
           cutlass.Int64(m * out.shape[1] if groups == 1 else out.stride(2)),
           cutlass.Int64(0 if alpha is None or alpha.numel() == 1 else 1),
           cuda.CUstream(torch.cuda.current_stream(a.device).cuda_stream if stream_int is None else stream_int))


@_execute.register_fake
def _execute_fake(a, b, sfa, sfb, out, alpha, recipe, c_dtype, stream_int):
    return None
