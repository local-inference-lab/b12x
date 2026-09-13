"""Tensor-scaled and compact K128 FP8 GEMM with ordinary warp MMA.

The SM103 entry retains the shared TMA pipeline and arithmetic. Static weight
geometry selects a conservative tile; live rows and group strides are launch
arguments. The same entry can run on SM12x for regression qualification.
"""

from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.dense_gemm import DenseGemmKernel, _empty_dense_gemm_output
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import cuda_stream_to_int
from ._sm103 import OUTPUT_TYPES, _grouped_layout, _span, pointer


class DenseFp8Launch:
    def __init__(self, n, k, groups, c_dtype, block_fp8, alpha_is_one, sm_count):
        self.n, self.k, self.groups = n, k, groups
        self.block_fp8, self.alpha_is_one = block_fp8, alpha_is_one
        self.swap = n * c_dtype.width % 128 != 0
        self.tile = (64, 16) if self.swap else (16, 64)
        self.sm_count = sm_count
        if min(n, k, groups, sm_count) <= 0 or k % 128:
            raise ValueError("FP8 GEMM requires positive geometry and K divisible by 128")
        if block_fp8 and (n % 128 or groups != 1):
            raise ValueError("compact block-FP8 requires N divisible by 128 and one group")
        if self.swap and groups != 1:
            raise ValueError("unaligned FP8 output rows require one group")

    @cute.jit
    def __call__(self, a: cute.Pointer, b: cute.Pointer, sfa: cute.Pointer,
                 sfb: cute.Pointer, c: cute.Pointer, alpha: cute.Pointer,
                 rows: cutlass.Int32, a_stride: cutlass.Int64,
                 c_stride: cutlass.Int64, stream: cuda.CUstream):
        m = cutlass.Int64(rows)
        a_stride = cute.assume(a_stride, divby=16)
        a_tensor = cute.make_tensor(a, cute.make_layout(
            (m, self.k, self.groups), stride=(cutlass.Int64(self.k), 1, a_stride)))
        b_tensor = cute.make_tensor(b, cute.make_layout(
            (self.n, self.k, self.groups),
            stride=(cutlass.Int64(self.k), 1, cutlass.Int64(self.n) * self.k)))
        c_tensor = cute.make_tensor(c, cute.make_layout(
            (m, self.n, self.groups), stride=(cutlass.Int64(self.n), 1, c_stride)))
        alpha_tensor = cute.make_tensor(alpha, cute.make_layout((1,)))
        if cutlass.const_expr(self.block_fp8):
            sfa_tensor = cute.make_tensor(sfa, cute.make_layout(
                (m, self.k // 128, 1),
                stride=(cutlass.Int64(self.k // 128), 1, m * (self.k // 128))))
            sfb_tensor = cute.make_tensor(sfb, cute.make_layout(
                (self.n // 128, self.k // 128, 1),
                stride=(cutlass.Int64(self.k // 128), 1,
                        cutlass.Int64(self.n // 128) * (self.k // 128))))
        else:
            sfa_tensor = cute.make_tensor(sfa, cute.make_layout((1,)))
            sfb_tensor = cute.make_tensor(sfb, cute.make_layout((1,)))
        DenseGemmKernel(
            sf_vec_size=128 if self.block_fp8 else 32,
            mma_tiler_mn=self.tile, cluster_shape_mn=(1, 1),
            mma_k=32, tile_k=128, load_path="tma", swap_ab=self.swap,
            plain_fp8=True, block_fp8=self.block_fp8,
            alpha_is_one=self.alpha_is_one, target_occupancy=1,
        )(a_tensor, a_tensor, alpha_tensor, alpha_tensor, b_tensor,
          sfa_tensor, sfb_tensor, c_tensor, alpha_tensor, alpha_tensor,
          alpha_tensor, alpha_tensor, self.sm_count, stream)


@lru_cache(maxsize=1024)
def compile_kernel(n, k, groups, c_dtype, block_fp8, alpha_is_one,
                   device_ordinal, sm_count, architecture):
    raise_if_kernel_resolution_frozen("FP8 warp GEMM")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("FP8 warp GEMM must be prewarmed before graph capture")
    if architecture not in ("sm_103a", "sm_120a", "sm_121a"):
        raise ValueError("FP8 warp GEMM requires SM103/SM120/SM121")
    kernel = DenseFp8Launch(n, k, groups, OUTPUT_TYPES[c_dtype][1],
                            block_fp8, alpha_is_one, sm_count)
    sf_type = cutlass.Float32 if block_fp8 else cutlass.Float8E8M0FNU
    types = (cutlass.Float8E4M3FN, cutlass.Float8E4M3FN, sf_type, sf_type,
             OUTPUT_TYPES[c_dtype][1], cutlass.Float32)
    spec = KernelCompileSpec.from_facts(
        "gemm.blockscaled.fp8_warp", 1, ("n", n), ("k", k), ("groups", groups),
        ("c_dtype", c_dtype), ("block_fp8", block_fp8), ("alpha_is_one", alpha_is_one),
        ("device", device_ordinal), ("sm_count", sm_count), ("architecture", architecture),
    )
    return b12x_compile(
        kernel, *(pointer(t) for t in types), cutlass.Int32(1),
        cutlass.Int64(k), cutlass.Int64(n), cuda.CUstream(0),
        options=f"--gpu-arch={architecture}", compile_spec=spec,
    )


def execute(lhs, rhs, out, *, ab_dtype, sf_dtype, c_dtype, sf_vec_size,
            block_fp8=False, alpha=None, stream=None):
    if ab_dtype != "float8_e4m3fn" or c_dtype not in OUTPUT_TYPES:
        raise ValueError("FP8 GEMM requires E4M3 operands and BF16/FP16/FP32 output")
    if (sf_dtype, sf_vec_size) != (("float32", 128) if block_fp8 else ("float8_e8m0fnu", 32)):
        raise ValueError("FP8 GEMM scale dtype and group size do not match the recipe")
    a, sfa = lhs
    b, sfb = rhs
    if a.device.type != "cuda" or b.device != a.device or a.ndim != 3 or b.ndim != 3:
        raise ValueError("FP8 operands must have shape [rows,K,groups] on one CUDA device")
    m, k, groups = a.shape
    n = b.shape[0]
    if min(n, k, groups) <= 0 or max(m, n, k) >= 2**31 or groups > 65535 or k % 128:
        raise ValueError("FP8 GEMM requires valid geometry and K divisible by 128")
    if b.shape[1:] != a.shape[1:] or a.dtype != torch.float8_e4m3fn or b.dtype != a.dtype:
        raise ValueError("FP8 operands must have matching K/groups and E4M3 storage")
    for tensor, rows in ((a, m), (b, n)):
        _grouped_layout(tensor, rows, k, groups)
    if groups > 1 and (a.stride(2) % 16 or b.stride(2) != n * k):
        raise ValueError("FP8 GEMM requires aligned activation groups and contiguous weight groups")
    dtype = OUTPUT_TYPES[c_dtype][0]
    if groups > 1 and n * OUTPUT_TYPES[c_dtype][1].width % 128:
        raise ValueError("unaligned FP8 output rows require one group")
    if block_fp8:
        if n % 128 or groups != 1:
            raise ValueError("compact block-FP8 requires N divisible by 128 and one group")
        for scale, shape in ((sfa, (m, k // 128)), (sfb, (n // 128, k // 128))):
            if (scale.device != a.device or scale.dtype != torch.float32
                    or tuple(scale.shape) != shape or not scale.is_contiguous()
                    or (scale.numel() and scale.data_ptr() % 16)):
                raise ValueError("compact block-FP8 scales require aligned contiguous FP32 [rows,K/128] storage")
    else:
        # Tensor scaling uses alpha exclusively. No scale allocation or transfer
        # may depend on the live row count.
        sfa = sfb = None
    if out is None:
        out = _empty_dense_gemm_output(m, n, groups, dtype=dtype, device=a.device)
    if out.device != a.device or out.dtype != dtype or tuple(out.shape) != (m, n, groups):
        raise ValueError("FP8 output must match the operand device, shape, and selected dtype")
    _grouped_layout(out, m, n, groups)
    if groups > 1 and out.stride(2) * out.element_size() % 16:
        raise ValueError("FP8 output group strides must be 16-byte aligned")
    if alpha is not None and (alpha.device != a.device or alpha.dtype != torch.float32
            or alpha.numel() != 1 or not alpha.is_contiguous() or alpha.data_ptr() % 16):
        raise ValueError("FP8 alpha must be an aligned FP32 scalar on the operand device")
    lo, hi = _span(out)
    if any(t is not None and lo < _span(t)[1] and _span(t)[0] < hi for t in (a, b, sfa, sfb, alpha)):
        raise ValueError("FP8 output must not overlap inputs")
    if m:
        _execute(a, b, sfa, sfb, out, alpha, block_fp8, c_dtype, cuda_stream_to_int(stream))
    return out


@torch.library.custom_op("b12x::fp8_warp", mutates_args=("out",))
def _execute(a: torch.Tensor, b: torch.Tensor, sfa: torch.Tensor | None,
             sfb: torch.Tensor | None, out: torch.Tensor, alpha: torch.Tensor | None,
             block_fp8: bool, c_dtype: str, stream_int: int | None) -> None:
    m, k, groups = a.shape
    with torch.cuda.device(a.device):
        props = torch.cuda.get_device_properties(a.device)
        fn = compile_kernel(b.shape[0], k, groups, c_dtype, block_fp8, alpha is None,
                            a.device.index, props.multi_processor_count, f"sm_{props.major}{props.minor}a")
        sf_type = cutlass.Float32 if block_fp8 else cutlass.Float8E8M0FNU
        types = (cutlass.Float8E4M3FN, cutlass.Float8E4M3FN, sf_type, sf_type,
                 OUTPUT_TYPES[c_dtype][1], cutlass.Float32)
        tensors = (a, b, sfa, sfb, out, alpha)
        fn(*(pointer(t, x) for t, x in zip(types, tensors, strict=True)), cutlass.Int32(m),
           cutlass.Int64(m * k if groups == 1 else a.stride(2)),
           cutlass.Int64(m * out.shape[1] if groups == 1 else out.stride(2)),
           cuda.CUstream(torch.cuda.current_stream(a.device).cuda_stream if stream_int is None else stream_int))


@_execute.register_fake
def _execute_fake(a, b, sfa, sfb, out, alpha, block_fp8, c_dtype, stream_int):
    return None
