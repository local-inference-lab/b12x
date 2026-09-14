"""Shared capacity-independent FP8 warp-MMA launch factory."""

from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from .compiler import KernelCompileSpec, compile as b12x_compile
from .dense_gemm import DenseGemmKernel
from .runtime_control import raise_if_kernel_resolution_frozen
from .utils import get_cutlass_dtype, make_ptr


def pointer(dtype):
    return make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=16)


class DenseFp8Launch:
    def __init__(self, n, k, groups, c_dtype, block_fp8, alpha_is_one, sm_count):
        self.n, self.k, self.groups = n, k, groups
        self.block_fp8, self.alpha_is_one = block_fp8, alpha_is_one
        self.swap = n * c_dtype.width % 128 != 0
        self.tile = (64, 16) if self.swap else (16, 64)
        self.sm_count = sm_count
        if min(n, k, groups, sm_count) <= 0 or k % 128:
            raise ValueError(
                "FP8 GEMM requires positive geometry and K divisible by 128"
            )
        if block_fp8 and (n % 128 or groups != 1):
            raise ValueError(
                "compact block-FP8 requires N divisible by 128 and one group"
            )
        if self.swap and groups != 1:
            raise ValueError("unaligned FP8 output rows require one group")

    @cute.jit
    def __call__(
        self,
        a: cute.Pointer,
        b: cute.Pointer,
        sfa: cute.Pointer,
        sfb: cute.Pointer,
        c: cute.Pointer,
        alpha: cute.Pointer,
        rows: cutlass.Int32,
        a_stride: cutlass.Int64,
        c_stride: cutlass.Int64,
        stream: cuda.CUstream,
    ):
        m = cutlass.Int64(rows)
        a_stride = cute.assume(a_stride, divby=16)
        a_tensor = cute.make_tensor(
            a,
            cute.make_layout(
                (m, self.k, self.groups), stride=(cutlass.Int64(self.k), 1, a_stride)
            ),
        )
        b_tensor = cute.make_tensor(
            b,
            cute.make_layout(
                (self.n, self.k, self.groups),
                stride=(cutlass.Int64(self.k), 1, cutlass.Int64(self.n) * self.k),
            ),
        )
        c_tensor = cute.make_tensor(
            c,
            cute.make_layout(
                (m, self.n, self.groups), stride=(cutlass.Int64(self.n), 1, c_stride)
            ),
        )
        alpha_tensor = cute.make_tensor(alpha, cute.make_layout((1,)))
        if cutlass.const_expr(self.block_fp8):
            sfa_tensor = cute.make_tensor(
                sfa,
                cute.make_layout(
                    (m, self.k // 128, 1),
                    stride=(cutlass.Int64(self.k // 128), 1, m * (self.k // 128)),
                ),
            )
            sfb_tensor = cute.make_tensor(
                sfb,
                cute.make_layout(
                    (self.n // 128, self.k // 128, 1),
                    stride=(
                        cutlass.Int64(self.k // 128),
                        1,
                        cutlass.Int64(self.n // 128) * (self.k // 128),
                    ),
                ),
            )
        else:
            sfa_tensor = cute.make_tensor(sfa, cute.make_layout((1,)))
            sfb_tensor = cute.make_tensor(sfb, cute.make_layout((1,)))
        DenseGemmKernel(
            sf_vec_size=128 if self.block_fp8 else 32,
            mma_tiler_mn=self.tile,
            cluster_shape_mn=(1, 1),
            mma_k=32,
            tile_k=128,
            load_path="tma",
            swap_ab=self.swap,
            plain_fp8=True,
            block_fp8=self.block_fp8,
            alpha_is_one=self.alpha_is_one,
            target_occupancy=1,
        )(
            a_tensor,
            a_tensor,
            alpha_tensor,
            alpha_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
            alpha_tensor,
            alpha_tensor,
            alpha_tensor,
            alpha_tensor,
            self.sm_count,
            stream,
        )


@lru_cache(maxsize=1024)
def compile_kernel(
    n,
    k,
    groups,
    c_dtype,
    block_fp8,
    alpha_is_one,
    device_ordinal,
    sm_count,
    architecture,
):
    raise_if_kernel_resolution_frozen("FP8 warp GEMM")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("FP8 warp GEMM must be prewarmed before graph capture")
    if architecture not in ("sm_103a", "sm_120a", "sm_121a"):
        raise ValueError("FP8 warp GEMM requires SM103/SM120/SM121")
    kernel = DenseFp8Launch(
        n, k, groups, get_cutlass_dtype(c_dtype), block_fp8, alpha_is_one, sm_count
    )
    sf_type = cutlass.Float32 if block_fp8 else cutlass.Float8E8M0FNU
    types = (
        cutlass.Float8E4M3FN,
        cutlass.Float8E4M3FN,
        sf_type,
        sf_type,
        get_cutlass_dtype(c_dtype),
        cutlass.Float32,
    )
    spec = KernelCompileSpec.from_facts(
        "gemm.blockscaled.fp8_warp",
        1,
        ("n", n),
        ("k", k),
        ("groups", groups),
        ("c_dtype", c_dtype),
        ("block_fp8", block_fp8),
        ("alpha_is_one", alpha_is_one),
        ("device", device_ordinal),
        ("sm_count", sm_count),
        ("architecture", architecture),
    )
    return b12x_compile(
        kernel,
        *(pointer(t) for t in types),
        cutlass.Int32(1),
        cutlass.Int64(k),
        cutlass.Int64(n),
        cuda.CUstream(0),
        options=f"--gpu-arch={architecture}",
        compile_spec=spec,
    )
