"""Execute the production Trellis operand staging on portable Blackwell SIMT."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import pytest
import torch

from b12x._lib.architecture import architecture_for
from b12x.moe._shared.kernels.sm103.launch import pointer
from b12x.moe._shared.kernels.sm103.trellis_gemm import RoutedTrellisGemm
from tests._reference.trellis_decode import (
    CODEBOOK_RATES,
    codebook_tensor,
    native_weight,
)


class InspectOperands:
    """Read the exact staged operand tiles without executing SM103 MMA."""

    def __init__(self, projection):
        self.projection = projection

    @cute.jit
    def __call__(
        self,
        a,
        packed: cute.Pointer,
        lut: cute.Pointer,
        out: cute.Pointer,
        expert: cutlass.Int64,
        route: cutlass.Int64,
        stride: cutlass.Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(a, packed, lut, out, expert, route, stride).launch(
            grid=(
                cute.ceil_div(self.projection.n, 128),
                cute.ceil_div(self.projection.k, 64),
                1,
            ),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        a,
        packed,
        lut,
        out,
        expert: cutlass.Int64,
        route: cutlass.Int64,
        stride: cutlass.Int64,
    ):
        thread, _, _ = cute.arch.thread_idx()
        nn, kk, _ = cute.arch.block_idx()
        layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout((128, 64), stride=(64, 1)),
        )
        allocator = utils.SmemAllocator()
        sA = allocator.allocate_tensor(
            cutlass.Float16, layout.outer, 128, swizzle=layout.inner
        )
        sB = allocator.allocate_tensor(
            cutlass.Float16, layout.outer, 128, swizzle=layout.inner
        )
        a_tensor = self.projection.input_tensors(a, stride)
        packed_tensor = cute.make_tensor(
            packed,
            cute.make_layout(
                self.projection.experts
                * (self.projection.k // 16)
                * (self.projection.n // 16)
                * 8
                * self.projection.bits
            ),
        )
        # Poison both tiles so omitted padding stores fail the exact comparison.
        for item in cutlass.range(thread, 128 * 64, 128):
            sA[item // 64, item % 64] = cutlass.Float16(float("nan"))
            sB[item // 64, item % 64] = cutlass.Float16(float("nan"))
        cute.arch.barrier()
        self.projection.stage_operands(
            a_tensor,
            packed_tensor,
            lut,
            expert,
            route,
            stride,
            cutlass.Int32(kk),
            cutlass.Int64(nn) * 128,
            sA,
            sB,
        )
        cute.arch.barrier()
        output = cute.make_tensor(
            out,
            cute.make_layout(
                (
                    cute.ceil_div(self.projection.n, 128),
                    cute.ceil_div(self.projection.k, 64),
                    2,
                    128,
                    64,
                ),
                stride=(
                    cute.ceil_div(self.projection.k, 64) * 16384,
                    16384,
                    8192,
                    64,
                    1,
                ),
            ),
        )
        for item in cutlass.range(thread, 128 * 64, 128):
            output[nn, kk, 0, item // 64, item % 64] = sA[item // 64, item % 64]
            output[nn, kk, 1, item // 64, item % 64] = sB[item // 64, item % 64]


def _require_gpu():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 3),
        (12, 0),
        (12, 1),
    ):
        pytest.skip("physical Blackwell GPU required")


def _check_staging(codebook, bits, *, high_offsets=False, dual_input=False):
    _require_gpu()
    device = torch.device("cuda", torch.cuda.current_device())
    n, k = 144, 80
    stride = 2**31 + 16 if high_offsets else k + 16
    per_expert_words = (n // 16) * (k // 16) * 8 * bits
    experts = (2**31 // per_expert_words + 2) if high_offsets else 3
    if high_offsets and torch.cuda.mem_get_info()[0] < 19 * 1024**3:
        pytest.skip("high-offset staging requires 19 GiB of free GPU memory")
    a = torch.empty((2, stride), dtype=torch.float16, device=device)
    a[:, :k].normal_(std=0.25)
    cpu = torch.randint(
        -32768,
        32768,
        (1, k // 16, n // 16, 16 * bits),
        dtype=torch.int16,
        generator=torch.Generator().manual_seed(711),
    )
    weights = torch.empty(
        (experts, k // 16, n // 16, 16 * bits), dtype=torch.int16, device=device
    )
    weights[-1].copy_(cpu[0])
    expected_weight = native_weight(cpu, bits, codebook)[0].to(device)
    lut = codebook_tensor(codebook, device)
    output = torch.full(
        (2, 2, 2, 128, 64), float("nan"), dtype=torch.float16, device=device
    )
    projection = RoutedTrellisGemm(
        n, k, experts, 2, bits=bits, codebook=codebook, dual_input=dual_input
    )
    args = [
        pointer(t, v)
        for t, v in (
            (cutlass.Float16, a),
            (cutlass.Uint32, weights),
            (cutlass.Uint8, lut),
            (cutlass.Float16, output),
        )
    ]
    alternate = -a if dual_input else a
    if dual_input:
        args[0] = (args[0], pointer(cutlass.Float16, alternate), cutlass.Int64(64))
    args += [cutlass.Int64(experts - 1), cutlass.Int64(1), cutlass.Int64(stride)]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    target = architecture_for(torch.cuda.get_device_capability()).compilation_target
    fn = cute.compile(
        InspectOperands(projection), *args, stream, options=f"--gpu-arch={target}"
    )
    fn(*args, stream)
    expected = torch.zeros_like(output)
    for nn in range(2):
        for kk in range(2):
            last_n, last_k = min(128, n - nn * 128), min(64, k - kk * 64)
            expected[nn, kk, 0, 0, :last_k] = a[1, kk * 64 : kk * 64 + last_k]
            if dual_input:
                expected[nn, kk, 0, 1, :last_k] = alternate[
                    1, kk * 64 : kk * 64 + last_k
                ]
            expected[nn, kk, 1, :last_n, :last_k] = expected_weight[
                nn * 128 : nn * 128 + last_n, kk * 64 : kk * 64 + last_k
            ]
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    for invalid in (-1, experts, 2**32 + 1):
        output.fill_(float("nan"))
        args[-3] = cutlass.Int64(invalid)
        fn(*args, stream)
        assert torch.isfinite(output).all() and not torch.count_nonzero(output)


@pytest.mark.parametrize("codebook,bits", CODEBOOK_RATES)
def test_production_operand_staging(codebook, bits):
    _check_staging(codebook, bits)


def test_production_operand_staging_above_int32_offsets():
    _check_staging("mcg", 3, high_offsets=True)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_dual_input_operand_staging(bits):
    _check_staging("sqg_e4m3", bits, dual_input=True)
