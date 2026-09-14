"""Planned vocabulary GEMV using the shared CuTe BF16 reduction kernel."""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile, run_compiled
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x.gemm.bf16_gemv._kernel import SmallNGemvKernel

_WARMED: set[tuple[int, int, int, str]] = set()


def validate_output(source, weight, output):
    lo, hi = (
        output.data_ptr(),
        output.data_ptr() + output.numel() * output.element_size(),
    )
    for tensor in (source, weight):
        start = tensor.data_ptr()
        if lo < start + tensor.numel() * tensor.element_size() and start < hi:
            raise ValueError("vocabulary projection out must not overlap its inputs")


def _pointer(address):
    return make_ptr(cutlass.BFloat16, address, cute.AddressSpace.gmem, assumed_align=2)


@cache
def compile_kernel(n: int, k: int, device: int, architecture: str):
    raise_if_kernel_resolution_frozen("CuTe vocabulary projection")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("vocabulary projection must be planned before graph capture")
    if architecture not in ("sm_103a", "sm_120a", "sm_121a"):
        raise ValueError("CuTe vocabulary projection requires SM103/SM120/SM121")
    return b12x_compile(
        SmallNGemvKernel(n, k, True, False),
        *(_pointer(16) for _ in range(4)),
        cutlass.Int32(1),
        cutlass.Int64(k),
        cutlass.Int64(k),
        cutlass.Int64(n),
        cutlass.Int64(1),
        cutlass.Int64(1),
        cutlass.Int32(0),
        cuda.CUstream(0),
        options=f"--gpu-arch={architecture}",
        compile_spec=KernelCompileSpec.from_key(
            "gemm.bf16_vocab_projection.cute", 1, (n, k, device, architecture)
        ),
    )


@torch.library.custom_op("b12x::bf16_vocab_projection_cute", mutates_args=("output",))
def execute(
    source: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    architecture: str,
) -> None:
    validate_output(source, weight, output)
    n, k = weight.shape
    device = source.device.index
    key = (n, k, device, architecture)
    with torch.cuda.device(source.device):
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and key not in _WARMED:
            raise RuntimeError(
                "CuTe vocabulary projection must run before graph capture"
            )
        compiled = compile_kernel(*key)
        vector_loads = int(
            k % 8 == 0 and source.data_ptr() % 16 == weight.data_ptr() % 16 == 0
        )
        run_compiled(
            compiled,
            (
                _pointer(source.data_ptr()),
                _pointer(weight.data_ptr()),
                _pointer(source.data_ptr()),
                _pointer(output.data_ptr()),
                int(source.shape[0]),
                int(source.stride(0)),
                int(weight.stride(0)),
                int(output.stride(0)),
                1,
                1,
                vector_loads,
                current_cuda_stream(),
            ),
        )
        if not capturing:
            _WARMED.add(key)


@execute.register_fake
def _execute_fake(source, weight, output, architecture):
    return None
