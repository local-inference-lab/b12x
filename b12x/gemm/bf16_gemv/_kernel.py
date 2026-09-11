"""Native unquantized projections with runtime row counts.

BF16 operands use a tiled tensor-core GEMM for broad multi-row projections,
or the original vectorized GEMV for small-row/skinny shapes. FP32 operands
retain their full precision in the strided SIMT reduction. Geometry and
types specialize both implementations; live rows and strides never do.
"""

from __future__ import annotations

from threading import RLock

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass.cute.nvgpu import warp, warpgroup
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils_basic

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.compiler import run_compiled
from b12x._lib.intrinsics import (
    block_reduce,
    get_ptr_as_int64,
    ld_global_v4_u32,
    shared_ptr_to_u32,
    st_shared_v4_u32,
    u32_as_f32,
    warp_reduce,
)
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr
from ._prefill import prefill_mm, supports_prefill

_THREADS = 128
SMALL_M_MAX = 8  # Rows sharing a weight load, not a live-row support limit.
_DTYPES = {torch.bfloat16: BFloat16, torch.float32: Float32}
_NAMES = {torch.bfloat16: "bf16", torch.float32: "fp32"}
_KERNEL_CACHE: dict[tuple, object] = {}
_WARMED: set[tuple] = set()
_LOCK = RLock()


def _fadd(a, b):
    return a + b


@cute.jit
def _flat(pointer: cute.Pointer):
    return cute.make_tensor(pointer, cute.make_layout((Int64(1) << Int64(50),)))


@cute.jit
def _dot_bf16x8(
    source: cute.Tensor,
    offset: Int64,
    w0: Uint32,
    w1: Uint32,
    w2: Uint32,
    w3: Uint32,
    accumulator: Float32,
):
    x0, x1, x2, x3 = ld_global_v4_u32(get_ptr_as_int64(source, offset))
    result = accumulator
    for wv, xv in ((w0, x0), (w1, x1), (w2, x2), (w3, x3)):
        result += u32_as_f32(wv << Uint32(16)) * u32_as_f32(xv << Uint32(16))
        result += u32_as_f32(wv & Uint32(0xFFFF0000)) * u32_as_f32(
            xv & Uint32(0xFFFF0000)
        )
    return result


@cute.jit
def _reduce_store(
    value: Float32,
    reduction: cute.Tensor,
    output: cute.Tensor,
    offset: Int64,
    bias: cute.Tensor,
    column: Int32,
    has_bias: cutlass.Constexpr,
):
    total = block_reduce(warp_reduce(value, _fadd), _fadd, reduction, Float32(0.0))
    thread, _, _ = cute.arch.thread_idx()
    if Int32(thread) == Int32(0):
        if cutlass.const_expr(has_bias):
            total += Float32(bias[column])
        output[offset] = total.to(output.element_type)
    cute.arch.barrier()


class SmallNGemvKernel:
    """One CTA per output column and fixed tile of up to eight live rows."""

    def __init__(self, n: int, k: int, bf16_operands: bool, has_bias: bool):
        self.n, self.k = int(n), int(k)
        self.bf16_operands = bool(bf16_operands)
        self.has_bias = bool(has_bias)

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        weight: cute.Pointer,
        bias: cute.Pointer,
        output: cute.Pointer,
        rows: Int32,
        x_stride: Int64,
        weight_stride: Int64,
        output_stride: Int64,
        x_column_stride: Int64,
        weight_column_stride: Int64,
        vector_loads: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            _flat(x),
            _flat(weight),
            _flat(bias),
            _flat(output),
            rows,
            x_stride,
            weight_stride,
            output_stride,
            x_column_stride,
            weight_column_stride,
            vector_loads,
        ).launch(
            grid=(self.n, (rows + Int32(SMALL_M_MAX - 1)) // Int32(SMALL_M_MAX), 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        source: cute.Tensor,
        weight: cute.Tensor,
        bias: cute.Tensor,
        output: cute.Tensor,
        rows: Int32,
        x_stride: Int64,
        weight_stride: Int64,
        output_stride: Int64,
        x_column_stride: Int64,
        weight_column_stride: Int64,
        vector_loads: Int32,
    ):
        thread, _, _ = cute.arch.thread_idx()
        column, row_tile, _ = cute.arch.block_idx()
        tid = Int32(thread)
        first_row = Int64(row_tile) * Int64(SMALL_M_MAX)
        w_base = Int64(column) * weight_stride
        acc = cute.make_rmem_tensor((SMALL_M_MAX,), Float32)
        for r in cutlass.range_constexpr(SMALL_M_MAX):
            acc[r] = Float32(0.0)
        if cutlass.const_expr(self.bf16_operands and self.k % 8 == 0):
            if vector_loads != Int32(0):
                index = tid
                while index < Int32(self.k // 8):
                    offset = Int64(index) * Int64(8)
                    w0, w1, w2, w3 = ld_global_v4_u32(
                        get_ptr_as_int64(weight, w_base + offset)
                    )
                    for r in cutlass.range_constexpr(SMALL_M_MAX):
                        row = first_row + Int64(r)
                        if row < Int64(rows):
                            acc[r] = _dot_bf16x8(
                                source, row * x_stride + offset, w0, w1, w2, w3, acc[r]
                            )
                    index += Int32(_THREADS)
            else:
                self.scalar_dot(
                    source,
                    weight,
                    acc,
                    first_row,
                    w_base,
                    rows,
                    x_stride,
                    x_column_stride,
                    weight_column_stride,
                    tid,
                )
        else:
            self.scalar_dot(
                source,
                weight,
                acc,
                first_row,
                w_base,
                rows,
                x_stride,
                x_column_stride,
                weight_column_stride,
                tid,
            )
        allocator = cutlass.utils.SmemAllocator()
        reduction = allocator.allocate_tensor(
            Float32, cute.make_layout((1, _THREADS // 32)), byte_alignment=16
        )
        for r in cutlass.range_constexpr(SMALL_M_MAX):
            row = first_row + Int64(r)
            if row < Int64(rows):
                _reduce_store(
                    acc[r],
                    reduction,
                    output,
                    row * output_stride + Int64(column),
                    bias,
                    Int32(column),
                    self.has_bias,
                )

    @cute.jit
    def scalar_dot(
        self,
        source: cute.Tensor,
        weight: cute.Tensor,
        acc: cute.Tensor,
        first_row: Int64,
        w_base: Int64,
        rows: Int32,
        x_stride: Int64,
        x_column_stride: Int64,
        weight_column_stride: Int64,
        tid: Int32,
    ):
        index = tid
        while index < Int32(self.k):
            value = Float32(weight[w_base + Int64(index) * weight_column_stride])
            for r in cutlass.range_constexpr(SMALL_M_MAX):
                row = first_row + Int64(r)
                if row < Int64(rows):
                    acc[r] += (
                        Float32(source[row * x_stride + Int64(index) * x_column_stride])
                        * value
                    )
            index += Int32(_THREADS)


class Bf16GemmKernel:
    """Cooperatively staged BF16 warp-MMA, with runtime strides and M tails."""

    def __init__(self, n: int, k: int, has_bias: bool):
        self.n, self.k = int(n), int(k)
        self.has_bias = bool(has_bias)
        self.tile_m, self.tile_n, self.tile_k = 32, 64, 64

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        weight: cute.Pointer,
        bias: cute.Pointer,
        output: cute.Pointer,
        rows: Int32,
        x_stride: Int64,
        weight_stride: Int64,
        output_stride: Int64,
        x_column_stride: Int64,
        weight_column_stride: Int64,
        vector_loads: Int32,
        stream: cuda.CUstream,
    ):
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)),
            (2, 2, 1),
            permutation_mnk=(self.tile_m, self.tile_n, 16),
        )
        layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR,
                BFloat16,
                self.tile_k,
            ),
            BFloat16,
        )
        layout_a = cute.tile_to_shape(
            layout_atom, (self.tile_m, self.tile_k), order=(0, 1)
        )
        layout_b = cute.tile_to_shape(
            layout_atom, (self.tile_n, self.tile_k), order=(0, 1)
        )
        self.kernel(
            _flat(x),
            _flat(weight),
            _flat(bias),
            _flat(output),
            rows,
            x_stride,
            weight_stride,
            output_stride,
            x_column_stride,
            weight_column_stride,
            vector_loads,
            tiled_mma,
            layout_a,
            layout_b,
        ).launch(
            grid=(
                (rows + Int32(self.tile_m - 1)) // Int32(self.tile_m),
                (self.n + self.tile_n - 1) // self.tile_n,
                1,
            ),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    @cute.jit
    def load_tile(
        self,
        source: cute.Tensor,
        shared: cute.Tensor,
        first_row: Int32,
        row_count: Int32,
        k_base: Int32,
        row_stride: Int64,
        column_stride: Int64,
        tid: Int32,
        vector_loads: Int32,
        tile_rows: cutlass.Constexpr,
    ):
        if vector_loads != Int32(0):
            for step in cutlass.range_constexpr(
                tile_rows * self.tile_k // (8 * _THREADS)
            ):
                index = (tid + Int32(step * _THREADS)) * Int32(8)
                row = index // Int32(self.tile_k)
                column = index % Int32(self.tile_k)
                v0, v1, v2, v3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                if first_row + row < row_count and k_base + column < Int32(self.k):
                    offset = Int64(first_row + row) * row_stride + Int64(
                        k_base + column
                    )
                    v0, v1, v2, v3 = ld_global_v4_u32(get_ptr_as_int64(source, offset))
                destination = shared_ptr_to_u32(
                    shared.iterator + cute.crd2idx((row, column), shared.layout)
                )
                st_shared_v4_u32(destination, v0, v1, v2, v3)
        else:
            for step in cutlass.range_constexpr(tile_rows * self.tile_k // _THREADS):
                index = tid + Int32(step * _THREADS)
                row = index // Int32(self.tile_k)
                column = index % Int32(self.tile_k)
                value = BFloat16(0.0)
                if first_row + row < row_count and k_base + column < Int32(self.k):
                    value = source[
                        Int64(first_row + row) * row_stride
                        + Int64(k_base + column) * column_stride
                    ]
                shared[row, column] = value

    @cute.kernel
    def kernel(
        self,
        source: cute.Tensor,
        weight: cute.Tensor,
        bias: cute.Tensor,
        output: cute.Tensor,
        rows: Int32,
        x_stride: Int64,
        weight_stride: Int64,
        output_stride: Int64,
        x_column_stride: Int64,
        weight_column_stride: Int64,
        vector_loads: Int32,
        tiled_mma: cute.TiledMma,
        layout_a: cute.ComposedLayout,
        layout_b: cute.ComposedLayout,
    ):
        thread, _, _ = cute.arch.thread_idx()
        m_tile, n_tile, _ = cute.arch.block_idx()
        allocator = cutlass.utils.SmemAllocator()
        shared_a = allocator.allocate_tensor(BFloat16, layout_a, byte_alignment=1024)
        shared_b = allocator.allocate_tensor(BFloat16, layout_b, byte_alignment=1024)
        thread_mma = tiled_mma.get_slice(thread)
        register_a = thread_mma.make_fragment_A(thread_mma.partition_A(shared_a))
        register_b = thread_mma.make_fragment_B(thread_mma.partition_B(shared_b))
        accumulator = cute.make_rmem_tensor(
            thread_mma.partition_shape_C((self.tile_m, self.tile_n)),
            Float32,
        )
        accumulator.fill(0.0)
        accumulated = cute.make_rmem_tensor(
            thread_mma.partition_shape_C((self.tile_m, self.tile_n)),
            Float32,
        )
        accumulated.fill(0.0)
        copy_atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            BFloat16,
        )
        copy_a = cute.make_tiled_copy_A(copy_atom, tiled_mma).get_slice(thread)
        copy_b = cute.make_tiled_copy_B(copy_atom, tiled_mma).get_slice(thread)
        source_a, source_b = copy_a.partition_S(shared_a), copy_b.partition_S(shared_b)
        target_a, target_b = copy_a.retile(register_a), copy_b.retile(register_b)
        for k_tile in cutlass.range((self.k + self.tile_k - 1) // self.tile_k):
            k_base = Int32(k_tile * self.tile_k)
            self.load_tile(
                source,
                shared_a,
                Int32(m_tile * self.tile_m),
                rows,
                k_base,
                x_stride,
                x_column_stride,
                Int32(thread),
                vector_loads,
                self.tile_m,
            )
            self.load_tile(
                weight,
                shared_b,
                Int32(n_tile * self.tile_n),
                Int32(self.n),
                k_base,
                weight_stride,
                weight_column_stride,
                Int32(thread),
                vector_loads,
                self.tile_n,
            )
            cute.arch.barrier()
            cute.copy(copy_a, source_a[None, None, 0], target_a[None, None, 0])
            cute.copy(copy_b, source_b[None, None, 0], target_b[None, None, 0])
            for k_step in cutlass.range_constexpr(self.tile_k // 16):
                if cutlass.const_expr(k_step + 1 < self.tile_k // 16):
                    cute.copy(
                        copy_a,
                        source_a[None, None, k_step + 1],
                        target_a[None, None, k_step + 1],
                    )
                    cute.copy(
                        copy_b,
                        source_b[None, None, k_step + 1],
                        target_b[None, None, k_step + 1],
                    )
                cute.gemm(
                    tiled_mma,
                    accumulator,
                    register_a[None, None, k_step],
                    register_b[None, None, k_step],
                    accumulator,
                )
            if cutlass.const_expr(self.k > 4096):
                # Limit long-K accumulation depth before combining FP32
                # partials; this preserves small cancellation residuals.
                if (k_tile + 1) % 16 == 0 or k_tile + 1 == (
                    self.k + self.tile_k - 1
                ) // self.tile_k:
                    for index in cutlass.range_constexpr(cute.size(accumulator)):
                        accumulated[index] += accumulator[index]
                        accumulator[index] = Float32(0.0)
            cute.arch.barrier()
        coordinates = thread_mma.partition_C(
            cute.make_identity_tensor((self.tile_m, self.tile_n))
        )
        for index in cutlass.range_constexpr(cute.size(accumulator)):
            coordinate = coordinates[index]
            row = Int32(m_tile * self.tile_m) + coordinate[0]
            column = Int32(n_tile * self.tile_n) + coordinate[1]
            if row < rows and column < Int32(self.n):
                if cutlass.const_expr(self.k > 4096):
                    value = accumulated[index]
                else:
                    value = accumulator[index]
                if cutlass.const_expr(self.has_bias):
                    value += Float32(bias[column])
                output[Int64(row) * output_stride + Int64(column)] = value.to(
                    output.element_type
                )


class ProjectionKernel:
    """One geometry-specialized callable for every live row count."""

    def __init__(self, n: int, k: int, bf16_operands: bool, has_bias: bool):
        self.simt = SmallNGemvKernel(n, k, bf16_operands, has_bias)
        self.mma = Bf16GemmKernel(n, k, has_bias)
        self.has_mma = bf16_operands and n >= 256 and k >= 16
        n_tiles = (n + 63) // 64
        self.minimum_mma_rows = (
            24 if n_tiles >= 64 else ((64 + n_tiles - 1) // n_tiles) * 32
        )

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        weight: cute.Pointer,
        bias: cute.Pointer,
        output: cute.Pointer,
        rows: Int32,
        x_stride: Int64,
        weight_stride: Int64,
        output_stride: Int64,
        x_column_stride: Int64,
        weight_column_stride: Int64,
        vector_loads: Int32,
        warm_all: Int32,
        stream: cuda.CUstream,
    ):
        arguments = (
            x,
            weight,
            bias,
            output,
            rows,
            x_stride,
            weight_stride,
            output_stride,
            x_column_stride,
            weight_column_stride,
            vector_loads,
            stream,
        )
        if cutlass.const_expr(self.has_mma):
            if warm_all == Int32(2):
                self.simt(*arguments)
            elif warm_all != Int32(0):
                self.mma(*arguments)
                self.simt(*arguments)
            elif rows >= Int32(self.minimum_mma_rows):
                self.mma(*arguments)
            else:
                self.simt(*arguments)
        else:
            self.simt(*arguments)


def _pointer(tensor: torch.Tensor):
    return make_ptr(
        _DTYPES[tensor.dtype],
        tensor.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=tensor.element_size(),
    )


def _key(x, weight, out, bias):
    device = x.device.index
    if device is None:
        device = torch.cuda.current_device()
    return (
        int(device),
        int(weight.shape[0]),
        int(weight.shape[1]),
        _NAMES[x.dtype],
        _NAMES[weight.dtype],
        _NAMES[out.dtype],
        None if bias is None else _NAMES[bias.dtype],
    )


def _compile(key, x_dtype, weight_dtype, out_dtype, bias_dtype, device):
    with _LOCK:
        cached = _KERNEL_CACHE.get(key)
        if cached is not None:
            return cached
        kernel = ProjectionKernel(
            key[1],
            key[2],
            x_dtype == weight_dtype == torch.bfloat16,
            bias_dtype is not None,
        )
        raise_if_kernel_resolution_frozen("cute.compile", target=kernel, cache_key=key)
        types = (
            x_dtype,
            weight_dtype,
            x_dtype if bias_dtype is None else bias_dtype,
            out_dtype,
        )
        pointers = [
            make_ptr(
                _DTYPES[dtype], 16, cute.AddressSpace.gmem, assumed_align=dtype.itemsize
            )
            for dtype in types
        ]
        with torch.cuda.device(device):
            compiled = b12x_compile(
                kernel,
                *pointers,
                Int32(1),
                Int64(key[2]),
                Int64(key[2]),
                Int64(key[1]),
                Int64(1),
                Int64(1),
                Int32(0),
                Int32(0),
                current_cuda_stream(),
                compile_spec=KernelCompileSpec.from_key(
                    "gemm.bf16_projection",
                    1,
                    key,
                ),
            )
        _KERNEL_CACHE[key] = compiled
        return compiled


def _validate(x, weight, out, bias):
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("projection requires x[M,K] and weight[N,K]")
    if not x.is_cuda or weight.device != x.device or out.device != x.device:
        raise ValueError("native unquantized projection requires one CUDA device")
    if (
        x.dtype not in _DTYPES
        or weight.dtype not in _DTYPES
        or out.dtype not in _DTYPES
    ):
        raise TypeError("native unquantized projection supports BF16 and FP32")
    if weight.shape[0] <= 0 or weight.shape[1] <= 0 or x.shape[0] > 2**31 - 1:
        raise ValueError("projection geometry must be positive and live rows fit int32")
    if tuple(out.shape) != (x.shape[0], weight.shape[0]):
        raise ValueError("out must have shape [M,N]")
    if out.stride(1) != 1 or out.stride(0) < out.shape[1]:
        raise ValueError("out needs unit column stride and disjoint rows")
    if any(stride < 0 for tensor in (x, weight) for stride in tensor.stride()):
        raise ValueError("projection input strides must be nonnegative")
    if bias is not None and (
        bias.device != x.device
        or bias.dtype not in _DTYPES
        or tuple(bias.shape) != (weight.shape[0],)
        or not bias.is_contiguous()
    ):
        raise ValueError("bias must be contiguous BF16/FP32 [N] on the input device")
    if not out.numel():
        return
    output_start = out.data_ptr()
    output_end = (
        output_start
        + ((out.shape[0] - 1) * out.stride(0) + out.shape[1]) * out.element_size()
    )
    for tensor in (x, weight) if bias is None else (x, weight, bias):
        span = 1 + sum(
            (size - 1) * stride
            for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
        )
        start, end = tensor.data_ptr(), tensor.data_ptr() + span * tensor.element_size()
        if output_start < end and start < output_end:
            raise ValueError("projection output must not overlap its inputs")


def _launch(x, weight, out, bias=None):
    # The 1024-column projection reaches the general MMA crossover at 128
    # rows. Keep its qualified FP32 accumulation accuracy through that range.
    prefill_min_rows = 128 if weight.ndim == 2 and weight.shape[0] == 1024 else 256
    if (
        x.ndim == weight.ndim == 2
        and x.shape[0] >= prefill_min_rows
        and supports_prefill(x, weight, out, bias)
    ):
        _validate(x, weight, out, bias)
        prefill_mm(x, weight, out)
        return
    _launch_projection(x, weight, out, bias)


def _launch_scalar(x, weight, out, bias=None):
    """Invoke the SIMT reference explicitly, regardless of the MMA crossover."""
    _launch_projection(x, weight, out, bias, force_simt=True)


def _launch_projection(x, weight, out, bias=None, *, force_simt=False):
    _validate(x, weight, out, bias)
    if x.shape[0] == 0:
        return
    key = _key(x, weight, out, bias)
    with torch.cuda.device(x.device):
        capturing = torch.cuda.is_current_stream_capturing()
        with _LOCK:
            compiled = _KERNEL_CACHE.get(key)
            warmed = key in _WARMED
        if capturing and not warmed:
            raise RuntimeError(
                "native unquantized projection must be warm-run before CUDA graph capture"
            )
        if compiled is None:
            compiled = _compile(
                key,
                x.dtype,
                weight.dtype,
                out.dtype,
                None if bias is None else bias.dtype,
                x.device,
            )
        vector_loads = int(
            x.data_ptr() % 16 == 0
            and weight.data_ptr() % 16 == 0
            and x.stride(0) % 8 == 0
            and weight.stride(0) % 8 == 0
            and x.stride(1) == 1
            and weight.stride(1) == 1
            and weight.shape[1] % 8 == 0
        )
        arguments = (
            _pointer(x),
            _pointer(weight),
            _pointer(x if bias is None else bias),
            _pointer(out),
            int(x.shape[0]),
            int(x.stride(0)),
            int(weight.stride(0)),
            int(out.stride(0)),
            int(x.stride(1)),
            int(weight.stride(1)),
            vector_loads,
            2 if force_simt else 0,
            current_cuda_stream(),
        )
        # The same compiled host program owns both GPU entrypoints. Its row
        # scalar controls execution without resolving a different callable.
        if not warmed:
            warm_arguments = (*arguments[:4], 1, *arguments[5:-2], 1, arguments[-1])
            run_compiled(compiled, warm_arguments)
        run_compiled(compiled, arguments)
        if not capturing:
            with _LOCK:
                _WARMED.add(key)


@torch.library.custom_op("b12x::bf16_gemv_small_n", mutates_args=())
def bf16_gemv_small_n(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    dtype = x.dtype if output_dtype is None else output_dtype
    out = torch.empty((x.shape[0], weight.shape[0]), dtype=dtype, device=x.device)
    _launch(x, weight, out, bias)
    return out


@bf16_gemv_small_n.register_fake
def _bf16_gemv_small_n_fake(x, weight, bias=None, output_dtype=None):
    return x.new_empty(
        (x.shape[0], weight.shape[0]),
        dtype=x.dtype if output_dtype is None else output_dtype,
    )


@torch.library.custom_op("b12x::bf16_gemv_small_n_out", mutates_args=("out",))
def bf16_gemv_small_n_out(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> None:
    _launch(x, weight, out, bias)


@bf16_gemv_small_n_out.register_fake
def _bf16_gemv_small_n_out_fake(x, weight, out, bias=None):
    return None


def precompile_bf16_gemv_small_n(
    weight: torch.Tensor,
    log=None,
    *,
    input_dtype: torch.dtype = torch.bfloat16,
    output_dtype: torch.dtype = torch.bfloat16,
    bias: torch.Tensor | None = None,
) -> None:
    """Compile and warm every eligible implementation, independent of live M."""
    if weight.ndim != 2 or not weight.is_cuda or weight.dtype not in _DTYPES:
        raise ValueError("precompile requires a CUDA BF16/FP32 weight matrix")
    if input_dtype not in _DTYPES or output_dtype not in _DTYPES:
        raise TypeError("projection input/output dtype must be BF16 or FP32")
    with torch.cuda.device(weight.device):
        x = torch.zeros((1, weight.shape[1]), dtype=input_dtype, device=weight.device)
        out = torch.empty(
            (1, weight.shape[0]), dtype=output_dtype, device=weight.device
        )
        key = _key(x, weight, out, bias)
        with _LOCK:
            scalar_warm = key in _WARMED
        if not scalar_warm:
            _launch(x, weight, out, bias)
        if supports_prefill(x, weight, out, bias):
            prefill_mm(x, weight, out)
        torch.cuda.current_stream(weight.device).synchronize()
    if log is not None:
        log.debug(
            "native projection warm: N=%d K=%d %s/%s -> %s",
            weight.shape[0],
            weight.shape[1],
            input_dtype,
            weight.dtype,
            output_dtype,
        )
