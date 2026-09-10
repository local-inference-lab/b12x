"""Weight-first brick GEMV kernel (CuTe DSL) and its opaque custom op.

Geometry. The concatenated BF16 weight ``w (N, K)`` is cut into bricks of
``NT`` rows by ``KT`` columns; the grid is ``ceil(N / NT)`` n-tiles by
``S = K / KT`` k-splits, one CTA of 256 threads (8 warps) per brick. Each CTA:

1. executes ``griddepcontrol.launch_dependents`` so a dependent launched with
   the programmatic-stream-serialization attribute may start early;
2. copies its brick into shared memory with 16-byte ``cp.async`` in groups of
   256 copies (4 KB); ``depth`` bounds the committed groups left in flight per
   CTA (``depth == 0`` issues the whole brick at once). Rows are padded by 8
   elements so ``ldmatrix`` rows spread over the banks;
3. executes ``griddepcontrol.wait`` (a no-op unless the launch carried the
   attribute and the previous kernel on the stream triggered early), then
   stages its ``[M, KT]`` slice of ``x``;
4. computes the products with ``mma.sync m16n8k16`` (bf16 in, fp32
   accumulate). Warp ``w`` owns columns ``[w * KT / 8, (w + 1) * KT / 8)`` of
   the brick; rows of ``x`` beyond ``M`` are never written out;
5. sums the eight warp tiles in warp order in shared memory, then either
   writes bf16 (``S == 1``) or writes its fp32 partial and, if it is the last
   CTA of its n-tile to arrive (acquire-release counter), sums the ``S``
   partials in split order and writes bf16. The counter is reset for the next
   launch, so graph replay needs no host reset.

The reduction order is fixed by the geometry alone, so the output is bitwise
repeatable across launches and row counts.

Compile keys hold only static geometry (``N``, ``K``, ``NT``, ``KT``) plus the
device and toolchain identity of the compile cache; the row count ``M``, the
first weight's row count ``N0``, the staging depth and the launch attribute
are runtime scalar arguments.
"""

from __future__ import annotations

from typing import Dict, Tuple

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import torch
from cutlass import Int64
from cutlass.cutlass_dsl import Int32

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.intrinsics import (
    atomic_add_global_acq_rel_i32,
    bf16_mma_m16n8k16_f32,
    cp_async4_shared_global,
    cvt_f32_to_bf16_bits,
    ld_global_cg_f32,
    ld_shared_f32,
    ld_shared_i32,
    ldmatrix_m8n8x2_b16,
    ldmatrix_m8n8x4_b16,
    shared_ptr_to_u32,
    st_global_f32,
    st_global_i32,
    st_global_u16,
    st_shared_f32,
    st_shared_i32,
)
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr

_THREADS = 256
_WARPS = _THREADS // 32

#: Largest row count the kernel serves; larger inputs use cuBLAS inside the op.
MAX_ROWS = 16

#: Staging depths the kernel honors: ``cp.async`` groups left in flight per
#: CTA while a brick is staged (the ``cp_async_wait_group`` chain of the
#: brick staging loop); 0 issues the whole brick at once. Any other value would
#: fall through that chain and stage unbounded, so callers reject it.
STAGE_DEPTHS = frozenset({0, 1, 2, 3, 4, 6, 8})

#: Brick shapes the kernel is compiled for (rows, columns).
BRICKS: Tuple[Tuple[int, int], ...] = ((48, 768), (32, 768), (64, 512), (128, 256))

_KERNEL_CACHE: Dict[Tuple[int, int, int, int], object] = {}


def brick_for(n: int, k: int) -> Tuple[int, int]:
    """Brick shape for a weight ``(n, k)``: the widest column count that
    divides ``k`` with enough rows that one CTA stays under the 99 KB
    shared-memory limit and enough CTAs stream the weight in the window."""
    if k % 768 == 0 and n >= 512:
        return 48, 768
    if k % 768 == 0:
        return 32, 768
    if k % 512 == 0:
        return 64, 512
    if k % 256 == 0:
        return 128, 256
    raise ValueError(f"no weight-first brick for N={n}, K={k}")


def smem_bytes(nt: int, kt: int) -> int:
    """Shared memory of one CTA: the padded brick plus a padded 16-row slice
    of the activation, followed by a 16-byte control word."""
    return (nt + 16) * (kt + 8) * 2 + 16


class WeightFirstGemvKernel:
    """One CTA per ``NT x KT`` brick of ``w``; see the module docstring."""

    def __init__(self, n: int, k: int, nt: int, kt: int):
        if (nt, kt) not in BRICKS:
            raise ValueError(f"unsupported brick {nt}x{kt}")
        if k % kt != 0:
            raise ValueError(f"K={k} is not a multiple of the brick width {kt}")
        self.n = int(n)
        self.k = int(k)
        self.nt = int(nt)
        self.kt = int(kt)
        self.kt8 = kt // 8  # 16-byte chunks per brick row
        self.ld8 = self.kt8 + 1  # padded row stride in 16-byte chunks
        self.ld = self.ld8 * 8  # padded row stride in elements
        self.k8 = k // 8  # 16-byte chunks per weight row
        self.ntiles = nt // 8  # 8-column MMA tiles per brick row block
        self.kw = kt // _WARPS  # columns per warp
        self.ksteps = self.kw // 16
        self.splits = k // kt
        self.grid_n = (n + nt - 1) // nt
        self.opt = (16 * nt + _THREADS - 1) // _THREADS
        self.x_offset = nt * self.ld * 2  # bytes: start of the x slice
        self.control_offset = (nt + 16) * self.ld * 2  # bytes: last-CTA flag
        self.smem_bytes = smem_bytes(nt, kt)
        assert self.kw % 16 == 0
        assert _WARPS * 16 * nt * 4 <= nt * self.ld * 2

    @cute.jit
    def __call__(
        self,
        x_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        y0_ptr: cute.Pointer,
        y1_ptr: cute.Pointer,
        partial_ptr: cute.Pointer,
        counters_ptr: cute.Pointer,
        m_rows: Int32,
        n0: Int32,
        depth: Int32,
        use_pdl: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            x_ptr,
            w_ptr,
            y0_ptr,
            y1_ptr,
            partial_ptr,
            counters_ptr,
            m_rows,
            n0,
            depth,
        ).launch(
            grid=(self.grid_n, self.splits, 1),
            block=[_THREADS, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        x_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        y0_ptr: cute.Pointer,
        y1_ptr: cute.Pointer,
        partial_ptr: cute.Pointer,
        counters_ptr: cute.Pointer,
        m_rows: Int32,
        n0: Int32,
        depth: Int32,
    ):
        # A dependent launched with the programmatic-stream-serialization
        # attribute may start now; its own griddepcontrol.wait still orders
        # its reads of this kernel's output after this grid completes.
        cute.arch.griddepcontrol_launch_dependents()

        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        warp = tid >> Int32(5)
        lane = tid & Int32(31)
        n_tile = Int32(bidx)
        split = Int32(bidy)
        tile_row0 = n_tile * Int32(self.nt)
        k0_chunk = split * Int32(self.kt8)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.smem_bytes // 4],
                128,
            ]

        storage = smem.allocate(Storage)
        sw = shared_ptr_to_u32(storage.words.data_ptr())
        sx = sw + Int32(self.x_offset)
        s_control = sw + Int32(self.control_offset)

        rows = Int32(self.nt)
        if Int32(self.n) - tile_row0 < Int32(self.nt):
            rows = Int32(self.n) - tile_row0

        # 1. Stage the brick: groups of 256 copies of 16 bytes; `depth`
        #    groups may remain in flight per CTA (0: unbounded), which bounds
        #    the request pressure this kernel puts beside the allreduce's
        #    peer reads of this GPU's memory.
        w_addr = Int64(w_ptr.toint())
        brick_chunk0 = Int64(tile_row0) * Int64(self.k8) + Int64(k0_chunk)
        total = rows * Int32(self.kt8)
        base = Int32(0)
        while base < total:
            i = base + tid
            if i < total:
                r = i // Int32(self.kt8)
                c = i - r * Int32(self.kt8)
                gmem = w_addr + (
                    brick_chunk0 + Int64(r) * Int64(self.k8) + Int64(c)
                ) * Int64(16)
                cp_async4_shared_global(
                    sw + (r * Int32(self.ld8) + c) * Int32(16), gmem
                )
            cute.arch.cp_async_commit_group()
            if depth == Int32(1):
                cute.arch.cp_async_wait_group(1)
            elif depth == Int32(2):
                cute.arch.cp_async_wait_group(2)
            elif depth == Int32(3):
                cute.arch.cp_async_wait_group(3)
            elif depth == Int32(4):
                cute.arch.cp_async_wait_group(4)
            elif depth == Int32(6):
                cute.arch.cp_async_wait_group(6)
            elif depth == Int32(8):
                cute.arch.cp_async_wait_group(8)
            base += Int32(_THREADS)

        # 2. The activation is produced by the kernel before this one.
        cute.arch.griddepcontrol_wait()
        x_addr = Int64(x_ptr.toint())
        total_x = m_rows * Int32(self.kt8)
        i = tid
        while i < total_x:
            m = i // Int32(self.kt8)
            c = i - m * Int32(self.kt8)
            gmem = x_addr + (
                Int64(m) * Int64(self.k8) + Int64(k0_chunk) + Int64(c)
            ) * Int64(16)
            cp_async4_shared_global(sx + (m * Int32(self.ld8) + c) * Int32(16), gmem)
            i += Int32(_THREADS)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # 3. Tensor-core products. ldmatrix x4 loads the 16x16 A tile of x
        #    (lanes 0-15 address rows 0-15 at column 0, lanes 16-31 the same
        #    rows at column 8); ldmatrix x2 loads the 8x16 B tile of the
        #    brick (lanes 0-7 rows at column 0, lanes 8-15 at column 8).
        kw0 = warp * Int32(self.kw)
        a_base = sx + (
            (lane & Int32(15)) * Int32(self.ld) + kw0 + (lane >> Int32(4)) * Int32(8)
        ) * Int32(2)
        b_base = sw + (
            (lane & Int32(7)) * Int32(self.ld)
            + kw0
            + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
        ) * Int32(2)
        acc = [
            [
                cutlass.Float32(0.0),
                cutlass.Float32(0.0),
                cutlass.Float32(0.0),
                cutlass.Float32(0.0),
            ]
            for _ in range(self.ntiles)
        ]
        for ks in cutlass.range_constexpr(self.ksteps):
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(a_base + Int32(ks * 32))
            for t in cutlass.range_constexpr(self.ntiles):
                b0, b1 = ldmatrix_m8n8x2_b16(
                    b_base + Int32(t * 8 * self.ld * 2 + ks * 32)
                )
                d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                    acc[t][0], acc[t][1], acc[t][2], acc[t][3], a0, a1, a2, a3, b0, b1
                )
                acc[t][0] = d0
                acc[t][1] = d1
                acc[t][2] = d2
                acc[t][3] = d3
        cute.arch.sync_threads()

        # 4. Warp tiles into shared memory (the brick region is free now):
        #    red[warp][16 rows][NT columns] fp32.
        row = lane >> Int32(2)
        col = (lane & Int32(3)) * Int32(2)
        red = sw
        dst = red + (warp * Int32(16 * self.nt)) * Int32(4)
        for t in cutlass.range_constexpr(self.ntiles):
            c0 = t * 8
            st_shared_f32(
                dst + (row * Int32(self.nt) + col + Int32(c0)) * Int32(4), acc[t][0]
            )
            st_shared_f32(
                dst + (row * Int32(self.nt) + col + Int32(c0 + 1)) * Int32(4), acc[t][1]
            )
            st_shared_f32(
                dst + ((row + Int32(8)) * Int32(self.nt) + col + Int32(c0)) * Int32(4),
                acc[t][2],
            )
            st_shared_f32(
                dst
                + ((row + Int32(8)) * Int32(self.nt) + col + Int32(c0 + 1)) * Int32(4),
                acc[t][3],
            )
        cute.arch.sync_threads()

        # 5. Sum the eight warp tiles in warp order; thread `tid` owns the
        #    outputs i = tid + j * 256 of the M x rows brick outputs.
        n_out = m_rows * rows
        y0_addr = Int64(y0_ptr.toint())
        y1_addr = Int64(y1_ptr.toint())
        n1 = Int32(self.n) - n0
        vals = [cutlass.Float32(0.0) for _ in range(self.opt)]
        for j in cutlass.range_constexpr(self.opt):
            i = tid + Int32(j * _THREADS)
            v = cutlass.Float32(0.0)
            if i < n_out:
                m = i // rows
                r = i - m * rows
                for wr in cutlass.range_constexpr(_WARPS):
                    v = v + ld_shared_f32(
                        red + ((Int32(wr * 16) + m) * Int32(self.nt) + r) * Int32(4)
                    )
            vals[j] = v

        if cutlass.const_expr(self.splits == 1):
            for j in cutlass.range_constexpr(self.opt):
                i = tid + Int32(j * _THREADS)
                if i < n_out:
                    m = i // rows
                    r = i - m * rows
                    n = tile_row0 + r
                    bits = cvt_f32_to_bf16_bits(vals[j])
                    if n < n0:
                        st_global_u16(
                            y0_addr + (Int64(m) * Int64(n0) + Int64(n)) * Int64(2), bits
                        )
                    else:
                        st_global_u16(
                            y1_addr + (Int64(m) * Int64(n1) + Int64(n - n0)) * Int64(2),
                            bits,
                        )
        else:
            partial_addr = Int64(partial_ptr.toint())
            split_stride = Int64(16 * self.n)  # fp32 elements per split
            out_partial = partial_addr + Int64(split) * split_stride * Int64(4)
            for j in cutlass.range_constexpr(self.opt):
                i = tid + Int32(j * _THREADS)
                if i < n_out:
                    m = i // rows
                    r = i - m * rows
                    st_global_f32(
                        out_partial
                        + (Int64(m) * Int64(self.n) + Int64(tile_row0) + Int64(r))
                        * Int64(4),
                        vals[j],
                    )
            # Last CTA of this n-tile (release/acquire counter; reset below
            # so graph replay needs no host fill) sums the S partials in
            # split order.
            cute.arch.sync_threads()
            counter_addr = Int64(counters_ptr.toint()) + Int64(n_tile) * Int64(4)
            if tid == Int32(0):
                old = atomic_add_global_acq_rel_i32(counter_addr, Int32(1))
                flag = Int32(0)
                if old == Int32(self.splits - 1):
                    flag = Int32(1)
                st_shared_i32(s_control, flag)
            cute.arch.sync_threads()
            is_last = ld_shared_i32(s_control)
            if is_last != Int32(0):
                cute.arch.fence_acq_rel_gpu()
                for j in cutlass.range_constexpr(self.opt):
                    i = tid + Int32(j * _THREADS)
                    if i < n_out:
                        m = i // rows
                        r = i - m * rows
                        src = partial_addr + (
                            Int64(m) * Int64(self.n) + Int64(tile_row0) + Int64(r)
                        ) * Int64(4)
                        acc4 = [
                            cutlass.Float32(0.0),
                            cutlass.Float32(0.0),
                            cutlass.Float32(0.0),
                            cutlass.Float32(0.0),
                        ]
                        for quad in cutlass.range_constexpr(self.splits // 4):
                            for q in cutlass.range_constexpr(4):
                                acc4[q] = acc4[q] + ld_global_cg_f32(
                                    src + Int64(quad * 4 + q) * split_stride * Int64(4)
                                )
                        v = cutlass.Float32(0.0)
                        for s in cutlass.range_constexpr(
                            (self.splits // 4) * 4, self.splits
                        ):
                            v = v + ld_global_cg_f32(
                                src + Int64(s) * split_stride * Int64(4)
                            )
                        v = ((acc4[0] + acc4[1]) + (acc4[2] + acc4[3])) + v
                        n = tile_row0 + r
                        bits = cvt_f32_to_bf16_bits(v)
                        if n < n0:
                            st_global_u16(
                                y0_addr + (Int64(m) * Int64(n0) + Int64(n)) * Int64(2),
                                bits,
                            )
                        else:
                            st_global_u16(
                                y1_addr
                                + (Int64(m) * Int64(n1) + Int64(n - n0)) * Int64(2),
                                bits,
                            )
                if tid == Int32(0):
                    st_global_i32(counter_addr, Int32(0))


def _dummy(dtype, alignment: int):
    return make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=alignment)


def compile_weight_first_gemv(n: int, k: int, nt: int, kt: int):
    """Compile the brick GEMV for the static geometry ``(n, k, nt, kt)``.

    Returns ``launch(x, w, y0, y1, partial, counters, m, n0, depth, pdl)``
    where ``x`` is ``(m, k)`` bf16 contiguous, ``w`` the concatenated
    ``(n, k)`` bf16 weight, ``y0`` ``(m, n0)`` and ``y1`` ``(m, n - n0)`` bf16
    outputs, ``partial`` an ``(k // kt, 16, n)`` fp32 workspace and
    ``counters`` an int32 workspace of ``ceil(n / nt)`` zeros.
    """
    cache_key = (int(n), int(k), int(nt), int(kt))
    cached = _KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    kernel = WeightFirstGemvKernel(n, k, nt, kt)
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    raw = b12x_compile(
        kernel,
        _dummy(cutlass.BFloat16, 16),
        _dummy(cutlass.BFloat16, 16),
        _dummy(cutlass.BFloat16, 16),
        _dummy(cutlass.BFloat16, 16),
        _dummy(cutlass.Float32, 16),
        _dummy(cutlass.Int32, 16),
        1,
        1,
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "gemm.weight_first_gemv",
            1,
            cache_key,
        ),
    )

    def launch(
        x: torch.Tensor,
        w: torch.Tensor,
        y0: torch.Tensor,
        y1: torch.Tensor,
        partial: torch.Tensor,
        counters: torch.Tensor,
        m: int,
        n0: int,
        depth: int,
        pdl: bool,
    ) -> None:
        raw(
            make_ptr(
                cutlass.BFloat16, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.BFloat16, w.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.BFloat16,
                y0.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.BFloat16,
                y1.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Float32,
                partial.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Int32,
                counters.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            int(m),
            int(n0),
            int(depth),
            1 if pdl else 0,
            current_cuda_stream(),
        )

    _KERNEL_CACHE[cache_key] = launch
    return launch


def get_cached_weight_first_gemv(n: int, k: int, nt: int, kt: int):
    """Cache-only lookup (no JIT). Returns the launch function or ``None``."""
    return _KERNEL_CACHE.get((int(n), int(k), int(nt), int(kt)))


def _kernel_applies(x: torch.Tensor, weight: torch.Tensor, nt: int, kt: int) -> bool:
    """Whether ``x`` and ``weight`` take the kernel: both CUDA tensors on one
    device (the raw launch dereferences their pointers on that device's
    stream), a compiled brick, a row count within ``MAX_ROWS`` and the
    BF16 layout contract."""
    if x.dim() != 2 or weight.dim() != 2:
        return False
    m, k = x.shape
    return (
        x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and 1 <= m <= MAX_ROWS
        and (int(nt), int(kt)) in BRICKS
        and k == weight.shape[1]
        and k % kt == 0
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
        and weight.data_ptr() % 16 == 0
    )


def _cublas_split(
    x: torch.Tensor, weight: torch.Tensor, n0: int
) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.nn.functional.linear(x, weight)
    return y[:, :n0].contiguous(), y[:, n0:].contiguous()


@torch.library.custom_op(
    "b12x::weight_first_gemv", mutates_args=("partial", "counters")
)
def weight_first_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    partial: torch.Tensor,
    counters: torch.Tensor,
    n0: int,
    nt: int,
    kt: int,
    depth: int,
    pdl: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(x @ weight[:n0].T, x @ weight[n0:].T)`` for bf16 ``x (m, K)`` and a
    concatenated bf16 ``weight (N, K)``.

    Rows ``1 <= m <= MAX_ROWS`` with a compiled brick run the weight-first
    kernel (``pdl`` sets the programmatic-stream-serialization launch
    attribute; ``depth`` bounds the ``cp.async`` groups in flight per CTA).
    Every other shape, and a shape whose kernel is not compiled while a
    stream is capturing, falls back to cuBLAS inside the op.
    """
    if int(depth) not in STAGE_DEPTHS:
        raise ValueError(
            f"stage depth {depth} is not honored by the kernel; use 0 "
            f"(unbounded) or one of {sorted(STAGE_DEPTHS - {0})}"
        )
    if not _kernel_applies(x, weight, nt, kt):
        return _cublas_split(x, weight, n0)
    if not x.is_contiguous() or x.data_ptr() % 16 != 0:
        x = x.contiguous()
        if x.data_ptr() % 16 != 0:
            return _cublas_split(x, weight, n0)
    m, k = x.shape
    n = weight.shape[0]
    # The compiled callable and the launch take the current stream of the
    # current device; select the weight's device so a call made while
    # another device is current does not submit these pointers to a
    # foreign stream.
    with torch.cuda.device(weight.device):
        launch = get_cached_weight_first_gemv(n, k, nt, kt)
        if launch is None:
            if torch.cuda.is_current_stream_capturing():
                return _cublas_split(x, weight, n0)
            launch = compile_weight_first_gemv(n, k, nt, kt)
        y0 = torch.empty((m, n0), dtype=torch.bfloat16, device=x.device)
        y1 = torch.empty((m, n - n0), dtype=torch.bfloat16, device=x.device)
        launch(
            x, weight, y0, y1 if n > n0 else y0, partial, counters, m, n0, depth, pdl
        )
    return y0, y1


@weight_first_gemv.register_fake
def _weight_first_gemv_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    partial: torch.Tensor,
    counters: torch.Tensor,
    n0: int,
    nt: int,
    kt: int,
    depth: int,
    pdl: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    m = x.shape[0]
    n = weight.shape[0]
    return (
        x.new_empty((m, n0), dtype=torch.bfloat16),
        x.new_empty((m, n - n0), dtype=torch.bfloat16),
    )
