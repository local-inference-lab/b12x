"""Small-N BF16 GEMV for decode-time unquantized projections.

The Qwen3.6 GDN block keeps ``in_proj_ba`` in BF16 (its N is far too small to
quantize), and at BS1 decode cuBLAS routes that ``(m<=16, N<=few hundred) x
(N, K)`` GEMM to a 16x16 WMMA tile kernel that runs ~28 us per call — pure
launch/tile overhead for a job whose real traffic is one ~1 MB read of W.
At 48 GDN layers/step that bf16 tail was ~1.35 ms/step in the serving profile.

This kernel does the obvious thing instead: one CTA per output column ``n``,
128 threads strided over K with 128-bit loads of the W row, each thread
carrying ``m`` f32 accumulators (x rows are L2-resident — they are re-read per
CTA, at most ``N * m * K * 2`` bytes through L2, which is noise at these
shapes). bf16 pairs are unpacked with shift/mask bitcasts (no convert
instructions), accumulated in f32, warp+block reduced, and stored as bf16.
Expected cost is ~2-3 us per call vs the ~28 us cuBLAS pick.

Math note: accumulation is f32 in a (thread-strided, tree-reduced) order, the
same accumulator precision cuBLAS uses for bf16 GEMM but a different reduction
order, so results match torch/cuBLAS to bf16 rounding rather than bitwise.

Compiled per ``(m, n, k)`` like the small-M quantizer; decode sees a handful
of m values (1..SMALL_M_MAX), everything larger must use the cuBLAS path.

This module also registers ``sparkinfer::bf16_gemv_small_n`` so a Dynamo /
CUDA-graph path treats the GEMV (compile-cache lookup + CUTE launch) as a
single opaque node. The fallback for shapes the kernel does not cover
(prefill m > SMALL_M_MAX, odd K, misalignment) lives INSIDE the op, so the
calling graph never branches on data-dependent shapes.
"""
from __future__ import annotations

from typing import Dict, Tuple

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import torch
from cutlass import Int64
from cutlass.cutlass_dsl import Int32, Uint32

from sparkinfer._lib.compiler import KernelCompileSpec
from sparkinfer._lib.compiler import compile as sparkinfer_compile
from sparkinfer._lib.intrinsics import (
    block_reduce,
    get_ptr_as_int64,
    ld_global_v4_u32,
    u32_as_f32,
    warp_reduce,
)
from sparkinfer._lib.runtime_control import raise_if_kernel_resolution_frozen
from sparkinfer._lib.utils import current_cuda_stream

_THREADS = 128

#: Largest M routed to this kernel (decode is 1, MTP verify a handful). The
#: kernel's cost grows linearly in m (each thread re-dots every x row against
#: its W chunk), so it wins big at m<=4 (2.3-3.9 us vs cuBLAS 4-19 us on
#: N=64..256, K=5120, graph-replay timed) but loses by ~2x at m=16 — cap at 8.
SMALL_M_MAX = 8

_KERNEL_CACHE: Dict[Tuple, object] = {}
_PRECOMPILED: set = set()


def _fadd(a, b):
    return a + b


class SmallNGemvKernel:
    """One CTA per output column; ``y[0:m, n] = x[0:m, :] @ w[n, :]``."""

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        w: cute.Tensor,
        y: cute.Tensor,
        stream: cuda.CUstream,
    ):
        n = w.shape[0]
        self.kernel(x, w, y).launch(
            grid=(n, 1, 1),
            block=[_THREADS, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mW: cute.Tensor, mY: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        m = mX.shape[0]  # static (compile key)
        n = mW.shape[0]  # static (compile key)
        k = Int32(mX.shape[1])
        nvec = k // Int32(8)  # 8 bf16 per 128-bit load
        # ID * stride offsets are computed in Int64 (coding guideline) so they
        # cannot wrap before get_ptr_as_int64 widens them into the pointer.
        w_base = Int64(bidx) * Int64(k)

        smem = cutlass.utils.SmemAllocator()
        red_buf = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((1, _THREADS // 32)),
            byte_alignment=16,
        )

        acc = cute.make_rmem_tensor((m,), cutlass.Float32)
        for r in cutlass.range_constexpr(m):
            acc[r] = cutlass.Float32(0.0)

        # Strided K loop: each thread reads its 8-wide chunk of the W row once
        # and dots it against the same chunk of every x row. bf16 -> f32 is a
        # shift (low half) / mask (high half) bitcast, exact by construction.
        i = Int32(tidx)
        while i < nvec:
            col = Int64(i * Int32(8))
            w0, w1, w2, w3 = ld_global_v4_u32(
                get_ptr_as_int64(mW, w_base + col)
            )
            for r in cutlass.range_constexpr(m):
                x0, x1, x2, x3 = ld_global_v4_u32(
                    get_ptr_as_int64(mX, Int64(r) * Int64(k) + col)
                )
                s = acc[r]
                for wv, xv in ((w0, x0), (w1, x1), (w2, x2), (w3, x3)):
                    s = s + u32_as_f32(wv << Uint32(16)) * u32_as_f32(
                        xv << Uint32(16)
                    )
                    s = s + u32_as_f32(wv & Uint32(0xFFFF0000)) * u32_as_f32(
                        xv & Uint32(0xFFFF0000)
                    )
                acc[r] = s
            i += Int32(_THREADS)

        # One reduction per output row; red_buf is reused, so a barrier keeps
        # iteration r+1's lane-0 writes from racing iteration r's reads.
        for r in cutlass.range_constexpr(m):
            v = warp_reduce(acc[r], _fadd)
            total = block_reduce(v, _fadd, red_buf, cutlass.Float32(0.0))
            if tidx == Int32(0):
                # mY is the flat (m*n,) view: a 2D (m, n) tensor degenerates
                # to all-stride-1 at n == 1 and breaks dlpack layout deduction.
                mY[Int32(r * n) + bidx] = cutlass.BFloat16(total)
            cute.arch.barrier()


def compile_bf16_gemv_small_n(m: int, n: int, k: int):
    """Compile the small-N bf16 GEMV for ``(m, n, k)``.

    Returns ``launch(x, w, y)`` where ``x`` is ``(m, k)`` bf16 contiguous,
    ``w`` is ``(n, k)`` bf16 contiguous (row-major weight, as ``F.linear``
    takes it), and ``y`` a preallocated ``(m, n)`` bf16 output.
    """
    assert 1 <= m <= SMALL_M_MAX, f"small-N GEMV requires m<={SMALL_M_MAX}, got {m}"
    assert k % 8 == 0, f"K must be a multiple of 8, got {k}"
    assert n >= 1
    cache_key = (m, n, k)
    cached = _KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (m, k), stride_order=(1, 0), assumed_align=16
    )
    w_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (n, k), stride_order=(1, 0), assumed_align=16
    )
    y_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (m * n,), assumed_align=16
    )
    kernel = SmallNGemvKernel()
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    raw = sparkinfer_compile(
        kernel,
        x_fake,
        w_fake,
        y_fake,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "gemm.bf16_gemv_small_n",
            2,
            cache_key,
        ),
    )

    def launch(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor):
        raw(x, w, y.view(-1), current_cuda_stream())

    _KERNEL_CACHE[cache_key] = launch
    return launch


def get_cached_bf16_gemv_small_n(m: int, n: int, k: int):
    """Cache-only lookup (no JIT). Returns the launch fn or ``None``."""
    return _KERNEL_CACHE.get((m, n, k))


def precompile_bf16_gemv_small_n(weight: torch.Tensor, log=None) -> None:
    """Compile AND warm-run every decode-m variant for ``weight``'s shape.

    Called at weight-load time (``process_weights_after_loading``) so the hot
    path never JITs *or* lazily initializes: serving may be inside CUDA-graph
    capture when a given m first shows up, and both a mid-capture
    cute.compile (ptxas + syncs) and a first-launch lazy module load
    (cuModuleLoad) corrupt the capture — downstream kernels then die with
    CUDA 'invalid resource handle' / segfaults. The warm-run below forces the
    full first-launch path (module load included) on a normal stream now.
    All 48 GDN layers share one (n, k), so this runs once per shape
    (compilation is disk-cached after the first process).
    """
    if log is None:
        import logging

        log = logging.getLogger("sparkinfer.bf16_gemv")
    n, k = int(weight.shape[0]), int(weight.shape[1])
    if (
        k % 8 != 0
        or weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
        or weight.data_ptr() % 16 != 0
    ):
        log.info(
            "bf16 GEMV precompile: weight n=%d k=%d unsupported, skipping", n, k
        )
        return
    if (n, k) in _PRECOMPILED:
        return
    log.info(
        "bf16 GEMV precompile: n=%d k=%d, m=1..%d", n, k, SMALL_M_MAX
    )
    for m in range(1, SMALL_M_MAX + 1):
        log.debug("bf16 GEMV precompile m=%d n=%d k=%d: compiling", m, n, k)
        launch = compile_bf16_gemv_small_n(m, n, k)
        x = torch.zeros(m, k, dtype=torch.bfloat16, device=weight.device)
        y = torch.empty(m, n, dtype=torch.bfloat16, device=weight.device)
        # Warm-run against a DUMMY weight first, then the given one: if the
        # dummy passes and the given weight faults, the tensor's storage
        # (allocator/placement) is the problem, not the kernel. NOTE: the
        # vLLM plugin passes a fresh CLONE of the layer weight here — the
        # loader-placed parameter storage itself faults under our raw
        # launches (observed: dummy passed, loader weight IMA'd).
        w_dummy = torch.zeros_like(weight)
        launch(x, w_dummy, y)
        torch.cuda.synchronize()
        del w_dummy
        launch(x, weight, y)
        torch.cuda.synchronize()
        log.debug("bf16 GEMV precompile m=%d: done", m)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    _PRECOMPILED.add((n, k))


def _use_kernel(x: torch.Tensor, weight: torch.Tensor) -> bool:
    m, k = x.shape
    return (
        1 <= m <= SMALL_M_MAX
        and k % 8 == 0
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
        and weight.data_ptr() % 16 == 0
    )


@torch.library.custom_op("sparkinfer::bf16_gemv_small_n", mutates_args=())
def bf16_gemv_small_n(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``y = x @ weight.T`` for bf16 ``x (m, K)`` / ``weight (N, K)``.

    Routes small decode shapes (``m <= SMALL_M_MAX``) through the CUTE GEMV;
    anything else falls back to ``F.linear`` (cuBLAS) inside the op.
    """
    if not _use_kernel(x, weight):
        return torch.nn.functional.linear(x, weight)
    if not x.is_contiguous() or x.data_ptr() % 16 != 0:
        x = x.contiguous()
        if x.data_ptr() % 16 != 0:
            return torch.nn.functional.linear(x, weight)
    m, k = x.shape
    n = weight.shape[0]
    launch = get_cached_bf16_gemv_small_n(m, n, k)
    if launch is None:
        # Never JIT while a stream is capturing: cute.compile (ptxas +
        # cuModuleLoad + syncs) corrupts an in-flight CUDA-graph capture.
        # Serving precompiles all m at weight load, so this only triggers
        # for shapes precompile never saw — cuBLAS is the safe answer there.
        if torch.cuda.is_current_stream_capturing():
            return torch.nn.functional.linear(x, weight)
        launch = compile_bf16_gemv_small_n(m, n, k)
    y = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    launch(x, weight, y)
    return y


@bf16_gemv_small_n.register_fake
def _bf16_gemv_small_n_fake(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.bfloat16)
