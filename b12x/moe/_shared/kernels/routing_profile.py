"""Prepared device counters adjacent to routing; no activation or weight math."""
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.utils import SmemAllocator

from b12x._lib.intrinsics import atomic_add_global_u64


@dsl_user_op
def _mark_overflow(address, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int64(address).ir_value(loc=loc, ip=ip)],
        "red.relaxed.gpu.global.or.b64 [$0], 1;", "l", has_side_effects=True,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@cute.jit
def increment(pointer: cute.Pointer, overflow: cute.Pointer, value: Int64):
    old = Uint64(atomic_add_global_u64(pointer.toint(), value))
    if old + Uint64(value) < old:
        _mark_overflow(overflow.toint())
    return old


class CountRoutes:
    """One CTA samples complete invocations, including duplicate selections.

    Counter additions wrap modulo 2**64 and set a sticky overflow bit. Host
    snapshots reject that epoch. Atomic updates permit concurrent producer
    streams; snapshot/control operations require quiescent producers.
    """
    def __init__(self, experts, sample_every):
        self.experts, self.sample_every = experts, sample_every

    @cute.jit
    def __call__(self, ids: cute.Pointer, counts: cute.Pointer, enabled: cute.Pointer,
                 tokens: Int32, top_k: Int32, stream: cuda.CUstream):
        self.kernel(ids, counts, enabled, tokens, top_k).launch(
            grid=(1, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, ids: cute.Pointer, counts: cute.Pointer, enabled: cute.Pointer,
               tokens: Int32, top_k: Int32):
        tid, _, _ = cute.arch.thread_idx()
        sampled = SmemAllocator().allocate_tensor(Int32, 1, byte_alignment=4)
        overflow = counts + self.experts + 4
        if tid == 0:
            sampled[0] = 0
            if enabled[0] != 0:
                call = increment(counts + self.experts, overflow, Int64(1))
                increment(counts + self.experts + 2, overflow, Int64(tokens))
                if call % Uint64(self.sample_every) == Uint64(0):
                    sampled[0] = 1
                    increment(counts + self.experts + 1, overflow, Int64(1))
                    increment(counts + self.experts + 3, overflow, Int64(tokens))
        cute.arch.sync_threads()
        if sampled[0] != 0:
            for offset in cutlass.range(tid, tokens*top_k, 128):
                expert = Int64(ids[Int64(offset)])
                if (expert >= 0) & (expert < self.experts):
                    increment(counts + expert, overflow, Int64(1))
