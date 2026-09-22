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
    def __init__(self, experts, sample_every, runtime_token_limit=False):
        self.experts, self.sample_every = experts, sample_every
        self.runtime_token_limit = runtime_token_limit

    @cute.jit
    def __call__(self, ids: cute.Pointer, counts: cute.Pointer, enabled: cute.Pointer,
                 tokens: Int32, top_k: Int32, stream: cuda.CUstream):
        self.kernel(ids, counts, enabled, tokens, top_k).launch(
            grid=(1, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, ids: cute.Pointer, counts: cute.Pointer, enabled: cute.Pointer,
               tokens: Int32, top_k: Int32):
        tid, _, _ = cute.arch.thread_idx()
        if cutlass.const_expr(self.runtime_token_limit):
            tokens = cutlass.max(Int32(0), cutlass.min(tokens, enabled[1]))
        sampled = SmemAllocator().allocate_tensor(Int32, 1, byte_alignment=4)
        overflow = counts + self.experts + 4
        if tid == 0:
            sampled[0] = 0
            if enabled[0] != 0 and tokens > 0:
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


class SetTokenLimit:
    """Publish an engine-labelled observation extent on the producer stream."""
    @cute.jit
    def __call__(self, enabled: cute.Pointer, tokens: Int32, stream: cuda.CUstream):
        self.kernel(enabled, tokens).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, enabled: cute.Pointer, tokens: Int32):
        enabled[1] = tokens


class SetPhaseRange:
    """Set engine-declared row labels on the producer stream, outside replay."""
    @cute.jit
    def __call__(self, labels: cute.Pointer, begin: Int32, end: Int32, phase: Int32, stream: cuda.CUstream):
        self.kernel(labels, begin, end, phase).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, labels: cute.Pointer, begin: Int32, end: Int32, phase: Int32):
        tid, _, _ = cute.arch.thread_idx()
        for row in cutlass.range(begin + tid, end, 128):
            labels[Int64(row)] = phase


class CountPhasedRoutes:
    """One route addition into either decode or prefill; zero labels are padding."""
    def __init__(self, experts):
        self.experts = experts
        self.stride = (experts + 6) // 2 * 2

    @cute.jit
    def __call__(self, ids: cute.Pointer, counts: cute.Pointer, enabled: cute.Pointer,
                 tokens: Int32, top_k: Int32, stream: cuda.CUstream):
        self.kernel(ids, counts, enabled, tokens, top_k).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, ids: cute.Pointer, counts: cute.Pointer, enabled: cute.Pointer,
               tokens: Int32, top_k: Int32):
        tid, _, _ = cute.arch.thread_idx()
        if enabled[0] != 0:
            if tid == 0:
                decode, prefill = Int64(0), Int64(0)
                for row in cutlass.range(tokens):
                    label = enabled[4 + Int64(row)]
                    decode += Int64(label == 1)
                    prefill += Int64(label == 2)
                for phase in cutlass.range_constexpr(2):
                    amount = decode if phase == 0 else prefill
                    target = counts + Int64(phase * self.stride)
                    if amount > 0:
                        overflow = target + self.experts + 4
                        increment(target + self.experts, overflow, Int64(1))
                        increment(target + self.experts + 1, overflow, Int64(1))
                        increment(target + self.experts + 2, overflow, amount)
                        increment(target + self.experts + 3, overflow, amount)
            for offset in cutlass.range(tid, tokens * top_k, 128):
                label = enabled[4 + Int64(offset // top_k)]
                expert = Int64(ids[Int64(offset)])
                if (label >= 1) & (label <= 2) & (expert >= 0) & (expert < self.experts):
                    target = counts + Int64(label - 1) * Int64(self.stride)
                    increment(target + expert, target + self.experts + 4, Int64(1))
