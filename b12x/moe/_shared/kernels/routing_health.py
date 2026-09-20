"""Read canonical counters on their producer stream; never modify route counters."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint64
from cutlass.utils import SmemAllocator


class RoutingHealth:
    @cute.jit
    def __call__(
        self,
        descriptors: cute.Pointer,
        output: cute.Pointer,
        layers: Int32,
        baseline: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(descriptors, output, baseline).launch(
            grid=(layers, 1, 1), block=(128, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, descriptors: cute.Pointer, output: cute.Pointer, baseline: Int32):
        layer, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        offset = Int64(layer) * 4
        counts = cute.make_ptr(
            Uint64, descriptors[offset], cute.AddressSpace.gmem, assumed_align=8
        )
        mapping = cute.make_ptr(
            Int32, descriptors[offset + 1], cute.AddressSpace.gmem, assumed_align=8
        )
        previous = cute.make_ptr(
            Uint64, descriptors[offset + 2], cute.AddressSpace.gmem, assumed_align=8
        )
        experts = descriptors[offset + 3]
        total, cold, once, repeated, repeats, bad = (
            Uint64(0),
            Uint64(0),
            Uint64(0),
            Uint64(0),
            Uint64(0),
            Uint64(0),
        )
        for expert in cutlass.range(Int64(tid), experts, 128):
            now, old = counts[expert], previous[expert]
            tier = mapping[expert * Int64(2)]
            delta = Uint64(0)
            if baseline == 0:
                if now < old:
                    bad = Uint64(1)
                else:
                    delta = Uint64(now - old)
            previous[expert] = now
            if Uint64(total + delta) < total:
                bad = Uint64(1)
            total = Uint64(total + delta)
            if (tier < 0) | (tier > 1):
                bad = Uint64(1)
            if tier == 1:
                cold = Uint64(cold + delta)
                if delta == 1:
                    once = Uint64(once + Uint64(1))
                if delta > 1:
                    repeated = Uint64(repeated + Uint64(1))
                    repeats = Uint64(repeats + delta - Uint64(1))
        scratch = SmemAllocator().allocate_tensor(
            Uint64, cute.make_layout((128, 6)), byte_alignment=8
        )
        scratch[tid, 0], scratch[tid, 1] = total, cold
        scratch[tid, 2], scratch[tid, 3] = once, repeated
        scratch[tid, 4], scratch[tid, 5] = repeats, bad
        cute.arch.sync_threads()
        for shift in cutlass.range_constexpr(7):
            width = 64 >> shift
            if tid < width:
                for column in cutlass.range_constexpr(5):
                    left, right = scratch[tid, column], scratch[tid + width, column]
                    if Uint64(left + right) < left:
                        scratch[tid, 5] = Uint64(1)
                    scratch[tid, column] = left + right
                scratch[tid, 5] += scratch[tid + width, 5]
            cute.arch.sync_threads()
        if tid == 0:
            if counts[experts + 4] != 0:
                scratch[0, 5] += Uint64(1)
            for column in cutlass.range_constexpr(6):
                output[Int64(layer) * 6 + column] = scratch[0, column]
