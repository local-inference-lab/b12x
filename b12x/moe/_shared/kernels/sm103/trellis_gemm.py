"""SM103 routed FP16 projection with inline t256 weight reconstruction.

Each CTA computes one route and 128 output columns. Native compressed weights
are decoded into a 128x64 shared-memory tile; no decoded weight matrix is
written to global memory. Input and output are in the quantizer basis. Expert
rotations, activation, mixed-rate selection, and weighted reduction belong to
the enclosing MoE execution plan.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import Float16, Float32, Int32, Int64
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05

from .trellis_decode import decode_lane, decoder_contract, tile_coordinates


class RoutedTrellisGemm:
    """Project FP16 [routes,K] through compressed [E,K/16,N/16,16*bits].

    The payload uses Int16 storage, passed as a Uint32 pointer. Route IDs may
    be Int32 or Int64; invalid IDs produce zero without reading input or weight
    rows. Row strides and live route counts are runtime launch arguments.
    """

    def __init__(
        self,
        n: int,
        k: int,
        experts: int,
        capacity: int,
        *,
        bits: int,
        codebook: str,
        dual_input: bool = False,
    ):
        self.codebook = decoder_contract(bits, codebook)
        if min(n, k, experts, capacity) <= 0 or n % 16 or k % 16:
            raise ValueError(
                "Trellis projection requires positive geometry and K/N divisible by 16"
            )
        if capacity * ((n + 127) // 128) > 2**31 - 1:
            raise ValueError(
                "Trellis projection exceeds the one-dimensional CUDA grid limit"
            )
        self.n, self.k, self.experts, self.capacity = n, k, experts, capacity
        self.bits = bits
        self.dual_input = dual_input

    @cute.jit
    def __call__(
        self,
        a,
        packed: cute.Pointer,
        lut: cute.Pointer,
        ids: cute.Pointer,
        out: cute.Pointer,
        live_routes: Int32,
        a_stride: Int64,
        out_stride: Int64,
        stream: cuda.CUstream,
    ):
        mma = cute.make_tiled_mma(
            tcgen05.MmaF16BF16Op(
                Float16,
                Float32,
                (128, 128, 16),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
                OperandMajorMode.K,
                OperandMajorMode.K,
            )
        )
        atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW128, Float16
        )
        layout = cute.tile_to_shape(atom, (128, 64), order=(0, 1))
        self.kernel(mma, layout, a, packed, lut, ids, out, a_stride, out_stride).launch(
            grid=(live_routes * cute.ceil_div(self.n, 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self, mma, layout, a, packed, lut, ids, out, a_stride: Int64, out_stride: Int64
    ):
        block, _, _ = cute.arch.block_idx()
        route = Int64(block // cute.ceil_div(self.n, 128))
        n_base = Int64(block % cute.ceil_div(self.n, 128)) * 128
        route_ids = cute.make_tensor(ids, cute.make_layout(self.capacity))
        expert = Int64(route_ids[route])
        source = self.input_tensors(a, a_stride)
        weights = cute.make_tensor(
            packed,
            cute.make_layout(
                self.experts * (self.k // 16) * (self.n // 16) * (8 * self.bits)
            ),
        )
        output = cute.make_tensor(
            out, cute.make_layout(Int64(self.capacity) * out_stride)
        )

        self.execute_tiles(
            mma,
            layout,
            source,
            weights,
            lut,
            expert,
            route,
            a_stride,
            output,
            n_base,
            out_stride,
        )

    @cute.jit
    def execute_tiles(
        self,
        mma,
        layout,
        source,
        weights,
        lut,
        selection,
        route: Int64,
        a_stride: Int64,
        output,
        n_base: Int64,
        out_stride: Int64,
    ):
        """Share TMEM lifetime, MMA completion, and the FP16 epilogue across decoders."""
        thread, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        @cute.struct
        class Storage:
            completion: Int64
            tmem_address: Int32

        allocator = utils.SmemAllocator()
        storage = allocator.allocate(Storage)
        sA = allocator.allocate_tensor(Float16, layout.outer, 128, swizzle=layout.inner)
        sB = allocator.allocate_tensor(Float16, layout.outer, 128, swizzle=layout.inner)
        if thread == 0:
            cute.arch.mbarrier_init(storage.completion.ptr, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        thr_mma = mma.get_slice(0)
        frag_a = mma.make_fragment_A(thr_mma.partition_A(sA))
        frag_b = mma.make_fragment_B(thr_mma.partition_B(sB))
        acc_layout = mma.make_fragment_C(mma.partition_shape_C((128, 128))).layout
        tmem = utils.TmemAllocator(
            storage.tmem_address.ptr,
            barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=128),
        )
        tmem.allocate(128)
        tmem.wait_for_alloc()
        acc_ptr = tmem.retrieve_ptr(Float32)
        acc = cute.make_tensor(acc_ptr, acc_layout)
        tmem.relinquish_alloc_permit()

        for stage in cutlass.range(cute.ceil_div(self.k, 64)):
            self.stage_operands(
                source, weights, lut, selection, route, a_stride, stage, n_base, sA, sB
            )

            # Every writer publishes its stores to the async MMA proxy before
            # warp zero issues MMA. Completion precedes reuse of either tile.
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier()
            if warp == 0:
                for kk in cutlass.range_constexpr(4):
                    mma.set(
                        tcgen05.Field.ACCUMULATE,
                        (stage != 0) | cutlass.Boolean(kk != 0),
                    )
                    cute.gemm(
                        mma, acc, frag_a[None, None, kk], frag_b[None, None, kk], acc
                    )
                with cute.arch.elect_one():
                    tcgen05.commit(storage.completion.ptr)
            cute.arch.mbarrier_wait(storage.completion.ptr, phase=stage & 1)
            cute.arch.barrier()

        copy = tcgen05.make_tmem_copy(
            cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition.x128),
                Float32,
            ),
            acc,
        )
        thr_copy = copy.get_slice(thread)
        acc_source = thr_copy.partition_S(acc)
        coordinates = thr_copy.partition_D(
            thr_mma.partition_C(cute.make_identity_tensor((128, 128)))
        )
        split = self.output_split(source, n_base)
        values = cute.make_rmem_tensor(coordinates.shape, Float32)
        cute.copy(copy, acc_source, values)
        cute.arch.fence_view_async_tmem_load()
        for item in cutlass.range_constexpr(cute.size(values)):
            row, col = coordinates[item]
            if self.output_row(split, row, Int32(col)) & (
                n_base + col < self.n
            ):
                output[route * out_stride + n_base + Int64(col)] = values[item].to(
                    Float16
                )
        cute.arch.barrier()
        tmem.free(acc_ptr)

    @cute.jit
    def stage_operands(
        self,
        source,
        weights: cute.Tensor,
        lut: cute.Pointer,
        expert: Int64,
        route: Int64,
        a_stride: Int64,
        stage: Int32,
        n_base: Int64,
        sA: cute.Tensor,
        sB: cute.Tensor,
    ):
        """Fill both operand tiles, including every padded and invalid element."""
        thread, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = Int32(thread % 32)
        valid = (expert >= 0) & (expert < self.experts)
        self.stage_inputs(source, valid, route, a_stride, stage, sA)
        # A warp reconstructs a native 16x16 tile with its original lane map.
        for tile in cutlass.range(warp, 32, 4):
            local_k, local_n = tile // 8, tile % 8
            global_k = Int64(stage) * 4 + Int64(local_k)
            global_n = n_base // 16 + Int64(local_n)
            decoded = cute.make_rmem_tensor(8, Float16)
            decoded.fill(Float16(0))
            if valid & (global_k < self.k // 16) & (global_n < self.n // 16):
                tile_id = (expert * Int64(self.k // 16) + global_k) * Int64(
                    self.n // 16
                ) + global_n
                values = decode_lane(
                    weights, tile_id, lane, lut, self.bits, self.codebook
                )
                decoded.store(values.load())
            for j in cutlass.range_constexpr(8):
                nn, kk = tile_coordinates(lane, j)
                sB[local_n * 16 + nn, local_k * 16 + kk] = decoded[j]

    @cute.jit
    def input_tensors(self, a, stride: Int64):
        layout = cute.make_layout(Int64(self.capacity) * stride)
        if cutlass.const_expr(self.dual_input):
            source = (
                cute.make_tensor(a[0], layout),
                cute.make_tensor(a[1], layout),
                a[2],
            )
        else:
            source = cute.make_tensor(a, layout)
        return source

    @cute.jit
    def stage_inputs(
        self, source, valid, route: Int64, stride: Int64, stage: Int32, sA
    ):
        thread, _, _ = cute.arch.thread_idx()
        for item in cutlass.range(thread, 128 * 64, 128):
            row, col = item // 64, item % 64
            value = Float16(0)
            kk = Int64(stage) * 64 + Int64(col)
            if cutlass.const_expr(self.dual_input):
                if valid & (kk < self.k):
                    if row == 0:
                        value = source[0][route * stride + kk]
                    elif row == 1:
                        value = source[1][route * stride + kk]
            else:
                if valid & (row == 0) & (kk < self.k):
                    value = source[route * stride + kk]
            sA[row, col] = value

    @cute.jit
    def output_split(self, source, n_base: Int64):
        if cutlass.const_expr(self.dual_input):
            # Subtract in Int64 before narrowing the bounded tile coordinate.
            # One CTA-local cutoff avoids a 64-bit comparison for every value
            # while the TMEM accumulator fragment is live in registers.
            relative = source[2] - n_base
            split = Int32(cute.min(cute.max(relative, Int64(0)), Int64(128)))
        else:
            split = Int32(0)
        return split

    @cute.jit
    def output_row(self, split: Int32, row, column: Int32):
        # Both physical FC1 slots select the same input half per output column.
        if cutlass.const_expr(self.dual_input):
            selected = ((row == 0) & (column < split)) | (
                (row == 1) & (column >= split)
            )
        else:
            selected = row == 0
        return selected
