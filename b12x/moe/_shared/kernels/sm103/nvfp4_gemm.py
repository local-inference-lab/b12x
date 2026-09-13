# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""Routed native NVFP4 projection using CUTLASS TMA/UMMA/TMEM primitives.

Adapted from NVIDIA CUTLASS's nvfp4_gemm_0 tutorial. A CTA computes one
selected expert row; TMA zero-fills the unused M lanes. This deliberately
simple schedule is unqualified on B300 and makes no tiny-M speed claim.
The activation descriptor covers planned capacity; live routes only size the
launch grid. Expert tensors retain b12x's source-native packed representation.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils

mma_tiler_mn = (128, 128)
mma_inst_shape_k = 64
ab_dtype = cutlass.Float4E2M1FN
sf_dtype = cutlass.Float8E4M3FN
c_dtype = cutlass.BFloat16
sf_vec_size = 16


class RoutedNvfp4Gemm:
    def __init__(self, n: int, k: int, experts: int, capacity: int):
        self.n, self.k = n, k
        self.experts, self.capacity = experts, capacity
        self.threads_per_cta = 128
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_103")
        self.num_tmem_alloc_cols = 512

        self.num_acc_stage = 1
        self.num_ab_stage = 2

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        ids: cute.Pointer,
        alpha: cute.Pointer,
        live_routes: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        ids = cute.make_tensor(ids, cute.make_layout(self.capacity))
        alpha = cute.make_tensor(alpha, cute.make_layout(self.experts))
        self.c_layout = utils.LayoutEnum.ROW_MAJOR
        m, n, k, l = 1, self.n, self.k, self.capacity

        mma_inst_tile_k = 4
        self.mma_tiler = (
            mma_tiler_mn[0],
            mma_tiler_mn[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        a_tensor = cute.make_tensor(
            a_ptr,
            cute.make_layout(
                (m, cute.assume(k, 32), l),
                stride=(cute.assume(k, 32), 1, cute.assume(m * k, 32)),
            ),
        )
        b_tensor = cute.make_tensor(
            b_ptr,
            cute.make_layout(
                (n, cute.assume(k, 32), self.experts),
                stride=(cute.assume(k, 32), 1, cute.assume(n * k, 32)),
            ),
        )
        c_tensor = cute.make_tensor(
            c_ptr,
            cute.make_layout(
                (m, n, l),
                stride=(cute.assume(n, 16), 1, n),
            ),
        )
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a_tensor.shape, sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)

        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_tensor.shape, sf_vec_size
        )
        sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)

        mma_op = tcgen05.MmaMXF4NVF4Op(
            sf_dtype,
            (*mma_tiler_mn, mma_inst_shape_k),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
        )
        tiled_mma = cute.make_tiled_mma(mma_op)

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)),
            (tiled_mma.thr_id.shape,),
        )

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            ab_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            ab_dtype,
            self.num_ab_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            sf_vec_size,
            self.num_ab_stage,
        )

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            a_tensor,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            b_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            sfa_tensor,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            sfb_tensor,
            sfb_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        a_copy_size = cute.size_in_bytes(ab_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(ab_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        grid = (
            cute.ceil_div(c_tensor.shape[0], self.cta_tile_shape_mnk[0]),
            cute.ceil_div(c_tensor.shape[1], self.cta_tile_shape_mnk[1]),
            live_routes,
        )

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            c_tensor,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            ids,
            alpha,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        mC_mnl: cute.Tensor,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        ids: cute.Tensor,
        alpha: cute.Tensor,
    ):
        """
        GPU device kernel performing the batched GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        tidx, _, _ = cute.arch.thread_idx()

        bidx, bidy, bidz = cute.arch.block_idx()

        route_id = cutlass.Int64(bidz)
        expert = cutlass.Int64(ids[route_id])
        valid = (expert >= 0) & (expert < self.experts)
        safe_expert = cutlass.Int32(0)
        if valid:
            safe_expert = cutlass.Int32(expert)
        cta_coord = (bidx, bidy, bidz)
        mma_tile_coord_mnl = (
            cta_coord[0] // cute.size(tiled_mma.thr_id.shape),
            cta_coord[1],
            cta_coord[2],
        )

        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sA = smem.allocate_tensor(
            element_type=ab_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=ab_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sSFA = smem.allocate_tensor(
            element_type=sf_dtype,
            layout=sfa_smem_layout_staged,
            byte_alignment=128,
        )
        sSFB = smem.allocate_tensor(
            element_type=sf_dtype,
            layout=sfb_smem_layout_staged,
            byte_alignment=128,
        )

        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
        ).make_participants()
        acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_cta,
            ),
        ).make_participants()

        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        thr_mma = tiled_mma.get_slice(0)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        tCgSFB = thr_mma.partition_B(gSFB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            0,
            cute.make_layout(1),
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb,
            0,
            cute.make_layout(1),
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
        )
        tmem.allocate(self.num_tmem_alloc_cols)
        tmem.wait_for_alloc()
        acc_tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
        tCtAcc = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

        sfa_tmem_ptr = cute.recast_ptr(
            acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc),
            dtype=sf_dtype,
        )
        tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            sf_vec_size,
            cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
        )
        tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
        sfb_tmem_ptr = cute.recast_ptr(
            acc_tmem_ptr
            + tcgen05.find_tmem_tensor_col_offset(tCtAcc)
            + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
            dtype=sf_dtype,
        )
        tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            sf_vec_size,
            cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
        )
        tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE),
            sf_dtype,
        )
        tCsSFA_compact = cute.filter_zeros(sSFA)
        tCtSFA_compact = cute.filter_zeros(tCtSFA)
        tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFA_compact)
        thr_copy_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
        tCsSFA_compact_s2t_ = thr_copy_s2t_sfa.partition_S(tCsSFA_compact)
        tCsSFA_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t_sfa, tCsSFA_compact_s2t_
        )
        tCtSFA_compact_s2t = thr_copy_s2t_sfa.partition_D(tCtSFA_compact)

        tCsSFB_compact = cute.filter_zeros(sSFB)
        tCtSFB_compact = cute.filter_zeros(tCtSFB)
        tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFB_compact)
        thr_copy_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
        tCsSFB_compact_s2t_ = thr_copy_s2t_sfb.partition_S(tCsSFB_compact)
        tCsSFB_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t_sfb, tCsSFB_compact_s2t_
        )
        tCtSFB_compact_s2t = thr_copy_s2t_sfb.partition_D(tCtSFB_compact)

        tAgA = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
        tBgB = tBgB[(None, mma_tile_coord_mnl[1], None, safe_expert)]
        tAgSFA = tAgSFA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
        tBgSFB = tBgSFB[(None, mma_tile_coord_mnl[1], None, safe_expert)]

        if warp_idx == 0:
            acc_empty = acc_producer.acquire_and_advance()
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for _k_tile in cutlass.range(
                k_tile_cnt, prefetch_stages=self.num_ab_stage - 2
            ):
                ab_empty = ab_producer.acquire_and_advance()

                cute.copy(
                    tma_atom_a,
                    tAgA[(None, ab_empty.count)],
                    tAsA[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, ab_empty.count)],
                    tBsB[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                )
                cute.copy(
                    tma_atom_sfa,
                    tAgSFA[(None, ab_empty.count)],
                    tAsSFA[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                )
                cute.copy(
                    tma_atom_sfb,
                    tBgSFB[(None, ab_empty.count)],
                    tBsSFB[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                )

                ab_full = ab_consumer.wait_and_advance()

                s2t_stage_coord = (None, None, None, None, ab_full.index)
                tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
                cute.copy(
                    tiled_copy_s2t_sfa,
                    tCsSFA_compact_s2t_staged,
                    tCtSFA_compact_s2t,
                )
                cute.copy(
                    tiled_copy_s2t_sfb,
                    tCsSFB_compact_s2t_staged,
                    tCtSFB_compact_s2t,
                )

                num_kblocks = cute.size(tCrA, mode=[2])
                for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                    kblock_coord = (
                        None,
                        None,
                        kblock_idx,
                        ab_full.index,
                    )

                    sf_kblock_coord = (None, None, kblock_idx)
                    tiled_mma.set(
                        tcgen05.Field.SFA,
                        tCtSFA[sf_kblock_coord].iterator,
                    )
                    tiled_mma.set(
                        tcgen05.Field.SFB,
                        tCtSFB[sf_kblock_coord].iterator,
                    )

                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[kblock_coord],
                        tCrB[kblock_coord],
                        tCtAcc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                ab_full.release()
            acc_empty.commit()

        op = tcgen05.Ld32x32bOp(tcgen05.Repetition.x128, tcgen05.Pack.NONE)
        copy_atom_t2r = cute.make_copy_atom(op, cutlass.Float32)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc)
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc)
        tTR_gC = thr_copy_t2r.partition_D(tCgC)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[None, None, None, None, 0, 0, 0].shape, cutlass.Float32
        )
        tTR_gC = tTR_gC[(None, None, None, None, *mma_tile_coord_mnl)]

        tmem.relinquish_alloc_permit()

        acc_full = acc_consumer.wait_and_advance()

        cute.copy(tiled_copy_t2r, tTR_tAcc, tTR_rAcc)
        identity = cute.make_identity_tensor(self.mma_tiler[:2])
        coords = thr_copy_t2r.partition_D(thr_mma.partition_C(identity))
        for idx in cutlass.range_constexpr(cute.size(tTR_rAcc)):
            row, col = coords[idx]
            if row == 0:
                value = cutlass.Float32(0.0)
                if valid:
                    value = tTR_rAcc[idx] * alpha[safe_expert]
                tTR_gC[idx] = value.to(c_dtype)

        acc_full.release()

        cute.arch.barrier()
        tmem.free(acc_tmem_ptr)

        return
