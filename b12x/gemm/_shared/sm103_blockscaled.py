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


"""Shared SM103 block-scaled GEMM using CUTLASS TMA, UMMA, and TMEM.

Dense tiles cover runtime rows; routed tiles cover one selected expert row.
Both schedules consume F8_128x4 scale storage and retain FP32 accumulators.
The pipeline is adapted from NVIDIA CUTLASS's nvfp4_gemm_0 tutorial.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils

class BlockscaledGemm:
    def __init__(self, n, k, groups, *, recipe, c_dtype, routed_capacity=None, alpha_is_one=False,
                 a_fmt=None, b_fmt=None, a_preexpanded=False, b_preexpanded=False,
                 apply_row_scale=False):
        if recipe not in ("nvfp4", "mxfp4", "mxfp8", "mxfp6", "w4a8_mx"):
            raise ValueError("SM103 blockscaled GEMM requires NVFP4, MXFP4, MXFP6, or MXFP8")
        self.n, self.k = n, k
        self.experts = groups
        self.routed = routed_capacity is not None
        self.capacity = routed_capacity
        self.recipe = recipe
        self.mma_tiler_mn = (128, 128)
        self.mma_inst_shape_k = 32 if recipe in ("mxfp6", "mxfp8", "w4a8_mx") else 64
        self.ab_dtype = cutlass.Float8E4M3FN if recipe == "mxfp8" else cutlass.Float4E2M1FN
        self.a_dtype = self.b_dtype = self.ab_dtype
        if recipe == "w4a8_mx":
            self.a_dtype = cutlass.Float8E4M3FN
        self.pack_a_smem = self.pack_b_smem = False
        if recipe == "mxfp6":
            if routed_capacity is not None:
                raise ValueError("MXFP6 dense GEMM does not use the routed NVFP4 contract")
            types = {"e2m3": cutlass.Float6E2M3FN, "e3m2": cutlass.Float6E3M2FN,
                     "e4m3": cutlass.Float8E4M3FN}
            if a_fmt not in types or b_fmt not in types:
                raise ValueError("MXFP6 operand formats must be e2m3, e3m2, or e4m3")
            self.a_dtype, self.b_dtype = types[a_fmt], types[b_fmt]
            self.pack_a_smem = a_preexpanded and self.a_dtype.width == 6
            self.pack_b_smem = b_preexpanded and self.b_dtype.width == 6
        self.a_gmem_dtype = cutlass.Uint8 if self.pack_a_smem else self.a_dtype
        self.b_gmem_dtype = cutlass.Uint8 if self.pack_b_smem else self.b_dtype
        self.a_smem_dtype = cutlass.Uint8 if self.a_dtype.width == 6 else self.a_dtype
        self.b_smem_dtype = cutlass.Uint8 if self.b_dtype.width == 6 else self.b_dtype
        # kind::mxf8f6f4 reads E2M1 with byte addressing: each sixteen codes
        # occupy their own sixteen-byte slot (eight code bytes, eight ignored).
        # TMA stages packed checkpoint rows; the MMA warp expands them.
        self.unpack_b_fp4 = recipe == "w4a8_mx"
        if self.unpack_b_fp4:
            self.b_smem_dtype = cutlass.Uint8
        self.sf_dtype = cutlass.Float8E4M3FN if recipe == "nvfp4" else cutlass.Float8E8M0FNU
        self.sf_vec_size = 16 if recipe == "nvfp4" else 32
        self.c_dtype = c_dtype
        self.alpha_is_one = alpha_is_one
        self.apply_row_scale = apply_row_scale
        self.threads_per_cta = 128
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_103")
        self.num_tmem_alloc_cols = 512

        self.num_acc_stage = 1
        self.num_ab_stage = 2

    @cute.jit
    def _launch(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        ids,
        alpha: cute.Pointer,
        live_rows: cutlass.Int32,
        a_group_stride: cutlass.Int64,
        c_group_stride: cutlass.Int64,
        alpha_stride: cutlass.Int64,
        stream: cuda.CUstream,
        row_scale=None,
        route_indices=None,
        route_count=None,
    ):
        if cutlass.const_expr(self.routed):
            ids = cute.make_tensor(ids, cute.make_layout(self.capacity))
            m, l = 1, self.capacity
        else:
            m, l = cutlass.Int64(live_rows), self.experts
        alpha = cute.make_tensor(alpha, cute.make_layout(self.experts, stride=alpha_stride))
        if cutlass.const_expr(self.apply_row_scale):
            row_scale = cute.make_tensor(row_scale, cute.make_layout(m))
        self.c_layout = utils.LayoutEnum.ROW_MAJOR
        n, k = self.n, self.k

        # Never tile K beyond the whole reduction: a 256-wide FP4 tile over K=128
        # gives MXFP4 a single scale-factor K atom smaller than its TMA box,
        # which faults with an illegal instruction on SM103.
        mma_inst_tile_k = max(1, min(4, k // self.mma_inst_shape_k))
        self.mma_tiler = (
            self.mma_tiler_mn[0],
            self.mma_tiler_mn[1],
            self.mma_inst_shape_k * mma_inst_tile_k,
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
                stride=(cute.assume(k, 32), 1,
                        cute.assume(a_group_stride, 128 if self.a_gmem_dtype.width == 6 else 128 // self.a_gmem_dtype.width)),
            ),
        )
        b_tensor = cute.make_tensor(
            b_ptr,
            cute.make_layout(
                (n, cute.assume(k, 32), self.experts),
                stride=(cute.assume(k, 32), 1, cutlass.Int64(n) * k),
            ),
        )
        c_tensor = cute.make_tensor(
            c_ptr,
            cute.make_layout(
                (m, n, l),
                stride=(cutlass.Int64(n), 1, c_group_stride),
            ),
        )
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a_tensor.shape, self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)

        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_tensor.shape, self.sf_vec_size
        )
        sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)

        if cutlass.const_expr(self.recipe == "nvfp4"):
            mma_op = tcgen05.MmaMXF4NVF4Op(
                self.sf_dtype, (*self.mma_tiler_mn, self.mma_inst_shape_k),
                tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            )
        elif cutlass.const_expr(self.recipe == "mxfp4"):
            mma_op = tcgen05.MmaMXF4Op(
                (*self.mma_tiler_mn, self.mma_inst_shape_k),
                tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            )
        else:
            mma_op = tcgen05.MmaMXF8F6F4Op(
                self.a_dtype, self.b_dtype,
                (*self.mma_tiler_mn, self.mma_inst_shape_k),
                tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
                tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
            )
        tiled_mma = cute.make_tiled_mma(mma_op)

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)),
            (tiled_mma.thr_id.shape,),
        )

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_smem_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_smem_dtype,
            self.num_ab_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
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
            internal_type=self.a_smem_dtype if cutlass.const_expr(self.a_dtype.width == 6) else None,
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        if cutlass.const_expr(self.unpack_b_fp4):
            self._check_unpacked_b_layout()
            packed_row_bytes = self.mma_tiler[2] // 2
            b_bytes = cute.make_tensor(
                cute.recast_ptr(b_ptr, dtype=cutlass.Uint8),
                cute.make_layout(
                    (n, cute.assume(k // 2, 16), self.experts),
                    stride=(cute.assume(k // 2, 16), 1, cutlass.Int64(n) * (k // 2)),
                ),
            )
            tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                b_bytes,
                cute.make_layout((self.mma_tiler[1], packed_row_bytes), stride=(packed_row_bytes, 1)),
                (self.mma_tiler[1], packed_row_bytes),
            )
        else:
            tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
                cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
                b_tensor,
                b_smem_layout,
                self.mma_tiler,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
                internal_type=self.b_smem_dtype if cutlass.const_expr(self.b_dtype.width == 6) else None,
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

        # TMA completion counts the transferred global bytes, excluding the
        # padding it appends to packed FP6 groups in shared memory.
        a_copy_size = cute.size(a_smem_layout) * self.a_gmem_dtype.width // 8
        b_copy_size = cute.size(b_smem_layout) * self.b_gmem_dtype.width // 8
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        grid = (
            live_rows if cutlass.const_expr(self.routed) else
            cute.ceil_div(c_tensor.shape[0], self.cta_tile_shape_mnk[0]),
            cute.ceil_div(c_tensor.shape[1], self.cta_tile_shape_mnk[1]),
            1 if cutlass.const_expr(self.routed) else self.experts,
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
            row_scale,
            route_indices,
            route_count,
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
        ids,
        alpha: cute.Tensor,
        row_scale,
        route_indices,
        route_count,
    ):
        """
        GPU device kernel performing the batched GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        tidx, _, _ = cute.arch.thread_idx()

        bidx, bidy, bidz = cute.arch.block_idx()
        active = cutlass.Boolean(True)
        if cutlass.const_expr(route_count is not None):
            active = bidx < route_count[0]
        if active:
            if cutlass.const_expr(self.routed):
                # Routed matrices have one M tile; X carries the runtime route count.
                bidz = bidx
                bidx = cutlass.Int32(0)

            valid = cutlass.Boolean(True)
            safe_expert = bidz
            if cutlass.const_expr(self.routed):
                route_id = cutlass.Int64(bidz)
                expert = cutlass.Int64(ids[route_id])
                if cutlass.const_expr(route_indices is not None):
                    bidz = route_indices[route_id]
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
                element_type=self.a_smem_dtype,
                layout=a_smem_layout_staged.outer,
                byte_alignment=128,
                swizzle=a_smem_layout_staged.inner,
            )
            sB = smem.allocate_tensor(
                element_type=self.b_smem_dtype,
                layout=b_smem_layout_staged.outer,
                byte_alignment=1024 if self.unpack_b_fp4 else 128,
                swizzle=b_smem_layout_staged.inner,
            )
            if cutlass.const_expr(self.unpack_b_fp4):
                packed_row_bytes = self.mma_tiler[2] // 2
                sBp = smem.allocate_tensor(
                    element_type=cutlass.Uint8,
                    layout=cute.make_layout(
                        (self.mma_tiler[1], packed_row_bytes, self.num_ab_stage),
                        stride=(packed_row_bytes, 1, self.mma_tiler[1] * packed_row_bytes),
                    ),
                    byte_alignment=128,
                )
            sSFA = smem.allocate_tensor(
                element_type=self.sf_dtype,
                layout=sfa_smem_layout_staged,
                byte_alignment=128,
            )
            sSFB = smem.allocate_tensor(
                element_type=self.sf_dtype,
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
            if cutlass.const_expr(self.unpack_b_fp4):
                gB_nkl = cute.local_tile(
                    mB_nkl, (self.mma_tiler[1], self.mma_tiler[2] // 2), (None, None, None)
                )
            else:
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
            if cutlass.const_expr(self.unpack_b_fp4):
                tBsB, tBgB = cpasync.tma_partition(
                    tma_atom_b,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sBp, 0, 2),
                    cute.group_modes(gB_nkl, 0, 2),
                )
            else:
                tBsB, tBgB = cpasync.tma_partition(
                    tma_atom_b,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sB, 0, 3),
                    cute.group_modes(thr_mma.partition_B(gB_nkl), 0, 3),
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
                dtype=self.sf_dtype,
            )
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr
                + tcgen05.find_tmem_tensor_col_offset(tCtAcc)
                + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
                dtype=self.sf_dtype,
            )
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            copy_atom_s2t = cute.make_copy_atom(
                tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE),
                self.sf_dtype,
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

                    if cutlass.const_expr(self.pack_a_smem):
                        self._pack_fp6_smem(sA[(None, None, None, ab_full.index)], self.mma_tiler[0] * self.mma_tiler[2])
                    if cutlass.const_expr(self.pack_b_smem):
                        self._pack_fp6_smem(sB[(None, None, None, ab_full.index)], self.mma_tiler[1] * self.mma_tiler[2])
                    if cutlass.const_expr(self.pack_a_smem or self.pack_b_smem):
                        cute.arch.sync_warp()
                        cute.arch.fence_proxy("async.shared", space="cta")
                    if cutlass.const_expr(self.unpack_b_fp4):
                        self._unpack_fp4_smem(
                            sBp[(None, None, ab_full.index)],
                            sB[(None, None, None, ab_full.index)],
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        cute.arch.sync_warp()

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
            cute.arch.fence_view_async_tmem_load()
            identity = cute.make_identity_tensor(self.mma_tiler[:2])
            coords = thr_copy_t2r.partition_D(thr_mma.partition_C(identity))
            for idx in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                row, col = coords[idx]
                global_row = cutlass.Int64(bidx) * self.mma_tiler[0] + row
                global_col = cutlass.Int64(bidy) * self.mma_tiler[1] + col
                if (global_row < mC_mnl.shape[0]) & (global_col < self.n):
                    value = cutlass.Float32(0.0)
                    if valid:
                        value = tTR_rAcc[idx]
                        if cutlass.const_expr(not self.alpha_is_one):
                            value = value * alpha[safe_expert]
                        if cutlass.const_expr(self.apply_row_scale):
                            value = value.to(self.c_dtype).to(cutlass.Float32) * row_scale[global_row].to(cutlass.Float32)
                    tTR_gC[idx] = value.to(self.c_dtype)

            acc_full.release()

            cute.arch.barrier()
            tmem.free(acc_tmem_ptr)

    def _check_unpacked_b_layout(self):
        """Pin the byte-addressed operand geometry that ``_unpack_fp4_smem`` writes.

        Each row holds one 128-byte SW128 atom row: code k of row r lives at byte
        r*128 + k before the swizzle, stages are 16 KiB apart, and the swizzle
        permutes sixteen-byte slots by r % 8.
        """
        outer, inner = self.b_smem_layout_staged.outer, self.b_smem_layout_staged.inner
        rows, k_tile, inst_k = self.mma_tiler[1], self.mma_tiler[2], self.mma_inst_shape_k
        if k_tile != 128 or str(inner) != "S<3,4,3>":
            raise ValueError(f"unexpected E2M1 operand layout {inner} o {outer}")
        for row, k, stage in ((0, 0, 0), (1, 17, 0), (5, 70, 1), (rows - 1, k_tile - 1, 1)):
            if outer(((row, k % inst_k), 0, k // inst_k, stage)) != stage * rows * k_tile + row * k_tile + k:
                raise ValueError(f"unexpected E2M1 operand layout {inner} o {outer}")

    @cute.jit
    def _unpack_fp4_smem(self, packed: cute.Tensor, tile: cute.Tensor):
        """Copy each packed group of sixteen E2M1 codes into its UMMA slot.

        ``packed`` holds one stage of checkpoint-native rows, two codes per byte.
        ``tile`` is the matching stage of the byte-addressed SW128 operand: group g
        of row r occupies sixteen-byte slot g ^ (r % 8) of the row's 128 bytes,
        and the MMA ignores the slot's trailing eight bytes. The buffer is
        1024-byte aligned, so the swizzle is relative to the stage. Each lane moves
        whole eight-byte groups; a warp owns disjoint groups.
        """
        lane = cute.arch.lane_idx()
        rows, groups = self.mma_tiler[1], self.mma_tiler[2] // 16
        source = cute.make_tensor(
            cute.recast_ptr(packed.iterator, dtype=cutlass.Uint64), cute.make_layout(rows * groups)
        )
        target = cute.make_tensor(
            cute.recast_ptr(tile.iterator, dtype=cutlass.Uint64), cute.make_layout(rows * groups * 2)
        )
        values = cute.make_rmem_tensor(rows * groups // 32, cutlass.Uint64)
        for i in cutlass.range_constexpr(rows * groups // 32):
            values[i] = source[lane + 32 * i]
        for i in cutlass.range_constexpr(rows * groups // 32):
            item = lane + 32 * i
            row = item // groups
            slot = (item % groups) ^ (row % 8)
            target[row * groups * 2 + slot * 2] = values[i]

    @cute.jit
    def _pack_fp6_smem(self, tile: cute.Tensor, byte_count: cutlass.Constexpr):
        """Pack sixteen byte codes into twelve bytes plus four ignored bytes.

        A warp owns disjoint aligned sixteen-byte groups. Every code is loaded
        before its group is overwritten; the caller publishes stores to UMMA.
        """
        lane = cute.arch.lane_idx()
        storage = cute.make_tensor(tile.iterator, cute.make_layout(byte_count))
        for group in cutlass.range(lane, byte_count // 16, 32):
            codes = cute.make_rmem_tensor(16, cutlass.Uint32)
            for i in cutlass.range_constexpr(16):
                codes[i] = cutlass.Uint32(storage[group * 16 + i]) & 63
            for i in cutlass.range_constexpr(4):
                word = codes[4*i] | (codes[4*i+1] << 6) | (codes[4*i+2] << 12) | (codes[4*i+3] << 18)
                for j in cutlass.range_constexpr(3):
                    storage[group * 16 + 3*i + j] = cutlass.Uint8(word >> (8*j))
