"""Deterministic FP32 top-k reducer plus final normalized H512 for P8.

This replaces the ordinary B12X top-k reducer; it is not an extra GEMM and it
does not decode weights. FC2 has already emitted ``H128(FP16(MMA))*down_svh``
per route. One CTA owns a full H512 residual block, performs router-order FP32
summation, then the final outer H512 and one BF16 output store.
"""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64

from b12x.moe._shared.kernels.w4a8_trellis_decode import _w4a8_had128_quad


class P8CoupledTopKSumKernel:
    threads_per_cta = 128

    def __init__(self, *, topk: int = 8, hidden: int = 4096):
        if topk != 8 or hidden != 4096 or hidden % 512:
            raise ValueError("P8 coupled reducer requires topk=8 and H=4096")
        self.topk = int(topk)
        self.hidden = int(hidden)

    @cute.jit
    def __call__(
        self,
        route_ptr: cute.Pointer,
        weight_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        active_m: Int32,
        stream: cuda.CUstream,
    ):
        routes = cute.make_tensor(
            route_ptr,
            cute.make_layout((Int64(active_m) * Int64(self.topk * self.hidden),), stride=(1,)),
        )
        weights = cute.make_tensor(
            weight_ptr,
            cute.make_layout((Int64(active_m) * Int64(self.topk),), stride=(1,)),
        )
        output = cute.make_tensor(
            output_ptr,
            cute.make_layout((Int64(active_m) * Int64(self.hidden),), stride=(1,)),
        )
        self.kernel(routes, weights, output, active_m).launch(
            grid=(active_m * Int32(self.hidden // 512), 1, 1),
            block=(self.threads_per_cta, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        routes: cute.Tensor,
        weights: cute.Tensor,
        output: cute.Tensor,
        active_m: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        lane = tid & Int32(31)
        warp = tid >> Int32(5)
        unit = Int32(bidx)
        blocks_per_token = Int32(self.hidden // 512)
        token = unit // blocks_per_token
        block = unit - token * blocks_per_token
        if token < active_m:
            # Four warps own four independent H128 quarters. FC2 already
            # applied the route-local H128 and shared down_svh; linearity lets
            # us sum those FP32 values before the final outer H512.
            col = block * Int32(512) + warp * Int32(128) + lane * Int32(4)
            a0 = cutlass.Float32(0.0)
            a1 = cutlass.Float32(0.0)
            a2 = cutlass.Float32(0.0)
            a3 = cutlass.Float32(0.0)
            for route in cutlass.range_constexpr(self.topk):
                row = Int64(token) * Int64(self.topk) + Int64(route)
                weight = weights[row].to(cutlass.Float32)
                base = row * Int64(self.hidden) + Int64(col)
                a0 += weight * routes[base].to(cutlass.Float32)
                a1 += weight * routes[base + Int32(1)].to(cutlass.Float32)
                a2 += weight * routes[base + Int32(2)].to(cutlass.Float32)
                a3 += weight * routes[base + Int32(3)].to(cutlass.Float32)

            reduced_ptr = cute.arch.alloc_smem(cutlass.Float32, 512)
            reduced = cute.make_tensor(reduced_ptr, cute.make_layout(512))
            local = warp * Int32(128) + lane * Int32(4)

            # H512 = H4 tensor H128. H128 is applied within each quarter;
            # normalized H4 then couples equal coordinates across quarters.
            h0, h1, h2, h3 = _w4a8_had128_quad(a0, a1, a2, a3, lane)
            reduced[local] = h0
            reduced[local + Int32(1)] = h1
            reduced[local + Int32(2)] = h2
            reduced[local + Int32(3)] = h3
            cute.arch.sync_threads()
            quarter_col = tid
            x0 = reduced[quarter_col]
            x1 = reduced[quarter_col + Int32(128)]
            x2 = reduced[quarter_col + Int32(256)]
            x3 = reduced[quarter_col + Int32(384)]
            half = cutlass.Float32(0.5)
            out_base = Int64(token) * Int64(self.hidden) + Int64(block) * Int64(512) + Int64(quarter_col)
            output[out_base] = cutlass.BFloat16(half * (x0 + x1 + x2 + x3))
            output[out_base + Int32(128)] = cutlass.BFloat16(half * (x0 - x1 + x2 - x3))
            output[out_base + Int32(256)] = cutlass.BFloat16(half * (x0 + x1 - x2 - x3))
            output[out_base + Int32(384)] = cutlass.BFloat16(half * (x0 - x1 - x2 + x3))


__all__ = ["P8CoupledTopKSumKernel"]
