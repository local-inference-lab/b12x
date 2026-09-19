"""Canonical-ID remapping and native W4A16 ordered route reduction."""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda

from b12x.moe._shared.kernels.w4a16.kernel import _materialize_w4a16_topk_route_f32


class Remap:
    def __init__(self, experts):
        self.experts = experts

    @cute.jit
    def __call__(
        self,
        mapping: cute.Pointer,
        maps: cute.Pointer,
        ids: cute.Pointer,
        safe: cute.Pointer,
        live: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(mapping, maps, ids, safe, live).launch(
            grid=(cute.ceil_div(cutlass.max(live, self.experts), 256), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mapping: cute.Pointer,
        maps: cute.Pointer,
        ids: cute.Pointer,
        safe: cute.Pointer,
        live: cutlass.Int32,
    ):
        block, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        idx = cutlass.Int64(block) * 256 + tid
        if idx < self.experts:
            tier, row = mapping[2 * idx], mapping[2 * idx + 1]
            maps[idx] = cutlass.Int32(-1)
            maps[self.experts + idx] = cutlass.Int32(-1)
            maps[cutlass.Int64(tier) * self.experts + idx] = row
        if idx < live:
            expert = cutlass.Int64(ids[idx])
            value = cutlass.Int32(-1)
            if expert >= 0 and expert < self.experts:
                value = cutlass.Int32(expert)
            safe[idx] = value


class OrderedSum:
    def __init__(self, experts, hidden, topk):
        self.experts, self.hidden, self.topk = experts, hidden, topk

    @cute.jit
    def __call__(
        self,
        hot: cute.Pointer,
        cold: cute.Pointer,
        ids: cute.Pointer,
        mapping: cute.Pointer,
        output: cute.Pointer,
        live: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(hot, cold, ids, mapping, output, live).launch(
            grid=(cute.ceil_div(live * self.hidden, 256), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        hot: cute.Pointer,
        cold: cute.Pointer,
        ids: cute.Pointer,
        mapping: cute.Pointer,
        output: cute.Pointer,
        live: cutlass.Int32,
    ):
        block, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        idx = cutlass.Int64(block) * 256 + tid
        if idx < cutlass.Int64(live) * self.hidden:
            token, col = idx // self.hidden, idx % self.hidden
            acc = cutlass.Float32(0)
            for rank in cutlass.range_constexpr(self.topk):
                route = token * self.topk + rank
                expert = cutlass.Int64(ids[route])
                if expert >= 0 and expert < self.experts:
                    value = cutlass.Float32(0)
                    if mapping[2 * expert] == 0:
                        value = hot[route * self.hidden + col].to(cutlass.Float32)
                    else:
                        value = cold[route * self.hidden + col].to(cutlass.Float32)
                    acc += _materialize_w4a16_topk_route_f32(value)
            output[idx] = acc.to(cutlass.BFloat16)

