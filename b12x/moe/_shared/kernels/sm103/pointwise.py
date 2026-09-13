"""CuTe route quantization, SwiGLU requantization, and deterministic reduction."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint8
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def pack_fp4_pair(lo, hi, *, loc=None, ip=None):
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(lo).ir_value(loc=loc, ip=ip),
                Float32(hi).ir_value(loc=loc, ip=ip),
            ],
            "{ .reg .b8 pair; cvt.rn.satfinite.e2m1x2.f32 pair, $2, $1; cvt.u32.u8 $0, pair; }",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    ).to(Uint8)


class RoutedQuantize:
    def __init__(
        self, k, top_k, experts, *, activation=False, gate_first=True, limit=None
    ):
        self.k, self.top_k, self.experts = k, top_k, experts
        self.activation, self.gate_first, self.limit = activation, gate_first, limit

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        ids: cute.Pointer,
        gs: cute.Pointer,
        q: cute.Pointer,
        sf: cute.Pointer,
        live_routes: Int32,
        scale_stride: Int32,
        stream: cuda.CUstream,
    ):
        x = cute.make_tensor(x, cute.make_layout(Int64(live_routes) * self.k * 2))
        ids = cute.make_tensor(ids, cute.make_layout(live_routes))
        gs = cute.make_tensor(gs, cute.make_layout(self.experts))
        q = cute.make_tensor(q, cute.make_layout(Int64(live_routes) * (self.k // 2)))
        sf = cute.make_tensor(
            sf, cute.make_layout(Int64(live_routes) * 128 * (self.k // 16))
        )
        self.kernel(x, ids, gs, q, sf, live_routes, scale_stride).launch(
            grid=(cute.ceil_div(self.k // 16, 128), live_routes, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        ids: cute.Tensor,
        gs: cute.Tensor,
        q: cute.Tensor,
        sf: cute.Tensor,
        live_routes: Int32,
        scale_stride: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, route, _ = cute.arch.block_idx()
        group = block * 128 + tid
        route64 = Int64(route)
        if group < self.k // 16:
            expert = Int64(ids[route64])
            valid = (expert >= 0) & (expert < self.experts)
            values = cute.make_rmem_tensor(16, Float32)
            maximum = Float32(0.0)
            global_scale = Float32(1.0)
            if valid:
                global_scale = gs[Int64(expert) * scale_stride]
            for j in cutlass.range_constexpr(16):
                value = Float32(0.0)
                if valid:
                    if cutlass.const_expr(self.activation):
                        offset = route64 * (2 * self.k) + group * 16 + j
                        if cutlass.const_expr(self.gate_first):
                            gate = Float32(x[offset])
                            up = Float32(x[offset + self.k])
                        else:
                            up = Float32(x[offset])
                            gate = Float32(x[offset + self.k])
                        if cutlass.const_expr(self.limit is not None):
                            gate = cute.arch.fmin(gate, Float32(self.limit))
                            up = cute.arch.fmax(
                                cute.arch.fmin(up, Float32(self.limit)),
                                Float32(-self.limit),
                            )
                        value = (
                            (gate / (1.0 + cute.math.exp(-gate)) * up)
                            .to(cutlass.BFloat16)
                            .to(Float32)
                        )
                    else:
                        value = Float32(
                            x[(route64 // self.top_k) * self.k + group * 16 + j]
                        )
                values[j] = value
                maximum = cute.arch.fmax(maximum, cute.math.abs(value))
            scale = (global_scale * (maximum / 6.0)).to(cutlass.Float8E4M3FN)
            scale_f32 = scale.to(Float32)
            reciprocal = Float32(0.0)
            if scale_f32 > 0:
                reciprocal = global_scale / scale_f32
            for j in cutlass.range_constexpr(8):
                packed = pack_fp4_pair(
                    values[2 * j] * reciprocal, values[2 * j + 1] * reciprocal
                )
                q[route64 * (self.k // 2) + group * 8 + j] = packed
            # CUTLASS scale atom ((32,4),4), strides ((16,4),1).
            # All padding is written on every launch, including invalid routes.
            for row in cutlass.range_constexpr(128):
                offset = route64 * (128 * (self.k // 16)) + (group // 4) * 512
                offset += (row % 32) * 16 + (row // 32) * 4 + group % 4
                sf[offset] = scale


class TopKSum:
    def __init__(self, hidden, top_k):
        self.hidden, self.top_k = hidden, top_k

    @cute.jit
    def __call__(
        self,
        routes: cute.Pointer,
        weights: cute.Pointer,
        output: cute.Pointer,
        tokens: Int32,
        stream: cuda.CUstream,
    ):
        routes = cute.make_tensor(
            routes, cute.make_layout(Int64(tokens) * self.top_k * self.hidden)
        )
        weights = cute.make_tensor(
            weights, cute.make_layout(Int64(tokens) * self.top_k)
        )
        output = cute.make_tensor(output, cute.make_layout(Int64(tokens) * self.hidden))
        self.kernel(routes, weights, output, tokens).launch(
            grid=(cute.ceil_div(self.hidden, 128), tokens, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        routes: cute.Tensor,
        weights: cute.Tensor,
        output: cute.Tensor,
        tokens: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, token, _ = cute.arch.block_idx()
        col = block * 128 + tid
        if col < self.hidden:
            value = Float32(0.0)
            for rank in cutlass.range_constexpr(self.top_k):
                route = Int64(token) * self.top_k + rank
                value += Float32(routes[route * self.hidden + col]) * Float32(
                    weights[route]
                )
            output[Int64(token) * self.hidden + col] = value.to(cutlass.BFloat16)
