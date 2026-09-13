"""Trellis expert transforms with fixed storage and runtime route counts.

The ordinary path rounds at each FP16 checkpoint boundary. Coupled transforms
retain FP32 between their two Hadamards and use the checkpoint's interleaved
gate/up basis. Projection kernels consume only the resulting FP16 operands.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float16, Float32, Int32, Int64

from ..w4a8_trellis_decode import _w4a8_had128_quad as had128_quad


@cute.jit
def had128(values: cute.Tensor, lane: Int32):
    result = cute.make_rmem_tensor(4, Float32)
    result[0], result[1], result[2], result[3] = had128_quad(
        values[0], values[1], values[2], values[3], lane
    )
    return result


@cute.jit
def had512(values: cute.Tensor, lane: Int32):
    quarters = cute.make_rmem_tensor((4, 4), Float32)
    for group in cutlass.range_constexpr(4):
        (
            quarters[group, 0],
            quarters[group, 1],
            quarters[group, 2],
            quarters[group, 3],
        ) = had128_quad(
            values[4 * group],
            values[4 * group + 1],
            values[4 * group + 2],
            values[4 * group + 3],
            lane,
        )
    result = cute.make_rmem_tensor(16, Float32)
    for j in cutlass.range_constexpr(4):
        a, b = quarters[0, j] + quarters[1, j], quarters[0, j] - quarters[1, j]
        c, d = quarters[2, j] + quarters[3, j], quarters[2, j] - quarters[3, j]
        result[j], result[4 + j] = (a + c) * 0.5, (b + d) * 0.5
        result[8 + j], result[12 + j] = (a - c) * 0.5, (b - d) * 0.5
    return result


@cute.jit
def activate(gate: Float32, up: Float32, kind: cutlass.Constexpr):
    if cutlass.const_expr(kind == "situ"):
        gate_clip = 4.0 * cute.math.tanh(gate * 0.25)
        up_clip = 25.0 * cute.math.tanh(up * 0.04)
        value = gate_clip / (1.0 + cute.math.exp(-gate)) * up_clip
    else:
        value = gate / (1.0 + cute.math.exp(-gate)) * up
    return value


class MapRoutes:
    """Map global routing and output-scale IDs without narrowing Int64 IDs."""

    def __init__(self, capacity, experts, route_experts, *, mapped, output_mapped):
        if min(capacity, experts, route_experts) <= 0 or capacity > 2**31 - 1:
            raise ValueError(
                "Trellis route geometry must be positive with Int32 live capacity"
            )
        self.capacity, self.experts, self.route_experts = (
            capacity,
            experts,
            route_experts,
        )
        self.mapped, self.output_mapped = mapped, output_mapped

    @cute.jit
    def __call__(
        self,
        ids: cute.Pointer,
        route_map: cute.Pointer,
        output_map: cute.Pointer,
        local: cute.Pointer,
        output_ids: cute.Pointer,
        live_routes: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(ids, route_map, output_map, local, output_ids, live_routes).launch(
            grid=(cute.ceil_div(live_routes, 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, ids, route_map, output_map, local, output_ids, live_routes: Int32):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        row = Int64(block) * 128 + Int64(thread)
        source = cute.make_tensor(ids, cute.make_layout(self.capacity))
        routes = cute.make_tensor(local, cute.make_layout(self.capacity))
        outputs = cute.make_tensor(output_ids, cute.make_layout(self.capacity))
        mapping = cute.make_tensor(route_map, cute.make_layout(self.route_experts))
        output_mapping = cute.make_tensor(
            output_map, cute.make_layout(self.route_experts)
        )
        if row < live_routes:
            expert = Int64(source[row])
            compute, output = Int64(-1), Int64(-1)
            if (expert >= 0) & (expert < self.route_experts):
                compute = expert
                if cutlass.const_expr(self.mapped):
                    compute = Int64(mapping[expert])
                output = compute
                if cutlass.const_expr(self.output_mapped):
                    output = Int64(output_mapping[expert])
            if (compute < 0) | (compute >= self.experts):
                compute = Int64(-1)
            if (output < 0) | (output >= self.experts) | (compute < 0):
                output = Int64(-1)
            routes[row], outputs[row] = compute, output


class InputRotation:
    def __init__(self, hidden, experts, top_k, capacity, *, coupled):
        if min(hidden, experts, top_k, capacity) <= 0 or hidden % (
            512 if coupled else 128
        ):
            raise ValueError(
                "Trellis input rotation requires positive geometry and complete Hadamard blocks"
            )
        if capacity * (hidden // 128) > 2**31 - 1:
            raise ValueError("Trellis input rotation exceeds the Int32 grid capacity")
        self.hidden, self.experts, self.top_k, self.capacity = (
            hidden,
            experts,
            top_k,
            capacity,
        )
        self.coupled = coupled
        self.block = 512 if coupled else 128

    @cute.jit
    def __call__(
        self,
        source: cute.Pointer,
        ids: cute.Pointer,
        suh: cute.Pointer,
        output: cute.Pointer,
        scale_stride: Int64,
        live_routes: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(source, ids, suh, output, scale_stride, live_routes).launch(
            grid=(cute.ceil_div(live_routes * (self.hidden // self.block), 4), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, source, ids, suh, output, scale_stride: Int64, live_routes: Int32):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        lane = Int32(thread % 32)
        unit = Int64(block) * 4 + Int64(thread // 32)
        blocks = self.hidden // self.block
        x = cute.make_tensor(
            source, cute.make_layout((self.capacity // self.top_k) * self.hidden)
        )
        routes = cute.make_tensor(ids, cute.make_layout(self.capacity))
        scales = cute.make_tensor(
            suh, cute.make_layout(Int64(self.experts) * scale_stride + self.hidden)
        )
        dest = cute.make_tensor(output, cute.make_layout(self.capacity * self.hidden))
        if unit < Int64(live_routes) * blocks:
            route, slab = unit // blocks, unit % blocks
            expert = Int64(routes[route])
            valid = (expert >= 0) & (expert < self.experts)
            values = cute.make_rmem_tensor(self.block // 32, Float32)
            for j in cutlass.range_constexpr(self.block // 32):
                column = slab * self.block + (j // 4) * 128 + Int64(lane) * 4 + j % 4
                value = Float32(0)
                if valid:
                    value = (
                        x[(route // self.top_k) * Int64(self.hidden) + column]
                        .to(Float16)
                        .to(Float32)
                    )
                values[j] = value
            if cutlass.const_expr(self.coupled):
                values = had512(values, lane)
            for group in cutlass.range_constexpr(self.block // 128):
                scaled = cute.make_rmem_tensor(4, Float32)
                for j in cutlass.range_constexpr(4):
                    column = slab * self.block + group * 128 + Int64(lane) * 4 + j
                    value = Float32(0)
                    if valid:
                        value = (
                            (
                                values[4 * group + j]
                                * scales[expert * scale_stride + column].to(Float32)
                            )
                            .to(Float16)
                            .to(Float32)
                        )
                    scaled[j] = value
                rotated = had128(scaled, lane)
                for j in cutlass.range_constexpr(4):
                    column = slab * self.block + group * 128 + Int64(lane) * 4 + j
                    dest[route * Int64(self.hidden) + column] = rotated[j].to(Float16)


class IntermediateRotation:
    def __init__(self, intermediate, experts, capacity, *, coupled, activation):
        if min(intermediate, experts, capacity) <= 0 or intermediate % 128:
            raise ValueError(
                "Trellis intermediate rotation requires positive geometry and complete H128 blocks"
            )
        if capacity * (intermediate // 128) > 2**31 - 1:
            raise ValueError(
                "Trellis intermediate rotation exceeds the Int32 grid capacity"
            )
        if activation not in {"silu", "situ"} or (coupled and activation != "situ"):
            raise ValueError(
                "coupled Trellis requires SiTU; ordinary Trellis supports SiLU or SiTU"
            )
        self.width, self.experts, self.capacity = intermediate, experts, capacity
        self.coupled, self.activation = coupled, activation

    @cute.jit
    def __call__(
        self,
        gate: cute.Pointer,
        up: cute.Pointer,
        ids: cute.Pointer,
        rotations: cute.Pointer,
        output: cute.Pointer,
        live_routes: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(gate, up, ids, rotations, output, live_routes).launch(
            grid=(cute.ceil_div(live_routes * (self.width // 128), 4), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, gate, up, ids, rotations, output, live_routes: Int32):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        lane = Int32(thread % 32)
        unit = Int64(block) * 4 + Int64(thread // 32)
        blocks = self.width // 128
        gate_t = cute.make_tensor(gate, cute.make_layout(self.capacity * self.width))
        up_t = cute.make_tensor(up, cute.make_layout(self.capacity * self.width))
        routes = cute.make_tensor(ids, cute.make_layout(self.capacity))
        rot_width = self.width * (6 if self.coupled else 3)
        scales = cute.make_tensor(rotations, cute.make_layout(self.experts * rot_width))
        dest = cute.make_tensor(output, cute.make_layout(self.capacity * self.width))
        if unit < Int64(live_routes) * blocks:
            route, slab = unit // blocks, unit % blocks
            expert = Int64(routes[route])
            valid = (expert >= 0) & (expert < self.experts)
            rot_base = expert * Int64(rot_width)
            values = cute.make_rmem_tensor(4, Float32)
            values.fill(Float32(0))
            if cutlass.const_expr(not self.coupled):
                g, u = (
                    cute.make_rmem_tensor(4, Float32),
                    cute.make_rmem_tensor(4, Float32),
                )
                g.fill(Float32(0))
                u.fill(Float32(0))
                for j in cutlass.range_constexpr(4):
                    column = slab * 128 + Int64(lane) * 4 + j
                    if valid:
                        g[j], u[j] = (
                            gate_t[route * Int64(self.width) + column].to(Float32),
                            up_t[route * Int64(self.width) + column].to(Float32),
                        )
                g, u = had128(g, lane), had128(u, lane)
                for j in cutlass.range_constexpr(4):
                    column = slab * 128 + Int64(lane) * 4 + j
                    if valid:
                        gv = (
                            (g[j] * scales[rot_base + column].to(Float32))
                            .to(Float16)
                            .to(Float32)
                        )
                        uv = (
                            (u[j] * scales[rot_base + self.width + column].to(Float32))
                            .to(Float16)
                            .to(Float32)
                        )
                        activated = (
                            activate(gv, uv, self.activation).to(Float16).to(Float32)
                        )
                        values[j] = (
                            (
                                activated
                                * scales[rot_base + 2 * self.width + column].to(Float32)
                            )
                            .to(Float16)
                            .to(Float32)
                        )
                result = had128(values, lane)
            else:
                adjacent = cute.make_rmem_tensor(4, Float32)
                for group in cutlass.range_constexpr(2):
                    raw, scale = (
                        cute.make_rmem_tensor(4, Float32),
                        cute.make_rmem_tensor(4, Float32),
                    )
                    raw.fill(Float32(0))
                    scale.fill(Float32(0))
                    for j in cutlass.range_constexpr(4):
                        interleaved = group * 128 + Int64(lane) * 4 + j
                        column = (
                            slab * 128 + (interleaved // 64) * 32 + interleaved % 32
                        )
                        projection = (interleaved // 32) % 2
                        if valid:
                            value = gate_t[route * Int64(self.width) + column].to(
                                Float32
                            )
                            if projection != 0:
                                value = up_t[route * Int64(self.width) + column].to(
                                    Float32
                                )
                            raw[j] = value
                            scale[j] = scales[
                                rot_base + projection * Int64(self.width) + column
                            ].to(Float32)
                    raw = had128(raw, lane)
                    for j in cutlass.range_constexpr(4):
                        raw[j] = raw[j] * scale[j]
                    raw = had128(raw, lane)
                    for j in cutlass.range_constexpr(4):
                        if valid:
                            raw[j] = raw[j] * scales[
                                rot_base
                                + 3 * self.width
                                + slab * 256
                                + group * 128
                                + Int64(lane) * 4
                                + j
                            ].to(Float32)
                    adjacent[2 * group] = activate(raw[0], raw[1], "situ")
                    adjacent[2 * group + 1] = activate(raw[2], raw[3], "situ")
                # Each lane initially holds two values from each half. Shuffle
                # both halves before selecting four consecutive output columns.
                for j in cutlass.range_constexpr(4):
                    peer = (lane * 4 + j) % 64 // 2
                    low = cute.arch.shuffle_sync(adjacent[j % 2], peer)
                    high = cute.arch.shuffle_sync(adjacent[2 + j % 2], peer)
                    value = low
                    if lane >= 16:
                        value = high
                    column = slab * 128 + Int64(lane) * 4 + j
                    if valid:
                        values[j] = value * scales[
                            rot_base + 5 * self.width + column
                        ].to(Float32)
                values = had128(values, lane)
                for j in cutlass.range_constexpr(4):
                    column = slab * 128 + Int64(lane) * 4 + j
                    if valid:
                        values[j] = values[j] * scales[
                            rot_base + 2 * self.width + column
                        ].to(Float32)
                result = had128(values, lane)
            for j in cutlass.range_constexpr(4):
                column = slab * 128 + Int64(lane) * 4 + j
                dest[route * Int64(self.width) + column] = result[j].to(Float16)


class OutputRotation:
    def __init__(self, hidden, experts, top_k, capacity, *, coupled):
        if min(hidden, experts, top_k, capacity) <= 0 or hidden % (
            512 if coupled else 128
        ):
            raise ValueError(
                "Trellis output rotation requires positive geometry and complete Hadamard blocks"
            )
        if capacity * (hidden // 128) > 2**31 - 1:
            raise ValueError("Trellis output rotation exceeds the Int32 grid capacity")
        self.hidden, self.experts, self.top_k, self.capacity = (
            hidden,
            experts,
            top_k,
            capacity,
        )
        self.coupled = coupled
        self.block = 512 if coupled else 128

    @cute.jit
    def __call__(
        self,
        source: cute.Pointer,
        ids: cute.Pointer,
        svh: cute.Pointer,
        weights: cute.Pointer,
        output: cute.Pointer,
        scale_stride: Int64,
        live_tokens: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            source, ids, svh, weights, output, scale_stride, live_tokens
        ).launch(
            grid=(cute.ceil_div(live_tokens * (self.hidden // self.block), 4), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self, source, ids, svh, weights, output, scale_stride: Int64, live_tokens: Int32
    ):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        lane = Int32(thread % 32)
        unit = Int64(block) * 4 + Int64(thread // 32)
        blocks = self.hidden // self.block
        x = cute.make_tensor(
            source, cute.make_layout(self.capacity * self.top_k * self.hidden)
        )
        routes = cute.make_tensor(ids, cute.make_layout(self.capacity * self.top_k))
        scales = cute.make_tensor(
            svh, cute.make_layout(Int64(self.experts) * scale_stride + self.hidden)
        )
        router = cute.make_tensor(weights, cute.make_layout(self.capacity * self.top_k))
        dest = cute.make_tensor(output, cute.make_layout(self.capacity * self.hidden))
        if unit < Int64(live_tokens) * blocks:
            token, slab = unit // blocks, unit % blocks
            result = cute.make_rmem_tensor(self.block // 32, Float32)
            result.fill(Float32(0))
            for slot in cutlass.range_constexpr(self.top_k):
                route = token * Int64(self.top_k) + slot
                expert = Int64(routes[route])
                valid = (expert >= 0) & (expert < self.experts)
                for group in cutlass.range_constexpr(self.block // 128):
                    values = cute.make_rmem_tensor(4, Float32)
                    values.fill(Float32(0))
                    for j in cutlass.range_constexpr(4):
                        column = slab * self.block + group * 128 + Int64(lane) * 4 + j
                        if valid:
                            values[j] = x[route * Int64(self.hidden) + column].to(
                                Float32
                            )
                    values = had128(values, lane)
                    for j in cutlass.range_constexpr(4):
                        column = slab * self.block + group * 128 + Int64(lane) * 4 + j
                        if valid:
                            result[4 * group + j] += (
                                values[j]
                                * scales[expert * scale_stride + column].to(Float32)
                            ) * router[route].to(Float32)
            if cutlass.const_expr(self.coupled):
                result = had512(result, lane)
            for j in cutlass.range_constexpr(self.block // 32):
                column = slab * self.block + (j // 4) * 128 + Int64(lane) * 4 + j % 4
                dest[token * Int64(self.hidden) + column] = result[j].to(
                    dest.element_type
                )
