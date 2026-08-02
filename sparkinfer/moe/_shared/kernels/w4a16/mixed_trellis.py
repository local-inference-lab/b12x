"""One-launch mixed-bitrate EXL3 Trellis MoE execution.

The route packer assigns every global expert to one combined expert namespace.
Input/intermediate rotations therefore run once. Per-tile dispatch resolves the
combined expert to a bitrate-specialized K3 or K4 decoder while preserving the
single cooperative FC1/activation/FC2 grid used by homogeneous Trellis.

The module stays internal because checkpoint interpretation and runtime planning
belong to the serving framework; SparkInfer owns only the prepared kernel path.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Protocol, Sequence

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.base_dsl.compiler import OptLevel
from cutlass.cutlass_dsl import Int32

from sparkinfer._lib.compiler import KernelCompileSpec, compile as sparkinfer_compile
from sparkinfer._lib.intrinsics import shared_ptr_to_u32
from sparkinfer._lib.runtime_control import raise_if_kernel_resolution_frozen
from sparkinfer._lib.utils import current_cuda_stream, make_ptr

from .host import max_packed_route_slots, packed_gemm_scratch_elements
from .kernel import (
    W4A16FusedMoeKernel,
    _cutlass_element_dtype,
    _fake_m_for_specialization,
    _query_w4a16_kernel_resources,
    compile_w4a16_topk_sum,
    pack_topk_routes_by_expert,
)


@dataclass(frozen=True)
class MixedTrellisCompileResult:
    compiled: object
    topk_sum: object
    size_m: int
    hidden_size: int
    intermediate_size: int
    top_k: int
    tier0_num_experts: int
    tier1_num_experts: int
    tier0_bits: int
    tier1_bits: int
    fc1_tile_k: int
    fc1_tile_n: int
    fc2_tile_k: int
    fc2_tile_n: int
    moe_block_size: int
    fc2_moe_block_size: int
    fc2_schedule_route_block_factor: int
    fc2_paired_m8_routes: bool
    max_m_blocks: int
    blocks_per_sm: int
    sms: int
    shared_memory_bytes: int
    registers_per_thread: int
    local_memory_bytes: int
    rotation_input_dtype: str
    route_ids_dtype: torch.dtype


@dataclass(frozen=True)
class MixedTrellisRotations:
    intermediate: torch.Tensor
    gate_suh: torch.Tensor
    up_suh: torch.Tensor
    down_svh: torch.Tensor


@dataclass
class MixedTrellisBuffers:
    rotation_gate: torch.Tensor
    rotation_up: torch.Tensor
    fc1: torch.Tensor
    activated: torch.Tensor
    fc2: torch.Tensor
    output: torch.Tensor
    packed_route_indices: torch.Tensor
    block_expert_ids: torch.Tensor
    packed_route_count: torch.Tensor
    expert_offsets: torch.Tensor
    expert_counts: torch.Tensor
    fc1_scratch: torch.Tensor
    fc2_scratch: torch.Tensor
    workspace: torch.Tensor


class MixedTrellisTier(Protocol):
    """Prepared Trellis tier fields consumed by the mixed launch."""

    intermediate_rotations: torch.Tensor
    gate_suh: torch.Tensor
    up_suh: torch.Tensor
    down_svh: torch.Tensor
    w13: torch.Tensor
    w2: torch.Tensor
    w13_scale: torch.Tensor
    w2_scale: torch.Tensor
    w13_global_scale: torch.Tensor
    w2_global_scale: torch.Tensor


class W4A16MixedTrellisKernel:
    """One cooperative grid over two native Trellis bitrates."""

    ABI_VERSION = 4

    def __init__(
        self,
        *,
        driver: W4A16FusedMoeKernel,
        tier0: W4A16FusedMoeKernel,
        tier1: W4A16FusedMoeKernel,
    ):
        for name, moe in (("driver", driver), ("tier0", tier0), ("tier1", tier1)):
            if not moe.full_rotation or not moe.intermediate_rotation:
                raise ValueError(f"mixed Trellis {name} requires full rotation")
            if moe.direct_topk_routes or moe.tc_decode_fused_sum:
                raise ValueError(f"mixed Trellis {name} requires route packing")
            if moe.weight_layout != "trellis3_t256":
                raise ValueError(f"mixed Trellis {name} requires native t256 weights")
            if moe.element_dtype != "fp16":
                raise ValueError(f"mixed Trellis {name} requires fp16 GEMM operands")
        for attr in (
            "size_m",
            "hidden_size",
            "intermediate_size",
            "fc1_cols",
            "top_k",
            "moe_block_size",
            "activation",
            "rotation_input_dtype",
            "cta_threads",
            "sms",
        ):
            values = tuple(getattr(moe, attr) for moe in (driver, tier0, tier1))
            if values[1:] != values[:-1]:
                raise ValueError(f"mixed Trellis kernels disagree on {attr}: {values}")
        for phase in ("fc1", "fc2"):
            gemms = tuple(getattr(moe, phase) for moe in (driver, tier0, tier1))
            geometry = tuple(
                (
                    g.n_tiles,
                    g.k_tiles,
                    g.tile_n,
                    g.tile_k,
                    g.cta_threads,
                    g.moe_block_size,
                    g.schedule_route_block_factor,
                    g.paired_m8_routes,
                )
                for g in gemms
            )
            if geometry[1:] != geometry[:-1]:
                raise ValueError(
                    f"mixed Trellis kernels disagree on {phase} geometry: {geometry}"
                )
        fc2_factor = int(driver.fc2.schedule_route_block_factor)
        expected_factor = int(driver.moe_block_size // driver.fc2.moe_block_size)
        if fc2_factor < 1 or expected_factor % fc2_factor != 0:
            raise ValueError(
                "mixed Trellis FC2 schedule factor must divide one packed "
                f"route block: factor={fc2_factor}, maximum={expected_factor}"
            )
        expected_pair = fc2_factor == 2 and driver.fc2.moe_block_size == 8
        if bool(driver.fc2.paired_m8_routes) != expected_pair:
            raise ValueError(
                "mixed Trellis FC2 pair contract mismatch: "
                f"factor={fc2_factor}, m={driver.fc2.moe_block_size}, "
                f"paired={driver.fc2.paired_m8_routes}"
            )
        if tier0.num_experts > 256 or tier1.num_experts > 256:
            raise ValueError("tier-local expert ids must fit in eight bits")
        if driver.num_experts != tier0.num_experts + tier1.num_experts:
            raise ValueError("driver expert count must equal the sum of both tiers")
        self.driver = driver
        self.tier0 = tier0
        self.tier1 = tier1
        self.total_experts = driver.num_experts
        self.size_m = driver.size_m
        self.hidden_size = driver.hidden_size
        self.intermediate_size = driver.intermediate_size
        self.top_k = driver.top_k
        self.cta_threads = driver.cta_threads
        self.sms = driver.sms
        self.blocks_per_sm = min(
            driver.blocks_per_sm, tier0.blocks_per_sm, tier1.blocks_per_sm
        )
        self.shared_words = max(
            driver.shared_words, tier0.shared_words, tier1.shared_words
        )

    @property
    def __cache_key__(self) -> tuple[object, ...]:
        return (
            "w4a16_mixed_trellis",
            self.ABI_VERSION,
            self.driver.__cache_key__,
            self.tier0.__cache_key__,
            self.tier1.__cache_key__,
            self.blocks_per_sm,
            self.shared_words,
        )

    @cute.jit
    def _dispatch_tier_gemm(
        self,
        gemm,
        a_flat: cute.Tensor,
        a_alt_flat: cute.Tensor,
        b_flat: cute.Tensor,
        c_flat: cute.Tensor,
        scales_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        topk_weights: cute.Tensor,
        c_tmp: cute.Tensor,
        locks: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        route_block_idx: Int32,
        local_expert: Int32,
        output_n_tile: Int32,
        reduce_k_tile: Int32,
        reduce_tile_count: Int32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
        active_size_m: Int32,
    ):
        factor = gemm.schedule_route_block_factor
        first_route_block = route_block_idx * Int32(factor)
        first_lock_slot = lock_slot * Int32(factor)
        if cutlass.const_expr(gemm.paired_m8_routes):
            gemm._run_tile_m8_pair(
                a_flat,
                a_alt_flat,
                b_flat,
                c_flat,
                scales_flat,
                global_scale,
                packed_route_indices,
                topk_weights,
                c_tmp,
                locks,
                smem_base,
                tid,
                first_route_block,
                local_expert,
                output_n_tile,
                reduce_k_tile,
                reduce_tile_count,
                reduce_slice_count,
                reduce_slice_idx,
                first_lock_slot,
                active_size_m,
            )
        else:
            for subtile in cutlass.range_constexpr(factor):
                gemm._run_tile(
                    a_flat,
                    a_alt_flat,
                    b_flat,
                    c_flat,
                    scales_flat,
                    global_scale,
                    packed_route_indices,
                    topk_weights,
                    c_tmp,
                    locks,
                    smem_base,
                    tid,
                    first_route_block + Int32(subtile),
                    local_expert,
                    output_n_tile,
                    reduce_k_tile,
                    reduce_tile_count,
                    reduce_slice_count,
                    reduce_slice_idx,
                    first_lock_slot + Int32(subtile),
                    active_size_m,
                )

    @cute.jit
    def _emit_tier_tile(
        self,
        is_fc1: cutlass.Constexpr,
        a_flat: cute.Tensor,
        a_alt_flat: cute.Tensor,
        t0_b_flat: cute.Tensor,
        t0_scales_flat: cute.Tensor,
        t0_global_scale: cute.Tensor,
        t1_b_flat: cute.Tensor,
        t1_scales_flat: cute.Tensor,
        t1_global_scale: cute.Tensor,
        c_flat: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        descriptor_map: cute.Tensor,
        topk_weights: cute.Tensor,
        c_tmp: cute.Tensor,
        locks: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        active_size_m: Int32,
        route_block_idx: Int32,
        output_n_tile: Int32,
        reduce_k_tile: Int32,
        reduce_tile_count: Int32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
    ):
        metadata_block_idx = route_block_idx
        if cutlass.const_expr(not is_fc1):
            metadata_block_idx = route_block_idx // Int32(
                self.driver.moe_block_size
                // (
                    self.driver.fc2.moe_block_size
                    * self.driver.fc2.schedule_route_block_factor
                )
            )
        combined_expert = block_expert_ids[metadata_block_idx].to(Int32)
        if combined_expert >= Int32(0) and combined_expert < Int32(self.total_experts):
            descriptor = descriptor_map[combined_expert].to(Int32)
            if descriptor >= Int32(0):
                tier = descriptor >> Int32(8)
                local_expert = descriptor & Int32(0xFF)
                if tier == Int32(0) and local_expert < Int32(self.tier0.num_experts):
                    if cutlass.const_expr(is_fc1):
                        gemm = self.tier0.fc1
                    else:
                        gemm = self.tier0.fc2
                    self._dispatch_tier_gemm(
                        gemm,
                        a_flat,
                        a_alt_flat,
                        t0_b_flat,
                        c_flat,
                        t0_scales_flat,
                        t0_global_scale,
                        packed_route_indices,
                        topk_weights,
                        c_tmp,
                        locks,
                        smem_base,
                        tid,
                        route_block_idx,
                        local_expert,
                        output_n_tile,
                        reduce_k_tile,
                        reduce_tile_count,
                        reduce_slice_count,
                        reduce_slice_idx,
                        lock_slot,
                        active_size_m,
                    )
                elif tier == Int32(1) and local_expert < Int32(self.tier1.num_experts):
                    if cutlass.const_expr(is_fc1):
                        gemm = self.tier1.fc1
                    else:
                        gemm = self.tier1.fc2
                    self._dispatch_tier_gemm(
                        gemm,
                        a_flat,
                        a_alt_flat,
                        t1_b_flat,
                        c_flat,
                        t1_scales_flat,
                        t1_global_scale,
                        packed_route_indices,
                        topk_weights,
                        c_tmp,
                        locks,
                        smem_base,
                        tid,
                        route_block_idx,
                        local_expert,
                        output_n_tile,
                        reduce_k_tile,
                        reduce_tile_count,
                        reduce_slice_count,
                        reduce_slice_idx,
                        lock_slot,
                        active_size_m,
                    )

    @cute.jit
    def __call__(
        self,
        rotation_input_ptr: cute.Pointer,
        rotation_gate: cute.Tensor,
        rotation_up: cute.Tensor,
        t0_w13: cute.Tensor,
        t0_w2: cute.Tensor,
        t0_w13_scales: cute.Tensor,
        t0_w2_scales: cute.Tensor,
        t0_w13_global: cute.Tensor,
        t0_w2_global: cute.Tensor,
        t1_w13: cute.Tensor,
        t1_w2: cute.Tensor,
        t1_w13_scales: cute.Tensor,
        t1_w2_scales: cute.Tensor,
        t1_w13_global: cute.Tensor,
        t1_w2_global: cute.Tensor,
        fc1: cute.Tensor,
        activated: cute.Tensor,
        fc2: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        descriptor_map: cute.Tensor,
        topk_weights_ptr: cute.Pointer,
        fc1_scratch: cute.Tensor,
        fc2_scratch: cute.Tensor,
        workspace: cute.Tensor,
        intermediate_rotations: cute.Tensor,
        gate_suh: cute.Tensor,
        up_suh: cute.Tensor,
        active_m: cutlass.Int32,
        grid_x: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        rotation_input = cute.make_tensor(
            rotation_input_ptr,
            layout=cute.make_layout(
                (active_m.to(cutlass.Int64) * cutlass.Int64(self.hidden_size),),
                stride=(1,),
            ),
        )
        topk_weights = cute.make_tensor(
            topk_weights_ptr,
            layout=cute.make_layout(
                (active_m.to(cutlass.Int64) * cutlass.Int64(self.top_k),),
                stride=(1,),
            ),
        )
        self.kernel(
            rotation_input,
            rotation_gate,
            rotation_up,
            t0_w13,
            t0_w2,
            t0_w13_scales,
            t0_w2_scales,
            t0_w13_global,
            t0_w2_global,
            t1_w13,
            t1_w2,
            t1_w13_scales,
            t1_w2_scales,
            t1_w13_global,
            t1_w2_global,
            fc1,
            activated,
            fc2,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            descriptor_map,
            topk_weights,
            fc1_scratch,
            fc2_scratch,
            workspace,
            intermediate_rotations,
            gate_suh,
            up_suh,
            active_m,
        ).launch(
            grid=(grid_x, 1, 1),
            block=[self.cta_threads, 1, 1],
            min_blocks_per_mp=self.blocks_per_sm,
            cooperative=True,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        rotation_input: cute.Tensor,
        rotation_gate: cute.Tensor,
        rotation_up: cute.Tensor,
        t0_w13: cute.Tensor,
        t0_w2: cute.Tensor,
        t0_w13_scales: cute.Tensor,
        t0_w2_scales: cute.Tensor,
        t0_w13_global: cute.Tensor,
        t0_w2_global: cute.Tensor,
        t1_w13: cute.Tensor,
        t1_w2: cute.Tensor,
        t1_w13_scales: cute.Tensor,
        t1_w2_scales: cute.Tensor,
        t1_w13_global: cute.Tensor,
        t1_w2_global: cute.Tensor,
        fc1: cute.Tensor,
        activated: cute.Tensor,
        fc2: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        descriptor_map: cute.Tensor,
        topk_weights: cute.Tensor,
        fc1_scratch: cute.Tensor,
        fc2_scratch: cute.Tensor,
        workspace: cute.Tensor,
        intermediate_rotations: cute.Tensor,
        gate_suh: cute.Tensor,
        up_suh: cute.Tensor,
        active_m: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_x_raw, _, _ = cute.arch.grid_dim()
        tid = Int32(tidx)
        cta = Int32(bidx)
        grid_x = Int32(grid_x_raw)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.shared_words], 1024
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())
        fc1_emit = partial(
            self._emit_tier_tile,
            True,
            rotation_gate,
            rotation_up,
            t0_w13,
            t0_w13_scales,
            t0_w13_global,
            t1_w13,
            t1_w13_scales,
            t1_w13_global,
            fc1,
            packed_route_indices,
            block_expert_ids,
            descriptor_map,
            topk_weights,
            fc1_scratch,
            workspace,
            smem_base,
            tid,
            active_m,
        )
        fc2_emit = partial(
            self._emit_tier_tile,
            False,
            activated,
            activated,
            t0_w2,
            t0_w2_scales,
            t0_w2_global,
            t1_w2,
            t1_w2_scales,
            t1_w2_global,
            fc2,
            packed_route_indices,
            block_expert_ids,
            descriptor_map,
            topk_weights,
            fc2_scratch,
            workspace,
            smem_base,
            tid,
            active_m * Int32(self.top_k),
        )
        self.driver._moe_body(
            rotation_gate,
            rotation_up,
            rotation_input,
            t0_w13,
            t0_w2,
            fc1,
            activated,
            fc2,
            t0_w13_scales,
            t0_w2_scales,
            t0_w13_global,
            t0_w2_global,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            t0_w13_global,
            Int32(0),
            topk_weights,
            fc1_scratch,
            fc2_scratch,
            workspace,
            intermediate_rotations,
            gate_suh,
            up_suh,
            descriptor_map,
            Int32(self.total_experts),
            Int32(self.total_experts),
            smem_base,
            tid,
            cta,
            grid_x,
            active_m,
            fc1_emit,
            fc2_emit,
        )


_CACHE: dict[tuple[object, ...], MixedTrellisCompileResult] = {}


def _trellis_weight_fake(
    *, num_experts: int, size_k: int, size_n: int, bits: int
) -> cute.Tensor:
    return cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (num_experts * (size_k // 16) * (size_n // 16) * (8 * bits),),
        assumed_align=16,
    )


def compile_mixed_trellis(
    *,
    size_m: int,
    hidden_size: int,
    intermediate_size: int,
    tier0_num_experts: int,
    tier1_num_experts: int,
    top_k: int,
    max_m_blocks: int,
    sms: int,
    max_shared_mem: int,
    force_tile_config: tuple[int, int, int, int],
    tier0_bits: int = 3,
    tier1_bits: int = 4,
    moe_block_size: int = 8,
    rotation_input_dtype: str = "bf16",
    route_ids_dtype: torch.dtype = torch.int32,
) -> MixedTrellisCompileResult:
    if route_ids_dtype not in (torch.int32, torch.int64):
        raise TypeError("mixed Trellis route IDs must be int32 or int64")
    if int(size_m) * int(top_k) > torch.iinfo(torch.int32).max:
        raise ValueError("mixed Trellis routed-row count must fit in int32")
    fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n = (
        int(value) for value in force_tile_config
    )
    if fc1_tile_k < 128:
        raise ValueError(
            "mixed Trellis FC1 requires tile_k >= 128; narrower K tiles lose "
            "large-M cross-tier partial reductions"
        )
    total_experts = int(tier0_num_experts) + int(tier1_num_experts)
    paired_m8_fc2 = int(moe_block_size) in (32, 64)

    def make_kernel(num_experts: int, bits: int) -> W4A16FusedMoeKernel:
        return W4A16FusedMoeKernel(
            size_m=size_m,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            activation="silu",
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            fc1_tile_n=fc1_tile_n,
            fc1_tile_k=fc1_tile_k,
            fc2_tile_n=fc2_tile_n,
            fc2_tile_k=fc2_tile_k,
            moe_block_size=moe_block_size,
            max_m_blocks=max_m_blocks,
            fc2_moe_block_size=(8 if paired_m8_fc2 else moe_block_size),
            fc2_schedule_route_block_factor=(2 if paired_m8_fc2 else 1),
            element_dtype="fp16",
            weight_layout="trellis3_t256",
            scale_format="e4m3_k32",
            w13_layout="trellis3_t256_proj",
            trellis_bits=bits,
            intermediate_rotation=True,
            full_rotation=True,
            rotation_input_dtype=rotation_input_dtype,
            schedule_whole_tiles=True,
        )

    kernel = W4A16MixedTrellisKernel(
        driver=make_kernel(total_experts, tier0_bits),
        tier0=make_kernel(int(tier0_num_experts), int(tier0_bits)),
        tier1=make_kernel(int(tier1_num_experts), int(tier1_bits)),
    )
    # shared_words is the complete dynamically allocated MemRange used by the
    # cooperative kernel. CUDA permits a launch exactly at the device's
    # opt-in shared-memory limit; rejecting an additional 512 bytes here
    # unnecessarily excludes the stock mixed-K tile geometry at block-64.
    if kernel.shared_words * 4 > int(max_shared_mem):
        raise ValueError(
            "mixed Trellis shared-memory requirement exceeds the device limit: "
            f"required={kernel.shared_words * 4} "
            f"limit={int(max_shared_mem)}"
        )
    device = int(torch.cuda.current_device())
    cache_key = (
        "mixed_trellis",
        device,
        kernel.__cache_key__,
        str(route_ids_dtype),
        int(size_m),
        int(max_m_blocks),
    )
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    compile_m = _fake_m_for_specialization(size_m)
    compile_rows = compile_m * top_k
    fc1_cols = 2 * intermediate_size
    cutlass_dtype = cutlass.Float16
    rotation_dtype = _cutlass_element_dtype(rotation_input_dtype)

    def tensor(dtype, elements: int, *, align: int = 16):
        return cute.runtime.make_fake_compact_tensor(
            dtype, (max(int(elements), 1),), assumed_align=align
        )

    def tier_args(experts: int, bits: int):
        return (
            _trellis_weight_fake(
                num_experts=experts,
                size_k=hidden_size,
                size_n=fc1_cols,
                bits=bits,
            ),
            _trellis_weight_fake(
                num_experts=experts,
                size_k=intermediate_size,
                size_n=hidden_size,
                bits=bits,
            ),
            tensor(cutlass.Int32, 1),
            tensor(cutlass.Int32, 1),
            tensor(cutlass.Float32, experts),
            tensor(cutlass.Float32, experts),
        )

    scratch_elements = max(
        fc1_cols * compile_rows,
        hidden_size * compile_rows,
        4 * 256 * moe_block_size * 256,
    )
    compile_args = (
        make_ptr(rotation_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        tensor(cutlass_dtype, compile_rows * hidden_size),
        tensor(cutlass_dtype, compile_rows * hidden_size),
        *tier_args(int(tier0_num_experts), int(tier0_bits)),
        *tier_args(int(tier1_num_experts), int(tier1_bits)),
        tensor(cutlass_dtype, compile_rows * fc1_cols),
        tensor(cutlass_dtype, compile_rows * intermediate_size),
        tensor(cutlass_dtype, compile_rows * hidden_size),
        tensor(cutlass.Int32, moe_block_size),
        tensor(cutlass.Int32, 1),
        tensor(cutlass.Int32, 1, align=4),
        tensor(cutlass.Int32, total_experts),
        make_ptr(cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4),
        tensor(cutlass.Float32, scratch_elements),
        tensor(cutlass.Float32, scratch_elements),
        tensor(cutlass.Int32, 4 * 256 + 2),
        tensor(cutlass.Float16, total_experts * 3 * intermediate_size),
        tensor(cutlass.Float16, total_experts * hidden_size),
        tensor(cutlass.Float16, total_experts * hidden_size),
        1,
        1,
        current_cuda_stream(),
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    compiled = sparkinfer_compile(
        kernel,
        *compile_args,
        compile_spec=KernelCompileSpec.from_key(
            "moe.w4a16.mixed_trellis", W4A16MixedTrellisKernel.ABI_VERSION, cache_key
        ),
        dsl_compile_options=OptLevel(2),
    )
    registers = -1
    local_bytes = -1
    resources = _query_w4a16_kernel_resources(compiled)
    if resources is not None:
        _, registers, local_bytes = resources
        if local_bytes != 0:
            raise RuntimeError(
                "mixed Trellis codegen spills to local memory "
                f"({local_bytes} bytes/thread)"
            )
    topk_sum = compile_w4a16_topk_sum(
        m=size_m,
        topk=top_k,
        hidden_size=hidden_size,
        element_dtype="fp16",
        full_rotation=True,
        num_experts=total_experts,
        route_num_experts=total_experts,
        route_ids_dtype=route_ids_dtype,
        use_expert_map=True,
    )
    result = MixedTrellisCompileResult(
        compiled=compiled,
        topk_sum=topk_sum,
        size_m=int(size_m),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        top_k=int(top_k),
        tier0_num_experts=int(tier0_num_experts),
        tier1_num_experts=int(tier1_num_experts),
        tier0_bits=int(tier0_bits),
        tier1_bits=int(tier1_bits),
        fc1_tile_k=fc1_tile_k,
        fc1_tile_n=fc1_tile_n,
        fc2_tile_k=fc2_tile_k,
        fc2_tile_n=fc2_tile_n,
        moe_block_size=int(moe_block_size),
        fc2_moe_block_size=int(kernel.driver.fc2.moe_block_size),
        fc2_schedule_route_block_factor=int(
            kernel.driver.fc2.schedule_route_block_factor
        ),
        fc2_paired_m8_routes=bool(kernel.driver.fc2.paired_m8_routes),
        max_m_blocks=int(max_m_blocks),
        blocks_per_sm=int(kernel.blocks_per_sm),
        sms=int(sms),
        shared_memory_bytes=int(kernel.shared_words * 4),
        registers_per_thread=registers,
        local_memory_bytes=local_bytes,
        rotation_input_dtype=str(rotation_input_dtype),
        route_ids_dtype=route_ids_dtype,
    )
    _CACHE[cache_key] = result
    return result


def make_mixed_trellis_buffers(
    launch: MixedTrellisCompileResult,
    *,
    device: torch.device,
    sms: int,
) -> MixedTrellisBuffers:
    capacity_rows = launch.size_m * launch.top_k
    total_experts = launch.tier0_num_experts + launch.tier1_num_experts
    route_slots = max_packed_route_slots(
        capacity_rows, launch.moe_block_size, total_experts
    )
    route_blocks = (route_slots + launch.moe_block_size - 1) // launch.moe_block_size
    if route_blocks > launch.max_m_blocks:
        raise ValueError(
            "mixed Trellis route buffers require "
            f"{route_blocks} blocks, but the launch was compiled for "
            f"{launch.max_m_blocks}"
        )
    fc1_cols = 2 * launch.intermediate_size
    # FC1 consumes rotation_gate before the grid-wide activation barrier. FC2
    # starts only after that barrier, so its output can safely reuse the same
    # storage without changing either phase's tensor layout.
    rotation_gate = torch.empty(
        (capacity_rows, launch.hidden_size), dtype=torch.float16, device=device
    )
    return MixedTrellisBuffers(
        rotation_gate=rotation_gate,
        rotation_up=torch.empty(
            (capacity_rows, launch.hidden_size), dtype=torch.float16, device=device
        ),
        fc1=torch.empty((capacity_rows, fc1_cols), dtype=torch.float16, device=device),
        activated=torch.empty(
            (capacity_rows, launch.intermediate_size),
            dtype=torch.float16,
            device=device,
        ),
        fc2=rotation_gate,
        output=torch.empty(
            (launch.size_m, launch.hidden_size), dtype=torch.float32, device=device
        ),
        packed_route_indices=torch.empty(route_slots, dtype=torch.int32, device=device),
        block_expert_ids=torch.empty(route_blocks, dtype=torch.int32, device=device),
        packed_route_count=torch.empty(1, dtype=torch.int32, device=device),
        expert_offsets=torch.empty(total_experts + 1, dtype=torch.int32, device=device),
        expert_counts=torch.empty(total_experts, dtype=torch.int32, device=device),
        fc1_scratch=torch.empty(
            packed_gemm_scratch_elements(
                size_n=fc1_cols,
                route_slots=route_slots,
                moe_block_size=launch.moe_block_size,
                sms=sms,
            ),
            dtype=torch.float32,
            device=device,
        ),
        fc2_scratch=torch.empty(
            packed_gemm_scratch_elements(
                size_n=launch.hidden_size,
                route_slots=route_slots,
                moe_block_size=launch.moe_block_size,
                sms=sms,
            ),
            dtype=torch.float32,
            device=device,
        ),
        workspace=torch.zeros(
            max(sms * 4, launch.blocks_per_sm * sms) + 2,
            dtype=torch.int32,
            device=device,
        ),
    )


def build_ordered_maps(
    tier0_num_experts: int,
    tier1_num_experts: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    tier0_num_experts = int(tier0_num_experts)
    tier1_num_experts = int(tier1_num_experts)
    return build_tiered_maps(
        range(tier0_num_experts),
        range(tier0_num_experts, tier0_num_experts + tier1_num_experts),
        device=device,
    )


def build_tiered_maps(
    tier0_global_ids: Sequence[int],
    tier1_global_ids: Sequence[int],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build immutable route and descriptor maps for one mixed expert layer."""

    tier0_ids = tuple(int(expert_id) for expert_id in tier0_global_ids)
    tier1_ids = tuple(int(expert_id) for expert_id in tier1_global_ids)
    if len(tier0_ids) > 256 or len(tier1_ids) > 256:
        raise ValueError("each mixed Trellis tier supports at most 256 experts")
    total = len(tier0_ids) + len(tier1_ids)
    if sorted((*tier0_ids, *tier1_ids)) != list(range(total)):
        raise ValueError(
            f"mixed Trellis tier ids must be a disjoint partition of [0, {total})"
        )
    global_to_combined_host = [-1] * total
    for local_id, global_id in enumerate(tier0_ids):
        global_to_combined_host[global_id] = local_id
    for local_id, global_id in enumerate(tier1_ids):
        global_to_combined_host[global_id] = len(tier0_ids) + local_id
    global_to_combined = torch.tensor(
        global_to_combined_host, dtype=torch.int32, device=device
    )
    descriptor = torch.tensor(
        [*range(len(tier0_ids)), *((1 << 8) | i for i in range(len(tier1_ids)))],
        dtype=torch.int32,
        device=device,
    )
    return global_to_combined, descriptor


def combine_trellis_rotations(
    tier0: MixedTrellisTier, tier1: MixedTrellisTier
) -> MixedTrellisRotations:
    """Materialize one tier-ordered table set once during model preparation."""
    return MixedTrellisRotations(
        intermediate=torch.cat(
            (tier0.intermediate_rotations, tier1.intermediate_rotations), dim=0
        ).contiguous(),
        gate_suh=torch.cat((tier0.gate_suh, tier1.gate_suh), dim=0).contiguous(),
        up_suh=torch.cat((tier0.up_suh, tier1.up_suh), dim=0).contiguous(),
        down_svh=torch.cat((tier0.down_svh, tier1.down_svh), dim=0).contiguous(),
    )


def run_mixed_trellis(
    x: torch.Tensor,
    tier0: MixedTrellisTier,
    tier1: MixedTrellisTier,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    global_to_combined: torch.Tensor,
    descriptor_map: torch.Tensor,
    rotations: MixedTrellisRotations,
    launch: MixedTrellisCompileResult,
    buffers: MixedTrellisBuffers,
) -> torch.Tensor:
    m = int(x.shape[0])
    if m <= 0:
        raise ValueError(f"mixed Trellis requires at least one active row, got {m}")
    if m > launch.size_m:
        raise ValueError(f"active rows {m} exceed launch capacity {launch.size_m}")
    expected_input_dtype = (
        torch.bfloat16 if launch.rotation_input_dtype == "bf16" else torch.float16
    )
    for name, tensor, expected_dtype in (
        ("input", x, expected_input_dtype),
        ("topk_ids", topk_ids, launch.route_ids_dtype),
        ("topk_weights", topk_weights, torch.float32),
    ):
        if tensor.dtype != expected_dtype:
            raise TypeError(
                f"mixed Trellis {name} must be {expected_dtype}, got {tensor.dtype}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"mixed Trellis {name} must be contiguous")
    total_experts = launch.tier0_num_experts + launch.tier1_num_experts
    if int(global_to_combined.numel()) != total_experts:
        raise ValueError("global_to_combined must cover every routed expert")
    if int(descriptor_map.numel()) != total_experts:
        raise ValueError("descriptor_map must cover the combined namespace")
    required_route_slots = max_packed_route_slots(
        m * launch.top_k, launch.moe_block_size, total_experts
    )
    required_route_blocks = (
        required_route_slots + launch.moe_block_size - 1
    ) // launch.moe_block_size
    if required_route_blocks > launch.max_m_blocks:
        raise RuntimeError(
            "mixed Trellis request requires "
            f"{required_route_blocks} route blocks, but the launch supports "
            f"{launch.max_m_blocks}"
        )
    if buffers.packed_route_indices.numel() < required_route_slots:
        raise RuntimeError(
            "mixed Trellis packed-route buffer is below request capacity"
        )
    if buffers.block_expert_ids.numel() < required_route_blocks:
        raise RuntimeError(
            "mixed Trellis block-expert buffer is below request capacity"
        )
    packed, block_experts, packed_count = pack_topk_routes_by_expert(
        topk_ids,
        launch.moe_block_size,
        total_experts,
        expert_map=global_to_combined,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
        expert_counts=buffers.expert_counts,
    )
    stream = current_cuda_stream()
    launch.compiled(
        make_ptr(
            _cutlass_element_dtype(launch.rotation_input_dtype),
            x.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        buffers.rotation_gate.view(-1),
        buffers.rotation_up.view(-1),
        tier0.w13.view(torch.int32).view(-1),
        tier0.w2.view(torch.int32).view(-1),
        tier0.w13_scale.view(torch.uint8).view(torch.int32).view(-1),
        tier0.w2_scale.view(torch.uint8).view(torch.int32).view(-1),
        tier0.w13_global_scale,
        tier0.w2_global_scale,
        tier1.w13.view(torch.int32).view(-1),
        tier1.w2.view(torch.int32).view(-1),
        tier1.w13_scale.view(torch.uint8).view(torch.int32).view(-1),
        tier1.w2_scale.view(torch.uint8).view(torch.int32).view(-1),
        tier1.w13_global_scale,
        tier1.w2_global_scale,
        buffers.fc1.view(-1),
        buffers.activated.view(-1),
        buffers.fc2.view(-1),
        packed,
        block_experts,
        packed_count,
        descriptor_map,
        make_ptr(
            cutlass.Float32,
            topk_weights.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        buffers.fc1_scratch,
        buffers.fc2_scratch,
        buffers.workspace,
        rotations.intermediate.view(-1),
        rotations.gate_suh.view(-1),
        rotations.up_suh.view(-1),
        m,
        max(int(launch.blocks_per_sm) * int(launch.sms), 1),
        stream,
    )
    launch.topk_sum.compiled(
        make_ptr(
            cutlass.Float16,
            buffers.fc2.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float32,
            buffers.output.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float32,
            topk_weights.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        make_ptr(
            cutlass.Int32 if topk_ids.dtype == torch.int32 else cutlass.Int64,
            topk_ids.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4 if topk_ids.dtype == torch.int32 else 8,
        ),
        make_ptr(
            cutlass.Int32,
            global_to_combined.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        make_ptr(
            cutlass.Float16,
            rotations.down_svh.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        Int32(launch.topk_sum.num_experts),
        Int32(launch.topk_sum.route_num_experts),
        m,
        stream,
    )
    return buffers.output[:m]


__all__ = [
    "MixedTrellisBuffers",
    "MixedTrellisCompileResult",
    "MixedTrellisRotations",
    "build_ordered_maps",
    "build_tiered_maps",
    "combine_trellis_rotations",
    "compile_mixed_trellis",
    "make_mixed_trellis_buffers",
    "run_mixed_trellis",
]
