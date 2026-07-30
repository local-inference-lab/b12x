"""Private one-grid mixed-K full-rotation Trellis MoE comparison path.

This module composes the existing production Trellis weight preparation and
scratch contract with the full-rotation K3/K4 hybrid kernel.  It is kept
separate from the stable uniform API so a deployment must explicitly opt in
and can retain the two-pass implementation as a rollback oracle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import torch

from ..._lib.scratch import ScratchBufferSpec, scratch_buffer_spec, scratch_tensor
from .._shared.kernels.w4a16.host import (
    W4A16BufferPlan,
    max_packed_route_slots,
    plan_w4a16_buffers,
)
from .._shared.kernels.w4a16.kernel import (
    W4A16FusedMoeFullRotationHybridCompileResult,
    W4A16TopKSumCompileResult,
    compile_w4a16_fused_moe_full_rotation_hybrid,
    compile_w4a16_topk_sum,
    run_w4a16_moe_full_rotation_hybrid,
)
from ._impl import (
    TrellisMoECaps,
    TrellisMoEWeights,
    _ARENA_ALIGNMENT,
    _input_dtype_name,
    _make_arena_layout,
    _resolve_cuda_device,
    _validate_runtime_tensor,
)


@dataclass(frozen=True)
class MixedTrellisMoEPlan:
    caps: TrellisMoECaps
    tier0_num_experts: int
    tier0_trellis_bits: int
    tier1_num_experts: int
    tier1_trellis_bits: int
    buffer_plan: W4A16BufferPlan
    fused_launches: tuple[
        tuple[int, W4A16FusedMoeFullRotationHybridCompileResult], ...
    ] = field(repr=False)
    identity_sums: tuple[W4A16TopKSumCompileResult, ...] = field(repr=False)
    _arena_layout: object = field(repr=False)
    _scratch_specs: tuple[ScratchBufferSpec, ...] = field(repr=False)

    @property
    def scratch_nbytes(self) -> int:
        return int(self._arena_layout.nbytes)

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._scratch_specs

    def topk_sum_launch(self, ids_dtype: torch.dtype) -> W4A16TopKSumCompileResult:
        for launch in self.identity_sums:
            if launch.route_ids_dtype == ids_dtype:
                return launch
        raise TypeError(f"mixed Trellis plan has no top-k sum for {ids_dtype}")

    def fused_launch(self, tokens: int) -> W4A16FusedMoeFullRotationHybridCompileResult:
        tokens = int(tokens)
        for planned_tokens, launch in self.fused_launches:
            if planned_tokens == tokens:
                return launch
        # Small decode plans contain an exact launch for every admitted M.
        # Large prefill plans intentionally retain one capacity launch, just
        # like r12's stock Trellis scratch plan.  That launch accepts any live
        # M covered by its caller-owned arena.
        for planned_tokens, launch in self.fused_launches:
            if planned_tokens >= tokens:
                return launch
        raise ValueError(
            f"mixed Trellis plan has no launch covering {tokens} tokens; "
            f"planned={[m for m, _ in self.fused_launches]}"
        )


@dataclass(frozen=True, kw_only=True)
class MixedTrellisMoEBinding:
    plan: MixedTrellisMoEPlan
    tier0_weights: TrellisMoEWeights
    tier1_weights: TrellisMoEWeights
    a: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    tier_local_map: torch.Tensor
    global_suh_gate: torch.Tensor
    global_suh_up: torch.Tensor
    global_intermediate_rotations: torch.Tensor
    global_svh_down: torch.Tensor
    output: torch.Tensor
    prepared_tier0: object = field(repr=False)
    prepared_tier1: object = field(repr=False)
    intermediate_cache13: torch.Tensor = field(repr=False)
    intermediate_cache2: torch.Tensor = field(repr=False)
    fc1_c_tmp: torch.Tensor = field(repr=False)
    fc2_c_tmp: torch.Tensor = field(repr=False)
    packed_route_indices: torch.Tensor = field(repr=False)
    block_expert_ids: torch.Tensor = field(repr=False)
    packed_route_count: torch.Tensor = field(repr=False)
    expert_offsets: torch.Tensor = field(repr=False)
    expert_counts: torch.Tensor = field(repr=False)
    rotation_a_gate: torch.Tensor = field(repr=False)
    rotation_a_up: torch.Tensor = field(repr=False)
    fused_launch: W4A16FusedMoeFullRotationHybridCompileResult = field(repr=False)
    topk_sum_launch: W4A16TopKSumCompileResult = field(repr=False)


def plan_mixed_trellis_moe(
    caps: TrellisMoECaps,
    *,
    tier0_weights: TrellisMoEWeights,
    tier1_weights: TrellisMoEWeights,
) -> MixedTrellisMoEPlan:
    """Compile the fused K3/K4 launch and one global full-rotation sum."""

    if not isinstance(caps, TrellisMoECaps):
        raise TypeError("caps must be TrellisMoECaps")
    device = _resolve_cuda_device(caps.device)
    if device != caps.device:
        caps = replace(caps, device=device)
    tiers = (tier0_weights, tier1_weights)
    if any(not isinstance(weights, TrellisMoEWeights) for weights in tiers):
        raise TypeError("tier weights must come from trellis_moe.prepare_weights")
    if tier0_weights.device != device or tier1_weights.device != device:
        raise ValueError("tier weights must be on the planned CUDA device")
    if (
        tier0_weights.hidden_size != caps.hidden_size
        or tier1_weights.hidden_size != caps.hidden_size
        or tier0_weights.intermediate_size != caps.intermediate_size
        or tier1_weights.intermediate_size != caps.intermediate_size
    ):
        raise ValueError("mixed tier dimensions do not match caps")
    if tier0_weights.tile_config != tier1_weights.tile_config:
        raise ValueError("mixed tiers require an identical tile_config")
    if tier0_weights.tile_config != caps.tile_config:
        raise ValueError("mixed tier tile_config does not match caps")
    if tier0_weights.trellis_bits == tier1_weights.trellis_bits:
        raise ValueError("mixed tiers must use distinct Trellis bitrates")
    if tier0_weights.num_experts + tier1_weights.num_experts != caps.num_experts:
        raise ValueError("mixed tier expert counts must exactly cover caps.num_experts")
    assert caps.route_num_experts is not None
    if caps.route_num_experts != caps.num_experts:
        raise ValueError(
            "fused mixed-K routes must use the global expert count directly"
        )

    with torch.cuda.device(device):
        props = torch.cuda.get_device_properties(device)
        sms = int(props.multi_processor_count)
        max_shared_mem = int(getattr(props, "shared_memory_per_block_optin", 101_376))
        buffer_plan = plan_w4a16_buffers(
            caps,
            m=caps.max_tokens,
            topk=caps.num_topk,
            route_num_experts=caps.route_num_experts,
            sms=sms,
            full_rotation=True,
            block_size_m=caps.block_size_m,
        )
        route_slots = max_packed_route_slots(
            caps.max_tokens * caps.num_topk,
            caps.block_size_m,
            caps.route_num_experts,
        )
        route_blocks = (route_slots + caps.block_size_m - 1) // caps.block_size_m
        # Match r12's consolidated full-rotation contract exactly: small decode
        # gets an exact live-M specialization, while every specialization keeps
        # the full caller-owned route-arena capacity.  The retired compatibility
        # wrapper instead reused one M=32 kernel (v8) or shrank max_m_blocks with
        # each exact M (v6); neither is the production r12 launch contract.
        fused_token_counts = (
            tuple(range(1, int(caps.max_tokens) + 1))
            if int(caps.max_tokens) <= 32
            else (int(caps.max_tokens),)
        )
        fused_launches = tuple(
            (
                token_count,
                compile_w4a16_fused_moe_full_rotation_hybrid(
                    size_m=token_count,
                    route_capacity_m_blocks=route_blocks,
                    hidden_size=caps.hidden_size,
                    intermediate_size=caps.intermediate_size,
                    tier0_num_experts=tier0_weights.num_experts,
                    tier0_trellis_bits=tier0_weights.trellis_bits,
                    tier1_num_experts=tier1_weights.num_experts,
                    tier1_trellis_bits=tier1_weights.trellis_bits,
                    top_k=caps.num_topk,
                    activation=caps.activation,
                    map_slots=caps.num_experts,
                    moe_block_size=caps.block_size_m,
                    rotation_input_dtype=_input_dtype_name(caps.input_dtype),
                    fast_math=caps.fast_math,
                    sms=sms,
                    max_shared_mem=max_shared_mem,
                    force_tile_config=caps.tile_config,
                ),
            )
            for token_count in fused_token_counts
        )
        if any(
            int(launch.max_m_blocks) < int(route_blocks) for _, launch in fused_launches
        ):
            raise RuntimeError("compiled mixed-K launch under-covers route capacity")
        identity_sums = tuple(
            compile_w4a16_topk_sum(
                m=caps.max_tokens,
                topk=caps.num_topk,
                hidden_size=caps.hidden_size,
                element_dtype="fp16",
                full_rotation=True,
                num_experts=caps.num_experts,
                route_num_experts=0,
                route_ids_dtype=ids_dtype,
                use_expert_map=False,
            )
            for ids_dtype in (torch.int32, torch.int64)
        )

    arena_layout = _make_arena_layout(caps, buffer_plan, sms=sms)
    specs = (
        scratch_buffer_spec(
            "mixed_trellis_moe",
            nbytes=arena_layout.nbytes,
            device=device,
        ),
    )
    return MixedTrellisMoEPlan(
        caps=caps,
        tier0_num_experts=tier0_weights.num_experts,
        tier0_trellis_bits=tier0_weights.trellis_bits,
        tier1_num_experts=tier1_weights.num_experts,
        tier1_trellis_bits=tier1_weights.trellis_bits,
        buffer_plan=buffer_plan,
        fused_launches=fused_launches,
        identity_sums=identity_sums,
        _arena_layout=arena_layout,
        _scratch_specs=specs,
    )


def _validate_global_table(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if (
        tensor.dtype != torch.float16
        or tensor.device != device
        or tuple(tensor.shape) != shape
        or not tensor.is_contiguous()
    ):
        raise ValueError(
            f"{name} must be contiguous fp16 {shape} on {device}; got "
            f"{tuple(tensor.shape)}/{tensor.dtype}/{tensor.device}"
        )


def bind_mixed_trellis_moe(
    plan: MixedTrellisMoEPlan,
    *,
    scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    a: torch.Tensor,
    tier0_weights: TrellisMoEWeights,
    tier1_weights: TrellisMoEWeights,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    tier_local_map: torch.Tensor,
    global_suh_gate: torch.Tensor,
    global_suh_up: torch.Tensor,
    global_intermediate_rotations: torch.Tensor,
    global_svh_down: torch.Tensor,
    output: torch.Tensor | None = None,
) -> MixedTrellisMoEBinding:
    if not isinstance(plan, MixedTrellisMoEPlan):
        raise TypeError("plan must come from plan_mixed_trellis_moe")
    caps = plan.caps
    tokens = int(a.shape[0])
    _validate_runtime_tensor(
        "a",
        a,
        shape=(tokens, caps.hidden_size),
        dtype=caps.input_dtype,
        device=caps.device,
    )
    if tokens < 1 or tokens > caps.max_tokens:
        raise ValueError("input token count exceeds mixed Trellis plan")
    _validate_runtime_tensor(
        "topk_weights",
        topk_weights,
        shape=(tokens, caps.num_topk),
        dtype=torch.float32,
        device=caps.device,
    )
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("topk_ids must be int32 or int64")
    _validate_runtime_tensor(
        "topk_ids",
        topk_ids,
        shape=(tokens, caps.num_topk),
        dtype=topk_ids.dtype,
        device=caps.device,
    )
    if (
        tier_local_map.dtype != torch.int32
        or tier_local_map.device != caps.device
        or tuple(tier_local_map.shape) != (caps.num_experts,)
        or not tier_local_map.is_contiguous()
    ):
        raise ValueError("tier_local_map must be contiguous int32 [global_num_experts]")
    _validate_global_table(
        "global_suh_gate",
        global_suh_gate,
        shape=(caps.num_experts, caps.hidden_size),
        device=caps.device,
    )
    _validate_global_table(
        "global_suh_up",
        global_suh_up,
        shape=(caps.num_experts, caps.hidden_size),
        device=caps.device,
    )
    _validate_global_table(
        "global_svh_down",
        global_svh_down,
        shape=(caps.num_experts, caps.hidden_size),
        device=caps.device,
    )
    _validate_global_table(
        "global_intermediate_rotations",
        global_intermediate_rotations,
        shape=(caps.num_experts, 3 * caps.intermediate_size),
        device=caps.device,
    )
    for name, weights, expected_experts, expected_bits in (
        (
            "tier0_weights",
            tier0_weights,
            plan.tier0_num_experts,
            plan.tier0_trellis_bits,
        ),
        (
            "tier1_weights",
            tier1_weights,
            plan.tier1_num_experts,
            plan.tier1_trellis_bits,
        ),
    ):
        if (
            weights.num_experts != expected_experts
            or weights.trellis_bits != expected_bits
            or weights.device != caps.device
        ):
            raise ValueError(f"{name} does not match the mixed Trellis plan")

    storage = scratch_tensor(scratch, plan._scratch_specs, owner="mixed Trellis MoE")
    if int(storage.data_ptr()) % _ARENA_ALIGNMENT != 0:
        raise ValueError(
            f"mixed Trellis scratch must be {_ARENA_ALIGNMENT}-byte aligned"
        )
    views = plan._arena_layout.materialize(storage)
    views["kernel_workspace"].zero_()
    if output is None:
        output_view = views["output"][:tokens]
    else:
        if (
            output.dtype != torch.float32
            or output.device != caps.device
            or not output.is_contiguous()
            or tuple(output.shape)
            not in (
                (tokens, caps.hidden_size),
                (caps.max_tokens, caps.hidden_size),
            )
        ):
            raise ValueError("output must be a live/capacity contiguous FP32 view")
        output_view = output[:tokens]
    prepared0 = replace(tier0_weights._prepared, workspace=views["kernel_workspace"])
    prepared1 = replace(tier1_weights._prepared, workspace=views["kernel_workspace"])
    return MixedTrellisMoEBinding(
        plan=plan,
        tier0_weights=tier0_weights,
        tier1_weights=tier1_weights,
        a=a,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        tier_local_map=tier_local_map,
        global_suh_gate=global_suh_gate,
        global_suh_up=global_suh_up,
        global_intermediate_rotations=global_intermediate_rotations,
        global_svh_down=global_svh_down,
        output=output_view,
        prepared_tier0=prepared0,
        prepared_tier1=prepared1,
        intermediate_cache13=views["intermediate_cache13"],
        intermediate_cache2=views["intermediate_cache2"],
        fc1_c_tmp=views["fc1_c_tmp"],
        fc2_c_tmp=views["fc2_c_tmp"],
        packed_route_indices=views["packed_route_indices"],
        block_expert_ids=views["block_expert_ids"],
        packed_route_count=views["packed_route_count"],
        expert_offsets=views["expert_offsets"],
        expert_counts=views["expert_counts"],
        rotation_a_gate=views["rotation_a_gate"],
        rotation_a_up=views["rotation_a_up"],
        fused_launch=plan.fused_launch(tokens),
        topk_sum_launch=plan.topk_sum_launch(topk_ids.dtype),
    )


def run_mixed_trellis_moe(*, binding: MixedTrellisMoEBinding) -> torch.Tensor:
    if not isinstance(binding, MixedTrellisMoEBinding):
        raise TypeError("binding must come from bind_mixed_trellis_moe")
    caps = binding.plan.caps
    return run_w4a16_moe_full_rotation_hybrid(
        binding.a,
        binding.prepared_tier0,
        binding.prepared_tier1,
        binding.topk_weights,
        binding.topk_ids,
        binding.tier_local_map,
        activation=caps.activation,
        intermediate_cache13=binding.intermediate_cache13,
        intermediate_cache2=binding.intermediate_cache2,
        output=binding.output,
        fc1_c_tmp=binding.fc1_c_tmp,
        fc2_c_tmp=binding.fc2_c_tmp,
        packed_route_indices=binding.packed_route_indices,
        block_expert_ids=binding.block_expert_ids,
        packed_route_count=binding.packed_route_count,
        expert_offsets=binding.expert_offsets,
        expert_counts=binding.expert_counts,
        rotation_a_gate=binding.rotation_a_gate,
        rotation_a_up=binding.rotation_a_up,
        global_intermediate_rotations=binding.global_intermediate_rotations,
        global_suh_gate=binding.global_suh_gate,
        global_suh_up=binding.global_suh_up,
        global_svh_down=binding.global_svh_down,
        fused_launch=binding.fused_launch,
        topk_sum_launch=binding.topk_sum_launch,
        fast_math=caps.fast_math,
    )


__all__ = [
    "MixedTrellisMoEBinding",
    "MixedTrellisMoEPlan",
    "bind_mixed_trellis_moe",
    "plan_mixed_trellis_moe",
    "run_mixed_trellis_moe",
]
