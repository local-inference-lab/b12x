"""Native NVFP4 MoE planning and binding for the SM103 architecture.

Status: implemented, awaiting B300 runtime qualification. The materialized
pipeline owns no model-sized repack and resolves kernels only at prewarm.
"""

from dataclasses import dataclass, replace
from enum import Enum

import torch

from b12x._lib.architecture import UnsupportedArchitectureError
from b12x._lib.scratch import scratch_buffer_spec, scratch_tensor
from b12x._lib.scratch_layout import align_up, materialize_scratch_view
from b12x.policy import PolicyResolution
from .._shared.execution import (
    MoEExecutionPlan,
    MoERegime,
    RouteLayout,
    WorkAvailability,
    WorkScheduler,
    GraphPartition,
    GemmEngine,
    PreparedWeightLayout,
    OutputReduction,
    make_moe_spec,
)
from ._policy import MoeDecodeQuery, MoeDecodeConfig, MOE_DECODE_POLICY

BACKEND = "tcgen05_nvfp4"


class Strategy(str, Enum):
    MATERIALIZED = "split_materialized"
    MONOLITHIC = "monolithic"
    TMEM_PIPELINED = "tmem_pipelined"


def capacity_regime(tokens: int) -> str:
    if tokens <= 0:
        raise ValueError("capacity must be positive")
    if tokens == 1:
        return "m1"
    if tokens <= 4:
        return "m2_m4"
    if tokens <= 8:
        return "m5_m8"
    if tokens <= 32:
        return "m9_m32"
    return "prefill"


def heuristic(query):
    if query.source_format in {"b12x_trellis", "btx"}:
        from ._sm103_trellis import BACKEND as trellis_backend

        config = MoeDecodeConfig(trellis_backend, "internal", None)
        validate_policy(query, config)
        return config
    config = MoeDecodeConfig(
        backend=BACKEND, route_planner="internal", max_active_clusters=None
    )
    validate_policy(query, config)
    return config


def independent_reference(a, experts, ids, weights):
    """Evaluate the backend's numeric contract for offline qualification."""
    from .._shared.kernels.materialized_nvfp4_reference import reference

    return reference(a, experts, ids, weights)


def validate_policy(query, config):
    if query.source_format in {"b12x_trellis", "btx"}:
        from ._sm103_trellis import validate_policy as validate_trellis

        return validate_trellis(query, config)
    if (
        query.quant_mode not in {"nvfp4", "nvfp4_auto"}
        or query.source_format != "modelopt_nvfp4"
        or query.activation != "silu"
    ):
        raise UnsupportedArchitectureError(
            "SM103 MoE implements ModelOpt NVFP4 A4 SiLU only"
        )
    if query.hidden_size % 256 or query.intermediate_size % 256:
        raise UnsupportedArchitectureError(
            "SM103 NVFP4 projection requires K and N divisible by 256"
        )
    if (
        min(
            query.num_tokens,
            query.top_k,
            query.num_experts,
            query.hidden_size,
            query.intermediate_size,
        )
        <= 0
    ):
        raise ValueError("SM103 MoE geometry and capacity must be positive")
    if (
        query.top_k > query.num_experts
        or query.routed_rows != query.num_tokens * query.top_k
    ):
        raise ValueError("invalid SM103 MoE routing capacity")
    if query.routed_rows > 65535:
        raise UnsupportedArchitectureError(
            "SM103 direct route grid supports at most 65535 planned routes"
        )
    expected = MoeDecodeConfig(
        backend=BACKEND, route_planner="internal", max_active_clusters=None
    )
    if config != expected:
        raise UnsupportedArchitectureError(
            "SM103 MoE requires the tcgen05_nvfp4 materialized backend; "
            "warp-MMA and unimplemented strategy overrides are rejected"
        )


def plan_execution(
    *,
    num_tokens,
    num_topk,
    device,
    weight_plan,
    quant_mode,
    swiglu_limit,
    swiglu_alpha,
    swiglu_beta,
    apply_router_weight_on_input,
    policy_context,
    policy_resolution=None,
):
    from ._impl import TPMoEPlan

    if weight_plan.source_format in {"b12x_trellis", "btx"}:
        from ._sm103_trellis import plan_execution as plan_trellis

        return plan_trellis(
            num_tokens=num_tokens,
            num_topk=num_topk,
            device=device,
            weight_plan=weight_plan,
            quant_mode=quant_mode,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            apply_router_weight_on_input=apply_router_weight_on_input,
            policy_context=policy_context,
            policy_resolution=policy_resolution,
        )
    if weight_plan.io_dtype != "bfloat16":
        raise UnsupportedArchitectureError("SM103 NVFP4 MoE requires BF16 I/O")
    if apply_router_weight_on_input:
        raise UnsupportedArchitectureError(
            "SM103 NVFP4 applies router weights after FC2"
        )
    if swiglu_alpha not in (None, 1.0) or swiglu_beta not in (None, 0.0):
        raise UnsupportedArchitectureError("SM103 NVFP4 implements standard SiLU only")
    query = MoeDecodeQuery(
        quant_mode=quant_mode,
        source_format=weight_plan.source_format,
        activation=weight_plan.activation,
        num_experts=weight_plan.num_experts,
        hidden_size=weight_plan.hidden_size,
        intermediate_size=weight_plan.intermediate_size,
        top_k=num_topk,
        num_tokens=num_tokens,
        routed_rows=num_tokens * num_topk,
    )
    if policy_resolution is None:
        resolution = policy_context.resolve(MOE_DECODE_POLICY, query)
    else:
        if (
            not isinstance(policy_resolution, PolicyResolution)
            or policy_resolution.component_id != MOE_DECODE_POLICY.component_id
            or policy_resolution.device != policy_context.device
        ):
            raise ValueError(
                "MoE policy resolution must match the component and device"
            )
        validate_policy(query, policy_resolution.config)
        resolution = policy_resolution
    spec = make_moe_spec(
        quant_mode="nvfp4",
        source_format=weight_plan.source_format,
        activation=weight_plan.activation,
        io_dtype=weight_plan.io_dtype,
        w13_layout=weight_plan.w13_layout,
    )
    execution = MoEExecutionPlan(
        regime=MoERegime.DIRECT,
        route_layout=RouteLayout.DIRECT_TOPK,
        work_availability=WorkAvailability.INLINE,
        scheduler=WorkScheduler.DIRECT,
        graph_partition=GraphPartition.MATERIALIZED,
        gemm_engine=GemmEngine.NVFP4_TCGEN05,
        weight_layout=PreparedWeightLayout.SOURCE_NATIVE,
        reduction=OutputReduction.ROUTE_BUFFER_TOPK_SUM,
        tile_m=128,
        tile_n=128,
    )
    return TPMoEPlan(
        spec=spec,
        execution=execution,
        implementation=BACKEND,
        quant_mode="nvfp4",
        activation=weight_plan.activation,
        swiglu_limit=swiglu_limit,
        state_E=weight_plan.num_experts,
        weight_E=weight_plan.num_experts,
        routed_rows=query.routed_rows,
        max_rows=num_tokens,
        k=query.hidden_size,
        n=query.intermediate_size,
        num_topk=num_topk,
        device=torch.device(device),
        dtype=torch.bfloat16,
        max_tokens_per_launch=num_tokens,
        policy_resolution=resolution,
    )


@dataclass(frozen=True)
class Buffer:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    offset: int
    nbytes: int


def scratch_layout(tokens, top_k, hidden, intermediate):
    """Return disjoint, aligned buffers; every extent is planned capacity."""
    routes = tokens * top_k
    shapes = (
        ("q1", (routes, hidden // 2), torch.uint8),
        ("s1", (routes, 128 * (hidden // 16)), torch.float8_e4m3fn),
        ("fc1", (routes, 2 * intermediate), torch.bfloat16),
        ("q2", (routes, intermediate // 2), torch.uint8),
        ("s2", (routes, 128 * (intermediate // 16)), torch.float8_e4m3fn),
        ("fc2", (routes, hidden), torch.bfloat16),
        ("output", (tokens, hidden), torch.bfloat16),
    )
    buffers, cursor = [], 0
    for name, shape, dtype in shapes:
        cursor = align_up(cursor, 1024)
        nbytes = shape[0] * shape[1] * dtype.itemsize
        buffers.append(Buffer(name, shape, dtype, cursor, nbytes))
        cursor += nbytes
    return tuple(buffers), align_up(cursor, 1024)


@dataclass(frozen=True)
class BackendPlan:
    buffers: tuple[Buffer, ...]
    launches: dict | None
    strategy: Strategy = Strategy.MATERIALIZED

    def prewarm(self, plan):
        """Compile the retained capacity plan without resolving policy again."""
        if self.launches is not None:
            return plan
        from .._shared.kernels.sm103.launch import compile_launches

        return replace(
            plan,
            _backend_plan=replace(self, launches=compile_launches(plan.caps)),
        )

    def bind(
        self,
        plan,
        *,
        scratch,
        a,
        experts,
        topk_weights,
        topk_ids,
        output,
        activation_amax,
        route_expert_map,
        output_expert_map,
        unit_scale_contract,
    ):
        from ._impl import B12XFP4ExpertWeights, TPMoEFP4Binding
        from .._shared.kernels.sm103.launch import bind_launches

        if self.launches is None:
            raise RuntimeError("SM103 plan must be prewarmed before binding")
        if (
            not isinstance(experts, B12XFP4ExpertWeights)
            or experts.plan != plan.caps.weight_plan
        ):
            raise ValueError("prepared experts do not match the SM103 plan")
        if (
            any(
                x is not None
                for x in (activation_amax, route_expert_map, output_expert_map)
            )
            or unit_scale_contract
        ):
            raise UnsupportedArchitectureError(
                "SM103 MoE does not implement expert maps, A16, or calibration"
            )
        caps = plan.caps
        if a.ndim != 2 or a.shape[1] != caps.k or not 0 < a.shape[0] <= caps.max_tokens:
            raise ValueError("activation shape exceeds the SM103 plan")
        m = a.shape[0]
        for name, tensor, shape, dtypes in (
            ("a", a, (m, caps.k), (torch.bfloat16,)),
            ("topk_ids", topk_ids, (m, caps.num_topk), (torch.int32, torch.int64)),
            ("topk_weights", topk_weights, (m, caps.num_topk), (torch.float32,)),
        ):
            if (
                tensor.shape != shape
                or tensor.dtype not in dtypes
                or tensor.device != caps.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"invalid {name}: expected contiguous {shape} {dtypes} on {caps.device}"
                )
        storage = scratch_tensor(scratch, plan.scratch_specs(), owner="SM103 MoE")
        views = {
            b.name: materialize_scratch_view(
                storage, offset_bytes=b.offset, shape=b.shape, dtype=b.dtype
            )[0]
            for b in self.buffers
        }
        if output is None:
            output = views["output"][:m]
        if (
            output.shape != a.shape
            or output.dtype != a.dtype
            or output.device != a.device
            or not output.is_contiguous()
        ):
            raise ValueError("SM103 output must match contiguous BF16 activations")
        if (
            output.untyped_storage().data_ptr() == storage.untyped_storage().data_ptr()
            and output.data_ptr() != views["output"].data_ptr()
        ):
            raise ValueError("SM103 output may alias only the planned output buffer")
        for tensor in (
            a,
            topk_ids,
            topk_weights,
            experts.w1_fp4,
            experts.w2_fp4,
            experts.w1_blockscale,
            experts.w2_blockscale,
            experts.w1_alphas,
            experts.w2_alphas,
            experts.a1_gscale,
            experts.a2_gscale,
        ):
            if (
                tensor.untyped_storage().data_ptr()
                == storage.untyped_storage().data_ptr()
            ):
                raise ValueError("SM103 scratch must not alias inputs or weights")
            if (
                tensor.untyped_storage().data_ptr()
                == output.untyped_storage().data_ptr()
            ):
                raise ValueError("SM103 output must not alias inputs or weights")
        bound = bind_launches(
            self.launches, caps, a, experts, topk_ids, topk_weights, views, output
        )
        return TPMoEFP4Binding(
            a=a,
            experts=experts,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            implementation=BACKEND,
            state_E=caps.weight_E,
            weight_E=caps.weight_E,
            max_rows=caps.max_tokens,
            k=caps.k,
            n=caps.n,
            num_topk=caps.num_topk,
            device=caps.device,
            dtype=caps.dtype,
            execution_plan=plan.launch_plan,
            output=output,
            quant_mode="nvfp4",
            deterministic_output=True,
            _backend_binding=bound,
        )


def plan_scratch(caps, *, prewarm_launches, policy_resolution=None):
    from ._impl import TPMoEArenaLayout, TPMoEScratchPlan

    if caps.weight_plan.source_format in {"b12x_trellis", "btx"}:
        from ._sm103_trellis import plan_scratch as plan_trellis

        return plan_trellis(
            caps, prewarm_launches=prewarm_launches, policy_resolution=policy_resolution
        )
    if caps.collect_activation_amax or caps.route_logits_dtype is not None:
        raise UnsupportedArchitectureError(
            "SM103 MoE requires preselected top-k routes without calibration"
        )
    if caps.route_num_experts not in (None, 0, caps.weight_E):
        raise UnsupportedArchitectureError("SM103 MoE requires local expert routing")
    if caps.core_token_counts and max(caps.core_token_counts) > caps.max_tokens:
        raise ValueError("warmup counts exceed SM103 capacity")
    launch_plan = plan_execution(
        num_tokens=caps.max_tokens,
        num_topk=caps.num_topk,
        device=caps.device,
        weight_plan=caps.weight_plan,
        quant_mode=caps.quant_mode,
        swiglu_limit=caps.swiglu_limit,
        swiglu_alpha=caps.swiglu_alpha,
        swiglu_beta=caps.swiglu_beta,
        apply_router_weight_on_input=caps.apply_router_weight_on_input,
        policy_context=caps.policy_context,
        policy_resolution=policy_resolution,
    )
    buffers, nbytes = scratch_layout(caps.max_tokens, caps.num_topk, caps.k, caps.n)
    plan = TPMoEScratchPlan(
        caps=caps,
        layout=TPMoEArenaLayout(
            route_workspace_nbytes=0,
            core_workspace_nbytes=nbytes,
            total_nbytes=nbytes,
            core_token_counts=caps.core_token_counts or (caps.max_tokens,),
        ),
        launch_plan=launch_plan,
        _core_workspace_plan=None,
        _scratch_specs=(
            scratch_buffer_spec("tp_moe.scratch", nbytes=nbytes, device=caps.device),
        ),
        _backend_plan=BackendPlan(buffers, launches=None),
    )
    return plan._backend_plan.prewarm(plan) if prewarm_launches else plan
