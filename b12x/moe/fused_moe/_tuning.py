"""Configuration contract for fused-MoE decode preparation."""

from __future__ import annotations

from dataclasses import dataclass, replace

from b12x.preparation import (
    DeviceIdentity,
    FrozenMapping,
    Knob,
    ParameterBinding,
    TuningContract,
    make_fixed_contract,
)


@dataclass(frozen=True)
class MoeDecodeQuery:
    """Numerical and lowering contract for a planned token capacity."""
    quant_mode: str
    quant_modes: tuple[str, ...]
    source_format: str
    activation: str
    io_dtype: str
    num_experts: int
    hidden_size: int
    intermediate_size: int
    top_k: int
    num_tokens: int
    routed_rows: int
    route_num_experts: int | None
    route_logits_dtype: str | None
    apply_router_weight_on_input: bool
    collect_activation_amax: bool
    deterministic_output: bool | None
    swiglu_limit: object
    swiglu_alpha: object
    swiglu_beta: object
    w13_layout: str
    weight_layouts: tuple[str, ...]
    w4a16_weight_layout: str | None
    w4a16_scale_format: str | None
    w4a16_block_size_m: int | None
    fast_math: bool
    numerical_recipe: str | None
    controls: FrozenMapping
    shared_input_scales: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "activation": self.activation,
            "apply_router_weight_on_input": self.apply_router_weight_on_input,
            "collect_activation_amax": self.collect_activation_amax,
            "controls": self.controls.to_dict(),
            "shared_input_scales": self.shared_input_scales,
            "deterministic_output": self.deterministic_output,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "io_dtype": self.io_dtype,
            "num_experts": self.num_experts,
            "num_tokens": self.num_tokens,
            "quant_mode": self.quant_mode,
            "quant_modes": self.quant_modes,
            "route_logits_dtype": self.route_logits_dtype,
            "route_num_experts": self.route_num_experts,
            "routed_rows": self.routed_rows,
            "source_format": self.source_format,
            "swiglu_alpha": self.swiglu_alpha,
            "swiglu_beta": self.swiglu_beta,
            "swiglu_limit": self.swiglu_limit,
            "numerical_recipe": self.numerical_recipe,
            "top_k": self.top_k,
            "w13_layout": self.w13_layout,
            "w4a16_scale_format": self.w4a16_scale_format,
            "w4a16_weight_layout": self.w4a16_weight_layout,
            "w4a16_block_size_m": self.w4a16_block_size_m,
            "fast_math": self.fast_math,
            "weight_layouts": self.weight_layouts,
        }


@dataclass(frozen=True)
class MoeDecodeConfig:
    backend: str
    route_planner: str
    max_active_clusters: int | None
    dynamic_tile_m: int | None = None
    dynamic_route_mode: str | None = None
    w4a16_route_mode: str | None = None
    nvfp4_share_input: bool = False
    nvfp4_materialize_intermediate: bool = False

    @classmethod
    def from_config(cls, payload: FrozenMapping) -> "MoeDecodeConfig":
        expected = {
            "backend",
            "dynamic_route_mode",
            "dynamic_tile_m",
            "max_active_clusters",
            "route_planner",
            "w4a16_route_mode",
            "nvfp4_share_input",
            "nvfp4_materialize_intermediate",
        }
        if set(payload) != expected:
            raise ValueError(
                "MoE decode config fields must be exactly "
                f"{sorted(expected)}; got {sorted(payload)}"
            )
        backend = payload["backend"]
        route_planner = payload["route_planner"]
        max_active_clusters = payload["max_active_clusters"]
        dynamic_tile_m = payload["dynamic_tile_m"]
        dynamic_route_mode = payload["dynamic_route_mode"]
        w4a16_route_mode = payload["w4a16_route_mode"]
        if not isinstance(backend, str) or not isinstance(route_planner, str):
            raise TypeError("MoE backend and route_planner must be strings")
        if max_active_clusters is not None and (
            not isinstance(max_active_clusters, int)
            or isinstance(max_active_clusters, bool)
        ):
            raise TypeError("max_active_clusters must be an integer or null")
        if dynamic_tile_m is not None and (
            not isinstance(dynamic_tile_m, int) or isinstance(dynamic_tile_m, bool)
        ):
            raise TypeError("dynamic_tile_m must be an integer or null")
        if dynamic_route_mode is not None and not isinstance(dynamic_route_mode, str):
            raise TypeError("dynamic_route_mode must be a string or null")
        if w4a16_route_mode is not None and not isinstance(w4a16_route_mode, str):
            raise TypeError("w4a16_route_mode must be a string or null")
        if any(type(payload[name]) is not bool for name in (
            "nvfp4_share_input", "nvfp4_materialize_intermediate",
        )):
            raise TypeError("NVFP4 lowering controls must be boolean")
        return cls(
            backend=backend,
            route_planner=route_planner,
            max_active_clusters=max_active_clusters,
            dynamic_tile_m=dynamic_tile_m,
            dynamic_route_mode=dynamic_route_mode,
            w4a16_route_mode=w4a16_route_mode,
            nvfp4_share_input=payload["nvfp4_share_input"],
            nvfp4_materialize_intermediate=payload["nvfp4_materialize_intermediate"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "route_planner": self.route_planner,
            "max_active_clusters": self.max_active_clusters,
            "dynamic_tile_m": self.dynamic_tile_m,
            "dynamic_route_mode": self.dynamic_route_mode,
            "w4a16_route_mode": self.w4a16_route_mode,
            "nvfp4_share_input": self.nvfp4_share_input,
            "nvfp4_materialize_intermediate": self.nvfp4_materialize_intermediate,
        }


def _validate_query(query: MoeDecodeQuery, _device: DeviceIdentity | None) -> None:
    if not isinstance(query, MoeDecodeQuery):
        raise TypeError("query must be MoeDecodeQuery")
    if not isinstance(query.controls, FrozenMapping):
        raise TypeError("MoE controls must be frozen declaration metadata")
    if query.io_dtype not in {"bfloat16", "float16"}:
        raise TypeError("MoE I/O dtype must be bfloat16 or float16")
    if query.route_logits_dtype not in {None, "float16", "bfloat16", "float32"}:
        raise TypeError("MoE router logits dtype is unsupported")
    if query.numerical_recipe is not None and not isinstance(query.numerical_recipe, str):
        raise TypeError("MoE numerical_recipe must be a string or null")
    if type(query.shared_input_scales) is not bool:
        raise TypeError("shared_input_scales must be boolean")
    if not query.weight_layouts:
        raise ValueError("MoE query requires declared weight layouts")

def _nvfp4_query(query):
    return query.quant_mode in {"nvfp4", "nvfp4_auto"} or (
        query.quant_mode == "multi" and "nvfp4" in query.quant_modes
    )


def _nvfp4_materialization_eligible(query, config):
    tile = query.controls.get("dynamic_tile_mn")
    return bool(
        _nvfp4_query(query)
        and config.backend == "dynamic"
        and config.dynamic_tile_m == 128
        and (tile is None or tuple(tile) == (128, 128))
        and query.activation == "silu"
        and not query.deterministic_output
        and query.hidden_size % 128 == 0
        and query.intermediate_size % 128 == 0
        and query.controls.get("dynamic_work_source", "materialized_queue") != "ready_queue"
        and not query.controls.get("dynamic_down_scale", False)
        and query.controls.get("dynamic_swap_ab") in (None, "0")
        and query.shared_input_scales
        and config.nvfp4_share_input
    )


def validate_moe_decode_config(
    query: MoeDecodeQuery,
    config: MoeDecodeConfig,
    _device: DeviceIdentity | None,
) -> None:
    from ._backends import architecture_backend

    backend = architecture_backend(_device)
    if backend is not None:
        backend.validate_policy(query, config)
        return
    if config.backend.startswith("tcgen05_"):
        raise ValueError("tcgen05 MoE requires an SM103 device")
    if query.quant_mode == "multi":
        if config.backend == "w4a16":
            if "w4a16" not in query.quant_modes:
                raise ValueError("W4A16 backend is absent from the declared MoE recipes")
            query = replace(query, quant_mode="w4a16")
        else:
            non_a16 = tuple(mode for mode in query.quant_modes if mode != "w4a16")
            if len(non_a16) != 1:
                raise ValueError("non-W4A16 multi-recipe MoE requires an explicit recipe route")
            query = replace(query, quant_mode=non_a16[0])
    if query.quant_mode == "nvfp4_auto":
        if query.source_format != "modelopt_nvfp4" or query.activation != "silu":
            raise ValueError(
                "automatic MoE precision requires ModelOpt NVFP4 weights and SiLU"
            )
        if (
            config.backend == "w4a16"
            and config.w4a16_route_mode == "direct"
            and query.num_tokens > 8
        ):
            raise ValueError(
                "source-native A16 direct decode requires capacity at most 8"
            )
        query = replace(query, quant_mode="w4a16" if config.backend == "w4a16" else "nvfp4")
    if any(type(value) is not bool for value in (
        config.nvfp4_share_input, config.nvfp4_materialize_intermediate,
    )):
        raise TypeError("NVFP4 lowering controls must be boolean")
    if config.nvfp4_share_input and not (
        query.quant_mode == "nvfp4" and config.backend == "dynamic" and query.shared_input_scales
    ):
        raise ValueError("shared NVFP4 input requires validated uniform input scales and dynamic execution")
    if config.nvfp4_materialize_intermediate and not _nvfp4_materialization_eligible(query, config):
        raise ValueError("NVFP4 split materialization requires shared input and the nondeterministic SiLU M128 contract")
    if (
        query.quant_mode == "w4a8_mx" and query.source_format == "fp4_e8m0_k32"
        and query.intermediate_size % 128 == 64 and config.backend == "dynamic"
        and (config.dynamic_tile_m != 16 or config.dynamic_route_mode != "grouped"
             or config.route_planner != "internal")
    ):
        raise ValueError("compact N64 W4A8 dynamic execution requires grouped internal M16 routing")
    if config.backend not in {"micro", "dynamic", "w4a16"}:
        raise ValueError(f"unsupported MoE backend {config.backend!r}")
    if query.quant_mode == "w4a16":
        if config.backend != "w4a16":
            raise ValueError("W4A16 queries require the W4A16 backend")
        if config.w4a16_route_mode not in {"direct", "packed"}:
            raise ValueError("W4A16 route mode must be 'direct' or 'packed'")
    else:
        if config.backend == "w4a16":
            raise ValueError("the W4A16 backend requires quant_mode='w4a16'")
        if config.w4a16_route_mode is not None:
            raise ValueError("w4a16_route_mode is only valid for W4A16")
    if query.quant_mode == "w6a8_mx" and config.backend != "dynamic":
        raise ValueError("W6A8-MX queries require the dynamic backend")
    if config.route_planner not in {"internal", "triton"}:
        raise ValueError(f"unsupported MoE route planner {config.route_planner!r}")
    if config.route_planner == "triton" and config.backend != "dynamic":
        raise ValueError("the Triton route planner requires dynamic MoE")
    if config.route_planner == "triton" and config.dynamic_route_mode != "grouped":
        raise ValueError("the Triton route planner requires grouped dynamic routing")
    if config.route_planner == "triton" and config.dynamic_tile_m != 16:
        raise ValueError("the Triton route planner requires dynamic_tile_m=16")
    if config.route_planner == "triton" and not (
        query.quant_mode == "nvfp4" and query.activation == "silu" and 0 < query.routed_rows <= 256
    ):
        raise ValueError("the Triton route planner only supports small NVFP4 SiLU workloads")
    if config.max_active_clusters is not None and config.max_active_clusters <= 0:
        raise ValueError("max_active_clusters must be positive when set")
    if config.route_planner != "triton" and config.max_active_clusters is not None:
        raise ValueError("max_active_clusters requires the Triton route planner")
    if config.backend == "dynamic":
        if config.dynamic_tile_m not in {16, 32, 64, 128}:
            raise ValueError("dynamic_tile_m must be one of 16, 32, 64, 128")
        if config.dynamic_route_mode not in {"direct", "grouped"}:
            raise ValueError("dynamic_route_mode must be 'direct' or 'grouped'")
    elif config.dynamic_tile_m is not None:
        raise ValueError("dynamic_tile_m is only valid for dynamic MoE")
    elif config.dynamic_route_mode is not None:
        raise ValueError("dynamic_route_mode is only valid for dynamic MoE")


def _default_config(query: MoeDecodeQuery, device: DeviceIdentity | None) -> MoeDecodeConfig:
    from ._backends import architecture_backend

    backend = architecture_backend(device)
    if backend is not None:
        return backend.heuristic(query)
    from ._impl import _heuristic_moe_decode_config

    if query.quant_mode == "multi":
        non_a16 = tuple(mode for mode in query.quant_modes if mode != "w4a16")
        if len(non_a16) != 1:
            raise ValueError("multi-recipe MoE requires an explicit default recipe route")
        query = replace(query, quant_mode=non_a16[0])
    config = _heuristic_moe_decode_config(query, device)
    config = replace(config, nvfp4_share_input=bool(
        _nvfp4_query(query) and config.backend == "dynamic" and query.shared_input_scales
    ))
    return replace(config, nvfp4_materialize_intermediate=bool(
        _nvfp4_materialization_eligible(query, config)
        and query.controls.get("dynamic_nvfp4_materialized") is not False
    ))


def _materialize_tuning(query, device, choice):
    """Apply the same concrete route eligibility used by production resolution."""
    from ._impl import (
        _dynamic_direct_routing_selected,
        _dynamic_kernel_intermediate_size,
        _policy_micro_supported,
        _w4a16_direct_routing_supported,
    )

    config = MoeDecodeConfig.from_config(choice)
    validate_moe_decode_config(query, config, device)
    if config.backend.startswith("tcgen05_"):
        return config
    if (
        min(query.num_experts, query.hidden_size, query.intermediate_size, query.top_k, query.num_tokens) <= 0
        or query.top_k > query.num_experts
        or query.routed_rows != query.num_tokens * query.top_k
    ):
        raise ValueError("MoE tuning requires consistent concrete routing geometry")
    effective_query = query
    if query.quant_mode == "multi":
        if config.backend == "w4a16":
            effective_query = replace(query, quant_mode="w4a16")
        else:
            non_a16 = tuple(mode for mode in query.quant_modes if mode != "w4a16")
            if len(non_a16) != 1:
                raise ValueError("non-W4A16 multi-recipe MoE requires an explicit recipe route")
            effective_query = replace(query, quant_mode=non_a16[0])
    elif query.quant_mode == "nvfp4_auto" and config.backend != "w4a16":
        effective_query = replace(query, quant_mode="nvfp4")
    if (config.backend == "micro" and effective_query.quant_mode == "w4a8_mx"
            and query.intermediate_size % 128 == 64 and query.io_dtype != "bfloat16"):
        raise ValueError("compact W4A8 micro requires BF16 activations")
    if config.backend == "micro" and not _policy_micro_supported(effective_query):
        raise ValueError("micro MoE does not support this concrete query")
    if config.backend == "dynamic":
        if effective_query.quant_mode == "w6a8_mx" and config.dynamic_tile_m != 128:
            raise ValueError("W6A8-MX requires the production M128 dynamic tile")
        try:
            _dynamic_direct_routing_selected(
                route_mode=config.dynamic_route_mode,
                quant_mode=effective_query.quant_mode,
                activation=query.activation,
                routed_rows=query.routed_rows,
                num_experts=query.num_experts,
                n=_dynamic_kernel_intermediate_size(query.intermediate_size, effective_query.quant_mode),
                deterministic_output=False,
                planned_tile_m=config.dynamic_tile_m,
            )
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
    if config.backend == "w4a16" and config.w4a16_route_mode == "direct" and not _w4a16_direct_routing_supported(query):
        raise ValueError("W4A16 direct routing does not support this concrete query")
    return config


def _tuning_parameters(query, device):
    from ._backends import architecture_backend

    backend = architecture_backend(device)
    if backend is not None:
        config = backend.heuristic(query)
        return {"backend": (config.backend,), "route_planner": ("internal",),
                "max_active_clusters": (None,)}
    if device is None or device.sm_count <= 0:
        raise ValueError("MoE launch tuning requires the device SM count")
    from ._impl import (
        _LEVEL_TILE_N,
        _dynamic_kernel_intermediate_size,
        _dynamic_task_geometry,
    )

    # The Triton planner is legal only for grouped NVFP4 dynamic routing at tile
    # M16. Its grid is clamped to the resident SMs, and a cluster beyond the
    # planned task queue takes no task and only joins the resident-grid barrier.
    tasks = _dynamic_task_geometry(
        query.num_experts,
        _dynamic_kernel_intermediate_size(query.intermediate_size, "nvfp4"),
        query.routed_rows,
        tile_m=16,
        tile_n=_LEVEL_TILE_N,
    )[2]
    clamp = min(device.sm_count, tasks)
    # Grid width is a runtime-only knob: the powers of two up to that clamp,
    # plus the clamp itself, bracket every resident grid within a factor of two.
    ladder = sorted({1 << exponent for exponent in range(clamp.bit_length())} | {clamp})
    return {"max_active_clusters": (None, *ladder)}


@dataclass(frozen=True, kw_only=True)
class MoeRouteQuery:
    """Immutable ABI for the native top-k routing program."""
    num_tokens: int
    num_experts: int
    top_k: int
    logits_dtype: str
    score_func: str
    renormalize: bool
    has_correction_bias: bool
    has_image_correction_bias: bool
    has_image_mask: bool
    routed_scaling_factor: float
    logits_row_stride: int
    topk_row_stride: int
    ids_row_stride: int
    weights_row_stride: int


@dataclass(frozen=True, kw_only=True)
class MoeFC2Query:
    """Immutable ABI for the standalone W4A16 FC2 launch."""
    max_routes: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    route_ids_dtype: str


def _validate_route_query(query: MoeRouteQuery, _device) -> None:
    if min(query.num_tokens, query.num_experts, query.top_k) <= 0:
        raise ValueError("native route preparation requires positive geometry")
    if query.num_experts > 1024 or query.top_k > query.num_experts:
        raise ValueError("native route preparation supports top-k within 1024 experts")
    if query.logits_dtype not in {"float16", "bfloat16", "float32"}:
        raise TypeError("router logits must be float16, bfloat16, or float32")
    if query.score_func not in {"softmax", "sqrtsoftplus"}:
        raise ValueError("unsupported router score function")
    if query.has_image_correction_bias != query.has_image_mask:
        raise ValueError("image correction bias and image mask must be supplied together")
    if min(query.logits_row_stride, query.topk_row_stride, query.ids_row_stride,
           query.weights_row_stride) <= 0:
        raise ValueError("route row strides must be positive")


def _validate_fc2_query(query: MoeFC2Query, _device) -> None:
    if min(query.max_routes, query.hidden_size, query.intermediate_size, query.num_experts) <= 0:
        raise ValueError("FC2 preparation requires positive geometry")
    if query.route_ids_dtype not in {"int32", "int64"}:
        raise TypeError("FC2 route IDs must be int32 or int64")


ROUTE_TUNING = make_fixed_contract(
    component_id="moe.route_topk", query_type=MoeRouteQuery, backend="triton"
)
ROUTE_TUNING = replace(ROUTE_TUNING, validate_query=_validate_route_query)

FC2_TUNING = make_fixed_contract(
    component_id="moe.fc2_w4a16", query_type=MoeFC2Query, backend="cutedsl"
)
FC2_TUNING = replace(FC2_TUNING, validate_query=_validate_fc2_query)


TUNING = TuningContract(
    component_id="moe.decode",
    query_schema_version=8,
    config_schema_version=5,
    query_fields=frozenset(MoeDecodeQuery.__dataclass_fields__),
    config_fields=frozenset(MoeDecodeConfig.__dataclass_fields__),
    encode_query=MoeDecodeQuery.to_dict,
    encode_config=MoeDecodeConfig.to_dict,
    decode_config=MoeDecodeConfig.from_config,
    validate_query=_validate_query,
    validate_config=validate_moe_decode_config,
    default_config=_default_config,
    candidate_contract_version=5,
    knobs=(
        Knob(name="backend", values=("micro", "dynamic", "w4a16"), binding=ParameterBinding.COMPILE),
        Knob(name="route_planner", values=("internal", "triton"), binding=ParameterBinding.COMPILE),
        Knob(name="max_active_clusters", values=None, binding=ParameterBinding.RUNTIME, when=FrozenMapping({"route_planner": "triton"})),
        Knob(name="dynamic_tile_m", values=(16, 32, 64, 128), binding=ParameterBinding.COMPILE, when=FrozenMapping({"backend": "dynamic"})),
        Knob(name="dynamic_route_mode", values=("direct", "grouped"), binding=ParameterBinding.COMPILE, when=FrozenMapping({"backend": "dynamic"})),
        Knob(name="nvfp4_share_input", values=(False, True), binding=ParameterBinding.COMPILE, when=FrozenMapping({"backend": "dynamic"}), otherwise=False),
        Knob(name="nvfp4_materialize_intermediate", values=(False, True), binding=ParameterBinding.COMPILE, when=FrozenMapping({"nvfp4_share_input": True}), otherwise=False),
        Knob(name="w4a16_route_mode", values=("direct", "packed"), binding=ParameterBinding.COMPILE, when=FrozenMapping({"backend": "w4a16"})),
    ),
    materialize=_materialize_tuning,
    parameters=_tuning_parameters,
)


__all__ = [
    "TUNING", "ROUTE_TUNING", "FC2_TUNING", "MoeDecodeConfig",
    "MoeDecodeQuery", "MoeRouteQuery", "MoeFC2Query",
    "validate_moe_decode_config",
]
