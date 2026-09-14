"""Planned native SM103 execution for uniform and projection-tiered Trellis.

The materialized schedule retains compressed weights and FP16 transform
boundaries. All kernels are resolved during prewarm; bind retains pointers
and runtime counts, and replay performs no allocation or policy lookup.
"""

from dataclasses import dataclass, replace
from math import prod

import torch

from b12x._lib.architecture import UnsupportedArchitectureError
from b12x._lib.scratch import scratch_buffer_spec, scratch_tensor
from b12x._lib.scratch_layout import align_up, materialize_scratch_view
from b12x.policy import PolicyResolution
from .._shared.execution import (
    GemmEngine,
    GraphPartition,
    MoEExecutionPlan,
    MoERegime,
    OutputReduction,
    PreparedWeightLayout,
    RouteLayout,
    WorkAvailability,
    WorkScheduler,
    make_moe_spec,
)
from ._policy import MOE_DECODE_POLICY, MoeDecodeConfig, MoeDecodeQuery

BACKEND = "tcgen05_trellis"


def projection_mixed(weight_plan):
    return (
        weight_plan.source_format == "b12x_trellis"
        and weight_plan.trellis_codebook == "mcg"
    )


def projection_rates(weight_plan):
    # Canonical rate tensors are loaded after weight planning. Resolve every
    # uniform SQG rate before capture; binding selects the prepared rate.
    if (
        weight_plan.source_format == "b12x_trellis"
        and weight_plan.trellis_codebook == "sqg_e4m3"
    ):
        return (2, 3, 4)
    return (weight_plan.trellis_bits,)


def validate_policy(query, config):
    if (
        query.source_format not in {"btx", "b12x_trellis"}
        or query.quant_mode != "w4a16"
        or query.activation not in {"silu", "situ"}
    ):
        raise UnsupportedArchitectureError(
            "SM103 Trellis MoE requires A16 SiLU or SiTU"
        )
    if (
        min(
            query.hidden_size,
            query.intermediate_size,
            query.num_experts,
            query.num_tokens,
            query.top_k,
        )
        <= 0
    ):
        raise ValueError("SM103 Trellis geometry and capacity must be positive")
    if query.hidden_size % 128 or query.intermediate_size % 128:
        raise UnsupportedArchitectureError(
            "SM103 Trellis transforms require H/I divisible by 128"
        )
    if (
        query.top_k > query.num_experts
        or query.routed_rows != query.num_tokens * query.top_k
    ):
        raise ValueError("invalid SM103 Trellis route capacity")
    if (
        query.routed_rows * (max(query.hidden_size, query.intermediate_size) // 128)
        > 2**31 - 1
    ):
        raise UnsupportedArchitectureError(
            "SM103 Trellis route capacity exceeds the CUDA grid limit"
        )
    if config != MoeDecodeConfig(BACKEND, "internal", None):
        raise UnsupportedArchitectureError(
            "SM103 Trellis requires the native materialized tcgen05 backend"
        )


def validate_weight_plan(weight_plan):
    from .._shared.trellis_codebooks import validate_codebook_bits

    if weight_plan.io_dtype not in {"float16", "bfloat16"}:
        raise UnsupportedArchitectureError(
            "SM103 Trellis requires FP16 or BF16 activations"
        )
    granularities = (
        (None, "uniform", "per_layer", "per_expert", "per_expert_projection")
        if projection_mixed(weight_plan)
        else (None, "uniform", "per_layer")
    )
    if (
        weight_plan.trellis_pair_kinds
        or weight_plan.trellis_rate_granularity not in granularities
    ):
        raise UnsupportedArchitectureError(
            "SM103 Trellis requires uniform or MCG projection rates; paired/grouped records remain unsupported"
        )
    validate_codebook_bits(weight_plan.trellis_codebook, weight_plan.trellis_bits)
    if weight_plan.coupled_hadamard and (
        weight_plan.hidden_size % 512 or weight_plan.activation != "situ"
    ):
        raise UnsupportedArchitectureError(
            "coupled Trellis requires H divisible by 512 and SiTU activation"
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

    validate_weight_plan(weight_plan)
    if (
        apply_router_weight_on_input
        or swiglu_limit is not None
        or swiglu_alpha not in (None, 1.0)
        or swiglu_beta not in (None, 0.0)
    ):
        raise UnsupportedArchitectureError(
            "SM103 Trellis applies router weights after the standard expert transform"
        )
    query = MoeDecodeQuery(
        quant_mode,
        weight_plan.source_format,
        weight_plan.activation,
        weight_plan.num_experts,
        weight_plan.hidden_size,
        weight_plan.intermediate_size,
        num_topk,
        num_tokens,
        num_tokens * num_topk,
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
                "Trellis policy resolution must match the component and device"
            )
        validate_policy(query, policy_resolution.config)
        resolution = policy_resolution
    spec = make_moe_spec(
        quant_mode=quant_mode,
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
        gemm_engine=GemmEngine.TRELLIS_TCGEN05,
        weight_layout=PreparedWeightLayout.TRELLIS_NATIVE,
        reduction=OutputReduction.ROUTE_BUFFER_TOPK_SUM,
        tile_m=128,
        tile_n=128,
    )
    return TPMoEPlan(
        spec=spec,
        execution=execution,
        implementation=BACKEND,
        quant_mode=quant_mode,
        activation=weight_plan.activation,
        state_E=weight_plan.num_experts,
        weight_E=weight_plan.num_experts,
        routed_rows=query.routed_rows,
        max_rows=num_tokens,
        k=query.hidden_size,
        n=query.intermediate_size,
        num_topk=num_topk,
        device=torch.device(device),
        dtype=getattr(torch, weight_plan.io_dtype),
        max_tokens_per_launch=num_tokens,
        policy_resolution=resolution,
    )


def scratch_layout(caps):
    from ._sm103 import Buffer

    routes = caps.max_tokens * caps.num_topk
    shapes = [
        ("ids", (routes,), torch.int64),
        ("output_ids", (routes,), torch.int64),
        ("input_gate", (routes, caps.k), torch.float16),
        ("input_up", (routes, caps.k), torch.float16),
        ("gate", (routes, caps.n), torch.float16),
        ("up", (routes, caps.n), torch.float16),
        ("activated", (routes, caps.n), torch.float16),
        ("down", (routes, caps.k), torch.float16),
        ("output", (caps.max_tokens, caps.k), torch.float32),
    ]
    buffers, offset = [], 0
    for name, shape, dtype in shapes:
        offset = align_up(offset, 1024)
        size = prod(shape) * dtype.itemsize
        buffers.append(Buffer(name, shape, dtype, offset, size))
        offset += size
    return tuple(buffers), align_up(offset, 1024)


def compile_launches(caps, *, offline=False, artifact_dir=None, artifact_prefix=""):
    import cuda.bindings.driver as cuda
    import cutlass as c
    import cutlass.cute as cute
    from pathlib import Path
    from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
    from .._shared.kernels.sm103.launch import pointer
    from .._shared.kernels.sm103.trellis_gemm import RoutedTrellisGemm
    from .._shared.kernels.sm103.trellis_mixed_gemm import RoutedMixedTrellisGemm
    from .._shared.kernels.sm103.trellis_transforms import (
        InputRotation,
        IntermediateRotation,
        OutputRotation,
        MapRoutes,
        ComposeExpertMaps,
    )

    if not offline and torch.cuda.get_device_capability(caps.device) != (10, 3):
        raise UnsupportedArchitectureError(
            "native Trellis launch compilation requires a physical SM103 device"
        )
    if (
        not offline
        and torch.cuda.get_device_properties(caps.device).shared_memory_per_block_optin
        < 33 * 1024
    ):
        raise UnsupportedArchitectureError(
            "SM103 Trellis requires at least 33 KiB opt-in shared memory"
        )
    weight_plan = caps.weight_plan
    mixed = projection_mixed(weight_plan)
    coupled, bits, codebook = (
        weight_plan.coupled_hadamard,
        weight_plan.trellis_bits,
        weight_plan.trellis_codebook,
    )
    routes = caps.max_tokens * caps.num_topk
    route_experts = caps.route_num_experts or caps.weight_E
    launches = {}
    io_type = c.BFloat16 if caps.dtype == torch.bfloat16 else c.Float16

    def compile_case(name, kernel, types, scalars):
        args = [pointer(t) for t in types] + list(scalars) + [cuda.CUstream(0)]
        if getattr(kernel, "dual_input", False):
            args[0] = (args[0], pointer(c.Float16), c.Int64(caps.n // 2))
        options = "--gpu-arch=sm_103a"
        directory = None
        if artifact_dir is not None:
            directory = Path(artifact_dir) / (artifact_prefix + name)
            directory.mkdir(parents=True, exist_ok=False)
            options += f" --keep-ptx --keep-cubin --dump-dir={directory}"
        spec = KernelCompileSpec.from_facts(
            "moe.sm103.trellis." + name,
            3,
            ("hidden", caps.k),
            ("intermediate", caps.n),
            ("expert_capacity", caps.weight_E),
            ("route_expert_capacity", route_experts),
            ("token_capacity", caps.max_tokens),
            ("top_k", caps.num_topk),
            ("bits", getattr(kernel, "bits", bits)),
            ("codebook", codebook),
            ("coupled", coupled),
            ("activation", caps.activation),
            ("io_dtype", str(caps.dtype)),
            ("projection_mixed", mixed),
            ("dual_input", getattr(kernel, "dual_input", False)),
        )
        if offline:
            fn = cute.compile(kernel, *args, options=options, no_jit_engine=True)
        else:
            with torch.cuda.device(caps.device):
                fn = b12x_compile(kernel, *args, options=options, compile_spec=spec)
        if directory is not None:
            (directory / (name + ".mlir")).write_text(str(fn.ir_module))
        launches[name] = fn

    for dtype in (c.Int32, c.Int64):
        for mapped in (False, True):
            for output_mapped in (False, True):
                name = f"map_{dtype.__name__}_{int(mapped)}_{int(output_mapped)}"
                compile_case(
                    name,
                    MapRoutes(
                        routes,
                        caps.weight_E,
                        route_experts,
                        mapped=mapped,
                        output_mapped=output_mapped,
                    ),
                    (dtype, c.Int32, c.Int32, c.Int64, c.Int64),
                    (c.Int32(1),),
                )
    if mixed:
        compile_case(
            "compose_maps",
            ComposeExpertMaps(routes, caps.weight_E),
            (c.Int64, c.Int64, c.Int32),
            (c.Int32(1),),
        )
    compile_case(
        "input",
        InputRotation(caps.k, caps.weight_E, caps.num_topk, routes, coupled=coupled),
        (io_type, c.Int64, c.Float16, c.Float16),
        (c.Int64(caps.k), c.Int32(1)),
    )
    compile_case(
        "intermediate",
        IntermediateRotation(
            caps.n, caps.weight_E, routes, coupled=coupled, activation=caps.activation
        ),
        (c.Float16, c.Float16, c.Int64, c.Float16, c.Float16),
        (c.Int32(1),),
    )
    for rate in () if mixed else projection_rates(weight_plan):
        for name, n, k in (("fc1", caps.n, caps.k), ("fc2", caps.k, caps.n)):
            for dual in (False, True) if coupled and name == "fc1" else (False,):
                compile_case(
                    name + f"_k{rate}" + ("_dual" if dual else ""),
                    RoutedTrellisGemm(
                        n,
                        k,
                        caps.weight_E,
                        routes,
                        bits=rate,
                        codebook=codebook,
                        dual_input=dual,
                    ),
                    (c.Float16, c.Uint32, c.Uint8, c.Int64, c.Float16),
                    (c.Int32(1), c.Int64(k), c.Int64(n)),
                )
    if mixed:
        for name, n, k in (
            ("fc1_mixed", caps.n, caps.k),
            ("fc2_mixed", caps.k, caps.n),
        ):
            for dual in (False, True) if coupled and name == "fc1_mixed" else (False,):
                compile_case(
                    name + ("_dual" if dual else ""),
                    RoutedMixedTrellisGemm(
                        n,
                        k,
                        caps.weight_E,
                        routes,
                        descriptor_local_bits=24 if caps.weight_E > 256 else 8,
                        dual_input=dual,
                    ),
                    (c.Float16, c.Uint32, c.Uint8, c.Int64, c.Int32, c.Float16),
                    (
                        c.Int32(0),
                        c.Int64(3 * caps.weight_E),
                        c.Int64(1),
                        (c.Int64(0),) * 3,
                        (c.Int32(0),) * 3,
                        c.Int32(1),
                        c.Int64(k),
                        c.Int64(n),
                    ),
                )
    for dtype in (c.Float32, io_type):
        compile_case(
            "output_" + dtype.__name__,
            OutputRotation(
                caps.k, caps.weight_E, caps.num_topk, caps.max_tokens, coupled=coupled
            ),
            (c.Float16, c.Int64, c.Float16, c.Float32, dtype),
            (c.Int64(caps.k), c.Int32(1)),
        )
    return launches


def _require_tensor(tensor, name, shape, dtype, device):
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.dtype != dtype
        or tuple(tensor.shape) != tuple(shape)
        or tensor.device != device
        or not tensor.is_contiguous()
    ):
        raise ValueError(
            f"{name} must be contiguous {dtype} {tuple(shape)} on {device}"
        )
    if tensor.data_ptr() % 16:
        raise ValueError(f"{name} must be 16-byte aligned")


def _mixed_contract(caps, prepared):
    from .trellis import PreparedProjectionTrellisWeights
    from .._shared.kernels.w4a16.prepare import TrellisWeightState

    if not isinstance(prepared, PreparedProjectionTrellisWeights):
        raise ValueError(
            "SM103 mixed Trellis requires canonical projection-tiered weights"
        )
    if (prepared.num_experts, prepared.hidden_size, prepared.intermediate_size) != (
        caps.weight_E,
        caps.k,
        caps.n,
    ):
        raise ValueError(
            "mixed Trellis prepared geometry differs from the capacity plan"
        )
    expected_bits = 24 if caps.weight_E > 256 else 8
    if (
        prepared.descriptor_local_bits != expected_bits
        or prepared.trellis_codebook != "mcg"
    ):
        raise ValueError(
            "mixed Trellis descriptor format or codebook differs from the plan"
        )
    _require_tensor(
        prepared.descriptor_map,
        "projection descriptors",
        (9 * caps.weight_E,),
        torch.int32,
        caps.device,
    )
    _require_tensor(
        prepared.global_to_combined,
        "prepared expert map",
        (caps.weight_E,),
        torch.int32,
        caps.device,
    )
    counts = (prepared.gate_counts, prepared.up_counts, prepared.down_counts)
    if any(
        row is None
        or len(row) != 3
        or any(not isinstance(v, int) or v < 0 for v in row)
        or sum(row) != caps.weight_E
        for row in counts
    ):
        raise ValueError(
            "mixed Trellis projection counts must partition the model experts"
        )
    for name, value in (
        ("gate/up payload", prepared.w13),
        ("down payload", prepared.w2),
    ):
        _require_tensor(value, name, (value.numel(),), torch.int32, caps.device)
    offsets = [[], [], []]
    cursor13, cursor2 = 0, 0
    if len(prepared.tiers) != 3:
        raise ValueError("mixed Trellis requires three native rate tiers")
    for tier, bit in enumerate((3, 4, 5)):
        record_words = (caps.k // 16) * (caps.n // 16) * 8 * bit
        payload = prepared.tiers[tier]
        if payload.trellis_codebook != "mcg" or payload.trellis_bits != bit:
            raise ValueError("mixed Trellis tier order must be MCG K3/K4/K5")
        if payload.coupled_hadamard != prepared.coupled_hadamard:
            raise ValueError(
                "mixed Trellis tier transforms must match the prepared owner"
            )
        if payload.trellis.input_scale_split != prepared.input_scale_split:
            raise ValueError(
                "mixed Trellis tier input-scale splits must match the prepared owner"
            )
        for value, owner, cursor, count in (
            (payload.w13, prepared.w13, cursor13, counts[0][tier] + counts[1][tier]),
            (payload.w2, prepared.w2, cursor2, counts[2][tier]),
        ):
            if (
                value.dtype != torch.int32
                or value.device != caps.device
                or not value.is_contiguous()
                or value.numel() != max(count, 1) * record_words
                or value.untyped_storage().data_ptr()
                != owner.untyped_storage().data_ptr()
                or value.data_ptr() != owner.data_ptr() + 4 * cursor
            ):
                raise ValueError(
                    "mixed Trellis tiers must be exact coalesced compressed payload views"
                )
        offsets[0].append(cursor13)
        offsets[1].append(cursor13 + counts[0][tier] * record_words)
        offsets[2].append(cursor2)
        cursor13 += payload.w13.numel()
        cursor2 += payload.w2.numel()
    if cursor13 != prepared.w13.numel() or cursor2 != prepared.w2.numel():
        raise ValueError(
            "mixed Trellis payload lengths disagree with projection counts"
        )
    rotations = prepared.rotations

    def prefix(value, width, *, broadcast=False):
        rows = 1 if broadcast and value.shape[0] == 1 else 3 * caps.weight_E
        _require_tensor(
            value,
            "mixed Trellis transform table",
            (rows, width),
            torch.float16,
            caps.device,
        )
        return value[: min(rows, caps.weight_E)]

    state = TrellisWeightState(
        codebook="mcg",
        bits=3,
        gate_suh=prefix(rotations.gate_suh, caps.k, broadcast=True),
        up_suh=prefix(rotations.up_suh, caps.k, broadcast=True),
        down_svh=prefix(rotations.down_svh, caps.k, broadcast=True),
        intermediate_rotations=prefix(
            rotations.intermediate, (6 if prepared.coupled_hadamard else 3) * caps.n
        ),
        coupled_hadamard=prepared.coupled_hadamard,
        input_scale_split=prepared.input_scale_split,
    )
    return state, tuple(tuple(row) for row in offsets), counts


@dataclass(frozen=True)
class BackendPlan:
    buffers: tuple
    launches: dict | None
    lut: torch.Tensor | None = None
    full_rotation: bool = True

    def prewarm(self, plan):
        if self.launches is not None:
            return plan
        caps = plan.caps
        with torch.cuda.device(caps.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "SM103 Trellis must be prewarmed before graph capture"
                )
        codebook = caps.weight_plan.trellis_codebook
        if codebook == "sqg_e4m3":
            from b12x._lib.quant.sqg_e4m3 import sqg_xor_cheb_t12_direct_lut_cpu

            lut = sqg_xor_cheb_t12_direct_lut_cpu().to(caps.device)
        elif codebook == "sqg_fp16":
            from b12x._lib.quant.sqg_fp16_d3l import sqg_fp16_d3l_descriptors_cpu

            lut = sqg_fp16_d3l_descriptors_cpu().to(caps.device)
        else:
            lut = torch.zeros(16, device=caps.device, dtype=torch.uint8)
        return replace(
            plan, _backend_plan=replace(self, launches=compile_launches(caps), lut=lut)
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
        import cutlass as c
        from ._impl import B12XFP4ExpertWeights, TPMoEFP4Binding
        from .._shared.kernels.sm103.launch import BoundLaunches, pointer
        from .._shared.kernels.w4a16.prepare import PreparedW4A16MoeWeights

        if self.launches is None or self.lut is None:
            raise RuntimeError("SM103 Trellis plans must be prewarmed before binding")
        if activation_amax is not None:
            raise UnsupportedArchitectureError(
                "Trellis experts do not use activation calibration"
            )
        # The public A16 binder sets unit_scale_contract. Trellis consumes
        # FP16 activations directly and has no activation-scale arithmetic.
        caps = plan.caps
        if (
            not isinstance(experts, B12XFP4ExpertWeights)
            or experts.plan != caps.weight_plan
        ):
            raise ValueError("prepared Trellis experts do not match the plan")
        prepared = experts.representation_for("w4a16")
        mixed = projection_mixed(caps.weight_plan)
        if mixed:
            state, offsets, counts = _mixed_contract(caps, prepared)
            if state.coupled_hadamard != caps.weight_plan.coupled_hadamard:
                raise ValueError(
                    "prepared mixed Trellis transform differs from the plan"
                )
        else:
            if (
                not isinstance(prepared, PreparedW4A16MoeWeights)
                or prepared.trellis is None
            ):
                raise ValueError(
                    "SM103 Trellis requires prepared native uniform-rate weights"
                )
            state = prepared.trellis
            if (
                state.bits not in projection_rates(caps.weight_plan)
                or state.codebook != caps.weight_plan.trellis_codebook
                or state.coupled_hadamard != caps.weight_plan.coupled_hadamard
                or state.fc1_pair_kind is not None
                or state.fc2_pair_kind is not None
                or prepared.w13_layout != "trellis_t256_proj"
            ):
                raise ValueError(
                    "prepared Trellis layout, codebook, or transform differs from the plan"
                )
        if a.ndim != 2 or not 0 < a.shape[0] <= caps.max_tokens:
            raise ValueError("Trellis activation rows exceed capacity")
        tokens, routes = a.shape[0], a.shape[0] * caps.num_topk
        _require_tensor(a, "activations", (tokens, caps.k), caps.dtype, caps.device)
        if topk_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Trellis route IDs must be Int32 or Int64")
        _require_tensor(
            topk_ids, "route IDs", (tokens, caps.num_topk), topk_ids.dtype, caps.device
        )
        _require_tensor(
            topk_weights,
            "router weights",
            (tokens, caps.num_topk),
            torch.float32,
            caps.device,
        )
        route_experts = caps.route_num_experts or caps.weight_E
        for name, tensor in (
            ("route_expert_map", route_expert_map),
            ("output_expert_map", output_expert_map),
        ):
            if tensor is not None:
                _require_tensor(
                    tensor, name, (route_experts,), torch.int32, caps.device
                )
        words = 0
        if not mixed:
            words = caps.weight_E * (caps.k // 16) * (caps.n // 16) * 8 * state.bits
            _require_tensor(
                prepared.w13, "gate/up payload", (2 * words,), torch.int32, caps.device
            )
            _require_tensor(
                prepared.w2, "down payload", (words,), torch.int32, caps.device
            )
        for name, tensor in (
            ("gate input scales", state.gate_suh),
            ("up input scales", state.up_suh),
            ("output scales", state.down_svh),
        ):
            if (
                tensor is None
                or tensor.ndim != 2
                or tensor.shape[0] not in (1, caps.weight_E)
            ):
                raise ValueError(f"{name} must have one or E rows")
            _require_tensor(
                tensor, name, (tensor.shape[0], caps.k), torch.float16, caps.device
            )
        _require_tensor(
            state.intermediate_rotations,
            "intermediate transforms",
            (caps.weight_E, caps.n * (6 if state.coupled_hadamard else 3)),
            torch.float16,
            caps.device,
        )
        split = state.input_scale_split
        if split is not None and (
            type(split) is not int
            or not state.coupled_hadamard
            or not 0 < split < caps.n
        ):
            raise ValueError(
                "coupled input-scale split must be an integer inside the local extent"
            )
        if (
            state.coupled_hadamard
            and split is None
            and state.gate_suh.data_ptr() != state.up_suh.data_ptr()
        ):
            raise ValueError(
                "coupled Trellis requires the shared input-scale tensor produced by preparation"
            )
        storage = scratch_tensor(
            scratch, plan.scratch_specs(), owner="SM103 Trellis MoE"
        )
        views = {
            b.name: materialize_scratch_view(
                storage, offset_bytes=b.offset, shape=b.shape, dtype=b.dtype
            )[0]
            for b in self.buffers
        }
        if output is None:
            output = views["output"][:tokens]
        if output.dtype not in (torch.float32, caps.dtype) or output.shape[0] not in (
            tokens,
            caps.max_tokens,
        ):
            raise ValueError(
                "Trellis output must have live/capacity rows and FP32 or input dtype"
            )
        _require_tensor(
            output, "output", (output.shape[0], caps.k), output.dtype, caps.device
        )
        output = output[:tokens]
        if (
            output.untyped_storage().data_ptr() == storage.untyped_storage().data_ptr()
            and (
                output.data_ptr() != views["output"].data_ptr()
                or output.dtype != torch.float32
            )
        ):
            raise ValueError("Trellis output may alias only the planned output buffer")
        inputs = (
            a,
            topk_ids,
            topk_weights,
            prepared.w13,
            prepared.w2,
            state.gate_suh,
            state.up_suh,
            state.down_svh,
            state.intermediate_rotations,
            self.lut,
        )
        inputs += tuple(
            t for t in (route_expert_map, output_expert_map) if t is not None
        )
        if mixed:
            inputs += (prepared.descriptor_map, prepared.global_to_combined)
        for tensor in inputs:
            if tensor.untyped_storage().data_ptr() in (
                storage.untyped_storage().data_ptr(),
                output.untyped_storage().data_ptr(),
            ):
                raise ValueError(
                    "Trellis scratch/output must not alias inputs or weights"
                )
        pids, poutput_ids = (
            pointer(c.Int64, views["ids"]),
            pointer(c.Int64, views["output_ids"]),
        )
        id_type = c.Int32 if topk_ids.dtype == torch.int32 else c.Int64
        calls = [
            (
                self.launches[
                    f"map_{id_type.__name__}_{int(route_expert_map is not None)}_{int(output_expert_map is not None)}"
                ],
                (
                    pointer(id_type, topk_ids),
                    pointer(c.Int32, route_expert_map),
                    pointer(c.Int32, output_expert_map),
                    pids,
                    poutput_ids,
                    c.Int32(routes),
                ),
            )
        ]
        if mixed:
            calls.append(
                (
                    self.launches["compose_maps"],
                    (
                        pids,
                        poutput_ids,
                        pointer(c.Int32, prepared.global_to_combined),
                        c.Int32(routes),
                    ),
                )
            )

        def projection_call(phase, source, target):
            fc1 = phase < 2
            dual = fc1 and split is not None
            source_pointer = (
                (
                    pointer(c.Float16, views["input_gate"]),
                    pointer(c.Float16, views["input_up"]),
                    c.Int64(split),
                )
                if dual
                else pointer(c.Float16, source)
            )
            suffix = "_dual" if dual else ""
            if mixed:
                payload = prepared.w13 if fc1 else prepared.w2
                fn = self.launches[("fc1_mixed" if fc1 else "fc2_mixed") + suffix]
                args = (
                    source_pointer,
                    pointer(c.Uint32, payload),
                    pointer(c.Uint8, self.lut),
                    pids,
                    pointer(c.Int32, prepared.descriptor_map),
                    pointer(c.Float16, target),
                    c.Int32(phase),
                    c.Int64(prepared.descriptor_map.numel() // 3),
                    c.Int64(payload.numel()),
                    tuple(c.Int64(v) for v in offsets[phase]),
                    tuple(c.Int32(v) for v in counts[phase]),
                )
            else:
                payload = (
                    (prepared.w13[:words] if phase == 0 else prepared.w13[words:])
                    if fc1
                    else prepared.w2
                )
                fn = self.launches[f"{'fc1' if fc1 else 'fc2'}_k{state.bits}" + suffix]
                args = (
                    source_pointer,
                    pointer(c.Uint32, payload),
                    pointer(c.Uint8, self.lut),
                    pids,
                    pointer(c.Float16, target),
                )
            return fn, args + (
                c.Int32(routes),
                c.Int64(caps.k if fc1 else caps.n),
                c.Int64(caps.n if fc1 else caps.k),
            )

        io_type = c.BFloat16 if caps.dtype == torch.bfloat16 else c.Float16
        for name, scales in (
            (("input_gate", state.gate_suh),)
            if state.coupled_hadamard and split is None
            else (("input_gate", state.gate_suh), ("input_up", state.up_suh))
        ):
            calls.append(
                (
                    self.launches["input"],
                    (
                        pointer(io_type, a),
                        pids,
                        pointer(c.Float16, scales),
                        pointer(c.Float16, views[name]),
                        c.Int64(0 if scales.shape[0] == 1 else caps.k),
                        c.Int32(routes),
                    ),
                )
            )
        calls.append(projection_call(0, views["input_gate"], views["gate"]))
        calls.append(
            projection_call(
                1,
                views["input_gate"] if state.coupled_hadamard else views["input_up"],
                views["up"],
            )
        )
        calls.append(
            (
                self.launches["intermediate"],
                (
                    pointer(c.Float16, views["gate"]),
                    pointer(c.Float16, views["up"]),
                    pids,
                    pointer(c.Float16, state.intermediate_rotations),
                    pointer(c.Float16, views["activated"]),
                    c.Int32(routes),
                ),
            )
        )
        calls.append(projection_call(2, views["activated"], views["down"]))
        out_type = c.Float32 if output.dtype == torch.float32 else io_type
        calls.append(
            (
                self.launches["output_" + out_type.__name__],
                (
                    pointer(c.Float16, views["down"]),
                    poutput_ids,
                    pointer(c.Float16, state.down_svh),
                    pointer(c.Float32, topk_weights),
                    pointer(out_type, output),
                    c.Int64(0 if state.down_svh.shape[0] == 1 else caps.k),
                    c.Int32(tokens),
                ),
            )
        )
        bound = BoundLaunches(
            tuple(calls), output, (storage, experts, self.lut, *inputs), caps.device
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
            quant_mode="w4a16",
            deterministic_output=True,
            _backend_binding=bound,
        )


def plan_scratch(caps, *, prewarm_launches, policy_resolution=None):
    from ._impl import TPMoEArenaLayout, TPMoEScratchPlan

    if caps.collect_activation_amax or caps.route_logits_dtype is not None:
        raise UnsupportedArchitectureError(
            "Trellis MoE requires preselected routes without calibration"
        )
    if caps.core_token_counts and max(caps.core_token_counts) > caps.max_tokens:
        raise ValueError("warmup counts exceed SM103 Trellis capacity")
    if caps.route_num_experts and caps.route_num_experts < caps.weight_E:
        raise ValueError("Trellis route expert capacity must cover the local experts")
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
    buffers, size = scratch_layout(caps)
    plan = TPMoEScratchPlan(
        caps=caps,
        layout=TPMoEArenaLayout(
            route_workspace_nbytes=0,
            core_workspace_nbytes=size,
            total_nbytes=size,
            core_token_counts=caps.core_token_counts or (caps.max_tokens,),
        ),
        launch_plan=launch_plan,
        _core_workspace_plan=None,
        _scratch_specs=(
            scratch_buffer_spec("tp_moe.scratch", nbytes=size, device=caps.device),
        ),
        _backend_plan=BackendPlan(buffers, None),
    )
    return plan._backend_plan.prewarm(plan) if prewarm_launches else plan
