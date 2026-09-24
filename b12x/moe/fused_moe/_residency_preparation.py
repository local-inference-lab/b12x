"""Declared programs and owned storage for hierarchical MXFP4 experts."""
from dataclasses import asdict, dataclass
import math

import torch

from b12x._lib.compile_pool import CompileJob
from b12x._lib.compile_plan import attach_programs, load_programs
from b12x._lib.program_cache import program_cache
from b12x.preparation import FrozenMapping, MemoryRequirements, PersistentMemory, Plan, current_plan
from ._residency_tuning import ResidencyQuery, TUNING
from ._residency_storage import (
    accounting, host_available_bytes, materialize_tier, validate_source, workspace_layout,
)
from .residency import ExpertMemoryBudget, ExpertResidencyPlan, ResidencyUpdateCapacity


@program_cache
def compile_programs(payload: FrozenMapping, ordinal: int):
    with torch.cuda.device(ordinal):
        return _compile_programs(ResidencyQuery(**dict(payload)), ordinal=ordinal)


def _compile_programs(query, *, ordinal=None, offline_dir=None, metadata_only=False, target="sm_103a"):
    """Compile the same entry points for preparation and offline qualification."""
    from pathlib import Path
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from b12x._lib.compiler import KernelCompileSpec, compile as compile_kernel
    from b12x.moe._shared.kernels.sm103.launch import pointer
    from b12x.moe._shared.kernels.sm103.residency import (
        OrderedFinalize, PartitionRoutes, QuantizeMxRoutes, RoutedMxGemm,
    )
    q, r = query, query.max_tokens * query.max_top_k
    ptr, i32, stream = pointer, cutlass.Int32, cuda.CUstream(0)
    programs = {}
    def add(name, kernel, dtypes, scalars):
        args = [ptr(t) for t in dtypes] + [i32(1) for _ in range(scalars)] + [stream]
        options = f"--gpu-arch={target}"
        if offline_dir is not None:
            destination = Path(offline_dir, name)
            destination.mkdir(parents=True, exist_ok=True)
            options += f" --keep-ptx --keep-cubin --dump-dir={destination}"
            program = cute.compile(kernel, *args, options=options, no_jit_engine=True)
            (destination / "module.mlir").write_text(str(program.ir_module))
        else:
            spec = KernelCompileSpec.from_facts("moe.residency." + name, 1,
                ("hidden", q.hidden), ("intermediate", q.intermediate),
                ("experts", q.experts), ("hot_experts", q.hot_experts),
                ("capacity", q.max_tokens), ("top_k_capacity", q.max_top_k),
                ("gate_first", q.gate_first), ("swiglu_limit", q.swiglu_limit),
                ("numerical_mode", q.numerical_mode), ("target", target))
            program = compile_kernel(kernel, *args, options=options, compile_spec=spec)
        programs[name] = program
    for dtype, suffix in ((cutlass.Int32, "i32"), (cutlass.Int64, "i64")):
        add("partition_" + suffix, PartitionRoutes(q.experts, (r+3)//4*4),
            (dtype, i32, i32, i32, i32), 1)
        add("finalize_" + suffix, OrderedFinalize(q.hidden, q.experts),
            (cutlass.BFloat16, dtype, cutlass.Float32, cutlass.BFloat16), 2)
        if not metadata_only:
            for name, width, activation in (("q1", q.hidden, False), ("q2", q.intermediate, True)):
                add(name + "_" + suffix, QuantizeMxRoutes(width, q.experts,
                    activation=activation, gate_first=q.gate_first, limit=q.swiglu_limit),
                    (cutlass.Float32 if activation else cutlass.BFloat16,
                     dtype, cutlass.Float8E4M3FN, cutlass.Uint8), 2)
    if not metadata_only:
        for tier, experts in (("hot", q.hot_experts), ("cold", q.experts-q.hot_experts)):
            if not experts:
                continue
            for name, n, k, fc1 in (("fc1", 2*q.intermediate, q.hidden, True),
                                    ("fc2", q.hidden, q.intermediate, False)):
                add(tier + "_" + name, RoutedMxGemm(n, k, experts, r, fc1=fc1),
                    (cutlass.Float8E4M3FN, cutlass.Float4E2M1FN,
                     cutlass.Float8E8M0FNU, cutlass.Float8E8M0FNU,
                     cutlass.Float32 if fc1 else cutlass.BFloat16,
                     i32, i32, i32, cutlass.Float32), 1)
    return programs


@dataclass(frozen=True)
class ResidencyBinding:
    a: torch.Tensor
    calls: tuple
    output: torch.Tensor
    owners: tuple
    plan: object = None

    def run(self):
        if self.owners[0].updates is not None:
            self.owners[0].updates.require_healthy()
        import cuda.bindings.driver as cuda
        stream = cuda.CUstream(torch.cuda.current_stream(self.a.device).cuda_stream)
        for program, args in self.calls:
            program(*args, stream)
        return self.output


@dataclass(frozen=True)
class _ResidencyState:
    query: ResidencyQuery
    device: torch.device
    programs: dict
    tiers: tuple
    mapping: torch.Tensor
    slab: torch.Tensor
    workspace: dict
    memory: object
    updates: object = None

    def bind(self, *, a, topk_ids, topk_weights, output=None):
        if self.updates is not None:
            self.updates.require_healthy()
        import cutlass
        from b12x.moe._shared.kernels.sm103.launch import pointer
        q, v = self.query, self.workspace
        if a.ndim != 2 or a.shape[1] != q.hidden or not 0 < a.shape[0] <= q.max_tokens:
            raise ValueError("activations exceed the declared expert geometry or capacity")
        m = a.shape[0]
        if topk_ids.ndim != 2 or topk_ids.shape[0] != m or not 0 < topk_ids.shape[1] <= q.max_top_k:
            raise ValueError("routes exceed the declared top-k capacity")
        top_k = topk_ids.shape[1]
        if output is None:
            output = v["out"][:m]
        for name, tensor, shape, dtypes in (
            ("activations", a, (m, q.hidden), (torch.bfloat16,)),
            ("route IDs", topk_ids, (m, top_k), (torch.int32, torch.int64)),
            ("route weights", topk_weights, (m, top_k), (torch.float32,)),
            ("output", output, (m, q.hidden), (torch.bfloat16,)),
        ):
            if tensor.device != self.device or tensor.shape != shape or tensor.dtype not in dtypes or not tensor.is_contiguous():
                raise ValueError(f"{name} differs from the prepared shape, device, dtype, or layout")
            if tensor.data_ptr() % 16:
                raise ValueError(f"{name} must be 16-byte aligned")
        # Caller buffers may not overwrite routes, checkpoint slabs, or workspace.
        def overlaps(left, right):
            return (left.data_ptr() < right.data_ptr() + right.numel()*right.element_size()
                    and right.data_ptr() < left.data_ptr() + left.numel()*left.element_size())
        readonly = (a, topk_ids, topk_weights, self.mapping) + tuple(t.slab for t in self.tiers if t)
        if any(overlaps(output, tensor) for tensor in readonly):
            raise ValueError("output overlaps a read-only execution operand")
        for tensor in (a, topk_ids, topk_weights):
            if overlaps(tensor, self.slab):
                raise ValueError("input overlaps preparation-owned workspace")
        if overlaps(output, self.slab) and output.data_ptr() != v["out"].data_ptr():
            raise ValueError("output overlaps preparation-owned scratch")
        suffix = "i32" if topk_ids.dtype == torch.int32 else "i64"
        id_type = cutlass.Int32 if suffix == "i32" else cutlass.Int64
        p = pointer
        live, k_live = cutlass.Int32(m*top_k), cutlass.Int32(top_k)
        ids = p(id_type, topk_ids)
        calls = [(self.programs["partition_"+suffix], (
            ids, p(cutlass.Int32, self.mapping), p(cutlass.Int32, v["local_ids"]),
            p(cutlass.Int32, v["indices"]), p(cutlass.Int32, v["counts"]), live))]
        calls.append((self.programs["q1_"+suffix], (p(cutlass.BFloat16, a), ids,
            p(cutlass.Float8E4M3FN, v["q1"]), p(cutlass.Uint8, v["s1"]), live, k_live)))
        for projection in ("fc1", "fc2"):
            if projection == "fc2":
                calls.append((self.programs["q2_"+suffix], (p(cutlass.Float32, v["fc1"]), ids,
                    p(cutlass.Float8E4M3FN, v["q2"]), p(cutlass.Uint8, v["s2"]), live, k_live)))
            first = projection == "fc1"
            for index, tier in enumerate(self.tiers):
                if tier is None:
                    continue
                prefix = "hot" if index == 0 else "cold"
                fields = tier.fields
                calls.append((self.programs[prefix+"_"+projection], (
                    p(cutlass.Float8E4M3FN, v["q1" if first else "q2"]),
                    p(cutlass.Float4E2M1FN, fields["w13" if first else "w2"]),
                    p(cutlass.Float8E8M0FNU, v["s1" if first else "s2"]),
                    p(cutlass.Float8E8M0FNU, fields["s13" if first else "s2"]),
                    p(cutlass.Float32 if first else cutlass.BFloat16, v[projection]),
                    p(cutlass.Int32, v["local_ids"][index]), p(cutlass.Int32, v["indices"][index]),
                    p(cutlass.Int32, v["counts"][index*4:]),
                    p(cutlass.Float32, self.mapping), live)))
        calls.append((self.programs["finalize_"+suffix], (
            p(cutlass.BFloat16, v["fc2"]), ids, p(cutlass.Float32, topk_weights),
            p(cutlass.BFloat16, output), cutlass.Int32(m), k_live)))
        return ResidencyBinding(a, tuple(calls), output, (self, a, topk_ids, topk_weights, output))


def plan(*, weight_plan, weights, capacity, placement, memory_budget, routing, invocation, override, updates=None):
    from .planning import WeightPlan, ActivationMode
    from .source import PackedSource, PackedSourceFormat
    from .execution import ExecutionCapacity, RoutingSpec
    if not isinstance(weight_plan, WeightPlan) or not isinstance(weight_plan.source, PackedSource):
        raise TypeError("hierarchical execution requires a packed WeightPlan")
    if not isinstance(placement, ExpertResidencyPlan) or not isinstance(memory_budget, ExpertMemoryBudget):
        raise TypeError("hierarchical execution requires placement and memory budget contracts")
    if not isinstance(capacity, ExecutionCapacity):
        raise TypeError("capacity must be ExecutionCapacity")
    if updates is not None and not isinstance(updates, ResidencyUpdateCapacity):
        raise TypeError("updates requires ResidencyUpdateCapacity")
    from .weights import WeightPacking
    if weight_plan.prepared_format.packing is not WeightPacking.SOURCE_NATIVE:
        raise ValueError("hierarchical execution requires a source_native weight packing declaration")
    activation = weight_plan.activation
    if (weight_plan.source.format != PackedSourceFormat.MXFP4_E8M0_K32
            or activation.mode != ActivationMode.A8 or activation.io_dtype != torch.bfloat16
            or activation.nonlinearity != "silu" or activation.swiglu_alpha is not None
            or activation.swiglu_beta is not None):
        raise ValueError("hierarchical execution supports BF16 I/O, SiLU, and native MXFP4 with A8 activations")
    routing = routing or RoutingSpec()
    if routing != RoutingSpec() and routing != RoutingSpec(deterministic_output=True):
        raise ValueError("hierarchical execution consumes preselected routes with unchanged FP32 weights")
    if capacity.route_num_experts not in (None, placement.total_experts):
        raise ValueError("hierarchical execution requires layer-local original expert IDs")
    if placement.total_experts != weight_plan.geometry.num_experts or invocation:
        raise ValueError("placement geometry or execution invocation differs from the declaration")
    q = ResidencyQuery(hidden=weight_plan.geometry.hidden_size,
        intermediate=weight_plan.geometry.intermediate_size, experts=placement.total_experts,
        hot_experts=len(placement.hbm_expert_ids), max_tokens=capacity.max_tokens, max_top_k=capacity.top_k,
        profile_hash=placement.profile_hash, model_fingerprint=placement.model_fingerprint,
        gate_first=weight_plan.source.w13_layout.value == "w31", swiglu_limit=activation.swiglu_limit,
        max_swap_pairs=0 if updates is None else updates.max_pairs)
    TUNING.validate_query(q, None)
    validate_source(weights, q)
    if weights.checkpoint_fingerprint != placement.model_fingerprint or weights.layer_name != placement.layer:
        raise ValueError("checkpoint fingerprint and layer must match the expert placement profile")
    memory = accounting(q)
    memory_budget.admit(memory)

    def materialize(selection, device):
        from b12x._lib.platform import probe_platform
        target = torch.device("cuda", device.ordinal)
        platform = probe_platform(target)
        if placement.grace_expert_ids and not platform.grace_coherent:
            raise ValueError("Grace expert storage requires verified coherent host-page-table access")
        # Validate values only during preparation; declaration never synchronizes.
        for name in ("w13_global_scales", "w2_global_scales", "input_scale", "intermediate_scale"):
            value = getattr(weights, name)
            if value is not None and not bool(torch.all(value == 1)):
                raise ValueError(f"native MXFP4 requires unit {name}; source values cannot be discarded")
        free, _ = torch.cuda.mem_get_info(target)
        if memory.hbm_total_bytes + memory_budget.hbm_safety_bytes + memory_budget.kv_reserved_bytes > free:
            raise ValueError("expert placement exceeds free HBM after declared reservations")
        host_free = host_available_bytes()
        if memory.grace_total_bytes + memory_budget.grace_safety_bytes > host_free:
            raise ValueError("expert placement exceeds available host memory after safety reservation")
        programs = compile_programs(FrozenMapping(asdict(q)), device.ordinal)
        load_programs(programs)
        tiers, slot_updates = [], None
        try:
            for index, ids in enumerate((placement.hbm_expert_ids, placement.grace_expert_ids)):
                tiers.append(materialize_tier(ids, weights, q, target, grace=index == 1))
            map_slab = torch.empty(memory.route_map_bytes, dtype=torch.uint8, device=target)
            mapping = map_slab[:q.experts*8].view(torch.int32).view(q.experts, 2)
            mapping.copy_(torch.tensor(placement.expert_map, dtype=torch.int32))
            layout, nbytes = workspace_layout(q)
            slab = torch.empty(nbytes, dtype=torch.uint8, device=target)
            workspace = {name: slab[offset:offset+math.prod(shape)*dtype.itemsize].view(dtype).view(shape)
                         for name, offset, shape, dtype in layout}
            from ._residency_updates import materialize_updates
            slot_updates = materialize_updates(q, tuple(tiers), mapping, placement, target)
            state = _ResidencyState(q, target, programs, tuple(tiers), mapping, slab, workspace, memory, slot_updates)
            return attach_programs(state, *programs.values())
        except BaseException:
            if slot_updates is not None:
                slot_updates.owner.close()
            for tier in tiers:
                if tier is not None and tier.owner is not None:
                    tier.owner.close()
            raise

    return Plan(contract=TUNING, query=q, invocation=FrozenMapping(), override=override,
        _compile_jobs=lambda config, device: (CompileJob.create(
            "b12x.moe.fused_moe._residency_preparation:compile_programs", FrozenMapping(asdict(q)), device.ordinal),),
        _memory_requirements=lambda config, device: MemoryRequirements(persistent=(
            PersistentMemory(key=("expert_residency", current_plan()), required_nbytes=memory.hbm_total_bytes),)),
        _materialize=materialize)
