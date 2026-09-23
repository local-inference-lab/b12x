"""Preparation-owned canonical host backing and fixed SM120 execution slots."""

from dataclasses import asdict, dataclass
import math

import torch

from b12x._lib.compile_plan import attach_programs, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import (
    FrozenMapping,
    MemoryRequirements,
    PersistentMemory,
    Plan,
    current_plan,
)
from . import _preparation as native
from ._cache_tuning import ExpertCacheConfig, ExpertCacheQuery, TUNING
from ._residency_storage import align
from .execution import ExecutionCapacity, RoutingSpec
from .planning import MoEGeometry, plan_weights, prepare_weights
from .weights import PackedWeights


NATIVE_CONFIG = native.MoeDecodeConfig(
    backend="w4a16",
    route_planner="internal",
    max_active_clusters=None,
    w4a16_route_mode="packed",
)


def tier_plan(q, count):
    from .planning import ActivationSpec, WeightPlanConstraints
    from .source import PackedSource

    return plan_weights(
        source=PackedSource(format="modelopt_nvfp4", w13_layout=q.w13_layout),
        activation=ActivationSpec(
            mode="a16", nonlinearity="silu", io_dtype=torch.bfloat16
        ),
        geometry=MoEGeometry(
            num_experts=count, hidden_size=q.hidden, intermediate_size=q.intermediate
        ),
        constraints=WeightPlanConstraints(required_packing="source_native"),
    )


def native_query(q, count):
    p = tier_plan(q, count)
    capacity = ExecutionCapacity(
        max_tokens=q.max_tokens, top_k=q.top_k, route_num_experts=q.experts
    )
    # These controls cannot change after declaration or between compiler workers.
    controls = FrozenMapping(
        {
            "dynamic_nvfp4_materialized": None,
            "dynamic_down_scale": False,
            "dynamic_swap_ab": None,
            "dynamic_tile_mn": None,
            "dynamic_work_source": "routing",
            "dynamic_external_route_plan": False,
            "dynamic_w4a8_repacked": False,
            "dynamic_w4a8_share_input": False,
            "dynamic_w4a8_materialized": False,
        }
    )
    return p, native._query_for_weight_plan(
        p,
        capacity,
        q.max_tokens,
        RoutingSpec(deterministic_output=True),
        controls,
        FrozenMapping(),
    )


def tier_layout(count, h, i):
    offset, fields = 0, []
    for name, shape in (
        ("w13", (count, 2 * i, h // 2)),
        ("w2", (count, h, i // 2)),
        ("s13", (count, 2 * i, h // 16)),
        ("s2", (count, h, i // 16)),
        ("g13", (count, 4)),
        ("g2", (count, 4)),
    ):
        fields.append((name, offset, shape))
        offset = align(offset + math.prod(shape))
    return fields, offset


def field_views(slab, layout):
    return {
        name: slab[offset : offset + math.prod(shape)].view(shape)
        for name, offset, shape in layout
    }


@dataclass(frozen=True)
class ExpertCacheMemory:
    resident_bytes: int
    backing_bytes: int
    source_bytes: int
    workspace_bytes: int
    metadata_bytes: int
    update_host_bytes: int

    @property
    def hbm_total_bytes(self):
        return self.resident_bytes + self.workspace_bytes + self.metadata_bytes

    @property
    def grace_total_bytes(self):
        return self.backing_bytes + self.source_bytes + self.update_host_bytes


def memory_for(q, device, source_bytes):
    workspace = 0
    for count in (q.resident, q.experts):
        wp, nq = native_query(q, count)
        caps = native._lower_caps(nq, NATIVE_CONFIG, wp._impl, device)
        scratch = native.plan_tp_moe_scratch(caps, prewarm_launches=False)
        workspace += sum(
            math.prod(s.shape) * s.dtype.itemsize for s in scratch.scratch_specs()
        )
    workspace += 3 * q.max_tokens * q.hidden * 2 + q.max_tokens * q.top_k * 4
    # Each native weight owner retains its own semaphore workspace and unit scale.
    workspace += (
        2 * (torch.cuda.get_device_properties(device).multi_processor_count * 4 + 2) * 4
    )
    workspace += (q.resident + q.experts) * 4
    return ExpertCacheMemory(
        tier_layout(q.resident, q.hidden, q.intermediate)[1],
        tier_layout(q.experts, q.hidden, q.intermediate)[1],
        source_bytes,
        workspace,
        align(q.experts * 8) * 2 + 16,
        2 * align(q.experts * 8) if q.max_pairs else 0,
    )


@program_cache
def compile_programs(payload, ordinal, config_payload=None):
    import cutlass
    import cuda.bindings.driver as cuda
    from b12x._lib.architecture import architecture_for
    from b12x._lib.compiler import KernelCompileSpec, compile as compile_kernel
    from b12x.moe._shared.kernels.sm103.launch import pointer
    from b12x.moe._shared.kernels.w4a16.residency import Remap, OrderedSum

    q = ExpertCacheQuery(**dict(payload))
    config = ExpertCacheConfig(**dict(config_payload or {}))
    TUNING.validate_config(q, config, None)
    programs, scratches = {}, []
    with torch.cuda.device(ordinal):
        target = architecture_for(torch.cuda.get_device_capability()).compilation_target
        for count in (q.resident, q.experts):
            wp, nq = native_query(q, count)
            scratches.append(
                native.compile_fused_moe(
                    native.TUNING.encode_query(nq),
                    native.TUNING.encode_config(NATIVE_CONFIG),
                    native._weight_plan_payload(wp),
                    (1, 1),
                    ordinal,
                )
            )
        if config.cold_prefill != "fused":
            wp, nq = native_query(q, q.experts)
            caps = native._lower_caps(nq, NATIVE_CONFIG, wp._impl, torch.device("cuda", ordinal))
            launchers = native._w4a16_primary_launches(
                scratches[-1], caps, cold_prefill=config.cold_prefill
            )
            programs["cold_prefill"] = launchers.carriers()
        for dtype, suffix in ((cutlass.Int32, "i32"), (cutlass.Int64, "i64")):
            for name, kernel, types in (
                (
                    "remap",
                    Remap(q.experts),
                    (cutlass.Int32, cutlass.Int32, dtype, cutlass.Int32),
                ),
                (
                    "sum",
                    OrderedSum(q.experts, q.hidden, q.top_k),
                    (
                        cutlass.BFloat16,
                        cutlass.BFloat16,
                        dtype,
                        cutlass.Int32,
                        cutlass.BFloat16,
                    ),
                ),
            ):
                spec = KernelCompileSpec.from_facts(
                    "moe.cache." + name,
                    1,
                    ("experts", q.experts),
                    ("hidden", q.hidden),
                    ("top_k", q.top_k),
                    ("ids", suffix),
                    ("target", target),
                )
                programs[name + "_" + suffix] = compile_kernel(
                    kernel,
                    *(pointer(t) for t in types),
                    cutlass.Int32(1),
                    cuda.CUstream(0),
                    options=f"--gpu-arch={target}",
                    compile_spec=spec,
                )
    programs["native"] = tuple(scratches)
    return programs


@dataclass(frozen=True)
class ExpertCacheBinding:
    a: torch.Tensor
    state: object
    tiers: tuple
    ids: torch.Tensor
    output: torch.Tensor
    remap_args: tuple
    sum_args: tuple
    suffix: str
    plan: object = None

    def run(self):
        import cuda.bindings.driver as cuda

        self.state.validate()
        if self.state.updates is not None:
            self.state.updates.require_executable()
        stream = cuda.CUstream(torch.cuda.current_stream(self.a.device).cuda_stream)
        self.state.programs["remap_" + self.suffix](*self.remap_args, stream)
        for binding in self.tiers:
            binding.run()
        self.state.programs["sum_" + self.suffix](*self.sum_args, stream)
        return self.output


class ExpertCacheState:
    def __init__(self, q, source, placement, memory, device, programs, config=ExpertCacheConfig()):
        from b12x.sequence._shared.disk_table import MappedHostAllocation
        from ._cache_updates import CanonicalSlotUpdates
        from ._residency_updates import _CudaTransfer

        self.query, self.source, self.memory, self.device = q, source, memory, device
        self.programs, self.closed = programs, False
        self.backing_owner = self.map_owner = None
        try:
            source.validate_values()
            hot_layout, hot_bytes = tier_layout(q.resident, q.hidden, q.intermediate)
            all_layout, all_bytes = tier_layout(q.experts, q.hidden, q.intermediate)
            self.backing_owner = MappedHostAllocation(
                (all_bytes,), torch.uint8, device, write_combined=False
            )
            self.resident_slab = torch.empty(
                hot_bytes, dtype=torch.uint8, device=device
            )
            self.resident = field_views(self.resident_slab, hot_layout)
            self.canonical = field_views(self.backing_owner.host_view, all_layout)
            self.backing = field_views(self.backing_owner.device_view, all_layout)
            hot_rows = {e: r for r, e in enumerate(placement.hbm_expert_ids)}
            for expert in range(q.experts):
                fields = source.row(expert)
                for name, row in fields.items():
                    self.canonical[name][expert].copy_(row)
                    if expert in hot_rows:
                        self.resident[name][hot_rows[expert]].copy_(row)
            expert_map = tuple(
                (0, hot_rows[e]) if e in hot_rows else (1, e) for e in range(q.experts)
            )
            self.mapping = torch.tensor(expert_map, dtype=torch.int32, device=device)
            self.maps = torch.empty((2, q.experts), dtype=torch.int32, device=device)
            self.safe_ids = torch.empty(
                (q.max_tokens, q.top_k), dtype=torch.int32, device=device
            )
            self.outputs = tuple(
                torch.empty(
                    (q.max_tokens, q.hidden), dtype=torch.bfloat16, device=device
                )
                for _ in range(3)
            )
            states, scratches = [], []
            for tier, (count, fields) in enumerate((
                (q.resident, self.resident),
                (q.experts, self.backing),
            )):
                wp, nq = native_query(q, count)
                experts = prepare_weights(
                    plan=wp,
                    weights=PackedWeights(
                        w13=fields["w13"],
                        w2=fields["w2"],
                        w13_block_scales=fields["s13"],
                        w2_block_scales=fields["s2"],
                        w13_global_scales=fields["g13"]
                        .view(torch.float32)
                        .reshape(count),
                        w2_global_scales=fields["g2"]
                        .view(torch.float32)
                        .reshape(count),
                    ),
                )
                caps = native._lower_caps(nq, NATIVE_CONFIG, wp._impl, device)
                scratch = native.plan_tp_moe_scratch(caps, prewarm_launches=True)
                launchers = native._w4a16_primary_launches(
                    scratch, caps, cold_prefill=config.cold_prefill if tier else "fused"
                )
                states.append(
                    native._FusedMoeState(
                        experts,
                        scratch,
                        NATIVE_CONFIG,
                        None,
                        launchers.carriers(),
                        None,
                        launchers,
                    )
                )
                scratches.append(
                    tuple(
                        torch.empty(s.shape, dtype=s.dtype, device=s.device)
                        for s in scratch.scratch_specs()
                    )
                )
                value = experts._impl.representation_for("w4a16")
                for attr, field in (
                    ("w13", "w13"),
                    ("w2", "w2"),
                    ("w13_scale", "s13"),
                    ("w2_scale", "s2"),
                    ("w13_global_scale", "g13"),
                    ("w2_global_scale", "g2"),
                ):
                    if getattr(value, attr).data_ptr() != fields[field].data_ptr():
                        raise RuntimeError(
                            "native preparation relocated fixed expert storage"
                        )
            self.states, self.scratches = tuple(states), tuple(scratches)
            self.updates = None
            if q.max_pairs:
                self.map_owner = MappedHostAllocation(
                    (2, q.experts, 2), torch.int32, device, write_combined=False
                )
                self.updates = CanonicalSlotUpdates(
                    resident=self.resident,
                    canonical=self.canonical,
                    mapping=self.mapping,
                    expert_map=expert_map,
                    before=self.map_owner.host_view[0],
                    after=self.map_owner.host_view[1],
                    transfer=_CudaTransfer(device),
                    max_pairs=q.max_pairs,
                )
            self.initial_map = expert_map
        except BaseException:
            self.close()
            raise

    def validate(self):
        if self.closed:
            raise RuntimeError("prepared expert cache is closed")
        if self.updates is not None:
            self.updates.require_healthy()

    def close(self):
        if not self.closed:
            self.closed = True
            for owner in (self.map_owner, self.backing_owner):
                if owner is not None:
                    owner.close()

    def _buffers(self):
        return (
            self.resident_slab,
            self.backing_owner.device_view,
            self.mapping,
            self.maps,
            self.safe_ids,
            *self.outputs,
            *(t for scratch in self.scratches for t in scratch),
            *(
                t
                for state in self.states
                for t in (
                    state.experts._impl.a1_gscale,
                    state.experts._impl.representation_for("w4a16").workspace,
                )
            ),
        )

    def pointers(self):
        self.validate()
        return tuple(t.data_ptr() for t in self._buffers())

    def bind(self, *, a, topk_ids, topk_weights, output=None):
        import cutlass
        from b12x.moe._shared.kernels.sm103.launch import pointer

        self.validate()
        q, m = self.query, a.shape[0]
        if not 0 < m <= q.max_tokens:
            raise ValueError("live tokens exceed prepared cache capacity")
        output = self.outputs[2][:m] if output is None else output
        for name, tensor, shape, dtypes in (
            ("activations", a, (m, q.hidden), (torch.bfloat16,)),
            ("routes", topk_ids, (m, q.top_k), (torch.int32, torch.int64)),
            ("route weights", topk_weights, (m, q.top_k), (torch.float32,)),
            ("output", output, (m, q.hidden), (torch.bfloat16,)),
        ):
            if (
                tensor.device != self.device
                or tensor.shape != shape
                or tensor.dtype not in dtypes
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"cache {name} differs from prepared shape, dtype, device or layout"
                )

        def overlaps(left, right):
            return (
                left.data_ptr()
                < right.data_ptr() + right.numel() * right.element_size()
                and right.data_ptr()
                < left.data_ptr() + left.numel() * left.element_size()
            )

        inputs = (a, topk_ids, topk_weights)
        for buffer in self._buffers():
            if any(overlaps(tensor, buffer) for tensor in inputs):
                raise ValueError("cache input overlaps preparation-owned storage")
            if overlaps(output, buffer) and buffer is not self.outputs[2]:
                raise ValueError("cache output overlaps preparation-owned storage")
        if any(overlaps(output, tensor) for tensor in inputs):
            raise ValueError("cache output overlaps a read-only operand")
        bindings = tuple(
            state.bind(
                scratch=scratch,
                a=a,
                topk_ids=self.safe_ids[:m],
                topk_weights=topk_weights,
                output=self.outputs[tier][:m],
                input_scales_static=True,
                route_expert_map=self.maps[tier],
            )
            for tier, (state, scratch) in enumerate(
                zip(self.states, self.scratches, strict=True)
            )
        )
        suffix = "i32" if topk_ids.dtype == torch.int32 else "i64"
        ids = pointer(cutlass.Int32 if suffix == "i32" else cutlass.Int64, topk_ids)
        remap_args = (
            pointer(cutlass.Int32, self.mapping),
            pointer(cutlass.Int32, self.maps),
            ids,
            pointer(cutlass.Int32, self.safe_ids),
            cutlass.Int32(m * q.top_k),
        )
        sum_args = (
            *(pointer(cutlass.BFloat16, b.intermediate_cache13) for b in bindings),
            ids,
            pointer(cutlass.Int32, self.mapping),
            pointer(cutlass.BFloat16, output),
            cutlass.Int32(m),
        )
        return ExpertCacheBinding(
            a, self, bindings, topk_ids, output, remap_args, sum_args, suffix
        )


def plan(*, source, capacity, placement, memory_budget, updates=None, override=None):
    from .cache_source import NVFP4_CACHE_ADAPTER, NVFP4_CACHE_RECIPE
    from b12x.moe.residency.storage import ExpertStorageSource
    from .residency import ExpertResidencyPlan, ExpertMemoryBudget

    if not isinstance(source, ExpertStorageSource) or not isinstance(
        placement, ExpertResidencyPlan
    ):
        raise TypeError(
            "expert cache requires an ExpertStorageSource and ExpertResidencyPlan"
        )
    contract = source.storage
    contract.require_cold_execution()
    if contract.adapter != NVFP4_CACHE_ADAPTER or contract.recipe != NVFP4_CACHE_RECIPE:
        raise NotImplementedError("no prepared cache execution backend for this storage adapter/recipe")
    if not isinstance(memory_budget, ExpertMemoryBudget):
        raise TypeError("expert cache requires an admitted memory budget")
    if not isinstance(
        capacity, ExecutionCapacity
    ) or capacity.route_num_experts not in (None, placement.total_experts):
        raise ValueError("cache capacity must use canonical layer-local expert IDs")
    if (
        source.weights.checkpoint_fingerprint != placement.model_fingerprint
        or source.weights.layer_name != placement.layer
        or source.plan.geometry.num_experts != placement.total_experts
    ):
        raise ValueError("cache source and placement checkpoint/layer/geometry differ")
    g = source.plan.geometry
    q = ExpertCacheQuery(
        experts=g.num_experts,
        resident=len(placement.hbm_expert_ids),
        hidden=g.hidden_size,
        intermediate=g.intermediate_size,
        max_tokens=capacity.max_tokens,
        top_k=capacity.top_k,
        w13_layout=source.plan.source.w13_layout.value,
        checkpoint_fingerprint=placement.model_fingerprint,
        profile_hash=placement.profile_hash,
        max_pairs=0 if updates is None else updates.max_pairs,
    )
    TUNING.validate_query(q, None)

    def materialize(selection, device):
        target = torch.device("cuda", device.ordinal)
        memory = memory_for(q, target, source.source_bytes)
        memory_budget.admit(memory)
        programs = compile_programs(
            FrozenMapping(asdict(q)), device.ordinal, FrozenMapping(asdict(selection.config))
        )
        load_programs(programs)
        return attach_programs(
            ExpertCacheState(q, source, placement, memory, target, programs, selection.config), programs
        )

    return Plan(
        contract=TUNING,
        query=q,
        override=override,
        _compile_jobs=lambda c, d: (
            CompileJob.create(
                "b12x.moe.fused_moe._cache_preparation:compile_programs",
                FrozenMapping(asdict(q)),
                d.ordinal,
                FrozenMapping(asdict(c)),
            ),
        ),
        _memory_requirements=lambda c, d: MemoryRequirements(
            persistent=(
                PersistentMemory(
                    key=("expert_cache", current_plan()),
                    required_nbytes=memory_for(
                        q, torch.device("cuda", d.ordinal), source.source_bytes
                    ).hbm_total_bytes,
                ),
            )
        ),
        _materialize=materialize,
    )
