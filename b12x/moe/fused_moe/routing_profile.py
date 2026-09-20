"""PreparationSession-owned counters and explicit out-of-band worker controls."""
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter_ns

import torch

from b12x._lib.compile_pool import CompileJob
from b12x._lib.compile_plan import attach_programs, load_programs
from b12x._lib.program_cache import program_cache
from b12x.preparation import FrozenMapping, MemoryRequirements, PersistentMemory, Plan, current_plan
from ._routing_profile_tuning import RoutingProfileQuery, RoutingProfileConfig, TUNING
from ..residency.contracts import LayerRoutingCounts, RoutingSnapshot


def _compile(query, *, target, offline_dir=None):
    import cutlass
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda
    from b12x._lib.compiler import KernelCompileSpec, compile as compile_kernel
    from b12x.moe._shared.kernels.sm103.launch import pointer
    from b12x.moe._shared.kernels.routing_profile import CountRoutes
    programs = {}
    if query.rank != query.owner_rank:
        return programs
    if query.runtime_token_limit:
        from b12x.moe._shared.kernels.routing_profile import SetTokenLimit
        args = (pointer(cutlass.Int32), cutlass.Int32(1), cuda.CUstream(0))
        if offline_dir is not None:
            path = Path(offline_dir)/"set_token_limit"
            path.mkdir(parents=True, exist_ok=True)
            programs["set_token_limit"] = cute.compile(SetTokenLimit(), *args,
                options=f"--gpu-arch={target} --keep-ptx --keep-cubin --dump-dir={path}", no_jit_engine=True)
        else:
            programs["set_token_limit"] = compile_kernel(SetTokenLimit(), *args,
                options=f"--gpu-arch={target}", compile_spec=KernelCompileSpec.from_facts(
                    "moe.routing_profile.set_token_limit", 1, ("target", target)))
    if query.health_summary:
        from b12x.moe._shared.kernels.routing_health import RoutingHealth
        args = (pointer(cutlass.Int64), pointer(cutlass.Uint64), cutlass.Int32(1),
                cutlass.Int32(0), cuda.CUstream(0))
        if offline_dir is not None:
            path = Path(offline_dir)/"health"
            path.mkdir(parents=True, exist_ok=True)
            programs["health"] = cute.compile(RoutingHealth(), *args,
                options=f"--gpu-arch={target} --keep-ptx --keep-cubin --dump-dir={path}", no_jit_engine=True)
        else:
            programs["health"] = compile_kernel(RoutingHealth(), *args,
                options=f"--gpu-arch={target}", compile_spec=KernelCompileSpec.from_facts(
                    "moe.routing_health", 1, ("target", target)))
    for experts in sorted({e for _, e in query.layers}):
        for dtype, suffix in ((cutlass.Int32, "i32"), (cutlass.Int64, "i64")):
            key = f"count_{experts}_{suffix}"
            args = [pointer(dtype), pointer(cutlass.Uint64), pointer(cutlass.Int32),
                    cutlass.Int32(1), cutlass.Int32(1), cuda.CUstream(0)]
            kernel = CountRoutes(experts, query.sample_every, query.runtime_token_limit)
            options = f"--gpu-arch={target}"
            if offline_dir is not None:
                path = Path(offline_dir)/key
                path.mkdir(parents=True, exist_ok=True)
                program = cute.compile(kernel, *args, options=options+f" --keep-ptx --keep-cubin --dump-dir={path}", no_jit_engine=True)
                (path/"module.mlir").write_text(str(program.ir_module))
            else:
                spec = KernelCompileSpec.from_facts("moe.routing_profile", 1, ("experts", experts),
                    ("sample_every", query.sample_every), ("ids_dtype", suffix),
                    ("runtime_token_limit", query.runtime_token_limit),
                    ("max_tokens", query.max_tokens), ("max_top_k", query.max_top_k), ("target", target))
                program = compile_kernel(kernel, *args, options=options, compile_spec=spec)
            programs[key] = program
    return programs


@program_cache
def compile_programs(payload, ordinal):
    from b12x._lib.architecture import architecture_for
    with torch.cuda.device(ordinal):
        target = architecture_for(torch.cuda.get_device_capability(ordinal)).compilation_target
        return _compile(RoutingProfileQuery(**dict(payload)), target=target)


@dataclass(frozen=True)
class RoutingProfileBinding:
    state: object
    program: object
    args: tuple
    ids: torch.Tensor
    plan: object = None

    def run(self):
        if self.plan is not None and (self.plan.prepared is None or self.plan.prepared.state is not self.state):
            raise RuntimeError("routing profile binding has been released or replaced")
        if self.program is not None:
            import cuda.bindings.driver as cuda
            self.program(*self.args, cuda.CUstream(torch.cuda.current_stream(self.state.device).cuda_stream))


class _CounterState:
    def __init__(self, query, device, programs):
        self.query, self.device, self.programs = query, device, programs
        self.epoch = 0
        self.rows = {}
        self.storage = torch.zeros(query.storage_bytes, device=device, dtype=torch.uint8) if query.storage_bytes else None
        if self.storage is not None:
            self.enabled = self.storage[:16].view(torch.int32)
            offset = 16
            for layer, experts in query.layers:
                size = (experts+6)//2*2*8
                for phase in query.phases:
                    self.rows[layer, phase] = self.storage[offset:offset+size].view(torch.uint64)
                    offset += size
            self.enabled[0] = 1
            self.enabled[1] = query.max_tokens
        self.health = None
        if query.health_summary:
            from ._routing_health import RoutingHealthState
            self.health = RoutingHealthState(self)

    def bind(self, *, layer, phase, topk_ids):
        import cutlass
        from b12x.moe._shared.kernels.sm103.launch import pointer
        q = self.query
        experts = dict(q.layers).get(layer)
        if experts is None or phase not in q.phases:
            raise ValueError("layer/phase not declared for routing profiling")
        if (topk_ids.device != self.device or topk_ids.dtype not in (torch.int32, torch.int64)
                or topk_ids.ndim != 2 or not topk_ids.is_contiguous()
                or not 0 < topk_ids.shape[0] <= q.max_tokens or not 0 < topk_ids.shape[1] <= q.max_top_k):
            raise ValueError("routing IDs differ from prepared device, capacity or layout")
        if self.storage is None:
            return RoutingProfileBinding(self, None, (), topk_ids)
        left, right = topk_ids.data_ptr(), self.storage.data_ptr()
        if left < right+self.storage.numel() and right < left+topk_ids.numel()*topk_ids.element_size():
            raise ValueError("routing IDs overlap preparation-owned counter storage")
        suffix = "i32" if topk_ids.dtype == torch.int32 else "i64"
        args = (pointer(cutlass.Int32 if suffix == "i32" else cutlass.Int64, topk_ids),
                pointer(cutlass.Uint64, self.rows[layer, phase]), pointer(cutlass.Int32, self.enabled),
                cutlass.Int32(topk_ids.shape[0]), cutlass.Int32(topk_ids.shape[1]))
        return RoutingProfileBinding(self, self.programs[f"count_{experts}_{suffix}"], args, topk_ids)

    def _quiesce(self, quiescent):
        if not quiescent:
            raise RuntimeError("counter controls require the engine to pause graph submission")
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("counter controls cannot execute during graph capture")
            torch.cuda.synchronize(self.device)

    def set_token_limit(self, tokens):
        """Publish an engine-labelled unpadded prefix on the producer stream.

        Zero excludes warmup, prefill, mixed batches or verification. There is
        no host wait or copy. Concurrent lanes need independent counter plans.
        """
        if not self.query.runtime_token_limit or type(tokens) is not int or not 0 <= tokens <= self.query.max_tokens:
            raise ValueError("runtime observation extent is outside its prepared contract")
        if self.storage is not None:
            import cutlass
            import cuda.bindings.driver as cuda
            from b12x.moe._shared.kernels.sm103.launch import pointer
            self.programs["set_token_limit"](pointer(cutlass.Int32, self.enabled),
                cutlass.Int32(tokens), cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream))

    def reset(self, *, quiescent=False):
        self._quiesce(quiescent)
        for row in self.rows.values():
            row.zero_()
        torch.cuda.synchronize(self.device)
        self.epoch += 1

    def set_enabled(self, enabled, *, quiescent=False):
        if type(enabled) is not bool:
            raise TypeError("profiling enabled must be bool")
        self._quiesce(quiescent)
        if self.storage is not None:
            self.enabled[0] = int(enabled)
            torch.cuda.synchronize(self.device)

    def snapshot(self, *, quiescent=False):
        started = perf_counter_ns()
        self._quiesce(quiescent)
        drained = perf_counter_ns()
        rows = []
        # One slab transfer snapshots the model at a quiescent boundary. Reading
        # each layer separately would add one synchronous D2H operation per row.
        host = self.storage.cpu() if self.storage is not None else None
        copied = perf_counter_ns()
        for (layer, phase), value in self.rows.items():
            experts = dict(self.query.layers)[layer]
            offset = value.data_ptr() - self.storage.data_ptr()
            data = host[offset:offset + value.numel()*8].view(torch.uint64).tolist()
            if data[experts+4]:
                raise OverflowError("routing counters overflowed; discard this epoch and reset")
            rows.append(LayerRoutingCounts(layer=layer, phase=phase, counts=tuple(data[:experts]),
                calls=data[experts], sampled_calls=data[experts+1], tokens=data[experts+2], sampled_tokens=data[experts+3]))
        result = RoutingSnapshot(epoch=self.epoch, rank=self.query.rank, layers=tuple(rows))
        self.last_snapshot_timings_ns = {"counter_drain": drained-started,
            "counter_d2h": copied-drained, "counter_decode": perf_counter_ns()-copied}
        return result


def plan_routing_profile(query, *, override=None):
    """Declare an opt-in observer; preparing and running the Plan remains explicit."""
    TUNING.validate_query(query, None)
    def materialize(selection, device):
        programs = compile_programs(FrozenMapping(asdict(query)), device.ordinal)
        load_programs(programs)
        return attach_programs(_CounterState(query, torch.device("cuda", device.ordinal), programs), *programs.values())
    return Plan(contract=TUNING, query=query, override=override, invocation=FrozenMapping(),
        _compile_jobs=lambda c, d: (CompileJob.create(
            "b12x.moe.fused_moe.routing_profile:compile_programs", FrozenMapping(asdict(query)), d.ordinal),),
        _memory_requirements=lambda c, d: MemoryRequirements(persistent=(
            PersistentMemory(key=("routing_profile", current_plan()), required_nbytes=query.storage_bytes+query.health_device_bytes),)) if query.storage_bytes else MemoryRequirements(),
        _materialize=materialize)


def bind_routing_profile(plan, **kwargs):
    from dataclasses import replace
    if plan.component_id != "moe.routing_profile" or plan.prepared is None:
        raise RuntimeError("routing profiling requires a session-prepared counter Plan")
    return replace(plan.prepared.state.bind(**kwargs), plan=plan)


def routing_profile_state(plan):
    """Return explicit worker controls for a prepared counter plan."""
    if plan.component_id != "moe.routing_profile" or plan.prepared is None:
        raise RuntimeError("routing profiler is not prepared")
    return plan.prepared.state
