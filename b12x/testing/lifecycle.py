"""Explicit resource observations outside model execution and timed requests."""

import json
import os
from pathlib import Path
import time


def process_resources():
    status = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        name, _, value = line.partition(":")
        if name in ("VmRSS", "VmHWM", "VmLck", "VmPin", "Threads"):
            status[name] = value.strip()
    return dict(
        pid=os.getpid(),
        status=status,
        descriptors=len(list(Path("/proc/self/fd").iterdir())),
        children=Path(f"/proc/self/task/{os.getpid()}/children").read_text().split(),
    )


def worker_resources(worker):
    import torch

    runner = getattr(worker, "model_runner", None)
    provider = getattr(worker.vllm_config, "_b12x_expert_cache_provider", None)
    model = provider.model if provider is not None else None
    allocations = []
    if model is not None:
        for name, plan in model.plans.items():
            if plan.prepared is not None:
                state = plan.prepared.state
                for kind in ("backing_owner", "map_owner"):
                    owner = getattr(state, kind, None)
                    if owner is not None and not owner._closed:
                        allocations.append(
                            dict(layer=name, kind=kind, bytes=owner.nbytes)
                        )
    free, total = torch.cuda.mem_get_info()
    counter = getattr(model, "_counters", None)
    health = getattr(counter, "health", None)
    manager = getattr(runner, "cudagraph_manager", None)
    host_stats = getattr(torch.cuda.memory, "host_memory_stats", None)
    return dict(
        **process_resources(),
        allocated=torch.cuda.memory_allocated(),
        reserved=torch.cuda.memory_reserved(),
        peak_allocated=torch.cuda.max_memory_allocated(),
        device_free=free,
        device_total=total,
        mapped_allocations=allocations,
        mapped_bytes=sum(a["bytes"] for a in allocations),
        cpu_source_bytes=0 if model is None else model.source_reserved_bytes,
        graph_owners=len(getattr(manager, "graphs", {})),
        pending_health=bool(health is not None and health.pending),
        health_host_bytes=0
        if health is None
        else health.host.numel() * health.host.element_size(),
        torch_host_stats=host_stats() if host_stats else None,
        native_memory_note="Device free memory includes all allocator/context use; VmLck/VmPin are not CUDA pinning totals.",
    )


def record_worker_resources(worker, stage, path):
    result = dict(stage=stage, time_ns=time.time_ns(), **worker_resources(worker))
    artifact = os.environ.get("B12X_ACCEPTANCE_BUILD_MANIFEST")
    if artifact:
        from .artifacts import verify_from_file

        result["artifacts"] = verify_from_file(artifact, loaded_only=True)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a") as stream:
        stream.write(json.dumps(result) + "\n")
    return result
