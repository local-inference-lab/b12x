"""Interleave native QSRT verify variants over one real TP9 weight extent.

Status: research-only. Four uninstrumented CUDA graphs share native K2 weights
and input/routing storage, with independent plans, scratch and outputs. Eager,
graph and mutated-input outputs must agree byte for byte across all variants.
Only uninstrumented graphs contribute latency samples. Optional phase records
come from separate graphs and describe one launch, not a model step.

Run as a module with the intended B12X and vLLM runtime on PYTHONPATH. The
checkpoint is read once per process; --rank selects logical TP9 ownership,
while --device selects the visible CUDA device. No serving endpoints are used.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import struct
import sys
from typing import Any

import torch

from benchmarks.benchmark_qsrt_tp9_extent import (
    HIDDEN,
    NUM_EXPERTS,
    TOP_K,
    _load_topk_ids,
    _routing_histogram,
    _uniform_topk_ids,
    _zipf_topk_ids,
)
from benchmarks.common import nvidia_smi_gpu_mode_snapshot


VARIANTS = ((256, 0), (256, 1), (512, 0), (512, 1))
# Each variant occupies each position and follows every other variant once.
ORDERS = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))
BASE_ENV = {
    "B12X_SQG_XOR_CHEB_T12_SMEM": "1",
    "B12X_SQG_XOR_CHEB_T12_DIRECT_SMEM": "1",
    "B12X_SQG_XOR_CHEB_T12_DECODE_CHAIN": "funnel",
    "B12X_W4A16_SMALL_M_SPLITK": "1",
    "B12X_W4A16_TOKEN_MAJOR_ROTATION": "1",
    "B12X_W4A16_CROSS_TILE_PREFETCH": "1",
    "B12X_W4A16_STABLE_ROUTE_PACK": "1",
    "B12X_W4A16_PREFILL_FUSED_SUM": "1",
    "B12X_W4A16_TOPK_SUM_OUTPUT": "bf16",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def _tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "numel": tensor.numel(),
        "bytes": tensor.numel() * tensor.element_size(),
        "data_ptr": tensor.data_ptr(),
        "storage_ptr": tensor.untyped_storage().data_ptr(),
    }


def _cuda_elf(obj: bytes) -> bytes:
    cursor = 0
    while True:
        start = obj.find(b"\x7fELF", cursor)
        if start < 0:
            raise ValueError("compiled object has no ELF64 CUDA image")
        cursor = start + 4
        if len(obj) - start < 64 or obj[start + 4 : start + 6] != b"\x02\x01":
            continue
        header = struct.unpack_from("<16sHHIQQQIHHHHHH", obj, start)
        if header[2] != 190:
            continue
        end = max(
            header[5] + header[9] * header[10], header[6] + header[11] * header[12]
        )
        for index in range(header[12]):
            offset = start + header[6] + index * header[11]
            if offset + 64 > len(obj):
                raise ValueError("CUDA section header exceeds the object")
            section = struct.unpack_from("<IIQQQQIIQQ", obj, offset)
            if section[1] != 8:  # SHT_NOBITS has no file payload.
                end = max(end, section[4] + section[5])
        if start + end > len(obj):
            raise ValueError("CUDA image exceeds its object wrapper")
        return obj[start : start + end]


def _select_environment(compiler, *, threads: int, pairs: int, phase: bool) -> dict:
    os.environ.update(BASE_ENV)
    os.environ.update(
        {
            "B12X_W4A16_M8_CTA_THREADS": str(threads),
            "B12X_SQG_XOR_CHEB_T12_DIRECT_PAIRS": str(pairs),
            "B12X_W4A16_PHASE_PROFILE": str(int(phase)),
        }
    )
    # Explicit kernel keys include the variant, but these metadata caches also
    # retain environment values. Refresh only metadata, keeping graph owners.
    refreshed = {}
    for name in ("_compile_environment_key", "_static_compile_cache_context"):
        function = getattr(compiler, name, None)
        if function is None or not hasattr(function, "cache_clear"):
            raise RuntimeError(f"runtime lacks required metadata cache {name}")
        refreshed[name] = function.cache_info()._asdict()
        function.cache_clear()
    contexts = getattr(compiler, "_DEVICE_COMPILE_CACHE_CONTEXTS", None)
    if not isinstance(contexts, dict):
        raise RuntimeError("runtime lacks the device compile-context cache")
    refreshed["_DEVICE_COMPILE_CACHE_CONTEXTS"] = len(contexts)
    contexts.clear()
    environment = dict(compiler._compile_environment_key())
    for name in (
        *BASE_ENV,
        "B12X_W4A16_M8_CTA_THREADS",
        "B12X_SQG_XOR_CHEB_T12_DIRECT_PAIRS",
        "B12X_W4A16_PHASE_PROFILE",
    ):
        if environment.get(name) != os.environ[name]:
            raise RuntimeError(f"compile metadata has a stale value for {name}")
    return {"environment": environment, "metadata_cache_state_before_clear": refreshed}


class CompileEvidence:
    def __init__(self, kernel, compiler, directory: Path):
        self.kernel = kernel
        self.compiler = compiler
        self.directory = directory
        self.records: dict[int, dict] = {}
        self.arm = "unassigned"

    @contextmanager
    def capture(self):
        original = self.kernel.b12x_compile

        def wrapped(owner, *values, **options):
            compiled = original(owner, *values, **options)
            if type(owner).__name__ == "W4A16FusedMoeKernel":
                self.record(
                    compiled,
                    owner=owner,
                    spec=options.get("compile_spec"),
                    stage="compile_return_before_resource_query",
                )
            return compiled

        self.kernel.b12x_compile = wrapped
        try:
            yield
        finally:
            self.kernel.b12x_compile = original

    def record(
        self, compiled, *, owner=None, spec=None, stage="resolved_binding"
    ) -> dict:
        identity = id(compiled)
        if identity in self.records:
            return self.records[identity]
        obj = compiled.dump_to_object("qsrt_verify_variants")
        cubin = _cuda_elf(obj)
        tag = f"{self.arm}-{len(self.records):03d}-{_sha(cubin)[:12]}"
        object_path = self.directory / f"{tag}.o"
        cubin_path = self.directory / f"{tag}.cubin"
        object_path.write_bytes(obj)
        cubin_path.write_bytes(cubin)
        record = {
            "id": tag,
            "capture_stage": stage,
            "object_path": str(object_path),
            "object_bytes": len(obj),
            "object_sha256": _sha(obj),
            "cubin_path": str(cubin_path),
            "cubin_bytes": len(cubin),
            "cubin_sha256": _sha(cubin),
            "compile_spec_json": getattr(spec, "json_key", None),
            "compile_spec_sha256": getattr(spec, "hash_key", None),
            "compile_environment": dict(self.compiler._compile_environment_key()),
            "package_fingerprint": self.compiler._b12x_package_fingerprint(),
            "toolchain": self.compiler._runtime_toolchain_key(),
        }
        if owner is not None:
            record["kernel_key"] = owner.__cache_key__
            record["owner"] = {
                name: getattr(owner, name, None)
                for name in (
                    "hidden_size",
                    "intermediate_size",
                    "top_k",
                    "element_dtype",
                    "rotation_input_dtype",
                    "cta_threads",
                    "sms",
                    "blocks_per_sm",
                    "workspace_elements",
                    "phase_profile",
                    "full_rotation",
                    "coupled_hadamard",
                    "token_major_rotation",
                    "sqg_xor_cheb_t12_direct_smem",
                )
            }
            record["owner"]["fc1_splitk"] = owner.fc1.small_m_splitk
            record["owner"]["fc2_splitk"] = owner.fc2.small_m_splitk
            record["owner"]["direct_pairs"] = getattr(
                owner.fc1, "sqg_xor_cheb_t12_direct_pairs", None
            )
        self.records[identity] = record
        return record


def _assert_output(
    output: torch.Tensor, expected: torch.Tensor | None, label: str
) -> str:
    if not bool(torch.isfinite(output).all()) or not bool(torch.count_nonzero(output)):
        raise AssertionError(f"{label}: output must be finite and nonzero")
    if expected is not None and not torch.equal(
        output.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    ):
        raise AssertionError(f"{label}: output bytes differ")
    return _sha(_tensor_bytes(output))


def _load_analyzer(path: Path | None):
    if path is None:
        module = importlib.import_module("benchmarks.analyze_w4a16_phase_profile")
    else:
        spec = importlib.util.spec_from_file_location("qsrt_phase_analyzer", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot load phase analyzer {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    if not callable(getattr(module, "summarize", None)):
        raise ValueError("phase analyzer must expose summarize(data, sms=, ctas=)")
    return module


def _attest(graph, *, path: Path, threads: int) -> dict:
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        graph.replay()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(path))
    events = [
        event
        for event in json.loads(path.read_text())["traceEvents"]
        if event.get("cat") == "kernel"
        and "W4A16FusedMoeKernel" in event.get("name", "")
    ]
    if len(events) != 1:
        raise AssertionError(
            f"{path}: expected one fused MoE launch, got {len(events)}"
        )
    event = events[0]
    args = event["args"]
    if args.get("block") != [threads, 1, 1]:
        raise AssertionError(f"{path}: unexpected CTA block {args.get('block')}")
    grid = args.get("grid")
    if not isinstance(grid, list) or len(grid) != 3 or min(grid) <= 0:
        raise AssertionError(f"{path}: missing valid actual launch grid")
    return {
        "trace_path": str(path),
        "symbol": event["name"],
        "symbol_sha256": _sha(event["name"].encode()),
        "grid": grid,
        "block": args["block"],
        "registers_per_thread": args.get("registers per thread"),
        "shared_memory_bytes": args.get("shared memory"),
        "limitation": "Symbol identity is not a unique compiled-object identity.",
    }


def _native_evidence(weights, *, width: int, first: int, atom_count: int) -> dict:
    payload = weights.representation_for("w4a16")
    for name, expected in (
        ("trellis_bits", 2),
        ("hidden_size", HIDDEN),
        ("intermediate_size", width),
        ("num_experts", NUM_EXPERTS),
        ("weight_layout", "trellis3_t256"),
        ("coupled_hadamard", True),
    ):
        if getattr(payload, name) != expected:
            raise AssertionError(f"native payload {name} differs from {expected!r}")
    tensors = {name: _tensor_metadata(getattr(payload, name)) for name in ("w13", "w2")}
    if (
        tensors["w13"]["storage_ptr"] != weights.w1_fp4.untyped_storage().data_ptr()
        or tensors["w2"]["storage_ptr"] != weights.w2_fp4.untyped_storage().data_ptr()
    ):
        raise AssertionError(
            "native representation does not share canonical weight storage"
        )
    for name, projections in (("w13", 2), ("w2", 1)):
        expected = NUM_EXPERTS * HIDDEN * width * projections * 2 // 8
        if tensors[name]["bytes"] != expected:
            raise AssertionError(
                f"{name}: expected {expected} bytes of native 2-bit codes"
            )
    return {
        "source_format": weights.source_format,
        "trellis_codebook": payload.trellis_codebook,
        "trellis_bits": payload.trellis_bits,
        "tile_config": payload.tile_config,
        "first_atom_slot": first,
        "atom_count": atom_count,
        "source_intermediate_channels": atom_count * 32,
        "runtime_width": width,
        "code_tensors": tensors,
        "code_bytes": sum(t["bytes"] for t in tensors.values()),
        "limitation": "Code bytes exclude scales, rotations, scratch and allocator overhead.",
    }


def _inputs(ms, *, routing: str, captured, seed: int, device):
    torch.manual_seed(seed)
    result = {}
    for m in ms:
        x = (torch.randn(m, HIDDEN, device=device) * 0.1).to(torch.bfloat16)
        if captured is not None:
            ids = captured.topk_ids.clone()
        elif routing == "uniform":
            ids = _uniform_topk_ids(m, device)
        elif routing.startswith("zipf:") and float(routing.split(":", 1)[1]) > 0:
            ids = _zipf_topk_ids(m, float(routing.split(":", 1)[1]), device)
        else:
            raise ValueError("--routing must be uniform or zipf:<positive exponent>")
        weights = (
            captured.topk_weights.clone()
            if captured is not None and captured.topk_weights is not None
            else torch.softmax(torch.randn(m, TOP_K, device=device), dim=-1)
        )
        base = (x, ids, weights)
        mutated = (
            (torch.randn_like(x.float()) * 0.1).to(x.dtype),
            ((ids + 1) % NUM_EXPERTS).contiguous(),
            weights.flip(-1).contiguous(),
        )
        result[m] = {
            "base": base,
            "mutated": mutated,
            "live": tuple(t.clone() for t in base),
        }
    return result


def _copy_inputs(inputs, scenario: str) -> None:
    for destination, source in zip(inputs["live"], inputs[scenario], strict=True):
        destination.copy_(source)


def _build_arm(
    *,
    threads,
    pairs,
    phase,
    ms,
    inputs,
    references,
    weights,
    fused_moe,
    compiler,
    evidence,
    artifacts,
    trace_ms,
    analyzer,
):
    arm_id = f"cta{threads}-pairs{pairs}-phase{int(phase)}"
    state = _select_environment(compiler, threads=threads, pairs=pairs, phase=phase)
    evidence.arm = arm_id
    plan = fused_moe.plan(
        fused_moe.Caps(
            max_tokens=max(ms),
            num_topk=TOP_K,
            device=torch.cuda.current_device(),
            weight_plan=weights.plan,
            quant_mode="w4a16",
            route_num_experts=NUM_EXPERTS,
            w4a16_block_size_m=8,
        )
    )
    spec = plan.scratch_specs()[0]
    expert_map = torch.arange(NUM_EXPERTS, dtype=torch.int32, device=spec.device)
    cases = {}
    records = []
    for m in ms:
        # Independent scratch per graph prevents records and outputs from
        # aliasing other token-count or variant graphs.
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        _copy_inputs(inputs[m], "base")
        x, ids, router = inputs[m]["live"]
        binding = fused_moe.bind(
            plan,
            scratch=scratch,
            a=x,
            experts=weights,
            topk_weights=router,
            topk_ids=ids,
            route_expert_map=expert_map,
        )
        eager = fused_moe.run(binding=binding).clone()
        _assert_output(eager, references.get((m, "base")), f"{arm_id}/M{m}/eager")
        references.setdefault((m, "base"), eager)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replay_output = fused_moe.run(binding=binding)
        graph.replay()
        torch.cuda.synchronize()
        base_digest = _assert_output(replay_output, eager, f"{arm_id}/M{m}/graph")
        _copy_inputs(inputs[m], "mutated")
        mutated = fused_moe.run(binding=binding).clone()
        _assert_output(
            mutated, references.get((m, "mutated")), f"{arm_id}/M{m}/mutation"
        )
        if torch.equal(mutated.view(torch.uint8), eager.view(torch.uint8)):
            raise AssertionError(f"{arm_id}/M{m}: changed inputs did not change output")
        references.setdefault((m, "mutated"), mutated)
        graph.replay()
        torch.cuda.synchronize()
        mutation_digest = _assert_output(
            replay_output, mutated, f"{arm_id}/M{m}/mutation graph"
        )
        _copy_inputs(inputs[m], "base")
        graph.replay()
        torch.cuda.synchronize()
        _assert_output(replay_output, eager, f"{arm_id}/M{m}/restore")
        launch = binding.fused_launch
        if launch is None or launch.cta_threads != threads:
            raise AssertionError(f"{arm_id}/M{m}: missing matching compiled launch")
        compiled = evidence.record(launch.compiled)
        owner = compiled.get("owner")
        if owner is not None and (
            owner["cta_threads"] != threads
            or owner["direct_pairs"] != bool(pairs)
            or bool(owner["phase_profile"]) != phase
            or not owner["sqg_xor_cheb_t12_direct_smem"]
        ):
            raise AssertionError(
                f"{arm_id}/M{m}: compiled owner differs from requested variant"
            )
        record = {
            "m": m,
            "compiled_object_id": compiled["id"],
            "base_output_sha256": base_digest,
            "mutated_output_sha256": mutation_digest,
            "eager_graph_exact": True,
            "cross_variant_exact": True,
            "mutated_replay_exact": True,
            "restored_replay_exact": True,
            "output_dtype": str(replay_output.dtype),
            "launch": {
                name: getattr(launch, name, None)
                for name in (
                    "size_m",
                    "max_m_blocks",
                    "cta_threads",
                    "registers_per_thread",
                    "shared_memory_bytes",
                    "workspace_elements",
                    "element_dtype",
                    "rotation_input_dtype",
                    "direct_topk_routes",
                    "use_expert_map",
                )
            },
        }
        if m in trace_ms or phase:
            record["attestation"] = _attest(
                graph, path=artifacts / f"{arm_id}-m{m}.trace.json", threads=threads
            )
        if phase:
            workspace = binding.kernel_workspace
            if workspace is None or workspace.dtype != torch.int32:
                raise AssertionError(
                    "phase profiling needs the bound int32 kernel workspace"
                )
            raw = _tensor_bytes(workspace)
            path = artifacts / f"{arm_id}-m{m}.workspace.bin"
            path.write_bytes(raw)
            sms = torch.cuda.get_device_properties(
                workspace.device
            ).multi_processor_count
            ctas = math.prod(record["attestation"]["grid"])
            record["phase_workspace_path"] = str(path)
            record["phase_profile"] = analyzer.summarize(raw, sms=sms, ctas=ctas)
        cases[m] = {
            "graph": graph,
            "binding": binding,
            "scratch": scratch,
            "output": replay_output,
            "expert_map": expert_map,
        }
        records.append(record)
    return {
        "id": arm_id,
        "threads": threads,
        "pairs": pairs,
        "phase": phase,
        "cases": cases,
        "record": {
            "id": arm_id,
            **state,
            "cases": records,
            "plan_caps": {
                "max_tokens": max(ms),
                "num_topk": TOP_K,
                "quant_mode": "w4a16",
                "route_num_experts": NUM_EXPERTS,
                "w4a16_block_size_m": 8,
            },
        },
    }


def _time_interleaved(
    arms, *, ms, rounds: int, warmup: int, samples: list
) -> list[dict]:
    for _ in range(warmup):
        for arm in arms:
            for m in ms:
                arm["cases"][m]["graph"].replay()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for round_index in range(rounds):
        for m in ms:
            for order_index, arm_index in enumerate(ORDERS[round_index % len(ORDERS)]):
                arm = arms[arm_index]
                begin.record()
                arm["cases"][m]["graph"].replay()
                end.record()
                end.synchronize()
                elapsed = begin.elapsed_time(end) * 1000.0
                if not math.isfinite(elapsed) or elapsed <= 0:
                    raise AssertionError("CUDA event returned an invalid elapsed time")
                samples.append(
                    {
                        "round": round_index,
                        "position": order_index,
                        "m": m,
                        "arm": arm["id"],
                        "graph_replay_us": elapsed,
                    }
                )
    return samples


def _parse_ms(text: str) -> tuple[int, ...]:
    values = tuple(
        dict.fromkeys(int(value) for value in text.split(",") if value.strip())
    )
    if not values or any(m not in (1, 2, 4, 8) for m in values):
        raise ValueError("token counts must be a nonempty subset of 1,2,4,8")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--rank", type=int, choices=range(9), default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, choices=(256, 384))
    parser.add_argument("--m-values", default="1,2,4,8")
    parser.add_argument("--timed-m-values", default="4")
    parser.add_argument("--time-iters", type=int, default=100)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--trace-m-values", default="4")
    parser.add_argument("--routing", default="uniform")
    parser.add_argument("--topk-ids", type=Path)
    parser.add_argument("--seed", type=int, default=71903)
    parser.add_argument("--phase-profile", action="store_true")
    parser.add_argument("--phase-analyzer", type=Path)
    parser.add_argument("--runtime-label", default="unspecified")
    parser.add_argument("--isolation-note", default="not declared")
    args = parser.parse_args()
    ms, timed_ms, trace_ms = map(
        _parse_ms, (args.m_values, args.timed_m_values, args.trace_m_values)
    )
    if args.time_iters <= 0 or args.warmup_iters < 0:
        parser.error("--time-iters must be positive and --warmup-iters nonnegative")
    analyzer = _load_analyzer(args.phase_analyzer) if args.phase_profile else None
    artifacts = (args.artifact_dir or args.output.with_suffix("")).resolve()
    if artifacts == args.output.resolve() or args.output.exists():
        parser.error(
            "output must be a fresh JSON path distinct from the artifact directory"
        )
    artifacts.mkdir(parents=True, exist_ok=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from b12x._lib import compiler
    from b12x.moe import fused_moe
    from b12x.moe._shared.kernels.w4a16 import kernel
    from b12x.moe._shared.qsrt_sharding import plan_qsrt_tp9_rank
    from vllm.model_executor.layers.quantization import kquant_qsrt_atoms_v2 as reader

    if not hasattr(kernel, "_sqg_xor_cheb_t12_direct_pairs_enabled"):
        raise RuntimeError("runtime does not implement the direct-pair variant")
    if args.phase_profile and not hasattr(kernel, "w4a16_phase_profile_enabled"):
        raise RuntimeError("runtime does not implement phase-profile workspaces")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    evidence = CompileEvidence(kernel, compiler, artifacts)
    result: dict[str, Any] = {
        "status": "running",
        "schema": "b12x.qsrt_verify_variants.v1",
        "argv": sys.argv,
        "runtime_label": args.runtime_label,
        "isolation_note": args.isolation_note,
        "source": {},
        "arms": [],
        "samples": [],
        "limitations": "One native extent and fixed inputs/routes; no TP9 model-throughput claim.",
    }
    source_files = [
        Path(__file__).resolve(),
        Path(kernel.__file__),
        Path(compiler.__file__),
        Path(reader.__file__),
        Path(fused_moe.__file__).with_name("_impl.py"),
        Path(sys.modules[_load_topk_ids.__module__].__file__),
        Path(sys.modules[nvidia_smi_gpu_mode_snapshot.__module__].__file__),
    ]
    if analyzer is not None:
        source_files.append(Path(analyzer.__file__))
    for path in source_files:
        result["source"][str(path.resolve())] = _sha(path.read_bytes())
    try:
        result["gpu_before_setup"] = nvidia_smi_gpu_mode_snapshot()
        _select_environment(compiler, threads=256, pairs=0, phase=False)
        extent = plan_qsrt_tp9_rank(args.layer, args.rank)
        width = args.width or extent.intermediate_channels
        if width != extent.intermediate_channels:
            raise ValueError(
                "this benchmark requires the native extent width, without padding"
            )
        path = args.model / f"qsrt-layer-{args.layer:05d}.safetensors"
        metadata = reader.read_qsrt_atom_v2_layer_metadata(path, layer=args.layer)
        weight_kwargs = dict(
            quant_modes="w4a16",
            source_format="qsrt_sqg_e4m3",
            activation="situ",
            params_dtype=torch.bfloat16,
            num_experts=NUM_EXPERTS,
            hidden_size=HIDDEN,
            intermediate_size=width,
            w13_layout="w13",
            trellis_bits=2,
            trellis_tile_config=(128, 128, 128, 128),
            qsrt_storage_format="qsrt_atoms_v2",
            qsrt_profile=metadata.profile,
        )
        weight_plan = fused_moe.plan_weights(**weight_kwargs)
        result["weight_plan_kwargs"] = {
            key: str(value) if isinstance(value, torch.dtype) else value
            for key, value in weight_kwargs.items()
        }
        evidence.arm = "shared-weight-prepare"
        with (
            evidence.capture(),
            reader.open_qsrt_atom_v2_extent(
                metadata, shard_count=9, shard_index=args.rank, device=None
            ) as (first, atoms),
        ):
            atom_count = int(atoms.shape[0])
            if (first, atom_count) != (extent.first_atom, extent.atom_count):
                raise AssertionError(
                    "reader extent differs from the TP9 ownership plan"
                )
            weights = fused_moe.prepare_weights(
                plan=weight_plan,
                params_dtype=torch.bfloat16,
                qsrt_atom_payload=atoms,
                qsrt_first_atom_slot=first,
                qsrt_layer_index=args.layer,
                gate_suh=metadata.gate_suh.unsqueeze(0).to(device),
                up_suh=metadata.up_suh.unsqueeze(0).to(device),
                down_svh=metadata.down_svh.unsqueeze(0).to(device),
                qsrt_rotation_draws=metadata.rotation_draws,
            )
        result["checkpoint"] = {
            "path": str(path.resolve()),
            "file_bytes": path.stat().st_size,
            "layer": args.layer,
            "logical_tp9_rank": args.rank,
            "profile": metadata.profile,
        }
        result["native_payload"] = _native_evidence(
            weights, width=width, first=first, atom_count=atom_count
        )
        captured = _load_topk_ids(args.topk_ids, device) if args.topk_ids else None
        if captured is not None:
            if captured.layer is not None and captured.layer != args.layer:
                raise ValueError(
                    "captured routing layer differs from the requested weight layer"
                )
            ms = _parse_ms(str(captured.topk_ids.shape[0]))
        if not set(timed_ms + trace_ms).issubset(ms) or (
            args.phase_profile and 4 not in ms
        ):
            raise ValueError(
                "timed/trace counts must be tested; phase profiling requires M4"
            )
        result["effective_m_values"] = ms
        result["timed_m_values"] = timed_ms
        result["trace_m_values"] = trace_ms
        inputs = _inputs(
            ms, routing=args.routing, captured=captured, seed=args.seed, device=device
        )
        result["inputs"] = {
            str(m): {
                "base_sha256": [_sha(_tensor_bytes(t)) for t in data["base"]],
                "mutated_sha256": [_sha(_tensor_bytes(t)) for t in data["mutated"]],
                "routing_histogram": _routing_histogram(data["base"][1], 8),
            }
            for m, data in inputs.items()
        }
        references = {}
        arms = []
        with evidence.capture():
            for threads, pairs in VARIANTS:
                arm = _build_arm(
                    threads=threads,
                    pairs=pairs,
                    phase=False,
                    ms=ms,
                    inputs=inputs,
                    references=references,
                    weights=weights,
                    fused_moe=fused_moe,
                    compiler=compiler,
                    evidence=evidence,
                    artifacts=artifacts,
                    trace_ms=trace_ms,
                    analyzer=None,
                )
                arms.append(arm)
                result["arms"].append(arm["record"])
                print(f"validated {arm['id']} at M={ms}", flush=True)
            result["gpu_before_timing"] = nvidia_smi_gpu_mode_snapshot()
            result["samples"] = _time_interleaved(
                arms,
                ms=timed_ms,
                rounds=args.time_iters,
                warmup=args.warmup_iters,
                samples=result["samples"],
            )
            result["gpu_after_timing"] = nvidia_smi_gpu_mode_snapshot()
            result["timing_summary"] = {
                f"{arm['id']}/m{m}": {
                    "samples": len(values),
                    "median_us": statistics.median(values),
                    "min_us": min(values),
                    "max_us": max(values),
                }
                for arm in arms
                for m in timed_ms
                if (
                    values := [
                        s["graph_replay_us"]
                        for s in result["samples"]
                        if s["arm"] == arm["id"] and s["m"] == m
                    ]
                )
            }
            if args.phase_profile:
                for threads, pairs in VARIANTS:
                    arm = _build_arm(
                        threads=threads,
                        pairs=pairs,
                        phase=True,
                        ms=(4,),
                        inputs=inputs,
                        references=references,
                        weights=weights,
                        fused_moe=fused_moe,
                        compiler=compiler,
                        evidence=evidence,
                        artifacts=artifacts,
                        trace_ms=(4,),
                        analyzer=analyzer,
                    )
                    result["arms"].append(arm["record"])
        result["status"] = "qualified output bytes; isolated graph latency observations"
        result["interleave_orders"] = ORDERS
        result["native_payload_after"] = _native_evidence(
            weights, width=width, first=first, atom_count=atom_count
        )
        if result["native_payload_after"] != result["native_payload"]:
            raise AssertionError(
                "native weight storage metadata changed during the benchmark"
            )
        result["gpu_after_diagnostics"] = nvidia_smi_gpu_mode_snapshot()
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["compiled_objects"] = list(evidence.records.values())
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
