#!/usr/bin/env python3
"""Production-plan NVFP4 split/monolithic CUDA-graph A/B receipt.

Each arm plans, prewarms, binds caller-owned fixed scratch and captures one
fused_moe.run under its explicit split setting. Only graph.replay is timed;
production chooses the backend, tile and active-cluster count. Unengaged or
incorrect cases have no qualified timing. Synthetic data is not serving evidence.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from benchmarks.common import nvidia_smi_gpu_mode_snapshot
from b12x.moe import fused_moe
from b12x.moe.fused_moe import _impl
from b12x.moe._shared.kernels.reference import compare_to_reference, moe_reference_nvfp4
from tests.moe.test_nvfp4_phase_kernels import (
    _bf16_output_bound,
    _quantize_nvfp4_rows,
    _swizzle_scale_plane,
)

SPLIT_ENV = "B12X_NVFP4_DYNAMIC_MATERIALIZED"
RATIO_DIRECTION = "monolithic_us / split_us (>1.0 means split is faster)"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=ROOT, check=True,
    ).stdout.strip()


def _source_hashes() -> dict[str, str]:
    """Include every loaded local module, including transitive oracle helpers."""
    paths = {Path(__file__).resolve(), ROOT / "benchmarks/gen_nvfp4_split_readme.py"}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.is_relative_to(ROOT) and path.is_file():
                paths.add(path)
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def _package_versions() -> dict[str, str | None]:
    import cutlass

    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nvidia-cutlass-dsl": getattr(cutlass, "__version__", None),
    }


@contextmanager
def _split_setting(enabled: bool):
    """Scope both the environment and its import-time cache to one arm."""
    previous = os.environ.get(SPLIT_ENV)
    cache_names = (
        "_NVFP4_MATERIALIZED_ENV_RAW",
        "_NVFP4_MATERIALIZED_ENV_EXPLICIT",
        "_NVFP4_MATERIALIZED_ENV_IS_TRUE",
    )
    cached = {name: getattr(_impl, name) for name in cache_names}
    try:
        os.environ[SPLIT_ENV] = "1" if enabled else "0"
        _impl._nvfp4_materialized_env_refresh()
        yield
    finally:
        if previous is None:
            os.environ.pop(SPLIT_ENV, None)
        else:
            os.environ[SPLIT_ENV] = previous
        for name, value in cached.items():
            setattr(_impl, name, value)


@contextmanager
def _observe_production_launches(compiled_identities):
    """Observe, never replace, production compilation and launch resolution.

    Associate the actual compiler result with its actual backend object. At
    capture, record the same compiled object returned to the real launch site.
    A cache entry whose construction was not observed cannot qualify a case.
    Neither wrapper changes arguments, results, policy or launch geometry.
    """
    compile_original = _impl.b12x_compile
    resolve_original = _impl._get_dynamic_kernel
    calls = []

    def observe_compile(launch, *args, **kwargs):
        compiled = compile_original(launch, *args, **kwargs)
        kernel = getattr(launch, "_kernel", None)
        if kernel is not None and hasattr(kernel, "nvfp4_split_materialized"):
            identity = {
                "adapter": type(launch).__name__,
                "backend_class": type(kernel).__name__,
                "nvfp4_split_materialized": bool(kernel.nvfp4_split_materialized),
                "mma_tiler_mn": list(kernel.tile_shape_mnk[:2]),
                "share_input_across_experts": bool(kernel.share_input_across_experts),
                "dynamic_down_scale": bool(kernel.dynamic_down_scale),
                "swap_ab": bool(kernel.swap_ab),
                "deterministic_output": bool(kernel.deterministic_output),
                "compile_spec": repr(kwargs.get("compile_spec")),
            }
            compiled_identities[id(compiled)] = (compiled, identity)
        return compiled

    def observe_resolve(*args, **kwargs):
        compiled, mac = resolve_original(*args, **kwargs)
        observed = compiled_identities.get(id(compiled))
        calls.append({
            "compiled_identity_observed": observed is not None,
            "compiled_object_id": id(compiled),
            "max_active_clusters": int(mac),
            **(observed[1] if observed is not None else {}),
        })
        return compiled, mac

    _impl.b12x_compile = observe_compile
    _impl._get_dynamic_kernel = observe_resolve
    try:
        yield calls
    finally:
        _impl.b12x_compile = compile_original
        _impl._get_dynamic_kernel = resolve_original


def _build_inputs(*, E: int, K: int, n: int, M: int, top_k: int, seed: int):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    x = (torch.randn(M, K, device=device) * 2.0).to(torch.bfloat16)
    ids = torch.stack([
        torch.randperm(E, device=device)[:top_k] for _ in range(M)
    ]).to(torch.int32)
    route_weights = torch.softmax(torch.randn(M, top_k, device=device), dim=-1).float()

    def weights(rows, cols):
        payloads, scales = [], []
        for _ in range(E):
            values = torch.randn(rows, cols, device=device) * 0.05
            packed, _, scale = _quantize_nvfp4_rows(values, 1.0)
            payloads.append(packed)
            scales.append(_swizzle_scale_plane(scale, rows))
        return torch.stack(payloads).contiguous(), torch.stack(scales).contiguous()

    w13, sf13 = weights(2 * n, K)
    w2, sf2 = weights(K, n)
    bundle = fused_moe.PackedWeights(
        w13=w13, w2=w2, w13_block_scales=sf13, w2_block_scales=sf2,
        w13_global_scales=torch.ones(E, device=device),
        w2_global_scales=torch.ones(E, device=device),
        # One shared scalar, not an E-element vector: production can share A.
        input_scale=torch.ones(1, device=device),
        intermediate_scale=torch.ones(E, device=device),
    )
    oracle = moe_reference_nvfp4(
        x, w13, sf13, bundle.w13_global_scales,
        w2, sf2, bundle.w2_global_scales,
        bundle.input_scale, bundle.intermediate_scale, ids, route_weights,
        E, K, n, activation="silu", quant_scale_math="direct_division",
    )
    weight_plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
        geometry=fused_moe.MoEGeometry(
            num_experts=E, hidden_size=K, intermediate_size=n,
        ),
        activation=fused_moe.ActivationSpec(
            mode=fused_moe.ActivationMode.A4, nonlinearity="silu", io_dtype=torch.bfloat16,
        ),
        constraints=fused_moe.WeightPlanConstraints(
            required_packing=fused_moe.WeightPacking.SOURCE_NATIVE,
        ),
    )
    experts = fused_moe.prepare_weights(plan=weight_plan, weights=bundle)
    return x, ids, route_weights, experts, oracle


def _build_arm(*, name, x, ids, route_weights, experts, calls, fast_math):
    with _split_setting(name == "split"):
        plan = fused_moe.plan_execution(
            experts=experts,
            capacity=fused_moe.ExecutionCapacity(
                max_tokens=x.shape[0], top_k=ids.shape[1],
                warmup_token_counts=(x.shape[0],),
            ),
        )
        fused_moe.prewarm(plan)
        scratch = {
            spec.name: torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
            for spec in plan.scratch_specs()
        }
        output = torch.empty_like(x)
        binding = fused_moe.bind(
            plan, scratch=scratch, a=x, experts=experts,
            topk_ids=ids, topk_weights=route_weights, output=output,
            input_scales_static=True, fast_math=fast_math,
        )
        # Warm lazy launch state on a side stream before capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fused_moe.run(binding=binding)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        capture_begin = len(calls)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            fused_moe.run(binding=binding)
        torch.cuda.synchronize()
        captured = calls[capture_begin:]
        identity = {
            "name": name,
            "split_environment": {SPLIT_ENV: os.environ[SPLIT_ENV]},
            "path": "fused_moe.plan_execution/prewarm/bind/run -> CUDA graph replay",
            "activation_mode": "a4", "quant_mode": binding.quant_mode,
            "input_scale": "shared scalar 1.0", "input_scales_static": True,
            "fast_math": fast_math,
            "capacity": asdict(plan.capacity),
            "bound_policy": asdict(binding.execution_plan.policy_resolution.config),
            "scratch_bytes": sum(t.numel() * t.element_size() for t in scratch.values()),
            "capture_dynamic_launches": captured,
            "split_engaged": bool(captured) and all(
                c.get("compiled_identity_observed") and c.get("nvfp4_split_materialized")
                for c in captured
            ),
        }
    return {"graph": graph, "output": output, "binding": binding,
            "plan": plan, "scratch": scratch, "identity": identity}


def _addresses(arms):
    result = {}
    for name, arm in arms.items():
        tensors = {f"scratch.{key}": value for key, value in arm["scratch"].items()}
        for owner_name, owner in (("binding", arm["binding"]),
                                  ("experts", arm["plan"].experts._impl)):
            tensors.update({
                f"{owner_name}.{field.name}": value
                for field in fields(owner)
                if isinstance(value := getattr(owner, field.name), torch.Tensor)
            })
        result[name] = {key: value.data_ptr() for key, value in tensors.items()}
    return result


def _correctness(arms, oracle):
    bound = _bf16_output_bound(oracle)
    result = {"bound_abs": bound, "oracle_finite": bool(torch.isfinite(oracle).all())}
    for name, arm in arms.items():
        out = arm["output"]
        metrics = compare_to_reference(out.float(), oracle)
        finite = bool(torch.isfinite(out).all())
        nonzero = int(out.count_nonzero()) > 0
        result[name] = asdict(metrics)
        result[f"finite_{name}"] = finite
        result[f"nonzero_{name}"] = nonzero
        result[f"{name}_passed"] = bool(
            finite and nonzero and metrics.cos > 0.9999 and metrics.rmse <= bound
        )
    cross = compare_to_reference(arms["split"]["output"].float(),
                                 arms["monolithic"]["output"].float())
    result["split_vs_monolithic"] = asdict(cross)
    result["passed"] = bool(
        result["oracle_finite"] and result["split_passed"] and result["monolithic_passed"]
        and cross.cos > 0.9999 and cross.rmse <= bound
    )
    return result


def _graph_check(arms, oracle):
    """Poison both outputs and replay twice, checking fixed-address reuse."""
    addresses = _addresses(arms)
    allocated = torch.cuda.memory_allocated()
    checks = []
    replay_allocation_counts = []
    for order in (("monolithic", "split"), ("split", "monolithic")):
        before_replay_allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        for name in order:
            arms[name]["output"].fill_(float("nan"))
            arms[name]["graph"].replay()
        torch.cuda.synchronize()
        after_replay_allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        replay_allocation_counts.append({
            "before": before_replay_allocations,
            "after": after_replay_allocations,
        })
        checks.append(_correctness(arms, oracle))
    final_allocated = torch.cuda.memory_allocated()
    stable_addresses = _addresses(arms) == addresses
    no_replay_allocations = all(
        count["before"] == count["after"] for count in replay_allocation_counts
    )
    return {
        "poison_rewrite_checks": checks,
        "addresses": addresses,
        "allocated_bytes_before": allocated,
        "allocated_bytes_after": final_allocated,
        "allocation_stable": allocated == final_allocated,
        "replay_allocation_counts": replay_allocation_counts,
        "no_replay_allocations": no_replay_allocations,
        "addresses_stable": stable_addresses,
        "passed": all(c["passed"] for c in checks)
        and allocated == final_allocated and no_replay_allocations and stable_addresses,
    }


def _time_launch(launch, *, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    samples_us = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iterations):
        start.record()
        launch()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1e3)
    return samples_us


def _sample_under_load(launch) -> dict:
    import threading

    holder = {}

    def snapshot():
        holder["snapshot"] = nvidia_smi_gpu_mode_snapshot()

    for _ in range(20):
        launch()
    thread = threading.Thread(target=snapshot)
    thread.start()
    while thread.is_alive():
        for _ in range(20):
            launch()
        torch.cuda.synchronize()
    thread.join()
    return holder.get("snapshot", {"available": False})


def _gpu_mode_check(
    snapshots, *, initial_snapshot, expected_uuid, allow_software_power_cap,
    max_sm_clock_delta_percent
):
    """Qualify diagnostic timings against the declared physical-GPU conditions."""
    allowed_masks = (0, 4) if allow_software_power_cap else (0,)
    failures = []
    observed = {}
    result = {
        "allowed_throttle_masks": [hex(mask) for mask in allowed_masks],
        "required_pstate": "P1",
        "max_sm_clock_delta_percent": max_sm_clock_delta_percent,
        "sm_clock_delta_percent": None,
        "sm_clock_delta_reference": "minimum arm SM clock",
        "initial_throttle_mask": None,
        "initial_pstate": None,
        "active_throttle_masks": {},
        "initial_to_active_throttle_masks": {},
        "inter_arm_software_power_cap_transition": False,
        "software_power_cap_transition": False,
        "failures": failures,
    }
    initial_mask = None
    if initial_snapshot.get("available"):
        try:
            initial_mode = initial_snapshot["fields"]
            initial_mask = int(initial_mode["clocks_throttle_reasons.active"], 0)
            result["initial_throttle_mask"] = hex(initial_mask)
            result["initial_pstate"] = initial_mode["pstate"]
            if initial_mode["uuid"].removeprefix("GPU-") != expected_uuid.removeprefix("GPU-"):
                failures.append("initial snapshot: physical GPU identity changed")
            if initial_mask not in allowed_masks:
                failures.append(f"initial snapshot: disallowed throttle mask {hex(initial_mask)}")
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            failures.append(f"invalid initial GPU mode snapshot: {error}")
    else:
        failures.append("initial GPU mode snapshot unavailable")
    for name in ("monolithic", "split"):
        snapshot = snapshots.get(name, {})
        if not snapshot.get("available"):
            failures.append(f"{name}: GPU mode snapshot unavailable")
            continue
        try:
            mode = snapshot["fields"]
            uuid = mode["uuid"].removeprefix("GPU-")
            sm_clock = int(mode["clocks.current.sm"])
            memory_clock = int(mode["clocks.current.memory"])
            throttle_mask = int(mode["clocks_throttle_reasons.active"], 0)
            result["active_throttle_masks"][name] = hex(throttle_mask)
            if uuid != expected_uuid.removeprefix("GPU-"):
                failures.append(f"{name}: physical GPU identity changed")
            if mode["pstate"] != "P1":
                failures.append(f"{name}: GPU is not in P1")
            if throttle_mask not in allowed_masks:
                failures.append(f"{name}: disallowed throttle mask {hex(throttle_mask)}")
            if sm_clock <= 0 or memory_clock <= 0:
                failures.append(f"{name}: invalid GPU clock")
                continue
            observed[name] = (sm_clock, memory_clock, throttle_mask)
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            failures.append(f"{name}: invalid GPU mode snapshot: {error}")
    if len(observed) == 2:
        mono_sm, mono_memory, mono_mask = observed["monolithic"]
        split_sm, split_memory, split_mask = observed["split"]
        if mono_memory != split_memory:
            failures.append("memory clocks differ between arms")
        delta = 100.0 * abs(mono_sm - split_sm) / min(mono_sm, split_sm)
        result["sm_clock_delta_percent"] = delta
        if delta > max_sm_clock_delta_percent:
            failures.append("SM-clock delta exceeds the declared bound")
        result["inter_arm_software_power_cap_transition"] = {mono_mask, split_mask} == {0, 4}
        if initial_mask is not None:
            result["initial_to_active_throttle_masks"] = {
                name: [hex(initial_mask), hex(values[2])] for name, values in observed.items()
            }
            result["software_power_cap_transition"] = {initial_mask, mono_mask, split_mask} == {0, 4}
    result["passed"] = len(observed) == 2 and not failures
    return result


def _run_case(case, args, compiled_identities, initial_snapshot):
    x, ids, route_weights, experts, oracle = _build_inputs(**case["shape"], seed=args.seed)
    with _observe_production_launches(compiled_identities) as calls:
        arms = {
            name: _build_arm(name=name, x=x, ids=ids, route_weights=route_weights,
                             experts=experts, calls=calls, fast_math=args.fast_math)
            for name in ("monolithic", "split")
        }
    case["arms"] = {name: arm["identity"] for name, arm in arms.items()}
    case["split_engaged"] = arms["split"]["identity"]["split_engaged"]
    mono_calls = arms["monolithic"]["identity"]["capture_dynamic_launches"]
    case["arm_identity_passed"] = bool(
        case["split_engaged"] and mono_calls
        and all(c.get("compiled_identity_observed")
                and c.get("nvfp4_split_materialized") is False for c in mono_calls)
    )
    # Even unengaged shapes retain correctness diagnostics, never an A/B ratio.
    for arm in arms.values():
        arm["graph"].replay()
    torch.cuda.synchronize()
    case["correctness"] = _correctness(arms, oracle)
    case["graph_check"] = _graph_check(arms, oracle)
    if not (case["arm_identity_passed"] and case["correctness"]["passed"]
            and case["graph_check"]["passed"]):
        case["status"] = "failed_correctness_or_graph" if not (
            case["correctness"]["passed"] and case["graph_check"]["passed"]
        ) else "split_not_engaged_or_unverified"
        return

    addresses = _addresses(arms)
    allocated = torch.cuda.memory_allocated()
    allocation_count = torch.cuda.memory_stats()["allocation.all.allocated"]
    for round_idx in range(args.rounds):
        order = ("monolithic", "split") if round_idx % 2 == 0 else ("split", "monolithic")
        case["round_order"].append(list(order))
        for name in order:
            samples = _time_launch(arms[name]["graph"].replay,
                                   warmup=args.warmup, iterations=args.iters)
            case["samples_us"][name].append(samples)
    case["gpu_mode_active"] = {
        name: _sample_under_load(arm["graph"].replay) for name, arm in arms.items()
    }
    case["timing_allocation_count_before"] = allocation_count
    case["timing_allocation_count_after"] = torch.cuda.memory_stats()["allocation.all.allocated"]
    case["timing_allocation_stable"] = (
        torch.cuda.memory_allocated() == allocated
        and case["timing_allocation_count_after"] == allocation_count
    )
    case["post_timing_correctness"] = _correctness(arms, oracle)
    case["timing_addresses_stable"] = _addresses(arms) == addresses
    case["gpu_mode_check"] = _gpu_mode_check(
        case["gpu_mode_active"],
        initial_snapshot=initial_snapshot,
        expected_uuid=str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid),
        allow_software_power_cap=args.allow_software_power_cap,
        max_sm_clock_delta_percent=args.max_sm_clock_delta_percent,
    )
    case["qualified"] = bool(
        case["post_timing_correctness"]["passed"] and case["timing_allocation_stable"]
        and case["timing_addresses_stable"]
        and case["gpu_mode_check"]["passed"]
    )
    if not case["qualified"]:
        case["status"] = (
            "failed_gpu_mode_check"
            if not case["gpu_mode_check"]["passed"] else "failed_post_timing_check"
        )
        return
    case["status"] = "diagnostic_qualified"
    case["median_us"] = {
        name: statistics.median(statistics.median(s) for s in rounds)
        for name, rounds in case["samples_us"].items()
    }
    case["speedup_split_over_mono"] = (
        case["median_us"]["monolithic"] / case["median_us"]["split"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", action="append", type=str, default=None,
                        help="E:K:n:topk:M (repeatable); defaults cover NVFP4 prefill bands")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument(
        "--fast-math", action=argparse.BooleanOptionalAction, default=False,
        help="Enable approximate math; defaults off to preserve the original A/B precision mode.",
    )
    parser.add_argument(
        "--allow-software-power-cap", action="store_true",
        help="Allow only 0x0/0x4 throttle masks for interleaved Max-Q diagnostics, not release evidence.",
    )
    parser.add_argument("--max-sm-clock-delta-percent", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iters < 1 or args.rounds < 1 or args.warmup < 0:
        parser.error("iters/rounds must be positive and warmup must be nonnegative")
    if not math.isfinite(args.max_sm_clock_delta_percent) or args.max_sm_clock_delta_percent < 0:
        parser.error("max-sm-clock-delta-percent must be finite and nonnegative")
    if args.allow_software_power_cap and args.rounds < 2:
        parser.error("software-power-cap diagnostics require at least two alternating rounds")
    shapes = args.shape or [
        "8:4096:2048:2:2048", "8:4096:2048:2:4096", "8:4096:2048:2:8192",
        "64:4096:1024:8:2048", "64:4096:1024:8:4096", "64:4096:1024:8:8192",
    ]
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    argv = [sys.executable, os.path.relpath(Path(sys.argv[0]).resolve(), ROOT), *sys.argv[1:]]
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    report = {
        "schema": "b12x.moe.nvfp4_split_materialized.benchmark", "version": 4,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "argv": argv, "command": shlex.join(argv),
        "command_cwd": str(ROOT), "invocation_cwd": str(Path.cwd().resolve()),
        "worktree_path": str(ROOT), "commit": _git("rev-parse", "HEAD"),
        "commit_short": _git("rev-parse", "--short", "HEAD"),
        "worktree_status": _git("status", "--porcelain") or "clean",
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "gpu_name": properties.name,
        "gpu_device": {
            "logical_index": torch.cuda.current_device(),
            "uuid": str(getattr(properties, "uuid", "")),
            "compute_capability": [properties.major, properties.minor],
            "multiprocessor_count": properties.multi_processor_count,
            "total_memory": properties.total_memory,
        },
        "gpu_snapshot": nvidia_smi_gpu_mode_snapshot(),
        "package_versions": _package_versions(),
        "env": {key: value for key, value in os.environ.items()
                if key.startswith(("B12X_", "CUDA_", "NVIDIA_"))},
        "arm_settings": {"monolithic": {SPLIT_ENV: "0"}, "split": {SPLIT_ENV: "1"}},
        "timed_path": "one captured production fused_moe.run per graph.replay",
        "gpu_mode_policy": {
            "qualification": "diagnostic only; not formal release evidence",
            "allowed_throttle_masks": ["0x0", "0x4"] if args.allow_software_power_cap else ["0x0"],
            "required_pstate": "P1",
            "require_same_physical_gpu": True,
            "require_same_memory_clock": True,
            "max_sm_clock_delta_percent": args.max_sm_clock_delta_percent,
            "sm_clock_delta_reference": "minimum arm SM clock",
            "arm_order": "alternating lead arm each round",
            "transition_reference": "initial snapshot to each active arm, plus inter-arm comparison",
            "clock_condition_scope": "active arms; the initial snapshot may be idle",
        },
        "source_hash_scope": "all loaded local modules plus benchmark and renderer, including input/oracle helpers",
        "iters": args.iters, "warmup": args.warmup, "rounds": args.rounds, "seed": args.seed,
        "fast_math": args.fast_math,
        "cases": [],
    }

    def save():
        ratios = [case["speedup_split_over_mono"] for case in report["cases"]
                  if case.get("qualified")]
        report["summary"] = {
            "qualified_speedups": len(ratios),
            "geomean_speedup_split_over_mono": statistics.geometric_mean(ratios) if ratios else None,
            "ratio_direction": RATIO_DIRECTION,
            "note": "Unengaged/unverified arms and failed checks have no headline timing. "
                    "Raw samples survive post-timing failures as unqualified diagnostics only.",
        }
        report["source_sha256"] = _source_hashes()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        os.replace(tmp, args.output)

    compiled_identities = {}
    save()
    for spec in shapes:
        E, K, n, top_k, M = (int(value) for value in spec.split(":"))
        if min(E, K, n, top_k, M) < 1 or top_k > E:
            parser.error(f"invalid shape {spec}")
        case = {
            "shape": {"E": E, "K": K, "n": n, "top_k": top_k, "M": M},
            "routed_rows": M * top_k, "split_engaged": False, "qualified": False,
            "status": "preparing", "correctness": {}, "round_order": [],
            "samples_us": {"monolithic": [], "split": []},
            "median_us": None, "speedup_split_over_mono": None,
        }
        report["cases"].append(case)
        try:
            _run_case(case, args, compiled_identities, report["gpu_snapshot"])
        except Exception as error:
            case["status"] = "error"
            case["error"] = f"{type(error).__name__}: {error}"
            save()
            raise
        save()
        if case["qualified"]:
            print(f"{spec}: {case['median_us']} ratio={case['speedup_split_over_mono']:.3f}", flush=True)
        else:
            print(f"{spec}: {case['status']}; headline timings withheld", flush=True)
    print(f"qualified summary: {report['summary']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
