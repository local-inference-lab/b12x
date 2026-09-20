"""Real AsyncLLM cache calibration and static/adaptive serving receipts.

Requires the companion loader branch. Each arm starts a fresh engine. Prompts
are supplied as JSONL {text, workload}; no synthetic routes or gate weights are
substituted. Static has no observer. Epoch pauses are included in request times.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
import platform
import subprocess
import time


def maintenance_check_interval(current, *, minimum, maximum, health):
    """Experimental host cadence; pressure and absent observations reset it."""
    if not 0 < minimum <= current <= maximum:
        raise ValueError("maintenance intervals must be positive and ordered")
    return min(maximum, current * 2) if health == "healthy" else minimum


async def run(args):
    import torch
    from vllm import AsyncEngineArgs, SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM
    from b12x.integration.vllm.residency_epoch import VllmResidencyEpochs
    from b12x.integration.vllm.residency_maintenance import VllmResidencyMaintenance
    from b12x.moe.residency import ResidencyCacheConfig, ResidencyEpochBudget

    settings = dict(
        mode=args.mode,
        activation="w4a16",
        profile_path=str(args.profile),
        workload=args.workload,
        expert_device_bytes=args.cache_gib << 30,
        host_bytes=args.host_gib << 30,
        kv_reserved_bytes=args.kv_gib << 30,
        graph_reserved_bytes=512 << 20,
        device_safety_bytes=1 << 30,
        host_safety_bytes=1 << 30,
        max_pairs_per_layer=args.layer_pairs,
        health_probes=args.control == "health",
        history_depth=args.history_depth,
        anchor_health=args.anchor_health or args.anchor_advantage is not None,
    )
    engine_args = AsyncEngineArgs(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.context,
        max_num_seqs=args.concurrency,
        max_num_batched_tokens=args.capacity,
        kv_cache_memory_bytes=args.kv_gib << 30,
        enable_prefix_caching=False,
        kv_cache_dtype="bfloat16",
        attention_config={"backend": "FLASHINFER"},
        enforce_eager=args.eager,
        enable_chunked_prefill=True,
        kernel_config={
            "enable_b12x_autotune": False,
            "moe_backend": "flashinfer_cutlass" if args.mode == "native" else "b12x",
        },
        compilation_config={
            "cudagraph_capture_sizes": sorted({1, args.concurrency}),
            "mode": 3 if args.inductor and not args.eager else 0,
            "cudagraph_mode": "NONE" if args.eager else "FULL_DECODE_ONLY",
        },
        additional_config={}
        if args.mode == "native"
        else {"b12x_expert_cache": settings},
        worker_extension_cls=(
            "b12x.testing.vllm_execution_trace.ExecutionTraceWorker"
            if args.execution_trace
            else "b12x.integration.vllm.residency_epoch.ResidencyEpochWorkerExtension"
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def record(kind, **values):
        with args.output.open("a") as stream:
            stream.write(
                json.dumps(
                    {"kind": kind, "time_ns": time.time_ns(), **values}, default=str
                )
                + "\n"
            )
            stream.flush()

    record(
        "configuration",
        arguments=vars(args),
        settings=settings if args.mode != "native" else None,
        numerical_recipe=(
            "ordinary_modelopt_nvfp4_a4"
            if args.mode == "native"
            else "cache_w4a16_bf16_whole_k"
        ),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        host=platform.node(),
        source_receipt=os.environ.get("B12X_SERVING_SOURCE_RECEIPT"),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        topology=subprocess.run(
            ["nvidia-smi", "topo", "-m"], capture_output=True, text=True
        ).stdout,
        gpu=subprocess.run(
            ["nvidia-smi", "-q", "-x"], capture_output=True, text=True
        ).stdout,
    )
    engine = None
    from b12x.moe.residency import RoutingAnchorThresholds
    anchor_gate = (RoutingAnchorThresholds(
        advantage_fraction=args.anchor_advantage,
        minimum_layer_fraction=args.anchor_breadth,
    ) if args.anchor_advantage is not None else None)
    controller, epoch_task, traffic_task = None, None, None
    try:
        engine = AsyncLLM.from_engine_args(engine_args)
        status = (
            (await engine.collective_rpc("b12x_expert_cache_status"))[0]
            if args.mode != "native"
            else None
        )
        if status is not None and not args.eager and not status["graphs"]:
            raise AssertionError(
                "graph-enabled serving did not retain a captured graph"
            )
        record("prepared", status=status)
        if args.mode == "adaptive" and args.control != "observe":
            controller_type = (
                VllmResidencyMaintenance
                if args.control in ("maintenance", "health")
                else VllmResidencyEpochs
            )
            extra = (
                {
                    "cold_fraction_threshold": None
                    if args.control == "health"
                    else args.cold_threshold,
                    "policy_diagnostics": args.policy_diagnostics,
                    "anchor_thresholds": anchor_gate,
                    "recenter_budget": (
                        ResidencyEpochBudget(max_pairs=args.recenter_pairs,
                                             max_copy_bytes=args.recenter_mib << 20)
                        if args.recenter_pairs is not None else None
                    ),
                }
                if args.control in ("maintenance", "health")
                else {}
            )
            controller = controller_type(
                engine,
                configs={
                    name: ResidencyCacheConfig(
                        max_pairs=args.layer_pairs,
                        scoring="decayed_lfu",
                        minimum_cold_selections=2,
                        minimum_score_gain=1,
                        minimum_residency_windows=1,
                    )
                    for name, layer in status["layers"].items()
                    if layer["max_pairs"]
                },
                budget=ResidencyEpochBudget(
                    max_pairs=args.epoch_pairs, max_copy_bytes=args.epoch_mib << 20
                ),
                **extra,
            )
            record(
                "maintenance" if args.control in ("maintenance", "health") else "epoch",
                receipt=await controller.run(),
            )
        prompts = [
            json.loads(line)
            for line in args.prompts.read_text().splitlines()
            if line.strip()
        ]
        if not prompts:
            raise ValueError("real serving benchmark requires a nonempty prompt corpus")
        if args.execution_trace:
            record(
                "trace_begin",
                result=await engine.collective_rpc(
                    "begin_execution_trace",
                    kwargs={
                        "request_prefixes": args.trace_requests,
                        "output_limit": args.trace_output_limit,
                        "modules": args.trace_modules,
                    },
                ),
            )
        health_probe = thresholds = None
        last_maintenance_tokens = 0
        if args.control == "health":
            from b12x.integration.vllm.residency_health import VllmResidencyHealth
            from b12x.moe.residency.health import RoutingHealthThresholds

            health_probe = VllmResidencyHealth(
                engine, record_history=bool(args.history_depth)
            )
            thresholds = RoutingHealthThresholds(
                cold_fraction=args.cold_threshold,
                minimum_layer_fraction=args.health_layer_breadth,
                layer_cold_fraction=args.health_layer_threshold,
            )
        tokens_completed, next_epoch = 0, args.epoch_tokens
        check_interval = args.epoch_tokens
        finished = asyncio.Event()
        control_lock = asyncio.Lock()

        async def epochs():
            nonlocal next_epoch, check_interval, last_maintenance_tokens
            while not finished.is_set():
                await asyncio.sleep(0.01)
                if tokens_completed >= next_epoch:
                    async with control_lock:
                        reason = None
                        if health_probe is not None:
                            probe = await health_probe.probe()
                            assessment = thresholds.assess(probe["summary"])
                            if anchor_gate is not None:
                                assessment.update(anchor_gate.assess(probe["summary"]))
                            record(
                                "health_probe",
                                receipt=probe,
                                assessment=assessment,
                                output_tokens_so_far=tokens_completed,
                            )
                            reason = (
                                "anchor_advantage"
                                if assessment.get("anchor_better", False)
                                else "pressure"
                                if assessment["health"] == "pressure"
                                else "maximum_interval"
                                if tokens_completed - last_maintenance_tokens
                                >= args.health_max_tokens
                                else None
                            )
                            if reason is None:
                                next_epoch = tokens_completed + args.epoch_tokens
                                continue
                        receipt = await controller.run(movement_mode="recenter") if reason == "anchor_advantage" else await controller.run()
                        last_maintenance_tokens = tokens_completed
                    if args.healthy_check_max_tokens is not None:
                        check_interval = maintenance_check_interval(
                            check_interval,
                            minimum=args.epoch_tokens,
                            maximum=args.healthy_check_max_tokens,
                            health=receipt["worker"].get("health"),
                        )
                    record(
                        "maintenance"
                        if args.control in ("maintenance", "health")
                        else "epoch",
                        receipt=receipt,
                        output_tokens_so_far=tokens_completed,
                        next_check_tokens=check_interval,
                        trigger=reason,
                    )
                    next_epoch = tokens_completed + check_interval

        if controller is not None:
            epoch_task = asyncio.create_task(epochs())
        semaphore = asyncio.Semaphore(args.concurrency)

        async def admitted_stream(index, prompt, admitted):
            queue = await engine.add_request(
                f"cache-{index}",
                prompt["text"],
                SamplingParams(temperature=0, max_tokens=args.tokens, ignore_eos=True),
            )
            admitted.set()
            complete = False
            try:
                while not complete:
                    result = queue.get_nowait() or await queue.get()
                    complete = result.finished
                    yield result
            finally:
                if not complete:
                    await engine.abort(queue.request_id, internal=True)

        async def request(index, prompt, admitted=None):
            nonlocal tokens_completed
            async with semaphore:
                start = time.perf_counter_ns()
                start_wall_ns = time.time_ns()
                events, previous, final = [], 0, None
                stream = (
                    admitted_stream(index, prompt, admitted)
                    if admitted is not None
                    else engine.generate(
                        prompt["text"],
                        SamplingParams(
                            temperature=0, max_tokens=args.tokens, ignore_eos=True
                        ),
                        request_id=f"cache-{index}",
                    )
                )
                async for result in stream:
                    now = time.perf_counter_ns()
                    final = result.outputs[0]
                    total = len(final.token_ids)
                    if total > previous:
                        events.append(
                            {"ns": now - start, "new_tokens": total - previous}
                        )
                        tokens_completed += total - previous
                        previous = total
                elapsed = time.perf_counter_ns() - start
                first = events[0]["ns"]
                record(
                    "request",
                    index=index,
                    workload=prompt.get("workload", "unspecified"),
                    start_wall_ns=start_wall_ns,
                    output_tokens=previous,
                    elapsed_ns=elapsed,
                    ttft_ns=first,
                    events=events,
                    decode_tokens_per_s=(previous - 1)
                    * 1e9
                    / (events[-1]["ns"] - first)
                    if previous > 1
                    else None,
                    token_ids=list(final.token_ids),
                    text=final.text,
                )

        async def traffic():
            # Sequential corpus groups preserve explicit workload transitions.
            for offset in range(0, len(prompts), args.concurrency):
                indices = range(offset, min(len(prompts), offset + args.concurrency))
                if args.admission == "together":
                    # Diagnostic control: acknowledge every add while scheduling
                    # is paused, so asynchronous tokenization cannot split admission.
                    tasks = []
                    try:
                        async with control_lock:
                            await engine.pause_generation(
                                mode="keep", clear_cache=False
                            )
                            admitted = [asyncio.Event() for _ in indices]
                            tasks = [
                                asyncio.create_task(request(i, prompts[i], event))
                                for i, event in zip(indices, admitted, strict=True)
                            ]
                            await asyncio.wait_for(
                                asyncio.gather(*(e.wait() for e in admitted)), 60
                            )
                            await engine.resume_generation()
                        await asyncio.gather(*tasks)
                    finally:
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                    continue
                await asyncio.gather(
                    *(
                        request(i, prompts[i])
                        for i in range(
                            offset, min(len(prompts), offset + args.concurrency)
                        )
                    )
                )

        begin = time.perf_counter_ns()
        traffic_task = asyncio.create_task(traffic())
        if epoch_task is not None:
            done, _ = await asyncio.wait(
                (traffic_task, epoch_task), return_when=asyncio.FIRST_COMPLETED
            )
            if epoch_task in done:
                await epoch_task
                raise RuntimeError(
                    "residency epoch loop exited before traffic completed"
                )
        await traffic_task
        request_elapsed = time.perf_counter_ns() - begin
        finished.set()
        if epoch_task is not None:
            await epoch_task
        elapsed = time.perf_counter_ns() - begin
        if args.measure_control_floor:
            scheduler, maintenance = [], []
            for _ in range(20):
                started = time.perf_counter_ns()
                await engine.is_paused()
                scheduler.append(time.perf_counter_ns() - started)
                maintenance.append(await controller.run())
            started = time.perf_counter_ns()
            readback = await engine.collective_rpc("measure_idle_readback")
            record(
                "idle_control_floor",
                scheduler_rpc_ns=scheduler,
                maintenance=maintenance,
                readback=readback,
                readback_rpc_ns=time.perf_counter_ns() - started,
            )
        if args.execution_trace:
            record(
                "trace_end",
                result=await engine.collective_rpc(
                    "end_execution_trace", args=(str(args.execution_trace),)
                ),
            )
        record(
            "serving",
            output_tokens=tokens_completed,
            elapsed_ns=elapsed,
            request_elapsed_ns=request_elapsed,
            aggregate_tokens_per_s=tokens_completed * 1e9 / elapsed,
        )
        if args.mode == "profile":
            start = time.perf_counter_ns()
            await engine.pause_generation(mode="keep", clear_cache=False)
            record(
                "profile",
                result=await engine.collective_rpc(
                    "b12x_expert_cache_save_profile", kwargs={"quiescent": True}
                ),
                pause_ns=time.perf_counter_ns() - start,
            )
            await engine.resume_generation()
        after = (
            (await engine.collective_rpc("b12x_expert_cache_status"))[0]
            if args.mode != "native"
            else None
        )
        if status is not None and status["graphs"] != after["graphs"]:
            raise AssertionError("serving recaptured or replaced a CUDA graph")
        for name, before in status["layers"].items() if status else ():
            if before["pointers"] != after["layers"][name]["pointers"]:
                raise AssertionError("serving cache changed a captured pointer")
        record("complete", status=after)
    except BaseException as error:
        record(
            "failure",
            error=repr(error),
            epoch=controller.last_receipt if controller else None,
        )
        raise
    finally:
        if epoch_task is not None and not epoch_task.done():
            epoch_task.cancel()
        if traffic_task is not None and not traffic_task.done():
            traffic_task.cancel()
        if engine is not None:
            engine.shutdown()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument(
        "--mode", choices=("profile", "static", "adaptive", "native"), required=True
    )
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workload", default="general")
    p.add_argument("--cache-gib", type=int, default=8)
    p.add_argument("--host-gib", type=int, default=40)
    p.add_argument("--kv-gib", type=int, default=2)
    p.add_argument("--context", type=int, default=2048)
    p.add_argument("--capacity", type=int, default=64)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--epoch-tokens", type=int, default=32)
    p.add_argument("--layer-pairs", type=int, default=2)
    p.add_argument("--epoch-pairs", type=int, default=16)
    p.add_argument("--epoch-mib", type=int, default=64)
    p.add_argument("--recenter-pairs", type=int, help="Explicit recovery pair cap; omission uses the normal epoch budget")
    p.add_argument("--recenter-mib", type=int, help="Explicit recovery copy-byte cap in MiB")
    p.add_argument(
        "--control",
        choices=("external", "observe", "maintenance", "health"),
        default="external",
        help="Observe records routing counters without policy, epochs or promotions",
    )
    p.add_argument(
        "--cold-threshold",
        type=float,
        help="Experimental maintenance cold-fraction gate; omission permits all proposals",
    )
    p.add_argument(
        "--healthy-check-max-tokens",
        type=int,
        help="Experimental healthy-check backoff cap; requires conditional maintenance",
    )
    p.add_argument(
        "--health-max-tokens",
        type=int,
        default=1024,
        help="Experimental maximum delivered-token interval for a full snapshot",
    )
    p.add_argument("--anchor-health", action="store_true", help="Read-only learned-anchor counterfactual")
    p.add_argument("--anchor-advantage", type=float, help="Experimental recovery threshold; enables anchor health")
    p.add_argument("--anchor-breadth", type=float, default=0.75)
    p.add_argument("--health-layer-breadth", type=float, default=0.0)
    p.add_argument("--health-layer-threshold", type=float, default=0.15)
    p.add_argument(
        "--policy-diagnostics",
        action="store_true",
        help="Retain full policy scores/counts at maintenance only",
    )
    p.add_argument(
        "--history-depth",
        type=int,
        default=0,
        help="Opt-in retained counter cuts; zero allocates no history",
    )
    p.add_argument("--eager", action="store_true")
    p.add_argument(
        "--execution-trace",
        type=Path,
        help="Opt-in CPU batch diagnostics; traced runs are not timing evidence",
    )
    p.add_argument(
        "--trace-requests",
        nargs="*",
        default=[],
        help="Request ID prefixes whose inputs/hidden/logits are copied for diagnostics",
    )
    p.add_argument("--trace-output-limit", type=int, default=40)
    p.add_argument(
        "--trace-modules",
        nargs="*",
        default=[],
        help="Exact module names to record during selected eager prefill steps",
    )
    p.add_argument(
        "--measure-control-floor",
        action="store_true",
        help="Measure idle readback/control after traffic; requires traced maintenance",
    )
    p.add_argument(
        "--admission",
        choices=("streamed", "together"),
        default="streamed",
        help="Together controls batch admission; its idle barrier cost remains in timings",
    )
    p.add_argument(
        "--inductor",
        action="store_true",
        help="Also enable vLLM Inductor compilation; requires matching engine extensions",
    )
    args = p.parse_args()
    if (args.recenter_pairs is None) != (args.recenter_mib is None):
        p.error("recovery requires both pair and byte limits")
    if args.recenter_pairs is not None:
        if args.anchor_advantage is None or min(args.recenter_pairs, args.recenter_mib) < 0:
            p.error("recovery budgets require anchor control and nonnegative limits")
    if (args.anchor_health or args.anchor_advantage is not None) and args.control != "health":
        p.error("anchor observation/recovery requires explicit adaptive health control")
    if args.anchor_advantage is not None:
        from b12x.moe.residency import RoutingAnchorThresholds
        RoutingAnchorThresholds(advantage_fraction=args.anchor_advantage,
                                minimum_layer_fraction=args.anchor_breadth)
    if args.history_depth < 0 or (args.history_depth and args.control != "health"):
        p.error("history requires health control and a nonnegative depth")
    if args.control == "health" and (
        args.mode != "adaptive"
        or args.cold_threshold is None
        or args.health_max_tokens < args.epoch_tokens
        or args.epoch_tokens <= 0
        or args.healthy_check_max_tokens is not None
    ):
        p.error(
            "health control requires adaptive mode, an explicit threshold and positive ordered intervals"
        )
    if args.measure_control_floor and not (
        args.execution_trace
        and args.mode == "adaptive"
        and args.control == "maintenance"
        and args.epoch_pairs == 0
    ):
        p.error(
            "control-floor diagnostic requires traced maintenance with zero movement budget"
        )
    if args.healthy_check_max_tokens is not None and (
        args.mode != "adaptive"
        or args.control != "maintenance"
        or args.cold_threshold is None
        or not 0 < args.epoch_tokens <= args.healthy_check_max_tokens
    ):
        p.error(
            "healthy backoff requires conditional maintenance and ordered positive intervals"
        )
    if args.output.exists():
        raise FileExistsError(
            "serving receipts are append-only; choose a new output path"
        )
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
