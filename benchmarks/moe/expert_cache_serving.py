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


async def run(args):
    import torch
    from vllm import AsyncEngineArgs, SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM
    from b12x.integration.vllm.residency_epoch import VllmResidencyEpochs
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
        kernel_config={"enable_b12x_autotune": False, "moe_backend": "b12x"},
        compilation_config={
            "cudagraph_capture_sizes": sorted({1, args.concurrency}),
            "mode": 3 if args.inductor and not args.eager else 0,
            "cudagraph_mode": "NONE" if args.eager else "FULL_DECODE_ONLY",
        },
        additional_config={"b12x_expert_cache": settings},
        worker_extension_cls="b12x.integration.vllm.residency_epoch.ResidencyEpochWorkerExtension",
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
        settings=settings,
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
    controller, epoch_task, traffic_task = None, None, None
    try:
        engine = AsyncLLM.from_engine_args(engine_args)
        status = (await engine.collective_rpc("b12x_expert_cache_status"))[0]
        if not args.eager and not status["graphs"]:
            raise AssertionError(
                "graph-enabled serving did not retain a captured graph"
            )
        record("prepared", status=status)
        if args.mode == "adaptive":
            controller = VllmResidencyEpochs(
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
            )
            record("epoch", receipt=await controller.run())
        prompts = [
            json.loads(line)
            for line in args.prompts.read_text().splitlines()
            if line.strip()
        ]
        if not prompts:
            raise ValueError("real serving benchmark requires a nonempty prompt corpus")
        tokens_completed, next_epoch = 0, args.epoch_tokens
        finished = asyncio.Event()

        async def epochs():
            nonlocal next_epoch
            while not finished.is_set():
                await asyncio.sleep(0.01)
                if tokens_completed >= next_epoch:
                    receipt = await controller.run()
                    record(
                        "epoch", receipt=receipt, output_tokens_so_far=tokens_completed
                    )
                    next_epoch = tokens_completed + args.epoch_tokens

        if controller is not None:
            epoch_task = asyncio.create_task(epochs())
        semaphore = asyncio.Semaphore(args.concurrency)

        async def request(index, prompt):
            nonlocal tokens_completed
            async with semaphore:
                start = time.perf_counter_ns()
                start_wall_ns = time.time_ns()
                events, previous, final = [], 0, None
                async for result in engine.generate(
                    prompt["text"],
                    SamplingParams(
                        temperature=0, max_tokens=args.tokens, ignore_eos=True
                    ),
                    request_id=f"cache-{index}",
                ):
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
        after = (await engine.collective_rpc("b12x_expert_cache_status"))[0]
        if status["graphs"] != after["graphs"]:
            raise AssertionError("serving recaptured or replaced a CUDA graph")
        for name, before in status["layers"].items():
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
    p.add_argument("--mode", choices=("profile", "static", "adaptive"), required=True)
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
    p.add_argument("--eager", action="store_true")
    p.add_argument(
        "--inductor",
        action="store_true",
        help="Also enable vLLM Inductor compilation; requires matching engine extensions",
    )
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(
            "serving receipts are append-only; choose a new output path"
        )
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
