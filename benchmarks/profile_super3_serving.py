"""Collect HTTP timings and separate CUPTI traces from a Super3 vLLM server.

Start serve-super3-packed-spark.sh with the Torch profiler configured, no
iteration limit, ignore_frontend=true, and stacks/shapes/memory disabled.
Artifacts belong outside the source tree. No kernel or serving policy is changed.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import uuid

from benchmark_v41_serving import prompt_tokens, repository_state, request_json


def gpu_snapshot():
    command = [
        "nvidia-smi",
        "--query-gpu=uuid,name,compute_cap,pstate,clocks.sm,clocks.mem,"
        "clocks_event_reasons.active,power.draw",
        "--format=csv",
    ]
    return {"time_ns": time.time_ns(), "raw": subprocess.check_output(command).decode()}


def control(base, route):
    request = urllib.request.Request(base + route, data=b"", method="POST")
    with urllib.request.urlopen(request, timeout=900) as response:
        response.read()
        assert response.status == 200


def stream(base, route, payload):
    payload = dict(
        payload,
        temperature=0,
        seed=0,
        cache_salt=uuid.uuid4().hex,
        stream=True,
        stream_options={"include_usage": True},
        return_token_ids=True,
    )
    request = urllib.request.Request(
        base + route,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    chunks = []
    usage = None
    finish = None
    with urllib.request.urlopen(request, timeout=900) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            raw = line[6:].strip()
            if raw == b"[DONE]":
                break
            event = json.loads(raw)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                text = choice.get("text") or delta.get("content") or ""
                ids = choice.get("token_ids") or []
                if text or ids:
                    chunks.append(
                        dict(
                            elapsed_s=time.perf_counter() - start,
                            text=text,
                            token_ids=ids,
                        )
                    )
                finish = choice.get("finish_reason") or finish
    elapsed = time.perf_counter() - start
    assert usage and chunks, (usage, chunks)
    assert usage["completion_tokens"] == payload["max_tokens"], usage
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    assert cached == 0, usage
    ids = [token for chunk in chunks for token in chunk["token_ids"]]
    assert len(ids) == usage["completion_tokens"], (len(ids), usage)
    first = chunks[0]["elapsed_s"]
    remaining = len(ids) - len(chunks[0]["token_ids"])
    return dict(
        request=payload,
        ttft_s=first,
        elapsed_s=elapsed,
        decode_s_per_token=(chunks[-1]["elapsed_s"] - first) / remaining
        if remaining
        else None,
        usage=usage,
        finish_reason=finish,
        text="".join(c["text"] for c in chunks),
        chunks=chunks,
    )


def image_content():
    from PIL import Image, ImageDraw

    picture = Image.new("RGB", (512, 512), "white")
    ImageDraw.Draw(picture).rectangle((96, 96, 416, 416), fill="red")
    # Different image hashes prevent cross-request encoder-cache reuse.
    for x, value in enumerate(uuid.uuid4().bytes):
        picture.putpixel((x, 0), (value, value, value))
    data = io.BytesIO()
    picture.save(data, format="PNG")
    return [
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64,"
                + base64.b64encode(data.getvalue()).decode()
            },
        },
        {"type": "text", "text": "What color is the square? Answer only the color."},
    ]


def source_state(root, output):
    result = repository_state(root)
    patch = subprocess.check_output(["git", "-C", str(root), "diff", "--binary"])
    (output / (root.name + ".patch")).write_bytes(patch)
    result["diff_sha256"] = hashlib.sha256(patch).hexdigest()
    package = root / ("b12x" if root.name == "b12x" else "vllm")
    result["source_sha256"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*.py"))
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="super3-packed")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--vllm-root", default="/home/luke/projects/vllm-upstream-main", type=Path
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    assert args.repeats > 0
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = args.base_url.rstrip("/")
    root = Path(__file__).resolve().parents[1]
    result = dict(
        command=[sys.executable, *sys.argv],
        cwd=str(Path.cwd()),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        repositories=[source_state(p, out) for p in (root, args.vllm_root)],
        gpu_before=gpu_snapshot(),
        records=[],
        correctness={},
        timing_scope="Unprofiled HTTP latency; traced GPU times separately.",
        ratio_direction="No A/B speedup comparison.",
    )
    result["toolchain"] = {}
    for package in ("torch", "vllm", "nvidia-cutlass-dsl", "triton"):
        with suppress(importlib.metadata.PackageNotFoundError):
            result["toolchain"][package] = importlib.metadata.version(package)

    def save():
        (out / "measurements.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    for _ in range(120):
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as response:
                if response.status == 200:
                    break
        except OSError:
            time.sleep(5)
    else:
        raise RuntimeError("Server did not become ready within ten minutes")

    for name, content, expected in (
        ("arithmetic", "What is 17 + 25? Answer only the number.", "42"),
        ("vision", image_content(), "red"),
    ):
        reply = request_json(
            base,
            "/v1/chat/completions",
            dict(
                model=args.model,
                messages=[dict(role="user", content=content)],
                temperature=0,
                max_tokens=32,
                chat_template_kwargs={"enable_thinking": False},
            ),
        )
        text = reply["choices"][0]["message"]["content"] or ""
        assert expected in text.lower(), reply
        result["correctness"][name] = reply
        save()
        print("correctness", name, repr(text), flush=True)

    long = prompt_tokens(
        base,
        args.model,
        "Background notes: "
        + "The oak tree has green leaves. " * 800
        + "\nWrite a long detailed story about a forest expedition.",
        {"enable_thinking": False},
    )
    prompts = {n: long[: n - 64] + long[-64:] for n in (256, 2048)}
    cases = [
        ("decode_b1", 256, 64, 1),
        ("prefill_2048", 2048, 1, 1),
        ("decode_b2", 256, 32, 2),
        ("vision_prefill", 0, 1, 1),
    ]

    def run_case(case):
        name, length, generated, concurrency = case
        if length:
            route = "/v1/completions"
            payload = dict(
                model=args.model,
                prompt=prompts[length],
                max_tokens=generated,
                ignore_eos=True,
            )
        else:
            route = "/v1/chat/completions"
            payload = dict(
                model=args.model,
                messages=[dict(role="user", content=image_content())],
                max_tokens=generated,
                ignore_eos=True,
                chat_template_kwargs={"enable_thinking": False},
            )
        if concurrency == 1:
            return [stream(base, route, payload)]
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            jobs = [
                pool.submit(stream, base, route, payload) for _ in range(concurrency)
            ]
            return [job.result() for job in jobs]

    telemetry = (out / "gpu-telemetry.csv").open("w")
    monitor = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,uuid,pstate,clocks.sm,clocks.mem,"
            "clocks_event_reasons.active,power.draw",
            "--format=csv",
            "-l",
            "1",
        ],
        stdout=telemetry,
        stderr=subprocess.STDOUT,
    )
    try:
        for case in cases:
            run_case(case)
            print("warmed", case[0], flush=True)
        for repeat in range(args.repeats):
            for case in cases:
                record = dict(
                    case=case[0],
                    repeat=repeat,
                    profiled=False,
                    gpu_before=gpu_snapshot(),
                    requests=run_case(case),
                    gpu_after=gpu_snapshot(),
                )
                result["records"].append(record)
                save()
                print(
                    "timing",
                    case[0],
                    repeat,
                    [
                        (round(r["ttft_s"], 4), round(r["elapsed_s"], 4))
                        for r in record["requests"]
                    ],
                    flush=True,
                )
        for case in cases:
            previous = set((out / "traces").glob("*.pt.trace.json.gz"))
            control(base, "/start_profile")
            try:
                replies = run_case(case)
            finally:
                control(base, "/stop_profile")
            traces = sorted(set((out / "traces").glob("*.pt.trace.json.gz")) - previous)
            assert traces, "No new worker trace was written"
            record = dict(case=case[0], profiled=True, requests=replies, traces=[])
            for index, trace in enumerate(traces):
                target = trace.with_name(f"{case[0]}-{index}.pt.trace.json.gz")
                trace.rename(target)
                record["traces"].append(str(target))
            result["records"].append(record)
            save()
            print("traced", case[0], record["traces"], flush=True)
        result["complete"] = True
    finally:
        monitor.terminate()
        monitor.wait(timeout=10)
        telemetry.close()
        result["gpu_after"] = gpu_snapshot()
        save()


if __name__ == "__main__":
    main()
