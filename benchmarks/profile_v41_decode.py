"""Capture steady-state V4.1 decode after prefill and an output-token warmup.

The server must already have a bounded Torch profiler window configured. This
collector deliberately records no latency benchmark: profiling changes timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_v41_serving import gpu_snapshot, repository_state


def profiler_control(base_url, route):
    request = urllib.request.Request(
        base_url.rstrip("/") + route, data=b"", method="POST"
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        response.read()
        return response.status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup-output-tokens", type=int, default=128)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.warmup_output_tokens < 1
        or args.max_output_tokens < args.warmup_output_tokens + 32
    ):
        parser.error("leave at least 32 output tokens after a positive warmup")
    source = json.loads(args.benchmark.read_text())
    matches = [
        record
        for record in source["records"]
        if str(record["case"]) == args.case and record["repeat"] == args.repeat
    ]
    if len(matches) != 1:
        parser.error("benchmark must contain exactly one matching case/sample")
    selected = matches[0]
    model = source["model"]["data"][0]["id"]
    request_body = {
        "model": model,
        "prompt": selected["prompt_token_ids"],
        "max_tokens": args.max_output_tokens,
        "temperature": 0,
        "seed": 41,
        "return_token_ids": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    result = {
        "command": [sys.executable, *sys.argv],
        "worktree": str(ROOT),
        "repository": repository_state(ROOT),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(args.benchmark.read_bytes()).hexdigest(),
        "case": args.case,
        "repeat": args.repeat,
        "request": request_body,
        "warmup_output_tokens": args.warmup_output_tokens,
        "scope": "Diagnostic steady-state decode profile; not a latency benchmark. Server-side launch configuration bounds the capture window.",
        "gpu_before": gpu_snapshot(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(request_body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started_profile = False
    emitted_ids = []
    text_chunks = []
    usage = None
    finish_reason = None
    failure = None
    try:
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
                    emitted_ids.extend(choice.get("token_ids") or [])
                    if choice.get("text"):
                        text_chunks.append(choice["text"])
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                if (
                    not started_profile
                    and len(emitted_ids) >= args.warmup_output_tokens
                ):
                    result["profile_start_unix_ns"] = time.time_ns()
                    result["profile_trigger_output_tokens"] = len(emitted_ids)
                    result["profile_start_status"] = profiler_control(
                        args.base_url, "/start_profile"
                    )
                    started_profile = True
    except Exception as error:
        failure = error
        result["stream_error"] = f"{type(error).__name__}: {error}"
    finally:
        if started_profile:
            try:
                result["profile_stop_status"] = profiler_control(
                    args.base_url, "/stop_profile"
                )
            except Exception as error:
                result["profile_stop_error"] = f"{type(error).__name__}: {error}"
                if failure is None:
                    failure = error
            result["profile_stop_unix_ns"] = time.time_ns()
        result.update(
            {
                "token_ids": emitted_ids,
                "text": "".join(text_chunks),
                "usage": usage,
                "finish_reason": finish_reason,
                "gpu_after": gpu_snapshot(),
            }
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if failure is not None:
        raise failure
    if not started_profile:
        raise RuntimeError("generation ended before the decode profiling trigger")
    if usage is None or not text_chunks:
        raise RuntimeError("profiled generation returned no text or usage")
    if finish_reason == "length" and len(emitted_ids) != usage["completion_tokens"]:
        raise RuntimeError("streamed token IDs disagree with final usage")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "case",
                    "repeat",
                    "profile_trigger_output_tokens",
                    "profile_start_status",
                    "profile_stop_status",
                    "usage",
                )
            }
        )
    )


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
