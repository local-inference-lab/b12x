#!/usr/bin/env python3
"""Measure a cold prefill and optionally qualify an LMCache prefix restore.

The probe avoids tokenizer-dependent prompt sizing by submitting deterministic
token IDs through the OpenAI completions endpoint. The default mode computes
the prefix once, waits for asynchronous LMCache writes to drain, clears the
local vLLM prefix cache, and submits the identical token sequence again. A pass
requires the configured restore contract, identical greedy output, and a healthy
serving endpoint after both requests. Aligned caching restores complete chunks;
semantic checkpoint caching must restore the entire prompt without recomputing
any tokens. ``--cold-only`` records the same first request against a server
without an external cache so cache-store overhead can be measured directly.
``--restore-only`` reconstructs the request recorded by a prior qualification
result and verifies a filesystem-backed restore after both serving processes
have restarted.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

from qualify_glm53_lmcache_tiers import wait_for_store

_SOURCES = ("external_kv_transfer", "local_compute", "local_cache_hit")


def request_json(
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 1800,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def reset_prefix_cache(base_url: str, timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        result = request_json(f"{base_url}/reset_prefix_cache", {})
        if result.get("success") is True:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("vLLM did not release its local prefix-cache blocks")
        time.sleep(0.1)


def check_health(base_url: str) -> None:
    with urllib.request.urlopen(f"{base_url}/health", timeout=30) as response:
        if response.status >= 400:
            raise RuntimeError(f"serving health endpoint returned {response.status}")


def restore_checks(
    sources: dict[str, int], prompt_tokens: int, chunk_size: int, contract: str
) -> dict[str, bool]:
    """Require source attribution matching the selected external cache contract."""
    if contract == "semantic":
        return {
            "restore_covers_exact_prompt": (
                sources["external_kv_transfer"] == prompt_tokens
            ),
            "restore_does_not_recompute": sources["local_compute"] == 0,
            "restore_does_not_use_local_prefix_cache": sources["local_cache_hit"] == 0,
        }
    expected = prompt_tokens // chunk_size * chunk_size
    return {
        "restore_is_chunk_aligned": (
            sources["external_kv_transfer"] > 0
            and sources["external_kv_transfer"] % chunk_size == 0
        ),
        "restore_covers_complete_chunks": sources["external_kv_transfer"] == expected,
        "restore_recomputes_only_suffix": (
            sources["local_compute"] == prompt_tokens - expected
        ),
        "restore_does_not_use_local_prefix_cache": sources["local_cache_hit"] == 0,
    }


def metrics(base_url: str) -> dict[str, float]:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=30) as response:
        text = response.read().decode()
    metric_name = "vllm:prompt_tokens_by_source_total"
    values = {source: 0.0 for source in _SOURCES}
    for line in text.splitlines():
        if not line.startswith(metric_name + "{"):
            continue
        for source in _SOURCES:
            if f'source="{source}"' in line:
                values[source] = float(line.rsplit(" ", 1)[1])
    return values


def delta(before: dict[str, float], after: dict[str, float]) -> dict[str, int]:
    return {key: round(after[key] - before[key]) for key in _SOURCES}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5051")
    parser.add_argument("--cache-url", default="http://127.0.0.1:8085")
    parser.add_argument("--model", default="GLM-5.3-Flash-NVFP4")
    parser.add_argument("--prompt-tokens", type=int, default=1_000_000)
    parser.add_argument("--nonce", type=int)
    parser.add_argument("--cold-only", action="store_true")
    parser.add_argument("--restore-only", action="store_true")
    parser.add_argument("--reference-result", type=Path)
    parser.add_argument(
        "--checkpoint-contract", choices=("aligned", "semantic"), default="aligned"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cold_only and args.restore_only:
        parser.error("--cold-only and --restore-only are mutually exclusive")
    if args.restore_only and args.reference_result is None:
        parser.error("--restore-only requires --reference-result")
    if args.prompt_tokens < 8:
        parser.error("--prompt-tokens must be at least eight")

    nonce = args.nonce if args.nonce is not None else time.time_ns()
    nonce_ids = [1000 + int(value) for value in nonce.to_bytes(8, "little")]
    pattern = tuple(range(1400, 1527))
    token_ids = nonce_ids + [
        pattern[index % len(pattern)]
        for index in range(args.prompt_tokens - len(nonce_ids))
    ]
    payload = {
        "model": args.model,
        "prompt": token_ids,
        "temperature": 0,
        "max_tokens": 1,
        "seed": 43,
    }

    if args.restore_only:
        assert args.reference_result is not None
        reference = json.loads(args.reference_result.read_text())
        reference_conditions = reference["conditions"]
        if reference_conditions["model"] != args.model:
            parser.error("--model does not match --reference-result")
        if reference_conditions["prompt_tokens"] != args.prompt_tokens:
            parser.error("--prompt-tokens does not match --reference-result")
        if reference_conditions["nonce"] != nonce:
            parser.error("--nonce does not match --reference-result")
        if (
            reference_conditions.get("checkpoint_contract", "aligned")
            != args.checkpoint_contract
        ):
            parser.error("--checkpoint-contract does not match --reference-result")

        status = request_json(f"{args.cache_url}/status")
        chunk_size = int(status["chunk_size"])
        reset_prefix_cache(args.base_url)
        before_restore = metrics(args.base_url)
        restore_started = time.monotonic()
        restored = request_json(f"{args.base_url}/v1/completions", payload)
        restore_seconds = time.monotonic() - restore_started
        restore_delta = delta(before_restore, metrics(args.base_url))
        check_health(args.base_url)

        reference_text = reference["cold"]["text"]
        stored_bytes = int(reference_conditions.get("stored_bytes", 0))
        checks = {
            **restore_checks(
                restore_delta, args.prompt_tokens, chunk_size, args.checkpoint_contract
            ),
            "greedy_output_matches_reference": (
                restored["choices"][0]["text"] == reference_text
            ),
        }
        passed = all(checks.values())
        result = {
            "status": "qualified" if passed else "failed",
            "conditions": {
                "model": args.model,
                "prompt_tokens": args.prompt_tokens,
                "cache_chunk_tokens": chunk_size,
                "checkpoint_contract": args.checkpoint_contract,
                "temperature": 0,
                "max_tokens": 1,
                "nonce": nonce,
                "external_cache": "filesystem_l2_after_process_restart",
                "reference_result": str(args.reference_result),
            },
            "checks": checks,
            "restore": {
                "elapsed_seconds": round(restore_seconds, 3),
                "effective_stored_gib_per_second": (
                    round(stored_bytes / (1024**3) / restore_seconds, 3)
                    if stored_bytes
                    else None
                ),
                "prompt_sources": restore_delta,
                "request_id": restored["id"],
                "output_token_ids": restored["choices"][0].get("token_ids"),
                "text": restored["choices"][0]["text"],
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if passed else 1

    if not args.cold_only:
        reset_prefix_cache(args.base_url)
    initial_status = (
        None if args.cold_only else request_json(f"{args.cache_url}/status")
    )
    before_cold = metrics(args.base_url)
    cold_started = time.monotonic()
    cold = request_json(f"{args.base_url}/v1/completions", payload)
    cold_seconds = time.monotonic() - cold_started
    cold_delta = delta(before_cold, metrics(args.base_url))
    check_health(args.base_url)

    if args.cold_only:
        passed = cold_delta["local_compute"] == args.prompt_tokens
        result = {
            "status": "qualified" if passed else "failed",
            "conditions": {
                "model": args.model,
                "prompt_tokens": args.prompt_tokens,
                "temperature": 0,
                "max_tokens": 1,
                "nonce": nonce,
                "external_cache": "disabled",
            },
            "checks": {"cold_computed_complete_prompt": passed},
            "cold": {
                "elapsed_seconds": round(cold_seconds, 3),
                "prompt_sources": cold_delta,
                "request_id": cold["id"],
                "output_token_ids": cold["choices"][0].get("token_ids"),
                "text": cold["choices"][0]["text"],
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if passed else 1

    minimum_checkpoint_count = None
    if args.checkpoint_contract == "semantic":
        assert initial_status is not None
        minimum_checkpoint_count = (
            int(initial_status["recurrent_checkpoints"]["published_generations"]) + 1
        )
    store_wait_started = time.monotonic()
    status = wait_for_store(
        args.cache_url,
        timeout=600,
        minimum_checkpoint_count=minimum_checkpoint_count,
    )
    store_ready = time.monotonic()

    reset_prefix_cache(args.base_url)
    before_restore = metrics(args.base_url)
    restore_started = time.monotonic()
    restored = request_json(f"{args.base_url}/v1/completions", payload)
    restore_seconds = time.monotonic() - restore_started
    restore_delta = delta(before_restore, metrics(args.base_url))
    check_health(args.base_url)

    chunk_size = int(status["chunk_size"])
    assert initial_status is not None
    initial_l1_bytes = int(
        initial_status["storage_manager"]["l1_manager"]["memory_used_bytes"]
    )
    stored_l1_bytes = int(status["storage_manager"]["l1_manager"]["memory_used_bytes"])
    stored_bytes = max(0, stored_l1_bytes - initial_l1_bytes)
    checks = {
        "cold_computed_complete_prompt": (
            cold_delta["local_compute"] == args.prompt_tokens
        ),
        **restore_checks(
            restore_delta, args.prompt_tokens, chunk_size, args.checkpoint_contract
        ),
        "greedy_output_matches": cold["choices"][0]["text"]
        == restored["choices"][0]["text"],
    }
    passed = all(checks.values())
    result = {
        "status": "qualified" if passed else "failed",
        "conditions": {
            "model": args.model,
            "prompt_tokens": args.prompt_tokens,
            "cache_chunk_tokens": chunk_size,
            "checkpoint_contract": args.checkpoint_contract,
            "temperature": 0,
            "max_tokens": 1,
            "nonce": nonce,
            "stored_bytes": stored_bytes,
            "stored_bytes_measurement": "net L1 occupancy increase; excludes evicted payloads",
        },
        "checks": checks,
        "cold": {
            "elapsed_seconds": round(cold_seconds, 3),
            "prompt_sources": cold_delta,
            "request_id": cold["id"],
            "output_token_ids": cold["choices"][0].get("token_ids"),
            "text": cold["choices"][0]["text"],
        },
        "asynchronous_store": {
            "wait_after_response_seconds": round(store_ready - store_wait_started, 3),
            "cold_start_to_store_ready_seconds": round(store_ready - cold_started, 3),
            "contract": "All checkpoint writes and configured storage-tier tasks drained; reported separately from API response time.",
        },
        "restore": {
            "elapsed_seconds": round(restore_seconds, 3),
            "effective_stored_gib_per_second": round(
                stored_bytes / (1024**3) / restore_seconds, 3
            ),
            "prompt_sources": restore_delta,
            "request_id": restored["id"],
            "output_token_ids": restored["choices"][0].get("token_ids"),
            "text": restored["choices"][0]["text"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
