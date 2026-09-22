"""Summarize opt-in model events separately from client first-token delivery.

CUDA intervals include the model's ordinary TP communication. They do not
isolate collective time. Mixed iterations retain their complete duration and
are excluded from pure-prefill rates instead of assigning time by token ratio.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re


def summarize(records):
    if not records or records[-1]["kind"] != "complete":
        raise ValueError("phase timing requires a completed serving receipt")
    config = next(r["arguments"] for r in records if r["kind"] == "configuration")
    if config["admission"] != "together":
        raise ValueError("prefill groups require controlled admission")
    results = [r["result"] for r in records if r["kind"] == "phase_timing"]
    if len(results) != 1 or len(results[0]) != config.get("tp_size", 1):
        raise ValueError("phase timing must include every TP rank exactly once")
    ranks = deepcopy(results[0])

    def logical_request(name):
        match = re.fullmatch(r"(cache-\d+)(?:-[0-9a-f]{8})?", name)
        if match is None:
            raise ValueError("unexpected request identity in phase timing")
        return match[1]

    for rank in ranks:
        for row in rank["iterations"]:
            row["requests"] = [logical_request(n) for n in row["requests"]]
        mapped = {logical_request(n): r for n, r in rank["requests"].items()}
        if len(mapped) != len(rank["requests"]):
            raise ValueError("duplicate logical request in phase timing")
        rank["requests"] = mapped
    iterations = ranks[0]["iterations"]
    fields = ("requests", "scheduled", "computed", "prompt_lengths", "phase")
    for rank in ranks:
        if len(rank["iterations"]) != len(iterations) or any(
            any(a[k] != b[k] for k in fields)
            for a, b in zip(iterations, rank["iterations"], strict=True)
        ):
            raise ValueError("TP scheduling metadata differs across ranks")
    requests = {r["index"]: r for r in records if r["kind"] == "request"}
    if set(requests) != set(range(len(requests))):
        raise ValueError("request inventory is incomplete")
    groups = []
    for first in range(0, len(requests), config["concurrency"]):
        indices = range(first, min(first + config["concurrency"], len(requests)))
        names = {f"cache-{i}" for i in indices}
        selected = [i for i, r in enumerate(iterations) if names & set(r["requests"])]
        if any(set(iterations[i]["requests"]) - names for i in selected):
            raise ValueError("a model iteration crosses controlled admission groups")
        pure = [i for i in selected if iterations[i]["phase"] == "prefill"]
        mixed = [i for i in selected if iterations[i]["phase"] == "mixed"]
        prompt = [i for i in selected if iterations[i]["prompt_tokens"]]
        if not prompt:
            raise ValueError("request group has no observed prompt processing")
        spans, sums = [], []
        for rank in ranks:
            rows = rank["iterations"]
            sums.append(sum(rows[i]["model_device_ms"] for i in pure))
            spans.append(
                rows[prompt[-1]]["model_end_ms"] - rows[prompt[0]]["model_start_ms"]
            )
            if any(not rank["requests"][n]["complete"] for n in names):
                raise ValueError("prompt processing did not complete")
        pure_tokens = sum(iterations[i]["prompt_tokens"] for i in pure)
        tokens = sum(iterations[i]["prompt_tokens"] for i in prompt)
        groups.append(
            dict(
                first_request=first,
                requests=list(indices),
                prompt_tokens=tokens,
                pure_prefill_tokens=pure_tokens,
                pure_model_ms_per_rank=sums,
                pure_model_tokens_per_s=pure_tokens * 1000 / max(sums)
                if max(sums)
                else None,
                prompt_span_ms_per_rank=spans,
                prompt_span_tokens_per_s=tokens * 1000 / max(spans)
                if not mixed
                else None,
                mixed_iterations=len(mixed),
                mixed_device_ms_per_rank=[
                    sum(r["iterations"][i]["model_device_ms"] for i in mixed)
                    for r in ranks
                ],
                chunk_sizes=[iterations[i]["prompt_chunks"] for i in prompt],
                ttft_ms=[requests[i]["ttft_ns"] / 1e6 for i in indices],
            )
        )
    return dict(
        arguments=config,
        timing_scope="pure model CUDA duration and prompt-processing CUDA span; client TTFT separate",
        rank_reduction="maximum rank duration, never sum rank times",
        groups=groups,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize([json.loads(r) for r in args.receipt.read_text().splitlines()])
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
