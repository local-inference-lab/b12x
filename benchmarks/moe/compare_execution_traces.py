"""Compare logical requests across opt-in vLLM execution traces.

Batch signatures and numerical comparisons are separate: identical prompts do
not imply identical execution shapes. No tolerance replaces exact equality.
"""

import argparse
import json
from pathlib import Path
import re


def request_identity(value):
    """Remove the engine's internal nonce from this harness's request IDs."""
    return re.sub(r"-[0-9a-f]{8}$", "", value)


def execution_signature(trace):
    return [
        {
            key: [request_identity(r) for r in value] if key == "requests" else value
            for key, value in step.items()
            if key
            in (
                "requests",
                "scheduled",
                "computed",
                "prefill_lengths",
                "prefilling",
                "query_start",
                "tokens",
                "padded_tokens",
                "padded_requests",
                "full_graph",
            )
        }
        for step in trace["steps"]
    ]


def sample_rows(trace, devices):
    """Key sampled rows by request and generated-token index, excluding chunks."""
    result = {}
    for device in devices:
        step = trace["steps"][device["step"]]
        if "logits" not in device:
            continue
        if len(device["logits"]) != len(step["requests"]):
            raise ValueError("diagnostic supports one sampled row per request")
        for row, request in enumerate(step["requests"]):
            index = (
                step["computed"][row]
                + step["scheduled"][row]
                - step["prefill_lengths"][row]
            )
            if index < 0:
                continue
            key = request_identity(request), index
            if key in result:
                raise ValueError("duplicate logical sampled row")
            result[key] = step, device, row
    return result


def compare(left, right):
    import torch

    traces = [json.loads(p.read_text()) for p in (left, right)]
    if any(t["overflow"] for t in traces):
        raise ValueError("truncated execution trace is not comparison evidence")
    signatures = [execution_signature(t) for t in traces]
    result = {
        "same_execution_signature": signatures[0] == signatures[1],
        "steps": [len(s) for s in signatures],
        "requests": {},
    }
    devices = [torch.load(str(p) + ".pt", weights_only=True) for p in (left, right)]
    samples = [sample_rows(t, d) for t, d in zip(traces, devices, strict=True)]
    for key in sorted(samples[0].keys() & samples[1].keys()):
        request, index = key
        rows = [s[key] for s in samples]
        hidden = [d["hidden"][row] for _, d, row in rows]
        logits = [d["logits"][row] for _, d, row in rows]
        summary = result["requests"].setdefault(
            request,
            {
                "compared": 0,
                "first_hidden_difference": None,
                "first_logit_difference": None,
                "first_argmax_difference": None,
            },
        )
        summary["compared"] += 1
        observation = {
            "output_index": index,
            "steps": [s["step"] for s, _, _ in rows],
            "tokens": [s["tokens"] for s, _, _ in rows],
            "hidden_max_abs": (hidden[0].float() - hidden[1].float())
            .abs()
            .max()
            .item(),
            "top5": [
                list(
                    zip(
                        v.topk(5).indices.tolist(),
                        v.topk(5).values.tolist(),
                        strict=True,
                    )
                )
                for v in logits
            ],
        }
        for name, differs in (
            ("hidden", not torch.equal(*hidden)),
            ("logit", not torch.equal(*logits)),
            ("argmax", logits[0].argmax().item() != logits[1].argmax().item()),
        ):
            field = f"first_{name}_difference"
            if differs and summary[field] is None:
                summary[field] = observation
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("left", type=Path)
    p.add_argument("right", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = compare(args.left, args.right)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
