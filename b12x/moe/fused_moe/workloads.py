"""Deterministic, untimed routing inputs for MoE candidate comparisons."""

from __future__ import annotations

from collections import Counter
import random

import torch


TUNING_WORKLOAD_VERSION = "shared_40_v1"


def make_routing_ids(
    tokens: int,
    top_k: int,
    num_experts: int,
    *,
    workload: str = "shared_40",
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build routes with distinct experts per token.

    ``shared_40`` reuses 40% of the batch's token/expert assignments, rounded
    to the nearest realizable expert count. Reuse favors already popular
    experts, rather than giving every expert exactly one or two tokens.
    One-token batches cannot share; a small expert pool may force more reuse.
    ``disjoint`` retains the cyclic, maximally spread baseline.
    """
    if tokens < 1 or not 1 <= top_k <= num_experts:
        raise ValueError("require positive tokens and 1 <= top_k <= num_experts")
    if workload == "disjoint":
        return (
            torch.arange(tokens * top_k, device=device, dtype=torch.int32)
            .reshape(tokens, top_k)
            .remainder_(num_experts)
        )
    if workload != "shared_40":
        raise ValueError(f"unknown routing workload: {workload!r}")

    unique = min(num_experts, max(top_k, (3 * tokens * top_k + 2) // 5))
    rng = random.Random(seed)
    expert_ids = rng.sample(range(num_experts), unique)
    counts: Counter[int] = Counter()
    rows = []
    introduced = 0
    for token in range(tokens):
        target = top_k + (unique - top_k) * token // max(tokens - 1, 1)
        new_count = target - introduced
        row = expert_ids[introduced:target]
        candidates = list(counts)
        for _ in range(top_k - new_count):
            expert = rng.choices(
                candidates, weights=[counts[e] ** 2 for e in candidates], k=1
            )[0]
            row.append(expert)
            candidates.remove(expert)
        rng.shuffle(row)
        rows.append(row)
        counts.update(row)
        introduced = target
    return torch.tensor(rows, dtype=torch.int32, device=device)


def make_tuning_routes(
    tokens: int, top_k: int, num_experts: int, *, device: torch.device | str
) -> torch.Tensor:
    """Use four sharing realizations for small verification batches."""
    if 2 <= tokens <= 8:
        return torch.stack([
            make_routing_ids(tokens, top_k, num_experts, seed=seed, device=device)
            for seed in range(42, 46)
        ])
    return make_routing_ids(
        tokens, top_k, num_experts, workload="disjoint", device=device
    ).unsqueeze(0)
