"""GPU timing helpers shared by every profile generator session."""

from __future__ import annotations

import statistics


def _l2_flush_fn(device: object, *, enabled: bool):
    if not enabled:
        return None
    import torch

    properties = torch.cuda.get_device_properties(device)
    flush_bytes = max(2 * int(properties.L2_cache_size), 64 << 20)
    buffer = torch.ones(
        (flush_bytes + 3) // 4,
        dtype=torch.float32,
        device=device,
    )
    reduction = torch.empty((), dtype=torch.float32, device=device)

    def flush() -> None:
        torch.sum(buffer, dim=0, out=reduction)

    return flush


def _median_of_group_medians(
    samples: tuple[float, ...],
    *,
    groups: int,
    repetitions: int,
) -> float:
    expected = int(groups) * int(repetitions)
    if len(samples) != expected:
        raise ValueError(f"expected {expected} timing samples, received {len(samples)}")
    medians = [
        statistics.median(samples[start : start + repetitions])
        for start in range(0, expected, repetitions)
    ]
    return float(statistics.median(medians))


def _bounded_repetitions(settings, *, pilot_us: float) -> int:
    budget_us = float(settings.max_candidate_seconds) * 1_000_000.0
    budgeted = int(budget_us / (max(float(pilot_us), 1.0) * settings.groups))
    return max(1, min(settings.repetitions, budgeted))


def _cuda_event_samples_us(
    run,
    *,
    count: int,
    device: object,
    flush=None,
    before_each=None,
) -> tuple[float, ...]:
    import torch

    starts = tuple(torch.cuda.Event(enable_timing=True) for _ in range(count))
    ends = tuple(torch.cuda.Event(enable_timing=True) for _ in range(count))
    for start, end in zip(starts, ends, strict=True):
        if before_each is not None:
            before_each()
        if flush is not None:
            flush()
        start.record()
        run()
        end.record()
    torch.cuda.synchronize(device)
    return tuple(
        float(start.elapsed_time(end)) * 1_000.0
        for start, end in zip(starts, ends, strict=True)
    )


__all__ = [
    "_bounded_repetitions",
    "_cuda_event_samples_us",
    "_l2_flush_fn",
    "_median_of_group_medians",
]
