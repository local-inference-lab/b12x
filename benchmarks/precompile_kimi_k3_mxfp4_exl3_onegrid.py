#!/usr/bin/env python3
"""Warm Kimi-K3 MXFP4+EXL3 one-grid kernels without opening weights."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sparkinfer.moe._shared.kernels.w4a16.host import max_packed_route_slots
from sparkinfer.moe._shared.kernels.w4a16.mixed_trellis import (
    compile_mixed_mxfp4_trellis,
)

_EXPERTS = 896
_TOPK = 16


def _allocation_splits(path: Path) -> list[tuple[int, int]]:
    document = json.loads(path.read_text())
    layers = document.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise ValueError(f"{path} does not contain a non-empty layers mapping")
    splits = []
    for layer, allocation in layers.items():
        kept = len(allocation.get("keep", ()))
        exl3 = len(allocation.get("exl3", ()))
        if kept <= 0 or exl3 <= 0 or kept + exl3 != _EXPERTS:
            raise ValueError(
                f"layer {layer} has invalid MXFP4/EXL3 split {kept}+{exl3}"
            )
        splits.append((kept, exl3))
    return sorted(set(splits))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("allocation", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    splits = _allocation_splits(args.allocation)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        splits = splits[: args.limit]

    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    sms = int(props.multi_processor_count)
    max_shared_mem = int(props.shared_memory_per_block_optin)
    route_slots = max_packed_route_slots(_TOPK, 8, _EXPERTS)
    started = time.perf_counter()
    timings = []
    for index, (kept, exl3) in enumerate(splits, 1):
        split_started = time.perf_counter()
        launch = compile_mixed_mxfp4_trellis(
            size_m=1,
            hidden_size=3584,
            intermediate_size=192,
            tier0_num_experts=kept,
            tier1_num_experts=exl3,
            top_k=_TOPK,
            max_m_blocks=(route_slots + 7) // 8,
            sms=sms,
            max_shared_mem=max_shared_mem,
            force_tile_config=(128, 64, 64, 128),
            activation="situ",
            intermediate_hadamard_tail=64,
            tier1_broadcast_suh=True,
            tier1_broadcast_svh=True,
        )
        elapsed = time.perf_counter() - split_started
        if int(launch.local_memory_bytes) != 0:
            raise RuntimeError(f"split {kept}+{exl3} spills local memory")
        timings.append(elapsed)
        print(
            json.dumps(
                {
                    "index": index,
                    "total": len(splits),
                    "split": [kept, exl3],
                    "seconds": elapsed,
                    "shared_memory_bytes": launch.shared_memory_bytes,
                    "registers_per_thread": launch.registers_per_thread,
                }
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "unique_splits": len(splits),
                "seconds": time.perf_counter() - started,
                "slowest_split_seconds": max(timings, default=0.0),
                "checkpoint_tensors_opened": 0,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
