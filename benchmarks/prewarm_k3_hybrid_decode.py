"""Precompile every Kimi K3 TP16 hybrid one-grid expert split.

Compilation depends on the two tier counts but not on resident checkpoint
weights. Running this once against the persistent SparkInfer cache avoids 16
TP workers racing to compile the same kernels during the expensive model load.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from sparkinfer.moe._shared.kernels.w4a16.kernel import (
    compile_w4a16_fused_moe_hybrid,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_json")
    args = parser.parse_args()

    with open(args.config_json) as config_file:
        config = json.load(config_file)
    bit_map = config["quantization_config"]["hybrid_bit_map"]
    splits = sorted(
        {
            (values.count(4), values.count(3))
            for values in bit_map.values()
            if 3 in values and 4 in values
        }
    )

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    start = time.perf_counter()
    for index, (kept, nf3) in enumerate(splits, 1):
        split_start = time.perf_counter()
        launch = compile_w4a16_fused_moe_hybrid(
            size_m=1,
            hidden_size=3584,
            intermediate_size=192,
            tier0_num_experts=kept,
            tier1_num_experts=nf3,
            top_k=16,
            activation="situ",
            map_slots=896,
            element_dtype="bf16",
            fast_math=True,
            sms=int(props.multi_processor_count),
            max_shared_mem=int(
                getattr(props, "shared_memory_per_block_optin", 101_376)
            ),
            tier0_weight_layout="modelopt",
            tier0_scale_format="e8m0_k32",
            tier0_w13_layout="w31",
            tier1_weight_layout="nf3_2p1",
            tier1_scale_format="e4m3_k32",
            tier1_w13_layout="packed",
            force_tile_config=(128, 64, 64, 128),
            schedule_whole_tiles=True,
        )
        if launch.local_memory_bytes > 0:
            raise RuntimeError(
                f"split {kept}/{nf3} spills {launch.local_memory_bytes} bytes/thread"
            )
        print(
            f"[{index:02d}/{len(splits):02d}] MXFP4={kept:3d} NF3={nf3:3d} "
            f"{time.perf_counter() - split_start:.3f}s",
            flush=True,
        )
    print(
        f"prewarmed {len(splits)} unique hybrid splits in "
        f"{time.perf_counter() - start:.3f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
