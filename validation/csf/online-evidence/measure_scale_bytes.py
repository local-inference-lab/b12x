"""Measure lossless encoding on native layer-3, rank-0 expert scale bytes.

Cases use DS4.1 TP4, Qwen TP1, Kimi TP16, and GLM TP2. No checkpoint is written.
"""

import argparse
import json
import time
from pathlib import Path
from contextlib import ExitStack
import torch
from safetensors import safe_open
from b12x._lib.quant.csf_encode import encode_scale_bytes
from b12x._lib.quant.x4t_scales import decode_x4t_scales
from b12x._lib.quant.nvfp4_csf import decode_nvfp4_csf_pair
from b12x._lib.quant.nvfp4_csf_inline import prepare_inline_scales

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--model",
    choices=("ds41", "qwen", "kimi", "glm"),
    nargs="+",
    default=["ds41", "qwen"],
)
parser.add_argument(
    "--output", type=Path, default=Path("/results/real-scale-encoder.json")
)
args = parser.parse_args()
hub = Path("/root/.cache/huggingface/hub")
cases = [
    (
        "ds41",
        "mxfp4",
        "models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277",
        384,
        5120,
        576,
    ),
    (
        "qwen",
        "nvfp4",
        "models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/b797d2e1160b9596b2570e56c1d3590faa09d4ed",
        None,
        None,
        None,
    ),
    (
        "kimi",
        "mxfp4",
        "models--moonshotai--Kimi-K3/snapshots/2496450e92e425c886db095102a52a6682ca3970",
        896,
        3584,
        192,
    ),
    (
        "glm",
        "nvfp4",
        "models--local-inference-lab--GLM-5.3-Flash-NVFP4/snapshots/46aaae8a82032f77100f2f03e9cc11b391df3b4d",
        288,
        4096,
        1024,
    ),
]
results = []
for model, fmt, revision, e, h, n in cases:
    if model not in args.model:
        continue
    root = hub / revision
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    if model == "qwen":
        config = json.loads((root / "config.json").read_text())
        config = config.get("text_config", config)
        e = config["num_experts"]
        h = config["hidden_size"]
        n = config["moe_intermediate_size"]
    group = 16 if fmt == "nvfp4" else 32
    arrays = [
        torch.empty(e, 2 * n, h // group, dtype=torch.uint8),
        torch.empty(e, h, n // group, dtype=torch.uint8),
    ]
    with ExitStack() as stack:
        handles = {}
        for expert in range(e):
            for part, offset in [("gate", 0), ("up", n), ("down", 0)]:
                if fmt == "nvfp4":
                    name = f"model.language_model.layers.3.mlp.experts.{expert}.{part}_proj.weight_scale"
                elif model == "kimi":
                    name = f"language_model.model.layers.3.block_sparse_moe.experts.{expert}.{dict(gate='w1', up='w3', down='w2')[part]}.weight_scale"
                else:
                    name = f"layers.3.ffn.experts.{expert}.{dict(gate='w1', up='w3', down='w2')[part]}.scale"
                filename = index[name]
                if filename not in handles:
                    handles[filename] = stack.enter_context(
                        safe_open(root / filename, framework="pt", device="cpu")
                    )
                value = handles[filename].get_slice(name)
                if part == "down":
                    arrays[1][expert].copy_(value[:, : n // group].view(torch.uint8))
                else:
                    arrays[0][expert, offset : offset + n].copy_(
                        value[:n, :].view(torch.uint8)
                    )
    arrays = [a.cuda() for a in arrays]
    samples = []
    index_samples = []
    for _repeat in range(6):
        torch.cuda.synchronize()
        t = time.perf_counter()
        batches = [
            encode_scale_bytes(
                a,
                format=fmt,
                exception_row_rotation=n if model == "kimi" and i == 0 else 0,
            )
            for i, a in enumerate(arrays)
        ]
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t)
        if fmt == "nvfp4":
            t = time.perf_counter()
            planes = [prepare_inline_scales(b) for b in batches]
            torch.cuda.synchronize()
            index_samples.append(time.perf_counter() - t)
    outputs = [torch.empty_like(a) for a in arrays]
    ids = torch.arange(e, device="cuda", dtype=torch.int32)
    if fmt == "mxfp4":
        for b, o in zip(batches, outputs, strict=True):
            decode_x4t_scales(b, ids, o)
        expected = arrays
    else:
        decode_nvfp4_csf_pair(*batches, ids, *outputs)
        expected = [
            a.reshape(e, a.shape[1] // 128, 4, 32, a.shape[2] // 4, 4)
            .permute(0, 1, 4, 3, 2, 5)
            .contiguous()
            .reshape_as(a)
            for a in arrays
        ]
    assert all(torch.equal(x, y) for x, y in zip(outputs, expected, strict=True))
    packed = sum(
        b.fixed.numel() + b.exceptions.numel() * b.exceptions.element_size()
        for b in batches
    )
    row = dict(
        format=fmt,
        revision=revision,
        layer=3,
        geometry=[e, h, n],
        source_bytes=sum(a.numel() for a in arrays),
        fixed_and_exception_bytes=packed,
        encode_pair_seconds=samples,
        index_pair_seconds=index_samples,
        exact=True,
    )
    results.append(row)
    print(json.dumps(row), flush=True)
args.output.write_text(json.dumps(results, indent=2))
