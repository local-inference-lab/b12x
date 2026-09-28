"""Race native V4.1 attention on shared-pool strides and high physical page IDs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
from statistics import median
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from b12x.preparation import PreparationSession
from benchmarks.attention_preparation import prepare_compressed
from b12x.attention import compressed_sparse_mla as mla
from b12x.attention._shared.mla.compressed_reference import (
    compressed_sparse_mla_reference,
    pack_deepseek_v41_cache_reference,
)
from benchmarks.common import nvidia_smi_gpu_mode_snapshot


def tensor_hash(tensor):
    return hashlib.sha256(
        tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


@torch.inference_mode()
def run_case(args, rows):
    device = torch.device("cuda", args.device)
    gen = torch.Generator(device=device).manual_seed(41071 + rows)
    swa_page, indexed_page = 32, args.indexed_page_size
    context = max(args.context_tokens, rows + 128)
    context = (context + 255) // 256 * 256
    indexed_tokens = context // args.ratio
    indexed_tokens = (indexed_tokens + indexed_page - 1) // indexed_page * indexed_page
    swa_pages, indexed_pages = context // swa_page, indexed_tokens // indexed_page
    stride = args.page_stride
    swa_base = 2**31 // stride + 1 if args.high_pid else 1
    indexed_base = swa_base + swa_pages
    total_pages = indexed_base + indexed_pages
    required = total_pages * stride
    if stride < max(swa_page * 528, indexed_page * 288):
        raise ValueError("page stride cannot contain both source formats")
    if torch.cuda.mem_get_info(device)[0] < required + (2 << 30):
        raise RuntimeError("insufficient VRAM for the high-PID shared pool and scratch")
    pool = torch.empty(required, dtype=torch.uint8, device=device)
    swa = torch.as_strided(pool, (total_pages, swa_page * 528), (stride, 1))
    indexed = torch.as_strided(pool, (total_pages, indexed_page * 288), (stride, 1))
    swa[0].fill_(127)
    indexed[0].fill_(127)
    group_scale = torch.linspace(0.03, 0.7, 32, device=device).repeat_interleave(16)
    values = (
        torch.randn((context, 512), generator=gen, device=device) * group_scale
    ).bfloat16()
    swa_reference = pack_deepseek_v41_cache_reference(
        values, page_size=swa_page, cache_kind="swa"
    )
    swa[swa_base : swa_base + swa_pages].copy_(swa_reference)
    values = (
        torch.randn((indexed_tokens, 512), generator=gen, device=device)
        * group_scale.flip(0)
        * 3
    ).bfloat16()
    indexed_reference = pack_deepseek_v41_cache_reference(
        values, page_size=indexed_page, cache_kind="indexed"
    )
    indexed[indexed_base : indexed_base + indexed_pages].copy_(indexed_reference)
    q = (
        torch.randn((rows, args.heads, 512), generator=gen, device=device) * 0.2
    ).bfloat16()
    q[:, 0, :448].zero_()
    q[:, 0, 448:].mul_(4)
    positions = torch.arange(context - rows, context, device=device, dtype=torch.int32)
    swa_indices = (
        positions[:, None]
        - 127
        + torch.arange(128, device=device)[None]
        + swa_base * swa_page
    ).int()
    visible = ((positions + 1) // args.ratio).clamp_min(1)
    logical = (
        (
            torch.arange(512, device=device)[None]
            + torch.arange(rows, device=device)[:, None] * 17
        )
        % visible[:, None]
    ).int()
    table = torch.arange(
        indexed_base, indexed_base + indexed_pages, device=device, dtype=torch.int32
    )[None].expand(rows, -1)
    swa_lengths = torch.full((rows,), 128, device=device, dtype=torch.int32)
    indexed_lengths = torch.minimum(visible, torch.full_like(visible, 512))
    swa_lengths[0] = indexed_lengths[0] = 0
    sink = torch.linspace(-0.3, 0.3, args.heads, device=device)
    sample_rows = torch.tensor(
        sorted(set([0, rows - 1, rows // 2, min(1, rows - 1), min(7, rows - 1)])),
        device=device,
        dtype=torch.int64,
    )

    def oracle():
        selected_logical = logical[sample_rows]
        # Keep oracle storage compact and independent of physical page IDs.
        # The production path still consumes the large strided shared pool.
        return compressed_sparse_mla_reference(
            q[sample_rows],
            swa_reference,
            swa_indices[sample_rows] - swa_base * swa_page,
            swa_lengths[sample_rows],
            extra_k_cache=indexed_reference,
            extra_indices=selected_logical,
            extra_topk_lengths=indexed_lengths[sample_rows],
            swa_page_size=swa_page,
            extra_page_size=indexed_page,
            sm_scale=1 / math.sqrt(512),
            attn_sink=sink,
            return_lse=True,
            cache_format="deepseek_v41",
        )

    expected, expected_lse = oracle()
    session = PreparationSession(device=device, autotune=False, compile_workers=2)
    arms = {}
    for mode in args.modes:
        for chunks in [1] if mode == "extend" else args.decode_chunks:
            out = torch.empty_like(q)
            caps = mla.Caps(
                device=device, num_q_heads=args.heads, max_q_rows=rows, max_width=640,
                swa_width=128, indexed_width=512, max_page_table_width=indexed_pages,
                swa_page_size=swa_page, indexed_page_size=indexed_page,
                cache_format="deepseek_v41", mode=mode,
                max_chunks_per_row=chunks, use_cuda_graph=True,
            )
            plan, binding = prepare_compressed(session, caps,
                bind_args=dict(q=q, swa_indices=swa_indices, swa_lengths=swa_lengths,
                               indexed_indices=logical, indexed_lengths=indexed_lengths, indexed_page_table=table),
                run_args=dict(swa_k_cache=swa, indexed_k_cache=indexed, sm_scale=1 / math.sqrt(512),
                              attn_sink=sink, return_lse=True, lse_scale="natural", out=out))
            scratch = binding.scratch.shared_scratch

            def execute(binding=binding, out=out):
                return mla.run(
                    binding=binding,
                    swa_k_cache=swa,
                    swa_page_size=swa_page,
                    indexed_k_cache=indexed,
                    indexed_page_size=indexed_page,
                    sm_scale=1 / math.sqrt(512),
                    attn_sink=sink,
                    return_lse=True,
                    lse_scale="natural",
                    out=out,
                )

            actual, lse = execute()
            torch.cuda.synchronize(device)
            try:
                torch.testing.assert_close(
                    actual[sample_rows], expected, rtol=0.035, atol=0.035
                )
                torch.testing.assert_close(
                    lse[sample_rows], expected_lse, rtol=0.01, atol=0.025
                )
            except AssertionError:
                failure_path = args.output.with_suffix(".failure.pt")
                torch.save(
                    {
                        "mode": mode,
                        "chunks": chunks,
                        "rows": rows,
                        "planned_num_chunks": binding.scratch.num_chunks_value,
                        "tmp_lse_shape": tuple(binding.scratch.tmp_lse.shape),
                        "tmp_lse_stride": tuple(binding.scratch.tmp_lse.stride()),
                        "chunk_control": binding.scratch.num_chunks_ptr.cpu(),
                        "actual": actual[sample_rows].cpu(),
                        "expected": expected.cpu(),
                        "lse": lse[sample_rows].cpu(),
                        "expected_lse": expected_lse.cpu(),
                        "q": q[sample_rows].cpu(),
                        "sample_rows": sample_rows.cpu(),
                    },
                    failure_path,
                )
                print(
                    f"Correctness failure: {mode}-c{chunks}; details {failure_path}",
                    flush=True,
                )
                raise
            if not torch.isfinite(actual).all() or not torch.count_nonzero(actual):
                raise AssertionError("native attention must be finite and nonzero")
            arms[f"{mode}-c{chunks}"] = {
                "execute": execute,
                "out": out,
                "lse": lse,
                "scratch": scratch,
                "plan": plan,
                "num_chunks": binding.scratch.num_chunks_value,
            }
    session.freeze()
    try:
        for arm in arms.values():
            for _ in range(3):
                arm["execute"]()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                arm["execute"]()
            arm["graph"] = graph
            for _ in range(args.warmup):
                graph.replay()
        torch.cuda.synchronize(device)
        timings = {name: [] for name in arms}
        begin, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for repeat in range(args.samples):
            order = list(arms) if repeat % 2 == 0 else list(reversed(arms))
            for name in order:
                begin.record()
                for _ in range(args.replays):
                    arms[name]["graph"].replay()
                end.record()
                end.synchronize()
                timings[name].append(begin.elapsed_time(end) * 1000 / args.replays)
        hashes = {name: tensor_hash(arm["out"]) for name, arm in arms.items()}
        q.mul_(0.7)
        expected, expected_lse = oracle()
        for arm in arms.values():
            arm["graph"].replay()
        torch.cuda.synchronize(device)
        for arm in arms.values():
            torch.testing.assert_close(
                arm["out"][sample_rows], expected, rtol=0.035, atol=0.035
            )
            torch.testing.assert_close(
                arm["lse"][sample_rows], expected_lse, rtol=0.01, atol=0.025
            )
        return {
            "rows": rows,
            "heads": args.heads,
            "ratio": args.ratio,
            "context_tokens": context,
            "page_stride": stride,
            "swa_base_pid": swa_base,
            "indexed_base_pid": indexed_base,
            "high_pid_byte_offset": swa_base * stride,
            "samples_us": timings,
            "median_us": {name: median(v) for name, v in timings.items()},
            "output_sha256": hashes,
            "planned_chunks": {name: arm["num_chunks"] for name, arm in arms.items()},
            "extend_over_decode": {
                name: median(timings["extend-c1"]) / median(values)
                for name, values in timings.items()
                if name.startswith("decode") and "extend-c1" in timings
            },
            "correctness": "V4.1 source-specific oracle on boundary/interior/tail rows, finite/nonzero full output, mutated-query graph replay, frozen resolution, shared strided pool beyond2^31-byte offsets.",
        }
    finally:
        for arm in arms.values():
            if "graph" in arm:
                arm["graph"].reset()
        session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", default="6,128,4096")
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--context-tokens", type=int, default=16384)
    parser.add_argument("--ratio", type=int, choices=(1, 2), default=2)
    parser.add_argument("--indexed-page-size", type=int, default=128)
    parser.add_argument("--page-stride", type=int, default=700416)
    parser.add_argument("--modes", default="extend,decode")
    parser.add_argument("--decode-chunks", default="1,4")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--high-pid", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.rows = [int(value) for value in args.rows.split(",")]
    args.decode_chunks = [int(value) for value in args.decode_chunks.split(",")]
    args.modes = args.modes.split(",")
    if any(row <= 0 for row in args.rows) or args.samples <= 0 or args.replays <= 0:
        parser.error("rows and repetition counts must be positive")
    if any(mode not in ("extend", "decode") for mode in args.modes):
        parser.error("modes must be extend/decode")
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    payload = {
        "command": [sys.executable, *sys.argv],
        "worktree": str(ROOT),
        "commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "status": subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--short"], text=True
        ).strip(),
        "arguments": {**vars(args), "output": str(args.output)},
        "toolchain": {
            name: importlib.metadata.version(name)
            for name in ("torch", "nvidia-cutlass-dsl")
        },
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("CUDA_", "CUTE_", "B12X_", "OMP_"))
        },
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                ROOT / "b12x/attention/_shared/mla/decode_math.py",
                ROOT / "b12x/attention/_shared/mla/kernel.py",
                ROOT / "b12x/attention/_shared/mla/prefill_mg.py",
            ]
        },
        "scope": "Public native plans and caller-owned scratch. Balanced graph replay; warm shared V4.1 cache data. No full-model performance inference. Output hashes compare identical inputs across source revisions, not different backends.",
        "ratio_direction": "extend latency / named decode latency; >1 means decode backend faster",
        "gpu_before": nvidia_smi_gpu_mode_snapshot(),
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for rows in args.rows:
        result = run_case(args, rows)
        payload["results"].append(result)
        payload["gpu_after"] = nvidia_smi_gpu_mode_snapshot()
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in ("rows", "heads", "median_us", "high_pid_byte_offset")
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
