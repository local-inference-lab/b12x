# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the b12x project
"""Benchmark complete io_uring + O_DIRECT PLE transactions on changing queries.

Uses physical temporary files, not sparse files. The default NVFP4 geometry
creates a 26.82 GiB table. Only the fixture's write cache is dropped; no global
cache settings are changed. Wall time includes hashing, CPU ID transfer,
read planning, reads/scatter, GPU dequantization and completion. GPU L2 is
flushed outside each timed transaction. One transaction per launch shape warms
compilation before measurement; there is no whole-table prewarm.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any

import torch
from cuda.bindings import runtime as cudart

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from b12x.sequence import ple_embedding


@dataclass
class Case:
    plan: Any
    table: Any
    binding: Any

    def prepare(self, tokens: int) -> None:
        self.binding.num_tokens.fill_(tokens)
        self.binding.query_start_loc[1].fill_(tokens)
        self.run(tokens)
        self.check(tokens)

    def run(self, tokens: int) -> None:
        ple_embedding.run(self.binding, token_count=tokens)

    def check(self, tokens: int) -> None:
        ids = self.binding._ids[:tokens]
        local = (
            (ids >= self.plan.shard_start)
            & (ids < self.plan.shard_end)
            & (ids < self.plan.table_vocab_size)
        )
        expected = torch.where(local, 0.5, 0).to(torch.bfloat16)
        expected = expected.unsqueeze(-1).expand(-1, -1, self.plan.head_dim)
        torch.testing.assert_close(
            self.binding.out[:tokens], expected.flatten(1), rtol=0, atol=0
        )


class Files:
    def __init__(self, directory: pathlib.Path, plan: Any, shard_rows: int):
        self.sources: list[tuple[int, bool, pathlib.Path, int]] = []
        offset = 4104
        for index, start in enumerate(range(0, plan.padded_vocab_size, shard_rows)):
            rows = min(shard_rows, plan.padded_vocab_size - start)
            for scale, width, value in [(False, 80, 0x22), (True, 10, 0x38)]:
                path = directory / f"{index}-{'scale' if scale else 'weight'}.bin"
                block = bytes([value]) * (8 << 20)
                with path.open("wb", buffering=0) as stream:
                    stream.write(b"H" * offset)
                    remaining = rows * width
                    while remaining:
                        piece = memoryview(block)[: min(remaining, len(block))]
                        remaining -= stream.write(piece)
                    os.fsync(stream.fileno())
                    os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                self.sources.append((index, scale, path, offset))


def make_plan(args: argparse.Namespace) -> Any:
    return ple_embedding.plan(
        ple_embedding.Caps(
            device=torch.device("cuda", args.device),
            max_tokens=max(args.tokens),
            max_seqs=1,
            vocab_size=248320,
            eos_token_id=248044,
            max_order=3,
            heads_per_order=8,
            dense_layer_ordinal=0,
            base_table_size=args.base_rows,
            embedding_dim=2560,
            tp_size=args.tp_size,
            tp_rank=args.tp_rank,
            table_alignment=128,
            quant_mode="nvfp4_group16",
            table_memory="io_uring",
        )
    )


def make_case(args: argparse.Namespace, plan: Any, table: Any, files: Files) -> Case:
    for index, scale, path, offset in files.sources:
        table.add_shard(index, str(path), offset, scale=scale)
    device = plan.caps.device
    generator = torch.Generator(device=device).manual_seed(args.seed)
    token_ids = torch.randint(
        1,
        200000,
        (plan.caps.max_tokens,),
        dtype=torch.int64,
        device=device,
        generator=generator,
    )
    spec = plan.scratch_specs()[0]
    binding = plan.bind(
        scratch=torch.empty(spec.shape, dtype=spec.dtype, device=device),
        weight=None,
        weight_scale=None,
        weight_scale_2=torch.tensor([0.5], device=device),
        disk_table=table,
        token_ids=token_ids,
        query_start_loc=torch.tensor(
            [0, plan.caps.max_tokens], dtype=torch.int32, device=device
        ),
        committed_history=torch.full(
            (1, 2), plan.caps.eos_token_id, dtype=torch.int64, device=device
        ),
        num_seqs=torch.ones(1, dtype=torch.int32, device=device),
        num_tokens=torch.tensor(
            [plan.caps.max_tokens], dtype=torch.int32, device=device
        ),
        out=torch.empty(plan.output_shape, dtype=plan.output_dtype, device=device),
    )
    return Case(plan, table, binding)


def measure(case: Case, tokens: int, l2_flush: torch.Tensor) -> dict[str, Any]:
    l2_flush.zero_()
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    case.run(tokens)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    case.check(tokens)
    return {"wall_ms": elapsed_ms, "io": case.table.stats()}


def query_stream(args: argparse.Namespace, tokens: int) -> list[torch.Tensor]:
    device = torch.device("cuda", args.device)
    if args.query_file is None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        return list(
            torch.randint(
                1,
                200000,
                (args.stream_queries, tokens),
                dtype=torch.int64,
                device=device,
                generator=generator,
            )
        )
    raw = json.loads(args.query_file.read_text())
    if not isinstance(raw, list) or len(raw) < args.stream_queries:
        raise ValueError("query-file must contain enough token-ID lists")
    queries = []
    for row in raw[: args.stream_queries]:
        if (
            not isinstance(row, list)
            or len(row) < tokens
            or any(
                type(token) is not int or not 0 <= token < 248320
                for token in row[:tokens]
            )
        ):
            raise ValueError("query-file has an invalid or short token-ID list")
        queries.append(torch.tensor(row[:tokens], dtype=torch.int64, device=device))
    return queries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-rows", type=int, default=20_000_000)
    parser.add_argument("--tokens", default="4096")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--shards", type=int, default=128)
    parser.add_argument("--queue-depth", type=int, default=64)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument(
        "--directory", type=pathlib.Path, default=pathlib.Path("/var/tmp")
    )
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--stream-queries", type=int, default=8)
    parser.add_argument("--query-file", type=pathlib.Path)
    args = parser.parse_args()
    args.tokens = [int(value) for value in args.tokens.split(",")]
    if not args.tokens or min(args.tokens) <= 0 or args.repeats <= 0:
        parser.error("tokens and repeats must be positive")
    if args.shards <= 0 or args.stream_queries <= 0:
        parser.error("shards and stream-queries must be positive")
    torch.cuda.set_device(args.device)
    root = pathlib.Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    )
    error, l2_bytes = cudart.cudaDeviceGetAttribute(
        cudart.cudaDeviceAttr.cudaDevAttrL2CacheSize, args.device
    )
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"Cannot query GPU L2 size: {error}")
    l2_flush = torch.empty(
        max(1, 2 * l2_bytes), dtype=torch.uint8, device=f"cuda:{args.device}"
    )
    metadata = {
        "gpu": torch.cuda.get_device_name(args.device),
        "torch": torch.__version__,
        "b12x_revision": revision,
        "source_tree_dirty": dirty,
        "page_size": os.sysconf("SC_PAGESIZE"),
        "mode": "io_uring",
        "direct": True,
        "cold_l2_cache": True,
        "tp_size": args.tp_size,
        "tp_rank": args.tp_rank,
        "queue_depth": args.queue_depth,
        "boundary": "complete hash/read/scatter/dequant transaction with completion",
        "warmup": "one transaction per launch shape before measurement",
        "stream_queries": args.stream_queries,
        "query_file": str(args.query_file) if args.query_file else None,
    }
    results: list[dict[str, Any]] = []
    with (
        torch.inference_mode(),
        tempfile.TemporaryDirectory(
            prefix="ple-disk-benchmark-", dir=args.directory
        ) as directory,
    ):
        plan = make_plan(args)
        shard_rows = (plan.padded_vocab_size + args.shards - 1) // args.shards
        table = ple_embedding.DiskTable(plan, shard_rows, queue_depth=args.queue_depth)
        metadata["table_bytes"] = plan.padded_vocab_size * 90
        print(
            json.dumps({"phase": "writing physical temporary files", **metadata}),
            flush=True,
        )
        files = Files(pathlib.Path(directory), plan, shard_rows)
        case = make_case(args, plan, table, files)
        for tokens in args.tokens:
            case.prepare(tokens)
            queries = query_stream(args, tokens)
            for repeat in range(args.repeats):
                for index, query in enumerate(queries):
                    case.binding.token_ids[:tokens].copy_(query)
                    row = {
                        "mode": "io_uring",
                        "tokens": tokens,
                        "repeat": repeat,
                        "query_index": index,
                        **measure(case, tokens, l2_flush),
                    }
                    results.append(row)
                    print(json.dumps(row), flush=True)
                if args.output is not None:
                    args.output.write_text(
                        json.dumps({"metadata": metadata, "results": results}, indent=2)
                    )
            selected = [row for row in results if row["tokens"] == tokens]
            first = [row["wall_ms"] for row in selected if row["query_index"] == 0]
            later = [row["wall_ms"] for row in selected if row["query_index"] > 0]
            print(
                json.dumps(
                    {
                        "summary": "io_uring",
                        "tokens": tokens,
                        "first_median_ms": statistics.median(first),
                        "subsequent_median_ms": statistics.median(later)
                        if later
                        else None,
                        "max_ms": max(row["wall_ms"] for row in selected),
                        "mean_stream_ms": sum(row["wall_ms"] for row in selected)
                        / args.repeats,
                    }
                ),
                flush=True,
            )
        torch.cuda.synchronize()
        del case, table


if __name__ == "__main__":
    main()
