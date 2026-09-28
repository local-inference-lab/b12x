# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the b12x project
"""Matched, isolated native Engram/PLE checkpoint SSD transactions, NOT inference.

No TP collective, projection, model-state preparation, speculative verification,
or graph consumer is timed. O_DIRECT does not imply a cold SSD/controller cache.
Only headers, tiny hash/scalar tensors, and selected CPU oracle rows are loaded;
row tables remain immutable file sources. See benchmark_ple_disk.py for the
synthetic-file counterpart. Run each TP rank explicitly on authorized GPUs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from contextlib import ExitStack
from typing import Any

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.common import make_l2_flush_fn, nvidia_smi_gpu_mode_snapshot
from b12x.sequence import engram, ple_embedding
from b12x.preparation import PreparationSession, PreparedCall
from b12x.sequence.engram._impl import _bind_state, _bind_lookup_state
from b12x.sequence.engram.geometry import build_compressed_token_map, build_geometry
from b12x.sequence.engram.reference import hash_reference
from b12x.sequence.ple_hash.reference import ple_hash_packed_reference
from b12x.sequence.ple_embedding.reference import _dequantize_selected_nvfp4

ENGRAM_PATH = "/data/cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/fb2764a5cf321eaa5070ca8f9e892818f477c16d"
QWEN_PATH = "/data/models/qwen3.8-flash-next-mixed/qwen3.8-flash-next-180b-nvfp4-ple-mxfp8-attn-shared_vv1"
PLE_PREFIX = "model.language_model.layers.1.ple.ple_embedding."
WARNING = "Native hash/D2H/event/disk/GPU-decode transaction timing; NOT full model inference or TP projection timing."


def digest(path: pathlib.Path) -> str:
    # Used only for source, metadata, and native-object files, never row payloads.
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_identity(path: pathlib.Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.absolute()),
        "realpath": str(path.resolve()),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


_READER_API = (
    "ple_reader",
    "ple_reader_add",
    "ple_reader_run",
    "ple_reader_stats",
)


def load_reader_library(path: pathlib.Path) -> Any:
    """Load and validate an exact cached reader object for this process."""
    path = path.expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"--reader-library must name a native library file: {path}")
    spec = importlib.util.spec_from_file_location("_b12x_loader_storage", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load native reader library: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if getattr(module, "ABI_VERSION", None) != 1:
        raise RuntimeError(
            f"Native reader ABI mismatch in {path}: "
            f"expected 1, got {getattr(module, 'ABI_VERSION', None)!r}"
        )
    missing = [
        name for name in _READER_API if not callable(getattr(module, name, None))
    ]
    if missing:
        raise RuntimeError(
            f"Native reader library {path} is missing reader API: {', '.join(missing)}"
        )
    return module


def select_native_reader(reader_library: pathlib.Path | None) -> pathlib.Path:
    """Select the native reader before DiskRowCache imports its loader."""
    from b12x.loader import _native

    if reader_library is None:
        module = _native.load()
    else:
        module = load_reader_library(reader_library)
        _native.load = lambda: module
    return pathlib.Path(module.__file__).resolve()


@dataclass(frozen=True)
class FilePlane:
    path: str
    offset: int
    shape: tuple[int, ...]
    dtype: torch.dtype


class Checkpoint:
    def __init__(self, root: pathlib.Path):
        self.root = root
        self.config = json.loads((root / "config.json").read_text())
        self.index = json.loads((root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        self.headers: dict[str, Any] = {}
        self.sources: dict[str, Any] = {}
        self.files: dict[str, Any] = {}

    def source(self, key: str) -> Any:
        filename = self.index[key]
        if filename not in self.headers:
            path = self.root / filename
            self.files[filename] = file_identity(path)
            from b12x.loader._checkpoint import _unique_object
            with path.open("rb", buffering=0) as stream:
                size = int.from_bytes(stream.read(8), "little")
                if not 0 < size <= 64 << 20 or size + 8 > self.files[filename]["size"]:
                    raise ValueError(f"Invalid safetensors header length: {path}")
                header = json.loads(stream.read(size), object_pairs_hook=_unique_object)
            self.headers[filename] = (8 + size, header)
        origin, header = self.headers[filename]
        entry = header[key]
        dtypes = {"F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.float8_e8m0fnu,
                  "U8": torch.uint8, "BF16": torch.bfloat16, "F32": torch.float32,
                  "I64": torch.int64, "I32": torch.int32}
        dtype = dtypes[entry["dtype"]]
        begin, end = entry["data_offsets"]
        shape = tuple(entry["shape"])
        if any(type(x) is not int or x < 0 for x in (*shape, begin, end)) or not begin <= end:
            raise ValueError(f"Invalid safetensors range: {key}")
        if math.prod(shape) * dtype.itemsize != end - begin or origin + end > self.files[filename]["size"]:
            raise ValueError(f"Safetensors extent mismatch: {key}")
        source = FilePlane(str(self.root / filename), origin + begin, shape, dtype)
        self.sources[key] = source
        return source

    def small(self, key: str) -> torch.Tensor:
        source = self.source(key)
        count = math.prod(source.shape) * source.dtype.itemsize
        if not 0 < count <= 65536:
            raise ValueError(f"Refusing non-small tensor payload: {key}, {count} bytes")
        return (
            torch.frombuffer(bytearray(self.read(source, 0, count)), dtype=source.dtype)
            .reshape(source.shape)
            .clone()
        )

    @staticmethod
    def read(source: Any, relative: int, count: int) -> bytes:
        if (
            relative < 0
            or count < 0
            or relative + count > math.prod(source.shape) * source.dtype.itemsize
        ):
            raise ValueError("CPU oracle slice outside declared source")
        with open(source.path, "rb", buffering=0) as stream:
            data = os.pread(stream.fileno(), count, source.offset + relative)
        if len(data) != count:
            raise OSError(f"Short checkpoint slice: {source.path}")
        return data

    def unchanged(self) -> None:
        for filename, identity in self.files.items():
            if file_identity(self.root / filename) != identity:
                raise RuntimeError(
                    f"Immutable checkpoint changed during benchmark: {filename}"
                )

    def report(self) -> dict[str, Any]:
        return {
            "root": str(self.root.resolve()),
            "files": self.files,
            "small_file_sha256": {
                name: digest(self.root / name)
                for name in (
                    "config.json",
                    "model.safetensors.index.json",
                    "tokenizer.json",
                )
            },
            "sources": {
                key: {
                    "path": s.path,
                    "offset": s.offset,
                    "offset_mod_4096": s.offset % 4096,
                    "shape": list(s.shape),
                    "dtype": str(s.dtype),
                    "bytes": math.prod(s.shape) * s.dtype.itemsize,
                }
                for key, s in self.sources.items()
            },
            "identity_contract": "Path/stat plus header ranges; immutable files required, no full payload checksum",
        }


@dataclass
class Case:
    model: str
    owner: int
    layout: Any
    session: Any
    table: Any
    binding: Any
    lookup: Any
    checkpoint: Checkpoint
    planes: dict[int, tuple[str, str]]
    source_rows: int
    token_map: list[int] | None
    token_bound: bool
    last_prepared: int = 0

    def __post_init__(self):
        if self.model == "engram" and self.token_bound:
            self.out.zero_()

    @property
    def heads(self) -> int:
        return 24 if self.model == "engram" else 16

    @property
    def dim(self) -> int:
        return 256 if self.model == "engram" else 160

    @property
    def rows(self) -> int:
        return (
            self.layout.table_rows
            if self.model == "engram"
            else self.layout.table_vocab_size
        )

    @property
    def ids(self) -> torch.Tensor:
        return self.binding.hash_ids if self.model == "engram" else self.binding._ids

    @property
    def out(self) -> torch.Tensor:
        return self.lookup.out if self.model == "engram" else self.binding.out

    def close(self):
        if self.model == "engram":
            self.table.close()
        else:
            self.table._cache.close()
        self.session.close()

    def run(self, tokens: int) -> None:
        if self.model == "engram":
            if self.token_bound:
                engram.run(self.binding, token_count=tokens)
                if tokens < self.last_prepared:
                    self.out[tokens : self.last_prepared].zero_()
                engram.run_lookup(self.lookup, token_count=tokens, clear_tail=False)
                self.last_prepared = tokens
            else:
                engram.run(self.binding)
                engram.run_lookup(self.lookup, token_count=tokens)
        else:
            ple_embedding.run(self.binding, token_count=tokens)

    def prepare(self, query: dict[str, Any]) -> None:
        _set_query(vars(self.binding), query, self.layout.caps, self.model, self.token_map)
        live = len(query["tokens"])
        if self.model == "engram" and self.token_bound:
            self.out[: max(live, self.last_prepared)].fill_(7)
        else:
            self.out.fill_(7)  # Untimed sentinel exposes model-specific tail contracts.

    def expected_hashes(self, query: dict[str, Any]) -> torch.Tensor:
        if self.model == "engram":
            history = [
                [self.token_map[x] if x >= 0 and x != 129264 else -1 for x in row]
                for row in query["history"]
            ]
            return hash_reference(
                query["tokens"],
                [x != 129264 for x in query["tokens"]],
                query["starts"],
                list(range(len(query["starts"]) - 1)),
                history,
                self.token_map,
                self.layout.geometry,
                self.owner,
            )
        return ple_hash_packed_reference(
            torch.tensor(query["tokens"], dtype=torch.int64),
            torch.tensor(query["starts"], dtype=torch.int32),
            torch.tensor(
                query["history"][: len(query["starts"]) - 1], dtype=torch.int64
            ),
            eos_token_id=self.layout.caps.eos_token_id,
            multipliers=self.binding._hash_binding.geometry.multipliers.cpu(),
            prime_sizes=self.binding._hash_binding.geometry.prime_sizes.cpu(),
            table_offsets=self.binding._hash_binding.geometry.table_offsets.cpu(),
            heads_per_order=8,
        )

    def cpu_row(self, row: int) -> tuple[torch.Tensor, dict[str, Any]]:
        shard, relative_row = divmod(row, self.source_rows)
        weight_key, scale_key = self.planes[shard]
        weight, scale = (
            self.checkpoint.source(weight_key),
            self.checkpoint.source(scale_key),
        )
        widths = (256, 8) if self.model == "engram" else (80, 10)
        raw_w = self.checkpoint.read(weight, relative_row * widths[0], widths[0])
        raw_s = self.checkpoint.read(scale, relative_row * widths[1], widths[1])
        w = torch.frombuffer(bytearray(raw_w), dtype=torch.uint8)
        s = torch.frombuffer(bytearray(raw_s), dtype=torch.uint8)
        if self.model == "engram":
            exponent = s.to(torch.int32)
            factors = torch.pow(
                torch.tensor(2.0, dtype=torch.float32), exponent.float() - 127
            )
            factors[exponent == 255] = float("nan")
            decoded = w.view(torch.float8_e4m3fn).float() * factors.repeat_interleave(
                32
            )
        else:
            decoded = _dequantize_selected_nvfp4(
                w,
                s.view(torch.float8_e4m3fn),
                self.binding.weight_scale_2.cpu(),
                head_dim=160,
            )
        return decoded.to(torch.bfloat16), {
            "row": row,
            "weight_key": weight_key,
            "scale_key": scale_key,
            "weight_offset": weight.offset + relative_row * widths[0],
            "scale_offset": scale.offset + relative_row * widths[1],
            "weight_sha256": hashlib.sha256(raw_w).hexdigest(),
            "scale_sha256": hashlib.sha256(raw_s).hexdigest(),
        }

    def check_rows(
        self, ids: torch.Tensor, output: torch.Tensor, count: int
    ) -> dict[str, Any]:
        flat = ids.reshape(-1)
        local = (
            (flat >= self.layout.shard_start)
            & (flat < self.layout.shard_end)
            & (flat < self.rows)
        )
        values = output.reshape(-1, self.dim)
        torch.testing.assert_close(
            values[~local], torch.zeros_like(values[~local]), rtol=0, atol=0
        )
        positions = local.nonzero().flatten()
        selected = (
            positions[
                torch.linspace(0, len(positions) - 1, min(count, len(positions))).long()
            ]
            if len(positions)
            else []
        )
        slices = []
        for position in selected:
            index = int(position)
            expected, source = self.cpu_row(int(flat[index]))
            torch.testing.assert_close(
                values[index], expected, rtol=0, atol=0, equal_nan=True
            )
            slices.append({"slot": index, **source})
        return {
            "passed": True,
            "local_rows": int(local.sum()),
            "remote_or_invalid_rows_checked": int((~local).sum()),
            "checked_slices": slices,
            "unique_local_rows": int(torch.unique(flat[local]).numel()),
        }

    def check(self, query: dict[str, Any], prepared: int, count: int) -> dict[str, Any]:
        live = len(query["tokens"])
        torch.testing.assert_close(
            self.ids[:live].cpu(), self.expected_hashes(query), rtol=0, atol=0
        )
        result = self.check_rows(self.ids[:live].cpu(), self.out[:live].cpu(), count)
        end = self.layout.caps.max_tokens if self.model == "engram" else prepared
        torch.testing.assert_close(
            self.out[live:end], torch.zeros_like(self.out[live:end]), rtol=0, atol=0
        )
        if self.model == "qwen":
            torch.testing.assert_close(
                self.out[prepared:],
                torch.full_like(self.out[prepared:], 7),
                rtol=0,
                atol=0,
            )
        result.update(
            hash_exact=True,
            live=live,
            prepared=prepared,
            zero_tail_rows=end - live,
            untouched_tail_rows=self.layout.caps.max_tokens - end,
        )
        return result

    def boundary_check(self) -> dict[str, Any]:
        # Untimed decoder/reader control: actual local edge rows plus invalid and
        # remote IDs. PLE has no public hash-bypass API; use its native launcher.
        start, end = self.layout.shard_start, min(self.layout.shard_end, self.rows)
        probes = [-1, self.rows, start, end - 1]
        if start > 0:
            probes.append(start - 1)
        if end < self.rows:
            probes.append(end)
        for boundary in range(
            (start // self.source_rows + 1) * self.source_rows, end, self.source_rows
        ):
            probes.extend([boundary - 1, boundary])
        ids = torch.full((self.layout.caps.max_tokens, self.heads), -1, dtype=torch.int64)
        selected = probes[: ids.numel()]
        ids.view(-1)[: len(selected)] = torch.tensor(selected, dtype=torch.int64)
        prepared = max(1, math.ceil(len(selected) / self.heads))
        self.ids.copy_(ids.to(self.layout.caps.device))
        self.binding.num_tokens.fill_(prepared)
        self.out.fill_(7)
        if self.model == "engram":
            engram.run_lookup(self.lookup, token_count=prepared)
            self.last_prepared = prepared
        else:
            state = self.binding.plan.prepared.state
            with self.table._cache.transaction():
                self.table._cache.read_rows(self.ids, prepared * self.heads)
                state.run_lookup(
                    self.table.weight, self.table.weight_scale, self.binding.weight_scale_2,
                    self.ids, self.binding.num_tokens, self.out, token_count=prepared,
                )
        torch.cuda.synchronize()
        result = self.check_rows(
            ids[:prepared], self.out[:prepared].cpu(), len(selected)
        )
        result["purpose"] = (
            "Untimed injected source/TP boundary, remote and invalid row control"
        )
        result["probe_ids"] = selected
        return result


def _set_query(tensors, query, caps, model, token_map):
    device = caps.device
    live, starts = len(query["tokens"]), query["starts"]
    tensors["token_ids"].fill_(2 if model == "engram" else caps.eos_token_id)
    tensors["token_ids"][:live].copy_(torch.tensor(query["tokens"], dtype=torch.int64, device=device))
    tensors["query_start_loc"].fill_(live)
    tensors["query_start_loc"][:len(starts)].copy_(torch.tensor(starts, dtype=torch.int32, device=device))
    tensors["num_tokens"].fill_(live)
    tensors["num_seqs"].fill_(len(starts) - 1)
    history = query["history"]
    if model == "engram":
        history = [[token_map[x] if x >= 0 and x != 129264 else -1 for x in row] for row in history]
        tensors["token_mask"].fill_(False)
        tensors["token_mask"][:live].copy_(tensors["token_ids"][:live] != 129264)
    tensors["committed_history"].copy_(torch.tensor(history, dtype=torch.int64, device=device))


def make_case(args, checkpoint, model, owner, capacity, token_map, initial_query, prepared_tokens) -> Case:
    config = checkpoint.config["text_config"]
    reader_options = {} if args.queue_depth is None else {"queue_depth": args.queue_depth}
    device = torch.device("cuda", args.device)
    common = dict(device=device, max_tokens=capacity, max_seqs=args.max_seqs,
                  tp_size=args.tp_size, tp_rank=args.tp_rank)
    tensors = dict(
        token_ids=torch.empty(capacity, dtype=torch.int64, device=device),
        query_start_loc=torch.empty(args.max_seqs + 1, dtype=torch.int32, device=device),
        committed_history=torch.empty((args.max_seqs, 3 if model == "engram" else 2), dtype=torch.int64, device=device),
        num_seqs=torch.empty(1, dtype=torch.int32, device=device),
        num_tokens=torch.empty(1, dtype=torch.int32, device=device),
    )
    if model == "engram":
        geometry = build_geometry(layer_ids=config["engram_layer_ids"], base_table_size=config["engram_vocab_size"],
                                  compressed_vocab_size=config["engram_compressed_vocab_size"])
        if tuple(config["engram_num_embeddings"]) != geometry.num_embeddings:
            raise ValueError("Engram checkpoint row counts differ from planned geometry")
        caps = engram.Caps(**common, max_requests=args.max_seqs, vocab_size=config["vocab_size"], layer_id=owner)
        plan = engram.plan(caps, token_map=token_map, geometry=geometry)
        reader_options["resident_scales"] = args.engram_resident_scales
        lookup_plan = engram.plan(caps, token_map=token_map, geometry=geometry,
                                  invocation={"operation": "lookup", "compact_rows": True,
                                              "resident_scales": args.engram_resident_scales})
        source_rows = plan.query.table_rows
        planes = {0: (f"layers.{owner}.engram.embed.weight", f"layers.{owner}.engram.embed.scale")}
        widths, dtypes = (256, 8), (torch.float8_e4m3fn, (torch.uint8, torch.float8_e8m0fnu))
        tensors.update(token_mask=torch.empty(capacity, dtype=torch.bool, device=device),
                       request_slots=torch.arange(args.max_seqs, dtype=torch.int32, device=device))
    else:
        if config["ple_layer_ids"] != [2] or config["split_ngram_parts"] != 128:
            raise ValueError("Expected actual Qwen layer-1 owner and 128 row shards")
        caps = ple_embedding.Caps(
            **common, vocab_size=config["vocab_size"], eos_token_id=config["eos_token_id"],
            max_order=config["ngram_size"], heads_per_order=config["heads_per_ngram"],
            dense_layer_ordinal=0, base_table_size=config["ngram_vocab_size_base"],
            embedding_dim=config["ple_embed_dim"], table_alignment=config["make_ngram_vocab_size_divisible_by"],
            quant_mode="nvfp4_group16", table_memory="io_uring",
        )
        layout = ple_embedding.storage_layout(caps)
        for name, values in (("layer_multipliers", layout.geometry.multipliers),
                             ("ngram_heads_vocab_sizes", layout.geometry.prime_sizes),
                             ("ngram_heads_offsets", layout.geometry.table_offsets)):
            torch.testing.assert_close(checkpoint.small(PLE_PREFIX + name).reshape(-1),
                                       torch.tensor(values, dtype=torch.int64), rtol=0, atol=0)
        if layout.padded_vocab_size % 128:
            raise ValueError("PLE padded table must split evenly across 128 checkpoint shards")
        source_rows = layout.padded_vocab_size // 128
        planes = {i: (f"{PLE_PREFIX}ngram_embedding.shard_{i}.weight",
                      f"{PLE_PREFIX}ngram_embedding.shard_{i}.weight_scale")
                  for i in range(128) if i * source_rows < layout.shard_end and (i + 1) * source_rows > layout.shard_start}
        widths, dtypes = (80, 10), (torch.uint8, torch.float8_e4m3fn)

    _set_query(tensors, initial_query, caps, model, token_map)

    def register_sources(table):
        for index, keys in planes.items():
            for scale, key in enumerate(keys):
                source = checkpoint.source(key)
                allowed = dtypes[scale] if isinstance(dtypes[scale], tuple) else (dtypes[scale],)
                if tuple(source.shape) != (source_rows, widths[scale]) or source.dtype not in allowed:
                    raise ValueError(f"Unexpected row plane {key}: {source.shape}, {source.dtype}")
                table.add_shard(index, source.path, source.offset, scale=bool(scale))

    with ExitStack() as cleanup:
        session = PreparationSession(device=device, autotune=False, compile_workers=2)
        cleanup.callback(session.close)
        if model == "engram":
            def prime_hash(state):
                spec, = state.scratch_specs()
                scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
                ids = torch.empty((capacity, 24), dtype=torch.int64, device=device)
                trial = _bind_state(state, scratch=scratch, hash_ids=ids, **tensors)
                return PreparedCall(run=lambda: state.run(trial, prepared_tokens), owners=(scratch, trial))
            session.prepare((plan.request(name="engram-hash", prepare_call=prime_hash),))
            layout = plan.prepared.state
            spec, = plan.scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
            binding = engram.bind(plan, scratch=scratch,
                                  hash_ids=torch.empty((capacity, 24), dtype=torch.int64, device=device), **tensors)
            engram.run(binding, token_count=prepared_tokens)
            tables = []
            def prime_lookup(state):
                table = engram.DiskTable(state, **reader_options)
                cleanup.callback(table.close)
                register_sources(table)
                tables.append(table)
                output = torch.empty((capacity, 6144), dtype=torch.bfloat16, device=device)
                trial = _bind_lookup_state(state, hash_ids=binding.hash_ids, num_tokens=binding.num_tokens,
                                           out=output, disk_table=table)
                def run():
                    with table._cache.transaction():
                        table._cache.read_rows(binding.hash_ids, prepared_tokens * 24)
                        state.run_lookup(trial, prepared_tokens, clear_tail=True)
                return PreparedCall(run=run, output=output, owners=(trial,))
            session.prepare((lookup_plan.request(name="engram-lookup", prepare_call=prime_lookup),))
            table = tables[0]
            lookup = engram.bind_lookup(lookup_plan, hash_ids=binding.hash_ids, num_tokens=binding.num_tokens,
                                       out=torch.empty((capacity, 6144), dtype=torch.bfloat16, device=device), disk_table=table)
        else:
            table = ple_embedding.DiskTable(layout, source_rows, **reader_options)
            cleanup.callback(table._cache.close)
            register_sources(table)
            scalar = checkpoint.small(PLE_PREFIX + "ngram_embedding.weight_scale_2")
            if scalar.numel() != 1 or scalar.dtype != torch.float32:
                raise ValueError("Expected FP32 scalar Qwen global scale")
            scale = scalar.reshape(1).to(device)
            spec, = layout.scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
            output = torch.empty(layout.output_shape, dtype=layout.output_dtype, device=device)
            arguments = dict(**tensors, scratch=scratch, weight=None, weight_scale=None,
                             weight_scale_2=scale, disk_table=table, out=output)
            plan = ple_embedding.plan(caps, geometry=layout.geometry,
                                      invocation=ple_embedding.invocation_from_tensors(**arguments))
            def prime(state):
                trial_args = dict(arguments, scratch=torch.empty_like(scratch), out=torch.empty_like(output))
                trial = state.bind(**trial_args)
                return PreparedCall(run=lambda: state.run(trial, token_count=prepared_tokens),
                                    output=trial.out, owners=(trial,))
            session.prepare((plan.request(name="ple-embedding", prepare_call=prime),))
            binding = ple_embedding.bind(plan, **arguments)
            lookup = None
        result = Case(model, owner, layout, session, table, binding, lookup,
                      checkpoint, planes, source_rows, token_map, args.engram_token_bound)
        cleanup.pop_all()
        return result


def query_stream(
    args: argparse.Namespace,
    model: str,
    config: dict[str, Any],
    prepared: int,
    repeat: int,
    text_ids: list[int] | None,
):
    live = prepared - args.tail_tokens
    seqs = min(args.seqs, live)
    starts = [i * live // seqs for i in range(seqs + 1)]
    history_size = 3 if model == "engram" else 2
    pad = -1 if model == "engram" else config["eos_token_id"]
    generator = torch.Generator().manual_seed(
        args.seed + repeat * 100003 + (0 if model == "engram" else 7919)
    )
    history = [[pad] * history_size for _ in range(args.max_seqs)]
    cursor = [repeat * 97 + s * 17 for s in range(seqs)]
    if args.history_tokens:
        for s in range(seqs):
            prefix = (
                [
                    text_ids[(cursor[s] + i) % len(text_ids)]
                    for i in range(args.history_tokens)
                ]
                if text_ids
                else torch.randint(
                    config["vocab_size"], (args.history_tokens,), generator=generator
                ).tolist()
            )
            history[s] = (history[s] + prefix)[-history_size:]
            cursor[s] += len(prefix)
    for index in range(args.stream_queries):
        tokens, accepted = [], []
        for s, (begin, end) in enumerate(zip(starts[:-1], starts[1:], strict=True)):
            count = end - begin
            query = (
                [text_ids[(cursor[s] + i) % len(text_ids)] for i in range(count)]
                if text_ids
                else torch.randint(
                    config["vocab_size"], (count,), generator=generator
                ).tolist()
            )
            tokens.extend(query)
            accepted.append(max(1, math.floor(count * args.accept_fraction)))
        yield {
            "index": index,
            "tokens": tokens,
            "starts": starts,
            "history": [row[:] for row in history],
            "accepted_per_request": accepted,
            "scheduled": live,
            "accepted": sum(accepted),
        }
        for s, (begin, take) in enumerate(zip(starts[:-1], accepted, strict=True)):
            history[s] = (history[s] + tokens[begin : begin + take])[-history_size:]
            cursor[s] += take


def measure(case: Case, prepared: int, flush: Any) -> dict[str, Any]:
    # Same synchronized wall boundary as benchmark_ple_disk.measure. CUDA-event
    # timing alone is inappropriate for a transaction containing blocking CPU I/O.
    if flush is not None:
        flush()
    torch.cuda.synchronize()
    gds = case.table._cache._gds
    before = gds.native.transport_stats() if gds else None
    start = time.perf_counter_ns()
    case.run(prepared)
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter_ns() - start) / 1e6
    io = case.table.stats()
    requested = io["requested_bytes"]
    transport = None
    if gds:
        after = gds.native.transport_stats()
        transport = {key: after[key] - before[key] for key in after}
        if any(transport[key] for key in ("posix_ops", "aio_ops", "iouring_ops", "read_errors")):
            raise RuntimeError(f"GDS direct-path qualification failed: {transport}")
    return {
        "backend": case.table._cache._backend,
        "transport": transport,
        "wall_ms": wall_ms,
        "io": io,
        "d2h_id_bytes": prepared * case.heads * 8,
        "local_row_occurrences": requested // ((256 if case.table.resident_scales else 264) if case.model == "engram" else 90),
        "physical_over_logical_bytes": io["read_bytes"] / requested
        if requested
        else None,
        "native_reader_ms": io["execution_seconds"] * 1000,
        "boundary": "pre-sync; native hash through final GPU decode sync; no input preparation/checks",
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engram-checkpoint", type=pathlib.Path, default=pathlib.Path(ENGRAM_PATH)
    )
    parser.add_argument(
        "--qwen-checkpoint", type=pathlib.Path, default=pathlib.Path(QWEN_PATH)
    )
    parser.add_argument("--backends", default=os.environ.get("B12X_DISK_BACKEND", "io_uring"))
    parser.add_argument("--models", choices=("both", "engram", "qwen"), default="both")
    parser.add_argument("--capacities", default="256,1024,4096")
    parser.add_argument("--tokens", default="1,6,24,256,4096")
    parser.add_argument(
        "--tail-tokens", type=int, default=0, help="T-L; cases T<=tail are skipped"
    )
    parser.add_argument("--max-seqs", type=int, default=4)
    parser.add_argument(
        "--seqs", type=int, default=4, help="Live requests, clamped to L"
    )
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--queue-depth",
        type=int,
        default=None,
        help="Override the component's fixed reader depth (Engram 128, PLE 64)",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--stream-queries", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument(
        "--history-tokens",
        type=int,
        default=8,
        help="Accepted prefix length, capped at 4096",
    )
    parser.add_argument(
        "--accept-fraction",
        type=float,
        default=1.0,
        help="Commit only each request's accepted prefix; at least one token",
    )
    parser.add_argument(
        "--text-file",
        type=pathlib.Path,
        help="Same UTF-8 text tokenized separately with each actual tokenizer; cyclic continuation",
    )
    parser.add_argument("--check-rows", type=int, default=8)
    parser.add_argument("--engram-resident-scales", action="store_true",
                        help="Keep the Engram E8M0 plane in mapped host memory")
    parser.add_argument(
        "--engram-token-bound",
        action="store_true",
        help="Limit Engram hashing and lookup to token_count=T",
    )
    parser.add_argument(
        "--l2-flush", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--reader-library",
        type=pathlib.Path,
        help="Exact cached native b12x reader library for a diagnostic A/B arm",
    )
    parser.add_argument(
        "--output", type=pathlib.Path, default=pathlib.Path("/tmp/b12x-ngram-ssd.json")
    )
    args = parser.parse_args()
    args.backends = args.backends.split(",")
    if not args.backends or len(set(args.backends)) != len(args.backends) or set(args.backends) - {"io_uring", "gds"}:
        parser.error("backends must be io_uring, gds, or io_uring,gds")
    if args.reader_library and args.backends != ["io_uring"]:
        parser.error("reader-library selects only an io_uring object")
    try:
        args.capacities = sorted(set(map(int, args.capacities.split(","))))
        args.tokens = sorted(set(map(int, args.tokens.split(","))))
    except ValueError:
        parser.error("capacities and tokens must be comma-separated integers")
    if (
        not args.capacities
        or not args.tokens
        or min(args.capacities + args.tokens) < 1
        or max(args.capacities + args.tokens) > 4096
    ):
        parser.error("Safe bounded benchmark requires 1<=T,C<=4096")
    if (
        not 1 <= args.seqs <= args.max_seqs <= 256
        or not 1 <= args.tp_size <= 128
        or not 0 <= args.tp_rank < args.tp_size
    ):
        parser.error("Invalid request/TP topology")
    if (
        (args.queue_depth is not None and not 1 <= args.queue_depth <= 128)
        or not 1 <= args.repeats <= 100
        or not 1 <= args.stream_queries <= 128
    ):
        parser.error(
            "queue-depth<=128, repeats<=100, stream-queries<=128 must be positive"
        )
    if (
        not 1 <= args.check_rows <= 256
        or not 0 <= args.history_tokens <= 4096
        or not 0 < args.accept_fraction <= 1
        or args.tail_tokens < 0
    ):
        parser.error("Invalid correctness/history/acceptance/tail controls")
    if not any(args.tail_tokens < t <= c for c in args.capacities for t in args.tokens):
        parser.error("No eligible T/L/C case")
    if args.text_file and args.text_file.stat().st_size > 1 << 20:
        parser.error("text-file must be at most 1 MiB")
    if args.reader_library and not args.reader_library.is_file():
        parser.error("--reader-library must name an existing native library file")
    if not 0 <= args.device < torch.cuda.device_count():
        parser.error("device must name a visible CUDA device")
    return args


def main() -> None:
    from transformers import PreTrainedTokenizerFast

    args = arguments()
    torch.cuda.set_device(args.device)
    root = pathlib.Path(__file__).resolve().parents[1]
    native_reader_objects = {}
    for backend in args.backends:
        if backend == "gds":
            from b12x.loader._gds_native import load
            native_reader_path = pathlib.Path(load().__file__).resolve()
        else:
            native_reader_path = select_native_reader(args.reader_library)
        native_reader_objects[backend] = {
            **file_identity(native_reader_path), "sha256": digest(native_reader_path),
            "selection": "override" if args.reader_library else "default",
        }
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    worktree_identity = {
        **file_identity(root),
        "git_revision": revision,
        "git_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            ).strip()
        ),
    }
    flush = make_l2_flush_fn(args.l2_flush)
    print(WARNING, flush=True)
    metadata = {
        "warning": WARNING,
        "arguments": {
            k: str(v) if isinstance(v, pathlib.Path) else v
            for k, v in vars(args).items()
        },
        "command_argv": [sys.executable, *sys.argv],
        "worktree_identity": worktree_identity,
        "native_reader_objects": native_reader_objects,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("triton", "transformers", "tokenizers", "safetensors", "vllm")
        },
        "b12x_revision": revision,
        "source_sha256": {
            name: digest(root / name)
            for name in (
                "benchmarks/benchmark_ngram_ssd.py",
                "benchmarks/common.py",
                "b12x/sequence/engram/api.py",
                "b12x/sequence/engram/_kernels.py",
                "b12x/sequence/engram/geometry.py",
                "b12x/sequence/ple_embedding/_disk.py",
                "b12x/sequence/ple_embedding/_contracts.py",
                "b12x/sequence/ple_embedding/_kernels.py",
                "b12x/sequence/ple_hash/_kernels.py",
                "b12x/sequence/ple_hash/_contracts.py",
                "b12x/sequence/_shared/disk_table.py",
                "b12x/loader/_native.py",
                "b12x/loader/_ple_reader.c",
                "b12x/loader/_row_plan.h",
                "b12x/loader/_gds_reader.c",
                "b12x/loader/_gds_native.py",
                "b12x/sequence/_shared/_gds.py",
                "b12x/sequence/engram/_preparation.py",
                "b12x/sequence/ple_embedding/_preparation.py",
            )
        },
        "gpu_start": nvidia_smi_gpu_mode_snapshot(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mode": "O_DIRECT eager disk transaction, isolated single rank; no graph capture or TP collective",
        "ordering": "reverse backend order on alternate query/repeat pairs",
        "ratio_direction": "io_uring latency / GDS latency; greater than one favors GDS",
        "cufile_config": os.environ.get("CUFILE_ENV_PATH_JSON"),
        "cufile_config_sha256": digest(pathlib.Path(os.environ["CUFILE_ENV_PATH_JSON"])) if os.environ.get("CUFILE_ENV_PATH_JSON") else None,
        "query_domain": "matched text, separately tokenized cyclic streams"
        if args.text_file
        else "valid-domain random raw IDs; distinct model seeds, NOT semantically matched IDs",
        "native_reader_boundary": "execution_seconds includes native planning/dedup/read/scatter, not just SSD service time",
        "cache_contract": "No full tables or generated datasets; batch staging only. O_DIRECT bypasses host page cache, not device caches. No cold-disk claim.",
        "numerical_contracts": {
            "engram": "E4M3 * E8M0 group32 -> BF16; DEAD/image compression; all inactive output zero",
            "qwen": "NVFP4 E2M1 * E4M3 group16 * FP32 global -> BF16; raw-ID EOS history; beyond T untouched",
        },
        "text_sha256": digest(args.text_file) if args.text_file else None,
    }
    payload: dict[str, Any] = {
        "metadata": metadata,
        "checkpoints": {},
        "cases": [],
        "results": [],
        "summaries": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # JSONL spooling bounds memory even for a large user-requested sweep. The final
    # JSON embeds all raw records by streaming, never retaining all query tensors.
    spool = args.output.with_suffix(args.output.suffix + ".jsonl")
    with torch.inference_mode(), spool.open("w") as raw:
        for model in ("engram", "qwen") if args.models == "both" else (args.models,):
            checkpoint = Checkpoint(
                args.engram_checkpoint if model == "engram" else args.qwen_checkpoint
            )
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(checkpoint.root / "tokenizer.json")
            )
            token_map = None
            if model == "engram":
                token_map, compressed = build_compressed_token_map(tokenizer)
                if (
                    compressed
                    != checkpoint.config["text_config"]["engram_compressed_vocab_size"]
                ):
                    raise ValueError(
                        "Actual tokenizer compressed geometry differs from checkpoint"
                    )
            text_ids = (
                tokenizer.encode(args.text_file.read_text(), add_special_tokens=False)
                if args.text_file
                else None
            )
            if text_ids is not None and (
                not text_ids
                or min(text_ids) < 0
                or max(text_ids) >= checkpoint.config["text_config"]["vocab_size"]
            ):
                raise ValueError("Text has no tokens or is outside this model's domain")
            for capacity in args.capacities:
                for owner in (1, 14) if model == "engram" else (1,):
                    planned_tokens = next((tokens for tokens in args.tokens
                                           if args.tail_tokens < tokens <= capacity), None)
                    if planned_tokens is None:
                        continue
                    initial_query = next(query_stream(args, model, checkpoint.config["text_config"],
                                                      planned_tokens, 0, text_ids))
                    with ExitStack() as resources:
                        cases = {}
                        selected_backend = os.environ.get("B12X_DISK_BACKEND")
                        try:
                            for backend in args.backends:
                                os.environ["B12X_DISK_BACKEND"] = backend
                                case = make_case(args, checkpoint, model, owner, capacity, token_map,
                                                 initial_query, planned_tokens)
                                resources.callback(case.close)
                                cases[backend] = case
                                if case.table._cache._gds:
                                    case.table._cache._gds.native.start_stats()
                        finally:
                            if selected_backend is None:
                                os.environ.pop("B12X_DISK_BACKEND", None)
                            else:
                                os.environ["B12X_DISK_BACKEND"] = selected_backend
                        for backend, case in cases.items():
                            case.session.freeze()
                            boundary = case.boundary_check()
                            payload["cases"].append({
                                "model": model, "backend": backend, "owner": owner,
                                "capacity": capacity, "shard_start": case.layout.shard_start,
                                "shard_end": case.layout.shard_end, "table_rows": case.rows,
                                "source_rows": case.source_rows, "registered_shards": list(case.planes),
                                "boundary_correctness": boundary, "allocation_stats": case.table.stats(),
                                "output_bytes": case.out.numel() * case.out.element_size(),
                            })
                        for prepared in args.tokens:
                            if not args.tail_tokens < prepared <= capacity:
                                continue
                            first = next(query_stream(args, model, checkpoint.config["text_config"], prepared, 0, text_ids))
                            for case in cases.values():
                                case.prepare(first)
                                case.run(prepared)
                                torch.cuda.synchronize()
                                case.check(first, prepared, args.check_rows)
                            times = {backend: [] for backend in args.backends}
                            for repeat in range(args.repeats):
                                for query in query_stream(args, model, checkpoint.config["text_config"], prepared, repeat, text_ids):
                                    order = args.backends if (repeat + query["index"]) % 2 == 0 else args.backends[::-1]
                                    for backend in order:
                                        case = cases[backend]
                                        case.prepare(query)
                                        measured = measure(case, prepared, flush)
                                        correctness = case.check(query, prepared, args.check_rows)
                                        if measured["local_row_occurrences"] != correctness["local_rows"]:
                                            raise AssertionError("Reader logical byte count differs from actual local hashes")
                                        if measured["io"]["lookups"] != prepared * case.heads:
                                            raise AssertionError("Reader did not process exactly the prepared ID extent")
                                        row = {
                                            "model": model, "owner": owner, "capacity": capacity,
                                            "prepared_tokens": prepared, "repeat": repeat, "query": query,
                                            "engram_token_bound": args.engram_token_bound if model == "engram" else None,
                                            "correctness": correctness, **measured,
                                        }
                                        times[backend].append(measured["wall_ms"])
                                        raw.write(json.dumps(row) + "\n")
                                        raw.flush()
                            medians = {backend: statistics.median(values) for backend, values in times.items()}
                            for backend, values in times.items():
                                summary = {
                                    "model": model, "backend": backend, "owner": owner,
                                    "capacity": capacity, "prepared_tokens": prepared,
                                    "live_tokens": prepared - args.tail_tokens,
                                    "median_ms": medians[backend], "min_ms": min(values),
                                    "max_ms": max(values), "samples": len(values),
                                }
                                if len(medians) == 2:
                                    summary["io_uring_over_gds"] = medians["io_uring"] / medians["gds"]
                                payload["summaries"].append(summary)
                                print(json.dumps(summary), flush=True)
                        torch.cuda.synchronize()
                    del case, cases
                    gc.collect()
            checkpoint.unchanged()
            payload["checkpoints"][model] = checkpoint.report()
        metadata["gpu_end"] = nvidia_smi_gpu_mode_snapshot()
    metadata["loaded_cufile"] = sorted({line.split()[-1] for line in pathlib.Path("/proc/self/maps").read_text().splitlines() if "libcufile" in line})
    for backend in args.backends:
        for capacity in args.capacities:
            for prepared in args.tokens:
                owners = [r for r in payload["summaries"] if r["model"] == "engram" and
                          r["capacity"] == capacity and r["prepared_tokens"] == prepared and r["backend"] == backend]
                if len(owners) == 2:
                    payload["summaries"].append({
                        "model": "engram_both_owners", "backend": backend,
                        "capacity": capacity, "prepared_tokens": prepared,
                        "sum_isolated_owner_medians_ms": sum(r["median_ms"] for r in owners),
                    })
    del payload["results"]
    with args.output.open("w") as final, spool.open() as raw:
        prefix = json.dumps(payload, indent=2)
        final.write(prefix[:-1] + ',\n"results": [\n')
        separator = ""
        for line in raw:
            final.write(separator + line.rstrip())
            separator = ",\n"
        final.write("\n]}\n")
    print(f"Raw JSON: {args.output}; incremental raw JSONL: {spool}", flush=True)


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
