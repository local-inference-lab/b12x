"""Packed Engram hashing and resident or SSD-backed, ceil-row-sharded FP8 lookup.

History contains three *compressed* IDs in chronological order, right-aligned
and filled with DEAD=-1 before sequence start. request_slots maps packed query
requests to history rows, so reordering never changes ownership. run never
commits history: the caller commits accepted compressed tokens separately.
EOS has no special meaning. False token_mask positions are DEAD.

Lookup returns a local contribution only. The integration sums it with the
b12x collective before applying wkv; there is no per-head TP partition.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
import operator
import os

import torch

from ..._lib.gating import default_is_supported
from ..._lib.scratch import ScratchBufferSpec, scratch_buffer_spec, scratch_tensor
from ..._lib.scratch_layout import (
    SCRATCH_ALIGN_BYTES,
    align_up,
    materialize_scratch_view,
)
from ...policy import PolicyContext, get_auto_policy
from ..ple_hash._contracts import (
    _canonical_device,
    _require_tensor,
    _require_mutation_alias_contract,
)
from .geometry import Geometry, build_compressed_token_map, build_geometry
from ._policy import ENGRAM_POLICY, EngramConfig, EngramQuery


@dataclass(frozen=True, kw_only=True)
class Caps:
    device: torch.device | str
    max_tokens: int
    max_seqs: int
    max_requests: int
    vocab_size: int = 129280
    layer_id: int = 1
    tp_size: int = 8
    tp_rank: int = 0

    def __post_init__(self):
        object.__setattr__(self, "device", _canonical_device(self.device))
        for name in ("max_tokens", "max_seqs", "max_requests", "vocab_size", "tp_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("tp_rank must be within tp_size")
        if self.max_tokens >= 2**31 or self.max_seqs >= 2**31:
            raise ValueError("packed count metadata must fit int32")


@dataclass(frozen=True, kw_only=True)
class Plan:
    caps: Caps
    geometry: Geometry
    token_map: torch.Tensor
    multipliers: torch.Tensor
    primes: torch.Tensor
    offsets: torch.Tensor
    pad_id: int
    table_rows: int
    shard_start: int
    shard_end: int
    shard_rows: int
    _scratch_specs: tuple[ScratchBufferSpec, ...]
    policy_resolution: object

    def scratch_specs(self):
        return self._scratch_specs

    @property
    def weight_shape(self):
        return (self.shard_rows, 256)

    @property
    def scale_shape(self):
        return (self.shard_rows, 8)

    def bind(self, **kwargs):
        return bind(self, **kwargs)


@dataclass(frozen=True, kw_only=True)
class Binding:
    plan: Plan
    token_ids: torch.Tensor
    token_mask: torch.Tensor
    query_start_loc: torch.Tensor
    request_slots: torch.Tensor
    committed_history: torch.Tensor
    num_seqs: torch.Tensor
    num_tokens: torch.Tensor
    hash_ids: torch.Tensor
    compressed: torch.Tensor
    request_ids: torch.Tensor
    error_code: torch.Tensor
    scratch: torch.Tensor


def _require_disk_eager(device: torch.device) -> None:
    if torch.compiler.is_compiling():
        raise RuntimeError("disk Engram preparation cannot run under torch.compile")
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "disk Engram preparation must run outside CUDA graph capture"
            )


class DiskTable:
    """Caller-owned immutable weight/scale files and a batch-bounded row cache.

    Register global FP8E4M3 rows of 256 bytes and separate raw E8M0 rows of
    eight bytes. By default each plane is one whole-table source. Offsets
    locate each plane's first byte, including inside a checkpoint container.
    Files must remain immutable for this owner's lifetime. Preparation is
    eager-only; downstream graphs consume the binding's stable BF16 output.

    resident_scales retains the owned original E8M0 bytes in mapped host RAM
    and removes scale-plane disk reads. prefetch allows one outstanding read
    per table. Both are opt-in; callers must budget scale RAM and drain reads
    before reusing request state.
    """

    def __init__(
        self,
        plan: Plan,
        shard_rows: int | None = None,
        queue_depth: int = 128,
        *,
        resident_scales: bool = False,
        prefetch: bool = False,
    ) -> None:
        from .._shared.disk_table import DiskPrefetch, DiskRowCache, MappedHostAllocation

        if not isinstance(plan, Plan):
            raise TypeError("plan must be Plan")
        if plan.caps.device.type != "cuda":
            raise ValueError("disk Engram requires CUDA")
        _require_disk_eager(plan.caps.device)
        if not isinstance(resident_scales, bool) or not isinstance(prefetch, bool):
            raise TypeError("resident_scales and prefetch must be bool")
        self.plan = plan
        self.resident_scales = resident_scales
        self._scale_sources: set[int] = set()
        self._scale_owner = None
        self._cache = DiskRowCache(
            device=plan.caps.device,
            max_lookups=plan.caps.max_tokens * 24,
            table_rows=plan.table_rows,
            shard_start=min(plan.shard_start, plan.table_rows),
            shard_end=min(plan.shard_end, plan.table_rows),
            shard_rows=plan.table_rows if shard_rows is None else shard_rows,
            weight_row_bytes=256,
            scale_row_bytes=0 if resident_scales else 8,
            queue_depth=queue_depth,
        )
        self.weight = self._cache.weight.view(torch.float8_e4m3fn)
        self.scale_bytes = self._cache.scale
        if resident_scales:
            # Retain the original E8M0 bytes, not decoded/requantized scales.
            # The caller must budget this allocation: ceil(rows/TP) * 8 bytes.
            self._scale_owner = MappedHostAllocation(
                plan.scale_shape, torch.uint8, plan.caps.device
            )
            self._scale_owner.host_view.zero_()
            self.scale_bytes = self._scale_owner.device_view
        self._prefetch = DiskPrefetch(self._cache) if prefetch else None
        self._prefetch_binding = None

    @property
    def prefetch_pending(self) -> bool:
        """Whether staging is still owned by an unconsumed prefetch."""
        return self._prefetch is not None and self._prefetch.pending

    def prefetch(self, binding: LookupBinding, token_count: int) -> None:
        """Start a bounded read; run_lookup consumes it without repeating I/O.

        All IDs and num_tokens must stay immutable until consumption. Independent
        tables may begin reads before either is consumed. Both calls must be on
        the same host thread and CUDA stream, outside compilation/capture.
        """
        if self._prefetch is None:
            raise RuntimeError("disk prefetch must be enabled at construction")
        if binding.disk_table is not self:
            raise ValueError("binding belongs to another disk table")
        if self._prefetch_binding is not None:
            raise RuntimeError("disk table has an unconsumed prefetch")
        token_count = operator.index(token_count)
        if not 0 <= token_count <= self.plan.caps.max_tokens:
            raise ValueError("token_count exceeds planned capacity")
        self._prefetch.begin(binding.hash_ids, token_count * 24)
        self._prefetch_binding = binding

    def abort_prefetch(self) -> None:
        """Drain a failed/cancelled request before its staging can be reused."""
        try:
            if self._prefetch is not None:
                self._prefetch.abort()
        finally:
            if self._prefetch is None or not self._prefetch.pending:
                self._prefetch_binding = None

    def close(self) -> None:
        """Join any outstanding I/O worker; existing GPU storage stays owned."""
        try:
            if self._prefetch is not None:
                self._prefetch.close()
        finally:
            if self._prefetch is None or not self._prefetch.pending:
                self._prefetch_binding = None

    def add_shard(
        self, index: int, path: str, offset: int, *, scale: bool = False
    ) -> None:
        """Register an immutable global source shard before binding."""
        if not (scale and self.resident_scales):
            self._cache.add_shard(index, path, offset, scale=scale)
            return
        with self._cache._lock:
            if self._cache._frozen:
                raise RuntimeError("cannot change disk shards after binding")
            index, offset = operator.index(index), operator.index(offset)
            if not 0 <= index < self._cache.shard_count or offset < 0:
                raise ValueError("invalid scale shard index or offset")
            if index in self._scale_sources:
                raise ValueError("checkpoint scale shard is already registered")
            start = index * self._cache.shard_rows
            end = min(start + self._cache.shard_rows, self.plan.table_rows)
            first, last = max(start, self.plan.shard_start), min(end, self.plan.shard_end)
            if first >= last:
                return
            view = self._scale_owner.host_view[
                first - self.plan.shard_start : last - self.plan.shard_start
            ]
            data = memoryview(view.numpy()).cast("B")
            with open(os.fspath(path), "rb", buffering=0) as source:
                if offset + (end - start) * 8 > os.fstat(source.fileno()).st_size:
                    raise ValueError("scale plane exceeds checkpoint file bounds")
                source.seek(offset + (first - start) * 8)
                done = 0
                while done < len(data):
                    count = source.readinto(data[done : done + (16 << 20)])
                    if not count:
                        raise ValueError("short resident Engram scale read")
                    done += count
            self._scale_sources.add(index)

    def _require_complete(self) -> None:
        self._cache.require_complete()
        if self.resident_scales:
            first = self._cache.shard_start // self._cache.shard_rows
            last = (self._cache.shard_end + self._cache.shard_rows - 1) // self._cache.shard_rows
            for shard in range(first, last):
                if shard not in self._scale_sources:
                    raise ValueError(f"missing resident scale shard {shard}")

    def stats(self) -> dict[str, int | float]:
        """Return shared reader counters and batch-bounded staging sizes."""
        result = self._cache.stats()
        result["resident_scale_bytes"] = self._scale_owner.nbytes if self._scale_owner else 0
        result["owned_host_bytes"] = result["owned_staging_bytes"] + result["resident_scale_bytes"]
        return result


@dataclass(frozen=True, kw_only=True)
class LookupBinding:
    plan: Plan
    weight: torch.Tensor
    scale_bytes: torch.Tensor
    hash_ids: torch.Tensor
    num_tokens: torch.Tensor
    out: torch.Tensor
    disk_table: DiskTable | None = None


def plan(
    caps: Caps,
    *,
    token_map: Sequence[int],
    geometry: Geometry | None = None,
    policy: PolicyContext | None = None,
) -> Plan:
    """Resolve once; copy host immutable tokenizer/hash geometry to the device.

    Default geometry is the checkpoint's 99092 compressed vocabulary and layers
    1/14. Explicit Geometry from build_geometry permits small oracle tables.
    Persistent tables are supplied by the loader, never allocated by bind.
    """
    geometry = geometry or build_geometry()
    if caps.layer_id not in geometry.layer_ids:
        raise ValueError("layer_id is not present in hash geometry")
    token_map = tuple(token_map)
    if (
        len(token_map) != caps.vocab_size
        or len(token_map) <= 2
        or any(not isinstance(x, int) for x in token_map)
        or set(token_map) != set(range(geometry.compressed_vocab_size))
    ):
        raise ValueError("token_map must cover the exact compressed vocabulary")
    # Do not admit arbitrary mutable/malformed geometry under a qualified policy.
    expected = build_geometry(
        layer_ids=geometry.layer_ids,
        base_table_size=geometry.primes[0][0],
        compressed_vocab_size=geometry.compressed_vocab_size,
    )
    if geometry != expected:
        raise ValueError(
            "geometry must match PCG64 and globally unreused prime construction"
        )
    index = geometry.layer_ids.index(caps.layer_id)
    rows = geometry.num_embeddings[index]
    shard_rows = (rows + caps.tp_size - 1) // caps.tp_size
    policy = policy or get_auto_policy(caps.device)
    policy.require_device(caps.device)
    resolution = policy.resolve(
        ENGRAM_POLICY,
        EngramQuery(
            max_tokens=caps.max_tokens,
            max_seqs=caps.max_seqs,
            max_requests=caps.max_requests,
            vocab_size=caps.vocab_size,
            compressed_vocab_size=geometry.compressed_vocab_size,
            layer_id=caps.layer_id,
            table_rows=rows,
            tp_size=caps.tp_size,
        ),
    )
    request_offset = align_up(caps.max_tokens * 8, SCRATCH_ALIGN_BYTES)
    error_offset = align_up(request_offset + caps.max_tokens * 4, SCRATCH_ALIGN_BYTES)
    return Plan(
        caps=caps,
        geometry=geometry,
        token_map=torch.tensor(token_map, dtype=torch.int64, device=caps.device),
        multipliers=torch.tensor(
            geometry.multipliers[index], dtype=torch.int64, device=caps.device
        ),
        primes=torch.tensor(
            geometry.primes[index], dtype=torch.int64, device=caps.device
        ),
        offsets=torch.tensor(
            geometry.offsets[index], dtype=torch.int64, device=caps.device
        ),
        pad_id=token_map[2],
        table_rows=rows,
        shard_rows=shard_rows,
        shard_start=caps.tp_rank * shard_rows,
        shard_end=(caps.tp_rank + 1) * shard_rows,
        _scratch_specs=(
            scratch_buffer_spec("engram", nbytes=error_offset + 4, device=caps.device),
        ),
        policy_resolution=resolution,
    )


def bind(
    plan: Plan,
    *,
    scratch,
    token_ids: torch.Tensor,
    token_mask: torch.Tensor,
    query_start_loc: torch.Tensor,
    request_slots: torch.Tensor,
    committed_history: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    hash_ids: torch.Tensor,
) -> Binding:
    """Bind views only; compressed query output is available for later commit.

    error_code bits: 1 invalid live counts, 2 invalid starts/slots, 4 invalid
    token/history IDs. Invalid metadata writes -1 hashes rather than gathering.
    """
    c = plan.caps
    tensors = dict(
        token_ids=token_ids,
        token_mask=token_mask,
        query_start_loc=query_start_loc,
        request_slots=request_slots,
        committed_history=committed_history,
        num_seqs=num_seqs,
        num_tokens=num_tokens,
        hash_ids=hash_ids,
    )
    layouts = dict(
        token_ids=((c.max_tokens,), torch.int64),
        token_mask=((c.max_tokens,), torch.bool),
        query_start_loc=((c.max_seqs + 1,), torch.int32),
        request_slots=((c.max_seqs,), torch.int32),
        committed_history=((c.max_requests, 3), torch.int64),
        num_seqs=((1,), torch.int32),
        num_tokens=((1,), torch.int32),
        hash_ids=((c.max_tokens, 24), torch.int64),
    )
    for name, tensor in tensors.items():
        shape, dtype = layouts[name]
        _require_tensor(name, tensor, shape=shape, dtype=dtype, device=c.device)
    storage = scratch_tensor(scratch, plan.scratch_specs(), owner="Engram")
    compressed, _ = materialize_scratch_view(
        storage, offset_bytes=0, shape=(c.max_tokens,), dtype=torch.int64
    )
    req_offset = align_up(c.max_tokens * 8, SCRATCH_ALIGN_BYTES)
    requests, _ = materialize_scratch_view(
        storage, offset_bytes=req_offset, shape=(c.max_tokens,), dtype=torch.int32
    )
    error, _ = materialize_scratch_view(
        storage,
        offset_bytes=align_up(req_offset + c.max_tokens * 4, SCRATCH_ALIGN_BYTES),
        shape=(1,),
        dtype=torch.int32,
    )
    _require_mutation_alias_contract(
        mutable=(("scratch", storage), ("hash_ids", hash_ids)),
        read_only=tuple((k, v) for k, v in tensors.items() if k != "hash_ids")
        + (
            ("token_map", plan.token_map),
            ("primes", plan.primes),
            ("offsets", plan.offsets),
            ("multipliers", plan.multipliers),
        ),
    )
    return Binding(
        plan=plan,
        **tensors,
        compressed=compressed,
        request_ids=requests,
        error_code=error,
        scratch=storage,
    )


def bind_lookup(
    plan: Plan,
    *,
    weight: torch.Tensor | None = None,
    scales: torch.Tensor | None = None,
    hash_ids: torch.Tensor,
    num_tokens: torch.Tensor,
    out: torch.Tensor,
    disk_table: DiskTable | None = None,
) -> LookupBinding:
    """Bind resident FP8E4M3/E8M0 planes or an explicitly registered disk owner.

    Resident shapes are [ceil(rows/TP),256] and [ceil(rows/TP),8]. Scales accept
    uint8 checkpoint bytes or torch.float8_e8m0fnu without conversion. Disk
    bindings require weight=scales=None and retain the frozen owner. Output is
    BF16 [max_tokens,24*256]; missing rows and inactive tokens are exact zero.
    """
    c = plan.caps
    if disk_table is not None:
        if not isinstance(disk_table, DiskTable):
            raise TypeError("disk_table must be DiskTable")
        if disk_table.plan is not plan:
            raise ValueError("disk_table must own this exact Plan")
        if weight is not None or scales is not None:
            raise ValueError("disk lookup requires weight=None and scales=None")
        _require_disk_eager(c.device)
        disk_table._require_complete()
        weight, scales = disk_table.weight, disk_table.scale_bytes
        weight_shape, scale_shape = (c.max_tokens * 24, 256), (c.max_tokens * 24, 8)
        if disk_table.resident_scales:
            scale_shape = plan.scale_shape
    else:
        if weight is None or scales is None:
            raise ValueError("resident lookup requires weight and scales")
        weight_shape, scale_shape = plan.weight_shape, plan.scale_shape
    if scales.dtype not in (torch.uint8, torch.float8_e8m0fnu):
        raise TypeError("scales must be E8M0 bytes or float8_e8m0fnu")
    scale_bytes = scales.view(torch.uint8)
    for name, tensor, shape, dtype in (
        ("weight", weight, weight_shape, torch.float8_e4m3fn),
        ("scales", scale_bytes, scale_shape, torch.uint8),
        ("hash_ids", hash_ids, (c.max_tokens, 24), torch.int64),
        ("num_tokens", num_tokens, (1,), torch.int32),
        ("out", out, (c.max_tokens, 6144), torch.bfloat16),
    ):
        _require_tensor(name, tensor, shape=shape, dtype=dtype, device=c.device)
    _require_mutation_alias_contract(
        mutable=(("out", out),),
        read_only=(
            ("weight", weight),
            ("scales", scales),
            ("hash_ids", hash_ids),
            ("num_tokens", num_tokens),
        ),
    )
    if disk_table is not None:
        disk_table._cache.freeze()
    return LookupBinding(
        plan=plan,
        weight=weight,
        scale_bytes=scale_bytes,
        hash_ids=hash_ids,
        num_tokens=num_tokens,
        out=out,
        disk_table=disk_table,
    )


def run(binding: Binding, token_count: int | None = None) -> torch.Tensor:
    """Hash immutable history/query without commits.

    ``token_count`` is a host-known upper bound, never a read of the device
    live-count scalar. Only that prefix is written; later output rows remain
    untouched. The bound changes launch grids, not compiled capacity.
    """
    from ._kernels import hash_op

    p, b = binding.plan, binding
    if p.caps.device.type != "cuda":
        raise ValueError("Engram run requires CUDA")
    prepared = p.caps.max_tokens if token_count is None else operator.index(token_count)
    if not 0 <= prepared <= p.caps.max_tokens:
        raise ValueError("token_count must be within the planned token capacity")
    hash_op(
        b.token_ids,
        b.token_mask,
        p.token_map,
        b.query_start_loc,
        b.request_slots,
        b.committed_history,
        b.num_seqs,
        b.num_tokens,
        p.multipliers,
        p.primes,
        p.offsets,
        b.compressed,
        b.request_ids,
        b.error_code,
        b.hash_ids,
        p.caps.max_requests,
        p.geometry.compressed_vocab_size,
        p.pad_id,
        prepared,
    )
    return b.hash_ids


def run_lookup(
    binding: LookupBinding, token_count: int | None = None, *, clear_tail: bool = True
) -> torch.Tensor:
    """Prepare local rows; the caller performs its existing b12x all-reduce.

    ``token_count`` is a host-known padded preparation capacity, defaulting to
    max_tokens, never a host read of the GPU live-count scalar. Tokens outside
    that capacity are zero. Disk reads and dequantization run eagerly before
    any downstream graph replay; resident lookup remains capture-compatible.
    ``clear_tail=False`` leaves rows beyond the preparation bound untouched;
    callers using it must own initialization and retired-row clearing.
    """
    from ._kernels import lookup_op

    p, b = binding.plan, binding
    if p.caps.device.type != "cuda":
        raise ValueError("Engram lookup requires CUDA")
    prepared = p.caps.max_tokens if token_count is None else operator.index(token_count)
    if not 0 <= prepared <= p.caps.max_tokens:
        raise ValueError("token_count must be within the planned token capacity")
    if b.disk_table is None:
        lookup_op(
            b.weight,
            b.scale_bytes,
            b.hash_ids,
            b.num_tokens,
            b.out,
            p.table_rows,
            p.shard_start,
            p.shard_end,
            prepared_tokens=prepared,
            clear_tail=clear_tail,
        )
    else:
        _require_disk_eager(p.caps.device)
        cache = b.disk_table._cache
        prefetched = b.disk_table._prefetch_binding
        if prefetched is not None and prefetched is not b:
            raise ValueError("disk table has a prefetch for a different binding")
        context = (
            b.disk_table._prefetch.consume(b.hash_ids, prepared * 24)
            if prefetched is not None else cache.transaction()
        )
        try:
            with context:
                if prefetched is None:
                    cache.read_rows(b.hash_ids, prepared * 24)
                lookup_op(
                    b.weight,
                    b.scale_bytes,
                    b.hash_ids,
                    b.num_tokens,
                    b.out,
                    p.table_rows,
                    p.shard_start,
                    p.shard_end,
                    compact_rows=True,
                    resident_scales=b.disk_table.resident_scales,
                    prepared_tokens=prepared,
                    clear_tail=clear_tail,
                )
        finally:
            if prefetched is not None and not b.disk_table._prefetch.pending:
                b.disk_table._prefetch_binding = None
    return b.out


def is_supported(device=None) -> bool:
    return default_is_supported(device, requires=("triton",))


__all__ = [
    "Caps",
    "Plan",
    "Binding",
    "LookupBinding",
    "DiskTable",
    "EngramQuery",
    "EngramConfig",
    "Geometry",
    "build_geometry",
    "build_compressed_token_map",
    "plan",
    "bind",
    "bind_lookup",
    "run",
    "run_lookup",
    "is_supported",
]
