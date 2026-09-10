"""PLE hashing/decoding over shared batch-bounded io_uring row storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .._shared.disk_table import DiskRowCache

if TYPE_CHECKING:
    from ._contracts import Binding, Plan


class DiskTable:
    """Own PLE file sources and decode a compact batch cache outside graphs."""

    def __init__(self, plan: Plan, shard_rows: int, *, queue_depth: int = 64) -> None:
        from ._contracts import Plan

        if not isinstance(plan, Plan):
            raise TypeError("plan must be Plan")
        if plan.caps.table_memory != "io_uring":
            raise ValueError("DiskTable requires table_memory='io_uring'")
        self.plan = plan
        self._cache = DiskRowCache(
            device=plan.caps.device,
            max_lookups=plan.caps.max_tokens * plan.head_count,
            table_rows=plan.padded_vocab_size,
            shard_start=plan.shard_start,
            shard_end=plan.shard_end,
            shard_rows=shard_rows,
            weight_row_bytes=plan.weight_shape[1] * plan.weight_dtype.itemsize,
            scale_row_bytes=(
                plan.head_dim // 16 if plan.caps.quant_mode == "nvfp4_group16" else 0
            ),
            queue_depth=queue_depth,
        )
        self.weight = self._cache.weight.view(plan.weight_dtype)
        self.weight_host = self._cache.weight_host.view(plan.weight_dtype)
        self.weight_scale = (
            self._cache.scale.view(torch.float8_e4m3fn)
            if self._cache.scale is not None
            else None
        )
        self.weight_scale_host = (
            self._cache.scale_host.view(torch.float8_e4m3fn)
            if self._cache.scale_host is not None
            else None
        )

    def add_shard(
        self, shard_index: int, path: str, offset: int, *, scale: bool = False
    ) -> None:
        self._cache.add_shard(shard_index, path, offset, scale=scale)

    def _require_complete(self) -> None:
        self._cache.require_complete()

    def _freeze(self) -> None:
        self._cache.freeze()

    def stats(self) -> dict[str, int | float]:
        return self._cache.stats()

    def _run(self, binding: Binding, *, token_count: int) -> None:
        from ._kernels import (
            _launch_bf16_lookup,
            _launch_fp8_lookup,
            _launch_hash,
            _launch_nvfp4_lookup,
        )

        plan = self.plan
        caps = plan.caps
        with self._cache.transaction():
            _launch_hash(
                binding.token_ids,
                binding.query_start_loc,
                binding.committed_history,
                binding.num_seqs,
                binding.num_tokens,
                plan.multipliers,
                plan.prime_sizes,
                plan.table_offsets,
                binding._ids,
                binding._hash_binding.request_ids,
                binding.error_code,
                caps.eos_token_id,
                caps.vocab_size,
                caps.max_order,
                caps.heads_per_order,
                caps.max_seqs,
                caps.max_tokens,
                token_count,
            )
            self._cache.read_rows(binding._ids, token_count * plan.head_count)
            if token_count == 0:
                return
            args = (
                binding._ids,
                binding.num_tokens,
                binding.out[:token_count],
                caps.max_tokens,
                plan.head_count,
                plan.head_dim,
                caps.embedding_dim,
                plan.table_vocab_size,
                plan.shard_start,
                plan.shard_end,
            )
            if caps.quant_mode == "bf16":
                _launch_bf16_lookup(self.weight, *args, compact_rows=True)
            elif caps.quant_mode == "fp8_e4m3_per_tensor":
                _launch_fp8_lookup(
                    self.weight, binding.weight_scale, *args, compact_rows=True
                )
            else:
                _launch_nvfp4_lookup(
                    self.weight,
                    self.weight_scale,
                    binding.weight_scale_2,
                    *args,
                    compact_rows=True,
                )
