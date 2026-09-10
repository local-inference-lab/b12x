"""Public surface for attention.dsa_indexer (docs in the op ``__init__``).

Naming: uniform ``_paged`` / ``_contiguous`` suffixes replace the upstream
``msa_`` / ``paged_`` / ``contiguous_`` prefix mix; the pipeline stage is the
verb (quantize -> logits/block_scores -> topk/q2k_indices).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from ..._lib.gating import default_is_supported
from ...policy import PolicyContext
from ._impl import (
    clear_indexer_caches as clear_caches,
)
from .paged import (
    INDEX_HEAD_DIM,
    PAGED_INDEX_PAGE_SIZE,
    index_topk_fp8 as _run_paged_topk,
)
from .scratch import (
    INDEXER_SOURCE_LAYOUT_PAGED as SOURCE_LAYOUT_PAGED,
)
from .scratch import (
    B12XIndexerPagedBinding as PagedBinding,
)
from .scratch import B12XIndexerScratchCaps as _ScratchCaps
from .scratch import (
    B12XIndexerScratchPlan as Plan,
)
from .scratch import (
    plan_indexer_scratch,
)
from . import META
from .mxfp4 import (
    MXFP4Runtime,
    MXFP4_INDEX_PAGE_BYTES,
    bind_mxfp4,
    index_mxfp4_page_bytes,
    quantize_q_mxfp4,
    quantize_write_index_k_mxfp4,
    score_mxfp4,
    select_mxfp4,
)


@dataclass(frozen=True, kw_only=True)
class Caps:
    """Capacity and semantic inputs for paged DSA planning."""

    device: torch.device | str
    num_q_heads: int
    max_q_rows: int
    max_page_table_width: int
    topk: int
    mode: Literal["decode", "prefill"] = "decode"
    max_batch: int | None = None
    output_index_space: Literal["logical", "physical"] = "logical"
    cache_format: Literal["fp8", "mxfp4"] = "fp8"
    page_size: int = PAGED_INDEX_PAGE_SIZE
    max_candidates: int = 0
    candidate_topk_blocks: int = 0

    def __post_init__(self) -> None:
        device = torch.device(self.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        object.__setattr__(self, "device", device)
        for name in (
            "num_q_heads",
            "max_q_rows",
            "max_page_table_width",
            "topk",
            "page_size",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
            object.__setattr__(self, name, value)
        if self.mode not in ("decode", "prefill"):
            raise ValueError(f"mode must be 'decode' or 'prefill', got {self.mode!r}")
        if self.output_index_space not in ("logical", "physical"):
            raise ValueError(
                "output_index_space must be 'logical' or 'physical', got "
                f"{self.output_index_space!r}"
            )
        max_batch = self.max_q_rows if self.max_batch is None else int(self.max_batch)
        if max_batch <= 0:
            raise ValueError(f"max_batch must be positive, got {max_batch}")
        object.__setattr__(self, "max_batch", max_batch)
        object.__setattr__(self, "max_candidates", int(self.max_candidates))
        object.__setattr__(
            self, "candidate_topk_blocks", int(self.candidate_topk_blocks)
        )
        if self.cache_format not in ("fp8", "mxfp4"):
            raise ValueError("cache_format must be 'fp8' or 'mxfp4'")
        if self.cache_format == "fp8":
            if self.page_size != PAGED_INDEX_PAGE_SIZE:
                raise ValueError("FP8 indexer requires page_size=64")
            if self.max_candidates or self.candidate_topk_blocks:
                raise ValueError("candidate recipes require cache_format='mxfp4'")
        else:
            index_mxfp4_page_bytes(self.page_size)
            if self.output_index_space != "logical":
                raise ValueError("MXFP4 candidate positions are always logical")
            if self.topk != 512:
                raise ValueError("V4.1 MXFP4 indexer selects topk=512")
            if self.num_q_heads > 32 or 32 % self.num_q_heads:
                raise ValueError("MXFP4 indexer requires a divisor of 32 global heads")
            if self.max_candidates < 0 or self.max_candidates > 16384:
                raise ValueError("max_candidates must be in [0,16384]")
            if self.candidate_topk_blocks not in (0, 2048):
                raise ValueError("candidate_topk_blocks must be zero or 2048")
            if self.max_candidates and self.candidate_topk_blocks:
                raise ValueError("source and reindex recipes are mutually exclusive")


@dataclass(frozen=True, kw_only=True)
class Binding:
    """A complete paged DSA invocation bound to one immutable plan."""

    plan: Plan
    runtime: PagedBinding | MXFP4Runtime
    q_fp8: torch.Tensor | None
    query_weights: torch.Tensor
    index_k_cache: torch.Tensor
    output_indices: torch.Tensor
    output_scores: torch.Tensor | None = None
    q_mxfp4: torch.Tensor | None = None
    q_scales: torch.Tensor | None = None


def plan(caps: Caps, *, policy: PolicyContext | None = None) -> Plan:
    """Resolve policy and scratch layout once for a fixed capacity."""

    if not isinstance(caps, Caps):
        raise TypeError("caps must be dsa_indexer.Caps")
    return plan_indexer_scratch(
        _ScratchCaps(
            device=caps.device,
            source_layout=SOURCE_LAYOUT_PAGED,
            num_q_heads=caps.num_q_heads,
            max_q_rows=caps.max_q_rows,
            max_page_table_width=caps.max_page_table_width,
            topk=caps.topk,
            mode=caps.mode,
            max_batch=caps.max_batch,
            page_size=caps.page_size,
            shared_page_table=caps.mode == "prefill",
            output_physical_slots=caps.output_index_space == "physical",
            cache_format=caps.cache_format,
            max_candidates=caps.max_candidates,
            candidate_topk_blocks=caps.candidate_topk_blocks,
        ),
        policy=policy,
    )


def bind(
    plan: Plan,
    *,
    scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    q_fp8: torch.Tensor | None = None,
    query_weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_lengths: torch.Tensor,
    active_width: torch.Tensor,
    output_indices: torch.Tensor,
    output_scores: torch.Tensor | None = None,
    q_mxfp4: torch.Tensor | None = None,
    q_scales: torch.Tensor | None = None,
    candidate_indices: torch.Tensor | None = None,
    candidate_lengths: torch.Tensor | None = None,
    candidate_output: torch.Tensor | None = None,
    candidate_output_lengths: torch.Tensor | None = None,
) -> Binding:
    """Bind every live tensor to a paged DSA plan without launching work."""

    if not isinstance(plan, Plan):
        raise TypeError("plan must be dsa_indexer.Plan")
    caps = plan.caps
    if caps.source_layout != SOURCE_LAYOUT_PAGED or caps.score_mode != "dsa":
        raise ValueError("dsa_indexer.bind requires a paged DSA plan")
    if caps.cache_format == "mxfp4":
        if q_fp8 is not None:
            raise ValueError("MXFP4 packed data must be passed as q_mxfp4, not q_fp8")
        return bind_mxfp4(
            plan,
            scratch=scratch,
            q_mxfp4=q_mxfp4,
            q_scales=q_scales,
            query_weights=query_weights,
            index_k_cache=index_k_cache,
            page_table=page_table,
            cache_lengths=cache_lengths,
            active_width=active_width,
            output_indices=output_indices,
            output_scores=output_scores,
            candidate_indices=candidate_indices,
            candidate_lengths=candidate_lengths,
            candidate_output=candidate_output,
            candidate_output_lengths=candidate_output_lengths,
        )
    if any(
        x is not None
        for x in (
            q_mxfp4,
            q_scales,
            candidate_indices,
            candidate_lengths,
            candidate_output,
            candidate_output_lengths,
        )
    ):
        raise ValueError("MXFP4 tensors require an MXFP4 plan")
    if q_fp8 is None:
        raise ValueError("FP8 recipe requires q_fp8")
    expected_q_shape = (caps.num_q_heads, INDEX_HEAD_DIM)
    if q_fp8.ndim != 3 or tuple(q_fp8.shape[1:]) != expected_q_shape:
        raise ValueError(
            "q_fp8 must have shape "
            f"(rows, {caps.num_q_heads}, {INDEX_HEAD_DIM}), got "
            f"{tuple(q_fp8.shape)}"
        )
    if int(q_fp8.shape[0]) > caps.max_q_rows:
        raise ValueError(
            f"q_fp8 rows {int(q_fp8.shape[0])} exceed plan capacity {caps.max_q_rows}"
        )
    if q_fp8.dtype != torch.float8_e4m3fn:
        raise TypeError(f"q_fp8 must have dtype torch.float8_e4m3fn, got {q_fp8.dtype}")
    if q_fp8.device != caps.device or not q_fp8.is_contiguous():
        raise ValueError(f"q_fp8 must be contiguous on {caps.device}")
    if query_weights.ndim == 3 and int(query_weights.shape[-1]) == 1:
        weight_shape = tuple(query_weights.shape[:2])
    elif query_weights.ndim == 2:
        weight_shape = tuple(query_weights.shape)
    else:
        raise ValueError(
            "query_weights must have shape (rows, heads) or "
            f"(rows, heads, 1), got {tuple(query_weights.shape)}"
        )
    if weight_shape != tuple(q_fp8.shape[:2]):
        raise ValueError(
            f"query_weights must match q_fp8 rows and heads, got "
            f"{tuple(query_weights.shape)}"
        )
    if query_weights.device != caps.device or not query_weights.is_contiguous():
        raise ValueError(f"query_weights must be contiguous on {caps.device}")
    expected_cache_width = caps.page_size * (INDEX_HEAD_DIM + 4)
    if (
        index_k_cache.ndim != 2
        or index_k_cache.dtype != torch.uint8
        or int(index_k_cache.shape[1]) != expected_cache_width
    ):
        raise ValueError(
            "index_k_cache must be a rank-2 uint8 tensor with width "
            f"{expected_cache_width}, got shape={tuple(index_k_cache.shape)} "
            f"dtype={index_k_cache.dtype}"
        )
    if index_k_cache.device != caps.device or int(index_k_cache.stride(1)) != 1:
        raise ValueError(f"index_k_cache must have unit inner stride on {caps.device}")
    runtime = plan.inner.bind(
        scratch=scratch,
        real_page_table=page_table,
        cache_seqlens_int32=cache_lengths,
        active_width=active_width,
        expected_num_q_heads=caps.num_q_heads,
        shared_page_table=caps.shared_page_table,
        output_physical_slots=caps.output_physical_slots,
        _initialize=False,
    )
    q_rows = int(q_fp8.shape[0])
    if output_indices.shape != (q_rows, caps.topk):
        raise ValueError(
            "output_indices must have shape "
            f"{(q_rows, caps.topk)}, got {tuple(output_indices.shape)}"
        )
    if (
        output_indices.dtype != torch.int32
        or output_indices.device != caps.device
        or not output_indices.is_contiguous()
    ):
        raise ValueError(
            f"output_indices must be contiguous torch.int32 on {caps.device}"
        )
    if output_scores is not None and output_scores.shape != output_indices.shape:
        raise ValueError(
            "output_scores must match output_indices shape, got "
            f"{tuple(output_scores.shape)} and {tuple(output_indices.shape)}"
        )
    if output_scores is not None and (
        output_scores.dtype != torch.float32
        or output_scores.device != caps.device
        or not output_scores.is_contiguous()
    ):
        raise ValueError(
            f"output_scores must be contiguous torch.float32 on {caps.device}"
        )
    return Binding(
        plan=plan,
        runtime=runtime,
        q_fp8=q_fp8,
        query_weights=query_weights,
        index_k_cache=index_k_cache,
        output_indices=output_indices,
        output_scores=output_scores,
    )


def run(binding: Binding) -> torch.Tensor:
    """Execute a complete binding through the route selected by its plan."""

    if not isinstance(binding, Binding):
        raise TypeError("binding must be dsa_indexer.Binding")
    if binding.plan.caps.cache_format == "mxfp4":
        score(binding)
        return select(binding)
    merge_state = binding.runtime.scratch.fused_indexer_merge_state
    if merge_state is not None:
        merge_state.zero_()
    return _run_paged_topk(
        q_fp8=binding.q_fp8,
        weights=binding.query_weights,
        index_k_cache=binding.index_k_cache,
        binding=binding.runtime,
        page_size=binding.plan.caps.page_size,
        topk=binding.plan.caps.topk,
        expected_num_q_heads=binding.plan.caps.num_q_heads,
        out_indices=binding.output_indices,
        out_scores=binding.output_scores,
        allow_transient_fold_buffers=False,
    )


def score(binding: Binding) -> torch.Tensor:
    """Write BF16 rank-local weighted head sums into stable plan scratch.

    MXFP4 only. Integration must all-reduce the returned tensor **in place**
    across index-head TP ranks before ``select(binding)``. Q/K and weights are
    BF16-rounded as in the published model; weights must already include the
    global normalization ``128**-0.5 * 32**-0.5``. No local top-k is applied.
    All ranks must use identical candidate ordering and visibility metadata.
    """
    if not isinstance(binding, Binding) or binding.plan.caps.cache_format != "mxfp4":
        raise ValueError("staged score requires an MXFP4 binding")
    return score_mxfp4(binding)


def select(binding: Binding) -> torch.Tensor:
    """Select globally reduced scores, sort logical positions, and pad -1.

    Candidate sources also write their sorted, bounded block8 position list.
    This never recomputes scores. ``run`` is the single-rank/replicated-head
    convenience path; head-sharded TP callers must use ``score`` then their
    BF16 all-reduce then ``select``.
    """
    if not isinstance(binding, Binding) or binding.plan.caps.cache_format != "mxfp4":
        raise ValueError("staged select requires an MXFP4 binding")
    return select_mxfp4(binding)


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0 and triton."""
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "Caps",
    "Plan",
    "Binding",
    "plan",
    "bind",
    "run",
    "score",
    "select",
    "quantize_q_mxfp4",
    "quantize_write_index_k_mxfp4",
    "index_mxfp4_page_bytes",
    "MXFP4_INDEX_PAGE_BYTES",
    "INDEX_HEAD_DIM",
    "PAGED_INDEX_PAGE_SIZE",
    "is_supported",
    "clear_caches",
]
