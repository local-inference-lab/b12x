"""Prepared public surface for compressed DSV4 sparse MLA."""

from __future__ import annotations

from b12x._lib.gating import default_is_supported
from b12x.preparation import Plan

from .._shared.mla.api import clear_mla_caches as clear_caches
from .._shared.mla.compressed_config import (
    compressed_sparse_mla_split_chunks_for_contract as split_chunks_for_contract,
)
from . import META
from ._preparation import (
    bind,
    invocation_from_descriptors,
    invocation_from_tensors,
    plan,
    run,
)
from .cache_writer import (
    CacheWriterQuery,
    page_nbytes,
    plan as plan_cache_writer,
    write_cache,
)
from ._scratch import (
    B12XCompressedSparseMLABinding as Binding,
    B12XCompressedSparseMLAScratch as Scratch,
    B12XCompressedSparseMLAScratchCaps as Caps,
)
from ._tuning import SparseMlaConfig, SparseMlaQuery


@torch.no_grad()
def prewarm(plan: Plan, *, scratch=None, sm_scale: float = 512**-0.5) -> None:
    """Resolve a capacity plan's bound kernels before capture or resolution freeze.

    Temporary query, metadata, caches and output belong only to eager warmup.
    Native dispatch uses the plan's capacity even for this single live row.
    Warmup includes sink, indexed-cache, page-mapping and both LSE-scale routes.
    """
    if not isinstance(plan, Plan):
        raise TypeError("plan must be compressed_sparse_mla.Plan")
    raise_if_kernel_resolution_frozen("compressed MLA prewarm")
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("compressed MLA prewarm requires eager execution")
    caps = plan.caps
    device = caps.device
    with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
        if scratch is None:
            scratch = tuple(
                torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
                for spec in plan.scratch_specs()
            )
        q = torch.ones((1, caps.num_q_heads, 512), dtype=torch.bfloat16, device=device)
        out = torch.empty_like(q)
        sink = torch.zeros(caps.num_q_heads, dtype=torch.float32, device=device)
        lengths = torch.zeros(1, dtype=torch.int32, device=device)
        selections = [
            torch.full((1, width), -1, dtype=torch.int32, device=device)
            for width in (caps.swa_width, caps.indexed_width)
        ]
        caches = [
            torch.zeros(
                (1, page_nbytes(size, cache_format=caps.cache_format, cache_kind=kind)),
                dtype=torch.uint8,
                device=device,
            )
            for size, kind in (
                (caps.swa_page_size, "swa"),
                (caps.indexed_page_size, "indexed"),
            )
        ]
        page_table = (
            torch.zeros(
                (1, caps.max_page_table_width), dtype=torch.int32, device=device
            )
            if caps.indexed_width
            and caps.cache_format == "deepseek_v41"
            and plan.execution_config.backend == "native"
            else None
        )
        for indexed in (False, True) if caps.indexed_width else (False,):
            for mapping in (
                (None, page_table) if indexed and page_table is not None else (None,)
            ):
                binding = bind(
                    plan,
                    scratch=scratch,
                    q=q,
                    swa_indices=selections[0],
                    swa_lengths=lengths,
                    indexed_indices=selections[1] if indexed else None,
                    indexed_lengths=lengths if indexed else None,
                    indexed_page_table=mapping,
                )
                for active_sink in (None, sink):
                    for lse_scale in ("natural", "base2"):
                        run(
                            binding=binding,
                            swa_k_cache=caches[0],
                            swa_page_size=caps.swa_page_size,
                            indexed_k_cache=caches[1] if indexed else None,
                            indexed_page_size=caps.indexed_page_size
                            if indexed
                            else None,
                            attn_sink=active_sink,
                            sm_scale=sm_scale,
                            out=out,
                            return_lse=True,
                            lse_scale=lse_scale,
                        )


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl and Triton available."""
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "CacheWriterQuery",
    "Binding",
    "Caps",
    "Plan",
    "Scratch",
    "SparseMlaConfig",
    "SparseMlaQuery",
    "bind",
    "page_nbytes",
    "plan_cache_writer",
    "clear_caches",
    "invocation_from_descriptors",
    "invocation_from_tensors",
    "is_supported",
    "write_cache",
    "plan",
    "run",
    "split_chunks_for_contract",
]
