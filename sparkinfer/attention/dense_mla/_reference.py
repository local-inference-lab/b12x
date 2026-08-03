"""High-precision paged dense-MLA oracle for the Kimi-K3 contract."""

from __future__ import annotations

import math

import torch

K3_ABSORBED_DIM = 576
K3_VALUE_DIM = 512
K3_RAW_QK_DIM = 192
K3_SM_SCALE = 1.0 / math.sqrt(K3_RAW_QK_DIM)


def _cache_rank3(cache: torch.Tensor) -> torch.Tensor:
    if cache.ndim == 4:
        if int(cache.shape[2]) != 1:
            raise ValueError("rank-4 dense MLA cache must have one KV head")
        cache = cache[:, :, 0, :]
    if cache.ndim != 3:
        raise ValueError(
            "cache must be [pages,page_size,576] or [pages,page_size,1,576]"
        )
    if int(cache.shape[-1]) != K3_ABSORBED_DIM:
        raise ValueError(f"dense MLA cache record must be {K3_ABSORBED_DIM} wide")
    return cache


def _scalar(value: torch.Tensor | float | None, *, name: str) -> float:
    if value is None:
        return 1.0
    if isinstance(value, torch.Tensor):
        if value.dtype != torch.float32 or value.numel() != 1:
            raise ValueError(f"{name} must be a scalar float32 tensor")
        return float(value.detach().cpu().item())
    return float(value)


def dense_mla_reference(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    kv_scale: torch.Tensor | float | None = None,
    q_scale: torch.Tensor | float | None = None,
    sm_scale: float = K3_SM_SCALE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the exact right-aligned causal K3 dense MLA definition.

    The cache already contains the current query chunk.  Query row ``i`` in a
    request of length ``Q`` therefore sees keys through
    ``cache_len - Q + i`` (inclusive).
    """
    cache = _cache_rank3(kv_cache)
    if q.ndim != 3 or tuple(q.shape[1:])[-1:] != (K3_ABSORBED_DIM,):
        raise ValueError("q must have shape [total_q, heads, 576]")
    if q.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise TypeError("q must be BF16 or E4M3")
    if cache.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise TypeError("kv_cache must be BF16 or E4M3")
    if page_table.ndim != 2 or page_table.dtype != torch.int32:
        raise TypeError("page_table must be rank-2 torch.int32")
    if cache_seqlens.ndim != 1 or cache_seqlens.dtype != torch.int32:
        raise TypeError("cache_seqlens must be rank-1 torch.int32")
    if cu_seqlens_q.ndim != 1 or cu_seqlens_q.dtype != torch.int32:
        raise TypeError("cu_seqlens_q must be rank-1 torch.int32")
    batch = int(page_table.shape[0])
    if tuple(cache_seqlens.shape) != (batch,):
        raise ValueError("cache_seqlens shape must match page_table batch")
    if tuple(cu_seqlens_q.shape) != (batch + 1,):
        raise ValueError("cu_seqlens_q shape must be [batch + 1]")
    if any(
        t.device != q.device for t in (cache, page_table, cache_seqlens, cu_seqlens_q)
    ):
        raise ValueError("dense MLA reference tensors must be on one device")

    kv_mul = _scalar(kv_scale, name="kv_scale")
    q_mul = _scalar(q_scale, name="q_scale")
    if cache.dtype == torch.float8_e4m3fn and kv_scale is None:
        raise ValueError("E4M3 kv_cache requires kv_scale")
    if q.dtype == torch.float8_e4m3fn and q_scale is None:
        raise ValueError("E4M3 q requires q_scale")

    cu_host = [int(v) for v in cu_seqlens_q.detach().cpu().tolist()]
    lens_host = [int(v) for v in cache_seqlens.detach().cpu().tolist()]
    if cu_host[0] != 0 or cu_host[-1] != int(q.shape[0]):
        raise ValueError("cu_seqlens_q must span exactly q.shape[0] rows")
    page_size = int(cache.shape[1])
    output = torch.empty(
        (int(q.shape[0]), int(q.shape[1]), K3_VALUE_DIM),
        dtype=torch.float32,
        device=q.device,
    )
    lse = torch.empty(
        (int(q.shape[0]), int(q.shape[1])),
        dtype=torch.float32,
        device=q.device,
    )

    for request in range(batch):
        q_begin, q_end = cu_host[request], cu_host[request + 1]
        q_len = q_end - q_begin
        kv_len = lens_host[request]
        if q_len <= 0:
            raise ValueError("every request must contain at least one query row")
        if kv_len < q_len:
            raise ValueError("cache length must include the complete query chunk")
        pages_needed = (kv_len + page_size - 1) // page_size
        if pages_needed > int(page_table.shape[1]):
            raise ValueError("page_table is too narrow for cache_seqlens")
        physical_pages = page_table[request, :pages_needed].to(torch.long)
        records = cache.index_select(0, physical_pages).reshape(-1, K3_ABSORBED_DIM)
        records = records[:kv_len].float() * kv_mul

        q_rows = q[q_begin:q_end].float() * q_mul
        for local_q in range(q_len):
            visible = kv_len - q_len + local_q + 1
            key = records[:visible]
            logits = torch.einsum("hd,kd->hk", q_rows[local_q], key)
            logits = logits * float(sm_scale)
            probs = torch.softmax(logits, dim=-1)
            output[q_begin + local_q] = torch.einsum(
                "hk,kd->hd", probs, key[:, :K3_VALUE_DIM]
            )
            lse[q_begin + local_q] = torch.logsumexp(logits, dim=-1)

    return output.to(torch.bfloat16), lse


__all__ = [
    "K3_ABSORBED_DIM",
    "K3_RAW_QK_DIM",
    "K3_SM_SCALE",
    "K3_VALUE_DIM",
    "dense_mla_reference",
]
