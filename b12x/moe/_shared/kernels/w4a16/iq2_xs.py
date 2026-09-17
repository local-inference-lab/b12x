"""Lossless preparation of safetensors IQ2_XS expert blocks."""

from __future__ import annotations

import torch

from b12x._lib.quant.iq2_xs import iq2_xs_execution_lut
from .prepare import PreparedW4A16MoeWeights, _make_workspace


def pack_iq2_xs_matrix(
    blocks: torch.Tensor, *, swap_halves: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack uint8[E,N,K/256,74] into descriptor and compact metadata planes.

    Descriptors are uint16[E,K/16,N/16,16,2], exposed as int32 words.
    Metadata holds contiguous FP16[E,K/256,N/16,16] bases followed by
    uint8[E,K/256,N/16,8,16] scale pairs. The total is 74 bytes per 256
    weights. Temporary copies contain at most 256 output rows of one expert.
    """
    if blocks.dtype != torch.uint8:
        raise TypeError("IQ2_XS blocks must be uint8")
    if blocks.ndim != 4 or blocks.shape[-1] != 74:
        raise ValueError("IQ2_XS blocks must have shape [E,N,K/256,74]")
    e, n, kb, _ = blocks.shape
    if min(e, n, kb) <= 0 or n % 16:
        raise ValueError("IQ2_XS requires positive geometry and N divisible by 16")
    if swap_halves and n % 32:
        raise ValueError("IQ2_XS projection halves must be divisible by 16")
    words = torch.empty(
        (e, kb * 16, n // 16, 16), dtype=torch.int32, device=blocks.device
    )
    metadata = torch.empty(e * kb * n * 10, dtype=torch.uint8, device=blocks.device)
    base_bytes = e * kb * n * 2
    bases = metadata[:base_bytes].view(torch.float16).reshape(e, kb, n // 16, 16)
    scales = metadata[base_bytes:].reshape(e, kb, n // 16, 8, 16)
    for expert in range(e):
        boundaries = (0, n // 2, n) if swap_halves else (0, n)
        for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            for row in range(begin, end, 256):
                stop = min(row + 256, end)
                source_row = (row + n // 2) % n if swap_halves else row
                chunk = blocks[expert, source_row : source_row + stop - row]
                d = (
                    chunk[..., :2]
                    .contiguous()
                    .view(torch.float16)
                    .reshape(stop - row, kb)
                )
                if not bool(torch.isfinite(d).all()):
                    raise ValueError("IQ2_XS block bases must be finite")
                qs = chunk[..., 2:66].contiguous().view(torch.uint16)
                q = qs.reshape(stop - row, kb * 16, 2).permute(1, 0, 2).contiguous()
                words[expert, :, row // 16 : stop // 16].copy_(
                    q.view(torch.int32).reshape(kb * 16, (stop - row) // 16, 16)
                )
                bases[expert, :, row // 16 : stop // 16].copy_(
                    d.T.reshape(kb, (stop - row) // 16, 16)
                )
                scales[expert, :, row // 16 : stop // 16].copy_(
                    chunk[..., 66:]
                    .reshape((stop - row) // 16, 16, kb, 8)
                    .permute(2, 0, 3, 1)
                )
    # Research control: materialize the K16 metadata alongside each descriptor tile.
    expanded = torch.empty((e, kb * 16, n // 16, 32), dtype=torch.int32, device=blocks.device)
    expanded[..., :16].copy_(words)
    base_bits = bases.view(torch.int16).to(torch.int32) & 0xFFFF
    for subblock in range(16):
        nibble = (scales[:, :, :, subblock // 2, :].to(torch.int32) >> (4 * (subblock % 2))) & 15
        expanded[:, subblock::16, :, 16:].copy_(base_bits | (nibble << 16))
    return expanded.reshape(-1), torch.zeros(4, dtype=torch.uint8, device=blocks.device)


def prepare_iq2_xs_moe_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    activation: str,
    w13_layout: str,
) -> PreparedW4A16MoeWeights:
    """Prepare compact IQ2_XS storage and device tables before execution."""
    if activation not in {"silu", "relu2"}:
        raise ValueError("IQ2_XS supports SiLU and ReLU²")
    if w13_layout not in {"w13", "w31"}:
        raise ValueError("IQ2_XS requires w13 or w31 projection order")
    if (
        min(hidden_size, intermediate_size, num_experts) <= 0
        or hidden_size % 256
        or intermediate_size % 256
    ):
        raise ValueError("IQ2_XS requires positive geometry and H/I divisible by 256")
    if not w13.is_cuda or w2.device != w13.device:
        raise ValueError("IQ2_XS preparation requires tensors on one CUDA device")
    gated = activation == "silu"
    expected13 = (
        num_experts,
        intermediate_size * (2 if gated else 1),
        hidden_size // 256,
        74,
    )
    expected2 = (num_experts, hidden_size, intermediate_size // 256, 74)
    if tuple(w13.shape) != expected13 or tuple(w2.shape) != expected2:
        raise ValueError(f"IQ2_XS blocks must have shapes {expected13} and {expected2}")
    q13, s13 = pack_iq2_xs_matrix(w13, swap_halves=gated and w13_layout == "w13")
    q2, s2 = pack_iq2_xs_matrix(w2)
    iq2_xs_execution_lut(w13.device, prepare=True)
    unit = torch.ones(num_experts, dtype=torch.float32, device=w13.device)
    return PreparedW4A16MoeWeights(
        w13=q13,
        w13_scale=s13,
        w13_global_scale=unit,
        w2=q2,
        w2_scale=s2,
        w2_global_scale=unit,
        workspace=_make_workspace(w13.device),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        is_gated=gated,
        params_dtype=torch.bfloat16,
        fc1_tile_n=128,
        fc2_tile_n=128,
        source_format="iq2_xs",
        w13_layout="packed",
        weight_layout="iq2_xs",
        scale_format="iq2_xs",
    )
