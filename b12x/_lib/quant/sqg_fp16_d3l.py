"""Canonical high-rate SQG FP16-D3L reconstruction data and references.

``sqg_fp16_d3l`` is defined for K5/K6 tiles. Its graph is the same
carry-mixed L16 permutation used by SQG-XOR-Cheb-T12 at K2/K3/K4; its scalar
law is a frozen 104-entry dyadic-linear descriptor evaluated with one FP16
FMA.  The interleaved ``(base, slope)`` table is 416 bytes.
"""

from __future__ import annotations

import functools
import hashlib

import torch


SQG_FP16_D3L = "sqg_fp16_d3l"
SQG_FP16_D3L_SUBDIVISION_BITS = 3
SQG_FP16_D3L_DESCRIPTOR_COUNT = 104
SQG_FP16_D3L_DESCRIPTOR_BYTES = 416
SQG_FP16_D3L_DESCRIPTOR_SHA256 = (
    "17cf4ca9ef1e3a07c3354c12f7ac887b4e081b1668bea61eb37d8f2b410bb968"
)

# 104 interleaved little-endian FP16 (base, slope) pairs. This is checkpoint-
# independent execution data and is byte-identical to kquant's canonical law.
_DESCRIPTOR_HEX = (
    "7d4600001d460000ef450000d0450000b8450000a54500009545000087450000"
    "7b45000070450000664500005d450000544500004c450000454500003e450000"
    "32450026274500251c450025134500250a45002502450024fb440024f4440022"
    "e7440022db440022d144cd20c744cd20be449a20b5449a20ae44661ea6440020"
    "9944921e8c44551e8144ab1d7744251d6e44921c6544001c5c44921c5544db1a"
    "46444b1b3944851a2d44df192244a31918442a190f44d61806445d18fb434818"
    "dd438d17c143f016a74350169043cd157a437115664313155343bf14414372141f"
    "43241401438013e642e012cd425a12b542ee119f4287118a422b117642e7105242"
    "8b10314222101242930ff641050fdc418f0ec441230eac41c70d9741740d6e4113"
    "0d4941a10c2741440c0741ee0be940680bcd40f90ab340980a9a40400a6b40d2"
    "09414057091940f408e83fa008a33f5908613f1c08223fd007e73e7407763efd"
    "060f3e7b06ae3d1106523db905fb3c6f05a83c3105593cfc040c3ccd04f23a93"
    "04dd395604d3382704a5370304b135e9038c33d603862fc9034c00c303"
)


@functools.cache
def sqg_fp16_d3l_descriptors_cpu() -> torch.Tensor:
    """Return the frozen descriptor payload as contiguous uint8 storage."""

    raw = bytes.fromhex(_DESCRIPTOR_HEX)
    if len(raw) != SQG_FP16_D3L_DESCRIPTOR_BYTES:
        raise AssertionError("SQG FP16-D3L descriptor byte count changed")
    if hashlib.sha256(raw).hexdigest() != SQG_FP16_D3L_DESCRIPTOR_SHA256:
        raise AssertionError("SQG FP16-D3L descriptor identity changed")
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone().contiguous()


_SQG_FP16_D3L_DESCRIPTORS_DEVICE: dict[tuple[str, int | None], torch.Tensor] = {}


def _sqg_fp16_d3l_descriptors_device(
    device_type: str,
    device_index: int | None,
) -> torch.Tensor:
    key = (device_type, device_index)
    cached = _SQG_FP16_D3L_DESCRIPTORS_DEVICE.get(key)
    if cached is None:
        cached = sqg_fp16_d3l_descriptors_cpu().to(
            device=torch.device(device_type, device_index)
        ).contiguous()
        _SQG_FP16_D3L_DESCRIPTORS_DEVICE[key] = cached
    return cached


def sqg_fp16_d3l_descriptors(device: torch.device | str) -> torch.Tensor:
    """Return the process-lifetime descriptor payload for ``device``."""

    resolved = torch.device(device)
    index = resolved.index
    if resolved.type == "cuda" and index is None:
        index = torch.cuda.current_device()
    return _sqg_fp16_d3l_descriptors_device(resolved.type, index)


def sqg_fp16_d3l_descriptors_resident(
    device: torch.device | str,
) -> torch.Tensor | None:
    """Return the resident device descriptors without allocating them."""
    resolved = torch.device(device)
    index = resolved.index
    if resolved.type == "cuda" and index is None:
        index = torch.cuda.current_device()
    return _SQG_FP16_D3L_DESCRIPTORS_DEVICE.get((resolved.type, index))


def sqg_xor_high_rate_rank_for_codewords(
    codewords: torch.Tensor, bits: int
) -> torch.Tensor:
    """Apply the canonical carry-mixed SQG graph at K5 or K6."""

    if bits not in (5, 6):
        raise ValueError(f"SQG FP16-D3L supports only K5/K6, got K{bits}")
    width = 16 - bits
    history_mask = (1 << width) - 1
    branch_mask = (1 << bits) - 1
    codeword = codewords.to(torch.int64) & 0xFFFF
    history = codeword >> bits
    branch = codeword & branch_mask
    mixed = history ^ (history >> 11)
    mixed ^= (mixed << 11) & history_mask
    product = (0x3FA7D929 * mixed + 0xC928FD8E) & 0xFFFFFFFF
    phase = product & history_mask
    syndrome = product >> (32 - bits)
    reversed_branch = torch.zeros_like(branch)
    for index in range(bits):
        reversed_branch |= ((branch >> index) & 1) << (bits - 1 - index)
    return ((reversed_branch ^ syndrome) << width) | phase


def decode_sqg_fp16_d3l_ranks_torch(ranks: torch.Tensor) -> torch.Tensor:
    """Decode ranks with the exact integer coordinates and FP16-FMA law."""

    rank = ranks.to(torch.int64) & 0xFFFF
    negative = rank < 0x8000
    magnitude = rank & 0x7FFF
    magnitude = torch.where(negative, magnitude ^ 0x7FFF, magnitude)
    odd_tail = 65535 - 2 * magnitude
    exponent = torch.floor(torch.log2(odd_tail.to(torch.float64))).to(torch.int64)
    shift = torch.clamp(exponent - SQG_FP16_D3L_SUBDIVISION_BITS, min=0)
    power = 1 << exponent
    delta = odd_tail - power
    subdivision = delta >> shift
    fitted_descriptor = (shift << SQG_FP16_D3L_SUBDIVISION_BITS) + subdivision
    maximum_odd = power + (subdivision << shift) + (1 << shift) - 1
    fitted_local = (maximum_odd - odd_tail) >> 1
    exact_descriptor = (1 << torch.clamp(exponent - 1, min=0)) + ((delta - 1) >> 1)
    exact = exponent <= SQG_FP16_D3L_SUBDIVISION_BITS
    descriptor = torch.where(exact, exact_descriptor, fitted_descriptor)
    local = torch.where(exact, torch.zeros_like(fitted_local), fitted_local)

    table = sqg_fp16_d3l_descriptors_cpu().view(torch.float16).reshape(-1, 2)
    table = table.to(device=rank.device)
    local_f32 = local.to(torch.float16).to(torch.float32)
    magnitude_value = (
        table[:, 1].index_select(0, descriptor).float() * local_f32
        + table[:, 0].index_select(0, descriptor).float()
    ).to(torch.float16)
    return torch.where(negative, -magnitude_value, magnitude_value).contiguous()


@functools.cache
def sqg_fp16_d3l_direct_lut_cpu(bits: int) -> torch.Tensor:
    """Return the canonical 65,536-entry FP16 codeword oracle for tests."""

    codewords = torch.arange(1 << 16, dtype=torch.int64)
    ranks = sqg_xor_high_rate_rank_for_codewords(codewords, bits)
    return decode_sqg_fp16_d3l_ranks_torch(ranks)


__all__ = [
    "SQG_FP16_D3L",
    "SQG_FP16_D3L_DESCRIPTOR_BYTES",
    "SQG_FP16_D3L_DESCRIPTOR_COUNT",
    "SQG_FP16_D3L_DESCRIPTOR_SHA256",
    "SQG_FP16_D3L_SUBDIVISION_BITS",
    "decode_sqg_fp16_d3l_ranks_torch",
    "sqg_fp16_d3l_descriptors",
    "sqg_fp16_d3l_descriptors_cpu",
    "sqg_fp16_d3l_descriptors_resident",
    "sqg_fp16_d3l_direct_lut_cpu",
    "sqg_xor_high_rate_rank_for_codewords",
]
