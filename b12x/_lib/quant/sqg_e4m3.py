"""Exact execution tables for the frozen SQG-Cheb E4M3 compander.

The SQG graph maps each L16 codeword to one of 65,536 quantile ranks.  After
the frozen full-tail Chebyshev normal compander and E4M3 projection, the
positive half of that rank space is a monotone staircase through raw E4M3
codes 0 through 74.  Each 32-rank bucket contains at most one code transition
except the final bucket; the complete numeric mapping therefore fits in 1,024
packed uint16 entries (2 KiB) plus two explicit terminal corrections.

``entry[7:0]``
    E4M3 magnitude code at the start of the bucket.

``entry[13:8]``
    First rank offset selecting the next code, or 32 when the bucket has no
    transition.

The graph permutation remains procedural and K-specific.  The packed state
table caches each history's phase and branch syndrome so device decoders can
skip both integer mixers; the direct tables cache the complete
codeword-to-byte map per rate.  None of these process-global tables is
checkpoint state.
"""

from __future__ import annotations

import functools

import torch


SQG_E4M3_RANK_LUT_ENTRIES = 1024
SQG_E4M3_DIRECT_LUT_ENTRIES = 3 * (1 << 16)
SQG_XOR_CHEB_T12_DIRECT_LUT_ENTRIES = 5 * (1 << 16)
SQG_E4M3_STATE_ENTRIES = (1 << 14) + (1 << 13) + (1 << 12)
SQG_E4M3_STATE_LUT_ENTRIES = SQG_E4M3_STATE_ENTRIES + SQG_E4M3_RANK_LUT_ENTRIES

# Positive-half transition ranks for the E4M3-aware, full-tail normal
# SQG-Cheb staircase.  Its last 32-rank bucket contains three transitions.
# The compact descriptor stores the first cut in that bucket and the decoder
# handles the two remaining tail cuts explicitly.  This retains the 2 KiB
# representation while reproducing all 65,536 labels exactly.
_CHEB_NORMAL_POSITIVE_TRANSITIONS = (
    17,
    51,
    85,
    119,
    153,
    187,
    221,
    255,
    289,
    323,
    357,
    391,
    426,
    460,
    494,
    528,
    579,
    647,
    715,
    783,
    851,
    919,
    987,
    1055,
    1157,
    1293,
    1429,
    1565,
    1701,
    1837,
    1973,
    2108,
    2312,
    2583,
    2854,
    3124,
    3395,
    3665,
    3934,
    4203,
    4606,
    5141,
    5674,
    6205,
    6732,
    7258,
    7779,
    8298,
    9070,
    10085,
    11084,
    12065,
    13026,
    13967,
    14885,
    15781,
    17081,
    18725,
    20265,
    21696,
    23017,
    24229,
    25332,
    26330,
    27637,
    29054,
    30143,
    30957,
    31548,
    31967,
    32255,
    32447,
    32617,
    32717,
    32753,
    32764,
    32767,
)


@functools.cache
def sqg_cheb_normal_e4m3_rank_lut_cpu() -> torch.Tensor:
    """Build the exact 2 KiB descriptor for the SQG-Cheb staircase.

    Each descriptor stores the magnitude at the start of a 32-rank bucket and
    the first transition offset.  Every bucket except the last contains at
    most one transition.  The decoder adds the two remaining final-bucket
    transitions at ranks 32764 and 32767 explicitly.
    """

    entries: list[int] = []
    transition_index = 0
    base_code = 0
    descriptor_transitions = _CHEB_NORMAL_POSITIVE_TRANSITIONS[:-2]
    for bucket in range(SQG_E4M3_RANK_LUT_ENTRIES):
        start = bucket * 32
        while (
            transition_index < len(descriptor_transitions)
            and descriptor_transitions[transition_index] <= start
        ):
            base_code += 1
            transition_index += 1

        cut = 32
        if transition_index < len(descriptor_transitions):
            candidate = descriptor_transitions[transition_index] - start
            if candidate < 32:
                if candidate <= 0:
                    raise AssertionError("SQG-Cheb transition ordering is malformed")
                cut = candidate

        entries.append(base_code | (cut << 8))

    if (
        base_code != 74
        or transition_index != len(descriptor_transitions) - 1
        or descriptor_transitions[transition_index] != 32753
    ):
        raise AssertionError("SQG-Cheb descriptor does not cover the staircase")
    table = torch.tensor(entries, dtype=torch.uint16)
    if int(table[-1]) != 0x114A:
        raise AssertionError("unexpected SQG-Cheb final descriptor")
    return table


@functools.cache
def _sqg_cheb_normal_e4m3_rank_lut_device(
    device_type: str,
    device_index: int | None,
) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    return sqg_cheb_normal_e4m3_rank_lut_cpu().to(device=device).contiguous()


def sqg_cheb_normal_e4m3_rank_lut(device: torch.device | str) -> torch.Tensor:
    """Return the process-lifetime exact Cheb descriptor for ``device``."""

    resolved = torch.device(device)
    index = resolved.index
    if resolved.type == "cuda" and index is None:
        index = torch.cuda.current_device()
    return _sqg_cheb_normal_e4m3_rank_lut_device(resolved.type, index)


@functools.cache
def sqg_xor_cheb_t12_lut_cpu() -> torch.Tensor:
    """Build the byte-exact modal 12-bit compression of profile 5.

    Each entry represents sixteen consecutive global SQG ranks. Ties use the
    lower E4M3 byte, matching the transition-teacher research contract.
    """

    ranks = torch.arange(1 << 16, dtype=torch.int64)
    codes = decode_sqg_cheb_normal_e4m3_ranks_torch(ranks).reshape(1 << 12, 16)
    result = torch.empty(1 << 12, dtype=torch.uint8)
    for index, block in enumerate(codes):
        labels, counts = torch.unique(block, return_counts=True)
        result[index] = labels[counts == counts.max()].min()
    return result.contiguous()


@functools.cache
def _sqg_xor_cheb_t12_lut_device(
    device_type: str,
    device_index: int | None,
) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    return sqg_xor_cheb_t12_lut_cpu().to(device=device).contiguous()


def sqg_xor_cheb_t12_lut(device: torch.device | str) -> torch.Tensor:
    """Return the process-lifetime 4 KiB modal T12 staircase."""

    resolved = torch.device(device)
    index = resolved.index
    if resolved.type == "cuda" and index is None:
        index = torch.cuda.current_device()
    return _sqg_xor_cheb_t12_lut_device(resolved.type, index)


@functools.cache
def _sqg_xor_cheb_t12_direct_lut_device(
    device_type: str,
    device_index: int | None,
) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    return sqg_xor_cheb_t12_direct_lut_cpu().to(device=device).contiguous()


def sqg_xor_cheb_t12_direct_lut(device: torch.device | str) -> torch.Tensor:
    """Return the process-lifetime rate-indexed direct state table.

    Rows are the K2-K6 slices in rate order: byte(state, bits) =
    table[((bits - 2) << 16) | state]. Each byte precomposes the frozen
    XOR-Cheb rank map with the modal T12 staircase, so lookups are
    bit-identical to the in-kernel T12 decode.
    """

    resolved = torch.device(device)
    index = resolved.index
    if resolved.type == "cuda" and index is None:
        index = torch.cuda.current_device()
    return _sqg_xor_cheb_t12_direct_lut_device(resolved.type, index)


def _sqg_xor_cheb_t12_rank_for_codewords(
    codewords: torch.Tensor, bits: int
) -> torch.Tensor:
    """Apply the frozen SQG-XOR graph to L16 codewords."""

    if bits not in (2, 3, 4, 5, 6):
        raise ValueError(f"unsupported SQG-XOR-Cheb-T12 rate K{bits}")
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
    stratum = reversed_branch ^ syndrome
    return (stratum << width) | phase


@functools.cache
def sqg_xor_cheb_t12_direct_lut_cpu() -> torch.Tensor:
    """Build independent K2-K6 codeword tables for SQG-XOR-Cheb-T12."""

    codewords = torch.arange(1 << 16, dtype=torch.int64)
    t12 = sqg_xor_cheb_t12_lut_cpu()
    return torch.cat(
        [
            t12[_sqg_xor_cheb_t12_rank_for_codewords(codewords, bits) >> 4]
            for bits in (2, 3, 4, 5, 6)
        ]
    ).contiguous()


def _sqg_rank_for_codewords(codewords: torch.Tensor, bits: int) -> torch.Tensor:
    """Apply the frozen SQG graph permutation using integer Torch ops."""

    width = 16 - bits
    width_mask = (1 << width) - 1
    branch_mask = (1 << bits) - 1
    history = codewords >> bits
    branch = codewords & branch_mask

    def mix(
        values: torch.Tensor,
        multiplier_a: int,
        multiplier_b: int,
        shifts: tuple[int, int, int],
    ) -> torch.Tensor:
        result = values & width_mask
        result ^= result >> shifts[0]
        result = (result * (multiplier_a | 1)) & width_mask
        result ^= result >> shifts[1]
        result = (result * (multiplier_b | 1)) & width_mask
        result ^= result >> shifts[2]
        return result & width_mask

    phase = mix(history, 0x65AF, 0x16BF, (6, 4, 5))
    syndrome = mix(history ^ 0x5105, 0x8693, 0x2A21, (2, 4, 4))
    syndrome &= branch_mask
    reversed_branch = torch.zeros_like(branch)
    for index in range(bits):
        reversed_branch |= ((branch >> index) & 1) << (bits - 1 - index)
    stratum = (7 * (reversed_branch ^ syndrome)) & branch_mask
    return (stratum << width) | phase


def _sqg_state_for_histories(histories: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack the exact SQG phase and syndrome for each history state."""

    width = 16 - bits
    width_mask = (1 << width) - 1
    branch_mask = (1 << bits) - 1

    def mix(
        values: torch.Tensor,
        multiplier_a: int,
        multiplier_b: int,
        shifts: tuple[int, int, int],
    ) -> torch.Tensor:
        result = values & width_mask
        result ^= result >> shifts[0]
        result = (result * (multiplier_a | 1)) & width_mask
        result ^= result >> shifts[1]
        result = (result * (multiplier_b | 1)) & width_mask
        result ^= result >> shifts[2]
        return result & width_mask

    phase = mix(histories, 0x65AF, 0x16BF, (6, 4, 5))
    syndrome = mix(histories ^ 0x5105, 0x8693, 0x2A21, (2, 4, 4))
    syndrome &= branch_mask
    return (phase | (syndrome << width)).to(torch.uint16)


@functools.cache
def sqg_cheb_normal_e4m3_state_lut_cpu() -> torch.Tensor:
    """Build the exact 58 KiB SQG-state plus Cheb-staircase table.

    The first 56 KiB stores one packed ``phase|syndrome`` word per K2, K3,
    and K4 history state; the trailing 2 KiB is the exact rank descriptor.
    The graph state is independent of the scalar reconstruction law.
    Profile-5 K2/Q8H4 reuses the K3 state segment after its codeword bit
    swap.
    """

    states = [
        _sqg_state_for_histories(
            torch.arange(1 << (16 - bits), dtype=torch.int64), bits
        )
        for bits in (2, 3, 4)
    ]
    return torch.cat((*states, sqg_cheb_normal_e4m3_rank_lut_cpu())).contiguous()


@functools.cache
def _sqg_cheb_normal_e4m3_state_lut_device(
    device_type: str,
    device_index: int | None,
) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    return sqg_cheb_normal_e4m3_state_lut_cpu().to(device=device).contiguous()


def sqg_cheb_normal_e4m3_state_lut(device: torch.device | str) -> torch.Tensor:
    """Return the process-lifetime Cheb state-plus-rank table for ``device``."""

    resolved = torch.device(device)
    index = resolved.index
    if resolved.type == "cuda" and index is None:
        index = torch.cuda.current_device()
    return _sqg_cheb_normal_e4m3_state_lut_device(resolved.type, index)


SQG_E4M3_EXEC_LUT_K3_DIRECT_OFF = 2 * SQG_E4M3_STATE_LUT_ENTRIES
SQG_E4M3_EXEC_LUT_ENTRIES = SQG_E4M3_STATE_LUT_ENTRIES + (1 << 15)


@functools.cache
def sqg_cheb_normal_e4m3_exec_lut_cpu() -> torch.Tensor:
    """Pack the state-plus-descriptor blob with the direct K3 table.

    K3 windows decode fastest through the 64 KiB direct table (a single
    byte load), while K2/K4 windows decode through the packed graph states
    so their combined footprint stays L1-resident.  The direct K3 segment
    begins at ``SQG_E4M3_EXEC_LUT_K3_DIRECT_OFF`` bytes.
    """

    direct_k3 = sqg_cheb_normal_e4m3_direct_lut_cpu()[1 << 16 : 2 << 16]
    return torch.cat(
        (
            sqg_cheb_normal_e4m3_state_lut_cpu(),
            direct_k3.view(torch.uint16),
        )
    ).contiguous()


@functools.cache
def _sqg_cheb_normal_e4m3_exec_lut_device(
    device_type: str,
    device_index: int | None,
) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    return sqg_cheb_normal_e4m3_exec_lut_cpu().to(device=device).contiguous()


def sqg_cheb_normal_e4m3_exec_lut(device: torch.device | str) -> torch.Tensor:
    """Return the process-lifetime W4A8 execution table for ``device``."""

    resolved = torch.device(device)
    index = resolved.index
    if resolved.type == "cuda" and index is None:
        index = torch.cuda.current_device()
    return _sqg_cheb_normal_e4m3_exec_lut_device(resolved.type, index)


def decode_sqg_cheb_normal_e4m3_ranks_torch(ranks: torch.Tensor) -> torch.Tensor:
    """Decode SQG ranks through the exact full-tail Chebyshev staircase."""

    rank = ranks.to(torch.int64) & 0xFFFF
    negative = rank < 0x8000
    positive_offset = rank & 0x7FFF
    positive_offset = torch.where(negative, positive_offset ^ 0x7FFF, positive_offset)
    transitions = torch.tensor(
        _CHEB_NORMAL_POSITIVE_TRANSITIONS,
        dtype=torch.int64,
        device=positive_offset.device,
    )
    magnitude = torch.bucketize(positive_offset, transitions, right=True)
    sign = negative.to(torch.int64) << 7
    sign = torch.where(magnitude == 0, torch.zeros_like(sign), sign)
    return (magnitude | sign).to(torch.uint8)


@functools.cache
def sqg_cheb_normal_e4m3_direct_lut_cpu() -> torch.Tensor:
    """Build exact K2/K3/K4 SQG-Cheb codeword tables (192 KiB total)."""

    codewords = torch.arange(1 << 16, dtype=torch.int64)
    return torch.cat(
        [
            decode_sqg_cheb_normal_e4m3_ranks_torch(
                _sqg_rank_for_codewords(codewords, bits)
            )
            for bits in (2, 3, 4)
        ]
    ).contiguous()


@functools.cache
def _sqg_cheb_normal_e4m3_direct_lut_device(
    device_type: str,
    device_index: int | None,
) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    return sqg_cheb_normal_e4m3_direct_lut_cpu().to(device=device).contiguous()


def sqg_cheb_normal_e4m3_direct_lut(device: torch.device | str) -> torch.Tensor:
    """Return the process-lifetime exact SQG-Cheb transition table."""

    resolved = torch.device(device)
    index = resolved.index
    if resolved.type == "cuda" and index is None:
        index = torch.cuda.current_device()
    return _sqg_cheb_normal_e4m3_direct_lut_device(resolved.type, index)


def _sqg_k2_q8h4_rank_for_codewords(codewords: torch.Tensor) -> torch.Tensor:
    """Map K2 windows through the frozen w2-only virtual-octile graph.

    Profile 5 reinterprets codeword bit 4 as the third coarse-stratum bit and
    moves the native second branch bit into that history position.  Expressing
    the permutation as a bit-2/bit-4 swap followed by the ordinary K3 graph
    keeps the global Cheb-normal E4M3 staircase byte-identical.
    """

    codeword = codewords.to(torch.int64) & 0xFFFF
    swap = ((codeword >> 2) ^ (codeword >> 4)) & 1
    permuted = codeword ^ (swap << 2) ^ (swap << 4)
    return _sqg_rank_for_codewords(permuted, 3)


@functools.cache
def sqg_cheb_normal_k2_q8h4_w2_e4m3_direct_lut_cpu() -> torch.Tensor:
    """Build profile-5 K2/K3/K4 direct tables (192 KiB total).

    Only the K2 segment differs from ordinary SQG-Cheb.  The table is intended
    for FC2, whose trellis records lie on the input/K axis.  FC1 must continue
    to use :func:`sqg_cheb_normal_e4m3_direct_lut_cpu`.
    """

    codewords = torch.arange(1 << 16, dtype=torch.int64)
    ranks = (
        _sqg_k2_q8h4_rank_for_codewords(codewords),
        _sqg_rank_for_codewords(codewords, 3),
        _sqg_rank_for_codewords(codewords, 4),
    )
    return torch.cat(
        [decode_sqg_cheb_normal_e4m3_ranks_torch(rank) for rank in ranks]
    ).contiguous()


@functools.cache
def _sqg_cheb_normal_k2_q8h4_w2_e4m3_direct_lut_device(
    device_type: str,
    device_index: int | None,
) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    return (
        sqg_cheb_normal_k2_q8h4_w2_e4m3_direct_lut_cpu().to(device=device).contiguous()
    )


def sqg_cheb_normal_k2_q8h4_w2_e4m3_direct_lut(
    device: torch.device | str,
) -> torch.Tensor:
    """Return the process-lifetime profile-5 FC2 table for ``device``."""

    resolved = torch.device(device)
    index = resolved.index
    if resolved.type == "cuda" and index is None:
        index = torch.cuda.current_device()
    return _sqg_cheb_normal_k2_q8h4_w2_e4m3_direct_lut_device(resolved.type, index)


__all__ = [
    "SQG_E4M3_DIRECT_LUT_ENTRIES",
    "SQG_XOR_CHEB_T12_DIRECT_LUT_ENTRIES",
    "SQG_E4M3_EXEC_LUT_ENTRIES",
    "SQG_E4M3_EXEC_LUT_K3_DIRECT_OFF",
    "SQG_E4M3_RANK_LUT_ENTRIES",
    "SQG_E4M3_STATE_ENTRIES",
    "SQG_E4M3_STATE_LUT_ENTRIES",
    "decode_sqg_cheb_normal_e4m3_ranks_torch",
    "sqg_xor_cheb_t12_direct_lut_cpu",
    "sqg_cheb_normal_e4m3_direct_lut",
    "sqg_cheb_normal_e4m3_direct_lut_cpu",
    "sqg_cheb_normal_e4m3_exec_lut",
    "sqg_cheb_normal_e4m3_exec_lut_cpu",
    "sqg_cheb_normal_e4m3_rank_lut",
    "sqg_cheb_normal_e4m3_rank_lut_cpu",
    "sqg_cheb_normal_e4m3_state_lut",
    "sqg_cheb_normal_e4m3_state_lut_cpu",
    "sqg_xor_cheb_t12_lut",
    "sqg_xor_cheb_t12_lut_cpu",
    "sqg_cheb_normal_k2_q8h4_w2_e4m3_direct_lut",
    "sqg_cheb_normal_k2_q8h4_w2_e4m3_direct_lut_cpu",
]
