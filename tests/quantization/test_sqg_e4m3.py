from __future__ import annotations

import hashlib

import torch

from b12x._lib.quant.sqg_e4m3 import (
    SQG_E4M3_DIRECT_LUT_ENTRIES,
    SQG_E4M3_RANK_LUT_ENTRIES,
    SQG_E4M3_STATE_ENTRIES,
    SQG_E4M3_STATE_LUT_ENTRIES,
    SQG_XOR_CHEB_T12_DIRECT_LUT_ENTRIES,
    _sqg_state_for_histories,
    decode_sqg_cheb_normal_e4m3_ranks_torch,
    sqg_xor_cheb_t12_direct_lut_cpu,
    sqg_cheb_normal_e4m3_direct_lut_cpu,
    sqg_cheb_normal_e4m3_rank_lut_cpu,
    sqg_cheb_normal_e4m3_state_lut_cpu,
    sqg_xor_cheb_t12_lut_cpu,
    sqg_cheb_normal_k2_q8h4_w2_e4m3_direct_lut_cpu,
)


def test_sqg_xor_cheb_t12_direct_lut_is_finite_and_bijective_by_rank() -> None:
    direct = sqg_xor_cheb_t12_direct_lut_cpu()
    assert direct.dtype == torch.uint8
    assert direct.shape == (SQG_XOR_CHEB_T12_DIRECT_LUT_ENTRIES,)
    t12 = sqg_xor_cheb_t12_lut_cpu()
    expected_histogram = torch.bincount(t12.to(torch.int64), minlength=256) * 16
    for rate_index in range(5):
        labels = direct[rate_index << 16 : (rate_index + 1) << 16]
        assert not bool(torch.any(labels == 0x80))
        assert not bool(torch.any((labels & 0x7F) == 0x7F))
        torch.testing.assert_close(
            torch.bincount(labels.to(torch.int64), minlength=256),
            expected_histogram,
            rtol=0,
            atol=0,
        )


def test_sqg_cheb_rank_descriptor_closes_all_ranks() -> None:
    table = sqg_cheb_normal_e4m3_rank_lut_cpu()
    assert table.dtype == torch.uint16
    assert table.shape == (SQG_E4M3_RANK_LUT_ENTRIES,)
    assert table.numel() * table.element_size() == 2048
    assert int(table[-1]) == 0x114A

    ranks = torch.arange(1 << 16, dtype=torch.int64)
    negative = ranks < 0x8000
    magnitude_rank = ranks & 0x7FFF
    magnitude_rank = torch.where(negative, magnitude_rank ^ 0x7FFF, magnitude_rank)
    entry = table[(magnitude_rank >> 5)].to(torch.int64)
    magnitude = (entry & 0xFF) + ((magnitude_rank & 31) >= (entry >> 8)).to(torch.int64)
    magnitude += (magnitude_rank >= 32764).to(torch.int64)
    magnitude += (magnitude_rank >= 32767).to(torch.int64)
    actual = magnitude | ((negative & (magnitude != 0)).to(torch.int64) << 7)
    expected = decode_sqg_cheb_normal_e4m3_ranks_torch(ranks).to(torch.int64)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not bool(torch.any(actual == 0x80))


def test_sqg_cheb_state_table_packs_graph_and_exact_descriptor() -> None:
    table = sqg_cheb_normal_e4m3_state_lut_cpu()
    assert table.dtype == torch.uint16
    assert table.shape == (SQG_E4M3_STATE_LUT_ENTRIES,)
    assert table.numel() * table.element_size() == 58 * 1024
    offset = 0
    for bits in (2, 3, 4):
        histories = torch.arange(1 << (16 - bits), dtype=torch.int64)
        expected = _sqg_state_for_histories(histories, bits)
        torch.testing.assert_close(
            table[offset : offset + histories.numel()],
            expected,
            rtol=0,
            atol=0,
        )
        offset += histories.numel()
    assert offset == SQG_E4M3_STATE_ENTRIES
    torch.testing.assert_close(
        table[SQG_E4M3_STATE_ENTRIES:],
        sqg_cheb_normal_e4m3_rank_lut_cpu(),
        rtol=0,
        atol=0,
    )


def test_sqg_cheb_direct_lut_matches_frozen_encoder_tables() -> None:
    direct = sqg_cheb_normal_e4m3_direct_lut_cpu()
    assert direct.dtype == torch.uint8
    assert direct.shape == (SQG_E4M3_DIRECT_LUT_ENTRIES,)
    expected_sha256 = {
        2: "1907c029b5e48254dfd3fe6c927fa1d756fcf4a2cdd3f505ac1eace24fdca6b8",
        3: "ca029489c764086d06512599256cb07947e9ee336b666bfcbf1aaa0f2785c257",
        4: "e48a0fc6a44a5d40c5a47f0313c9906041fcaf5738d442eb3817f225c910a520",
    }
    for rate_index, bits in enumerate((2, 3, 4)):
        labels = direct[rate_index << 16 : (rate_index + 1) << 16]
        assert (
            hashlib.sha256(labels.numpy().tobytes()).hexdigest()
            == expected_sha256[bits]
        )
        assert not bool(torch.any(labels == 0x80))


def test_sqg_cheb_k2_q8h4_w2_direct_lut_matches_profile_5() -> None:
    native = sqg_cheb_normal_e4m3_direct_lut_cpu()
    profile = sqg_cheb_normal_k2_q8h4_w2_e4m3_direct_lut_cpu()
    assert profile.dtype == torch.uint8
    assert profile.shape == (SQG_E4M3_DIRECT_LUT_ENTRIES,)
    expected_sha256 = {
        2: "8f1c5924211004230309288bfd45d709f99fbeacab1161e9ba0bdb55d888802e",
        3: "ca029489c764086d06512599256cb07947e9ee336b666bfcbf1aaa0f2785c257",
        4: "e48a0fc6a44a5d40c5a47f0313c9906041fcaf5738d442eb3817f225c910a520",
    }
    for rate_index, bits in enumerate((2, 3, 4)):
        labels = profile[rate_index << 16 : (rate_index + 1) << 16]
        assert (
            hashlib.sha256(labels.numpy().tobytes()).hexdigest()
            == expected_sha256[bits]
        )
        if bits > 2:
            native_labels = native[rate_index << 16 : (rate_index + 1) << 16]
            torch.testing.assert_close(labels, native_labels, rtol=0, atol=0)

    native_k2 = native[: 1 << 16]
    profile_k2 = profile[: 1 << 16]
    assert not torch.equal(profile_k2, native_k2)
    assert torch.equal(torch.sort(profile_k2).values, torch.sort(native_k2).values)
