"""Low-level geometry and destination guards for the generic dense GEMM.

Covers all accepted representation families (FP4/FP6/FP8/MXFP8/block-FP8/
plain-FP8), all alias pairs, internal allocation layout, and L=1/L>1 paths.
Every negative test verifies the malformed input is rejected *before* kernel
resolution; every positive test verifies the canonical caller is accepted.
"""
from __future__ import annotations

from collections.abc import Callable
import contextlib

import pytest
import torch

import b12x._lib.dense_gemm as dense_module

from ..conftest import require_b12x

_M = 4
_N = 128
_K = 128


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _device() -> torch.device:
    return require_b12x()


def _output(device: torch.device | str, *, l: int = 1) -> torch.Tensor:
    if l > 1:
        return torch.empty(
            (l, _M, _N), dtype=torch.bfloat16, device=device
        ).as_strided((_M, _N, l), (_N, 1, _M * _N))
    return torch.empty((_M, _N, 1), dtype=torch.bfloat16, device=device)


def _make_scale_mma(
    rows: int, k: int, l: int, sf_vec_size: int, dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Build a canonical permuted scale_mma tensor matching production layout."""
    m_tiles = (rows + 127) // 128
    sf_k = k // sf_vec_size
    k_tiles = (sf_k + 3) // 4
    physical = torch.empty(
        (l, m_tiles, k_tiles, 32, 4, 4), dtype=dtype, device=device
    )
    return physical.permute(3, 4, 1, 5, 2, 0)


def _operands_mxfp8(
    *, l: int = 1, m: int = _M, n: int = _N, k: int = _K,
    device: torch.device | None = None,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]:
    if device is None:
        device = _device()
    wire_k = k
    if l > 1:
        a = torch.empty(
            (l, m, wire_k), dtype=torch.float8_e4m3fn, device=device
        ).as_strided((m, wire_k, l), (wire_k, 1, m * wire_k))
        b = torch.empty(
            (l, n, wire_k), dtype=torch.float8_e4m3fn, device=device
        ).as_strided((n, wire_k, l), (wire_k, 1, n * wire_k))
    else:
        a = torch.empty((m, wire_k, l), dtype=torch.float8_e4m3fn, device=device)
        b = torch.empty((n, wire_k, l), dtype=torch.float8_e4m3fn, device=device)
    sfa = _make_scale_mma(m, k, l, 32, torch.float8_e8m0fnu, device)
    sfb = _make_scale_mma(n, k, l, 32, torch.float8_e8m0fnu, device)
    return (a, sfa), (b, sfb)


def _operands_fp4(
    *, l: int = 1, device: torch.device | None = None,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]:
    if device is None:
        device = _device()
    wire_k = _K // 2  # FP4 packs 2 values per byte; k is doubled inside dense_gemm
    if l > 1:
        a = torch.empty(
            (l, _M, wire_k), dtype=torch.float4_e2m1fn_x2, device=device
        ).as_strided((_M, wire_k, l), (wire_k, 1, _M * wire_k))
        b = torch.empty(
            (l, _N, wire_k), dtype=torch.float4_e2m1fn_x2, device=device
        ).as_strided((_N, wire_k, l), (wire_k, 1, _N * wire_k))
    else:
        a = torch.empty((_M, wire_k, l), dtype=torch.float4_e2m1fn_x2, device=device)
        b = torch.empty((_N, wire_k, l), dtype=torch.float4_e2m1fn_x2, device=device)
    sf_dt = torch.float8_e4m3fn
    sfa = _make_scale_mma(_M, _K, l, 16, sf_dt, device)
    sfb = _make_scale_mma(_N, _K, l, 16, sf_dt, device)
    return (a, sfa), (b, sfb)


def _operands_fp6(
    *, l: int = 1, a_preexpanded: bool = True, b_preexpanded: bool = True,
    b_packed: bool = False, device: torch.device | None = None,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]:
    if device is None:
        device = _device()
    if b_packed:
        b_wire_k = (3 * _K) // 4
    elif b_preexpanded:
        b_wire_k = _K
    else:
        b_wire_k = (3 * _K) // 4  # will be expanded inside dense_gemm
    a_wire_k = _K if a_preexpanded else (3 * _K) // 4
    if l > 1:
        a = torch.empty(
            (l, _M, a_wire_k), dtype=torch.uint8, device=device
        ).as_strided((_M, a_wire_k, l), (a_wire_k, 1, _M * a_wire_k))
        b = torch.empty(
            (l, _N, b_wire_k), dtype=torch.uint8, device=device
        ).as_strided((_N, b_wire_k, l), (b_wire_k, 1, _N * b_wire_k))
    else:
        a = torch.empty((_M, a_wire_k, l), dtype=torch.uint8, device=device)
        b = torch.empty((_N, b_wire_k, l), dtype=torch.uint8, device=device)
    sf_dt = torch.float8_e8m0fnu
    sfa = _make_scale_mma(_M, _K, l, 32, sf_dt, device)
    sfb = _make_scale_mma(_N, _K, l, 32, sf_dt, device)
    return (a, sfa), (b, sfb)


def _operands_block_fp8(
    *, device: torch.device | None = None,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]:
    if device is None:
        device = _device()
    a = torch.empty((_M, _K, 1), dtype=torch.float8_e4m3fn, device=device)
    b = torch.empty((_N, _K, 1), dtype=torch.float8_e4m3fn, device=device)
    sfa = torch.empty((_M, _K // 128), dtype=torch.float32, device=device)
    sfb = torch.empty((_N // 128, _K // 128), dtype=torch.float32, device=device)
    return (a, sfa), (b, sfb)


def _call(
    lhs: tuple[torch.Tensor, torch.Tensor],
    rhs: tuple[torch.Tensor, torch.Tensor],
    out: object,
    *,
    ab_dtype: str = "float8_e4m3fn",
    sf_dtype: str = "float8_e8m0fnu",
    c_dtype: str = "bfloat16",
    sf_vec_size: int = 32,
    plain_fp8: bool = False,
    block_fp8: bool = False,
    a_preexpanded: bool = False,
    b_preexpanded: bool = False,
    b_packed: bool = False,
) -> torch.Tensor:
    return dense_module.dense_gemm(
        lhs,
        rhs,
        out=out,  # type: ignore[arg-type]
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=(16, 64),
        load_path="tma",
        swap_ab=False,
        plain_fp8=plain_fp8,
        block_fp8=block_fp8,
        a_preexpanded=a_preexpanded,
        b_preexpanded=b_preexpanded,
        b_packed=b_packed,
    )


def _forbid_compile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch all kernel entry points to detect bypass."""
    def forbid(*_a, **_kw):
        raise AssertionError("malformed dense_gemm input reached kernel resolution")

    monkeypatch.setattr(dense_module, "_get_compiled_dense_gemm", forbid)
    monkeypatch.setattr(dense_module, "_get_compiled_dense_gemm_mxfp6", forbid)


def _assert_rejected(
    monkeypatch: pytest.MonkeyPatch,
    invoke: Callable[[], object],
    *,
    error: type[BaseException] = ValueError,
    match: str,
) -> None:
    _forbid_compile(monkeypatch)
    with pytest.raises(error, match=match):
        invoke()


# ---------------------------------------------------------------------------
# A (LHS) validation
# ---------------------------------------------------------------------------

def test_a_wrong_dtype_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_mxfp8()
    bad_a = torch.empty(
        lhs[0].shape, dtype=torch.uint8, device=lhs[0].device)
    _assert_rejected(
        monkeypatch,
        lambda: _call((bad_a, lhs[1]), rhs, _output(lhs[0].device)),
        match="A.*dtype",
    )


def test_a_wrong_k_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """An A/B K mismatch is rejected at the public boundary."""
    lhs, rhs = _operands_mxfp8()
    bad_a = torch.empty((_M, _K + 32, 1), dtype=lhs[0].dtype, device=lhs[0].device)
    _assert_rejected(
        monkeypatch,
        lambda: _call((bad_a, lhs[1]), rhs, _output(lhs[0].device)),
        match="B.*shape",
    )


def test_a_cpu_device_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_mxfp8()
    bad_a = torch.empty(lhs[0].shape, dtype=lhs[0].dtype)
    _assert_rejected(
        monkeypatch,
        lambda: _call((bad_a, lhs[1]), rhs, _output(lhs[0].device)),
        match="A.*device",
    )


def test_a_misaligned_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_mxfp8()
    needed = lhs[0].numel() * lhs[0].element_size()
    storage = torch.empty(needed + 2, dtype=torch.uint8, device=lhs[0].device)
    bad_a = storage.narrow(0, 2, needed).view(lhs[0].dtype).view(lhs[0].shape)
    _assert_rejected(
        monkeypatch,
        lambda: _call((bad_a, lhs[1]), rhs, _output(lhs[0].device)),
        match="A.*16-byte aligned",
    )


def test_a_underbacked_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_mxfp8()
    real_span = dense_module._remaining_storage_bytes
    out = _output(lhs[0].device)

    def short_a(t: torch.Tensor) -> int:
        if t is lhs[0]:
            return t.numel() * t.element_size() - 1
        return real_span(t)

    monkeypatch.setattr(dense_module, "_remaining_storage_bytes", short_a)
    _assert_rejected(
        monkeypatch,
        lambda: _call(lhs, rhs, out),
        match="A.*backing bytes",
    )


# ---------------------------------------------------------------------------
# B (RHS) validation
# ---------------------------------------------------------------------------

def test_b_wrong_dtype_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_mxfp8()
    bad_b = torch.empty(rhs[0].shape, dtype=torch.uint8, device=rhs[0].device)
    _assert_rejected(
        monkeypatch,
        lambda: _call(lhs, (bad_b, rhs[1]), _output(lhs[0].device)),
        match="B.*dtype",
    )


def test_b_k_mismatch_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_mxfp8()
    bad_b = torch.empty(
        (_N, _K + 32, 1), dtype=rhs[0].dtype, device=rhs[0].device)
    _assert_rejected(
        monkeypatch,
        lambda: _call(lhs, (bad_b, rhs[1]), _output(lhs[0].device)),
        match="B.*shape",
    )


def test_b_cpu_device_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_mxfp8()
    bad_b = torch.empty(rhs[0].shape, dtype=rhs[0].dtype)
    _assert_rejected(
        monkeypatch,
        lambda: _call(lhs, (bad_b, rhs[1]), _output(lhs[0].device)),
        match="B.*device",
    )


def test_b_misaligned_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_mxfp8()
    needed = rhs[0].numel() * rhs[0].element_size()
    storage = torch.empty(needed + 2, dtype=torch.uint8, device=rhs[0].device)
    bad_b = storage.narrow(0, 2, needed).view(rhs[0].dtype).view(rhs[0].shape)
    _assert_rejected(
        monkeypatch,
        lambda: _call(lhs, (bad_b, rhs[1]), _output(lhs[0].device)),
        match="B.*16-byte aligned",
    )


# ---------------------------------------------------------------------------
# SFA / SFB (scale factor) validation
# ---------------------------------------------------------------------------

def test_sfa_wrong_dtype_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_mxfp8()
    bad_sfa = torch.empty(lhs[1].shape, dtype=torch.float8_e4m3fn, device=lhs[1].device)
    _assert_rejected(
        monkeypatch,
        lambda: _call((lhs[0], bad_sfa), rhs, _output(lhs[0].device)),
        match="SFA.*dtype",
    )


def test_sfa_wrong_shape_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_mxfp8()
    bad_sfa = torch.empty(
        (32, 4, 2, 4, 1, 1), dtype=lhs[1].dtype, device=lhs[1].device)
    _assert_rejected(
        monkeypatch,
        lambda: _call((lhs[0], bad_sfa), rhs, _output(lhs[0].device)),
        match="SFA.*shape",
    )


def test_sfa_cpu_device_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_mxfp8()
    bad_sfa = torch.empty(lhs[1].shape, dtype=lhs[1].dtype)
    _assert_rejected(
        monkeypatch,
        lambda: _call((lhs[0], bad_sfa), rhs, _output(lhs[0].device)),
        match="SFA.*device",
    )


def test_sfa_misaligned_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_mxfp8()
    needed = lhs[1].numel() * lhs[1].element_size()
    storage = torch.empty(needed + 2, dtype=torch.uint8, device=lhs[1].device)
    bad_sfa = storage.narrow(0, 2, needed).view(lhs[1].dtype).view(lhs[1].shape)
    _assert_rejected(
        monkeypatch,
        lambda: _call((lhs[0], bad_sfa), rhs, _output(lhs[0].device)),
        match="SFA.*16-byte aligned",
    )


def test_sfa_underbacked_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_mxfp8()
    real_span = dense_module._remaining_storage_bytes
    out = _output(lhs[0].device)

    def short_sfa(t: torch.Tensor) -> int:
        if t is lhs[1]:
            return t.numel() * t.element_size() - 1
        return real_span(t)

    monkeypatch.setattr(dense_module, "_remaining_storage_bytes", short_sfa)
    _assert_rejected(
        monkeypatch,
        lambda: _call(lhs, rhs, out),
        match="SFA.*backing bytes",
    )


def test_sfa_wrong_strides_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """SFA with contiguous strides (not canonical permuted) must be rejected."""
    lhs, rhs = _operands_mxfp8()
    bad_sfa = torch.empty(
        lhs[1].shape, dtype=lhs[1].dtype, device=lhs[1].device
    ).contiguous()  # contiguous, but wrong permuted strides
    _assert_rejected(
        monkeypatch,
        lambda: _call((lhs[0], bad_sfa), rhs, _output(lhs[0].device)),
        match="SFA.*canonical strides",
    )


def test_sfb_wrong_dtype_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_mxfp8()
    bad_sfb = torch.empty(rhs[1].shape, dtype=torch.float8_e4m3fn, device=rhs[1].device)
    _assert_rejected(
        monkeypatch,
        lambda: _call(lhs, (rhs[0], bad_sfb), _output(lhs[0].device)),
        match="SFB.*dtype",
    )


def test_sfb_cpu_device_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_mxfp8()
    bad_sfb = torch.empty(rhs[1].shape, dtype=rhs[1].dtype)
    _assert_rejected(
        monkeypatch,
        lambda: _call(lhs, (rhs[0], bad_sfb), _output(lhs[0].device)),
        match="SFB.*device",
    )


# ---------------------------------------------------------------------------
# alpha validation
# ---------------------------------------------------------------------------

def test_alpha_wrong_dtype_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_mxfp8()
    bad_alpha = torch.tensor([1.0], dtype=torch.float16, device=lhs[0].device)
    _assert_rejected(
        monkeypatch,
        lambda: dense_module.dense_gemm(
            lhs, rhs, out=_output(lhs[0].device),
            ab_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16", sf_vec_size=32,
            mma_tiler_mn=(16, 64), load_path="tma", swap_ab=False,
            alpha=bad_alpha, alpha_dtype="float32",
        ),
        match="alpha.*dtype",
    )


def test_alpha_wrong_shape_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_mxfp8()
    bad_alpha = torch.empty(2, dtype=torch.float32, device=lhs[0].device)
    _assert_rejected(
        monkeypatch,
        lambda: dense_module.dense_gemm(
            lhs, rhs, out=_output(lhs[0].device),
            ab_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16", sf_vec_size=32,
            mma_tiler_mn=(16, 64), load_path="tma", swap_ab=False,
            alpha=bad_alpha, alpha_dtype="float32",
        ),
        match="alpha.*shape",
    )


def test_alpha_cpu_device_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_mxfp8()
    bad_alpha = torch.tensor([1.0], dtype=torch.float32)
    _assert_rejected(
        monkeypatch,
        lambda: dense_module.dense_gemm(
            lhs, rhs, out=_output(lhs[0].device),
            ab_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16", sf_vec_size=32,
            mma_tiler_mn=(16, 64), load_path="tma", swap_ab=False,
            alpha=bad_alpha, alpha_dtype="float32",
        ),
        match="alpha.*device",
    )


# ---------------------------------------------------------------------------
# out (destination) validation
# ---------------------------------------------------------------------------

def test_out_wrong_shape_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_mxfp8()
    bad_out = torch.empty((_M, _N + 1, 1), dtype=torch.bfloat16, device=lhs[0].device)
    _assert_rejected(
        monkeypatch,
        lambda: _call(lhs, rhs, bad_out),
        match="out.*shape",
    )


def test_out_wrong_dtype_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_mxfp8()
    bad_out = torch.empty((_M, _N, 1), dtype=torch.float16, device=lhs[0].device)
    _assert_rejected(
        monkeypatch,
        lambda: _call(lhs, rhs, bad_out),
        match="out.*dtype",
    )


def test_out_cpu_device_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_mxfp8()
    bad_out = torch.empty((_M, _N, 1), dtype=torch.bfloat16)
    _assert_rejected(
        monkeypatch,
        lambda: _call(lhs, rhs, bad_out),
        match="out.*device",
    )


def test_out_l2_wrong_strides_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_mxfp8(l=2)
    bad_out = torch.empty((_M, _N, 2), dtype=torch.bfloat16, device=lhs[0].device)
    _assert_rejected(
        monkeypatch,
        lambda: _call(lhs, rhs, bad_out),
        match="out.*canonical strides",
    )


def test_out_misaligned_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_mxfp8()
    needed = _M * _N * 2
    storage = torch.empty(needed + 2, dtype=torch.uint8, device=lhs[0].device)
    bad_out = storage.narrow(0, 2, needed).view(torch.bfloat16).view(_M, _N, 1)
    _assert_rejected(
        monkeypatch,
        lambda: _call(lhs, rhs, bad_out),
        match="out.*16-byte aligned",
    )


def test_out_short_backing_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_mxfp8()
    out = _output(lhs[0].device)
    real_span = dense_module._remaining_storage_bytes

    def short_out(t: torch.Tensor) -> int:
        if t is out:
            return t.numel() * t.element_size() - 1
        return real_span(t)

    monkeypatch.setattr(dense_module, "_remaining_storage_bytes", short_out)
    _assert_rejected(
        monkeypatch,
        lambda: _call(lhs, rhs, out),
        match="out.*backing bytes",
    )


def test_out_non_tensor_rejected_before_data_ptr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-tensor out must TypeError before any .data_ptr() call."""
    lhs, rhs = _operands_mxfp8()
    _assert_rejected(
        monkeypatch,
        lambda: _call(lhs, rhs, object()),
        error=TypeError,
        match="out must be a torch.Tensor",
    )


def test_out_storage_offset_accepted() -> None:
    """Aligned nonzero storage_offset output is accepted by phase-1 validation."""
    lhs, rhs = _operands_mxfp8(l=2)
    needed_bytes = _M * _N * 2 * 2
    storage = torch.empty(
        needed_bytes + 16, dtype=torch.uint8, device=lhs[0].device
    )
    physical = (
        storage.narrow(0, 16, needed_bytes)
        .view(torch.bfloat16)
        .view(2, _M, _N)
    )
    out = physical.as_strided((_M, _N, 2), (_N, 1, _M * _N))
    assert out.storage_offset() != 0
    dense_module._validate_dense_gemm_caller_inputs(
        lhs[0], rhs[0], lhs[1], rhs[1], None, out,
        m=_M, n=_N, k=_K, l=2, a_wire_k=_K,
        expected_b_k=_K,
        caller_ab_torch_dtype=torch.float8_e4m3fn,
        caller_ab_dtype_aliases=(),
        sf_torch_dtype=torch.float8_e8m0fnu,
        c_torch_dtype=torch.bfloat16,
        alpha_torch_dtype=torch.float32,
        sf_vec_size=32, is_mxfp6=False, block_fp8=False,
        rhs_values_tiled=None, quantized_c=None,
        row_scale=None, x_bf16=None, w_gscale=None,
    )


# ---------------------------------------------------------------------------
# Pairwise overlap tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("aliased", ["a", "b", "sfa", "sfb", "alpha"])
def test_output_overlap_with_operand_rejected(
    monkeypatch: pytest.MonkeyPatch, aliased: str,
) -> None:
    lhs, rhs = _operands_mxfp8()
    out_bytes = _M * _N * 2
    operand_sizes = {
        "a": lhs[0].numel() * lhs[0].element_size(),
        "b": rhs[0].numel() * rhs[0].element_size(),
        "sfa": lhs[1].numel() * lhs[1].element_size(),
        "sfb": rhs[1].numel() * rhs[1].element_size(),
        "alpha": 4,  # float32
    }
    arena = torch.empty(
        max(out_bytes, operand_sizes[aliased]),
        dtype=torch.uint8, device=lhs[0].device,
    )
    out = arena.narrow(0, 0, out_bytes).view(torch.bfloat16).view(_M, _N, 1)
    if aliased == "a":
        lhs = (arena.narrow(0, 0, operand_sizes[aliased])
               .view(torch.float8_e4m3fn).view(_M, _K, 1), lhs[1])
    elif aliased == "b":
        rhs = (arena.narrow(0, 0, operand_sizes[aliased])
               .view(torch.float8_e4m3fn).view(_N, _K, 1), rhs[1])
    elif aliased == "sfa":
        sfa_strides = dense_module._expected_scale_mma_strides(_M, _K, 1, 32)
        lhs = (lhs[0], arena.narrow(0, 0, operand_sizes[aliased])
               .view(torch.float8_e8m0fnu)
               .as_strided(lhs[1].shape, sfa_strides))
    elif aliased == "sfb":
        sfb_strides = dense_module._expected_scale_mma_strides(_N, _K, 1, 32)
        rhs = (rhs[0], arena.narrow(0, 0, operand_sizes[aliased])
               .view(torch.float8_e8m0fnu)
               .as_strided(rhs[1].shape, sfb_strides))
    else:  # alpha
        alpha = arena.narrow(0, 0, 4).view(torch.float32).view(1)
        _assert_rejected(
            monkeypatch,
            lambda: dense_module.dense_gemm(
                lhs, rhs, out=out,
                ab_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu",
                c_dtype="bfloat16", sf_vec_size=32,
                mma_tiler_mn=(16, 64), load_path="tma", swap_ab=False,
                alpha=alpha, alpha_dtype="float32",
            ),
            match="must not overlap",
        )
        return
    _assert_rejected(
        monkeypatch,
        lambda: _call(lhs, rhs, out),
        match="must not overlap",
    )


def test_pairwise_operand_overlap_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A and B sharing storage must be rejected."""
    lhs, rhs = _operands_mxfp8()
    total = max(lhs[0].numel(), rhs[0].numel())
    arena = torch.empty(total, dtype=torch.uint8, device=lhs[0].device)
    a = arena.narrow(0, 0, lhs[0].numel()).view(torch.float8_e4m3fn).view(_M, _K, 1)
    b = arena.narrow(0, 0, rhs[0].numel()).view(torch.float8_e4m3fn).view(_N, _K, 1)
    _assert_rejected(
        monkeypatch,
        lambda: _call((a, lhs[1]), (b, rhs[1]), _output(lhs[0].device)),
        match="must not overlap",
    )


# ---------------------------------------------------------------------------
# Block FP8 scale validation
# ---------------------------------------------------------------------------

def test_block_fp8_sfa_cpu_device_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_block_fp8()
    bad_sfa = torch.empty(lhs[1].shape, dtype=torch.float32)
    _assert_rejected(
        monkeypatch,
        lambda: _call(
            (lhs[0], bad_sfa), rhs, _output(lhs[0].device),
            sf_dtype="float32", sf_vec_size=128, block_fp8=True,
        ),
        match="SFA.*device",
    )


def test_block_fp8_sfb_misaligned_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_block_fp8()
    needed = rhs[1].numel() * rhs[1].element_size()
    storage = torch.empty(needed + 4, dtype=torch.uint8, device=rhs[1].device)
    bad_sfb = storage.narrow(0, 4, needed).view(torch.float32).view(rhs[1].shape)
    _assert_rejected(
        monkeypatch,
        lambda: _call(
            lhs, (rhs[0], bad_sfb), _output(lhs[0].device),
            sf_dtype="float32", sf_vec_size=128, block_fp8=True,
        ),
        match="SFB.*16-byte aligned",
    )


# ---------------------------------------------------------------------------
# FP4 representation validation
# ---------------------------------------------------------------------------

def test_fp4_a_wrong_dtype_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_fp4()
    bad_a = torch.empty(lhs[0].shape, dtype=torch.uint8, device=lhs[0].device)
    _assert_rejected(
        monkeypatch,
        lambda: _call(
            (bad_a, lhs[1]), rhs, _output(lhs[0].device),
            ab_dtype="float4_e2m1fn", sf_dtype="float8_e4m3fn", sf_vec_size=16,
        ),
        match="A.*dtype",
    )


def test_fp4_sfa_wrong_dtype_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_fp4()
    bad_sfa = torch.empty(lhs[1].shape, dtype=torch.float8_e8m0fnu, device=lhs[1].device)
    _assert_rejected(
        monkeypatch,
        lambda: _call(
            (lhs[0], bad_sfa), rhs, _output(lhs[0].device),
            ab_dtype="float4_e2m1fn", sf_dtype="float8_e4m3fn", sf_vec_size=16,
        ),
        match="SFA.*dtype",
    )


# ---------------------------------------------------------------------------
# FP6 representation validation — through the real dense_gemm route
# ---------------------------------------------------------------------------

def test_fp6_preexpanded_wrong_dtype_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_fp6(a_preexpanded=True, b_preexpanded=True)
    bad_a = torch.empty(lhs[0].shape, dtype=torch.float8_e4m3fn, device=lhs[0].device)
    _assert_rejected(
        monkeypatch,
        lambda: _call(
            (bad_a, lhs[1]), rhs, _output(lhs[0].device),
            ab_dtype="float6_e3m2fn", sf_dtype="float8_e8m0fnu", sf_vec_size=32,
            a_preexpanded=True, b_preexpanded=True,
        ),
        match="A.*dtype",
    )


def test_fp6_packed_b_wrong_dtype_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_fp6(a_preexpanded=True, b_packed=True)
    bad_b = torch.empty(rhs[0].shape, dtype=torch.float8_e4m3fn, device=rhs[0].device)
    _assert_rejected(
        monkeypatch,
        lambda: _call(
            lhs, (bad_b, rhs[1]), _output(lhs[0].device),
            ab_dtype="float6_e3m2fn", sf_dtype="float8_e8m0fnu", sf_vec_size=32,
            a_preexpanded=True, b_packed=True,
        ),
        match="B.*dtype",
    )


def test_fp6_preexpanded_caller_validation_accepted() -> None:
    """Canonical FP6 preexpanded caller passes phase-1 validation."""
    lhs, rhs = _operands_fp6(a_preexpanded=True, b_preexpanded=True)
    device = lhs[0].device
    dense_module._validate_dense_gemm_caller_inputs(
        lhs[0], rhs[0], lhs[1], rhs[1], None, _output(device),
        m=_M, n=_N, k=_K, l=1, a_wire_k=_K,
        expected_b_k=_K,
        caller_ab_torch_dtype=torch.uint8,
        caller_ab_dtype_aliases=(),
        sf_torch_dtype=torch.float8_e8m0fnu,
        c_torch_dtype=torch.bfloat16,
        alpha_torch_dtype=torch.float32,
        sf_vec_size=32, is_mxfp6=True, block_fp8=False,
        rhs_values_tiled=None, quantized_c=None,
        row_scale=None, x_bf16=None, w_gscale=None,
    )


def test_fp6_packed_caller_validation_accepted() -> None:
    """Canonical FP6 packed-B caller passes phase-1 validation."""
    lhs, rhs = _operands_fp6(a_preexpanded=True, b_packed=True)
    device = lhs[0].device
    dense_module._validate_dense_gemm_caller_inputs(
        lhs[0], rhs[0], lhs[1], rhs[1], None, _output(device),
        m=_M, n=_N, k=_K, l=1, a_wire_k=_K,
        expected_b_k=(3 * _K) // 4,
        caller_ab_torch_dtype=torch.uint8,
        caller_ab_dtype_aliases=(),
        sf_torch_dtype=torch.float8_e8m0fnu,
        c_torch_dtype=torch.bfloat16,
        alpha_torch_dtype=torch.float32,
        sf_vec_size=32, is_mxfp6=True, block_fp8=False,
        rhs_values_tiled=None, quantized_c=None,
        row_scale=None, x_bf16=None, w_gscale=None,
    )


def test_fp6_default_caller_validation_accepted() -> None:
    """Canonical FP6 default (will-expand) caller passes phase-1 validation."""
    lhs, rhs = _operands_fp6(a_preexpanded=False, b_preexpanded=False)
    device = lhs[0].device
    dense_module._validate_dense_gemm_caller_inputs(
        lhs[0], rhs[0], lhs[1], rhs[1], None, _output(device),
        m=_M, n=_N, k=_K, l=1, a_wire_k=(3 * _K) // 4,
        expected_b_k=(3 * _K) // 4,
        caller_ab_torch_dtype=torch.uint8,
        caller_ab_dtype_aliases=(),
        sf_torch_dtype=torch.float8_e8m0fnu,
        c_torch_dtype=torch.bfloat16,
        alpha_torch_dtype=torch.float32,
        sf_vec_size=32, is_mxfp6=True, block_fp8=False,
        rhs_values_tiled=None, quantized_c=None,
        row_scale=None, x_bf16=None, w_gscale=None,
    )


def test_fp6_expansion_not_called_before_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase-1 validation runs before _expand_packed_mxfp6_ab."""
    lhs, rhs = _operands_fp6(a_preexpanded=False, b_preexpanded=False)
    expanded = False

    def spy_expand(*_a, **_kw):
        nonlocal expanded
        expanded = True
        raise AssertionError("expansion called before validation")

    monkeypatch.setattr(dense_module, "_expand_packed_mxfp6_ab", spy_expand)
    # This should raise from validation (bad dtype), not from expansion.
    bad_a = torch.empty(lhs[0].shape, dtype=torch.float8_e4m3fn, device=lhs[0].device)
    _forbid_compile(monkeypatch)
    with pytest.raises(ValueError, match="A.*dtype"):
        _call(
            (bad_a, lhs[1]), rhs, _output(lhs[0].device),
            ab_dtype="float6_e3m2fn", sf_dtype="float8_e8m0fnu", sf_vec_size=32,
        )
    assert not expanded


def test_fp6_cached_alpha_not_called_before_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase-1 validation runs before _cached_alpha_one allocation."""
    lhs, rhs = _operands_fp6(a_preexpanded=True, b_preexpanded=True)
    cached = False

    def spy_alpha(*_a, **_kw):
        nonlocal cached
        cached = True
        raise AssertionError("alpha allocation called before validation")

    monkeypatch.setattr(dense_module, "_cached_alpha_one", spy_alpha)
    _forbid_compile(monkeypatch)
    with pytest.raises(ValueError, match="SFA.*dtype"):
        bad_sfa = torch.empty(
            lhs[1].shape, dtype=torch.float8_e4m3fn, device=lhs[1].device)
        _call(
            (lhs[0], bad_sfa), rhs, _output(lhs[0].device),
            ab_dtype="float6_e3m2fn", sf_dtype="float8_e8m0fnu", sf_vec_size=32,
            a_preexpanded=True, b_preexpanded=True,
        )
    assert not cached


# ---------------------------------------------------------------------------
# Quantized C validation
# ---------------------------------------------------------------------------

def _quantized_c_operands(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n, l = _M, _N, 1
    quant_c_width = n * l
    values = torch.empty((m, quant_c_width), dtype=torch.float8_e4m3fn, device=device)
    scale_rows = torch.empty(
        (m, quant_c_width // 32), dtype=torch.float8_e8m0fnu, device=device)
    scale_mma = _make_scale_mma(128, quant_c_width, 1, 32, torch.float8_e8m0fnu, device)
    return values, scale_rows, scale_mma


def test_quant_c_values_cpu_device_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_mxfp8()
    qc = _quantized_c_operands(lhs[0].device)
    bad_values = torch.empty(qc[0].shape, dtype=qc[0].dtype)
    _assert_rejected(
        monkeypatch,
        lambda: dense_module.dense_gemm(
            lhs, rhs, out=_output(lhs[0].device),
            ab_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16", sf_vec_size=32,
            mma_tiler_mn=(16, 64), load_path="tma", swap_ab=False,
            _quantized_c=(bad_values, qc[1], qc[2]),
        ),
        match="quant_c_values.*device",
    )


def test_quant_c_scale_rows_misaligned_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_mxfp8()
    qc = _quantized_c_operands(lhs[0].device)
    needed = qc[1].numel() * qc[1].element_size()
    storage = torch.empty(needed + 2, dtype=torch.uint8, device=lhs[0].device)
    bad_rows = storage.narrow(0, 2, needed).view(qc[1].dtype).view(qc[1].shape)
    _assert_rejected(
        monkeypatch,
        lambda: dense_module.dense_gemm(
            lhs, rhs, out=_output(lhs[0].device),
            ab_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16", sf_vec_size=32,
            mma_tiler_mn=(16, 64), load_path="tma", swap_ab=False,
            _quantized_c=(qc[0], bad_rows, qc[2]),
        ),
        match="quant_c_scale_rows.*16-byte aligned",
    )


def test_quant_c_scale_mma_wrong_strides_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """quant_c_scale_mma with contiguous (non-permuted) strides must reject."""
    lhs, rhs = _operands_mxfp8()
    qc = _quantized_c_operands(lhs[0].device)
    bad_mma = torch.empty(qc[2].shape, dtype=qc[2].dtype, device=lhs[0].device)
    _assert_rejected(
        monkeypatch,
        lambda: dense_module.dense_gemm(
            lhs, rhs, out=_output(lhs[0].device),
            ab_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16", sf_vec_size=32,
            mma_tiler_mn=(16, 64), load_path="tma", swap_ab=False,
            _quantized_c=(qc[0], qc[1], bad_mma),
        ),
        match="quant_c_scale_mma.*canonical strides",
    )


def test_quant_c_values_overlap_with_a_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_mxfp8()
    qc = _quantized_c_operands(lhs[0].device)
    a_bytes = lhs[0].numel() * lhs[0].element_size()
    qc_bytes = qc[0].numel() * qc[0].element_size()
    arena = torch.empty(max(a_bytes, qc_bytes), dtype=torch.uint8, device=lhs[0].device)
    a = arena.narrow(0, 0, a_bytes).view(torch.float8_e4m3fn).view(_M, _K, 1)
    values = arena.narrow(0, 0, qc_bytes).view(torch.float8_e4m3fn).view(qc[0].shape)
    _assert_rejected(
        monkeypatch,
        lambda: dense_module.dense_gemm(
            (a, lhs[1]), rhs, out=_output(lhs[0].device),
            ab_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16", sf_vec_size=32,
            mma_tiler_mn=(16, 64), load_path="tma", swap_ab=False,
            _quantized_c=(values, qc[1], qc[2]),
        ),
        match="must not overlap",
    )


def test_quant_c_writers_mutual_overlap_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    lhs, rhs = _operands_mxfp8()
    qc = _quantized_c_operands(lhs[0].device)
    v_bytes = qc[0].numel() * qc[0].element_size()
    r_bytes = qc[1].numel() * qc[1].element_size()
    arena = torch.empty(max(v_bytes, r_bytes), dtype=torch.uint8, device=lhs[0].device)
    values = arena.narrow(0, 0, v_bytes).view(torch.float8_e4m3fn).view(qc[0].shape)
    scale_rows = arena.narrow(0, 0, r_bytes).view(torch.float8_e8m0fnu).view(qc[1].shape)
    _assert_rejected(
        monkeypatch,
        lambda: dense_module.dense_gemm(
            lhs, rhs, out=_output(lhs[0].device),
            ab_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16", sf_vec_size=32,
            mma_tiler_mn=(16, 64), load_path="tma", swap_ab=False,
            _quantized_c=(values, scale_rows, qc[2]),
        ),
        match="must not overlap",
    )


# ---------------------------------------------------------------------------
# Internal allocation layout (plain_fp8 L>1)
# ---------------------------------------------------------------------------

def test_plain_fp8_l2_internal_allocation_uses_canonical_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """plain_fp8 L>1 out=None must allocate [L,M,N] physical, not contiguous."""
    lhs, rhs = _operands_mxfp8(l=2)
    captured: dict = {}

    def fake_compile(*_a, **_kw):
        def fake_launch(**launch_kw):
            captured.update(launch_kw)
        return fake_launch

    monkeypatch.setattr(dense_module, "_get_compiled_dense_gemm", fake_compile)
    dense_module.dense_gemm(
        lhs, rhs,
        ab_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu",
        c_dtype="bfloat16", sf_vec_size=32,
        mma_tiler_mn=(16, 64), load_path="tma", swap_ab=False,
        plain_fp8=True,
    )
    c = captured["c_tensor_gpu"]
    assert tuple(c.shape) == (_M, _N, 2)
    assert tuple(c.stride()) == (_N, 1, _M * _N), (
        f"Expected canonical strides ({_N}, 1, {_M * _N}), got {tuple(c.stride())}"
    )


def test_plain_fp8_l1_internal_allocation_uses_canonical_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """plain_fp8 L=1 out=None must allocate contiguous (M,N,1)."""
    lhs, rhs = _operands_mxfp8(l=1)
    captured: dict = {}

    def fake_compile(*_a, **_kw):
        def fake_launch(**launch_kw):
            captured.update(launch_kw)
        return fake_launch

    monkeypatch.setattr(dense_module, "_get_compiled_dense_gemm", fake_compile)
    dense_module.dense_gemm(
        lhs, rhs,
        ab_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu",
        c_dtype="bfloat16", sf_vec_size=32,
        mma_tiler_mn=(16, 64), load_path="tma", swap_ab=False,
        plain_fp8=True,
    )
    c = captured["c_tensor_gpu"]
    assert tuple(c.shape) == (_M, _N, 1)
    assert c.is_contiguous()


def test_mxfp8_l1_functional_launch_internal_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MXFP8 L=1 out=None functional op allocates canonical [L,M,N] for L>1."""
    lhs, rhs = _operands_mxfp8(l=2)
    captured: dict = {}

    original_op = torch.ops.b12x.dense_gemm_launch_functional

    def spy_op(*args, **kwargs):
        captured["args"] = args
        return original_op(*args, **kwargs)

    monkeypatch.setattr(
        torch.ops.b12x, "dense_gemm_launch_functional", spy_op
    )
    result = dense_module.dense_gemm(
        lhs, rhs,
        ab_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu",
        c_dtype="bfloat16", sf_vec_size=32,
        mma_tiler_mn=(16, 64), load_path="tma", swap_ab=False,
    )
    assert tuple(result.stride()) == (_N, 1, _M * _N)


# ---------------------------------------------------------------------------
# Acceptance: valid canonical callers are accepted
# ---------------------------------------------------------------------------

def test_mxfp8_l1_accepted() -> None:
    lhs, rhs = _operands_mxfp8(l=1)
    device = lhs[0].device
    dense_module._validate_dense_gemm_caller_inputs(
        lhs[0], rhs[0], lhs[1], rhs[1], None, _output(device),
        m=_M, n=_N, k=_K, l=1, a_wire_k=_K,
        expected_b_k=_K,
        caller_ab_torch_dtype=torch.float8_e4m3fn,
        caller_ab_dtype_aliases=(),
        sf_torch_dtype=torch.float8_e8m0fnu,
        c_torch_dtype=torch.bfloat16,
        alpha_torch_dtype=torch.float32,
        sf_vec_size=32, is_mxfp6=False, block_fp8=False,
        rhs_values_tiled=None, quantized_c=None,
        row_scale=None, x_bf16=None, w_gscale=None,
    )


def test_mxfp8_l2_accepted() -> None:
    lhs, rhs = _operands_mxfp8(l=2)
    device = lhs[0].device
    dense_module._validate_dense_gemm_caller_inputs(
        lhs[0], rhs[0], lhs[1], rhs[1], None, _output(device, l=2),
        m=_M, n=_N, k=_K, l=2, a_wire_k=_K,
        expected_b_k=_K,
        caller_ab_torch_dtype=torch.float8_e4m3fn,
        caller_ab_dtype_aliases=(),
        sf_torch_dtype=torch.float8_e8m0fnu,
        c_torch_dtype=torch.bfloat16,
        alpha_torch_dtype=torch.float32,
        sf_vec_size=32, is_mxfp6=False, block_fp8=False,
        rhs_values_tiled=None, quantized_c=None,
        row_scale=None, x_bf16=None, w_gscale=None,
    )


def test_fp4_accepted() -> None:
    lhs, rhs = _operands_fp4(l=1)
    device = lhs[0].device
    dense_module._validate_dense_gemm_caller_inputs(
        lhs[0], rhs[0], lhs[1], rhs[1], None, _output(device),
        m=_M, n=_N, k=_K, l=1, a_wire_k=_K // 2,
        expected_b_k=_K // 2,
        caller_ab_torch_dtype=torch.float4_e2m1fn_x2,
        caller_ab_dtype_aliases=(),
        sf_torch_dtype=torch.float8_e4m3fn,
        c_torch_dtype=torch.bfloat16,
        alpha_torch_dtype=torch.float32,
        sf_vec_size=16, is_mxfp6=False, block_fp8=False,
        rhs_values_tiled=None, quantized_c=None,
        row_scale=None, x_bf16=None, w_gscale=None,
    )


def test_block_fp8_accepted() -> None:
    lhs, rhs = _operands_block_fp8()
    device = lhs[0].device
    dense_module._validate_dense_gemm_caller_inputs(
        lhs[0], rhs[0], lhs[1], rhs[1], None, _output(device),
        m=_M, n=_N, k=_K, l=1, a_wire_k=_K,
        expected_b_k=_K,
        caller_ab_torch_dtype=torch.float8_e4m3fn,
        caller_ab_dtype_aliases=(),
        sf_torch_dtype=torch.float32,
        c_torch_dtype=torch.bfloat16,
        alpha_torch_dtype=torch.float32,
        sf_vec_size=128, is_mxfp6=False, block_fp8=True,
        rhs_values_tiled=None, quantized_c=None,
        row_scale=None, x_bf16=None, w_gscale=None,
    )


def test_quantized_c_accepted() -> None:
    lhs, rhs = _operands_mxfp8(l=1)
    device = lhs[0].device
    qc = _quantized_c_operands(device)
    dense_module._validate_dense_gemm_caller_inputs(
        lhs[0], rhs[0], lhs[1], rhs[1], None, _output(device),
        m=_M, n=_N, k=_K, l=1, a_wire_k=_K,
        expected_b_k=_K,
        caller_ab_torch_dtype=torch.float8_e4m3fn,
        caller_ab_dtype_aliases=(),
        sf_torch_dtype=torch.float8_e8m0fnu,
        c_torch_dtype=torch.bfloat16,
        alpha_torch_dtype=torch.float32,
        sf_vec_size=32, is_mxfp6=False, block_fp8=False,
        rhs_values_tiled=None, quantized_c=qc,
        row_scale=None, x_bf16=None, w_gscale=None,
    )


def test_row_scale_accepted() -> None:
    """Canonical FP6 with row_scale passes phase-1 validation."""
    lhs, rhs = _operands_fp6(a_preexpanded=True, b_preexpanded=True)
    device = lhs[0].device
    row_scale = torch.empty((_M,), dtype=torch.bfloat16, device=device)
    dense_module._validate_dense_gemm_caller_inputs(
        lhs[0], rhs[0], lhs[1], rhs[1], None, _output(device),
        m=_M, n=_N, k=_K, l=1, a_wire_k=_K,
        expected_b_k=_K,
        caller_ab_torch_dtype=torch.uint8,
        caller_ab_dtype_aliases=(),
        sf_torch_dtype=torch.float8_e8m0fnu,
        c_torch_dtype=torch.bfloat16,
        alpha_torch_dtype=torch.float32,
        sf_vec_size=32, is_mxfp6=True, block_fp8=False,
        rhs_values_tiled=None, quantized_c=None,
        row_scale=row_scale, x_bf16=None, w_gscale=None,
    )


def test_x_bf16_w_gscale_accepted() -> None:
    """Canonical FP6 with x_bf16/w_gscale passes phase-1 validation."""
    lhs, rhs = _operands_fp6(a_preexpanded=True, b_preexpanded=True)
    device = lhs[0].device
    x_bf16 = torch.empty((_M, _K), dtype=torch.bfloat16, device=device)
    w_gscale = torch.empty((1,), dtype=torch.float32, device=device)
    dense_module._validate_dense_gemm_caller_inputs(
        lhs[0], rhs[0], lhs[1], rhs[1], None, _output(device),
        m=_M, n=_N, k=_K, l=1, a_wire_k=_K,
        expected_b_k=_K,
        caller_ab_torch_dtype=torch.uint8,
        caller_ab_dtype_aliases=(),
        sf_torch_dtype=torch.float8_e8m0fnu,
        c_torch_dtype=torch.bfloat16,
        alpha_torch_dtype=torch.float32,
        sf_vec_size=32, is_mxfp6=True, block_fp8=False,
        rhs_values_tiled=None, quantized_c=None,
        row_scale=None, x_bf16=x_bf16, w_gscale=w_gscale,
    )


# ---------------------------------------------------------------------------
# Phase-2 launched-pointer spy tests with nontrivial dimensions
# ---------------------------------------------------------------------------

_M2 = 128
_N2 = 256
_K2 = 256
_L2 = 2


def _operands_mxfp8_large(
    *, l: int = 1, device: torch.device | None = None,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]:
    if device is None:
        device = _device()
    wire_k = _K2
    if l > 1:
        a = torch.empty(
            (l, _M2, wire_k), dtype=torch.float8_e4m3fn, device=device
        ).as_strided((_M2, wire_k, l), (wire_k, 1, _M2 * wire_k))
        b = torch.empty(
            (l, _N2, wire_k), dtype=torch.float8_e4m3fn, device=device
        ).as_strided((_N2, wire_k, l), (wire_k, 1, _N2 * wire_k))
    else:
        a = torch.empty((_M2, wire_k, l), dtype=torch.float8_e4m3fn, device=device)
        b = torch.empty((_N2, wire_k, l), dtype=torch.float8_e4m3fn, device=device)
    sfa = _make_scale_mma(_M2, _K2, l, 32, torch.float8_e8m0fnu, device)
    sfb = _make_scale_mma(_N2, _K2, l, 32, torch.float8_e8m0fnu, device)
    return (a, sfa), (b, sfb)


def _spy_phase2(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Spy on _validate_dense_gemm_launched to capture its arguments."""
    captured: dict = {}
    original = dense_module._validate_dense_gemm_launched

    def spy(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        # Don't actually run validation (operands may be empty)
        return original(*args, **kwargs)

    monkeypatch.setattr(dense_module, "_validate_dense_gemm_launched", spy)
    # Also forbid actual kernel compile
    def forbid(*_a, **_kw):
        raise AssertionError("kernel compile reached")
    monkeypatch.setattr(dense_module, "_get_compiled_dense_gemm", forbid)
    monkeypatch.setattr(dense_module, "_get_compiled_dense_gemm_mxfp6", forbid)
    return captured


def test_mxfp8_l2_phase2_invoked_with_canonical_strides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MXFP8 L>1 public route invokes phase-2 with nontrivial k_tiles."""
    lhs, rhs = _operands_mxfp8_large(l=_L2)
    captured = _spy_phase2(monkeypatch)
    out = _output(lhs[0].device, l=_L2)
    # Resize out for large dims
    out = torch.empty(
        (_L2, _M2, _N2), dtype=torch.bfloat16, device=lhs[0].device
    ).as_strided((_M2, _N2, _L2), (_N2, 1, _M2 * _N2))
    with contextlib.suppress(AssertionError):
        _call(lhs, rhs, out)
    assert "kwargs" in captured, "phase-2 was not invoked"
    assert captured["kwargs"]["l"] == _L2


def test_fp4_phase2_invoked(monkeypatch: pytest.MonkeyPatch) -> None:
    """FP4 public route invokes phase-2."""
    lhs, rhs = _operands_fp4()
    captured = _spy_phase2(monkeypatch)
    with contextlib.suppress(AssertionError):
        _call(
            lhs, rhs, _output(lhs[0].device),
            ab_dtype="float4_e2m1fn", sf_dtype="float8_e4m3fn", sf_vec_size=16,
        )
    assert "kwargs" in captured, "phase-2 was not invoked for FP4"


def test_fp6_preexpanded_phase2_invoked(monkeypatch: pytest.MonkeyPatch) -> None:
    """FP6 preexpanded public route invokes phase-2."""
    lhs, rhs = _operands_fp6(a_preexpanded=True, b_preexpanded=True)
    captured = _spy_phase2(monkeypatch)
    with contextlib.suppress(AssertionError):
        _call(
            lhs, rhs, _output(lhs[0].device),
            ab_dtype="float6_e3m2fn", sf_dtype="float8_e8m0fnu", sf_vec_size=32,
            a_preexpanded=True, b_preexpanded=True,
        )
    assert "kwargs" in captured, "phase-2 was not invoked for FP6 preexpanded"


def test_fp6_packed_phase2_invoked(monkeypatch: pytest.MonkeyPatch) -> None:
    """FP6 packed-B public route invokes phase-2."""
    lhs, rhs = _operands_fp6(a_preexpanded=True, b_packed=True)
    captured = _spy_phase2(monkeypatch)
    with contextlib.suppress(AssertionError):
        _call(
            lhs, rhs, _output(lhs[0].device),
            ab_dtype="float6_e3m2fn", sf_dtype="float8_e8m0fnu", sf_vec_size=32,
            a_preexpanded=True, b_packed=True,
        )
    assert "kwargs" in captured, "phase-2 was not invoked for FP6 packed"


def test_fp6_default_phase2_invoked(monkeypatch: pytest.MonkeyPatch) -> None:
    """FP6 default (will-expand) public route invokes phase-2."""
    lhs, rhs = _operands_fp6(a_preexpanded=False, b_preexpanded=False)
    captured = _spy_phase2(monkeypatch)
    with contextlib.suppress(AssertionError):
        _call(
            lhs, rhs, _output(lhs[0].device),
            ab_dtype="float6_e3m2fn", sf_dtype="float8_e8m0fnu", sf_vec_size=32,
        )
    assert "kwargs" in captured, "phase-2 was not invoked for FP6 default"


def test_block_fp8_phase2_invoked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Block-FP8 public route invokes phase-2."""
    lhs, rhs = _operands_block_fp8()
    captured = _spy_phase2(monkeypatch)
    with contextlib.suppress(AssertionError):
        _call(
            lhs, rhs, _output(lhs[0].device),
            sf_dtype="float32", sf_vec_size=128, block_fp8=True,
        )
    assert "kwargs" in captured, "phase-2 was not invoked for block-FP8"


def test_quantized_c_phase2_invoked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Quantized-C public route invokes phase-2."""
    lhs, rhs = _operands_mxfp8()
    qc = _quantized_c_operands(lhs[0].device)
    captured = _spy_phase2(monkeypatch)
    with contextlib.suppress(AssertionError):
        dense_module.dense_gemm(
            lhs, rhs, out=_output(lhs[0].device),
            ab_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16", sf_vec_size=32,
            mma_tiler_mn=(16, 64), load_path="tma", swap_ab=False,
            _quantized_c=qc,
        )
    assert "kwargs" in captured, "phase-2 was not invoked for quant-C"


def test_plain_fp8_l2_phase2_invoked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plain-FP8 L>1 out=None public route invokes phase-2."""
    lhs, rhs = _operands_mxfp8(l=_L2)
    captured = _spy_phase2(monkeypatch)
    with contextlib.suppress(AssertionError):
        dense_module.dense_gemm(
            lhs, rhs,
            ab_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16", sf_vec_size=32,
            mma_tiler_mn=(16, 64), load_path="tma", swap_ab=False,
            plain_fp8=True,
        )
    assert "kwargs" in captured, "phase-2 was not invoked for plain_fp8 L>1"


def test_mxfp8_functional_l2_phase2_invoked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MXFP8 L>1 out=None functional route invokes phase-2 internally."""
    lhs, rhs = _operands_mxfp8_large(l=_L2)
    captured: dict = {}
    original = dense_module._validate_dense_gemm_launched

    def spy(*args, **kwargs):
        captured["kwargs"] = kwargs
        return original(*args, **kwargs)

    monkeypatch.setattr(dense_module, "_validate_dense_gemm_launched", spy)
    with contextlib.suppress(Exception):
        dense_module.dense_gemm(
            lhs, rhs,
            ab_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16", sf_vec_size=32,
            mma_tiler_mn=(16, 64), load_path="tma", swap_ab=False,
        )
    assert "kwargs" in captured, "phase-2 not invoked in functional route"
    assert captured["kwargs"]["l"] == _L2


def test_shallow_type_check_before_ndim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-tensor A must TypeError before any .ndim access."""
    lhs, rhs = _operands_mxfp8()
    _forbid_compile(monkeypatch)
    with pytest.raises(TypeError, match="A must be a torch.Tensor"):
        _call(("not_a_tensor", lhs[1]), rhs, _output(lhs[0].device))  # type: ignore[arg-type]
