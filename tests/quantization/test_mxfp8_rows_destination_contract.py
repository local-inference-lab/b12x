"""Adversarial destination-contract tests for the raw MXFP8 row quantizer."""
from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager

import pytest
import torch

import b12x._lib.quant.mxfp8_rows as rows_impl
import b12x.gemm._shared.block_fp8 as block_fp8
from b12x.gemm._shared.wo_mxfp8 import empty_mxfp8_rows_for_dense_gemm

from ..conftest import require_b12x

_M = 4
_K = 128
_G = _K // 32
_M_TILES = 1
_K_TILES = 1
_VALUES_BYTES = _M * _K
_SCALE_ROWS_BYTES = _M * _G
_SCALE_MMA_BYTES = _M_TILES * _K_TILES * 512


def _scale_mma_from_u8(storage: torch.Tensor) -> torch.Tensor:
    physical = storage.narrow(0, 0, _SCALE_MMA_BYTES).view(
        1, _M_TILES, _K_TILES, 32, 4, 4
    )
    return physical.view(torch.float8_e8m0fnu).permute(3, 4, 1, 5, 2, 0)


def _valid_case() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = require_b12x()
    source = torch.randn((_M, _K), dtype=torch.bfloat16, device=device)
    out = empty_mxfp8_rows_for_dense_gemm(_M, _K, device=device)
    return source, out.values, out.scale_rows, out.scale_mma


def _assert_rejected_before_compile(
    monkeypatch: pytest.MonkeyPatch,
    tensors: tuple[torch.Tensor, object, object, object],
    *,
    error: type[BaseException],
    match: str,
) -> None:
    compiled = False

    def forbid_compile(*_args, **_kwargs):
        nonlocal compiled
        compiled = True
        raise AssertionError("destination validation reached kernel resolution")

    monkeypatch.setattr(rows_impl, "_get_compiled_mxfp8_rows_quant", forbid_compile)
    source, values, scale_rows, scale_mma = tensors
    with pytest.raises(error, match=match):
        rows_impl.quantize_mxfp8_rows_cute(
            source,
            values,  # type: ignore[arg-type]
            scale_rows,  # type: ignore[arg-type]
            scale_mma,  # type: ignore[arg-type]
        )
    assert not compiled


def test_canonical_destinations_pass_exact_contract() -> None:
    source, values, scale_rows, scale_mma = _valid_case()
    rows_impl.validate_mxfp8_rows_destinations(
        source, values, scale_rows, scale_mma
    )


def _misaligned_source(device: torch.device) -> torch.Tensor:
    storage = torch.empty(_M * _K * 2 + 2, dtype=torch.uint8, device=device)
    return storage.narrow(0, 2, _M * _K * 2).view(torch.bfloat16).view(_M, _K)


def test_misaligned_source_is_rejected_before_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, values, scale_rows, scale_mma = _valid_case()
    _assert_rejected_before_compile(
        monkeypatch,
        (_misaligned_source(source.device), values, scale_rows, scale_mma),
        error=ValueError,
        match="source base pointer must be 16-byte aligned",
    )


def test_public_alloc_quantizer_is_fullgraph_traceable() -> None:
    device = require_b12x()
    source = torch.randn((_M, _K), dtype=torch.bfloat16, device=device)

    def quantize_values(x: torch.Tensor) -> torch.Tensor:
        return block_fp8.quantize_block_fp8_linear_input_mxfp8(x).values

    compiled = torch.compile(quantize_values, backend="eager", fullgraph=True)
    result = compiled(source)
    assert result.shape == source.shape
    assert result.dtype == torch.float8_e4m3fn



@pytest.mark.parametrize(
    ("tokens", "in_features"),
    [(_M + 1, _K), (_M, _K + 32)],
)
def test_direct_alloc_op_rejects_dimensions_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
    tokens: int,
    in_features: int,
) -> None:
    source, _, _, _ = _valid_case()

    def forbid_allocation(*_args, **_kwargs):
        raise AssertionError("dimension validation reached output allocation")

    monkeypatch.setattr(block_fp8, "empty_mxfp8_rows_bases", forbid_allocation)
    with pytest.raises(ValueError, match="dimensions must match source shape"):
        torch.ops.b12x.quantize_block_fp8_linear_input_mxfp8_alloc(
            source, tokens, in_features
        )


def test_direct_alloc_op_rejects_misalignment_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = require_b12x()
    source = _misaligned_source(device)

    def forbid_allocation(*_args, **_kwargs):
        raise AssertionError("source validation reached output allocation")

    monkeypatch.setattr(block_fp8, "empty_mxfp8_rows_bases", forbid_allocation)
    with pytest.raises(
        ValueError, match="source base pointer must be 16-byte aligned"
    ):
        torch.ops.b12x.quantize_block_fp8_linear_input_mxfp8_alloc(
            source, _M, _K
        )

def test_aligned_disjoint_arena_slices_are_accepted() -> None:
    source, _, _, _ = _valid_case()
    scale_rows_offset = _VALUES_BYTES
    scale_mma_offset = scale_rows_offset + _SCALE_ROWS_BYTES
    arena = torch.empty(
        scale_mma_offset + _SCALE_MMA_BYTES,
        dtype=torch.uint8,
        device=source.device,
    )
    values = (
        arena.narrow(0, 0, _VALUES_BYTES)
        .view(torch.float8_e4m3fn)
        .view(_M, _K)
    )
    scale_rows = (
        arena.narrow(0, scale_rows_offset, _SCALE_ROWS_BYTES)
        .view(torch.float8_e8m0fnu)
        .view(1, _M, _G)
    )
    scale_mma = _scale_mma_from_u8(
        arena.narrow(0, scale_mma_offset, _SCALE_MMA_BYTES)
    )

    assert scale_rows.storage_offset() != 0
    assert scale_mma.storage_offset() != 0
    rows_impl.validate_mxfp8_rows_destinations(
        source, values, scale_rows, scale_mma
    )


@pytest.mark.parametrize(
    ("destination", "replacement", "match"),
    [
        (
            "values",
            lambda device: torch.empty(
                (_M, _K), dtype=torch.uint8, device=device
            ),
            "values must have dtype",
        ),
        (
            "scale_rows",
            lambda device: torch.empty(
                (1, _M, _G), dtype=torch.uint8, device=device
            ),
            "scale_rows must have dtype",
        ),
        (
            "scale_mma",
            lambda device: torch.empty(
                (32, 4, 1, 4, 1, 1), dtype=torch.uint8, device=device
            ),
            "scale_mma must have dtype",
        ),
    ],
)
def test_wrong_destination_dtype_is_rejected_before_compile(
    monkeypatch: pytest.MonkeyPatch,
    destination: str,
    replacement: Callable[[torch.device], torch.Tensor],
    match: str,
) -> None:
    source, values, scale_rows, scale_mma = _valid_case()
    tensors = {
        "values": values,
        "scale_rows": scale_rows,
        "scale_mma": scale_mma,
    }
    tensors[destination] = replacement(source.device)
    _assert_rejected_before_compile(
        monkeypatch,
        (source, tensors["values"], tensors["scale_rows"], tensors["scale_mma"]),
        error=TypeError,
        match=match,
    )


@pytest.mark.parametrize(
    ("destination", "replacement", "match"),
    [
        (
            "values",
            lambda device: torch.empty(
                (_M, _K + 32), dtype=torch.float8_e4m3fn, device=device
            ),
            "values must have shape",
        ),
        (
            "scale_rows",
            lambda device: torch.empty(
                (1, _M, _G + 1), dtype=torch.float8_e8m0fnu, device=device
            ),
            "scale_rows must have shape",
        ),
        (
            "scale_mma",
            lambda device: torch.empty(
                (32, 4, 1, 4, 2, 1),
                dtype=torch.float8_e8m0fnu,
                device=device,
            ),
            "scale_mma must have shape",
        ),
    ],
)
def test_wrong_destination_shape_is_rejected_before_compile(
    monkeypatch: pytest.MonkeyPatch,
    destination: str,
    replacement: Callable[[torch.device], torch.Tensor],
    match: str,
) -> None:
    source, values, scale_rows, scale_mma = _valid_case()
    tensors = {
        "values": values,
        "scale_rows": scale_rows,
        "scale_mma": scale_mma,
    }
    tensors[destination] = replacement(source.device)
    _assert_rejected_before_compile(
        monkeypatch,
        (source, tensors["values"], tensors["scale_rows"], tensors["scale_mma"]),
        error=ValueError,
        match=match,
    )


@pytest.mark.parametrize("destination", ["values", "scale_rows", "scale_mma"])
def test_wrong_destination_device_is_rejected_before_compile(
    monkeypatch: pytest.MonkeyPatch,
    destination: str,
) -> None:
    source, values, scale_rows, scale_mma = _valid_case()
    replacements = {
        "values": torch.empty((_M, _K), dtype=torch.float8_e4m3fn),
        "scale_rows": torch.empty(
            (1, _M, _G), dtype=torch.float8_e8m0fnu
        ),
        "scale_mma": _scale_mma_from_u8(
            torch.empty(_SCALE_MMA_BYTES, dtype=torch.uint8)
        ),
    }
    tensors = {
        "values": values,
        "scale_rows": scale_rows,
        "scale_mma": scale_mma,
    }
    tensors[destination] = replacements[destination]
    _assert_rejected_before_compile(
        monkeypatch,
        (source, tensors["values"], tensors["scale_rows"], tensors["scale_mma"]),
        error=ValueError,
        match="must be on the source/launch device",
    )


@pytest.mark.parametrize(
    ("destination", "replacement", "match"),
    [
        (
            "values",
            lambda device: torch.empty(
                (_K, _M), dtype=torch.float8_e4m3fn, device=device
            ).t(),
            "values has strides",
        ),
        (
            "scale_rows",
            lambda device: torch.empty(
                (1, _G, _M), dtype=torch.float8_e8m0fnu, device=device
            ).transpose(1, 2),
            "scale_rows has strides",
        ),
        (
            "scale_mma",
            lambda device: torch.empty(
                (32, 4, 1, 4, 1, 1),
                dtype=torch.float8_e8m0fnu,
                device=device,
            ),
            "scale_mma has noncanonical strides",
        ),
    ],
)
def test_noncanonical_destination_layout_is_rejected_before_compile(
    monkeypatch: pytest.MonkeyPatch,
    destination: str,
    replacement: Callable[[torch.device], torch.Tensor],
    match: str,
) -> None:
    source, values, scale_rows, scale_mma = _valid_case()
    tensors = {
        "values": values,
        "scale_rows": scale_rows,
        "scale_mma": scale_mma,
    }
    tensors[destination] = replacement(source.device)
    _assert_rejected_before_compile(
        monkeypatch,
        (source, tensors["values"], tensors["scale_rows"], tensors["scale_mma"]),
        error=ValueError,
        match=match,
    )


def _misaligned_values(device: torch.device) -> torch.Tensor:
    storage = torch.empty(_VALUES_BYTES + 1, dtype=torch.uint8, device=device)
    return (
        storage.narrow(0, 1, _VALUES_BYTES)
        .view(torch.float8_e4m3fn)
        .view(_M, _K)
    )


def _misaligned_scale_rows(device: torch.device) -> torch.Tensor:
    storage = torch.empty(_SCALE_ROWS_BYTES + 1, dtype=torch.uint8, device=device)
    return (
        storage.narrow(0, 1, _SCALE_ROWS_BYTES)
        .view(torch.float8_e8m0fnu)
        .view(1, _M, _G)
    )


def _misaligned_scale_mma(device: torch.device) -> torch.Tensor:
    storage = torch.empty(_SCALE_MMA_BYTES + 1, dtype=torch.uint8, device=device)
    return _scale_mma_from_u8(storage.narrow(0, 1, _SCALE_MMA_BYTES))


@pytest.mark.parametrize(
    ("destination", "replacement"),
    [
        ("values", _misaligned_values),
        ("scale_rows", _misaligned_scale_rows),
        ("scale_mma", _misaligned_scale_mma),
    ],
)
def test_misaligned_destination_is_rejected_before_compile(
    monkeypatch: pytest.MonkeyPatch,
    destination: str,
    replacement: Callable[[torch.device], torch.Tensor],
) -> None:
    source, values, scale_rows, scale_mma = _valid_case()
    tensors = {
        "values": values,
        "scale_rows": scale_rows,
        "scale_mma": scale_mma,
    }
    tensors[destination] = replacement(source.device)
    _assert_rejected_before_compile(
        monkeypatch,
        (source, tensors["values"], tensors["scale_rows"], tensors["scale_mma"]),
        error=ValueError,
        match="base pointer must be 16-byte aligned",
    )


def test_short_backing_span_is_rejected_before_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, values, scale_rows, scale_mma = _valid_case()
    real_backing_span = rows_impl._backing_span

    def short_values_span(tensor: torch.Tensor) -> int:
        if tensor is values:
            return _VALUES_BYTES - 1
        return real_backing_span(tensor)

    monkeypatch.setattr(rows_impl, "_backing_span", short_values_span)
    _assert_rejected_before_compile(
        monkeypatch,
        (source, values, scale_rows, scale_mma),
        error=ValueError,
        match="values backing storage is",
    )


def _alias_case(
    pair: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    source, values, scale_rows, scale_mma = _valid_case()
    if pair == "source-values":
        values = (
            source.view(torch.uint8)
            .reshape(-1)
            .narrow(0, 0, _VALUES_BYTES)
            .view(torch.float8_e4m3fn)
            .view(_M, _K)
        )
    elif pair == "source-scale_rows":
        scale_rows = (
            source.view(torch.uint8)
            .reshape(-1)
            .narrow(0, 0, _SCALE_ROWS_BYTES)
            .view(torch.float8_e8m0fnu)
            .view(1, _M, _G)
        )
    elif pair == "source-scale_mma":
        scale_mma = _scale_mma_from_u8(source.view(torch.uint8).reshape(-1))
    else:
        arena = torch.empty(
            _SCALE_MMA_BYTES, dtype=torch.uint8, device=source.device
        )
        if pair in {"values-scale_rows", "values-scale_mma"}:
            values = (
                arena.narrow(0, 0, _VALUES_BYTES)
                .view(torch.float8_e4m3fn)
                .view(_M, _K)
            )
        if pair in {"values-scale_rows", "scale_rows-scale_mma"}:
            scale_rows = (
                arena.narrow(0, 0, _SCALE_ROWS_BYTES)
                .view(torch.float8_e8m0fnu)
                .view(1, _M, _G)
            )
        if pair in {"values-scale_mma", "scale_rows-scale_mma"}:
            scale_mma = _scale_mma_from_u8(arena)
    return source, values, scale_rows, scale_mma


@pytest.mark.parametrize(
    "pair",
    [
        "source-values",
        "source-scale_rows",
        "source-scale_mma",
        "values-scale_rows",
        "values-scale_mma",
        "scale_rows-scale_mma",
    ],
)
def test_every_overlapping_storage_pair_is_rejected_before_compile(
    monkeypatch: pytest.MonkeyPatch,
    pair: str,
) -> None:
    _assert_rejected_before_compile(
        monkeypatch,
        _alias_case(pair),
        error=ValueError,
        match="overlapping storage",
    )


def test_non_tensor_destination_is_rejected_before_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _, scale_rows, scale_mma = _valid_case()
    _assert_rejected_before_compile(
        monkeypatch,
        (source, object(), scale_rows, scale_mma),
        error=TypeError,
        match="values must be a torch.Tensor",
    )


def test_partial_row_tile_reports_true_maximum_scale_offset() -> None:
    geometry = rows_impl._compute_mxfp8_rows_geometry(33, _K)
    assert geometry["scale_mma_max_offset"] == 499
    assert geometry["scale_mma_max_offset"] < geometry["scale_mma_bytes"]


@pytest.mark.parametrize(
    "field",
    [
        "source element index",
        "values word index",
        "scale_rows element index",
        "task count",
        "scale_mma byte offset",
    ],
)
def test_every_int32_index_family_is_guarded(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        rows_impl._compute_mxfp8_rows_geometry(1 << 30, _K)


def test_launch_resolves_stream_inside_source_device_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, values, scale_rows, scale_mma = _valid_case()
    raw_calls: list[tuple[object, ...]] = []

    def fake_compile(*_args, **_kwargs):
        def fake_raw(*args):
            raw_calls.append(args)

        return fake_raw

    rows_impl._get_compiled_mxfp8_rows_quant.cache_clear()
    monkeypatch.setattr(rows_impl, "b12x_compile", fake_compile)

    entered: list[torch.device] = []
    active: list[torch.device] = []

    @contextmanager
    def record_device(device):
        resolved = torch.device(device)
        entered.append(resolved)
        active.append(resolved)
        try:
            yield
        finally:
            active.pop()

    def checked_current_stream():
        assert active == [source.device]
        return object()

    monkeypatch.setattr(torch.cuda, "device", record_device)
    monkeypatch.setattr(rows_impl, "current_cuda_stream", checked_current_stream)
    try:
        rows_impl.quantize_mxfp8_rows_cute(
            source, values, scale_rows, scale_mma
        )
    finally:
        rows_impl._get_compiled_mxfp8_rows_quant.cache_clear()

    assert entered == [source.device]
    assert len(raw_calls) == 1


def test_compile_cache_separates_device_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compile_devices: list[torch.device] = []
    compile_specs = []

    def fake_compile(*_args, **kwargs):
        compile_devices.append(requested_device)
        compile_specs.append(kwargs["compile_spec"])
        return lambda *_launch_args: None

    monkeypatch.setattr(rows_impl, "b12x_compile", fake_compile)
    monkeypatch.setattr(rows_impl, "current_cuda_stream", object)
    rows_impl._get_compiled_mxfp8_rows_quant.cache_clear()
    try:
        for requested_device in (
            torch.device("cuda:0"),
            torch.device("cuda:1"),
            torch.device("cuda:0"),
        ):
            rows_impl._get_compiled_mxfp8_rows_quant(
                _K,
                torch.bfloat16,
                0,
                rows_impl._THREADS,
                "linear",
                requested_device,
            )
    finally:
        rows_impl._get_compiled_mxfp8_rows_quant.cache_clear()

    assert compile_devices == [torch.device("cuda:0"), torch.device("cuda:1")]
    assert compile_specs[0].json_key != compile_specs[1].json_key
