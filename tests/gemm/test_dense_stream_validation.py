"""Focused regression tests for issue #153: dense-GEMM stream contract.

Unit tests exercise the validation and lifetime-recording boundaries without
launching kernels. Wrapper-ordering tests use CPU tensors with CUDA guards
patched out; explicit launch-boundary and allocator-origin tests are CUDA-gated.
A real SM120/SM121 host is required for end-to-end dense-kernel coverage.

Run (from worktree root):

    pytest tests/gemm/test_dense_stream_validation.py -v
"""

from __future__ import annotations

import pytest
import torch

import b12x._lib.dense_gemm  # noqa: F401  # register custom ops
from b12x._lib.utils import (
    record_tensors_on_current_stream,
    validate_stream_is_current,
    validate_stream_int_is_current,
)


# ---------------------------------------------------------------------------
# Unit tests for the validation primitives themselves.
# ---------------------------------------------------------------------------


class _FakeStream:
    """Minimal stand-in for a torch.cuda.Stream-like object."""

    def __init__(self, handle: int, device_index: int = 0) -> None:
        self._handle = handle
        self._device_index = device_index

    @property
    def cuda_stream(self) -> int:
        return self._handle

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self._device_index)



def test_record_tensors_on_device_current_stream(monkeypatch) -> None:
    """Every non-None raw-launch operand is recorded on the device stream."""
    device = torch.device("cuda", 1)
    current = _FakeStream(handle=42, device_index=1)
    recorded: list[object] = []

    class _FakeTensor:
        def record_stream(self, stream) -> None:
            recorded.append(stream)

    tensors = [_FakeTensor(), None, _FakeTensor()]
    monkeypatch.setattr(torch.cuda, "current_stream", lambda dev: current)
    launch_stream = record_tensors_on_current_stream(tensors, device)

    assert int(launch_stream) == 42
    assert recorded == [current, current]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_record_tensors_defers_origin_stream_reuse() -> None:
    """A foreign-origin allocation stays live until current-stream work retires."""
    if not hasattr(torch.cuda, "_sleep"):
        pytest.skip("torch.cuda._sleep is unavailable")
    device = torch.device("cuda", torch.cuda.current_device())
    origin = torch.cuda.Stream(device=device)
    consumer = torch.cuda.Stream(device=device)
    elements = 8 * 1024 * 1024

    with torch.cuda.stream(origin):
        source = torch.ones(elements, dtype=torch.float32, device=device)
    consumer.wait_stream(origin)
    source_ptr = source.data_ptr()

    with torch.cuda.stream(consumer):
        record_tensors_on_current_stream([source], device)
        torch.cuda._sleep(50_000_000)
        observed = source[:1024].clone()
    del source

    with torch.cuda.stream(origin):
        churn = torch.empty(elements, dtype=torch.float32, device=device)
        churn.zero_()

    assert churn.data_ptr() != source_ptr
    consumer.synchronize()
    assert torch.equal(observed, torch.ones_like(observed))


def test_dense_gemm_records_every_caller_tensor_before_other_work(
    monkeypatch,
) -> None:
    """Packed inputs and caller output are retained before expansion/reduction."""
    import b12x._lib.dense_gemm as dg

    a = torch.empty((1, 3, 1), dtype=torch.uint8)
    sfa = torch.empty(1)
    b = torch.empty((1, 3, 1), dtype=torch.uint8)
    sfb = torch.empty(1)
    out = torch.empty((1, 1, 1))
    alpha = torch.empty(1)
    tiled = torch.empty(1)
    x_bf16 = torch.empty(1)
    w_gscale = torch.empty(1)
    row_scale = torch.empty(1)
    quantized = (torch.empty(1), torch.empty(1), torch.empty(1))
    recorded: list[torch.Tensor] = []

    def _record(tensors, _device):
        recorded.extend(tensor for tensor in tensors if tensor is not None)

    monkeypatch.setattr(dg, "record_tensors_on_current_stream", _record)
    with pytest.raises(ValueError, match="load_path"):
        dg.dense_gemm(
            (a, sfa),
            (b, sfb),
            out=out,
            ab_dtype="float6_e3m2fn",
            sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16",
            sf_vec_size=32,
            alpha=alpha,
            load_path="invalid",
            rhs_values_tiled=tiled,
            _quantized_c=quantized,
            x_bf16=x_bf16,
            w_gscale=w_gscale,
            row_scale=row_scale,
        )

    expected = [
        a,
        sfa,
        b,
        sfb,
        out,
        alpha,
        tiled,
        x_bf16,
        w_gscale,
        row_scale,
        *quantized,
    ]
    assert len(recorded) == len(expected)
    assert all(got is want for got, want in zip(recorded, expected, strict=True))


@pytest.mark.parametrize("kind", ["block", "mxfp8", "tensor_fp8"])
def test_linear_composites_record_source_and_bias_before_normalization(
    monkeypatch,
    kind,
) -> None:
    """Composite quantize/copy/bias stages retain caller storage before use."""
    from b12x.gemm._shared.wo_mxfp8 import MXFP8Rows

    rows = MXFP8Rows(
        values=torch.empty(1),
        scale_rows=torch.empty(1),
        scale_mma=torch.empty(1),
    )
    source = torch.empty((1, 128))
    bias = torch.empty(64)
    recorded: list[object] = []

    def _record(tensors, _device):
        recorded.extend(tensor for tensor in tensors if tensor is not None)

    def _normalize(_source):
        assert any(tensor is source for tensor in recorded)
        assert any(tensor is bias for tensor in recorded)
        raise RuntimeError("normalization reached after lifetime recording")

    if kind == "block":
        import b12x.gemm._shared.block_fp8 as module

        weight = module.BlockFP8LinearWeight(
            weight=rows,
            in_features=128,
            out_features=64,
            block_size=(128, 128),
        )
        call = module.block_fp8_linear_mxfp8
    elif kind == "mxfp8":
        import b12x.gemm.mxfp8_linear._kernel as module

        weight = module.MXFP8LinearWeight(
            weight=rows,
            in_features=128,
            padded_in_features=128,
            out_features=64,
        )
        call = module.mxfp8_linear
    else:
        import b12x.gemm.tensor_fp8_linear._kernel as module

        weight = module.TensorFP8LinearWeight(
            values=torch.empty(1),
            scale_mma=torch.empty(1),
            output_scale=torch.empty(1),
            in_features=128,
            padded_in_features=128,
            out_features=64,
        )
        call = module.tensor_fp8_linear

    monkeypatch.setattr(module, "_check_gpu_tensor", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "record_tensors_on_current_stream", _record)
    monkeypatch.setattr(module, "_source_2d", _normalize)

    with pytest.raises(RuntimeError, match="after lifetime recording"):
        call(source, weight, bias=bias)


def test_validate_stream_is_current_accepts_none(monkeypatch) -> None:
    """``stream=None`` always passes — it means 'use current stream'."""
    validate_stream_is_current(None, torch.device("cuda", 0))


def test_validate_stream_int_is_current_accepts_none() -> None:
    """``stream_int=None`` always passes — fallback to current stream."""
    validate_stream_int_is_current(None, torch.device("cuda", 0))


def test_validate_stream_is_current_rejects_non_current_stream(monkeypatch) -> None:
    """A valid stream that is *not* the current stream must fail."""
    device = torch.device("cuda", 0)
    fake_current = _FakeStream(handle=42, device_index=0)
    fake_other = _FakeStream(handle=99, device_index=0)

    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda dev: fake_current
    )
    with pytest.raises(ValueError, match="must be the current stream"):
        validate_stream_is_current(fake_other, device)


def test_validate_stream_int_is_current_rejects_non_current_int(monkeypatch) -> None:
    """A raw handle that does not match the current stream handle must fail."""
    device = torch.device("cuda", 0)
    fake_current = _FakeStream(handle=42, device_index=0)

    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda dev: fake_current
    )
    with pytest.raises(ValueError, match="must be the current stream"):
        validate_stream_int_is_current(99, device)


def test_validate_stream_int_is_current_accepts_current_int(monkeypatch) -> None:
    """A raw handle equal to the current stream handle passes."""
    device = torch.device("cuda", 0)
    fake_current = _FakeStream(handle=42, device_index=0)

    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda dev: fake_current
    )
    validate_stream_int_is_current(42, device)


def test_validate_stream_int_is_current_accepts_default_zero_when_current(
    monkeypatch,
) -> None:
    """Handle 0 (default stream) passes when the default stream is current."""
    device = torch.device("cuda", 0)
    fake_default = _FakeStream(handle=0, device_index=0)

    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda dev: fake_default
    )
    monkeypatch.setattr(
        torch.cuda, "default_stream", lambda dev: fake_default
    )
    validate_stream_int_is_current(0, device)


def test_validate_stream_int_is_current_rejects_default_zero_when_not_current(
    monkeypatch,
) -> None:
    """Handle 0 (default stream) fails when the current stream is not default."""
    device = torch.device("cuda", 0)
    fake_current = _FakeStream(handle=42, device_index=0)
    fake_default = _FakeStream(handle=0, device_index=0)

    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda dev: fake_current
    )
    monkeypatch.setattr(
        torch.cuda, "default_stream", lambda dev: fake_default
    )
    with pytest.raises(ValueError, match="must be the current stream"):
        validate_stream_int_is_current(0, device)

# ---------------------------------------------------------------------------
# Integration tests: public dense_gemm entry rejects before any work.
#
# These monkeypatch the heavy machinery so we can verify the *order* of
# checks: stream validation runs *before* any allocation, compile, or launch.
# ---------------------------------------------------------------------------


def _patch_dense_gemm_internals(monkeypatch):
    """Patch out GPU-dependent calls so we can test validation ordering."""
    import b12x._lib.dense_gemm as dg

    # Guard: these must never be called if validation fails first.
    called: dict[str, bool] = {}

    def _no_call(name):
        def _fail(*a, **kw):
            called[name] = True
            raise AssertionError(
                f"{name} was called before stream validation rejected the call"
            )
        return _fail

    monkeypatch.setattr(dg, "get_num_sm", _no_call("get_num_sm"))
    monkeypatch.setattr(dg, "_select_default_dense_gemm_plan", _no_call("_select_default_dense_gemm_plan"))
    monkeypatch.setattr(dg, "_get_compiled_dense_gemm", _no_call("_get_compiled_dense_gemm"))
    monkeypatch.setattr(dg, "_get_compiled_dense_gemm_mxfp6", _no_call("_get_compiled_dense_gemm_mxfp6"))
    monkeypatch.setattr(dg, "_dense_gemm_launch_flat", _no_call("_dense_gemm_launch_flat"))
    return called


def test_dense_gemm_rejects_non_current_stream_before_any_work(
    monkeypatch,
) -> None:
    """dense_gemm must reject a non-current stream before any tensor work."""
    called = _patch_dense_gemm_internals(monkeypatch)

    fake_current = _FakeStream(handle=42, device_index=0)
    fake_other = _FakeStream(handle=99, device_index=0)

    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda dev: fake_current
    )

    # Minimal operands — they won't be touched because validation fires first.
    a = torch.empty(1, 1, 1, dtype=torch.float8_e4m3fn)
    sfa = torch.empty(1, dtype=torch.float8_e8m0fnu)
    b = torch.empty(1, 1, 1, dtype=torch.float8_e4m3fn)
    sfb = torch.empty(1, dtype=torch.float8_e8m0fnu)

    from b12x._lib.dense_gemm import dense_gemm

    with pytest.raises(ValueError, match="must be the current stream"):
        dense_gemm(
            (a, sfa),
            (b, sfb),
            ab_dtype="float8_e4m3fn",
            sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16",
            sf_vec_size=32,
            stream=fake_other,
        )

    # Nothing downstream should have been called.
    assert not called, f"work ran before validation: {called}"


def test_dense_gemm_accepts_none_stream(monkeypatch) -> None:
    """dense_gemm with stream=None passes validation (may fail later on GPU)."""
    # Don't patch internals — just verify validation doesn't raise.
    # The call will fail at GPU/shape checks, but NOT at stream validation.
    a = torch.empty(1, 1, 1, dtype=torch.float8_e4m3fn)
    sfa = torch.empty(1, dtype=torch.float8_e8m0fnu)
    b = torch.empty(1, 1, 1, dtype=torch.float8_e4m3fn)
    sfb = torch.empty(1, dtype=torch.float8_e8m0fnu)

    from b12x._lib.dense_gemm import dense_gemm

    # We expect *some* error downstream (no GPU, bad shapes), but it must NOT
    # be a stream-validation ValueError.
    try:
        dense_gemm(
            (a, sfa),
            (b, sfb),
            ab_dtype="float8_e4m3fn",
            sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16",
            sf_vec_size=32,
            stream=None,
        )
    except ValueError as exc:
        assert "must be the current stream" not in str(exc), (
            f"stream=None should not fail stream validation: {exc}"
        )
    except Exception:
        pass  # Any non-stream error is fine — we only test validation.


def test_dense_gemm_fused_quant_a_rejects_non_current_stream(
    monkeypatch,
) -> None:
    """dense_gemm_fused_quant_a must reject a non-current stream."""
    fake_current = _FakeStream(handle=42, device_index=0)
    fake_other = _FakeStream(handle=99, device_index=0)

    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda dev: fake_current
    )

    from b12x._lib.dense_gemm import dense_gemm_fused_quant_a

    source = torch.empty(1, 128, dtype=torch.bfloat16)
    b = torch.empty(1, 128, 1, dtype=torch.float8_e4m3fn)
    sfb = torch.empty(1, dtype=torch.float8_e8m0fnu)

    with pytest.raises(ValueError, match="must be the current stream"):
        dense_gemm_fused_quant_a(
            source,
            b,
            sfb,
            stream=fake_other,
        )


def test_dense_gemm_fused_quant_a_grouped_rejects_non_current_stream(
    monkeypatch,
) -> None:
    """dense_gemm_fused_quant_a_grouped must reject a non-current stream."""
    fake_current = _FakeStream(handle=42, device_index=0)
    fake_other = _FakeStream(handle=99, device_index=0)

    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda dev: fake_current
    )

    from b12x._lib.dense_gemm import dense_gemm_fused_quant_a_grouped

    source = torch.empty(1, 1, 128, dtype=torch.bfloat16)
    b = torch.empty(1, 128, 1, dtype=torch.float8_e4m3fn)
    sfb = torch.empty(1, dtype=torch.float8_e8m0fnu)

    with pytest.raises(ValueError, match="must be the current stream"):
        dense_gemm_fused_quant_a_grouped(
            source,
            b,
            sfb,
            groups=1,
            stream=fake_other,
        )


# ---------------------------------------------------------------------------
# Custom-op boundary: stream_int validation inside the ops.
# ---------------------------------------------------------------------------


def test_dense_gemm_launch_op_rejects_non_current_stream_int(monkeypatch) -> None:
    """The dense_gemm_launch custom op validates stream_int internally."""
    fake_current = _FakeStream(handle=42, device_index=0)

    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda dev: fake_current
    )

    a = torch.empty(1, 1, 1, dtype=torch.float8_e4m3fn, device="cuda")
    sfa = torch.empty(1, dtype=torch.float8_e8m0fnu, device="cuda")
    b = torch.empty(1, 1, 1, dtype=torch.float8_e4m3fn, device="cuda")
    sfb = torch.empty(1, dtype=torch.float8_e8m0fnu, device="cuda")
    c = torch.empty(1, 1, 1, dtype=torch.bfloat16, device="cuda")
    alpha = torch.empty(1, dtype=torch.float32, device="cuda")

    with pytest.raises(ValueError, match="must be the current stream"):
        torch.ops.b12x.dense_gemm_launch(
            a, b, sfa, sfb, c, alpha,
            1, 1, 1, 1,
            "float8_e4m3fn", "float8_e8m0fnu", "bfloat16", "float32",
            32, 32, 128, 64, 64, 1, 1, 188,
            False, False, False, 1, False, False,
            "tma", False, False, False, 99,  # non-current stream_int
        )


def test_dense_gemm_launch_functional_op_rejects_non_current_stream_int(
    monkeypatch,
) -> None:
    """The functional custom op validates stream_int internally."""
    fake_current = _FakeStream(handle=42, device_index=0)

    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda dev: fake_current
    )

    a = torch.empty(1, 1, 1, dtype=torch.float8_e4m3fn, device="cuda")
    sfa = torch.empty(1, dtype=torch.float8_e8m0fnu, device="cuda")
    b = torch.empty(1, 1, 1, dtype=torch.float8_e4m3fn, device="cuda")
    sfb = torch.empty(1, dtype=torch.float8_e8m0fnu, device="cuda")
    alpha = torch.empty(1, dtype=torch.float32, device="cuda")

    with pytest.raises(ValueError, match="must be the current stream"):
        torch.ops.b12x.dense_gemm_launch_functional(
            a, b, sfa, sfb, alpha,
            1, 1, 1, 1,
            "float8_e4m3fn", "float8_e8m0fnu", "bfloat16", "bfloat16", "float32",
            32, 32, 128, 64, 64, 1, 1, 188,
            False, False, False, 1, False, False,
            "tma", False, False, False, 99,  # non-current stream_int
        )


# ---------------------------------------------------------------------------
# Split-K: atomic and non-atomic paths cannot mix streams.
#
# The single-stream contract means split-K zero (atomic) and reduction
# (non-atomic) run on the same stream as the GEMM.  An explicit non-current
# stream is rejected before any of these stages execute.
# ---------------------------------------------------------------------------


def test_split_k_atomic_path_rejects_non_current_stream(monkeypatch) -> None:
    """Atomic split-K path: out.zero_() and GEMM must be on the same stream.

    The non-current stream is rejected before zero_() or the GEMM launch.
    """
    called = _patch_dense_gemm_internals(monkeypatch)

    fake_current = _FakeStream(handle=42, device_index=0)
    fake_other = _FakeStream(handle=99, device_index=0)

    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda dev: fake_current
    )

    a = torch.empty(1, 1, 1, dtype=torch.float8_e4m3fn)
    sfa = torch.empty(1, dtype=torch.float8_e8m0fnu)
    b = torch.empty(1, 1, 1, dtype=torch.float8_e4m3fn)
    sfb = torch.empty(1, dtype=torch.float8_e8m0fnu)
    out = torch.empty(1, 1, 1, dtype=torch.bfloat16)

    from b12x._lib.dense_gemm import dense_gemm

    with pytest.raises(ValueError, match="must be the current stream"):
        dense_gemm(
            (a, sfa),
            (b, sfb),
            out=out,
            ab_dtype="float8_e4m3fn",
            sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16",
            sf_vec_size=32,
            stream=fake_other,
        )
    assert not called, f"split-K work ran before validation: {called}"


def test_split_k_non_atomic_path_rejects_non_current_stream(monkeypatch) -> None:
    """Non-atomic split-K: scratch alloc, GEMM, and reduction on one stream.

    The non-current stream is rejected before scratch allocation or reduction.
    """
    called = _patch_dense_gemm_internals(monkeypatch)

    fake_current = _FakeStream(handle=42, device_index=0)
    fake_other = _FakeStream(handle=99, device_index=0)

    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda dev: fake_current
    )

    a = torch.empty(1, 1, 1, dtype=torch.float8_e4m3fn)
    sfa = torch.empty(1, dtype=torch.float8_e8m0fnu)
    b = torch.empty(1, 1, 1, dtype=torch.float8_e4m3fn)
    sfb = torch.empty(1, dtype=torch.float8_e8m0fnu)
    out = torch.empty(1, 1, 1, dtype=torch.bfloat16)

    from b12x._lib.dense_gemm import dense_gemm

    with pytest.raises(ValueError, match="must be the current stream"):
        dense_gemm(
            (a, sfa),
            (b, sfb),
            out=out,
            ab_dtype="float8_e4m3fn",
            sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16",
            sf_vec_size=32,
            stream=fake_other,
        )
    assert not called, f"split-K work ran before validation: {called}"


# ---------------------------------------------------------------------------
# Wrapper-level tests: composites reject before quantize/bias.
# ---------------------------------------------------------------------------


def test_block_fp8_linear_rejects_non_current_stream(monkeypatch) -> None:
    """block_fp8_linear_mxfp8 validates stream before quantize or GEMM."""
    fake_current = _FakeStream(handle=42, device_index=0)
    fake_other = _FakeStream(handle=99, device_index=0)

    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda dev: fake_current
    )

    # Patch the quantizer to detect if it runs before validation.
    import b12x.gemm._shared.block_fp8 as bf

    quantizer_called = []
    original_quant = bf.quantize_block_fp8_linear_input_mxfp8

    def _spy_quant(*a, **kw):
        quantizer_called.append(True)
        return original_quant(*a, **kw)

    monkeypatch.setattr(bf, "quantize_block_fp8_linear_input_mxfp8", _spy_quant)

    source = torch.empty(1, 128, dtype=torch.bfloat16, device="cuda")

    from b12x.gemm._shared.block_fp8 import block_fp8_linear_mxfp8, BlockFP8LinearWeight

    # Only geometry is inspected before the stream guard.
    stub_weight = object.__new__(BlockFP8LinearWeight)
    object.__setattr__(stub_weight, "in_features", 128)
    object.__setattr__(stub_weight, "out_features", 128)

    with pytest.raises(ValueError, match="must be the current stream"):
        block_fp8_linear_mxfp8(
            source=source,
            packed_weight=stub_weight,
            stream=fake_other,
        )
    assert not quantizer_called, "quantizer ran before stream validation"


def test_mxfp8_linear_rejects_non_current_stream(monkeypatch) -> None:
    """mxfp8_linear validates stream before quantize or GEMM."""
    fake_current = _FakeStream(handle=42, device_index=0)
    fake_other = _FakeStream(handle=99, device_index=0)

    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda dev: fake_current
    )

    source = torch.empty(1, 128, dtype=torch.bfloat16, device="cuda")

    from b12x.gemm.mxfp8_linear._kernel import mxfp8_linear, MXFP8LinearWeight

    # Only geometry is inspected before the stream guard.
    stub_weight = object.__new__(MXFP8LinearWeight)
    object.__setattr__(stub_weight, "in_features", 128)
    object.__setattr__(stub_weight, "padded_in_features", 128)
    object.__setattr__(stub_weight, "out_features", 128)

    with pytest.raises(ValueError, match="must be the current stream"):
        mxfp8_linear(
            source=source,
            packed_weight=stub_weight,
            stream=fake_other,
        )


def test_tensor_fp8_linear_rejects_non_current_stream(monkeypatch) -> None:
    """tensor_fp8_linear validates stream before padding or GEMM."""
    fake_current = _FakeStream(handle=42, device_index=0)
    fake_other = _FakeStream(handle=99, device_index=0)

    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda dev: fake_current
    )

    source = torch.empty(1, 128, dtype=torch.float8_e4m3fn, device="cuda")

    from b12x.gemm.tensor_fp8_linear._kernel import (
        tensor_fp8_linear,
        TensorFP8LinearWeight,
    )

    class _StubWeight(TensorFP8LinearWeight):
        def __init__(self):
            pass

    with pytest.raises(ValueError, match="must be the current stream"):
        tensor_fp8_linear(
            source=source,
            packed_weight=_StubWeight(),
            stream=fake_other,
        )
