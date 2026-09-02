"""SM120 (RTX PRO 6000 Blackwell) mHC decode policy: source-split selection and
the prefill crossover, plus a bitwise check of the split decode kernel."""

from __future__ import annotations

import pytest
import torch

from b12x.norm import mhc
from b12x.norm.mhc import _impl as mhc_impl
from b12x.norm.mhc import _kernels as mhc_kernels

from ..conftest import require_b12x as require_sm120
from .test_mhc import _make_inputs, _make_mhc_binding, _mhc_pre_reference

_HIDDEN = 4096


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (1, (0, 0)),
        (8, (0, 0)),
        (16, (0, 0)),
        (31, (0, 0)),
        (32, (4, 6)),
        (64, (4, 6)),
        (96, (4, 6)),
        (128, (4, 6)),
        (256, (4, 6)),
    ],
)
def test_sm120_decode_split_policy(
    monkeypatch: pytest.MonkeyPatch, tokens: int, expected: tuple[int, int]
) -> None:
    monkeypatch.delenv("B12X_MHC_DECODE_SPLITS", raising=False)
    assert (
        mhc_kernels._selected_post_pre_decode_split_n(
            num_tokens=tokens, hidden_size=_HIDDEN, compute_capability=(12, 0)
        )
        == expected
    )


def test_sm120_decode_split_policy_requires_hidden_4096(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("B12X_MHC_DECODE_SPLITS", raising=False)
    assert mhc_kernels._selected_post_pre_decode_split_n(
        num_tokens=64, hidden_size=7168, compute_capability=(12, 0)
    ) == (0, 0)


def test_other_architectures_keep_their_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("B12X_MHC_DECODE_SPLITS", raising=False)
    select = mhc_kernels._selected_post_pre_decode_split_n
    assert select(num_tokens=8, hidden_size=_HIDDEN, compute_capability=(12, 1)) == (
        4,
        6,
    )
    assert select(num_tokens=10, hidden_size=_HIDDEN, compute_capability=(12, 1)) == (
        8,
        6,
    )
    assert select(num_tokens=7, hidden_size=_HIDDEN, compute_capability=(12, 1)) == (
        0,
        0,
    )
    assert select(num_tokens=64, hidden_size=_HIDDEN, compute_capability=(10, 0)) == (
        0,
        0,
    )
    assert select(num_tokens=64, hidden_size=_HIDDEN, compute_capability=(9, 0)) == (
        0,
        0,
    )


def test_explicit_split_override_wins_on_sm120(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("B12X_MHC_DECODE_SPLITS", "8")
    monkeypatch.setenv("B12X_MHC_DECODE_TILE_N", "6")
    assert mhc_kernels._selected_post_pre_decode_split_n(
        num_tokens=8, hidden_size=_HIDDEN, compute_capability=(12, 0)
    ) == (8, 6)
    monkeypatch.setenv("B12X_MHC_DECODE_SPLITS", "0")
    assert mhc_kernels._selected_post_pre_decode_split_n(
        num_tokens=128, hidden_size=_HIDDEN, compute_capability=(12, 0)
    ) == (0, 0)


def test_prefill_crossover_default_moves_to_192_rows_on_sm120() -> None:
    assert mhc_impl._mhc_prefill_min_tokens_for_capability((12, 0)) == 192
    assert mhc_impl._mhc_prefill_min_tokens_for_capability((12, 1)) == 96
    assert mhc_impl._mhc_prefill_min_tokens_for_capability((10, 0)) == 96
    assert mhc_impl._mhc_prefill_min_tokens_for_capability(None) == 96
    assert mhc_impl._default_mhc_prefill_min_tokens(torch.device("cpu")) == 96


@pytest.mark.parametrize("tokens", [32, 64, 128])
def test_sm120_split_decode_kernel_matches_unsplit_within_rounding(
    monkeypatch: pytest.MonkeyPatch, tokens: int
) -> None:
    """The split kernel reduces the fn partials over four source splits in
    the finalize, so gate/mix values may differ from the unsplit kernel by
    fp32 rounding and y by one bf16 ulp; the residual update does not depend
    on that reduction and must be bitwise identical."""
    device = require_sm120()
    if tuple(torch.cuda.get_device_capability(device)) != (12, 0):
        pytest.skip("SM120 decode policy test")
    monkeypatch.delenv("B12X_MHC_DECODE_SPLITS", raising=False)
    assert mhc_kernels._selected_post_pre_decode_split_n(
        num_tokens=tokens, hidden_size=_HIDDEN
    ) == (4, 6)
    residual, x, fn, scale, bias = _make_inputs(
        tokens=tokens, hidden_size=_HIDDEN, seed=4_242 + tokens, device=device
    )
    _, prev_post, prev_comb = _mhc_pre_reference(
        residual, fn, scale, bias, rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20
    )
    prev_post = prev_post.contiguous()
    prev_comb = prev_comb.contiguous()

    def run(splits_override: str | None) -> tuple[torch.Tensor, ...]:
        if splits_override is None:
            monkeypatch.delenv("B12X_MHC_DECODE_SPLITS", raising=False)
        else:
            monkeypatch.setenv("B12X_MHC_DECODE_SPLITS", splits_override)
        binding = _make_mhc_binding(tokens=tokens, hidden_size=_HIDDEN, device=device)
        mhc.run_post_pre(
            x,
            residual,
            prev_post,
            prev_comb,
            fn,
            scale,
            bias,
            rms_eps=1e-6,
            hc_eps=1e-6,
            sinkhorn_iters=20,
            binding=binding,
        )
        torch.cuda.synchronize(device)
        return (
            binding.y.clone(),
            binding.post_buffer.clone(),
            binding.comb_buffer.clone(),
            binding.out.clone(),
        )

    y_s, post_s, comb_s, out_s = run(None)
    y_u, post_u, comb_u, out_u = run("0")
    assert torch.equal(out_s, out_u), "residual update must not depend on the split"
    assert torch.equal(y_s, run(None)[0]), "split kernel must be deterministic"
    torch.testing.assert_close(y_s.float(), y_u.float(), rtol=2**-7, atol=2**-14)
    torch.testing.assert_close(post_s, post_u, rtol=2**-20, atol=2**-22)
    torch.testing.assert_close(comb_s, comb_u, rtol=2**-20, atol=2**-22)
    assert torch.isfinite(y_s.float()).all()
