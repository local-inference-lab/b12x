"""gemm.blockscaled: small-row GEMV regime of serialized 128x128 block-FP8 plans.

Fixed serialized block-FP8 plans with at most eight expected rows and enough
output features run the GEMV instead of the dense GEMM.  Both are checked
against an FP64 reference on the dequantized operands, against each other,
for determinism, and under CUDA graph replay.
"""

from __future__ import annotations

import pytest
import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.gemm import blockscaled
from b12x.gemm.blockscaled import _block_fp8_gemv
from b12x.preparation.types import require_prepared

from ._blockscaled import prepared
from ..conftest import require_b12x

_OPTIONS = dict(
    ab_dtype="float8_e4m3fn", sf_dtype="float32", c_dtype="bfloat16",
    sf_vec_size=128, block_fp8=True,
)

# (out_features, in_features): MiMo-V2 global (padded 3392) and SWA fused QKV,
# a DeepSeek-style 7168-wide input, a K that is not a multiple of 1024, a
# single 128-block, and the widest routed output.
SHAPES = [(3456, 4096), (3712, 4096), (1536, 7168), (2048, 1536), (1024, 128), (4096, 4096)]


def _operands(m, n, k, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    lhs = torch.randn((m, k), generator=gen, device="cuda").to(torch.float8_e4m3fn)
    rhs = torch.randn((n, k), generator=gen, device="cuda").to(torch.float8_e4m3fn)
    lhs_scale = torch.rand((m, k // 128), generator=gen, device="cuda") + 0.25
    rhs_scale = torch.rand(((n + 127) // 128, k // 128), generator=gen, device="cuda") + 0.25
    return lhs, lhs_scale, rhs, rhs_scale


def _reference(lhs, lhs_scale, rhs, rhs_scale):
    lhs64 = lhs.double() * lhs_scale.double().repeat_interleave(128, dim=1)
    rhs_rows = rhs_scale.double().repeat_interleave(128, dim=0)[: rhs.shape[0]]
    rhs64 = rhs.double() * rhs_rows.repeat_interleave(128, dim=1)
    return lhs64 @ rhs64.T


def _run(plan, lhs, lhs_scale, rhs, rhs_scale):
    return blockscaled.mm_block_fp8(lhs, lhs_scale, rhs, rhs_scale, plan=plan,
                                    out_dtype=torch.bfloat16)


def _assert_rounded(out, ref):
    # One BF16 rounding of an FP32-accumulated sum: within one BF16 ulp.
    err = (out.double() - ref).abs()
    ulp = ref.abs().clamp_min(1e-30) * 2.0 ** -7
    assert torch.isfinite(out).all()
    assert bool((err <= ulp + 1e-6 * ref.abs().max()).all()), float((err / ulp).max())


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("m", [1, 3, 8])
def test_block_fp8_gemv_matches_reference_and_dense(m, n, k, monkeypatch):
    require_b12x()
    lhs, lhs_scale, rhs, rhs_scale = _operands(m, n, k, seed=m * 7919 + n + k)
    assert _block_fp8_gemv.supports(m, rhs.shape[0], k)
    ref = _reference(lhs, lhs_scale, rhs, rhs_scale)
    with prepared((lhs, lhs_scale), (rhs, rhs_scale), expected_m=m, **_OPTIONS) as plan:
        assert require_prepared(plan, "gemm.blockscaled.fixed").gemv is not None
        gemv = _run(plan, lhs, lhs_scale, rhs, rhs_scale)
        again = _run(plan, lhs, lhs_scale, rhs, rhs_scale)
    # The same fixed plan without the GEMV regime: the dense GEMM it replaces.
    monkeypatch.setattr(_block_fp8_gemv, "supports", lambda *args: False)
    with prepared((lhs, lhs_scale), (rhs, rhs_scale), expected_m=m, **_OPTIONS) as plan:
        assert require_prepared(plan, "gemm.blockscaled.fixed").gemv is None
        dense = _run(plan, lhs, lhs_scale, rhs, rhs_scale)
    _assert_rounded(gemv, ref)
    assert torch.equal(gemv, again), "block-FP8 GEMV must be deterministic"
    # The dense GEMM computes the same products; its 2-way split-K for 2..6
    # rows may add BF16 partials (B12X_DENSE_SPLITK_TURBO), so compare norms.
    diff = (gemv.double() - dense.double()).norm() / ref.norm()
    assert diff < 4e-3, float(diff)


def test_block_fp8_gemv_graph_replay_reads_new_operands():
    require_b12x()
    m, n, k = 4, 3712, 4096
    lhs, lhs_scale, rhs, rhs_scale = _operands(m, n, k, seed=11)
    with prepared((lhs, lhs_scale), (rhs, rhs_scale), expected_m=m, **_OPTIONS) as plan:
        static_out = _run(plan, lhs, lhs_scale, rhs, rhs_scale)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = _run(plan, lhs, lhs_scale, rhs, rhs_scale)
        new_lhs, new_scale, _, _ = _operands(m, n, k, seed=12)
        lhs.copy_(new_lhs)
        lhs_scale.copy_(new_scale)
        graph.replay()
        torch.cuda.synchronize()
    _assert_rounded(static_out, _reference(lhs, lhs_scale, rhs, rhs_scale))


@pytest.mark.parametrize("m,n,k", [(9, 4096, 4096), (4, 8192, 4096), (4, 1024, 1000)])
def test_block_fp8_gemv_leaves_other_shapes_on_the_dense_gemm(m, n, k):
    assert not _block_fp8_gemv.supports(m, n, k)


def test_block_fp8_gemv_live_rows_reuse_one_compiled_callable():
    """Row counts are launch arguments: every M <= 8 runs the same program."""
    require_b12x()
    n, k = 3712, 4096
    ordinal = torch.cuda.current_device()
    program = _block_fp8_gemv.compile_block_fp8_gemv(ordinal, n, k)
    with kernel_resolution_guard("block-FP8 GEMV live rows"):
        for m in (1, 2, 5, 8):
            assert _block_fp8_gemv.compile_block_fp8_gemv(ordinal, n, k) is program
            lhs, lhs_scale, rhs, rhs_scale = _operands(m, n, k, seed=m)
            out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device="cuda")
            program(lhs, lhs_scale, rhs, rhs_scale, out)
            _assert_rounded(out, _reference(lhs, lhs_scale, rhs, rhs_scale))
