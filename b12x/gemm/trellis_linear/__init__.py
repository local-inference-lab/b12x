"""Trellis-coded dense linear for SM12x.

The operation consumes the checkpoint-native ``trellis_t256`` payload
(MCG, SQG-XOR-Cheb-T12, or SQG-FP16-D3L codebooks) and
its two Hadamard sign vectors.  Preparation validates and records zero-copy
views; execution performs input rotation, a W4A16 or direct E4M3-W4A8 GEMM,
and output rotation. Optional additive rank-16 branches evaluate native BF16
tensor-core stages while retaining the packed base weight. Equal-shaped
gate/up factors share one projection launch and one output launch.
Callers allocate stable capacity with :func:`make_buffers` or
:func:`make_pair_buffers` and obtain allocation-free exact-row views with
:func:`view_buffers` or :func:`view_pair_buffers`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="trellis_linear",
    group="gemm",
    api_style="oneshot",
    entry_points=(
        "PreparedWeight",
        "PreparedAdditiveWeight",
        "PreparedAdditivePair",
        "Buffers",
        "PairBuffers",
        "prepare_weight",
        "prepare_additive_weight",
        "prepare_additive_pair",
        "prepare_pair_weight",
        "make_buffers",
        "make_pair_buffers",
        "run",
        "run_additive",
        "run_additive_pair",
        "view_buffers",
        "view_pair_buffers",
        "is_supported",
        "clear_caches",
    ),
    dtypes=("bf16", "fp16"),
    recipes=(
        "w4a16/trellis_mcg",
        "w4a16/trellis_sqg_e4m3",
        "w4a16/trellis_sqg_fp16",
        "w4a8/trellis_sqg_e4m3",
    ),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/local-inference-lab/b12x",
        commit="c58a381",
        paths=(
            "b12x/gemm/trellis_linear/",
            "b12x/moe/_shared/kernels/w4a16/",
        ),
    ),
    test_path="tests/gemm/test_trellis_linear.py",
    since="1.0.1",
    notes="Trellis-coded W4A16, single/paired BF16 rank-16, and direct E4M3-W4A8 linear.",
)

if TYPE_CHECKING:
    from .api import (  # noqa: F401
        Buffers,
        PairBuffers,
        PreparedAdditivePair,
        PreparedAdditiveWeight,
        PreparedWeight,
        clear_caches,
        is_supported,
        make_buffers,
        make_pair_buffers,
        prepare_additive_pair,
        prepare_additive_weight,
        prepare_weight,
        prepare_pair_weight,
        run,
        run_additive,
        run_additive_pair,
        view_buffers,
        view_pair_buffers,
    )

install_lazy_api(globals(), META)
