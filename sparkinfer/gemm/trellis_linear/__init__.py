"""Native EXL3 Trellis W4A16 dense linear for SM12x.

The operation consumes the checkpoint-native ``trellis3_t256`` payload and
its two Hadamard sign vectors.  Preparation validates and records zero-copy
views; execution performs input rotation, the W4A16 GEMM, and output rotation.
Callers that need CUDA-graph capture must provide stable ``output``,
``gemm_output``, and ``c_tmp`` tensors to :func:`run`.
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
        "prepare_weight",
        "run",
        "is_supported",
        "clear_caches",
    ),
    dtypes=("bf16", "fp16"),
    recipes=("w4a16/exl3_trellis_mcg",),
    provenance=Provenance(
        repo="https://github.com/local-inference-lab/sparkinfer",
        commit="c58a381",
        paths=(
            "sparkinfer/gemm/trellis_linear/",
            "sparkinfer/moe/_shared/kernels/w4a16/",
        ),
    ),
    test_path="tests/gemm/test_trellis_linear.py",
    since="1.0.1",
    notes="Zero-copy native EXL3 Trellis dense W4A16 linear.",
)

if TYPE_CHECKING:
    from .api import (  # noqa: F401
        PreparedWeight,
        clear_caches,
        is_supported,
        prepare_weight,
        run,
    )

install_lazy_api(globals(), META)
