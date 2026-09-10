"""Native unquantized GEMV-style projections with BF16/FP32 operands.

``mm`` runs ``y = x @ weight.T + bias`` through opaque torch custom ops.
FP32 accumulation and bias addition happen before the final BF16/FP32 cast.
Caller-owned ``out`` avoids allocation; live rows and input strides reuse
one geometry/type specialization. ``precompile`` compiles and warm-runs that
specialization before CUDA graph capture. There is no cuBLAS fallback.

Example:
    from b12x.gemm import bf16_gemv

    bf16_gemv.precompile(layer.weight)        # one-time, at weight load
    out = bf16_gemv.mm(x, layer.weight)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="bf16_gemv",
    group="gemm",
    api_style="oneshot",
    entry_points=(
        "mm",
        "bf16_gemv_small_n",
        "precompile",
        "is_supported",
        "is_disabled",
        "SMALL_M_MAX",
        "SMALL_N_GEMV_MAX_OUT",
        "SMALL_N_GEMV_MIN_IN",
    ),
    dtypes=("bf16", "fp32"),
    provenance=Provenance(
        repo="https://github.com/phaelon74/b12x",
        commit="9c78d553",
        paths=(
            "b12x/gemm/bf16_gemv.py",
            "b12x/gemm/bf16_gemv_op.py",
            "b12x/integration/vllm_plugin.py",
        ),
    ),
    test_path="tests/gemm/test_bf16_gemv.py",
    since="1.0.1",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        SMALL_M_MAX,
        SMALL_N_GEMV_MAX_OUT,
        SMALL_N_GEMV_MIN_IN,
        bf16_gemv_small_n,
        is_disabled,
        is_supported,
        mm,
        precompile,
    )

install_lazy_api(globals(), META)
