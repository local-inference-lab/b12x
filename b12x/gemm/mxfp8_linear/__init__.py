"""ModelOpt MXFP8 linear for SM12x (one-shot).

``pack_weight`` converts a ModelOpt MXFP8 checkpoint weight into the
dense-GEMM layout (host-side, one-time); ``mm`` quantizes the BF16/FP16
input inline and runs the native MXFP8 GEMM.  Dispatches through an opaque
torch custom op, so it is torch.compile- and CUDA-graph-safe.

Example:
    from b12x.gemm import mxfp8_linear

    weight = mxfp8_linear.pack_weight(w_mxfp8, w_scale)   # one-time
    out = mxfp8_linear.mm(x, weight, expected_m=x.shape[0])
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="mxfp8_linear",
    group="gemm",
    api_style="oneshot",
    entry_points=(
        "Weight",
        "empty_input",
        "is_supported",
        "mm",
        "mm_quantized",
        "mm_quantized_into",
        "pack_weight",
        "quantize_input_slice",
    ),
    dtypes=("bf16", "fp16"),
    recipes=("mxfp8",),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=("b12x/gemm/mxfp8_linear.py",),
    ),
    test_path="tests/gemm/test_mxfp8_linear.py",
    since="0.7.0",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        Weight,
        empty_input,
        is_supported,
        mm,
        mm_quantized,
        mm_quantized_into,
        pack_weight,
        quantize_input_slice,
    )

install_lazy_api(globals(), META)
