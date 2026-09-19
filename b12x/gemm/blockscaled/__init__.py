"""Prepared dense block-scaled GEMM: ``C = (A·SFA) @ (B·SFB)``.

Functional and caller-buffer operations share the SM120 warp-MMA engine (no
TMEM, no tcgen05, no 2-CTA). Recipes: NVFP4 (Float4E2M1 values, e4m3 scales, vec 16),
MXFP4 (e8m0 scales, vec 32), MXFP8 (e4m3 values, e8m0 scales, vec 32), and
tensor-scaled FP8. ``mm`` accepts raw ``(values, scales)`` operand pairs or a
weight returned by ``pack_weight``. A packed MXFP8 weight accepts either a
BF16/FP16 activation or a prequantized ``(values, scales)`` pair with compact
row-major or F8_128x4-swizzled UE8M0 scales. Pass swizzled storage flattened
(or as its native 6D view); a 2D ``[M,K/32]`` scale is interpreted as compact.
``expected_m`` and precision constraints are immutable query coordinates.

``plan(query)`` declares an invocation without compiling or allocating.
``PreparationSession`` selects and primes its prepared ``Plan`` before
``mm`` or the explicit ``w4a16``/``w8a16`` entry points may execute. A16 uses
BF16 activations and inline weight dequantization; it performs no activation
scaling. ``plan_regimes`` combines exact static-shape variants with one bounded
dynamic-row execution while keeping runtime dispatch behind the custom-op boundary.
It accepts a ``BlockscaledQuery`` or a ``FixedBlockscaledQuery`` whose
``expected_m`` is unset; a fixed capacity serves serialized NVFP4/MXFP4/block-FP8
operands and packed tensor-FP8 at any row count up to ``max_rows``.

NVFP4 packed weights require ``pack_weight(..., recipe='nvfp4', global_scale=g,
global_scale_kind='multiplier')``. ``'reciprocal'`` accepts a weight quantizer
multiplier without creating another tensor. NVFP4 quantized activation calls
also require ``activation_global_scale``, the activation quantizer multiplier.
Global scales must be finite and strictly positive; reconstructed weights must
fit BF16. Quantized activation execution requires K divisible by 128 and N by 8.

Caller-owned output/workspace buffers are described by the query and reserved
before durable binding. Concurrent owners use disjoint buffers. Functional
forms retain their output allocation contract, while graph replay uses fixed
captured addresses. No standalone warmup or implicit execution path exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="blockscaled",
    group="gemm",
    api_style="planned",
    entry_points=(
        "Weight",
        "NVFP4LinearWeight",
        "BlockscaledQuery",
        "BlockscaledConfig",
        "FixedBlockscaledQuery",
        "plan",
        "plan_regimes",
        "query_from_call",
        "mm",
        "mm_mxfp4",
        "mm_nvfp4",
        "mm_block_fp8",
        "pack_weight",
        "is_supported",
        "w4a16",
        "w8a16",
        "workspace_size",
    ),
    dtypes=("bf16", "fp16", "fp32", "fp8_e4m3", "fp4_e2m1"),
    recipes=("nvfp4", "mxfp4", "mxfp8"),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=("b12x/gemm/dense.py",),
    ),
    test_path="tests/gemm/test_blockscaled.py",
    since="0.7.0",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        Weight,
        NVFP4LinearWeight,
        BlockscaledQuery,
        BlockscaledConfig,
        FixedBlockscaledQuery,
        plan,
        plan_regimes,
        query_from_call,
        is_supported,
        mm,
        mm_block_fp8,
        mm_mxfp4,
        mm_nvfp4,
        pack_weight,
        w4a16,
        w8a16,
        workspace_size,
    )

install_lazy_api(globals(), META)
