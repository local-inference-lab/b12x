"""Profiled launch plans for the packed dense linear recipes.

The dense SM120 GEMM behind ``blockscaled`` (NVFP4 / MXFP4 / block-FP8),
``mxfp8_linear`` and ``tensor_fp8_linear`` chooses its MMA tile, K tile,
load path, operand swap and split-K from a built-in heuristic. This op turns
that choice into a profiled policy decision: ``plan(Caps)`` resolves the
``gemm.dense_linear`` component once per (recipe, geometry, capacity), and
``mm``/``mm_serialized`` forward the resolved launch options to the existing
one-shot entry points without any run-time policy lookup. ``plan_table``
plans a whole serving capacity ladder so integrations can select by the
live ``expected_m`` before CUDA-graph capture.

Example:
    from b12x.gemm import dense_linear, mxfp8_linear

    weight = mxfp8_linear.pack_weight(w_fp8, w_scale)
    table = dense_linear.plan_table(
        dense_linear.Caps(device="cuda", recipe="mxfp8", in_features=K,
                          out_features=N, max_tokens=8),
        token_counts=(1, 2, 4, 8, 16, 32, 64, 128, 2048),
    )
    y = dense_linear.mm(x, weight, plan=table, expected_m=x.shape[0])
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="dense_linear",
    group="gemm",
    api_style="planned",
    entry_points=(
        "Caps",
        "Plan",
        "PlanTable",
        "DenseLinearConfig",
        "DenseLinearQuery",
        "plan",
        "plan_table",
        "mm",
        "mm_serialized",
        "is_supported",
    ),
    dtypes=("bf16", "fp16"),
    recipes=("nvfp4", "mxfp4", "mxfp8", "tensor_fp8", "block_fp8", "mxfp6"),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="e076bd6d",
        paths=("b12x/gemm/dense_linear/api.py",),
    ),
    test_path="tests/gemm/test_dense_linear.py",
    since="0.31.0",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        Caps,
        DenseLinearConfig,
        DenseLinearQuery,
        Plan,
        PlanTable,
        is_supported,
        mm,
        mm_serialized,
        plan,
        plan_table,
    )

install_lazy_api(globals(), META)
