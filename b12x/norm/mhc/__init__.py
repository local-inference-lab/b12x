"""mHC residual for SM12x: fused RMSNorm + hyper-connection mixing +
projection (DeepSeek-style), BF16 with TF32 projection paths.

Three phases share one plan/binding; the phase is the verb because the
signatures differ: ``run_pre`` (residual -> mixed input + carry),
``run_post`` (output mix-back), ``run_post_pre`` (fused post+pre between
layers). Sinkhorn-normalized mix matrices; hidden sizes 4096/5120/7168; mix
constants exposed as ``MIXES`` / ``MULT`` / ``PARTIALS``.

V4.1 lagged input mixing: pass incoming FP32 ``pre_mix[tokens, 4]`` to
``run_pre`` or ``run_post_pre`` and a disjoint caller-owned FP32 ``pre_out``
(directly or at bind time). The incoming coefficients collapse this residual;
newly predicted coefficients are written to ``pre_out`` for the next sublayer.
The four returned tensors remain ``residual, post, comb, y``. Initialize the
first incoming mix to one-hot. With ``norm_weight`` and ``norm_eps=1e-20``,
RMSNorm follows the BF16-rounded collapse. Omitting both mix arguments keeps
the original current-mix behavior. ``run_pre`` accepts either an expanded
``[tokens, 4, hidden]`` residual with full ``fn[24, 4 * hidden]`` or the older
broadcast ``[tokens, hidden]`` residual with pre-summed ``fn[24, hidden]``.
Scratch ``split_k`` defaults to ``4 * hidden / 256`` (80 for hidden 5120).
Lagged mixing specializes the existing finalize; projection remains a separate
pass, not a single Mega-mHC kernel. Caller-owned paths support CUDA graphs,
not Dynamo tracing.

Planned lifecycle: ``plan(Caps(...))`` -> ``bind`` (views only) ->
``run_*`` (capture safe; torch.compile-safe via opaque custom ops).

Example:
    from b12x.norm import mhc

    plan    = mhc.plan(mhc.Caps(...))
    spec    = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = mhc.bind(plan, scratch=scratch, ...)
    residual, post, comb, y = mhc.run_post_pre(..., binding=binding)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="mhc",
    group="norm",
    api_style="planned",
    entry_points=(
        "Caps",
        "Plan",
        "Binding",
        "MhcConfig",
        "MhcQuery",
        "plan",
        "bind",
        "run_pre",
        "run_post",
        "run_post_pre",
        "run_collapse",
        "MIXES",
        "MULT",
        "PARTIALS",
        "DEFAULT_SPLIT_K",
        "DEFAULT_BLOCK_K",
        "DEFAULT_BLOCK_H",
        "is_supported",
    ),
    dtypes=("bf16",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=(
            "b12x/integration/residual.py",
            "b12x/integration/residual_kernels.py",
        ),
    ),
    test_path="tests/norm/test_mhc.py",
    since="0.7.0",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        DEFAULT_BLOCK_H,
        DEFAULT_BLOCK_K,
        DEFAULT_SPLIT_K,
        MIXES,
        MULT,
        PARTIALS,
        Binding,
        Caps,
        MhcConfig,
        MhcQuery,
        Plan,
        bind,
        is_supported,
        plan,
        run_post,
        run_post_pre,
        run_pre,
        run_collapse,
    )

install_lazy_api(globals(), META)
