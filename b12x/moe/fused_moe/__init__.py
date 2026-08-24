"""Fused tensor-parallel MoE for SM12x: route -> FC1 -> activation -> FC2 ->
scatter, in one launch family.

Checkpoint encodings are described by ``PackedConfig`` or ``TrellisConfig``;
kernel recipes remain private planner policy. The
``quantization_config.b12x_trellis`` schema represents the mcg, sqg_e4m3, and
sqg_fp16 codebooks. Routed execution currently implements MCG K3/K4/K5 and
SQG-E4M3 K3.
Activations: silu, situ, relu2, swigluoai_uninterleave. Kernel regimes are
private planner policy selected by ``plan`` from checkpoint metadata and
serving capacity.

Weight lifecycle (host-side, one-time):
    ``plan_weights`` -> ``prepare_weights`` -> ``ExpertWeights``
Runtime lifecycle:
    ``plan(Caps)`` -> ``bind`` / ``bind_sparse`` / ``bind_route``
    (allocation-free views) -> ``run`` / ``run_sparse`` / ``route``
    (CUDA-graph capture safe)
``required_nbytes(Caps)`` prices that scratch without compiling launches or
retaining storage.
``route_topk`` is a standalone one-shot top-k router; ``run_sparse`` fuses
gate -> top-k -> experts from router logits.

Example:
    from b12x.moe import fused_moe

    config  = fused_moe.PackedConfig(source_format="modelopt_nvfp4")
    wplan   = fused_moe.plan_weights(config=config, ...)
    experts = fused_moe.prepare_weights(
        plan=wplan, weights=fused_moe.PackedWeights(...))
    plan    = fused_moe.plan(fused_moe.Caps(config=config, weight_plan=wplan, ...))
    spec    = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = fused_moe.bind(plan, scratch=scratch, a=x, experts=experts,
                             topk_weights=tw, topk_ids=ti)
    out     = fused_moe.run(binding=binding)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="fused_moe",
    group="moe",
    api_style="planned",
    entry_points=(
        "Caps",
        "PackedConfig",
        "PackedWeights",
        "Plan",
        "Binding",
        "SparseBinding",
        "RouteBinding",
        "ExpertWeights",
        "Routing",
        "ScaleFactors",
        "TrellisConfig",
        "TrellisWeights",
        "WeightsPlan",
        "plan",
        "required_nbytes",
        "plan_weights",
        "prepare_weights",
        "prepare_fc2_weights",
        "prewarm_fc2",
        "bind",
        "bind_sparse",
        "bind_route",
        "run",
        "run_fc2",
        "run_sparse",
        "route",
        "route_topk",
        "is_supported",
        "clear_caches",
    ),
    dtypes=("bf16", "fp16"),
    recipes=(
        "nvfp4",
        "w4a8_mx",
        "w4a8_nvfp4",
        "w6a8_mx",
        "w4a16",
        "b12x_trellis",
    ),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=("b12x/integration/tp_moe.py", "b12x/moe/"),
    ),
    test_path="tests/moe/test_fused_moe.py",
    since="0.7.0",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        Binding,
        Caps,
        ExpertWeights,
        PackedConfig,
        PackedWeights,
        Plan,
        RouteBinding,
        Routing,
        ScaleFactors,
        SparseBinding,
        TrellisConfig,
        TrellisWeights,
        WeightsPlan,
        bind,
        bind_route,
        bind_sparse,
        clear_caches,
        is_supported,
        plan,
        required_nbytes,
        plan_weights,
        prepare_weights,
        prepare_fc2_weights,
        prewarm_fc2,
        route,
        route_topk,
        run,
        run_fc2,
        run_sparse,
    )

install_lazy_api(globals(), META)
