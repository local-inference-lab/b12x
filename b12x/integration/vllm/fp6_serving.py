"""Framework enablement layer for serving b12x MX-FP6 (W6A6/W6A8) checkpoints.

This is the reference surface a vLLM / SGLang quantization backend calls to run an
FP6 checkpoint produced by :mod:`b12x.quantization.mxfp6.fp6_safetensors_export`.
It is framework-agnostic (no vLLM/SGLang imports) so it can be wired into any
fork; :mod:`b12x.integration.vllm.plugin` is the concrete vLLM adapter.

Enablement is twofold (mirrors how the FP4 b12x path is selected framework-side):

1. **Env gate** — ``B12X_ENABLE_FP6=1`` must be set. If unset,
   :func:`should_use_b12x_fp6` returns ``False``
   and the framework falls back to its native path.
2. **Checkpoint detection** — ``config.json`` carries
   ``quantization_config={"quant_method": "modelopt", "quant_algo": "W6A6", ...}``.

Exact FP6 call contract
-----------------------

**MoE** (gated SiLU). All tensors on CUDA; ``E`` experts, hidden ``K``,
intermediate ``N``; FP6 codes pack 4 values into 3 bytes (``3*dim/4``); block
scales are UE8M0 (``float8_e8m0fnu`` bytes) at ``sf_vec_size=32``:

* ``hidden_states``  ``(M, K)``      bfloat16 activations (quantized to FP6 in-kernel)
* ``topk_weights``   ``(M, topk)``   float32 router weights
* ``topk_ids``       ``(M, topk)``   int32 expert ids
* prepared experts from :func:`b12x.moe.fused_moe.prepare_weights` with a
  ``PackedConfig(source_format="mxfp6_e2m3", w13_layout="w13")`` and FC1 rows
  in ``[up; gate]``
* output ``(M, K)`` bfloat16.

Routing is the framework's responsibility.  :class:`B12XFP6MoEMethod.apply`
consumes ``topk_ids``/``topk_weights`` and returns the routed-and-combined output.

**Dense linear** ``y = x @ W.T``:

* ``x`` ``(M, in_features)`` bfloat16 -> ``y`` ``(M, out_features)`` bfloat16
* weight from :func:`b12x.quantization.mxfp6.load_fp6_dense_checkpoint` as an
  :class:`~b12x.quantization.mxfp6.fp6_dense_weights.FP6DenseWeight`.

End-to-end flow
---------------

    pip install b12x
    python scripts/quantize_model_fp6.py --model <bf16 model> --out <fp6 model> --arch auto
    export B12X_ENABLE_FP6=1
    vllm serve <fp6 model>      # plugin detects W6A6 and calls b12x
"""
from __future__ import annotations

import os
from typing import Any, Optional

import torch

ENABLE_ENV = "B12X_ENABLE_FP6"
QUANT_METHOD = "modelopt"
QUANT_ALGO = "W6A6"

# Process-wide scratch/plan cache shared across ALL MoE layers of a model.
# Layers run sequentially on one stream — eager and inside CUDA graphs alike —
# so scratch reuse is safe, and every layer of a model has identical
# (E, K, N, topk) geometry.  Per-layer caching multiplies the footprint by
# the layer count (~40x): fatal under vLLM's CUDA-graph memory estimator.
_SCRATCH_CACHE: dict[tuple, tuple[Any, tuple[torch.Tensor, ...]]] = {}

# Deduped fused_moe weight plans, keyed by geometry.  Every MoE layer of a
# model shares one plan object, which is what makes the scratch cache above
# actually shared (its key includes ``id(weight_plan)``) and keeps
# ``fused_moe.bind``'s ``experts.plan == caps.weight_plan`` check trivially
# true for scratch planned against the shared object.
_WEIGHT_PLAN_CACHE: dict[tuple, Any] = {}


def get_fp6_moe_weight_plan(
    *,
    source_format: str,
    activation: str,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> Any:
    """One shared ``fused_moe.plan_weights`` result per (geometry, activation)."""
    from b12x.moe import fused_moe

    key = (
        source_format,
        activation,
        int(num_experts),
        int(hidden_size),
        int(intermediate_size),
    )
    plan = _WEIGHT_PLAN_CACHE.get(key)
    if plan is None:
        config = fused_moe.PackedConfig(
            source_format=source_format,
            w13_layout="w13",  # [up; gate] FC1 rows.
        )
        plan = fused_moe.plan_weights(
            config=config,
            activation=activation,
            dtype=torch.bfloat16,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        _WEIGHT_PLAN_CACHE[key] = plan
    return plan


def _env_truthy(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() in ("1", "true", "yes", "on")


def is_b12x_fp6_enabled() -> bool:
    """True iff ``B12X_ENABLE_FP6`` is set."""
    return _env_truthy(ENABLE_ENV)


def _quant_config(config: Any) -> Optional[dict]:
    qc = (
        config.get("quantization_config")
        if isinstance(config, dict)
        else getattr(config, "quantization_config", None)
    )
    if qc is None:
        return None
    if isinstance(qc, dict):
        return qc
    if hasattr(qc, "to_dict"):
        return qc.to_dict()
    try:
        return dict(vars(qc))
    except TypeError:
        return None


def is_b12x_fp6_checkpoint(config: Any) -> bool:
    """True iff ``config`` declares a b12x FP6 (modelopt + W6A6) quantization."""
    qc = _quant_config(config)
    if not qc:
        return False
    return (
        str(qc.get("quant_method", "")).lower() == QUANT_METHOD
        and str(qc.get("quant_algo", "")).upper() == QUANT_ALGO
    )


def should_use_b12x_fp6(config: Any) -> bool:
    """Gate: env enabled AND the checkpoint is an FP6 checkpoint."""
    return is_b12x_fp6_enabled() and is_b12x_fp6_checkpoint(config)


def kernel_source_format_for_moe(_checkpoint_source_format: str) -> str:
    """Map a checkpoint ``source_format`` to the ``w6a8_mx`` fused_moe tag."""
    # The w6a8_mx preparation path requires the mxfp6_e2m3 source tag regardless
    # of whether the checkpoint was exported as mxfp6_default or mxfp6_w6a8.
    return "mxfp6_e2m3"


class B12XFP6MoEMethod:
    """Reference routed-MoE method backed by :mod:`b12x.moe.fused_moe`.

    Holds one layer's prepared experts and weight plan; scratch/plan tensors come
    from the process-wide shared cache (see ``_SCRATCH_CACHE``).
    """

    def __init__(
        self,
        experts_prepared: Any,
        weight_plan: Any,
        *,
        input_scales_static: bool = True,
        apply_router_weight_on_input: bool = False,
    ):
        self.experts_prepared = experts_prepared
        self.weight_plan = weight_plan
        self.input_scales_static = input_scales_static
        self.apply_router_weight_on_input = apply_router_weight_on_input

    def _plan_and_scratch(
        self, m: int, topk: int, device: torch.device
    ) -> tuple[Any, tuple[torch.Tensor, ...]]:
        from b12x.moe import fused_moe

        key = (
            int(m),
            int(topk),
            device,
            id(self.weight_plan),
            bool(self.apply_router_weight_on_input),
        )
        cached = _SCRATCH_CACHE.get(key)
        if cached is None:
            plan = fused_moe.plan(
                fused_moe.Caps(
                    max_tokens=m,
                    num_topk=topk,
                    device=device,
                    config=self.weight_plan.checkpoint_config,
                    weight_plan=self.weight_plan,
                    core_token_counts=(m,),
                    route_num_experts=0,
                    apply_router_weight_on_input=self.apply_router_weight_on_input,
                )
            )
            scratch = tuple(
                torch.empty(
                    shape,
                    dtype=dtype,
                    device=plan.scratch_specs()[i].device,
                )
                for i, (shape, dtype) in enumerate(plan.shapes_and_dtypes())
            )
            cached = (plan, scratch)
            _SCRATCH_CACHE[key] = cached
        return cached

    def apply(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        apply_router_weight_on_input: bool = False,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the FP6 fused MoE for one layer; returns ``(M, K)`` bf16.

        ``output`` (zeroed, ``(M, K)`` bf16) is required under CUDA-graph
        capture: the kernel scatter-accumulates into it and refuses to
        allocate internally while a capture is active.
        """
        from b12x.moe import fused_moe

        m = int(hidden_states.shape[0])
        topk = int(topk_ids.shape[1])
        device = hidden_states.device
        router_on_input = bool(apply_router_weight_on_input)
        if router_on_input != self.apply_router_weight_on_input:
            raise ValueError(
                "apply_router_weight_on_input mismatch: method was constructed "
                f"with {self.apply_router_weight_on_input}, apply() got "
                f"{router_on_input}"
            )
        if output is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "B12X FP6 MoE requires a caller-owned output buffer "
                    "during CUDA graph capture"
                )
            output = torch.zeros(
                m,
                hidden_states.shape[1],
                dtype=hidden_states.dtype,
                device=device,
            )
        plan, scratch = self._plan_and_scratch(m, topk, device)
        binding = fused_moe.bind(
            plan,
            scratch=scratch,
            a=hidden_states,
            experts=self.experts_prepared,
            topk_weights=topk_weights,
            topk_ids=topk_ids.to(torch.int32),
            output=output,
            input_scales_static=self.input_scales_static,
        )
        return fused_moe.run(binding=binding)


class B12XFP6LinearMethod:
    """Reference dense-linear method backed by :func:`dense_fp6_linear`."""

    def __init__(self, weight):
        self.weight = weight

    def apply(
        self, x: torch.Tensor, *, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute ``y = x @ W.T`` in MX-FP6; returns ``(M, out_features)`` bf16."""
        from b12x.quantization.mxfp6 import dense_fp6_linear

        return dense_fp6_linear(x, self.weight, out=out)


def load_b12x_fp6_moe_methods(
    model_path: str,
    *,
    activation: str = "silu",
    device: torch.device | str = "cuda",
    limit_layers: Optional[int] = None,
) -> dict[int, B12XFP6MoEMethod]:
    """Load every routed-MoE layer as ``{layer_index: B12XFP6MoEMethod}``."""
    from b12x.moe import fused_moe
    from b12x.quantization.mxfp6 import load_fp6_moe_checkpoint

    layers = load_fp6_moe_checkpoint(
        model_path, activation=activation, device=device, limit_layers=limit_layers
    )
    out: dict[int, B12XFP6MoEMethod] = {}
    for layer_idx, weights in layers.items():
        kernel_src = kernel_source_format_for_moe(weights.source_format)
        weight_plan = get_fp6_moe_weight_plan(
            source_format=kernel_src,
            activation=weights.activation,
            num_experts=weights.num_experts,
            hidden_size=weights.k,
            intermediate_size=weights.n,
        )
        # Artifact blockscales are swizzled; prepare_weights wants unswizzled.
        from b12x._lib.fp6 import unswizzle_mxfp6_scales

        def _unswizzle_grid(swizzled: torch.Tensor, rows: int, blocks: int) -> torch.Tensor:
            return torch.stack(
                [
                    unswizzle_mxfp6_scales(swizzled[eid], rows, blocks)
                    for eid in range(swizzled.shape[0])
                ]
            ).contiguous()

        prepared = fused_moe.prepare_weights(
            plan=weight_plan,
            weights=fused_moe.PackedWeights(
                w13=weights.w1_fp6,
                w2=weights.w2_fp6,
                w13_block_scales=_unswizzle_grid(
                    weights.w1_blockscale, 2 * weights.n, weights.k // 32
                ),
                w2_block_scales=_unswizzle_grid(
                    weights.w2_blockscale, weights.k, weights.n // 32
                ),
                w13_global_scales=weights.w1_alphas,
                w2_global_scales=weights.w2_alphas,
                input_scale=weights.a1_gscale,
                intermediate_scale=weights.a2_gscale,
            ),
        )
        out[layer_idx] = B12XFP6MoEMethod(prepared, weight_plan)
    return out


def load_b12x_fp6_linear_methods(
    model_path: str,
    *,
    device: torch.device | str = "cuda",
) -> dict[str, B12XFP6LinearMethod]:
    """Load every FP6 dense linear as ``{module_name: B12XFP6LinearMethod}``."""
    from b12x.quantization.mxfp6 import load_fp6_dense_checkpoint

    weights = load_fp6_dense_checkpoint(model_path, device=device)
    return {name: B12XFP6LinearMethod(w) for name, w in weights.items()}
