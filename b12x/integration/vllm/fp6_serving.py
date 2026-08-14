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
* prepared experts from :func:`b12x.moe.fused_moe.prepare_weights` with
  ``quant_mode="w6a8_mx"`` / ``source_format="mxfp6_e2m3"`` and FC1 rows in
  ``[up; gate]`` (``w13_layout="w13"``)
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

import contextlib
import os
import weakref
from typing import Any, Optional

import torch

ENABLE_ENV = "B12X_ENABLE_FP6"
QUANT_METHOD = "modelopt"
QUANT_ALGO = "W6A6"


class CaptureOutputCache:
    """Bounded output buffer cache keyed by the declared capture-size catalog.

    Only catalogued M values (CUDA-graph capture sizes) get retained
    ``(M, K)`` BF16 output buffers with stable addresses.  Uncatalogued
    eager M returns ``None`` (caller allocates transiently).  Uncatalogued
    M during CUDA-graph capture raises before any allocation.

    Owner invariants (dtype, device, hidden_size) are fixed at construction
    and validated on every ``get`` to prevent cross-model/cross-config aliasing.
    """

    def __init__(
        self,
        catalog: tuple[int, ...],
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self._catalog = frozenset(int(m) for m in catalog)
        self._hidden_size = int(hidden_size)
        self._dtype = dtype
        self._device = device
        self._bufs: dict[int, torch.Tensor] = {}

    @property
    def catalog(self) -> frozenset[int]:
        return self._catalog

    def get(
        self, m: int, dtype: torch.dtype, device: torch.device
    ) -> Optional[torch.Tensor]:
        if dtype != self._dtype or device != self._device:
            raise RuntimeError(
                f"B12X FP6 MoE: output cache dtype/device mismatch: "
                f"got ({dtype}, {device}), expected ({self._dtype}, {self._device})"
            )
        if m not in self._catalog:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"B12X FP6 MoE: uncatalogued M={m} during CUDA-graph "
                    "capture — output buffer not in capture-size catalog"
                )
            return None  # eager transient
        buf = self._bufs.get(m)
        if buf is None:
            buf = torch.zeros(
                m, self._hidden_size, dtype=self._dtype, device=self._device
            )
            self._bufs[m] = buf
        else:
            buf.zero_()
        return buf

    def clear(self) -> None:
        self._bufs.clear()

    def __len__(self) -> int:
        return len(self._bufs)


class ServingScratchArena:
    """Model-owned scratch cache bounded by the declared capture catalog.

    One arena instance is shared across ALL MoE layers of a model (layers
    run sequentially on one stream, so scratch reuse is safe).  Only
    catalogued M values get retained scratch with stable addresses (needed
    for CUDA-graph capture).  Uncatalogued eager M computes transient
    scratch that is never stored.  Uncatalogued capture M is rejected
    before allocation.

    The scratch key includes the full semantic geometry (source_format,
    activation, E, K, N, topk, device, router_on_input) so different model
    geometries never alias, and plan re-creation does not split generations.
    """

    def __init__(
        self,
        catalog: tuple[int, ...],
        weight_plan: Any,
        topk: int,
        device: torch.device,
        *,
        source_format: str = "",
        activation: str = "",
        num_experts: int = 0,
        hidden_size: int = 0,
        intermediate_size: int = 0,
        apply_router_weight_on_input: bool = False,
    ):
        self._catalog = frozenset(int(m) for m in catalog)
        self._weight_plan = weight_plan
        self._topk = int(topk)
        self._device = device
        self._router_on_input = bool(apply_router_weight_on_input)
        # Full semantic geometry key — stable across plan re-creation.
        self._geometry = (
            source_format,
            activation,
            int(num_experts),
            int(hidden_size),
            int(intermediate_size),
            self._topk,
            str(device),
            self._router_on_input,
        )
        self._scratch: dict[tuple, tuple[Any, tuple[torch.Tensor, ...]]] = {}

    def validate_geometry(
        self, *, weight_plan: Any, topk: int, device: torch.device,
        source_format: str, activation: str, num_experts: int,
        hidden_size: int, intermediate_size: int,
        apply_router_weight_on_input: bool,
    ) -> None:
        """Raise RuntimeError if caller geometry differs from arena geometry."""
        other = (
            source_format, activation, int(num_experts),
            int(hidden_size), int(intermediate_size),
            int(topk), str(device), bool(apply_router_weight_on_input),
        )
        if other != self._geometry:
            raise RuntimeError(
                f"B12X FP6 MoE: geometry mismatch — arena has "
                f"{self._geometry} but layer has {other}. All MoE layers "
                "of a model must share identical geometry."
            )

    @property
    def catalog(self) -> frozenset[int]:
        return self._catalog

    @property
    def weight_plan(self) -> Any:
        return self._weight_plan

    def _key(self, m: int) -> tuple:
        return (int(m), self._geometry)

    def get_or_build(
        self, m: int
    ) -> tuple[Any, tuple[torch.Tensor, ...]]:
        """Return retained scratch for catalogued M, transient for uncatalogued eager.

        Raises RuntimeError for uncatalogued M during CUDA-graph capture.
        """
        if m not in self._catalog:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"B12X FP6 MoE: uncatalogued M={m} during CUDA-graph "
                    "capture — scratch not in capture-size catalog"
                )
            # Uncatalogued eager: build transient, do not retain.
            return self._build(m)
        key = self._key(m)
        cached = self._scratch.get(key)
        if cached is not None:
            return cached
        plan, scratch = self._build(m)
        self._scratch[key] = (plan, scratch)
        return plan, scratch

    def _build(self, m: int) -> tuple[Any, tuple[torch.Tensor, ...]]:
        from b12x.moe import fused_moe

        plan = fused_moe.plan(
            fused_moe.Caps(
                max_tokens=m,
                num_topk=self._topk,
                device=self._device,
                weight_plan=self._weight_plan,
                core_token_counts=(m,),
                route_num_experts=0,
                quant_mode="w6a8_mx",
                apply_router_weight_on_input=self._router_on_input,
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
        return plan, scratch

    def clear(self) -> None:
        """Drop all retained scratch references.

        Safe only after all CUDA graphs referencing this arena's tensors
        have been destroyed.
        """
        self._scratch.clear()

    def __len__(self) -> int:
        return len(self._scratch)


# Deduped fused_moe weight plans, keyed by geometry.  Uses WeakValueDictionary
# so plans are garbage-collected when no live arena references them — no
# historical accumulation.  Plans are tensor-free metadata; the weak ref
# prevents stale geometries from accumulating across model churn.
_WEIGHT_PLAN_CACHE: "weakref.WeakValueDictionary[tuple, Any]" = (
    weakref.WeakValueDictionary()
)


def reset_serving_caches() -> None:
    """Drop all process-wide weight-plan references.

    Safe only after **all** live models' CUDA graphs have been destroyed.
    Per-model scratch is owned by :class:`ServingScratchArena` instances and
    cleared via their ``clear()`` method; this function only clears the
    tensor-free plan cache.
    """
    _WEIGHT_PLAN_CACHE.clear()


def capture_sizes_from_config(compilation_config: Any) -> tuple[int, ...]:
    """Extract cudagraph_capture_sizes from a vLLM compilation config.

    Handles both object and dict config forms.  Returns all positive declared
    sizes (no upper cap — vLLM can declare capture sizes above 512).
    Returns ``()`` if graph capture is disabled (``None`` or empty).
    Raises ``RuntimeError`` if the attribute exists but has an unsupported type,
    so discovery problems surface at model load, not during capture.
    """
    cap: Any = None
    if isinstance(compilation_config, dict):
        cap = compilation_config.get("cudagraph_capture_sizes")
    else:
        cap = getattr(compilation_config, "cudagraph_capture_sizes", None)
    if cap is None:
        return ()  # graph capture disabled
    try:
        sizes = tuple(int(s) for s in cap)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"B12X FP6: cudagraph_capture_sizes has unsupported type/value: "
            f"{cap!r} ({exc})"
        ) from exc
    return tuple(sorted(s for s in sizes if s > 0))


def get_fp6_moe_weight_plan(
    *,
    source_format: str,
    activation: str,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> Any:
    """One shared ``fused_moe.plan_weights`` result per (geometry, activation).

    The plan is cached weakly so it is GC'd when no live arena holds a
    strong reference, preventing historical geometry accumulation.
    """
    from b12x.moe import fused_moe

    key = (
        source_format,
        activation,
        int(num_experts),
        int(hidden_size),
        int(intermediate_size),
    )
    plan = _WEIGHT_PLAN_CACHE.get(key)
    if plan is not None:
        return plan
    plan = fused_moe.plan_weights(
        quant_modes="w6a8_mx",
        source_format=source_format,
        activation=activation,
        params_dtype=torch.bfloat16,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        w13_layout="w13",  # [up; gate] FC1 rows (the only w6a8_mx layout)
    )
    with contextlib.suppress(TypeError):
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

    Holds one layer's prepared experts and weight plan.  Scratch/plan tensors
    come from the owner's :class:`ServingScratchArena` (if provided) or are
    transient (if no arena — standalone eager path).

    Scratch retention is bounded by the arena's capture-size catalog.  Only
    catalogued M values get retained scratch with stable addresses.  Uncatalogued
    eager M computes transient scratch that is never stored.  Uncatalogued M
    during CUDA-graph capture is rejected before any cache lookup or allocation.
    """

    def __init__(
        self,
        experts_prepared: Any,
        weight_plan: Any,
        *,
        input_scales_static: bool = True,
        apply_router_weight_on_input: bool = False,
        arena: Optional[ServingScratchArena] = None,
    ):
        self.experts_prepared = experts_prepared
        self.weight_plan = weight_plan
        self.input_scales_static = input_scales_static
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.arena = arena

    def update_experts(self, new_prepared: Any) -> None:
        """Copy new prepared weight values into existing storage (reload-safe).

        vLLM's ``reload_weights()`` does NOT destroy CUDA graphs, so
        graph-captured weight pointers must stay stable.  This copies the
        tensor data from ``new_prepared`` into the existing
        ``self.experts_prepared`` storage, preserving every ``data_ptr``
        that a captured graph may reference.

        Raises RuntimeError if shapes/dtypes/devices don't match (topology
        change requires graph destruction and recapture, not in-place copy).
        """
        old = self.experts_prepared
        # The prepared object is opaque (from fused_moe.prepare_weights).
        # Copy every tensor attribute from new into old, preserving data_ptr.
        for attr_name in dir(new_prepared):
            if attr_name.startswith("_"):
                continue
            new_val = getattr(new_prepared, attr_name, None)
            old_val = getattr(old, attr_name, None)
            if not isinstance(new_val, torch.Tensor):
                continue
            if not isinstance(old_val, torch.Tensor):
                raise RuntimeError(
                    f"B12X FP6 MoE: reload attribute '{attr_name}' is a "
                    f"tensor in new prepared but not in old — topology "
                    f"changed, cannot copy in-place. Destroy CUDA graphs "
                    f"and call unload_serving() before reloading."
                )
            if (
                new_val.shape != old_val.shape
                or new_val.dtype != old_val.dtype
                or new_val.device != old_val.device
            ):
                raise RuntimeError(
                    f"B12X FP6 MoE: reload attribute '{attr_name}' has "
                    f"shape/dtype/device mismatch: "
                    f"new={tuple(new_val.shape)},{new_val.dtype},"
                    f"{new_val.device} vs "
                    f"old={tuple(old_val.shape)},{old_val.dtype},"
                    f"{old_val.device}. Topology changed — destroy CUDA "
                    f"graphs and call unload_serving() before reloading."
                )
            old_val.copy_(new_val)

    def _plan_and_scratch(
        self, m: int, topk: int, device: torch.device
    ) -> tuple[Any, tuple[torch.Tensor, ...]]:
        if self.arena is not None:
            return self.arena.get_or_build(m)
        # No arena (standalone eager path): always transient, never retained.
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"B12X FP6 MoE: uncatalogued M={m} during CUDA-graph "
                "capture — no scratch arena configured"
            )
        from b12x.moe import fused_moe

        plan = fused_moe.plan(
            fused_moe.Caps(
                max_tokens=m,
                num_topk=topk,
                device=device,
                weight_plan=self.weight_plan,
                core_token_counts=(m,),
                route_num_experts=0,
                quant_mode="w6a8_mx",
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
        return plan, scratch

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
        # Check catalog membership and capture state BEFORE cache lookup or
        # allocation.  A direct caller supplying a caller-owned output during
        # capture must not reach _build_scratch for an uncatalogued M.
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
        # _plan_and_scratch checks capture/catalog and raises for uncatalogued
        # capture before any allocation.
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
            w1_fp4=weights.w1_fp6,
            w1_blockscale=_unswizzle_grid(
                weights.w1_blockscale, 2 * weights.n, weights.k // 32
            ),
            w1_global_scale=weights.w1_alphas,
            a1_gscale=weights.a1_gscale,
            w2_fp4=weights.w2_fp6,
            w2_blockscale=_unswizzle_grid(
                weights.w2_blockscale, weights.k, weights.n // 32
            ),
            w2_global_scale=weights.w2_alphas,
            a2_gscale=weights.a2_gscale,
            params_dtype=torch.bfloat16,
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
