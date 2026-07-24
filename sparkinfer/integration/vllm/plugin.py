"""Installable vLLM plugin that registers the sparkinfer MX-FP6 (W6A6/W6A8) quantization.

Wire through vLLM's ``general_plugins`` entry point (see README)::

    [project.entry-points."vllm.general_plugins"]
    sparkinfer_fp6 = "sparkinfer.integration.vllm.plugin:register_sparkinfer_fp6"

vLLM loads general plugins in *every* process — the front process, the
engine-core process, and each tensor-parallel worker — so registration survives
worker spawn under TP.

Flow
----
1. Entry point :func:`register_sparkinfer_fp6` runs in each process and calls
   ``register_quantization_config("sparkinfer_fp6")`` once (idempotent).
2. :class:`SparkInferFp6Config.override_quantization_method` claims a
   ``modelopt`` + ``quant_algo=W6A6`` checkpoint when ``SPARKINFER_ENABLE_FP6=1``,
   so the launch script does not strictly need ``--quantization sparkinfer_fp6``.
3. The checkpoint directory is resolved from ``SPARKINFER_FP6_MODEL_DIR`` (or
   legacy ``B12X_FP6_MODEL_DIR``) or from ``maybe_update_config(model_name=...)``.
4. ``get_quant_method`` dispatches each layer to the kernels in
   :mod:`sparkinfer.integration.vllm.fp6_serving`:
     * ``FusedMoE``  -> ``_VllmMoEMethod`` -> :class:`SparkInferFP6MoEMethod`
       -> ``sparkinfer.moe.fused_moe`` (``w6a8_mx``)
     * ``LinearBase``-> ``_VllmLinearMethod`` -> ``sparkinfer::fp6_dense_linear``

Weight binding
--------------
Both methods register **real** vLLM params in ``create_weights`` (mirroring
vLLM's own ModelOpt NVFP4 methods) so the framework loader places the packed
FP6 tensors itself.

The module imports without vLLM installed (all vLLM imports are deferred); the
``QuantizationConfig`` subclass is defined inside :func:`register_sparkinfer_fp6`.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import torch

from sparkinfer.integration.vllm.fp6_serving import (
    QUANT_ALGO,
    QUANT_METHOD,
    SparkInferFP6MoEMethod,
    get_fp6_moe_weight_plan,
    is_sparkinfer_fp6_enabled,
    kernel_source_format_for_moe,
)

MODEL_DIR_ENV = "SPARKINFER_FP6_MODEL_DIR"
LEGACY_MODEL_DIR_ENV = "B12X_FP6_MODEL_DIR"
QUANT_NAME = "sparkinfer_fp6"

# Unquantized bf16 linears with N <= MAX_OUT and K >= MIN_IN route through the
# sparkinfer small-N GEMV.  See sparkinfer.gemm.bf16_gemv for the thresholds.
from sparkinfer.gemm.bf16_gemv import (  # noqa: E402
    SMALL_N_GEMV_MAX_OUT,
    SMALL_N_GEMV_MIN_IN,
    is_disabled as _bf16_gemv_disabled,
    precompile as precompile_bf16_gemv_small_n,
)

# Fallback decode token counts warm-run per MoE layer at load time.
_MOE_WARM_DECODE_MS = (1, 2, 3, 4, 5, 8)
_MOE_WARM_MAX_M = 64

# Dense decode/verify token counts warm-run at load time.
_DENSE_WARM_DECODE_MS = (1, 2, 4, 5, 8, 16)
_DENSE_WARMED_SHAPES: set[tuple] = set()

# Fused vLLM module suffix -> ordered HF constituent projection names.
_FUSED_PARTS: dict[str, tuple[str, ...]] = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
    "in_proj_qkvz": ("in_proj_qkv", "in_proj_z"),
    "in_proj_ba": ("in_proj_b", "in_proj_a"),
}

_registered = False
_CONFIG_CLS: Optional[type] = None
logger = logging.getLogger("sparkinfer.vllm_fp6")


def _moe_warm_decode_ms() -> tuple[int, ...]:
    """Decode Ms to warm-run per MoE layer before CUDA-graph capture."""
    env = os.environ.get("SPARKINFER_MOE_WARM_MS") or os.environ.get(
        "B12X_MOE_WARM_MS", ""
    )
    env = env.strip()
    if env:
        try:
            ms = {int(v) for v in env.replace(",", " ").split()}
            return tuple(sorted(m for m in ms if 1 <= m <= _MOE_WARM_MAX_M))
        except ValueError:
            logger.warning("Ignoring malformed SPARKINFER_MOE_WARM_MS=%r", env)
    sizes = set(_MOE_WARM_DECODE_MS)
    try:
        from vllm.config import get_current_vllm_config

        cfg = get_current_vllm_config()
        cap = getattr(cfg.compilation_config, "cudagraph_capture_sizes", None)
        if cap:
            sizes.update(int(s) for s in cap)
    except Exception:
        logger.debug("vLLM config unavailable for MoE warm sizes", exc_info=True)
    return tuple(sorted(s for s in sizes if 1 <= s <= _MOE_WARM_MAX_M))


def _rebuild_sparkinfer_fp6_config(model_dir: Optional[str]) -> Any:
    register_sparkinfer_fp6()
    assert _CONFIG_CLS is not None
    return _CONFIG_CLS(model_dir)


def _resolve_model_dir(explicit: Optional[str] = None) -> Optional[str]:
    return (
        explicit
        or os.environ.get(MODEL_DIR_ENV)
        or os.environ.get(LEGACY_MODEL_DIR_ENV)
        or None
    )


def _norm_key(name: str) -> str:
    name = name.removeprefix("model.")
    name = name.replace("language_model.model.", "language_model.")
    return name


def register_sparkinfer_fp6() -> None:
    """vLLM ``general_plugins`` entry point: register ``sparkinfer_fp6``."""
    global _registered, _CONFIG_CLS, logger
    if _registered:
        return

    try:
        from vllm.logger import init_logger

        logger = init_logger("vllm.sparkinfer_fp6")
    except Exception:
        pass

    try:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoEMethodBase
    except ImportError:
        try:
            from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase
        except ImportError:
            from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
                FusedMoEMethodBase,
            )
    from vllm.model_executor.layers.linear import (
        LinearMethodBase,
        UnquantizedLinearMethod,
    )
    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
    from vllm.model_executor.parameter import (
        ModelWeightParameter,
        PerTensorScaleParameter,
    )

    class _VllmLinearMethod(LinearMethodBase):  # type: ignore[misc]
        def __init__(
            self,
            source_format: str,
            *,
            default_act_fmt: str,
            act_fmt_overrides: list[tuple[str, str]] | None = None,
        ):
            self.source_format = source_format
            self.default_act_fmt = default_act_fmt
            self.act_fmt_overrides = list(act_fmt_overrides or [])

        def create_weights(
            self,
            layer: Any,
            input_size_per_partition: int,
            output_partition_sizes: list[int],
            input_size: int,
            output_size: int,
            params_dtype: torch.dtype,
            **extra_weight_attrs: Any,
        ) -> None:
            del input_size, params_dtype
            weight_loader = extra_weight_attrs.get("weight_loader")
            out_total = sum(output_partition_sizes)
            in_p = int(input_size_per_partition)
            if in_p % 32 != 0:
                raise ValueError(
                    f"SparkInfer FP6 requires in_features % 32 == 0, got {in_p}"
                )
            layer.logical_widths = output_partition_sizes
            layer.sparkinfer_out_features_unsharded = int(output_size)

            weight = ModelWeightParameter(
                data=torch.empty(out_total, (3 * in_p) // 4, dtype=torch.uint8),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight", weight)

            weight_scale = ModelWeightParameter(
                data=torch.empty(out_total, in_p // 32, dtype=torch.uint8),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight_scale", weight_scale)

            n_shards = len(output_partition_sizes)
            weight_scale_2 = PerTensorScaleParameter(
                data=torch.ones(n_shards, dtype=torch.float32),
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight_scale_2", weight_scale_2)
            input_scale = PerTensorScaleParameter(
                data=torch.ones(n_shards, dtype=torch.float32),
                weight_loader=weight_loader,
            )
            layer.register_parameter("input_scale", input_scale)

        def process_weights_after_loading(self, layer: Any) -> None:
            from sparkinfer.quantization.mxfp6.fp6_checkpoint import (
                resolve_activation_format,
            )
            from sparkinfer.quantization.mxfp6.fp6_dense_weights import PACKED_GEMM_MIN_N

            ws2 = layer.weight_scale_2.data.float()
            if not torch.allclose(ws2, torch.ones_like(ws2), atol=1e-3):
                raise NotImplementedError(
                    "SparkInfer FP6 vLLM path requires unit weight_scale_2 "
                    "(pure-MX W6A6 contract)."
                )
            prefix = str(getattr(layer, "prefix", "") or "")
            act_fmt = resolve_activation_format(
                prefix,
                default_fmt=self.default_act_fmt,
                overrides=self.act_fmt_overrides,
            )
            w = _dense_weight_from_loaded(
                layer.weight.data,
                layer.weight_scale.data,
                self.source_format,
                act_fmt=act_fmt,
                out_features_unsharded=getattr(
                    layer, "sparkinfer_out_features_unsharded", 0
                ),
            )
            dev, dt = layer.weight.device, layer.weight.dtype
            gemm_w = w.gemm_weight()
            if w.use_packed_gemm and w.out_features < PACKED_GEMM_MIN_N:
                logger.info(
                    "SparkInfer FP6: %s kept packed-B (per-GPU N=%d < %d, "
                    "full N=%d >= threshold)",
                    prefix,
                    w.out_features,
                    PACKED_GEMM_MIN_N,
                    w.out_features_unsharded,
                )
            if not w.use_packed_gemm:
                w.packed = torch.empty(0, dtype=torch.uint8, device=dev)
            layer.sparkinfer_fp6_weight = w
            layer.weight.data = torch.empty(0, dtype=dt, device=dev)
            layer.weight_scale.data = torch.empty(
                0, dtype=layer.weight_scale.dtype, device=dev
            )
            import sparkinfer.quantization.mxfp6.fp6_dense_op  # noqa: F401

            layer.sparkinfer_fp6_gemm_weight = gemm_w
            layer.sparkinfer_fp6_scales = w.scale_storage
            layer.sparkinfer_fp6_gscale = w.global_scale
            layer.sparkinfer_fp6_fmt = w.fmt
            layer.sparkinfer_fp6_act_fmt = w.act_fmt
            layer.sparkinfer_fp6_out_features = w.out_features
            layer.sparkinfer_fp6_in_features = w.in_features
            if self.act_fmt_overrides and act_fmt != self.default_act_fmt:
                logger.info(
                    "SparkInfer FP6: act_fmt override %s -> %s (default %s)",
                    prefix,
                    act_fmt,
                    self.default_act_fmt,
                )
            self._warm_dense_decode(layer)

        def _warm_dense_decode(self, layer: Any) -> None:
            gemm_w = layer.sparkinfer_fp6_gemm_weight
            if not gemm_w.is_cuda:
                return
            key = (
                layer.sparkinfer_fp6_out_features,
                layer.sparkinfer_fp6_in_features,
                layer.sparkinfer_fp6_fmt,
                layer.sparkinfer_fp6_act_fmt,
                tuple(gemm_w.shape),
            )
            if key in _DENSE_WARMED_SHAPES:
                return
            _DENSE_WARMED_SHAPES.add(key)
            for m in _DENSE_WARM_DECODE_MS:
                x = torch.zeros(
                    m,
                    layer.sparkinfer_fp6_in_features,
                    dtype=torch.bfloat16,
                    device=gemm_w.device,
                )
                self.apply(layer, x)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        def apply(
            self, layer: Any, x: torch.Tensor, bias: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            x2d = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x
            y = torch.ops.sparkinfer.fp6_dense_linear(
                x2d.to(torch.bfloat16),
                layer.sparkinfer_fp6_gemm_weight,
                layer.sparkinfer_fp6_scales,
                layer.sparkinfer_fp6_gscale,
                layer.sparkinfer_fp6_fmt,
                layer.sparkinfer_fp6_out_features,
                layer.sparkinfer_fp6_in_features,
                layer.sparkinfer_fp6_act_fmt,
            )
            if x.dim() > 2:
                y = y.reshape(*x.shape[:-1], y.shape[-1])
            return y + bias if bias is not None else y

    import vllm.model_executor.layers.linear as _vllm_linear

    _register_v2 = getattr(
        _vllm_linear, "register_weight_loader_v2_supported_method", None
    )
    if _register_v2 is not None:
        _register_v2(_VllmLinearMethod)
    elif _VllmLinearMethod.__name__ not in _vllm_linear.WEIGHT_LOADER_V2_SUPPORTED:
        _vllm_linear.WEIGHT_LOADER_V2_SUPPORTED.append(_VllmLinearMethod.__name__)

    class _VllmSmallNBF16Method(UnquantizedLinearMethod):  # type: ignore[misc]
        def process_weights_after_loading(self, layer: Any) -> None:
            super().process_weights_after_loading(layer)
            import sparkinfer.gemm.bf16_gemv  # noqa: F401

            w = layer.weight
            if w.dim() == 2 and w.is_cuda:
                layer.sparkinfer_gemv_weight = w.data.detach().clone().contiguous()
                precompile_bf16_gemv_small_n(layer.sparkinfer_gemv_weight, log=logger)

        def apply(
            self, layer: Any, x: torch.Tensor, bias: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            w = getattr(layer, "sparkinfer_gemv_weight", None)
            if (
                w is not None
                and bias is None
                and x.dtype == torch.bfloat16
                and w.dtype == torch.bfloat16
            ):
                x2d = x.reshape(-1, x.shape[-1])
                y = torch.ops.sparkinfer.bf16_gemv_small_n(x2d, w)
                return y.reshape(*x.shape[:-1], w.shape[0])
            return super().apply(layer, x, bias)

    try:
        from vllm.model_executor.layers.fused_moe import FusedMoeWeightScaleSupported
    except ImportError:
        from vllm.model_executor.layers.fused_moe.routed_experts import (
            FusedMoeWeightScaleSupported,
        )
    from vllm.model_executor.utils import set_weight_attrs

    class _VllmMoEMethod(FusedMoEMethodBase):  # type: ignore[misc]
        _OUT_BUF_MAX_M = 512

        def __init__(self, moe_config: Any, source_format: str):
            super().__init__(moe_config)
            self.source_format = source_format
            self.core: Optional[SparkInferFP6MoEMethod] = None
            self._out_bufs: dict[int, torch.Tensor] = {}

        def _output_for(self, x: torch.Tensor) -> Optional[torch.Tensor]:
            m = int(x.shape[0])
            if m > self._OUT_BUF_MAX_M:
                if not torch.cuda.is_current_stream_capturing():
                    return None
                return torch.zeros(m, x.shape[1], dtype=x.dtype, device=x.device)
            buf = self._out_bufs.get(m)
            if buf is None:
                buf = torch.zeros(m, x.shape[1], dtype=x.dtype, device=x.device)
                self._out_bufs[m] = buf
            else:
                buf.zero_()
            return buf

        def create_weights(
            self,
            layer: Any,
            num_experts: int,
            hidden_size: int,
            intermediate_size_per_partition: int,
            params_dtype: torch.dtype,
            **extra_weight_attrs: Any,
        ) -> None:
            del params_dtype
            weight_loader = extra_weight_attrs.get("weight_loader")
            e = int(num_experts)
            k = int(hidden_size)
            n = int(intermediate_size_per_partition)
            if k % 32 != 0 or n % 32 != 0:
                raise ValueError(
                    f"SparkInfer FP6 MoE requires hidden/intermediate % 32 == 0, "
                    f"got K={k}, N={n}"
                )
            if not getattr(self.moe, "is_act_and_mul", True):
                raise NotImplementedError(
                    "SparkInfer FP6 MoE supports gated (act-and-mul) experts only"
                )

            def _param(shape: tuple[int, ...], dtype: torch.dtype, **attrs: Any):
                p = torch.nn.Parameter(
                    torch.empty(*shape, dtype=dtype), requires_grad=False
                )
                set_weight_attrs(p, {"weight_loader": weight_loader, **attrs})
                return p

            group = FusedMoeWeightScaleSupported.GROUP.value
            tensor = FusedMoeWeightScaleSupported.TENSOR.value
            layer.register_parameter(
                "w13_weight", _param((e, 2 * n, (3 * k) // 4), torch.uint8)
            )
            layer.register_parameter(
                "w2_weight", _param((e, k, (3 * n) // 4), torch.uint8)
            )
            layer.register_parameter(
                "w13_weight_scale",
                _param((e, 2 * n, k // 32), torch.uint8, quant_method=group),
            )
            layer.register_parameter(
                "w2_weight_scale",
                _param((e, k, n // 32), torch.uint8, quant_method=group),
            )
            layer.register_parameter(
                "w13_weight_scale_2",
                _param((e, 2), torch.float32, quant_method=tensor),
            )
            layer.register_parameter(
                "w2_weight_scale_2",
                _param((e,), torch.float32, quant_method=tensor),
            )
            layer.register_parameter("w13_input_scale", _param((e,), torch.float32))
            layer.register_parameter("w2_input_scale", _param((e,), torch.float32))

        def uses_weight_scale_2_pattern(self) -> bool:
            return True

        def get_fused_moe_quant_config(self, layer: Any) -> Any:
            return None

        def process_weights_after_loading(self, layer: Any) -> None:
            from sparkinfer.moe import fused_moe

            for name in (
                "w13_weight_scale_2",
                "w2_weight_scale_2",
                "w13_input_scale",
                "w2_input_scale",
            ):
                t = getattr(layer, name).data.float()
                if not torch.allclose(t, torch.ones_like(t), atol=1e-3):
                    raise NotImplementedError(
                        f"SparkInfer FP6 MoE requires unit {name} "
                        "(pure-MX W6A6 contract)."
                    )

            act = getattr(layer, "activation", None) or getattr(
                self.moe, "activation", "silu"
            )
            activation = str(getattr(act, "value", act)).lower()
            e = int(layer.w13_weight.shape[0])
            k = int(layer.w2_weight.shape[1])
            n = int(layer.w13_weight.shape[1]) // 2
            dev = layer.w13_weight.device

            def _swap_halves(t: torch.Tensor) -> torch.Tensor:
                gate = t[:, :n].clone()
                t[:, :n] = t[:, n:]
                t[:, n:] = gate
                return t

            w1_fp6 = _swap_halves(layer.w13_weight.data)
            w1_scale = _swap_halves(layer.w13_weight_scale.data)
            w2_fp6 = layer.w2_weight.data
            w2_scale = layer.w2_weight_scale.data

            ones_e = torch.ones(e, dtype=torch.float32, device=dev)
            ones_1 = torch.ones(1, dtype=torch.float32, device=dev)
            kernel_src = kernel_source_format_for_moe(self.source_format)
            # Shared per-geometry plan object: makes the process-wide scratch
            # cache in fp6_serving actually shared across all MoE layers.
            weight_plan = get_fp6_moe_weight_plan(
                source_format=kernel_src,
                activation=activation,
                num_experts=e,
                hidden_size=k,
                intermediate_size=n,
            )
            prepared = fused_moe.prepare_weights(
                plan=weight_plan,
                w1_fp4=w1_fp6,
                w1_blockscale=w1_scale,
                w1_global_scale=ones_e,
                a1_gscale=ones_1,
                w2_fp4=w2_fp6,
                w2_blockscale=w2_scale,
                w2_global_scale=ones_e.clone(),
                a2_gscale=ones_1.clone(),
                params_dtype=torch.bfloat16,
            )

            for name in (
                "w13_weight",
                "w2_weight",
                "w13_weight_scale",
                "w2_weight_scale",
            ):
                p = getattr(layer, name)
                p.data = torch.empty(0, dtype=p.dtype, device=dev)
            if dev.type == "cuda":
                torch.cuda.empty_cache()

            self.core = SparkInferFP6MoEMethod(
                prepared,
                weight_plan,
                apply_router_weight_on_input=bool(
                    getattr(layer, "apply_router_weight_on_input", False)
                ),
            )
            if dev.type == "cuda":
                topk = int(getattr(self.moe, "experts_per_token", 0) or 8)
                router_on_input = bool(
                    getattr(layer, "apply_router_weight_on_input", False)
                )
                for m in _moe_warm_decode_ms():
                    x = torch.zeros(m, k, dtype=torch.bfloat16, device=dev)
                    ids = (
                        torch.arange(m * topk, dtype=torch.int32, device=dev)
                        .remainder(e)
                        .reshape(m, topk)
                    )
                    w = torch.full(
                        (m, topk), 1.0 / topk, dtype=torch.float32, device=dev
                    )
                    self.core.apply(
                        x,
                        w,
                        ids,
                        apply_router_weight_on_input=router_on_input,
                        output=self._output_for(x),
                    )
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        def apply(
            self,
            layer: Any,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            shared_experts: Any = None,
            shared_experts_input: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            del shared_experts, shared_experts_input
            assert self.core is not None, "process_weights_after_loading not run"
            return self.core.apply(
                x,
                topk_weights,
                topk_ids.to(torch.int32),
                apply_router_weight_on_input=bool(
                    getattr(layer, "apply_router_weight_on_input", False)
                ),
                output=self._output_for(x),
            )

    @register_quantization_config(QUANT_NAME)
    class SparkInferFp6Config(QuantizationConfig):  # type: ignore[misc]
        def __init__(self, model_dir: Optional[str] = None) -> None:
            super().__init__()
            self._model_dir = _resolve_model_dir(model_dir)
            self._fp6_modules: set[str] = set()
            self._source_format = "mxfp6_default"
            self._linear_method: Optional[Any] = None
            self._loaded = False
            self._match_hits = 0
            self._match_miss: list[str] = []
            self._miss_count = 0
            self._warned_zero_overlap = False

        def __reduce__(self):
            return (_rebuild_sparkinfer_fp6_config, (self._model_dir,))

        @classmethod
        def get_name(cls) -> str:
            return QUANT_NAME

        @classmethod
        def get_supported_act_dtypes(cls) -> list[torch.dtype]:
            return [torch.bfloat16]

        @classmethod
        def get_min_capability(cls) -> int:
            return 100

        @staticmethod
        def get_config_filenames() -> list[str]:
            return []

        @classmethod
        def from_config(cls, config: dict[str, Any]) -> "SparkInferFp6Config":
            return cls()

        @classmethod
        def override_quantization_method(
            cls,
            hf_quant_cfg: dict[str, Any],
            user_quant: Optional[str],
            hf_config: Any = None,
        ) -> Optional[str]:
            del user_quant, hf_config
            if not hf_quant_cfg:
                return None
            if not is_sparkinfer_fp6_enabled():
                return None
            method = str(hf_quant_cfg.get("quant_method", "")).lower()
            algo = str(hf_quant_cfg.get("quant_algo", "")).upper()
            if method == QUANT_METHOD and algo == QUANT_ALGO:
                return QUANT_NAME
            return None

        def maybe_update_config(
            self, model_name: str, hf_config: Any = None, revision: Any = None
        ) -> None:
            del hf_config, revision
            if self._model_dir is None and model_name:
                self._model_dir = model_name

        def _ensure_loaded(self) -> None:
            if self._loaded:
                return
            model_dir = _resolve_model_dir(self._model_dir)
            if not model_dir:
                raise RuntimeError(
                    "SparkInfer FP6: model directory unknown. Set "
                    "SPARKINFER_FP6_MODEL_DIR to the FP6 checkpoint path."
                )
            from sparkinfer.quantization.mxfp6.fp6_checkpoint import WEIGHT_SCALE_SUFFIX
            from sparkinfer.quantization.mxfp6.fp6_safetensors_load import (
                _source_format_from_config,
            )
            from sparkinfer.quantization.mxfp6.model_fp6 import SafetensorsModel
            from sparkinfer.quantization.mxfp6.fp6_checkpoint import (
                activation_format_for_source,
                load_act_fmt_overrides,
            )

            st = SafetensorsModel(model_dir)
            qcfg = st.config.get("quantization_config", {}) or {}
            self._source_format = _source_format_from_config(qcfg)
            self._default_act_fmt = activation_format_for_source(self._source_format)
            self._act_fmt_overrides = load_act_fmt_overrides(qcfg)
            self._fp6_modules = {
                _norm_key(k[: -len(WEIGHT_SCALE_SUFFIX)])
                for k in st.keys()
                if k.endswith(WEIGHT_SCALE_SUFFIX)
            }
            self._linear_method = _VllmLinearMethod(
                self._source_format,
                default_act_fmt=self._default_act_fmt,
                act_fmt_overrides=self._act_fmt_overrides,
            )
            self._loaded = True
            logger.info(
                "SparkInfer FP6: %d FP6 modules discovered in %s "
                "(source_format=%s act_fmt=%s overrides=%d)",
                len(self._fp6_modules),
                model_dir,
                self._source_format,
                self._default_act_fmt,
                len(self._act_fmt_overrides),
            )

        def _is_fp6_linear(self, prefix: str) -> bool:
            if _norm_key(prefix) in self._fp6_modules:
                return True
            parts = _FUSED_PARTS.get(prefix.rsplit(".", 1)[-1])
            if parts is None:
                return False
            parent = _norm_key(prefix.rsplit(".", 1)[0])
            return all(f"{parent}.{p}" in self._fp6_modules for p in parts)

        def _has_fp6_experts(self, prefix: str) -> bool:
            root = _norm_key(prefix).removesuffix(".routed_experts") + "."
            return any(m.startswith(root) for m in self._fp6_modules)

        def _maybe_warn_zero_overlap(self, prefix: str) -> None:
            del prefix
            if self._warned_zero_overlap or self._match_hits:
                return
            if self._miss_count < 200:
                return
            self._warned_zero_overlap = True
            logger.warning(
                "SparkInfer FP6: %d layers checked and NONE matched the FP6 index "
                "(%d modules from %s). SPARKINFER_FP6_MODEL_DIR likely points at a "
                "different model's checkpoint (first miss: %s).",
                self._miss_count,
                len(self._fp6_modules),
                _resolve_model_dir(self._model_dir),
                self._match_miss[0] if self._match_miss else "?",
            )

        def get_quant_method(self, layer: Any, prefix: str):
            from vllm.model_executor.layers.linear import (
                LinearBase,
                UnquantizedLinearMethod,
            )
            from vllm.model_executor.layers.vocab_parallel_embedding import (
                ParallelLMHead,
            )

            self._ensure_loaded()

            _moe_types: list[type] = []
            try:
                from vllm.model_executor.layers.fused_moe.routed_experts import (
                    RoutedExperts,
                )

                _moe_types.append(RoutedExperts)
            except ImportError:
                pass
            try:
                from vllm.model_executor.layers.fused_moe import FusedMoE

                if isinstance(FusedMoE, type):
                    _moe_types.append(FusedMoE)
            except ImportError:
                pass
            is_moe = bool(_moe_types) and isinstance(layer, tuple(_moe_types))

            if is_moe:
                if self._has_fp6_experts(prefix):
                    logger.info("SparkInfer FP6: bound FP6 MoE %s", prefix)
                    return _VllmMoEMethod(layer.moe_config, self._source_format)
                logger.info("SparkInfer FP6: vLLM-native MoE fallback for %s", prefix)
                return None

            if isinstance(layer, (LinearBase, ParallelLMHead)):
                if self._is_fp6_linear(prefix):
                    self._match_hits += 1
                    if self._match_hits <= 8:
                        logger.info("SparkInfer FP6: bound FP6 linear %s", prefix)
                    return self._linear_method
                self._miss_count += 1
                if len(self._match_miss) < 24:
                    self._match_miss.append(prefix)
                    logger.info("SparkInfer FP6: BF16 fallback for %s", prefix)
                self._maybe_warn_zero_overlap(prefix)
                out_size = int(getattr(layer, "output_size", 0) or 0)
                in_size = int(getattr(layer, "input_size", 0) or 0)
                if (
                    not _bf16_gemv_disabled()
                    and 0 < out_size <= SMALL_N_GEMV_MAX_OUT
                    and in_size >= SMALL_N_GEMV_MIN_IN
                ):
                    logger.debug(
                        "SparkInfer FP6: small-N bf16 GEMV for %s (N=%d, K=%d)",
                        prefix,
                        out_size,
                        in_size,
                    )
                    return _VllmSmallNBF16Method()
                return UnquantizedLinearMethod()

            return None

    _CONFIG_CLS = SparkInferFp6Config
    _registered = True


# Backward-compatible alias for forks still wired to the old entry-point name.
register_b12x_fp6 = register_sparkinfer_fp6


def _dense_weight_from_loaded(
    weight_u8: torch.Tensor,
    scale_u8: torch.Tensor,
    source_format: str,
    *,
    act_fmt: str | None = None,
    out_features_unsharded: int = 0,
):
    from sparkinfer._lib.intrinsics import swizzle_block_scale
    from sparkinfer.quantization.mxfp6.fp6_checkpoint import (
        activation_format_for_source,
        weight_format_for_source,
    )
    from sparkinfer.quantization.mxfp6.fp6_dense_weights import FP6DenseWeight

    out_f = int(weight_u8.shape[0])
    in_f = int(weight_u8.shape[1]) * 4 // 3
    scale_storage = (
        swizzle_block_scale(scale_u8.view(torch.float8_e8m0fnu))
        .reshape(-1)
        .view(torch.uint8)
        .contiguous()
    )
    return FP6DenseWeight(
        packed=weight_u8.contiguous(),
        scale_storage=scale_storage,
        global_scale=torch.ones(1, dtype=torch.float32, device=weight_u8.device),
        out_features=out_f,
        in_features=in_f,
        fmt=weight_format_for_source(source_format),
        act_fmt=(
            act_fmt
            if act_fmt is not None
            else activation_format_for_source(source_format)
        ),
        out_features_unsharded=out_features_unsharded,
    )
