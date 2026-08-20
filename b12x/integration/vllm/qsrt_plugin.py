"""vLLM quantization plugin for TP1 Qwen3.8 dense-MLP QSRT K5.

The plugin claims only checkpoint modules enumerated by the
``b12x_qsrt`` quantization configuration. It binds each removed BF16 dense-MLP
matrix to a native K5 trellis payload and BF16 rank-16 correction. No decoded
weight or unquantized fallback exists in the claimed linear method.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import torch


QUANT_NAME = "b12x_qsrt"
MODEL_DIR_ENV = "B12X_QSRT_MODEL_DIR"
MANIFEST_FILENAME = "b12x-qsrt-manifest.json"
CHECKPOINT_KIND = "qwen38_dense_mlp_qsrt_k5_rank16_checkpoint_v1"
DESCRIPTOR_SHA256 = "17cf4ca9ef1e3a07c3354c12f7ac887b4e081b1668bea61eb37d8f2b410bb968"
RANK = 16
BITS = 5
_DEFAULT_CAPTURE_MS = (1, 2, 4, 8, 16)
_COMPONENT_DTYPES = {
    "trellis": torch.int16,
    "suh": torch.float16,
    "svh": torch.float16,
    "a": torch.bfloat16,
    "b": torch.bfloat16,
}

_registered = False
_CONFIG_CLS: Optional[type] = None
_WARMED_GEOMETRIES: set[tuple[int, int, int, int, int, int]] = set()
_SHARED_BUFFERS: dict[tuple[int, int, int, int], Any] = {}
_SHARED_PAIR_BUFFERS: dict[tuple[int, int, int, int], Any] = {}
logger = logging.getLogger("b12x.vllm_qsrt")


def _norm_key(name: str) -> str:
    value = str(name).removeprefix("model.")
    return value.replace("language_model.model.", "language_model.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _projection_group(prefix: str, modules: set[str]) -> str | None:
    normalized = _norm_key(prefix)
    if normalized.endswith(".gate_up_proj"):
        parent = normalized.removesuffix(".gate_up_proj")
        if {f"{parent}.gate_proj", f"{parent}.up_proj"} <= modules:
            return "gate_up"
    if normalized.endswith(".down_proj") and normalized in modules:
        return "down"
    return None


def _select_linear_method(
    prefix: str,
    modules: set[str],
    *,
    packed_method: Any,
    unquantized_method: Any,
) -> Any:
    """Select packed execution only for the declared decoder MLP inventory."""

    if _projection_group(prefix, modules) is None:
        return unquantized_method
    return packed_method


def _validate_quantization_config(config: dict[str, Any]) -> tuple[str, ...]:
    _workspace_capacity_rows(config)
    modules = config.get("modules")
    if (
        config.get("quant_method") != QUANT_NAME
        or config.get("format") != "qwen38_dense_mlp_qsrt_k5_rank16"
        or config.get("bits") != BITS
        or config.get("codebook") != "sqg_fp16"
        or config.get("descriptor_sha256") != DESCRIPTOR_SHA256
        or config.get("rank") != RANK
        or config.get("tensor_parallel_size") != 1
        or not isinstance(modules, list)
        or len(modules) != 192
        or any(not isinstance(value, str) for value in modules)
    ):
        raise ValueError("B12X QSRT quantization configuration is incompatible")
    normalized = tuple(_norm_key(value) for value in modules)
    if len(set(normalized)) != len(normalized):
        raise ValueError("B12X QSRT module inventory contains duplicates")
    expected_suffixes = {"gate_proj", "up_proj", "down_proj"}
    counts: dict[int, set[str]] = {}
    prefix = "language_model.layers."
    for name in normalized:
        if not name.startswith(prefix) or ".mlp." not in name:
            raise ValueError(f"B12X QSRT module is outside the decoder MLP: {name}")
        layer_text, projection = name[len(prefix) :].split(".mlp.", 1)
        if not layer_text.isdigit() or projection not in expected_suffixes:
            raise ValueError(f"B12X QSRT module identity is invalid: {name}")
        layer = int(layer_text)
        if not 0 <= layer < 64:
            raise ValueError(f"B12X QSRT decoder layer is out of range: {name}")
        counts.setdefault(layer, set()).add(projection)
    if counts != {layer: expected_suffixes for layer in range(64)}:
        raise ValueError("B12X QSRT module inventory is not the complete 64-layer MLP")
    return normalized


def _device_index(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else int(device.index)


def _workspace_capacity_rows(config: dict[str, Any]) -> int:
    """Return the checkpoint-declared upper bound for shared row storage."""

    value = config.get("workspace_capacity_rows")
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(
            "B12X QSRT quantization configuration requires a positive "
            "workspace_capacity_rows"
        )
    return value


def _capture_sizes(capacity_rows: int) -> tuple[int, ...]:
    if capacity_rows <= 0:
        raise ValueError("B12X QSRT workspace capacity must be positive")
    sizes = set(_DEFAULT_CAPTURE_MS)
    configured_sizes: set[int] = set()
    try:
        from vllm.config import get_current_vllm_config

        configured = getattr(
            get_current_vllm_config().compilation_config,
            "cudagraph_capture_sizes",
            None,
        )
        if configured:
            configured_sizes = {int(value) for value in configured}
    except Exception:
        logger.debug("vLLM CUDA-graph sizes are unavailable", exc_info=True)
    unsupported = sorted(value for value in configured_sizes if value > capacity_rows)
    if unsupported:
        raise NotImplementedError(
            "B12X QSRT CUDA-graph rows exceed the declared workspace capacity "
            f"{capacity_rows}: {unsupported}"
        )
    sizes.update(configured_sizes)
    return tuple(sorted(value for value in sizes if 1 <= value <= capacity_rows))


def _shared_buffers(weight: Any, rows: int, capacity_rows: int) -> Any:
    from b12x.gemm import trellis_linear

    if rows <= 0 or rows > capacity_rows:
        raise ValueError(
            f"B12X QSRT row count must be in [1, {capacity_rows}], got {rows}"
        )
    base = weight.base
    key = (
        _device_index(base.trellis.device),
        int(capacity_rows),
        int(base.in_features),
        int(base.out_features),
    )
    buffers = _SHARED_BUFFERS.get(key)
    if buffers is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "B12X QSRT scratch is absent during CUDA-graph capture; "
                "include the row count in vLLM cudagraph_capture_sizes"
            )
        buffers = trellis_linear.make_buffers(
            weight,
            size_m=capacity_rows,
            input_dtype=torch.bfloat16,
        )
        _SHARED_BUFFERS[key] = buffers
    return trellis_linear.view_buffers(buffers, size_m=rows)


def _shared_pair_buffers(weight: Any, rows: int, capacity_rows: int) -> Any:
    from b12x.gemm import trellis_linear

    if rows <= 0 or rows > capacity_rows:
        raise ValueError(
            f"B12X QSRT row count must be in [1, {capacity_rows}], got {rows}"
        )
    base = weight.left
    key = (
        _device_index(base.trellis.device),
        int(capacity_rows),
        int(base.in_features),
        int(base.out_features),
    )
    buffers = _SHARED_PAIR_BUFFERS.get(key)
    if buffers is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "B12X QSRT gate/up pair storage is absent during CUDA-graph capture"
            )
        buffers = trellis_linear.make_pair_buffers(
            weight,
            size_m=capacity_rows,
            input_dtype=torch.bfloat16,
        )
        _SHARED_PAIR_BUFFERS[key] = buffers
    return trellis_linear.view_pair_buffers(buffers, size_m=rows)


def _run_weight(
    x: torch.Tensor,
    weight: Any,
    buffers: Any,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    from b12x.gemm import trellis_linear

    kwargs = buffers.run_kwargs()
    if output is not None:
        kwargs["output"] = output
    return trellis_linear.run_additive(
        x,
        weight,
        low_rank_hidden=buffers.low_rank_hidden,
        **kwargs,
    )


def _parameter_loader(
    *,
    component: str,
    stacked: bool,
    loaded_slots: set[tuple[str, int]],
):
    expected_dtype = _COMPONENT_DTYPES[component]

    def load(
        parameter: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: int | None = None,
    ) -> None:
        if loaded_weight.dtype != expected_dtype:
            raise TypeError(
                f"B12X QSRT {component} must use {expected_dtype}, "
                f"got {loaded_weight.dtype}"
            )
        if stacked:
            if not isinstance(shard_id, int) or shard_id not in (0, 1):
                raise ValueError(f"B12X QSRT {component} requires gate/up shard 0 or 1")
            target = parameter.data[shard_id]
            slot = shard_id
        else:
            if shard_id is not None:
                raise ValueError(f"B12X QSRT down {component} is not a fused shard")
            target = parameter.data
            slot = 0
        identity = (component, slot)
        if identity in loaded_slots:
            raise ValueError(f"B12X QSRT payload was loaded twice: {identity}")
        if tuple(target.shape) != tuple(loaded_weight.shape):
            raise ValueError(
                f"B12X QSRT {component} shape differs: expected "
                f"{tuple(target.shape)}, got {tuple(loaded_weight.shape)}"
            )
        target.copy_(loaded_weight)
        loaded_slots.add(identity)

    return load


def _rebuild_config(config: dict[str, Any], model_dir: Optional[str]) -> Any:
    register_b12x_qsrt()
    assert _CONFIG_CLS is not None
    return _CONFIG_CLS(config, model_dir)


def register_b12x_qsrt() -> None:
    """Register the ``b12x_qsrt`` quantization method in every vLLM process."""

    global _registered, _CONFIG_CLS, logger
    if _registered:
        return
    try:
        from vllm.logger import init_logger
    except ImportError:
        logger.debug("vLLM logger is unavailable; using stdlib logging", exc_info=True)
    else:
        logger = init_logger("vllm.b12x_qsrt")

    from vllm.model_executor.layers.linear import (
        LinearBase,
        LinearMethodBase,
        UnquantizedLinearMethod,
    )
    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
    from vllm.model_executor.utils import set_weight_attrs

    class _VllmQSRTLinearMethod(LinearMethodBase):  # type: ignore[misc]
        def __init__(self, workspace_capacity_rows: int) -> None:
            self.workspace_capacity_rows = int(workspace_capacity_rows)

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
            del extra_weight_attrs
            if (
                params_dtype != torch.bfloat16
                or input_size_per_partition != input_size
                or sum(output_partition_sizes) != output_size
            ):
                raise ValueError("B12X QSRT supports BF16 TP1 execution only")
            stacked = len(output_partition_sizes) == 2
            if stacked:
                if (
                    output_partition_sizes[0] != output_partition_sizes[1]
                    or input_size != 5120
                    or output_partition_sizes[0] != 17408
                ):
                    raise ValueError("B12X QSRT gate/up geometry is incompatible")
                size_k, size_n, stack = input_size, output_partition_sizes[0], 2
                layer.b12x_qsrt_group = "gate_up"
            else:
                if (
                    len(output_partition_sizes) != 1
                    or input_size != 17408
                    or output_size != 5120
                ):
                    raise ValueError("B12X QSRT down geometry is incompatible")
                size_k, size_n, stack = input_size, output_size, 1
                layer.b12x_qsrt_group = "down"
            shapes = {
                "trellis": (size_k // 16, size_n // 16, 16 * BITS),
                "suh": (size_k,),
                "svh": (size_n,),
                "a": (size_k, RANK),
                "b": (size_n, RANK),
            }
            loaded_slots: set[tuple[str, int]] = set()
            layer.b12x_qsrt_loaded_slots = loaded_slots
            for component, shape in shapes.items():
                storage_shape = (stack, *shape) if stacked else shape
                parameter = torch.nn.Parameter(
                    torch.empty(storage_shape, dtype=_COMPONENT_DTYPES[component]),
                    requires_grad=False,
                )
                set_weight_attrs(
                    parameter,
                    {
                        "weight_loader": _parameter_loader(
                            component=component,
                            stacked=stacked,
                            loaded_slots=loaded_slots,
                        )
                    },
                )
                layer.register_parameter(f"qsrt_{component}", parameter)

        def process_weights_after_loading(self, layer: Any) -> None:
            from b12x.gemm import trellis_linear

            if not trellis_linear.is_supported(layer.qsrt_trellis.device):
                raise NotImplementedError(
                    "B12X QSRT requires an SM120 or SM121 CUDA device with "
                    "the declared B12X kernel dependencies"
                )
            stacked = layer.b12x_qsrt_group == "gate_up"
            expected_slots = {
                (component, slot)
                for component in _COMPONENT_DTYPES
                for slot in range(2 if stacked else 1)
            }
            if layer.b12x_qsrt_loaded_slots != expected_slots:
                raise ValueError(
                    "B12X QSRT checkpoint did not load every payload tensor"
                )

            def tensor(component: str, slot: int | None) -> torch.Tensor:
                value = getattr(layer, f"qsrt_{component}").data
                return value if slot is None else value[slot]

            def prepare_base(slot: int | None) -> Any:
                return trellis_linear.prepare_weight(
                    tensor("trellis", slot),
                    tensor("suh", slot),
                    tensor("svh", slot),
                    codebook="sqg_fp16",
                    params_dtype=torch.bfloat16,
                )

            if stacked:
                weight = trellis_linear.prepare_additive_pair(
                    prepare_base(0),
                    prepare_base(1),
                    tensor("a", None),
                    tensor("b", None),
                )
            else:
                weight = trellis_linear.prepare_additive_weight(
                    prepare_base(None),
                    tensor("a", None),
                    tensor("b", None),
                )
            layer.b12x_qsrt_weight = weight
            a_parameter = layer.qsrt_a
            a_parameter.data = torch.empty(
                0,
                dtype=a_parameter.dtype,
                device=a_parameter.device,
            )
            capacity_rows = self.workspace_capacity_rows
            sizes = _capture_sizes(capacity_rows)
            if stacked:
                _shared_pair_buffers(weight, capacity_rows, capacity_rows)
            else:
                _shared_buffers(weight, capacity_rows, capacity_rows)
            for rows in sizes:
                if stacked:
                    _shared_pair_buffers(weight, rows, capacity_rows)
                else:
                    _shared_buffers(weight, rows, capacity_rows)
            for rows in sizes:
                if stacked:
                    geometry = (
                        2,
                        _device_index(weight.left.trellis.device),
                        rows,
                        int(weight.left.in_features),
                        int(weight.left.out_features),
                        capacity_rows,
                    )
                    device = weight.left.trellis.device
                    input_features = int(weight.left.in_features)
                else:
                    geometry = (
                        1,
                        _device_index(weight.base.trellis.device),
                        rows,
                        int(weight.base.in_features),
                        int(weight.base.out_features),
                        capacity_rows,
                    )
                    device = weight.base.trellis.device
                    input_features = int(weight.base.in_features)
                if geometry in _WARMED_GEOMETRIES:
                    continue
                x = torch.zeros(
                    (rows, input_features),
                    dtype=torch.bfloat16,
                    device=device,
                )
                if stacked:
                    trellis_linear.run_additive_pair(
                        x,
                        weight,
                        buffers=_shared_pair_buffers(weight, rows, capacity_rows),
                    )
                else:
                    _run_weight(
                        x,
                        weight,
                        _shared_buffers(weight, rows, capacity_rows),
                    )
                _WARMED_GEOMETRIES.add(geometry)
            device = (
                weight.left.trellis.device if stacked else weight.base.trellis.device
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        def apply(
            self,
            layer: Any,
            x: torch.Tensor,
            bias: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            if bias is not None:
                raise ValueError("B12X QSRT Qwen dense MLPs do not have bias")
            if x.dtype != torch.bfloat16 or not x.is_contiguous():
                raise ValueError("B12X QSRT requires contiguous BF16 activations")
            x2d = x.reshape(-1, x.shape[-1])
            rows = int(x2d.shape[0])
            capacity_rows = self.workspace_capacity_rows
            weight = layer.b12x_qsrt_weight
            if layer.b12x_qsrt_group == "gate_up":
                from b12x.gemm import trellis_linear

                result = trellis_linear.run_additive_pair(
                    x2d,
                    weight,
                    buffers=_shared_pair_buffers(weight, rows, capacity_rows),
                )
            else:
                buffers = _shared_buffers(weight, rows, capacity_rows)
                result = _run_weight(x2d, weight, buffers)
            if x.dim() > 2:
                result = result.reshape(*x.shape[:-1], result.shape[-1])
            return result

    import vllm.model_executor.layers.linear as _vllm_linear

    register_v2 = getattr(
        _vllm_linear,
        "register_weight_loader_v2_supported_method",
        None,
    )
    if register_v2 is not None:
        register_v2(_VllmQSRTLinearMethod)
    elif _VllmQSRTLinearMethod.__name__ not in _vllm_linear.WEIGHT_LOADER_V2_SUPPORTED:
        _vllm_linear.WEIGHT_LOADER_V2_SUPPORTED.append(_VllmQSRTLinearMethod.__name__)

    @register_quantization_config(QUANT_NAME)
    class B12XQSRTConfig(QuantizationConfig):  # type: ignore[misc]
        def __init__(
            self,
            config: dict[str, Any],
            model_dir: Optional[str] = None,
        ) -> None:
            super().__init__()
            self._config = dict(config)
            self._modules = set(_validate_quantization_config(self._config))
            self._workspace_capacity_rows = _workspace_capacity_rows(self._config)
            self._model_dir = model_dir or os.environ.get(MODEL_DIR_ENV) or None
            self._manifest_validated = False
            self._method = _VllmQSRTLinearMethod(self._workspace_capacity_rows)
            self._unquantized_method = UnquantizedLinearMethod()

        def __reduce__(self):
            return (_rebuild_config, (self._config, self._model_dir))

        @classmethod
        def get_name(cls) -> str:
            return QUANT_NAME

        @classmethod
        def get_supported_act_dtypes(cls) -> list[torch.dtype]:
            return [torch.bfloat16]

        @classmethod
        def get_min_capability(cls) -> int:
            return 120

        @staticmethod
        def get_config_filenames() -> list[str]:
            return []

        @classmethod
        def from_config(cls, config: dict[str, Any]) -> "B12XQSRTConfig":
            return cls(config)

        @classmethod
        def override_quantization_method(
            cls,
            hf_quant_cfg: dict[str, Any],
            user_quant: Optional[str],
            hf_config: Any = None,
        ) -> Optional[str]:
            del user_quant, hf_config
            if hf_quant_cfg.get("quant_method") == QUANT_NAME:
                return QUANT_NAME
            return None

        def maybe_update_config(
            self,
            model_name: str,
            hf_config: Any = None,
            revision: Any = None,
        ) -> None:
            del hf_config, revision
            if self._model_dir is None and model_name:
                self._model_dir = model_name

        def _validate_manifest(self) -> None:
            if self._manifest_validated:
                return
            model_dir = self._model_dir or os.environ.get(MODEL_DIR_ENV)
            if not model_dir:
                raise RuntimeError(
                    f"B12X QSRT model directory is unknown; set {MODEL_DIR_ENV}"
                )
            path = Path(model_dir).expanduser().resolve() / MANIFEST_FILENAME
            document = json.loads(path.read_text())
            if not isinstance(document, dict):
                raise TypeError(f"{path} must contain a JSON object")
            build = document.get("build", {})
            root = path.parent
            if (
                document.get("kind") != CHECKPOINT_KIND
                or document.get("status") != "implemented"
                or document.get("indexed_tensor_count") != 1967
                or _sha256(root / "config.json") != document.get("config_sha256")
                or _sha256(root / "model.safetensors.index.json")
                != document.get("index_sha256")
                or build.get("artifact_manifest_sha256")
                != self._config.get("artifact_manifest_sha256")
                or build.get("adapter_manifest_sha256")
                != self._config.get("adapter_manifest_sha256")
                or build.get("recovery_report_sha256")
                != self._config.get("recovery_report_sha256")
                or build.get("recovery_overlay_sha256")
                != self._config.get("recovery_overlay_sha256")
            ):
                raise ValueError("B12X QSRT checkpoint manifest is incompatible")
            self._manifest_validated = True

        def get_quant_method(self, layer: Any, prefix: str):
            if not isinstance(layer, LinearBase):
                return None
            method = _select_linear_method(
                prefix,
                self._modules,
                packed_method=self._method,
                unquantized_method=self._unquantized_method,
            )
            if method is self._method:
                self._validate_manifest()
            return method

    _CONFIG_CLS = B12XQSRTConfig
    _registered = True


__all__ = [
    "BITS",
    "CHECKPOINT_KIND",
    "DESCRIPTOR_SHA256",
    "MANIFEST_FILENAME",
    "MODEL_DIR_ENV",
    "QUANT_NAME",
    "RANK",
    "register_b12x_qsrt",
]
