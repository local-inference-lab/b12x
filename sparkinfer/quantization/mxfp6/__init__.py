"""BF16 → MX-FP6 TMA quantization kernel APIs + FP6 checkpoint/weight stack.

FP6 subset of the original ``b12x.quantization`` package, ported to
``sparkinfer.quantization.mxfp6`` (the FP4/NVFP4 surface lives elsewhere in
``sparkinfer.quantization``).
"""
from dataclasses import dataclass
from typing import Dict, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.typing import AddressSpace

from sparkinfer._lib.compiler import KernelCompileSpec, compile as sparkinfer_compile
from sparkinfer._lib.intrinsics import align_up
from sparkinfer._lib.utils import (
    MXFP6_SF_VEC_SIZE,
    current_cuda_stream,
    get_max_active_clusters,
    get_num_sm,
    make_ptr,
    mxfp6_packed_k_bytes,
)
from sparkinfer._lib.runtime_control import raise_if_kernel_resolution_frozen
from sparkinfer.quantization.mxfp6.bf16_to_fp6_tma import TestKernel as FP6TestKernel

_TILE_M = 128
_TILE_K = 128
_SF_VEC_SIZE_FP6 = MXFP6_SF_VEC_SIZE
_KERNEL_CACHE_FP6: Dict[Tuple, object] = {}


@dataclass
class BF16ToFP6TMAOutputs:
    packed_a_storage: torch.Tensor
    scale_storage: torch.Tensor
    packed_a_view: object
    sfa_ptr: object

    @property
    def packed_a_flat(self) -> torch.Tensor:
        return self.packed_a_storage.view(-1)

    @property
    def scale_flat(self) -> torch.Tensor:
        return self.scale_storage.view(-1)


def allocate_bf16_to_fp6_tma_outputs(
    M: int,
    K: int,
    *,
    device: torch.device = torch.device("cuda"),
    emit: str = "packed",
) -> BF16ToFP6TMAOutputs:
    """Allocate uint8 FP6 activations and swizzled UE8M0 scales.

    ``emit="packed"`` allocates the 3:4-packed wire format (``3K/4`` bytes per
    row); ``emit="bytes"`` the one-code-per-byte GEMM operand layout (``K``
    bytes per row). Zero-filled: the quantizer writes all in-range codes/scales,
    but padding rows (M..rows_pad-1) stay untouched and the GEMM's TMA tile may
    read them for m in 2..15. Zeroed padding guarantees cross-process
    reproducibility (torch.empty contents vary per CUDA context).
    """
    rows_pad = align_up(M, _TILE_M)
    cols_pad_sf = align_up(K // _SF_VEC_SIZE_FP6, 4)
    packed_k_bytes = K if emit == "bytes" else mxfp6_packed_k_bytes(K)
    packed_a_storage = torch.zeros(1, M, packed_k_bytes, dtype=torch.uint8, device=device)
    scale_storage = torch.zeros(rows_pad * cols_pad_sf, dtype=torch.uint8, device=device)
    packed_a_view = packed_a_storage.permute(1, 2, 0)
    sfa_ptr = make_ptr(
        cutlass.Float8E8M0FNU,
        scale_storage.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    return BF16ToFP6TMAOutputs(
        packed_a_storage=packed_a_storage,
        scale_storage=scale_storage,
        packed_a_view=packed_a_view,
        sfa_ptr=sfa_ptr,
    )


def compile_bf16_to_fp6_tma(M: int, K: int, fmt: str = "e3m2", emit: str = "packed"):
    """Compile the BF16→MX-FP6 TMA kernel for (M, K) in ``fmt`` (``e3m2``/``e2m3``).

    ``emit`` selects the code layout written: ``"packed"`` (3:4 wire format,
    ``3K/4`` bytes/row) or ``"bytes"`` (one code per byte, ``K`` bytes/row -
    directly consumable by the ``mxf8f6f4`` GEMM, no expansion step).

    The callable signature is: ``launch(bf16_input, global_scale, packed_a_flat, scale_flat)``
    where packed_a_flat and scale_flat come from ``BF16ToFP6TMAOutputs``.
    """
    assert M % _TILE_M == 0 and K % _TILE_K == 0
    assert fmt in ("e3m2", "e2m3", "e4m3"), f"unsupported act/weight fmt: {fmt}"
    if fmt == "e4m3" and emit != "bytes":
        raise ValueError("e4m3 activations require emit='bytes'")
    assert emit in ("packed", "bytes"), f"unsupported FP6 emit: {emit}"
    cache_key = (M, K, fmt, emit)
    cached = _KERNEL_CACHE_FP6.get(cache_key)
    if cached is not None:
        return cached

    sf = cutlass.Float8E8M0FNU
    bf = cutlass.BFloat16
    packed_k_bytes = K if emit == "bytes" else mxfp6_packed_k_bytes(K)
    bf16_fake = cute.runtime.make_fake_compact_tensor(
        bf, (M, K), stride_order=(1, 0), assumed_align=16
    )
    gs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (1,), assumed_align=4
    )
    pa_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (M, packed_k_bytes, 1), stride_order=(1, 0, 2), assumed_align=16
    )
    sfa_fake = make_ptr(sf, 16, AddressSpace.gmem, assumed_align=16)
    mac = min(get_max_active_clusters(1), get_num_sm(torch.device("cuda")))
    kernel = FP6TestKernel(fmt, emit=emit)
    raise_if_kernel_resolution_frozen("cute.compile", target=kernel, cache_key=cache_key)
    raw = sparkinfer_compile(
        kernel,
        bf16_fake,
        gs_fake,
        pa_fake,
        sfa_fake,
        mac,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "quantization.bf16_to_fp6_tma",
            1,
            cache_key,
        ),
    )

    def launch(bf16_input, global_scale, packed_a_flat, scale_flat):
        pa_storage = packed_a_flat.view(1, M, packed_k_bytes).permute(1, 2, 0)
        sfa_p = make_ptr(
            sf, scale_flat.data_ptr(), AddressSpace.gmem, assumed_align=16
        )
        raw(bf16_input, global_scale, pa_storage, sfa_p, current_cuda_stream())

    _KERNEL_CACHE_FP6[cache_key] = launch
    return launch


# Imported last: the offline MoE weight-quantization pipeline lazily depends on
# the BF16->FP6 quantizer helpers defined above.
from sparkinfer.quantization.mxfp6.fp6_moe_weights import (  # noqa: E402
    FP6MoEWeights,
    quantize_moe_weights_to_fp6,
    save_fp6_moe_weights,
    load_fp6_moe_weights,
)
from sparkinfer.quantization.mxfp6.fp6_dense_weights import (  # noqa: E402
    FP6DenseWeight,
    quantize_dense_weight_to_fp6,
    dense_fp6_linear,
    save_fp6_dense_weight,
    load_fp6_dense_weight,
)
from sparkinfer.quantization.mxfp6.fp6_checkpoint import (  # noqa: E402
    FP6LinearTensors,
    quantize_linear_to_fp6,
    dequantize_linear_from_fp6,
    fp6_decode_lut,
    build_quantization_config,
    weight_format_for_source,
    activation_format_for_source,
    resolve_activation_format,
    parse_act_fmt_overrides,
    load_act_fmt_overrides,
    SCHEMA_VERSION as FP6_SCHEMA_VERSION,
    QUANT_ALGO as FP6_QUANT_ALGO,
    GROUP_SIZE as FP6_GROUP_SIZE,
)
from sparkinfer.quantization.mxfp6.model_fp6 import (  # noqa: E402
    SafetensorsModel,
    MoEExpertScheme,
    DenseLinearScheme,
    discover_moe_experts,
    discover_dense_linears,
    convert_moe_model_to_fp6,
    convert_dense_model_to_fp6,
    ConvertReport,
)
from sparkinfer.quantization.mxfp6.fp6_safetensors_export import (  # noqa: E402
    ExportReport,
    export_model_to_fp6_safetensors,
    export_moe_model_to_fp6_safetensors,
    export_dense_model_to_fp6_safetensors,
    dequantize_fp6_checkpoint_to_bf16,
)
from sparkinfer.quantization.mxfp6.fp6_safetensors_load import (  # noqa: E402
    load_fp6_moe_weights_from_safetensors,
    load_fp6_moe_checkpoint,
    load_fp6_dense_weight_from_safetensors,
    load_fp6_dense_checkpoint,
)

__all__ = [
    "BF16ToFP6TMAOutputs",
    "allocate_bf16_to_fp6_tma_outputs",
    "compile_bf16_to_fp6_tma",
    "FP6MoEWeights",
    "quantize_moe_weights_to_fp6",
    "save_fp6_moe_weights",
    "load_fp6_moe_weights",
    "FP6DenseWeight",
    "quantize_dense_weight_to_fp6",
    "dense_fp6_linear",
    "save_fp6_dense_weight",
    "load_fp6_dense_weight",
    "FP6LinearTensors",
    "quantize_linear_to_fp6",
    "dequantize_linear_from_fp6",
    "fp6_decode_lut",
    "build_quantization_config",
    "weight_format_for_source",
    "activation_format_for_source",
    "resolve_activation_format",
    "parse_act_fmt_overrides",
    "load_act_fmt_overrides",
    "FP6_SCHEMA_VERSION",
    "FP6_QUANT_ALGO",
    "FP6_GROUP_SIZE",
    "SafetensorsModel",
    "MoEExpertScheme",
    "DenseLinearScheme",
    "discover_moe_experts",
    "discover_dense_linears",
    "convert_moe_model_to_fp6",
    "convert_dense_model_to_fp6",
    "ConvertReport",
    "ExportReport",
    "export_model_to_fp6_safetensors",
    "export_moe_model_to_fp6_safetensors",
    "export_dense_model_to_fp6_safetensors",
    "dequantize_fp6_checkpoint_to_bf16",
    "load_fp6_moe_weights_from_safetensors",
    "load_fp6_moe_checkpoint",
    "load_fp6_dense_weight_from_safetensors",
    "load_fp6_dense_checkpoint",
]
