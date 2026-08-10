"""Public API for :mod:`b12x.gemm.trellis_linear`."""

from __future__ import annotations

from typing import Optional

import torch

from ..._lib.gating import default_is_supported
from ...moe._shared.kernels.w4a16.kernel import (
    clear_w4a16_kernel_cache,
    run_trellis256_dense,
)
from ...moe._shared.kernels.w4a16.prepare import (
    PreparedTrellis256DenseWeight,
    prepare_trellis256_dense_weight,
    prepare_trellis256_pair_dense_weight,
)
from .w4a8 import _compile_dense as _compile_w4a8_dense
from .w4a8 import run_trellis256_dense_w4a8
from . import META

PreparedWeight = PreparedTrellis256DenseWeight


def prepare_weight(
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    *,
    mcg: Optional[torch.Tensor] = None,
    mul1_e4m3: Optional[torch.Tensor] = None,
    codebook: Optional[str | int] = None,
    params_dtype: torch.dtype = torch.float16,
    dummy_scale: Optional[torch.Tensor] = None,
) -> PreparedWeight:
    """Validate one native EXL3 dense weight and retain zero-copy views."""
    return prepare_trellis256_dense_weight(
        trellis,
        suh,
        svh,
        mcg=mcg,
        mul1_e4m3=mul1_e4m3,
        codebook=codebook,
        params_dtype=params_dtype,
        dummy_scale=dummy_scale,
    )


def prepare_pair_weight(
    payload: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    *,
    pair_kind: str,
    rate_axis: str,
    mcg: Optional[torch.Tensor] = None,
    mul1_e4m3: Optional[torch.Tensor] = None,
    codebook: Optional[str | int] = None,
    params_dtype: torch.dtype = torch.float16,
    dummy_scale: Optional[torch.Tensor] = None,
) -> PreparedWeight:
    """Prepare one compact TP12 P24/P33 pair for the SM12x decoder."""
    return prepare_trellis256_pair_dense_weight(
        payload,
        suh,
        svh,
        pair_kind=pair_kind,
        rate_axis=rate_axis,
        mcg=mcg,
        mul1_e4m3=mul1_e4m3,
        codebook=codebook,
        params_dtype=params_dtype,
        dummy_scale=dummy_scale,
    )


def run(
    x: torch.Tensor,
    weight: PreparedWeight,
    *,
    output: Optional[torch.Tensor] = None,
    gemm_output: Optional[torch.Tensor] = None,
    c_tmp: Optional[torch.Tensor] = None,
    input_f16: Optional[torch.Tensor] = None,
    rotated_f16: Optional[torch.Tensor] = None,
    rotated_compute: Optional[torch.Tensor] = None,
    gemm_output_f16: Optional[torch.Tensor] = None,
    output_f16: Optional[torch.Tensor] = None,
    hadamard_128=None,
    _moe_block_size: int | None = None,
    _force_tile_config: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Execute Trellis GEMM, optionally reusing all capture-time storage."""
    return run_trellis256_dense(
        x,
        weight,
        output=output,
        gemm_output=gemm_output,
        c_tmp=c_tmp,
        input_f16=input_f16,
        rotated_f16=rotated_f16,
        rotated_compute=rotated_compute,
        gemm_output_f16=gemm_output_f16,
        output_f16=output_f16,
        hadamard_128=hadamard_128,
        _moe_block_size=_moe_block_size,
        _force_tile_config=_force_tile_config,
    )


def run_w4a8(
    x: torch.Tensor,
    weight: PreparedWeight,
    *,
    output: Optional[torch.Tensor] = None,
    input_f16: Optional[torch.Tensor] = None,
    rotated_f16: Optional[torch.Tensor] = None,
    quantized=None,
    gemm_output_f16: Optional[torch.Tensor] = None,
    output_f16: Optional[torch.Tensor] = None,
    hadamard_128=None,
) -> torch.Tensor:
    """Execute direct E4M3 Trellis decode through SM120 W4A8 MMA."""
    return run_trellis256_dense_w4a8(
        x,
        weight,
        output=output,
        input_f16=input_f16,
        rotated_f16=rotated_f16,
        quantized=quantized,
        gemm_output_f16=gemm_output_f16,
        output_f16=output_f16,
        hadamard_128=hadamard_128,
    )


def is_supported(device=None) -> bool:
    """True when the SM120/SM121 Trellis kernel stack is available."""
    return default_is_supported(device, requires=META.requires)


def clear_caches() -> None:
    """Clear compiled W4A16 and direct-W4A8 specializations."""
    clear_w4a16_kernel_cache()
    _compile_w4a8_dense.cache_clear()


__all__ = list(META.entry_points)
