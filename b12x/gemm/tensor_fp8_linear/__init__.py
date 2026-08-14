"""Static per-tensor FP8 linear for SM12x.

``pack_weight`` keeps serialized E4M3 weights unchanged, prepares unit UE8M0
scale-factor storage once, and records the combined activation/weight
dequantization scale. ``mm`` consumes an already-quantized E4M3 activation and
runs the native SM12x dense GEMM with the combined scale in its FP32 epilogue.

Unit-scale MMA tensor cache (per-device, process-lifetime admission):
  Env vars (set before import):
    ``B12X_TENSOR_FP8_UNIT_SCALE_CACHE_MAX_ENTRIES`` — default 32, hard max 256
    ``B12X_TENSOR_FP8_UNIT_SCALE_CACHE_MAX_BYTES``  — default 64 MiB, hard max 256 MiB
  Admitted entries are never evicted (CUDA graph replay safety).  Row keys are
  canonicalized to 128-row tiles.  ``prewarm`` is non-atomic: a failure
  after partial admission leaves earlier geometries permanently admitted;
  restart the process to reset.  Plan capacity accordingly.

Example:
    from b12x.gemm import tensor_fp8_linear

    packed = tensor_fp8_linear.pack_weight(
        weight_fp8,
        input_scale * weight_scale,
    )
    output = tensor_fp8_linear.mm(input_fp8, packed)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="tensor_fp8_linear",
    group="gemm",
    api_style="oneshot",
    entry_points=("Weight", "mm", "pack_weight", "prewarm", "is_supported"),
    dtypes=("fp8_e4m3", "bf16", "fp16"),
    recipes=("tensor_fp8",),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="1bc4f82",
        paths=("b12x/gemm/tensor_fp8_linear",),
    ),
    test_path="tests/gemm/test_tensor_fp8_linear.py",
    since="1.0.1",
)

if TYPE_CHECKING:
    from .api import Weight, is_supported, mm, pack_weight, prewarm  # noqa: F401

install_lazy_api(globals(), META)
