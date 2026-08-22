"""Public surface for gemm.mxfp8_linear (docs in the op ``__init__``)."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from ._kernel import (
    MXFP8LinearWeight as Weight,
)
from ._kernel import (
    empty_mxfp8_linear_input as empty_input,
)
from ._kernel import (
    is_mxfp8_linear_supported as _kernel_is_supported,
)
from ._kernel import (
    mxfp8_linear as mm,
)
from ._kernel import (
    mxfp8_linear_quantized as mm_quantized,
)
from ._kernel import (
    pack_mxfp8_linear_weight as pack_weight,
)
from ._kernel import (
    quantize_mxfp8_linear_input_slice as quantize_input_slice,
)
from . import META


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0, triton, and
    the kernel's own capability checks."""
    return default_is_supported(device, requires=META.requires) and bool(
        _kernel_is_supported()
    )


__all__ = [
    "Weight",
    "empty_input",
    "is_supported",
    "mm",
    "mm_quantized",
    "pack_weight",
    "quantize_input_slice",
]
