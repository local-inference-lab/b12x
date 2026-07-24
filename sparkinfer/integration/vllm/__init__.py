"""vLLM quantization plugin for sparkinfer MX-FP6 (W6A6/W6A8).

Import :func:`register_sparkinfer_fp6` from :mod:`sparkinfer.integration.vllm.plugin`
and wire it through vLLM's ``general_plugins`` entry point (see README).
"""

from .fp6_serving import (
    QUANT_ALGO,
    QUANT_METHOD,
    SparkInferFP6LinearMethod,
    SparkInferFP6MoEMethod,
    is_sparkinfer_fp6_checkpoint,
    is_sparkinfer_fp6_enabled,
    load_sparkinfer_fp6_linear_methods,
    load_sparkinfer_fp6_moe_methods,
    should_use_sparkinfer_fp6,
)

__all__ = [
    "QUANT_ALGO",
    "QUANT_METHOD",
    "SparkInferFP6LinearMethod",
    "SparkInferFP6MoEMethod",
    "is_sparkinfer_fp6_checkpoint",
    "is_sparkinfer_fp6_enabled",
    "load_sparkinfer_fp6_linear_methods",
    "load_sparkinfer_fp6_moe_methods",
    "should_use_sparkinfer_fp6",
]
