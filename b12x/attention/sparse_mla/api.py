"""Public surface for attention.sparse_mla (docs in the op ``__init__``)."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from .._shared.mla.traits import ModelType
from .._shared.mla.api import (
    MLASparseDecodeMetadata as DecodeMetadata,
)
from .._shared.mla.api import (
    MLASparseExtendMetadata as ExtendMetadata,
)
from .._shared.mla.api import (
    clear_mla_caches as clear_caches,
)
from .._shared.mla.api import (
    sparse_mla_decode_forward as run_decode,
)
from .._shared.mla.api import (
    sparse_mla_extend_forward as run_extend,
)
from .._shared.mla.kv_cache import (
    compile_glm_next_mla_cache_writer,
    concat_and_cache_glm_next_mla,
    concat_and_cache_glm_next_mla_fp8,
    concat_and_cache_glm_next_mla_nvfp4,
)
from .pooled_selection import expand_pooled_topk_to_physical_slots
from ._scratch import (
    B12XSparseMLABinding as Binding,
)
from .physical_selection import expand_pooled_topk_to_physical_slots
from ._policy import SparseMlaConfig, SparseMlaQuery
from ._scratch import (
    B12XSparseMLAScratch as Scratch,
)
from ._scratch import (
    B12XSparseMLAScratchCaps as Caps,
)
from ._scratch import (
    B12XSparseMLAScratchPlan as Plan,
)
from ._scratch import (
    plan_sparse_mla_scratch as plan,
)
from . import META


def bind(plan: Plan, **kwargs) -> Binding:
    """Bind runtime tensors and caller-owned scratch to a plan.

    Views only — never allocates — so it is CUDA-graph-capture safe.
    Delegates to ``plan.bind(**kwargs)``.
    """
    return plan.bind(**kwargs)


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0 and triton."""
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "ModelType",
    "Caps",
    "Plan",
    "Binding",
    "Scratch",
    "DecodeMetadata",
    "ExtendMetadata",
    "SparseMlaConfig",
    "SparseMlaQuery",
    "plan",
    "bind",
    "run_decode",
    "run_extend",
    "compile_glm_next_mla_cache_writer",
    "concat_and_cache_glm_next_mla",
    "concat_and_cache_glm_next_mla_fp8",
    "concat_and_cache_glm_next_mla_nvfp4",
    "expand_pooled_topk_to_physical_slots",
    "is_supported",
    "clear_caches",
]
