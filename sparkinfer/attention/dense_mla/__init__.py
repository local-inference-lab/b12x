"""Dense Multi-head Latent Attention for Kimi K3 on SM12x.

The operator consumes the absorbed K3 query and the combined compressed-cache
record directly:

* query: ``[total_q, local_heads, 576]``;
* cache: ``[pages, page_size, 576]`` (or a singleton-head rank-4 view);
* key: all 576 record elements;
* value: the first 512 record elements.

The attention scale is based on K3's *raw* 192-wide QK head
(``1 / sqrt(192)``), not the 576-wide absorbed record.  BF16 and standard
E4M3 combined records are supported; E4M3 uses one caller-provided FP32 scalar
scale for the whole record.

Planned lifecycle: ``plan(Caps(...))`` -> caller allocates ``scratch_specs`` ->
``bind`` (views and validation only) -> ``compile``/``run``.  Decode and
single-request prefill share the same graph-safe dense core.  No selected-index
array is materialized.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="dense_mla",
    group="attention",
    api_style="planned",
    entry_points=(
        "Caps",
        "Budget",
        "Plan",
        "Binding",
        "Scratch",
        "plan",
        "bind",
        "compile",
        "run",
        "reference",
        "infer_mode",
        "is_supported",
        "clear_caches",
    ),
    dtypes=("bf16", "fp8_e4m3"),
    recipes=("kimi_k3_dense_mla",),
    requires=(),
    provenance=Provenance(
        repo="https://github.com/lukealonso/sparkinfer",
        commit="36cade0b",
        paths=(
            "sparkinfer/attention/paged/",
            "sparkinfer/attention/_shared/mla/",
            "sparkinfer/attention/nsa_indexer/",
        ),
    ),
    test_path="tests/attention/test_dense_mla.py",
    since="0.8.0",
    notes=(
        "K3 TP12 uses eight local heads, a 576-element absorbed Q/K record, "
        "a 512-element latent value, and 1/sqrt(192) scaling. Pool-scaled "
        "addressing is 64-bit and the runtime owns no allocations."
    ),
)

if TYPE_CHECKING:
    from .api import (  # noqa: F401
        Binding,
        Budget,
        Caps,
        Plan,
        Scratch,
        bind,
        clear_caches,
        compile,
        infer_mode,
        is_supported,
        plan,
        reference,
        run,
    )

install_lazy_api(globals(), META)
