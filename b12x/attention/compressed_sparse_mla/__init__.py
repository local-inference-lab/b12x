"""Compressed sparse MLA for DeepSeek V4 and V4.1 on SM12x.

Decode directly from compressed KV pages: a sliding-window cache plus an
indexed (top-k-selected) cache, fused-merged into the caller's output.
Head dim is fixed to 512. V4 uses 448 NoPE + 64 BF16 RoPE; V4.1
quantizes all 512 coordinates with independent SWA FP8 and indexed FP4 scales.

Planned lifecycle: ``plan(Caps(...))`` -> ``bind`` (views only) -> ``run``
(capture safe). ``split_chunks_for_contract`` exposes the fixed
split-planning contract integrations preplan against.

``Caps(cache_format="deepseek_v41")`` selects SWA E4M3 + UE8M0/group32
(528 bytes/token) and indexed E2M1 + E4M3/group16 (288 bytes/token).
The default ``deepseek_v4`` recipe is unchanged. ``page_nbytes`` sizes either
recipe; ``write_cache`` and ``compile_cache_writer`` encode V4.1 records.
``indexed_page_table`` maps logical indexed slots at run time into planned
scratch, with checked 64-bit page products. Bound calls inherit their recipe;
unbound ``run`` is an allocating convenience outside graph capture only.

Example:
    from b12x.attention import compressed_sparse_mla

    plan    = compressed_sparse_mla.plan(compressed_sparse_mla.Caps(...))
    spec    = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = compressed_sparse_mla.bind(plan, scratch=scratch, q=q,
                                  swa_indices=idx, swa_lengths=lens, ...)
    out = compressed_sparse_mla.run(swa_k_cache=swa, binding=binding,
                             sm_scale=scale, ...)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="compressed_sparse_mla",
    group="attention",
    api_style="planned",
    entry_points=(
        "Caps",
        "Plan",
        "Binding",
        "Scratch",
        "SparseMlaConfig",
        "SparseMlaQuery",
        "plan",
        "bind",
        "run",
        "split_chunks_for_contract",
        "page_nbytes",
        "write_cache",
        "compile_cache_writer",
        "is_supported",
        "clear_caches",
    ),
    dtypes=("bf16", "fp8_e4m3"),
    recipes=("dsv4", "dsv41"),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=(
            "b12x/attention/mla/compressed_api.py",
            "b12x/integration/compressed_scratch.py",
        ),
    ),
    test_path="tests/attention/test_compressed_sparse_mla.py",
    since="0.7.0",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        Binding,
        Caps,
        Plan,
        Scratch,
        SparseMlaConfig,
        SparseMlaQuery,
        bind,
        clear_caches,
        compile_cache_writer,
        is_supported,
        plan,
        page_nbytes,
        run,
        split_chunks_for_contract,
        write_cache,
    )

install_lazy_api(globals(), META)
