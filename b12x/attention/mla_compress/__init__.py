"""Projected DeepSeek V4.1 CSA1/CSA2 compression with caller-owned carry.

``Caps -> plan -> bind -> run`` exposes normalized BF16 pre-RoPE latents plus
fixed-capacity emission metadata. See ``api`` for packed requests, state tags,
replay counts and ownership. This operation does not project, rotate, allocate
cache slots, or quantize the latent before indexer K projection.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="mla_compress", group="attention", api_style="planned",
    entry_points=("Caps", "Plan", "Binding", "MlaCompressQuery", "MlaCompressConfig",
                  "plan", "bind", "run", "is_supported"),
    dtypes=("bf16", "fp32"), recipes=("deepseek_v41",),
    provenance=Provenance(
        repo="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash",
        commit="fb2764a5cf321eaa5070ca8f9e892818f477c16d",
        paths=("inference/model.py",)),
    test_path="tests/attention/test_mla_compress.py",
    notes="New CuTe implementation of reference Compressor math; no V4 overlap/APE or cache packing.",
)

if TYPE_CHECKING:
    from .api import (Binding, Caps, Plan, MlaCompressQuery, MlaCompressConfig,
                      bind, is_supported, plan, run)

install_lazy_api(globals(), META)
