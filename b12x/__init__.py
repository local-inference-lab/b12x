"""b12x — Blackwell inference kernels with per-operation architecture coverage.

CuTe-DSL and Triton kernels for NVFP4/MXFP4/MXFP8 GEMM, fused MoE, attention
(paged, dense/sparse/compressed MLA, DSA indexing, and QSA decode),
quantization, multi-stream residual mixing, recurrent/sequence features, and
PCIe collectives. One grammar everywhere:

- ops live at ``b12x.<group>.<op>`` and declare themselves via ``META``;
- operations declare ``Plan`` values without allocating or compiling;
- ``PreparationSession`` selects, compiles and primes each ``Plan``, which then
  carries its own prepared state and a stable integer ``handle``;
- binding and custom ops consume the prepared ``Plan``; run and graph replay
  never resolve kernels.

Importing this module is cheap and side-effect free; kernels, cutlass, and
torch custom ops load on first op use.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from ._lib.meta import OpMeta
from ._lib.runtime_control import (
    KernelResolutionFrozenError,
    kernel_resolution_frozen,
)

# Static logical-op registry, kept in lockstep with public op directories and
# the explicit private-module overrides below by tests/test_registry.py.
_OPS: tuple[str, ...] = (
    "attention.paged",
    "attention.dense_mla",
    "attention.sparse_mla",
    "attention.compressed_sparse_mla",
    "attention.mla_compress",
    "attention.dsa_indexer",
    "attention.mla_compress",
    "attention.qsa",
    "attention.varlen",
    "comm.pcie",
    "comm.roce",
    "gemm.bf16_gemv",
    "gemm.bf16_vocab_projection",
    "gemm.blockscaled",
    "gemm.block_fp8_linear",
    "gemm.bmm",
    "gemm.mxfp8_linear",
    "gemm.tensor_fp8_linear",
    "gemm.mla_query_projection",
    "gemm.trellis_linear",
    "gemm.wo_projection",
    "moe.fused_moe",
    "moe.ep_moe",
    "norm.hyperconnection",
    "norm.mhc",
    "quantization.mxfp8",
    "quantization.nvfp4",
    "sequence.ple_hash",
    "sequence.ple_embedding",
    "sequence.ple",
    "sequence.engram",
    "sequence.embedding",
    "sequence.gdn_decode",
    "sequence.kda_prefill",
    "sequence.gdn_prefill",
    "sequence.mtp_feedback",
    "sequence.engram",
)

# A group-level function cannot share its name with an imported child module.
# These registry entries keep their public qualname while their metadata and
# implementation live under a private package.
_OP_MODULE_OVERRIDES: dict[str, str] = {
    "gemm.bmm": "gemm._bmm",
}
_CACHE_CLEAR_OVERRIDES: dict[str, str] = {
    "gemm.bmm": "clear_bmm_caches",
}

_GROUPS = (
    "attention",
    "comm",
    "gemm",
    "moe",
    "norm",
    "quantization",
    "sequence",
)
_LAZY_ROOT_ATTRS: dict[str, tuple[str, str]] = {
    # public name -> (module, attribute)
    "ScratchBufferSpec": ("._lib.scratch", "ScratchBufferSpec"),
