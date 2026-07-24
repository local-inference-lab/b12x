"""Opaque ``torch.custom_op`` wrapper for the FP6 dense linear.

Registers ``sparkinfer::fp6_dense_linear`` so vLLM's ``torch.compile`` path
(Dynamo fullgraph tracing + CUDA-graph capture) treats the whole FP6 linear —
activation quantization, CUTE kernel JIT lookup, and the block-scaled GEMM
launch — as a single opaque node instead of tracing into the JIT machinery
(tempfile / os.getpid), which previously forced ``--enforce-eager``.

Importing this module performs the registration; ``vllm_plugin`` imports it in
``process_weights_after_loading`` so the op exists in every worker before the
model is compiled.
"""
from __future__ import annotations

import torch

from .fp6_dense_weights import dense_fp6_linear_expanded


@torch.library.custom_op("sparkinfer::fp6_dense_linear", mutates_args=())
def fp6_dense_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale_storage: torch.Tensor,
    global_scale: torch.Tensor,
    fmt: str,
    out_features: int,
    in_features: int,
    act_fmt: str = "",
) -> torch.Tensor:
    """``y = x @ W.T`` in MX-FP6; ``x`` is ``(M, in_features)`` bf16.

    ``weight`` is :meth:`FP6DenseWeight.gemm_weight`: either the cached
    1-byte-per-code ``(N, K, 1)`` expansion or, for wide-N layers, the raw
    3:4-packed ``(N, 3K/4)`` codes streamed natively by the GEMM (the layout is
    detected from the K extent). ``scale_storage`` is the flat swizzled UE8M0
    scales. ``fmt`` is the weight sub-format; ``act_fmt`` is the runtime
    activation sub-format (empty string -> same as ``fmt``; ``"e4m3"`` for W6A8).
    Returns ``(M, out_features)`` bf16.
    """
    return dense_fp6_linear_expanded(
        x,
        weight,
        scale_storage,
        global_scale,
        fmt,
        out_features,
        in_features,
        act_fmt=act_fmt if act_fmt else None,
    )


@fp6_dense_linear.register_fake
def _fp6_dense_linear_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale_storage: torch.Tensor,
    global_scale: torch.Tensor,
    fmt: str,
    out_features: int,
    in_features: int,
    act_fmt: str = "",
) -> torch.Tensor:
    return x.new_empty((x.shape[0], out_features), dtype=torch.bfloat16)
