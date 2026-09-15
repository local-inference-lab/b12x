"""Opaque functional block-FP8 execution through retained preparation state."""
from __future__ import annotations

import torch

from b12x._lib.utils import cuda_stream_to_int
from b12x.preparation import plan_from_handle, require_prepared


def _execute(
    source: torch.Tensor, values: torch.Tensor, scale_rows: torch.Tensor,
    scale_mma: torch.Tensor, bias: torch.Tensor | None,
    workspace: torch.Tensor | None, stream: int | None, plan_handle: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    from b12x.gemm._shared.block_fp8 import BlockFP8LinearWeight
    from b12x.gemm._shared.wo_mxfp8 import MXFP8Rows

    state = require_prepared(plan_from_handle(plan_handle), "gemm.block_fp8_linear", source.device)
    query = state.query
    if output_dtype != getattr(torch, query.output_dtype):
        raise ValueError("block-FP8 output dtype differs from preparation")
    weight = BlockFP8LinearWeight(
        MXFP8Rows(values, scale_rows, scale_mma.view(torch.float8_e8m0fnu)),
        query.in_features, query.out_features, (query.weight_block_size,) * 2,
    )
    return state.run(source, weight, bias=bias, workspace=workspace, stream=stream)


_run = torch.library.custom_op("b12x::block_fp8_prepared_workspace", mutates_args=("workspace",))(_execute)


@torch.library.custom_op("b12x::block_fp8_prepared", mutates_args=())
def _functional(source: torch.Tensor, values: torch.Tensor, scale_rows: torch.Tensor,
                scale_mma: torch.Tensor, bias: torch.Tensor | None,
                stream: int | None, plan_handle: int, output_dtype: torch.dtype) -> torch.Tensor:
    return _execute(source, values, scale_rows, scale_mma, bias, None, stream, plan_handle, output_dtype)


@_functional.register_fake
def _functional_fake(source: torch.Tensor, values: torch.Tensor, scale_rows: torch.Tensor, scale_mma: torch.Tensor, bias: torch.Tensor | None, stream: int | None, plan_handle: int, output_dtype: torch.dtype) -> torch.Tensor:
    return source.new_empty((*source.shape[:-1], values.shape[0]), dtype=output_dtype)


@_run.register_fake
def _fake(source: torch.Tensor, values: torch.Tensor, scale_rows: torch.Tensor, scale_mma: torch.Tensor, bias: torch.Tensor | None, workspace: torch.Tensor | None, stream: int | None, plan_handle: int, output_dtype: torch.dtype) -> torch.Tensor:
    return source.new_empty((*source.shape[:-1], values.shape[0]), dtype=output_dtype)


def run_functional(source, weight, *, plan, bias=None, workspace=None, stream=None):
    query = plan.query
    if (weight.in_features != query.in_features or weight.out_features != query.out_features
            or weight.block_size != (query.weight_block_size,) * 2):
        raise ValueError("block-FP8 weight geometry differs from preparation")
    packed = weight.weight
    args = (source, packed.values, packed.scale_rows, packed.scale_mma.view(torch.uint8), bias)
    options = (cuda_stream_to_int(stream), plan.handle, getattr(torch, query.output_dtype))
    if workspace is None:
        return torch.ops.b12x.block_fp8_prepared(*args, *options)
    return torch.ops.b12x.block_fp8_prepared_workspace(*args, workspace, *options)
