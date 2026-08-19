"""Native BF16 low-rank correction for packed dense Trellis weights."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_BLOCK_M = 16
_BLOCK_N = 128
_BLOCK_K = 64
_WARMED_SIGNATURES: set[tuple[int, int, int, int]] = set()
_WARMED_PAIR_PROJECT_SIGNATURES: set[tuple[int, int, int, int]] = set()
_WARMED_PAIR_ADD_SIGNATURES: set[tuple[int, int, int, int]] = set()


@triton.jit
def _project_kernel(
    x,
    a_t,
    hidden,
    rows,
    INPUT_FEATURES: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_start = tl.program_id(0) * BLOCK_M
    rows_offset = row_start + tl.arange(0, BLOCK_M)
    rank_offset = tl.arange(0, RANK)
    accumulator = tl.zeros((BLOCK_M, RANK), dtype=tl.float32)

    for feature_start in tl.range(
        0,
        INPUT_FEATURES,
        BLOCK_K,
        num_stages=2,
    ):
        feature_offset = feature_start + tl.arange(0, BLOCK_K)
        x_offsets = (
            rows_offset[:, None].to(tl.int64) * INPUT_FEATURES + feature_offset[None, :]
        )
        a_offsets = (
            rank_offset[None, :].to(tl.int64) * INPUT_FEATURES + feature_offset[:, None]
        )
        x_tile = tl.load(
            x + x_offsets,
            mask=(rows_offset[:, None] < rows)
            & (feature_offset[None, :] < INPUT_FEATURES),
            other=0.0,
        )
        a_tile = tl.load(
            a_t + a_offsets,
            mask=feature_offset[:, None] < INPUT_FEATURES,
            other=0.0,
        )
        accumulator = tl.dot(x_tile, a_tile, accumulator)

    hidden_offsets = rows_offset[:, None].to(tl.int64) * RANK + rank_offset[None, :]
    tl.store(
        hidden + hidden_offsets,
        accumulator.to(tl.bfloat16),
        mask=rows_offset[:, None] < rows,
    )


@triton.jit
def _add_kernel(
    hidden,
    b,
    output,
    rows,
    OUTPUT_FEATURES: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows_offset = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    output_offset = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    rank_offset = tl.arange(0, RANK)
    hidden_offsets = rows_offset[:, None].to(tl.int64) * RANK + rank_offset[None, :]
    factor_offsets = rank_offset[:, None] + output_offset[None, :].to(tl.int64) * RANK
    hidden_tile = tl.load(
        hidden + hidden_offsets,
        mask=rows_offset[:, None] < rows,
        other=0.0,
    )
    factor_tile = tl.load(
        b + factor_offsets,
        mask=output_offset[None, :] < OUTPUT_FEATURES,
        other=0.0,
    )
    correction = tl.dot(hidden_tile, factor_tile).to(tl.bfloat16)
    output_offsets = (
        rows_offset[:, None].to(tl.int64) * OUTPUT_FEATURES + output_offset[None, :]
    )
    mask = (rows_offset[:, None] < rows) & (output_offset[None, :] < OUTPUT_FEATURES)
    base = tl.load(output + output_offsets, mask=mask, other=0.0)
    result = base.to(tl.float32) + correction.to(tl.float32)
    tl.store(output + output_offsets, result, mask=mask)


@triton.jit
def _project_pair_kernel(
    x,
    a_t,
    hidden,
    rows,
    INPUT_FEATURES: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_start = tl.program_id(0) * BLOCK_M
    rows_offset = row_start + tl.arange(0, BLOCK_M)
    rank_offset = tl.arange(0, RANK)
    accumulator_0 = tl.zeros((BLOCK_M, RANK), dtype=tl.float32)
    accumulator_1 = tl.zeros((BLOCK_M, RANK), dtype=tl.float32)

    for feature_start in tl.range(
        0,
        INPUT_FEATURES,
        BLOCK_K,
        num_stages=2,
    ):
        feature_offset = feature_start + tl.arange(0, BLOCK_K)
        x_offsets = (
            rows_offset[:, None].to(tl.int64) * INPUT_FEATURES + feature_offset[None, :]
        )
        a_offsets = (
            rank_offset[None, :].to(tl.int64) * INPUT_FEATURES + feature_offset[:, None]
        )
        x_tile = tl.load(
            x + x_offsets,
            mask=(rows_offset[:, None] < rows)
            & (feature_offset[None, :] < INPUT_FEATURES),
            other=0.0,
        )
        a_tile_0 = tl.load(
            a_t + a_offsets,
            mask=feature_offset[:, None] < INPUT_FEATURES,
            other=0.0,
        )
        a_tile_1 = tl.load(
            a_t + RANK * INPUT_FEATURES + a_offsets,
            mask=feature_offset[:, None] < INPUT_FEATURES,
            other=0.0,
        )
        accumulator_0 = tl.dot(x_tile, a_tile_0, accumulator_0)
        accumulator_1 = tl.dot(x_tile, a_tile_1, accumulator_1)

    hidden_offsets = rows_offset[:, None].to(tl.int64) * RANK + rank_offset[None, :]
    mask = rows_offset[:, None] < rows
    tl.store(hidden + hidden_offsets, accumulator_0.to(tl.bfloat16), mask=mask)
    tl.store(
        hidden + rows * RANK + hidden_offsets,
        accumulator_1.to(tl.bfloat16),
        mask=mask,
    )


@triton.jit
def _add_pair_kernel(
    hidden,
    b,
    output,
    rows,
    OUTPUT_FEATURES: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows_offset = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    blocks_per_projection = OUTPUT_FEATURES // BLOCK_N
    projection = tl.program_id(1) // blocks_per_projection
    projection_block = tl.program_id(1) % blocks_per_projection
    output_offset = projection_block * BLOCK_N + tl.arange(0, BLOCK_N)
    rank_offset = tl.arange(0, RANK)
    hidden_offsets = (
        projection.to(tl.int64) * rows * RANK
        + rows_offset[:, None].to(tl.int64) * RANK
        + rank_offset[None, :]
    )
    factor_offsets = (
        projection.to(tl.int64) * OUTPUT_FEATURES * RANK
        + output_offset[None, :].to(tl.int64) * RANK
        + rank_offset[:, None]
    )
    hidden_tile = tl.load(
        hidden + hidden_offsets,
        mask=rows_offset[:, None] < rows,
        other=0.0,
    )
    factor_tile = tl.load(b + factor_offsets)
    correction = tl.dot(hidden_tile, factor_tile).to(tl.bfloat16)
    fused_width = 2 * OUTPUT_FEATURES
    output_offsets = (
        rows_offset[:, None].to(tl.int64) * fused_width
        + projection.to(tl.int64) * OUTPUT_FEATURES
        + output_offset[None, :]
    )
    mask = rows_offset[:, None] < rows
    base = tl.load(output + output_offsets, mask=mask, other=0.0)
    result = base.to(tl.float32) + correction.to(tl.float32)
    tl.store(output + output_offsets, result, mask=mask)


def _device_index(tensor: torch.Tensor) -> int:
    index = tensor.device.index
    return torch.cuda.current_device() if index is None else int(index)


def _signature(
    x: torch.Tensor,
    a_t: torch.Tensor,
    b: torch.Tensor,
) -> tuple[int, int, int, int]:
    return (
        _device_index(x),
        int(x.shape[1]),
        int(b.shape[0]),
        int(a_t.shape[0]),
    )


def _launch(
    x: torch.Tensor,
    a_t: torch.Tensor,
    b: torch.Tensor,
    hidden: torch.Tensor,
    output: torch.Tensor,
) -> None:
    rows = int(x.shape[0])
    input_features = int(x.shape[1])
    output_features = int(b.shape[0])
    rank = int(a_t.shape[0])
    signature = _signature(x, a_t, b)
    if torch.cuda.is_current_stream_capturing() and signature not in _WARMED_SIGNATURES:
        raise RuntimeError(
            "additive Trellis low-rank kernels are not initialized for CUDA "
            "graph capture; run this exact factor geometry eagerly first"
        )
    _project_kernel[(triton.cdiv(rows, _BLOCK_M),)](
        x,
        a_t,
        hidden,
        rows,
        INPUT_FEATURES=input_features,
        RANK=rank,
        BLOCK_M=_BLOCK_M,
        BLOCK_K=_BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    _add_kernel[(triton.cdiv(rows, _BLOCK_M), triton.cdiv(output_features, _BLOCK_N))](
        hidden,
        b,
        output,
        rows,
        OUTPUT_FEATURES=output_features,
        RANK=rank,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        num_warps=4,
        num_stages=2,
    )
    _WARMED_SIGNATURES.add(signature)


def _launch_pair_project(
    x: torch.Tensor,
    a_t: torch.Tensor,
    hidden: torch.Tensor,
) -> None:
    rows = int(x.shape[0])
    input_features = int(x.shape[1])
    rank = int(a_t.shape[1])
    signature = (_device_index(x), input_features, rank, rows)
    if (
        torch.cuda.is_current_stream_capturing()
        and signature not in _WARMED_PAIR_PROJECT_SIGNATURES
    ):
        raise RuntimeError(
            "paired additive Trellis projection is not initialized for CUDA "
            "graph capture; run this exact factor geometry eagerly first"
        )
    _project_pair_kernel[(triton.cdiv(rows, _BLOCK_M),)](
        x,
        a_t,
        hidden,
        rows,
        INPUT_FEATURES=input_features,
        RANK=rank,
        BLOCK_M=_BLOCK_M,
        BLOCK_K=_BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    _WARMED_PAIR_PROJECT_SIGNATURES.add(signature)


def _launch_pair_add(
    hidden: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
) -> None:
    rows = int(hidden.shape[1])
    output_features = int(b.shape[1])
    rank = int(hidden.shape[2])
    signature = (_device_index(hidden), output_features, rank, rows)
    if (
        torch.cuda.is_current_stream_capturing()
        and signature not in _WARMED_PAIR_ADD_SIGNATURES
    ):
        raise RuntimeError(
            "paired additive Trellis output is not initialized for CUDA graph "
            "capture; run this exact factor geometry eagerly first"
        )
    _add_pair_kernel[
        (
            triton.cdiv(rows, _BLOCK_M),
            2 * triton.cdiv(output_features, _BLOCK_N),
        )
    ](
        hidden,
        b,
        output,
        rows,
        OUTPUT_FEATURES=output_features,
        RANK=rank,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        num_warps=4,
        num_stages=2,
    )
    _WARMED_PAIR_ADD_SIGNATURES.add(signature)


@torch.library.custom_op(
    "b12x::trellis_low_rank_additive",
    mutates_args=("hidden", "output"),
)
def _low_rank_additive_op(
    x: torch.Tensor,
    a_t: torch.Tensor,
    b: torch.Tensor,
    hidden: torch.Tensor,
    output: torch.Tensor,
) -> None:
    _launch(x, a_t, b, hidden, output)


@_low_rank_additive_op.register_fake
def _low_rank_additive_fake(
    x: torch.Tensor,
    a_t: torch.Tensor,
    b: torch.Tensor,
    hidden: torch.Tensor,
    output: torch.Tensor,
) -> None:
    del x, a_t, b, hidden, output


@torch.library.custom_op(
    "b12x::trellis_low_rank_pair_project",
    mutates_args=("hidden",),
)
def _low_rank_pair_project_op(
    x: torch.Tensor,
    a_t: torch.Tensor,
    hidden: torch.Tensor,
) -> None:
    _launch_pair_project(x, a_t, hidden)


@_low_rank_pair_project_op.register_fake
def _low_rank_pair_project_fake(
    x: torch.Tensor,
    a_t: torch.Tensor,
    hidden: torch.Tensor,
) -> None:
    del x, a_t, hidden


@torch.library.custom_op(
    "b12x::trellis_low_rank_pair_add",
    mutates_args=("output",),
)
def _low_rank_pair_add_op(
    hidden: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
) -> None:
    _launch_pair_add(hidden, b, output)


@_low_rank_pair_add_op.register_fake
def _low_rank_pair_add_fake(
    hidden: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
) -> None:
    del hidden, b, output


def run_low_rank_additive(
    x: torch.Tensor,
    a_t: torch.Tensor,
    b: torch.Tensor,
    hidden: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Project through BF16 rank factors and add into caller-owned output."""
    if (
        x.ndim != 2
        or a_t.ndim != 2
        or b.ndim != 2
        or hidden.ndim != 2
        or output.ndim != 2
    ):
        raise ValueError("additive low-rank tensors must all be rank two")
    rows, input_features = (int(value) for value in x.shape)
    rank = int(a_t.shape[0])
    output_features = int(b.shape[0])
    if (
        tuple(a_t.shape) != (rank, input_features)
        or tuple(b.shape) != (output_features, rank)
        or tuple(hidden.shape) != (rows, rank)
        or tuple(output.shape) != (rows, output_features)
        or rows <= 0
        or rank != 16
        or input_features % _BLOCK_K != 0
        or output_features % _BLOCK_N != 0
    ):
        raise ValueError(
            "additive low-rank execution requires rank 16, K divisible by 64, "
            "N divisible by 128, and mutually compatible tensor geometry"
        )
    tensors = (x, a_t, b, hidden, output)
    if any(
        tensor.dtype != torch.bfloat16
        or not tensor.is_cuda
        or tensor.device != x.device
        or not tensor.is_contiguous()
        or int(tensor.data_ptr()) % 16 != 0
        for tensor in tensors
    ):
        raise ValueError(
            "additive low-rank tensors must be contiguous, aligned BF16 CUDA storage"
        )
    torch.ops.b12x.trellis_low_rank_additive(x, a_t, b, hidden, output)


def _validate_pair_storage(
    x: torch.Tensor,
    a_t: torch.Tensor,
    b: torch.Tensor,
    hidden: torch.Tensor,
    output: torch.Tensor,
) -> tuple[int, int, int, int]:
    if (
        x.ndim != 2
        or a_t.ndim != 3
        or b.ndim != 3
        or hidden.ndim != 3
        or output.ndim != 2
    ):
        raise ValueError("paired additive low-rank tensors have incompatible ranks")
    rows, input_features = (int(value) for value in x.shape)
    output_features = int(b.shape[1])
    rank = int(a_t.shape[1])
    if (
        tuple(a_t.shape) != (2, rank, input_features)
        or tuple(b.shape) != (2, output_features, rank)
        or tuple(hidden.shape) != (2, rows, rank)
        or tuple(output.shape) != (rows, 2 * output_features)
        or rows <= 0
        or rank != 16
        or input_features % _BLOCK_K != 0
        or output_features % _BLOCK_N != 0
    ):
        raise ValueError(
            "paired additive execution requires rank 16, K divisible by 64, "
            "N divisible by 128, and mutually compatible tensor geometry"
        )
    tensors = (x, a_t, b, hidden, output)
    if any(
        tensor.dtype != torch.bfloat16
        or not tensor.is_cuda
        or tensor.device != x.device
        or not tensor.is_contiguous()
        or int(tensor.data_ptr()) % 16 != 0
        for tensor in tensors
    ):
        raise ValueError(
            "paired additive tensors must be contiguous, aligned BF16 CUDA storage"
        )
    return rows, input_features, output_features, rank


def run_low_rank_pair_project(
    x: torch.Tensor,
    a_t: torch.Tensor,
    b: torch.Tensor,
    hidden: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Project one input through two rank-16 A factors in one native launch."""
    _validate_pair_storage(x, a_t, b, hidden, output)
    torch.ops.b12x.trellis_low_rank_pair_project(x, a_t, hidden)


def run_low_rank_pair_add(
    x: torch.Tensor,
    a_t: torch.Tensor,
    b: torch.Tensor,
    hidden: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Add two rank-16 B projections to one concatenated gate/up output."""
    _validate_pair_storage(x, a_t, b, hidden, output)
    torch.ops.b12x.trellis_low_rank_pair_add(hidden, b, output)


def clear_low_rank_caches() -> None:
    """Require an eager warmup before any later CUDA-graph capture."""
    _WARMED_SIGNATURES.clear()
    _WARMED_PAIR_PROJECT_SIGNATURES.clear()
    _WARMED_PAIR_ADD_SIGNATURES.clear()


__all__ = [
    "clear_low_rank_caches",
    "run_low_rank_additive",
    "run_low_rank_pair_add",
    "run_low_rank_pair_project",
]
