"""Metadata stages for draft-round selection reuse inside the QSA operation."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows"])
def _record_kernel(
    positions,
    errors,
    saved_positions,
    saved_errors,
    saved_rows,
    rows,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = row < rows
    tl.store(saved_positions + row, tl.load(positions + row, active, other=-1), active)
    tl.store(saved_errors + row, tl.load(errors + row, active, other=0), active)
    if tl.program_id(0) == 0:
        tl.store(saved_rows, rows)


@torch.library.custom_op(
    "b12x::qsa_record_draft_anchors",
    mutates_args=("saved_positions", "saved_errors", "saved_rows"),
)
def record_anchors(
    positions: torch.Tensor,
    errors: torch.Tensor,
    saved_positions: torch.Tensor,
    saved_errors: torch.Tensor,
    saved_rows: torch.Tensor,
) -> None:
    rows = int(positions.shape[0])
    _record_kernel[(triton.cdiv(rows, 128),)](
        positions,
        errors,
        saved_positions,
        saved_errors,
        saved_rows,
        rows,
        BLOCK=128,
    )


@record_anchors.register_fake
def _record_fake(positions, errors, saved_positions, saved_errors, saved_rows) -> None:
    return None


@triton.jit(do_not_specialize=["rows", "source_capacity", "max_requests"])
def _prepare_kernel(
    source_positions,
    source_errors,
    source_selection,
    source_rows,
    num_source_rows,
    request_ids,
    query_positions,
    selected,
    errors,
    rows,
    source_capacity,
    max_requests,
    WIDTH: tl.constexpr,
    TAIL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    column = tl.arange(0, BLOCK)
    request = tl.load(request_ids + row, row < rows, other=-1).to(tl.int64)
    active = request >= 0
    mapped = active & (request < max_requests)
    source = tl.load(source_rows + request, mapped, other=-1).to(tl.int64)
    source_valid = (
        mapped
        & (source >= 0)
        & (source < source_capacity)
        & (source < tl.load(num_source_rows))
    )
    anchor = tl.load(source_positions + source, source_valid, other=-1)
    source_error = tl.load(source_errors + source, source_valid, other=1)
    position = tl.load(query_positions + row, row < rows, other=-1)
    start = anchor + 1
    valid = (
        source_valid & (anchor >= 0) & (position >= start) & (position < start + TAIL)
    )
    original = tl.load(
        source_selection + source * tl.full((), WIDTH, tl.int64) + column,
        valid & (column < WIDTH),
        other=-1,
    )
    tail = start + column - WIDTH
    value = tl.where(
        column < WIDTH, original, tl.where(valid & (tail <= position), tail, -1)
    )
    tl.store(
        selected + row * tl.full((), WIDTH + TAIL, tl.int64) + column,
        value,
        column < WIDTH + TAIL,
    )
    tl.store(errors + row, tl.where(active, source_error | tl.where(valid, 0, 1), 0))


@torch.library.custom_op(
    "b12x::qsa_prepare_draft_selection", mutates_args=("selected", "errors")
)
def prepare_selection(
    source_positions: torch.Tensor,
    source_errors: torch.Tensor,
    source_selection: torch.Tensor,
    source_rows: torch.Tensor,
    num_source_rows: torch.Tensor,
    request_ids: torch.Tensor,
    query_positions: torch.Tensor,
    selected: torch.Tensor,
    errors: torch.Tensor,
) -> None:
    rows = int(query_positions.shape[0])
    width = int(source_selection.shape[1])
    tail = int(selected.shape[1]) - width
    _prepare_kernel[(rows,)](
        source_positions,
        source_errors,
        source_selection,
        source_rows,
        num_source_rows,
        request_ids,
        query_positions,
        selected,
        errors,
        rows,
        int(source_selection.shape[0]),
        int(source_rows.shape[0]),
        WIDTH=width,
        TAIL=tail,
        BLOCK=triton.next_power_of_2(width + tail),
        num_warps=4,
    )


@prepare_selection.register_fake
def _prepare_fake(
    source_positions,
    source_errors,
    source_selection,
    source_rows,
    num_source_rows,
    request_ids,
    query_positions,
    selected,
    errors,
) -> None:
    return None
