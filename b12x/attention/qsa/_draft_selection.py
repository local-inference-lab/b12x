"""Metadata stages for draft-round selection reuse inside the QSA operation."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@torch.library.custom_op("b12x::qsa_validate_draft_buffers", mutates_args=("mutable",))
def validate_buffers(mutable: list[torch.Tensor], inputs: list[torch.Tensor]) -> None:
    """Check addresses before writes, outside Dynamo's symbolic tracing.

    The mutation annotation orders validation before consumers of these buffers.
    The check launches no GPU work and does not change buffer contents.
    """
    from ._contract import _require_mutation_alias_contract

    _require_mutation_alias_contract(
        mutable=tuple((f"draft buffer {i}", t) for i, t in enumerate(mutable)),
        read_only=tuple((f"draft input {i}", t) for i, t in enumerate(inputs)),
    )


@validate_buffers.register_fake
def _validate_fake(mutable, inputs) -> None:
    return None


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
    scratch: torch.Tensor,
    errors_offset: int,
    saved_positions: torch.Tensor,
    saved_errors: torch.Tensor,
    saved_rows: torch.Tensor,
) -> None:
    from ._contract import _scratch_view

    rows = int(positions.shape[0])
    errors = _scratch_view(
        scratch, offset_bytes=errors_offset, shape=(rows,), dtype=torch.int32
    )
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
def _record_fake(
    positions, scratch, errors_offset, saved_positions, saved_errors, saved_rows
) -> None:
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
