"""GPU-free checkpoint metadata oracle for transactional KDA prefill."""

from __future__ import annotations


def validate_metadata(
    *,
    cu_seqlens,
    initial_state_indices,
    final_state_indices,
    checkpoint_state_indices,
    checkpoint_offsets,
    num_seqs,
    num_tokens,
    token_capacity,
    seq_capacity,
    state_slots,
    chunk=16,
    null_state_index=None,
    max_checkpoints=1,
):
    """Validate packed ownership and return spans without reading state tensors.

    Nonpositive offsets disable an entry. A null checkpoint never owns storage;
    its positive offset still must be aligned and within the request. Enabled,
    non-null exports need distinct offsets and globally unique destinations.
    Initial may alias only own final.
    """
    if type(max_checkpoints) is not int or max_checkpoints not in (1, 2, 4):
        raise ValueError("max_checkpoints must be 1, 2 or 4")
    if not 0 <= num_seqs <= seq_capacity or not 0 <= num_tokens <= token_capacity:
        raise ValueError("live counts exceed capacities")
    if int(cu_seqlens[0]) != 0 or int(cu_seqlens[num_seqs]) != num_tokens:
        raise ValueError("packed boundaries do not match live tokens")
    spans = []
    writes = set()

    def null(slot):
        return null_state_index is not None and slot == null_state_index

    def check_slot(slot):
        if not null(slot) and not 0 <= slot < state_slots:
            raise IndexError("state slot out of range")

    def write(slot):
        if null(slot):
            return
        check_slot(slot)
        if slot in writes:
            raise ValueError("duplicate checkpoint/final write slot")
        writes.add(slot)

    for seq in range(num_seqs):
        start, end = int(cu_seqlens[seq]), int(cu_seqlens[seq + 1])
        if not 0 <= start <= end <= num_tokens:
            raise ValueError("invalid packed sequence interval")
        spans.append((start, end))
        check_slot(int(initial_state_indices[seq]))
        write(int(final_state_indices[seq]))
        seen_offsets = set()
        for cp in range(max_checkpoints):
            slot = int(
                checkpoint_state_indices[seq]
                if max_checkpoints == 1
                else checkpoint_state_indices[seq][cp]
            )
            offset = int(
                checkpoint_offsets[seq]
                if max_checkpoints == 1
                else checkpoint_offsets[seq][cp]
            )
            if offset > end - start or (offset > 0 and offset % chunk):
                raise ValueError("checkpoint offset out of bounds or unaligned")
            if offset > 0 and not null(slot):
                if offset in seen_offsets:
                    raise ValueError("active checkpoints have the same offset")
                seen_offsets.add(offset)
                write(slot)
    for seq in range(num_seqs):
        initial, final = int(initial_state_indices[seq]), int(final_state_indices[seq])
        if not null(initial) and initial in writes - (
            {final} if not null(final) else set()
        ):
            raise ValueError(
                "initial slot conflicts with a checkpoint or another request write"
            )
    return spans
