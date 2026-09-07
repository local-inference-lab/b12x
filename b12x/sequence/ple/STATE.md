# PLE convolution-state contract

The PLE layer owns no runtime storage. The integration allocates one BF16 tensor
with shape
`[max_state_slots, streams * hidden_size, state_length + max_speculative_tokens]`,
where `state_length = dilation * (kernel_size - 1)`, and passes it to `bind`.
Each state-slot payload must be dense: its inner strides are
`[state_length + max_speculative_tokens, 1]`. The outer state-slot stride may be
larger than the payload extent so an integration can bind page-aligned hybrid
cache storage without copying. Outer strides smaller than the payload extent,
and non-dense inner layouts, are unsupported because live state slots must not
overlap.

For each live slot, positions `[0:state_length]` are the committed convolution
window ordered oldest to newest. Decode treats the newest entry as the
guaranteed first token from the preceding verification batch. Positions
`[state_length:]` retain normalized PLE inputs for the preceding batch's
additional speculative candidates, also ordered oldest to newest.

On a decode call, `num_accepted_tokens[r]` advances the prior state by
`num_accepted_tokens[r] - 1` entries. The count includes the guaranteed token
from the preceding verification interval, so one is the neutral non-speculative
value. The first normalized PLE input in the current query becomes the new
guaranteed entry, and remaining query inputs are written to the speculative
tail. Larger accepted counts promote the corresponding retained candidates
before committing the current first input.

Every live decode query length must be at most
`max_speculative_tokens + 1`, and every accepted count must be in
`[1, max_speculative_tokens + 1]`. The runtime kernels validate these bounds on
the device; they do not clamp them.

Prefill consumes the caller-provided base window, or zeros for a fresh slot,
persists the newest `state_length` normalized inputs, and clears the speculative
tail. `state_is_fresh[r]` makes an existing physical slot read as zero without
requiring the integration to clear recycled storage first.

A mixed plan binds a fixed-capacity device boolean `request_is_prefill` with one
entry per request row. `run_mixed` applies prefill semantics to true live rows
and decode semantics to false live rows without partitioning or reordering the
packed token tensor. Decode query-length and accepted-token validation applies
only to false live rows; prefill and inactive rows ignore
`num_accepted_tokens`.

A live request with zero query tokens leaves its entire physical state slot
unchanged in both decode and prefill, including the speculative tail. Distinct
live requests must use distinct nonnegative state-slot IDs; duplicate real slots
fail closed before any state mutation.

A `state_slot_ids[r]` value of `-1` is a dummy sink for CUDA-graph padding. Its
tokens produce zero output and no state mutation. Other negative values and
values at or above `max_state_slots` are invalid.

`run_decode`, `run_prefill`, and `run_mixed` reset and then populate
`binding.error_code`, a one-element device `int32` scratch view. A nonzero code
means the entire launch wrote zero output and did not mutate state. The masks
are `1` for capacity metadata, `2` for packed-query boundaries, `4` for decode
query length, `8` for accepted-token count, and `16` for state-slot ID. Serving
integrations must inspect this value after stream or graph completion and treat
any nonzero value as fatal.
The additional mask `32` reports duplicate nonnegative live state-slot IDs.

Norm weights bind as flat `[streams * hidden_size]` tensors. A checkpoint
depthwise-convolution weight shaped
`[streams * hidden_size, 1, kernel_size]` must be squeezed to the contiguous
runtime view `[streams * hidden_size, kernel_size]` before binding.
