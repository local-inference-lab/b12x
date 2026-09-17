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
`[1, max_speculative_tokens + 1]`. The kernels neither check nor clamp these
bounds; the integration must satisfy them.

Prefill consumes the caller-provided base window, or zeros for a fresh slot,
persists the newest `state_length` normalized inputs, and clears the speculative
tail. `state_is_fresh[r]` makes an existing physical slot read as zero without
requiring the integration to clear recycled storage first.

`export_checkpoint` copies an internal prefill window from the immediately
preceding `run_mixed` invocation. Its int32 offsets and int64 destination slots
have the plan's request capacity. An offset is relative to the request's query
start and must lie strictly inside that query. Zero offsets, negative slots,
decode rows and empty rows leave state unchanged. Each enabled destination must
be a valid, distinct pool slot separate from every live input/output slot.
The export kernel checks these destination constraints on the device before
copying. An out-of-range destination or one matching any live state-slot ID
is skipped. If enabled exports share a destination, all of those exports are
skipped; independent valid exports still run. Disabled and inactive exports
do not reserve destinations. These checks read runtime metadata on every
CUDA graph replay and require no additional storage or host readback.
Offsets shorter than the convolution window include the saved input history;
the speculative tail is cleared. Export must finish before the mixed binding's
scratch is reused. Kernel preparation includes the export path, and runtime
offsets and slot IDs may change during CUDA graph replay.

A mixed plan binds a fixed-capacity device boolean `request_is_prefill` with one
entry per request row. `run_mixed` applies prefill semantics to true live rows
and decode semantics to false live rows without partitioning or reordering the
packed token tensor. The decode query-length and accepted-token bounds apply
only to false live rows; prefill and inactive rows ignore
`num_accepted_tokens`.

A live request with zero query tokens leaves its entire physical state slot
unchanged in both decode and prefill, including the speculative tail. Distinct
live requests must use distinct nonnegative state-slot IDs.

A `state_slot_ids[r]` value of `-1` is a dummy sink for CUDA-graph padding. Its
tokens produce zero output and no state mutation. Other negative values and
values at or above `max_state_slots` are invalid.

The mixed, prefill and decode kernels read `num_seqs`, `num_tokens`, `query_start_loc`,
`state_slot_ids`, and `num_accepted_tokens` from the device without checking
them against the planned capacities or against each other; the only runtime
masks are the live token count, the live request count, and the `-1` slot
sink. Metadata outside the contract above reads or writes out of bounds.

Norm weights bind as flat `[streams * hidden_size]` tensors. A checkpoint
depthwise-convolution weight shaped
`[streams * hidden_size, 1, kernel_size]` must be squeezed to the contiguous
runtime view `[streams * hidden_size, kernel_size]` before binding.
