# NVFP4-CSF scale operands in dynamic MoE

Dynamic SiLU expert execution can reconstruct compressed scale operands in its
existing shared-memory slots. It reads the checkpoint's exact scale bytes into
the native MMA layout without first expanding a complete expert scale plane in
global memory. FP4 weight codes, FP32 calibration and A4 arithmetic are unchanged.

vLLM owns checkpoint reading and TP slicing. B12X receives numeric scale planes
through ordinary `prepare_weights`, where it constructs the device index
described below. This representation is an execution layout, not another
checkpoint format. No B12X kernel opens checkpoint files or interprets model
tensor names.

## Preparation and selection

Cooperative operand reconstruction supports A4 SiLU dynamic execution with
hidden dimensions divisible by 128 and intermediate dimensions divisible by
64. The latter includes the Qwen TP2 intermediate dimension of 320. The kernel
uses N128/K128 operand staging; an explicit alternative N tile cannot select it.
A16, micro kernels and materialized-intermediate execution retain complete-plane
expansion and their existing numerical contracts.

The typed preparation query records whether indexed compressed planes exist;
the configuration records whether cooperative reconstruction is selected.
Preparation can time both valid choices. Native weights cannot select a CSF
decoder. Resolution occurs before binding and graph capture, and both required
integer routing ABIs are declared before kernel resolution is frozen.
The query/configuration schema versions are 16/9 and the candidate contract is
18. Serialized tuning data must match those versions.

## Prepared device representation

Each expert projection has one contiguous byte buffer:

| Region | Representation |
| --- | --- |
| Padding header | 1024 zero bytes, with an invalid-atom marker at uint32 word 167 |
| Fixed stream | Each native-order 128-row slab contains 128 base bytes followed by four-bit scale offsets |
| Exception index | Eight uint32 words per 512 native scale bytes: replacement start, four packed uint8 prefix counts, four 32-bit masks, replacement end, invalid-atom marker |
| Replacement words | Complete uint32 scale words in increasing atom/word order, padded to 16-byte alignment |

One index atom describes 128 output rows and four K/16 scales. A normal
N128/K128 operand spans two atoms. A prefix count plus a bitmap population count
locates a replacement word. Each replacement contains four exact scale bytes;
at least one byte is an exception in the checkpoint encoding. Complete-plane
expansion retains a view of the same fixed stream and the checkpoint exception
records, so the fixed stream is not duplicated. The additional exception index
does consume part of the compression saving.

Offsets index replacement words across the prepared projection's experts.
Conversion to global byte addresses uses Int64. Preparation rejects a stream
requiring 2^32 replacement-word indices. None of these offsets changes the
checkpoint's row-relative scale bases or exception encoding.

## Indexed complete-plane expansion

Micro execution and dynamic configurations that require complete scale planes
can use the prepared exception-word index for short route lists. Each 256-thread
CTA covers 4096 output bytes. A thread reconstructs four adjacent uint32 words
from one eight-byte code load and one four-byte row-base load, then writes one
aligned sixteen-byte vector. Matrix dimensions guarantee that even the last
active vector is complete; inactive threads in a partial CTA perform no read
or write.

Each vector lies within one 32-word bitmap group. The thread reads that bitmap
once. If any of its four words require replacement, one prefix/popcount lookup
locates the first replacement; predicated reads advance through the selected
words in order. No arithmetic is performed on exception bytes. Expert-scaled
fixed-stream, metadata, replacement and output addresses use Int64.

Repeated routes are rejected before reconstruction, so only one route owns an
expert's output region. Fused barrier reset and route mutation remain valid
under allocation-free graph replay. The byte tests include empty and dense
exceptions, partial CTAs, invalid and repeated int64 routes, and an expert whose
fixed stream lies beyond 2 GiB and output begins at 4 GiB. Checkpoint storage,
prepared buffers, launch grids and preparation selection contracts are unchanged.

## Operand staging

The native shared scale slot remains 1024 bytes:

| Byte offset | Contents | Bytes |
| ---: | --- | ---: |
| 0 | Four-bit offsets | 512 |
| 512 | Row bases | 128 |
| 640 | Two index records | 64 |
| 704 | Aligned replacement-word prefix | At most 320 |

The producer uses asynchronous bulk copies. Its transaction barrier expects
704 fixed bytes plus the staged replacement payload. Payload bytes are registered
before the fixed transactions can complete. Dense exception tails are read from
global storage, preserving the native shared-memory allocation. A padded row
atom or partial K atom reads the zero header instead of another projection or
expert.

Consumer warps retain their assigned native words in registers before a
consumer-only barrier. They then overwrite the compressed slot with those words;
a second barrier precedes native fragment loads. Shared-memory PTX reads declare
side effects and a memory clobber because the same address contains different
operands on successive pipeline iterations.

When two operands are ready together, consumers retain both operands before
either slot is overwritten and use the same two barriers for the pair. This
applies to fused gate/up computation and to a gate whose row range straddles
two 128-row atoms. A 64-row gate boundary requires only the upper half of the
first atom and the lower half of the second atom; those halves are reconstructed
in their native positions. Unused rows remain compressed and are outside that
MMA's load range. Every consumer reaches both barriers, including consumers
without a word to reconstruct in a half operand.

For a partial K operand, the shared reader reconstructs a word from the staged
bytes and selects zero when its atom has the invalid marker. The selection is
part of the same PTX operation, using its existing metadata address. Padded
atoms have zero codes and exception masks, so reconstruction cannot access an
exception record before the zero selection. Complete operands omit this check
at compilation. This removes a separate validity branch from partial-geometry
scale reads without changing the native padded values.

## Complete-plane expansion for short routes

When preparation selects complete-plane expansion, route lists shorter than
64 entries can use the same exception-word index. Each CTA writes a 4096-byte
native output span, balancing work between gate/up and down projections.
Duplicate checks, inactive-expert behavior and fused barrier initialization are
shared with the slab decoder. Presence-mask and full-expert modes retain the
slab decoder. Live route counts are runtime launch data, not compile-cache keys.

Scale scratch remains caller-owned and shared across serialized layer calls.
Replay neither allocates nor reconstructs the index. Keeping native planes per
layer would violate this memory contract.

## Validation

The [serving report](nvfp4-serving.md) records the component qualification,
whole-model measurements, prepared-memory cost and residual overhead.
The [Qwen TP1/TP2 report](qwen-serving.md) qualifies paired reconstruction,
split-gate halves, and partial-atom zero selection against matched native controls.
[Dynamic MoE invariants](dynamic-moe-correctness.md) records independent native
routing/addressing fixes included in the measured source.
