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
[Dynamic MoE invariants](dynamic-moe-correctness.md) records independent native
routing/addressing fixes included in the measured source.
