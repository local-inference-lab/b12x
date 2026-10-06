# Lossless scale compression for FP4 experts

MXFP4-CSF and NVFP4-CSF retain FP4 weight nibbles and recover the original
scale bytes before native expert computation. Both represent ordinary scale bytes as a row base plus an
unsigned offset, with exact exception bytes for values outside the interval.

| Representation | Source scale bytes | Offset width | Native expert arithmetic |
| --- | --- | ---: | --- |
| MXFP4-CSF | E8M0 | 1 bit | MXFP4 weights, BF16 or MXFP8 activations |
| NVFP4-CSF | E4M3 | 4 bits | NVFP4 weights, BF16 or calibrated FP4 activations |

One base belongs to a row of block scales. A row contains multiple scales;
the base is not a replacement for the row's entire scale tensor.

Use `b12x.moe.fused_moe.Mxfp4CsfWeights` or `Nvfp4CsfWeights` with the
ordinary `plan_weights` / `prepare_weights` interface. Supply decoded scale
scratch buffers owned by the caller. Serialized layer executions may reuse
these buffers. Concurrent execution streams require independent scratch.

Pass compressed CPU scale planes as `CsfScalePlanes(fixed, exceptions)` in
`w13_scales` and `w2_scales`. Each tuple contains one rank-local tensor per
expert, in canonical 16-row slab order: fixed bytes are `torch.uint8` and
exception records are `torch.uint32`. Geometry and codec come from the weight
plan. MXFP4 uses 32-channel scale groups; NVFP4 uses 16-channel groups and retains
the source FP32 scalar weight and activation calibration.

`prepare_weights` uploads the compressed planes, partitions exception ranges,
and applies the planned scale layout. NVFP4 uses the ordinary packed-weight
preparation, including its W4A16 normalization when A16 is selected. MXFP4 uses
W4A16 preparation for BF16 activations and native W4A8 scale packing for MXFP8
activations.
These operations happen during weight preparation, before graph capture.
Callers with already resident scale batches may also pass those batches.

vLLM owns CSF manifests, model inventories, tensor names, shard lifetimes and
TP slicing. Its `mxfp4_csf_loader` and `nvfp4_csf_loader` readers slice packed
weight views before materialization and produce rank-local weight bundles.
B12X has no CSF checkpoint reader or model-file discovery API. Updating this
integration requires the matching B12X `CsfScalePlanes` support; serialized
checkpoints, tensor bytes and launch flags are unchanged.

GPU decoding writes the native scale layout directly. Exception ranges are
partitioned at load time; a thread block patches only its output rows.
The paired NVFP4 decoder expands FC1 and FC2 in one launch. Preparation
retains the integer routing ABIs before graph capture, and replay uses
caller-owned allocations.

## Serialized formats

The vLLM checkpoint readers accept `lil-mxfp4-csf-checkpoint/1` and
`lil-nvfp4-csf-checkpoint/1`, respectively. Scale tensor components use
`.mxfp4_csf_fixed` / `.mxfp4_csf_exceptions` or
`.nvfp4_csf_fixed` / `.nvfp4_csf_exceptions` suffixes. Predecessor schemas
and API aliases are not accepted. Migrate their headers, manifests and
receipts before loading; the compressed payload values do not need refitting.

MXFP4 and NVFP4 describe different source arithmetic. A shared compressed
storage concept does not make their weight, scale, or activation formats
interchangeable. The NVFP4 checkpoint reader supports GLM-5.3-Flash and
Qwen3.8-Flash-Next geometry (TP extents must contain a multiple of 64
intermediate channels);
the MXFP4 reader supports Kimi-K3 and DeepSeek-V4.1-Flash geometry.
