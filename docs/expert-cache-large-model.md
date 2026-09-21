# Larger-than-VRAM checkpoint qualification

Status: **metadata-compatible candidate; complete-model execution unqualified**.
The candidate is
[`nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4` at revision
`8fb2682f136cf94d932a498f18cb1e428832a912`](https://huggingface.co/nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4/tree/8fb2682f136cf94d932a498f18cb1e428832a912).
The [SM120 reference guide](expert-cache-reference.md) remains the qualified
Qwen3-30B configuration. This document does not extend its serving claim to
Qwen3-Next.

## Compatibility boundary

The target has 48 routed MoE layers, each with 512 experts, hidden size 2048,
intermediate size 512 and normalized top-10 routing. Each layer also has a
resident shared MLP and a separate BF16 sigmoid gate. The maintained companion
implements its 36 Gated DeltaNet layers and 12 full-attention layers; cache
integration does not replace attention, recurrent state or shared computation.

The routed recipe remains BF16 activations, native ModelOpt NVFP4 weights,
whole-K W4A16 and actual output route weights. Ordinary shared/dense quantized
linears retain the engine's ModelOpt backend; the routed W4A16 declaration is
not a claim that every model operation uses W4A16. Router BF16 reduction,
shared gating, output addition order and greedy generation remain unchanged.

`MoERunner` already executes and synchronizes external shared experts. The
cache method accepts that wrapper and returns routed output only. The runner
combines shared and routed outputs in its established order. Caller-owned
routed scratch remains rejected. Shared linears requiring coordinated external
scratch still fail through `prepare_workspace`; this change does not grant an
unimplemented scratch-sharing contract. Normal ModelOpt NVFP4 FlashInfer
linears report no caller workspace. Other linear backend selections require
their own qualification.

`Qwen3NextForCausalLM` explicitly excludes the `mtp.` prefix through its weight
mapper. These optional draft tensors are not target-model weights. Speculation,
LoRA, TP/DP/PP expansion, EP and DBO remain outside the cache lane.

## Metadata and required memory

The bounded header audit covers all 11 shards and 297,728 tensors. It requires
all 296,175 target tensors, including every routed projection, scale, shared
gate, attention projection and recurrent parameter. Missing target tensors,
unexpected target names, quantization-exclusion conflicts, shape/dtype changes,
corrupt offsets and inconsistent index entries fail before source allocation.
Header coverage is not a report of tensors actually consumed by a running
loader; that remains a physical qualification gate.

| Storage class | Tensor bytes | GiB |
|---|---:|---:|
| Required routed packed weights alone | 38,654,705,664 | 36.000 |
| Required routed weights plus scales | 43,487,133,696 | 40.501 |
| Required shared MLPs and their gates | 85,132,416 | 0.079 |
| Required router weights | 100,663,296 | 0.094 |
| Required full-attention tensors | 509,620,416 | 0.475 |
| Required recurrent/linear-attention tensors | 1,993,619,232 | 1.857 |
| Required embedding, head and other norms | 1,245,057,024 | 1.160 |
| **Complete required target** | **47,421,226,080** | **44.164** |
| Optional MTP, excluded from target proof | 3,300,942,848 | 3.074 |

Ripper's physical SM120 device exposes 24,467 MiB. Required routed packed weights
alone exceed it, without KV, graphs, dummy allocations or an artificially small
utilization limit. This is a genuine capacity constraint, unlike the admitted
all-resident Qwen3-30B control.

The cache retains CPU source parameters and a separate complete pinned/mapped
canonical representation. Their lower bound is **86,973,874,176 bytes
(81.001 GiB)**: 43,487,133,696 source bytes and 43,486,740,480 canonical bytes.
Alignment is included in the canonical formula. This is not a complete host
admission: loader/header objects, shard views, scale conversion, retained scalar
copies, update journals and safety reserves add storage. The companion rejects
an expert-source-plus-backing lower bound exceeding the explicit host envelope
before hashing full weights or constructing model layers. In particular, the
40-GiB Qwen3-30B envelope is invalid here.

The inspected host had about 495 GiB available RAM and 1 TB free on its existing
model volume. Its reported locked-memory limit was 65,998,848 KiB. These are
preflight observations, not evidence that a complete mapped allocation succeeds.
No OS limits were changed. A final resident budget must use live admission after
ordinary/shared loading, with measured conversion peaks, KV/recurrent state,
graph storage, private workspaces, counters and explicit safety headroom. No
resident count or full-model memory peak is qualified from headers alone.

## Reproduce the preflight

The metadata command downloads configuration, the tensor index and bounded
safetensors headers only. It requires an immutable revision, validates HTTP range
responses and refuses a full-shard fallback. Use a fresh output directory.
`DEVICE_BYTES` and `HOST_BYTES` are explicit observed capacities/envelopes, not
model labels. The resulting report preserves the command, source identity,
metadata hashes, shard header hashes, byte classes and uncompleted gates.

```bash
python scripts/inspect_expert_cache_checkpoint.py \
  --repository nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 \
  --revision 8fb2682f136cf94d932a498f18cb1e428832a912 \
  --metadata-output "$METADATA" \
  --device-bytes "$DEVICE_BYTES" --host-bytes "$HOST_BYTES" \
  --output "$REPORT"
```

For an explicitly supplied complete checkpoint, inspect local headers and every
routed global scale without loading model tensors:

```bash
python scripts/inspect_expert_cache_checkpoint.py \
  --model "$MODEL" \
  --device-bytes "$DEVICE_BYTES" --host-bytes "$HOST_BYTES" \
  --output "$LOCAL_REPORT"
```

Local scale validation requires finite positive global scales and exact gate/up
equality. It does not silently reconcile scales. Block-scale value validation
remains with `ExpertWeightSource`. The engine still computes the full content
fingerprint for profile identity; metadata hashes cannot replace it.

## Qualification still requiring the complete checkpoint

No full checkpoint download or complete-model execution is implied by the
metadata audit. The staged acceptance is:

1. Supply the immutable checkpoint and verify local metadata, scale values and
   the full content fingerprint. Retain actual loader coverage, distinguishing
   optional MTP from required target tensors.
2. Use the [source-built artifact verification](expert-cache-reference.md#source-matched-environment)
   and verify loaded libraries. Run the existing real-checkpoint layer oracle,
   then include the model's actual quantized shared MLP/gate in independent layer
   comparisons. Synthetic shared-composition tests are not this model oracle.
3. Admit one useful C1/C4 configuration using live device and host accounting.
   Measure ordinary loading, cache preparation, graph/KV/recurrent preparation,
   serving and shutdown separately. Do not reuse the 40-GiB host envelope or
   select an 8-GiB cache merely to reproduce a historical gain.
4. Calibrate a new profile from independent mixed-domain prompts. Retain counts,
   finite completion condition, construction provenance and admitted per-layer
   counts. Do not reuse the Qwen3-30B profile or evaluation routes.
5. Run short static and actual-promotion adaptive C1 smokes, natural EOS and one
   moderate-context smoke before three alternating C4 pairs. Keep health-16,
   existing thresholds, 32 pairs/128 MiB, prepared per-layer capacity, and
   history/protection/anchor recovery disabled. Compare exact paired output under
   controlled admission and fixed graph/storage addresses.
6. Run the existing cancellation, pending-health ownership and repeated-close
   checks, plus the qualified Qwen3-30B and ordinary non-cache regressions.

The existing serving acceptance runner starts an ordinary all-resident smoke.
**Do not point that unchanged runner at this checkpoint.** Run the cache harness
stages explicitly until an admitted ordinary offloading control is configured;
keep the ordinary Qwen3-30B regression separate. The acceptance runner's original
scope and required promotion gate are not weakened to obtain a green result.

## Practical offloading control

The pinned companion implements `OffloadConfig` with the UVA backend,
`cpu_offload_gb` and segment-matched `cpu_offload_params`. The selector
`{"experts"}` includes routed expert parameters and excludes `shared_expert`,
routers and dense attention. `make_layers` offloads modules as they are
constructed. Quantization postprocessing restores UVA ownership when it replaces
a parameter; complete-model allocation peaks and native operand-access legality
still require measurement.

One reasonable control is ordinary ModelOpt execution with explicitly admitted
selective UVA expert offloading. Derive its offload allowance from actual memory,
then check postprocessing, graph capture, loading peaks and which tensors remain
mapped. Its ordinary routed W4A4 arithmetic makes it a **deployment comparison**,
not a bitwise control or isolated cache-overhead measurement. No compatibility
failure, throughput result or performance advantage over that baseline is claimed
without executing it. The maintained implementation, not mutable online defaults,
is the baseline specification.

## Hardware boundary

No explicitly configured, authorized B300 endpoint was available during this
audit. The [native SM103 runbook](sm103-qualification.md) and ordered physical
gates remain unchanged. Ripper observations are **PCIe Gen4 x16** evidence; its
idle negotiated generation may downshift. Record the loaded link, GPU UUID,
NUMA placement, clocks and power during any timed run. Neither Gen5 nor Grace
performance follows from this preflight.
