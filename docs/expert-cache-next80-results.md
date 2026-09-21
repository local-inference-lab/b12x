# Qwen3-Next-80B full-checkpoint qualification

Status: **qualified experimental single-SM120 b12x serving**, within the pinned
checkpoint, numerical recipe and resource envelope below.

This report records physical SM120 qualification of
[`nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4`](https://huggingface.co/nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4/tree/8fb2682f136cf94d932a498f18cb1e428832a912)
at immutable revision `8fb2682f136cf94d932a498f18cb1e428832a912`.
The [checkpoint contract](expert-cache-large-model.md) defines the numerical,
loader and ownership boundaries. Qwen3.8 remains deferred; none of its runtime
or PLE integration is part of this qualification.

The complete model serves on one 24-GB SM120 without a full-expert GPU loading
peak. At the admitted 34.68% resident payload fraction, three C4 trials average
59.24 generated tokens/s for learned static and 69.36 for the unchanged adaptive
reference. All paired outputs match exactly. Adaptation improves throughput and
median delivery gaps but worsens tail delivery gaps. These results use the
current dual host representation and PCIe Gen4 x16; they do not establish another
topology's performance or production readiness.

## Sources and physical scope

The retained evidence directory is `b12x-next80-full-20260921` on the development
host and ripper. Source exports, commands, failed attempts, raw output IDs,
resource checkpoints and complete wheel manifests accompany the receipts.

The serving GPU is RTX PRO 4000 Blackwell, SM120, UUID
`GPU-47363510-b87a-13a5-4824-2542e97df76c`. CUDA exposes 25,151,012,864 bytes
(23.424 GiB). Its loaded link is PCIe Gen4 x16. No Gen5 or Grace performance is
inferred. A second SM120, UUID `GPU-cc109c01-9756-d0db-21ea-f1825d3f963f`, runs
separately identified arithmetic/sanitizer diagnostics. Diagnostics on that GPU
are excluded from serving performance qualification.

The environment uses driver 580.173.02, CUDA 13.3.73, Torch 2.13.0 with CUDA 13.3,
CUTLASS DSL 4.6.2 and CUDA Python bindings 13.0.3. Wheel manifests verify packaged
Python, native libraries and loaded artifacts; version suffixes are not build
identity. No Station/B300 endpoint is configured. Physical SM103 acceptance
remains unavailable and separate.

The principal serving source is b12x
`9d430f37eed2862c3f467db0c6718f99b884a011`, exported with SHA-256
`d09fe3b1d8f96539fc664d66fbef5617d28278c031510a6b1431f82eaea41bce`.
The companion source is `771c44da5338940687a4f507ad0fca6532b55bb9`;
its complete wheel SHA-256 is
`5f6d33bc60e76687eb7025b320732da802ce5850e4547a9febfa9e3699daa686`.
The artifact manifest retains all native-library hashes and verifies loaded
worker modules. Later audit-command and report changes do not relabel these
measurements as results from newer runtime sources.

## Complete checkpoint and loader coverage

The pinned repository transfer contains **50,802,129,136 bytes** in 24 files.
The canonical snapshot occupies **50,802,136,369 logical disk bytes**, including
local download metadata; allocated filesystem blocks occupy 50,802,294,784 bytes.
Repository payload bytes are distinct from HTTP/retry
traffic. No second checkpoint copy is required. All eleven safetensors shard
SHA-256 values match the published LFS identities.

The runtime/profile checkpoint fingerprint is
`f22fdcef6ae16e9a85415e35ec55069ba7ef7eab8220f48343747fc2adb4ec2e`.
It is computed once from complete local contents. Subsequent launches validate
the retained local identity receipt against the unchanged file inventory and
filesystem identities. This assumes a trusted, immutable local snapshot; it is
not a cryptographic substitute for rehashing untrusted modified files.

The strict local audit validates all 297,728 indexed tensors, shapes, dtypes,
offsets and shard bounds. Every routed global scale is finite and positive;
all 24,576 gate/up global-scale pairs match exactly. The streamed block-scale
check covers target-model native E4M3 scales, accepting valid zero blocks and
rejecting negative/nonfinite values.

Actual load callbacks account for **296,175 required target tensors**, each with
one accepted destination. They exclude **1,553 optional MTP tensors**. The
independent comparison with the strict inventory reports:

| Target class | Tensors | Native bytes |
|---|---:|---:|
| Routed experts and scales | 294,912 | 43,487,133,696 |
| Shared experts | 576 | 84,935,808 |
| Shared sigmoid gates | 48 | 196,608 |
| Routers | 48 | 100,663,296 |
| GDN/recurrent components | 360 | 1,993,619,232 |
| Full attention | 132 | 509,620,416 |
| Embeddings and head | 2 | 1,244,659,712 |
| Other norms | 97 | 397,312 |
| **Required target** | **296,175** | **47,421,226,080** |

Routed destinations are CPU parameters; ordinary target destinations are CUDA
parameters. The optional MTP iterator entries have no load callbacks. Required
routed packed weights alone are 36 GiB, so they exceed physical VRAM without KV,
workspace, competing processes or artificial allocations.

## Real-layer correctness and startup fixes

Layers 0, 24 and 47 use their actual native checkpoint bytes and actual BF16 router
projection, normalized top-10 weights, shared MLP and sigmoid gate. The tests
compare all-resident, mixed and mapped-host routed execution with an independent
dequantization/matvec reference at the declared BF16 boundaries. Different
placements agree exactly; the independent reference uses explicit tolerances.
Repeated/reversed routes, captured replay after promotion, pointer identity and
zero replay allocation are checked. The actual shared module executes once and
matches its explicit shared-MLP-plus-sigmoid decomposition and established
shared-plus-routed addition order. These are layer/composition gates, not a
W4A16-versus-W4A4 equivalence claim.

Complete loading exposed a GDN warmup shape error in the maintained companion:
warmup flattened value heads into the feature dimension, while execution uses
one row per value head and a shared per-head normalization weight. The correction
matches execution geometry and changes no arithmetic. A separate startup failure
fix explicitly closes worker resources when warmup raises before EngineCore
construction completes. Complete source-built wheels retain both fixes.

A separate calibration-boundary defect was caught before the timing matrix:
startup observations could remain in cumulative counters after engine warmup.
The harness now pauses the prepared engine, records and clears those observations,
and explicitly starts calibration before admitting its first prompt. Saving a
profile requires that start boundary. Static/adaptive serving cannot invoke the
reset, and no policy or numerical arithmetic changes. The initial profile and
its preliminary smoke receipts remain retained; final timing uses recalibration
from the same disjoint mixed-domain prompts. The explicit start discarded 140
startup selections per layer. Correcting this boundary changed 286 resident
memberships across 48 layers without changing their resident counts. The old
profile hash is `4100fbbe7aee75099faa793645fe3c4ab8b6111a1ed522a4b582217a28dc7fa3`;
none of its diagnostic timings are relabeled as corrected-profile performance.

## Memory admission and calibration

The real preparation path successfully allocates **43,486,740,480 mapped canonical
bytes**, alongside **43,487,133,696 retained CPU source bytes**. The explicit host
envelope is 128 GiB, including an 8 GiB safety reserve. No OS limit changes are
required. Host availability, process RSS and the locked-memory limit are retained;
`VmPin` and `VmLck` are not treated as CUDA pinning totals.

Calibration uses eight independently authored prompts, two each for general,
code, math and multilingual traffic, with 64 generated tokens per prompt. Its
512 generated tokens produce 4,980 completed decode selections per layer
(498 decode-eligible rows). Fourteen output-generating rows occur outside the
declared decode-only phase. These
are canonical execution counters, not an assumption of one execution per
client-delivered token. The finite run does not establish convergence.

The existing fair capacity allocator projects those immutable calibration counts
to the maximum envelope admitted by observed graph/serving storage. Context is
2,048, KV reservation 2 GiB, graph reservation 512 MiB, prepared token capacity 64,
and device safety 1 GiB. Safety is additional to observed runtime allocations.

| Planned quantity | Value |
|---|---:|
| Expert envelope | 16,325,990,988 bytes (15.205 GiB) |
| Resident experts per layer | 178 in 27 layers; 177 in 21 layers |
| Resident expert payload | 15,081,283,584 bytes |
| Resident payload fraction | 34.6802% |
| Private workspace | 1,242,738,732 bytes |
| Metadata and observer/health storage | 989,968 bytes |
| Declared host use including safety | 95,564,302,592 bytes |

Layers 0–3 and 10–32 have 178 residents each; layers 4–9 and 33–47 have
177 each. The profile records every canonical resident ID. One expert's canonical
payload is 1,769,480 bytes; transaction receipts additionally include map-copy
bytes. The 128-MiB movement envelope admits the unchanged 32-pair limit.

The learned profile hash is
`16b14690fa88d1532a39042420afedf81c082e58cc0fc709602235694fb232b2`.
Its construction binds the calibration profile, canonical counts and measured
reservation receipt. Static and adaptive arms use the identical immutable map.
Static allocates no adaptive observer. The envelope fraction is not the resident
payload fraction.

The calibration lifecycle records a 3,970,680,832-byte device peak after loading,
before cache preparation. Routed experts never first form a complete GPU
checkpoint. After explicit release, mapped bytes, retained CPU expert sources,
graph owners and pending health storage are all zero. Residual Torch/runtime
allocations are recorded separately from cache ownership.

Representative primary static memory checkpoints, in GiB, are:

| Checkpoint | Torch allocated | CUDA free | CPU routed sources | Owned mapped backing | Graph owners |
|---|---:|---:|---:|---:|---:|
| Before loading | <0.001 | 23.094 | 0 | 0 | 0 |
| After ordinary loading and CPU sources | 3.698 | 19.366 | 40.501 | 0 | 0 |
| After canonical/cache preparation | 18.901 | 4.094 | 40.501 | 40.500 | 0 |
| After recurrent/KV/graph preparation | 21.828 | 1.028 | 40.501 | 40.500 | 2 |
| After serving | 21.828 | 1.014 | 40.501 | 40.500 | 2 |
| After explicit worker release | 0.511 | 22.430 | 0 | 0 | 0 |

The intermediate resource checkpoints also retain recurrent and KV allocation
growth during preparation. Across primary trials, peak Torch allocation is
23,462,148,096 bytes for static and 23,462,745,088 for adaptive. Minimum observed
CUDA free memory is respectively 1,089,011,712 and 1,084,817,408 bytes, preserving
the explicit 1,073,741,824-byte safety margin. CUDA free/total includes runtime
and native allocations outside Torch; the Torch peak alone is not total device
consumption. Worker peak RSS is at most 90,126,217,216 bytes for static and
90,254,548,992 for adaptive, about 83.94 and 84.06 GiB. Client-process resources
are retained separately in lifecycle receipts.

Adaptive adds 393,216 mapped map bytes to canonical backing and a 2,304-byte
pinned health result. Static has neither that health result nor adaptive counters.
Host memory is on the single NUMA node with CPUs 0–63; the recorded locked-memory
limit is unchanged. Successful real preparation proves this allocation on this
host, without assuming that the process memlock limit alone describes CUDA host
registration behavior.

With the corrected profile, the complete generation-zero diagnostic observes
977,280 selections in each interval. General has 321,313 cold selections
(32.88%); code has 349,236 (35.74%). Overall learned route coverage is 65.69%,
versus 34.68% of expert payload resident. Its exact output hash matches the first
uninstrumented static trial. Per-layer counts and all four request-group windows
are retained in `route-coverage-clean.json`. This finite mixed calibration does
not make the general evaluation interval healthy under the 15% pressure gate.

## Repeated static/adaptive serving

Each fresh engine serves 16 held-out requests in four controlled groups of four:
eight general requests followed by eight code requests, 256 generated tokens
each. Calibration and evaluation prompts are disjoint. Engine order is
static/adaptive, adaptive/static, static/adaptive. All six runs use the same
profile, graph sizes, 2-GiB KV reservation and 2,048-token context. No unrelated
GPU work overlaps these trials. The adaptive settings remain health-16, cold
threshold 0.15, maximum interval 1,024, decayed LFU, two prepared pairs per layer,
and 32 pairs / 128 MiB per model-wide transaction. History, specialist protection
and anchor recovery are disabled.

Values below are mean ± sample standard deviation over **three engine trials**,
not confidence intervals over individual tokens.

| Arm | Overall tokens/s | General tokens/s | Code tokens/s |
|---|---:|---:|---:|
| Learned static | 59.240 ± 0.028 | 61.415 ± 0.026 | 57.286 ± 0.031 |
| Adaptive | 69.363 ± 0.021 | 78.702 ± 0.014 | 62.100 ± 0.038 |

Raw overall static samples are 59.27191, 59.22973 and 59.21801; their paired
adaptive samples are 69.33851, 69.37401 and 69.37534. Paired gains are 16.984%,
17.127% and 17.152%, averaging **17.088%**. Every run generates 4,096 tokens;
all six output-ID hashes are
`44804de497f4fe66580e172127765ca77b35cb35c28df1dad362613636222d58`.
Within-run graph/cache pointers remain fixed and per-layer generations match
the completed transaction log.

The latency table averages each engine's reported quantile or request mean.
Raw per-request delivery events remain available; these are client delivery
gaps, not instrumented GPU token latency.

| Metric | Static | Adaptive |
|---|---:|---:|
| Mean per-request decode rate, tokens/s | 15.31 | 18.80 |
| TTFT p50 / p95, ms | 518.9 / 588.3 | 533.8 / 661.5 |
| Delivery gap p50 / p95 / p99, ms | 65.39 / 83.49 / 89.73 | 48.11 / 118.34 / 140.45 |
| Startup through graph preparation, s | 136.44 | 138.25 |
| Worker loading within startup, s | 38.40 | 38.41 |
| Explicit shutdown, s | 15.87 | 16.06 |

Startup and shutdown are outside serving throughput. Serving includes every
maintenance operation and the final pending-control tail: 3.17, 5.17 and 5.17 ms
for adaptive. General improves by 28.15% and code by 8.40%. The gain in the
general interval matters: this finite mixed profile is not an already healthy
static placement, so the result is broader than recovery from the code transition
alone. The separate C1 smoke below is a retained short-run negative result.

Every adaptive trial performs 138 health probes and 97 full maintenance
operations, all triggered by pressure; no maximum-interval trigger confounds
the comparison. Each makes 3,104 promotions, copies 5,511,889,152 API bytes
(5.13 GiB), and observes 1,947,840 selections in completed policy windows, with
21.23% cold selections. These windows omit the final 6,720 selections covered by
the separate complete static diagnostic; the percentages are not identical
observation intervals. Static has no counters, health probes or maintenance.

Median proposals/selected pairs are 96/32, from 48 proposing layers. All 97
epochs saturate the pair cap, with median 64 skipped pairs and zero byte-cap
skips. The total is 6,203 skipped proposals per run. This is admissible movement
opportunity, not proof that increasing the cap would improve throughput.

Each run reloads 600 previously evicted experts, including initial residents.
The older repeat-promotion metric is zero because it counts only experts
promoted twice during the run; it does not include those initial-resident
reloads. There are 473 completed promoted lifetimes, nine with zero hits, and
2,631 right-censored lifetimes. Reported useful hits after promotion total
437,181. No demand or lifetime outcome is inferred beyond its observed window.

Mean accumulated scheduler-blocked time is 11.68 s per adaptive run, already
included in serving wall time. A mean transaction contains a 53.63-ms scheduler
drain and a 66.67-ms worker RPC; the drain includes already-submitted useful
work. Nested worker stages include snapshot 5.12 ms, pressure evaluation 7.41 ms,
policy 25.01 ms, preflight 3.65 ms, apply 12.00 ms and acknowledgment 11.56 ms.
These nested stages are not added to the RPC, blocked time or serving wall time.

## Ordinary selective-UVA deployment control

The maintained ordinary ModelOpt path uses `FLASHINFER_CUTLASS`, with UVA
offloading selected by the exact `experts` parameter-name segment and an explicit
26-GiB allowance. Physical CUDA pointer attributes after quantization identify
242 mapped parameters totaling **27,984,642,048 bytes**. All belong to routed
experts. Shared experts, sigmoid gates, routers, GDN/attention, embeddings and
the head remain ordinary device parameters. Actual target loading consumes
every required tensor and excludes optional MTP.

This offloader operates at parameter-tensor granularity: layers 0–29 have all
their routed parameters mapped; layer 30 has its two packed routed-weight
parameters mapped. It is not b12x's learned per-expert placement. Total ordinary
post-quantization routed storage is 43,486,937,088 bytes, leaving about 14.44 GiB
locally resident. The 26-GiB setting preserves the same 1-GiB safety requirement;
reducing offload by another full GiB would consume that reserve. The retained
27-GiB startup smoke is not a separately tuned headline arm.

Three fresh engines use the same checkpoint, rendered prompts, C4 admission,
context, output length, 2-GiB KV reservation and captured graph sizes as the
cache arms. Both paths disable Inductor and use the declared graph configuration;
this is not a claim about the fastest possible vLLM configuration. FlashInfer's
normal kernel configuration cache is prepared outside serving and retained.

| Ordinary control metric | Result |
|---|---:|
| Overall tokens/s, mean ± sample SD | 41.321 ± 0.045 |
| General tokens/s, mean ± sample SD | 41.364 ± 0.143 |
| Code tokens/s, mean ± sample SD | 41.317 ± 0.106 |
| Mean per-request decode rate | 10.58 tokens/s |
| TTFT p50 / p95 | 607.1 / 644.3 ms |
| Delivery gap p50 / p95 / p99 | 95.23 / 101.71 / 103.68 ms |
| Startup, raw trials | 123.40 / 105.98 / 105.93 s |
| Worker loading within startup, mean | 76.21 s |
| Explicit shutdown, mean | 11.01 s |
| Peak Torch device allocation, maximum | 22,726,150,656 bytes |
| Minimum observed CUDA free memory | 1,397,293,056 bytes |
| Peak worker RSS, maximum | 34,647,212,032 bytes (32.27 GiB) |

Raw overall rates are 41.30660, 41.37198 and 41.28514. The first startup includes
additional ordinary-kernel autotuning outside the serving interval. Each run
completes 4,096 tokens and releases mapped model ownership cleanly; the residual
Torch pinned allocator cache is 192 bytes. The ordinary control has no retained
b12x CPU source representation or duplicate canonical backing.

The observed rate ratios are 1.434 for learned-static b12x over ordinary and
1.679 for adaptive b12x over ordinary. These are **deployment observations**.
Ordinary routed execution uses W4A4 while b12x uses whole-K W4A16; placement granularity and host
representations also differ. The measurements do not isolate kernel efficiency
or establish equivalent answer quality. Ordinary trial output IDs also differ
between fresh engines. Two additional eight-token C4 diagnostics reproduce the
difference with **identical nine-step execution signatures**. In their first
64-token prefill, layer 0's input, GDN output, post-attention normalization,
router projection, shared expert and shared sigmoid gate match exactly. Its
ordinary MoE output differs by at most 0.0009765625. Later hidden/logit differences
produce different greedy IDs. All captured tensors remain finite.

This localizes the first observed discrepancy to ordinary MoE execution under
matched inputs and shapes; it does not identify the underlying kernel/reduction
cause. The throughput runs complete successfully, but **ordinary numerical
repeatability remains unresolved**. Their raw outputs, traces and module
comparisons are retained in `native-repeat-*`. The 41.32 tokens/s observation is
not a fully qualified numerical reference or a closed practical-competitiveness
claim. No arithmetic, tolerance or b12x equality gate is changed to conceal it.

## Measured execution cost

A separate C4 static trace records eight decode iterations after a 16-iteration
delay. The public engine profiler runs with selected-token log probabilities
enabled. All 256 diagnostic selected-token log probabilities are finite and the
output prefixes match the uninstrumented static trial. Graph/cache identities
and clean owner release pass. Profiling timings are not serving throughput.

Routed whole-K kernels account for **80.81% of summed GPU kernel duration**
(400.22 ms of 495.29 ms). The union of GPU kernel intervals is 487.49 ms within
a 500.30-ms span. Those duration shares are not percentages of end-to-end wall
time. The GDN recurrence kernel itself accounts for 0.61%; this does not include
all GDN projections or other dense computation. The trace retains every kernel
name rather than assigning generic GEMMs to model components without evidence.

A matched real-checkpoint layer-24 diagnostic separates residency cost while
holding four BF16 input rows, actual router top-10 IDs/weights and the whole-K
recipe fixed. Its inputs are synthetic; its 26/40 cold routes under the learned
map are not the full-model evaluation's cold fraction. Ten alternating rounds
retain warm and L2-scrubbed samples, with scrub time excluded from target GPU
events. L2 is 48 MiB and the scrub buffer is 96 MiB.

| Layer operation | Cold selections / 40 | Warm median | L2-scrubbed median |
|---|---:|---:|---:|
| All-resident routed experts | 0 | 156.84 µs | 176.30 µs |
| Learned mixed placement | 26 | 2,103.39 µs | 2,137.86 µs |
| One resident slot, all selected experts cold | 40 | 3,211.24 µs | 3,215.40 µs |
| Actual shared MLP and sigmoid gate | Separate shared computation | 22.58 µs | 27.76 µs |

Independent-reference arithmetic, exact placement parity and allocation-free
graph replay precede these timings. The all-resident control is one layer, not
a feasible all-resident full-model deployment. Cold routed execution is about
18.2 times the all-resident layer cost in the scrubbed comparison. Together with
the full-model trace, this identifies mapped cold-expert service as the dominant
measured limitation of this Gen4 host. It does not establish the transport
ceiling or the same ratio on Gen5 or Grace.

The immediate follow-on qualification task is to minimize the ordinary ModelOpt
MoE repeatability discrepancy at fixed real weights, inputs and routes, and
determine whether it is permitted reduction variation or a correctness defect.
That closes the remaining practical-baseline uncertainty before optimizing it.
For b12x performance, the measured area to address is cold-expert service, with
the controller and numerical recipe held fixed. No transport optimization is
implemented here. Measurements on an authorized Gen5 or Grace target are needed
before treating Gen4 costs as the intended platform's limit. If physical B300
access becomes available, the existing ordered native correctness and memory-tier
gates retain priority.

## Preliminary full-model gates

The learned-static and adaptive C1 smokes each complete 64 generated tokens
with identical output IDs, stable graph/cache addresses and explicit owner
release. Adaptive performs three bounded transactions, 96 promotions and
170,427,136 API copy bytes. Its 26.65 tokens/s is below the static smoke's
27.45 tokens/s; a short request does not necessarily repay movement. These
single-run diagnostics used companion `a45d74d6228b0ba6083bc5f781c035189bd689ee`
and overlapped a separately identified GPU diagnostic. They are not the repeated
performance comparison.

Fixed-length generation retains EOS IDs and deliberately continues afterward;
that continuation is not a quality test. A separate natural-EOS request answers
`12` to `7 + 5` and stops after three tokens, including EOS. A 1,386-token input
correctly retrieves `K043, 60` from 65 supplied records, stops after nine tokens,
and has an 11.31-second TTFT. This qualifies one moderate context extent, not the
model's maximum context or general answer quality.

A separate generation-zero counters-only C4 diagnostic covers complete
request-group boundaries. It subtracts the pre-serving baseline, performs no
policy updates or moves, and records 1,954,560 decode selections. The preliminary
profile covers 69.03% of general selections and 63.34% of code selections (66.18% overall). These historical diagnostics precede the calibration
boundary correction and are retained separately from the corrected-profile
routing diagnostic. Neither diagnostic is static timing or predicted throughput.

Ordinary selective-UVA loading initially exposed a shutdown defect: live weight
owners retired, but about 27 GiB remained cached by Torch's pinned allocator until
interpreter finalization exceeded the process manager's timeout. That failed run
is retained. The companion now releases unused pinned allocator blocks during
explicit worker shutdown when UVA offload is enabled, after graph/model readers
retire. No timeout or hot-path synchronization changes were made. A complete
rebuilt wheel passes the same full-model smoke with clean shutdown; the final
host allocator cache is 88 bytes of small runtime allocations rather than the
model-sized pool.

## Lifecycle and regression acceptance

Full-model normal static and adaptive shutdown release all explicitly owned
mapped canonical bytes, retained CPU expert sources, graph owners and health
storage. The pending-health test reaches shutdown with a submitted result and
2,304 pinned result bytes; shutdown waits for ownership to retire and reports
zero pending work and zero result bytes afterward.

Two full-model engines are reconstructed sequentially in one client process.
Each completes 96 promotions, observes caller cancellation during engine-owned
maintenance, preserves output IDs and releases all cache owners without a forced
kill or ignored destructor exception. Both workers retain the same 548,416,000
Torch-allocated bytes and 616,562,688 reserved bytes immediately after explicit
release; these runtime pools are distinct from the released cache. The workers
then exit. Client descriptors remain 38, threads remain 73, and the same Python
resource-tracker child remains. Client RSS grows from 1,519,744 to 1,634,112 KiB
between these two cycles. This proves the two tested reconstructions and cache
owner release, not a long-run parent-process RSS plateau.

The final complete companion wheel passes 32 focused coverage, shared-expert,
maintenance, warmup and shutdown tests. Physical GPU acceptance passes all 42
prepared-cache, observation and residency tests without skips. The real-checkpoint
suite passes four tests, including the three representative layers and physical
mapped-pointer inspection. Bounded real-checkpoint memcheck and synccheck each
pass with zero reported errors. Those sanitizer cases use a declared 16-expert
subset of real checkpoint bytes with unchanged hidden/intermediate geometry and
top-10 routes; they do not claim sanitizer coverage of all 512 experts. Earlier
interrupted sanitizer attempts remain retained as incomplete runs.

The Qwen3-30 regression passes ordinary non-cache serving plus learned-static and
adaptive cache arms. Its cache pair produces 1,024 tokens per arm with exact IDs,
384 adaptive promotions, fixed addresses and clean release. These GPU/regression
gates use b12x `6c6aeea1f0c9447e2bce1fca44095b6485d1319b` and the final companion
wheel. The later calibration-boundary change has focused host tests, independent
[host CI](https://github.com/local-inference-lab/b12x/actions/runs/35656922928),
physical recalibration and the subsequent repeated serving gates. Sanitizer
receipts retain their earlier source pair; later load-time and lifecycle fixes
do not modify the sanitized device execution.

Independent host acceptance on the measurement source reports 1,126 passed and
67 skipped tests, with zero failures or errors. These host skips are not GPU
acceptance. No required physical SM120 test is counted as passed through a skip.

The evidence retains failed startup, forced ordinary-UVA teardown, interrupted
sanitizer and preliminary calibration attempts. The fixes above explain their
disposition. An early pointer-inspection test also used an unindexed CUDA device
in its expectation; the corrected test checks the actual indexed device and
passes. Failed and preliminary runs are excluded from the repeated performance
samples.


## Reproduce the gates

Use the source-matched environment from the [reference guide](expert-cache-reference.md).
Supply explicit `MODEL`, `RESULTS`, `PROFILE` and built-artifact locations. The
model directory must contain the pinned complete checkpoint; these commands do
not download it. Keep the checkpoint immutable after generating its content
receipt. The initial audit streams block scales and hashes complete contents
once:

```bash
python scripts/inspect_expert_cache_checkpoint.py \
  --model "$MODEL" --check-block-scales \
  --identity-output "$RESULTS/checkpoint-identity.json" \
  --device-bytes 25151012864 --host-bytes 137438953472 \
  --output "$RESULTS/local-audit.json"
export B12X_CHECKPOINT_IDENTITY="$RESULTS/checkpoint-identity.json"
export B12X_TEST_NEXT80_CHECKPOINT="$MODEL"
python -m pytest tests/moe/test_next80_checkpoint.py -q
```

Render `benchmarks/moe/fixtures/next80_calibration_mixed.jsonl` and the held-out
`expert_health_chat_code.jsonl` with the pinned tokenizer's user chat template,
using `add_generation_prompt=True`. Retain the original text, rendered text,
prompt token count and rendered-file hash. The evidence bundle includes
`fixtures.py`, the rendered fixtures and hashes. It also includes natural-EOS,
moderate-context and cancellation receipts.

Calibration first used a conservative 15-GiB expert envelope and 64 output tokens
for each of eight prompts. Request load-time coverage only on this diagnostic;
normal timing does not need the large callback manifest:

```bash
python -m benchmarks.moe.expert_cache_serving \
  --model "$MODEL" --mode profile --profile "$RESULTS/calibrated.json" \
  --prompts "$RESULTS/calibration.jsonl" --tokens 64 --concurrency 4 \
  --workload next80-mixed-calibration-v1 --cache-gib 15 \
  --host-gib 128 --host-safety-gib 8 --context 2048 --kv-gib 2 \
  --capacity 64 --admission together \
  --loader-coverage "$RESULTS/loader-coverage.json" \
  --resources "$RESULTS/calibration-resources.jsonl" \
  --output "$RESULTS/calibration.jsonl.receipt"
python scripts/inspect_expert_cache_checkpoint.py \
  --model "$MODEL" --loader-coverage "$RESULTS/loader-coverage.json" \
  --device-bytes 25151012864 --host-bytes 137438953472 \
  --output "$RESULTS/coverage-audit.json"
python -m benchmarks.moe.expert_cache_capacity \
  --calibrated-profile "$RESULTS/calibrated.json" \
  --calibration-receipt "$RESULTS/calibration.jsonl.receipt" \
  --reference-receipt "$RESULTS/calibration.jsonl.receipt" \
  --maximum --output "$RESULTS/capacity"
```

The maximum option reuses immutable calibration counts and preserves observed
non-cache storage, KV, graph and safety reservations. It is a planning estimate;
verify the resulting geometry and live C4 headroom before timing. The recorded
run's equivalent planner calculation produced the envelope and profile listed
above. A newly constructed artifact has its own provenance/hash; never edit a
profile's declared geometry or hash to match these receipts.

For the recorded capacity, the principal cache invocation is:

```bash
python -m benchmarks.moe.expert_cache_serving \
  --model "$MODEL" --mode static --profile "$PROFILE" \
  --prompts "$RESULTS/evaluation.jsonl" --tokens 256 --concurrency 4 \
  --workload next80-mixed-calibration-v1 --cache-bytes 16325990988 \
  --host-gib 128 --host-safety-gib 8 --context 2048 --kv-gib 2 \
  --capacity 64 --admission together \
  --resources "$RESULTS/static-resources.jsonl" --output "$RESULTS/static.jsonl"
```

For adaptive, select `--mode adaptive` and add `--control health --epoch-tokens 16
--cold-threshold .15 --health-max-tokens 1024 --epoch-pairs 32 --epoch-mib 128`.
Use new output paths and a fresh engine for every arm. History, specialist
protection and anchor recovery remain disabled. Alternate static/adaptive,
adaptive/static, static/adaptive. The existing summarizer checks complete
receipts; additionally compare exact output hashes, per-layer generations,
within-run addresses, live safety headroom and released owners.

The observation-only diagnostic uses `--mode adaptive --control observe
--routing-diagnostics` with the same profile and requests. It must remain at
generation zero. Its routing counts are excluded from static timing. The ordinary
control uses `--mode native --expert-offload-gib 26` and the same serving
reservations. This explicitly selects UVA and the `experts` parameter segment.
Use `B12X_PARAMETER_STORAGE=1` for the load-time pointer inventory that proves
which post-quantization parameters are physically mapped.

`--shutdown-case health-pending` exercises a submitted result slot at close.
`--shutdown-case maintenance-cancelled --repeat-lifecycle 2` exercises cancelled
control callers and sequential reconstruction. Keep those diagnostics separate
from timing. The existing `qualify_expert_cache.py --tier serving` still begins
with an ordinary all-resident smoke: use it for the Qwen3-30 regression, not as
an unmodified launcher for this larger-than-VRAM model.
