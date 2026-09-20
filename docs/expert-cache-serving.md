# Prepared SM120 expert-cache serving

Status: **implemented experimental serving path**. The companion vLLM branch
`codex/b12x-expert-cache` at `1d1f870bd6`, based on the maintained PreparationSession branch at
`ef1aeaf080879865febd27a92c5644233d157987`, loads routed expert parameters on CPU
and submits their placement to b12x before CUDA graph capture. This path requires
physical SM120, one GPU, native ModelOpt NVFP4 weights and explicitly selected
W4A16 routed activations. It is opt-in; ordinary vLLM loading and static SM103
residency retain their existing behavior.

The [SM103 HBM/Grace backend](expert-residency.md) has a different numerical and
storage contract. Neither these PCIe serving tests nor the shared controller
qualify SM103 tcgen05 execution or Grace-backed TMA.

## Checkpoint ownership and numerical contract

`fused_moe.ExpertWeightSource` combines an immutable weight declaration,
contiguous CPU `PackedWeights`, checkpoint/layer identity and retained source
owners. Declaration performs no CUDA allocation, compilation or value inspection.
The serving loader uses the existing ModelOpt expert weight loader inside an
explicit CPU allocation context. Dense weights follow ordinary vLLM loading.
Routed experts never first exist as a complete GPU checkpoint.

The source contains the packed gate/up and down weights, logical E4M3 K16 block
scales, global scales and activation-scale metadata. The loader preserves
canonical expert IDs and declares checkpoint gate/up order as W31. Preparation
validates values and permutes scale bytes one expert at a time. It never
requantizes weights or reconciles unequal gate/up global scales. A checkpoint
with unequal gate/up global scales is rejected.

**W4A16 is an explicit numerical selection.** An NVFP4 checkpoint trained for A4
activations does not become A4-equivalent by preserving its weight bytes.
This backend uses BF16 activations, inline FP4 weight dequantization, native
SiLU, weighted BF16 route outputs, and the native W4A16 original-top-k-order FP32
sum followed by the final BF16 cast. The SM103 MXFP8/MXFP4 backend instead has its
own ordered FP32 FMA contract. Neither backend silently adopts the other's
rounding boundaries. Serving uses actual router IDs and actual routing weights.

Each cache projection computes a whole K reduction in one CTA. Split-K partial
sums can change their grouping when route packing changes, even with identical
expert bytes. Whole-K scheduling makes promotion independent of that grouping.
It preserves the FC1/activation/weighted-FC2 BF16 boundaries but can differ from
the ordinary split-K W4A16 result. The profile recipe explicitly records
`nvfp4_w4a16_whole_k_weighted_bf16_ordered_sum`; profiles from another recipe
are rejected. Prepared native ModelOpt W4A16 also exposes this schedule through
`RoutingSpec(deterministic_output=True)`, with decode query schema 9.

The qualified loader scope is TP/DP/PP=1, BF16 I/O, gated SiLU without bias,
H and local I divisible by 128, and one declared top-k across layers. EP, DBO,
speculation, LoRA, modified SwiGLU and router weighting on the input are rejected.
Distributed epoch contract tests do not qualify a distributed checkpoint loader.

## Preparation and storage

The public declaration stays inside the fused-MoE API:

```python
source = moe.ExpertWeightSource(
    plan=weight_plan,
    weights=cpu_packed_weights,
    owners=checkpoint_owners,
)
plan = moe.plan_execution(
    experts=source,
    capacity=moe.ExecutionCapacity(max_tokens=capacity, top_k=top_k),
    placement=learned_placement,
    memory_budget=admitted_layer_budget,
    updates=moe.ResidencyUpdateCapacity(max_pairs=pair_capacity),
)
# Submit plan.request(...) with a real operation primer to PreparationSession.
# After preparation, use moe.bind(plan, a=x, topk_ids=ids, topk_weights=weights).
```

Omit `updates` for static serving. The catalog variant `moe.fused_moe/cache`
resolves to `moe.expert_cache`, query/config schema 1. Its fixed backend is
`sm120_w4a16_canonical_mapped`. Preparation retains:

- one fixed VRAM expert slab and one cacheable mapped host slab per layer;
- a canonical row for every expert in the host slab;
- the device expert-to-tier/row map and native route remapping storage;
- private native W4A16 workspaces, route outputs and ordered reduction programs;
- bounded host map buffers when canonical fills are declared;
- CPU source owners, compiler carriers and all graph-referenced storage.

The native kernels and supporting CuTe kernels are resolved and primed before
capture. Positive live M up to capacity is a runtime argument. Both int32 and
int64 IDs are supported; invalid IDs contribute nothing and duplicates remain
separate routes. Binding rejects operand/storage overlap. A prepared plan serves
one serialized execution lane; concurrent lanes require separate workspaces.
Closing a plan requires the engine to retire all graphs and drain every reader.

The host representation is intentionally limited: **every canonical expert is
pinned/mapped**, and ordinary CPU loader parameters remain retained separately.
This is admitted and measured, but it is not the bounded-pinned-staging design
for arbitrary large PCIe checkpoints. Direct host miss execution requires a
mapped execution representation. Ordinary/mmap backing plus bounded staging
needs a different miss-service contract and is deferred.

## Model-wide admission

`ExpertCacheServingConfig` supplies explicit device-cache and host envelopes,
KV/graph reservations, device/host safety margins and prepared pair capacity.
`ExpertCacheModel` admits all participating layers before any cache materializes.
It accounts for resident payloads, canonical backing, retained CPU sources,
private workspaces, maps, counters, fill metadata, KV and graph reservations.
Already allocated dense weights and other device consumption are counted once
against total capacity; a free-memory snapshot is not charged twice.

Calibration bootstrap adds one expert per layer in deterministic rounds using
the exact marginal prepared-device bytes. It does not spend hypothetical shared
workspace savings. The calibration profile keeps those admitted per-layer slot
counts and ranks membership by observed decode frequency. Counts may differ
when geometry or capacity differs. This convenience builder does not implement
a learned global byte-density reallocator; the prepared backend accepts other
valid per-layer placements supplied by a model planner.

Host sources are admitted before parameter allocation. Full canonical backing
and fill buffers are admitted before preparation, against both the configured
host envelope and available physical host pages. The loader qualification is
single-rank, so it cannot independently promise the same host pool to several
ranks. Preparation failure requires reload; a partial declaration is not a
usable admitted model.

## Configuration and profile lifecycle

The companion reads the typed configuration from
`additional_config["b12x_expert_cache"]`. Absence leaves normal serving unchanged.
For example, with suitable byte envelopes for the selected machine:

```python
cache = dict(
    mode="static",                 # profile, static, or adaptive
    activation="w4a16",            # required explicit numerical choice
    profile_path="/profiles/general.json",
    workload="general",
    expert_device_bytes=8 << 30,   # includes expert workspaces/maps/counters
    host_bytes=40 << 30,           # sources + canonical backing + reserves
    kv_reserved_bytes=2 << 30,
    graph_reserved_bytes=512 << 20,
    device_safety_bytes=1 << 30,
    host_safety_bytes=1 << 30,
    max_pairs_per_layer=2,
)
```

Use vLLM's maintained V2 model runner (`VLLM_USE_V2_MODEL_RUNNER=1`), the b12x MoE
backend and the worker extension
`b12x.integration.vllm.residency_epoch.ResidencyEpochWorkerExtension`.
The engine's actual KV reservation must match the declared admission value.

- `profile` prepares a fair bootstrap and counters. At an explicit completed
  engine pause, `b12x_expert_cache_save_profile(quiescent=True)` saves observed
  membership atomically. It never changes the running placement.
- `static` loads the explicitly pinned learned profile. It declares no routing
  counter, phase setter, fill buffers or background epoch loop.
- `adaptive` starts from the same pinned learned profile and prepares counters
  and bounded canonical fills. The application explicitly drives epochs through
  `VllmResidencyEpochs`, or single-rank
  [scheduler-owned maintenance](expert-cache-maintenance.md) through
  `VllmResidencyMaintenance`. Neither driver installs an automatic cadence.

The artifact wrapper uses version 1 and SHA256 integrity. Its identity includes
checkpoint contents, numerical recipe, workload, top-k, layer names, E/H/I and
W13 order; its placements are ordinary hashed `ExpertResidencyPlan` objects.
Startup checks every field and memory admission. An unobserved positional map
cannot masquerade as a learned profile. The checkpoint fingerprint streams the
contents of all local safetensors shards and JSON configuration files through
SHA256; checkpoint files and retained source tensors must remain immutable.

Explicit calibration artifacts retain `converged=False` and
`termination="explicit_calibration_boundary"`. Selecting their path for this
experiment is deliberate acceptance of a bounded calibration. There is no
implicit auto-activation or cache search in this serving bridge. The separate
[automatic SM103 controller](expert-residency-automatic.md) keeps its schema-2
profile store, convergence rules and conservative activation policy.

## Observations, epochs and failure

The V2 runner supplies phase from scheduler request state. Pure decode rows form
an unpadded prefix; dummy work, prefill, mixed batches and verification are
excluded. An optional prepared CuTe setter publishes the valid row count once
per model invocation on the producer stream. Each observed layer counts selected
canonical IDs in its retained device counter slab. No host readback, allocation,
policy evaluation or synchronization enters graph replay. Routing query schema 2
adds `runtime_token_limit`; ordinary counters retain their existing behavior.

`bind_sm120_epoch_layer` connects each prepared cache to the shared
[model-wide epoch protocol](expert-residency-epochs.md). One engine pause covers
all layers. One counter readback precedes policy decisions, global pair/copy-byte
admission, all-rank preflight, backend fills, generation verification and
acknowledgement. Resume occurs only after success. A failed or unknown partial
layer/rank update leaves the lane stopped for coordinated reload.

For single-rank serving, worker-local maintenance combines those stages behind
one engine utility call. It retains the scheduler/device reader barrier while
avoiding the administrative output-settling delay. An explicit routing-pressure
gate can decline movement while retaining decayed history. Checks still pay
counter readback and host control cost; they are not free. The
[maintenance guide](expert-cache-maintenance.md) defines the boundary, opt-out,
failure behavior and counters-only comparison arm.

A canonical fill drains readers, verifies the device map, copies every selected
candidate into its victim's fixed VRAM slot, waits for completion, then publishes
one complete map generation. Every overwritten victim remains reconstructible
from immutable canonical backing. Failure restores every overwritten victim and
the prior map. Failed recovery poisons the preparation. This transport does not
weaken or replace the SM103 exclusive journaled exchange.

The static workload profile is never overwritten by runtime adaptation.
`recent_frequency` remains the shared default; the serving experiment explicitly
chooses `decayed_lfu`. A policy window is one completed snapshot interval, not
four token iterations inherited from an offline experiment. The benchmark's
`--epoch-tokens` counts delivered output tokens across requests; it is an
experimental trigger, not an exact scheduler-iteration interval.

## Reproducing serving comparisons

The maintained companion and b12x working branch must both be installed. The
checkpoint must fit the declared host and non-expert device reservations. The
following authored fixtures exercise stable general traffic followed by code;
they are not production traffic or a quality benchmark:

```sh
VLLM_USE_V2_MODEL_RUNNER=1 python -m benchmarks.moe.expert_cache_serving \
  --model /models/Qwen3-30B-A3B-NVFP4 --mode profile \
  --profile /profiles/general.json \
  --prompts benchmarks/moe/fixtures/expert_cache_calibration.jsonl \
  --output /receipts/calibration.jsonl --tokens 128

VLLM_USE_V2_MODEL_RUNNER=1 python -m benchmarks.moe.expert_cache_serving \
  --model /models/Qwen3-30B-A3B-NVFP4 --mode static \
  --profile /profiles/general.json \
  --prompts benchmarks/moe/fixtures/expert_cache_evaluation.jsonl \
  --output /receipts/static-c1.jsonl --tokens 128 --concurrency 1

VLLM_USE_V2_MODEL_RUNNER=1 python -m benchmarks.moe.expert_cache_serving \
  --model /models/Qwen3-30B-A3B-NVFP4 --mode adaptive \
  --profile /profiles/general.json \
  --prompts benchmarks/moe/fixtures/expert_cache_evaluation.jsonl \
  --output /receipts/adaptive-c1.jsonl --tokens 128 --concurrency 1 \
  --epoch-tokens 32 --epoch-pairs 16 --epoch-mib 64
```

Repeat matched arms with `--concurrency 4` and `8`, alternating order. Both arms
use native router weights, deterministic greedy requests, the same BF16 KV,
FlashInfer attention, context, cache capacity and full-decode graph sizes.
Inductor is independently optional through `--inductor`; it requires matching
engine binary extensions. Prefill remains eager in this experiment.

Summarize complete receipts without modifying their raw data:

```sh
python -m benchmarks.moe.summarize_expert_cache_serving \
  /receipts/static-c1.jsonl /receipts/adaptive-c1.jsonl \
  --output /receipts/summary-c1.json
```

Receipts retain every output token ID/text, TTFT, delivery intervals, per-request
decode rate, total serving wall time, admission, graph object identities, slab
pointers, epoch stages, proposals, budgets, outcomes and map generations. The
complete serving timer includes the final pending epoch; individual request
latencies include pauses while requests are active. Token delivery intervals
are not GPU iteration timings. Static has no observer, so its cold-rate analysis
requires a separately labelled instrumentation run; do not charge such a run's
throughput to the uninstrumented static arm.

The maintained AsyncLLM pause implementation waits for the scheduler's drain
acknowledgement and then sleeps 20 ms to settle output delivery. The epoch
adapter retains that supported boundary and measures the full cost. No serving
experiment bypasses the acknowledgement or removes the delay. A cheaper
scheduler-boundary API requires separate engine validation; it is not an
optimization of the expert copy primitive.

## Qualification boundaries

Status: **qualified for the recorded single-GPU serving experiment**. On an RTX
PRO 4000 Blackwell, Qwen3-30B-A3B-NVFP4 runs all 48 MoE layers with 58/128 resident
experts per layer. The authored calibration and evaluation corpora are separate.
Each evaluation arm generates 2,048 tokens across 16 requests. Static and
adaptive token IDs match exactly at C1, C4 and C8, including the longer-epoch
diagnostics. All captured graph objects and cache pointers remain unchanged.

| Concurrency | Learned static tok/s | Adaptive, 32-token epochs | Adaptive, 128-token epochs |
| ---: | ---: | ---: | ---: |
| 1 | 52.91 | 52.31 | 54.08 |
| 4 | 79.43 | 72.72 | 78.33 |
| 8 | 102.85 | 90.15 | Not measured |

These are complete serving-wall rates, including active epoch pauses. Epoch
triggers count delivered output tokens across requests, not scheduler iterations.
The 32-token configuration spends 10.06/10.17/9.54 seconds paused at C1/C4/C8.
The 128-token configuration spends 2.28/3.03 seconds paused at C1/C4. Each cell is
one run with dynamic clocks; the small C1 difference at 128 tokens is not a
robust performance win. Static remains the default.

At C1, a separately instrumented, zero-promotion control observes 24.11% cold
selections. Adaptation observes 15.77% with 32-token epochs and 20.92% with
128-token epochs. These cover completed decode observation windows, excluding
prefill and the final partial window. The instrumented control delivers 44.23
tok/s and is excluded from the static throughput column.

The stable general interval favors learned static placement: C1 throughput is
91.95 tok/s static, 66.83 with frequent adaptation, and 83.30 with longer epochs.
After the code transition, the corresponding rates are 37.14, 42.99 and 40.06.
Frequent movement improves the transition interval but does not repay the stable
interval's cost over the complete C1 run. No predictive break-even or production
quality claim follows from these authored prompts.

Device admission includes 6.882 GiB resident experts, 1.116 GiB private workspace,
1.668 GiB dense/model allocations, 2 GiB KV, 0.5 GiB graph reserve and 1 GiB safety.
Canonical host backing is 15.188 GiB; retained CPU sources consume another
15.188 GiB, plus host safety and update metadata. The measured loading device
peak is 1.668 GiB, before resident-cache preparation. All canonical host backing
is pinned in this implementation.

The physical environment uses maintained vLLM Python source with compiled
extensions from the recorded container image. Full decode graphs with compiler
mode 0 and FlashInfer attention are tested. Inductor fails on an unrelated
missing extension overload in that mixed build. A matching complete vLLM build
and normal PR CI remain independent integration gates. Engine shutdown sometimes
requires its five-second forced process cleanup; those warnings are retained.

A final source-bound C1 validation pair on the second RTX PRO 4000 records
53.07 tok/s static and 54.42 tok/s with a 128-token trigger. Another 240 promotions
and 4,096 tokens preserve exact outputs, graphs and pointers. It validates the
committed companion integration; it is retained separately from the nine-run
matrix and does not establish a general throughput benefit.

The [engineering ledger](expert-residency-ledger.md#prepared-sm120-serving-cache-september-19-2026)
records frozen source hashes, tests, measured serving results and all retained
failures. Same-graph output checks establish consistency for the tested recipe,
not checkpoint quality or equivalence to A4 activations. No adaptive default,
concurrent replacement, general TP loader or automatic engine restart is enabled.

Physical B300 qualification remains ordered: static all-HBM, all-Grace and mixed
native correctness; Grace-backed TMA; graph replay and sanitizers; fixed-address
promotion; direct Grace miss cost; then the same model-wide control experiment.
PCIe host backing costs cannot substitute for those measurements.

The measured optimization order is: reduce model-wide pause/RPC/bookkeeping
cost; avoid spending epochs on traffic already covered by the static profile;
then assess bounded host staging and explicit shared-workspace ownership. The
32-token zero-promotion control still loses throughput, so cheaper payload copies
alone cannot remove the main control cost. Concurrent replacement and fused-MoE
changes require separate evidence.

## Implementation map

| Responsibility | Files |
| --- | --- |
| CPU source ownership | [cache_source.py](../b12x/moe/fused_moe/cache_source.py) |
| Prepared SM120 cache, memory and fills | [_cache_tuning.py](../b12x/moe/fused_moe/_cache_tuning.py), [_cache_preparation.py](../b12x/moe/fused_moe/_cache_preparation.py), [_cache_updates.py](../b12x/moe/fused_moe/_cache_updates.py) |
| Public API and catalog | [api.py](../b12x/moe/fused_moe/api.py), [fused_moe exports](../b12x/moe/fused_moe/__init__.py), [catalog.py](../b12x/preparation/catalog.py) |
| Native whole-K scheduling and shared compiler lowering | [W4A16 kernel](../b12x/moe/_shared/kernels/w4a16/kernel.py), [_preparation.py](../b12x/moe/fused_moe/_preparation.py), [_tuning.py](../b12x/moe/fused_moe/_tuning.py) |
| Canonical remap and ordered sum | [W4A16 residency kernels](../b12x/moe/_shared/kernels/w4a16/residency.py) |
| Prepared decode-row counter extent | [counter kernels](../b12x/moe/_shared/kernels/routing_profile.py), [routing_profile.py](../b12x/moe/fused_moe/routing_profile.py), [_routing_profile_tuning.py](../b12x/moe/fused_moe/_routing_profile_tuning.py) |
| Model admission and worker adapter | [expert_cache.py](../b12x/integration/vllm/expert_cache.py), [residency_epoch.py](../b12x/integration/vllm/residency_epoch.py) |
| Serving measurement and analysis | [expert_cache_serving.py](../benchmarks/moe/expert_cache_serving.py), [summarize_expert_cache_serving.py](../benchmarks/moe/summarize_expert_cache_serving.py), [calibration prompts](../benchmarks/moe/fixtures/expert_cache_calibration.jsonl), [evaluation prompts](../benchmarks/moe/fixtures/expert_cache_evaluation.jsonl) |
| Admission, native graph and counter tests | [test_expert_cache_serving.py](../tests/moe/test_expert_cache_serving.py), [test_prepared_expert_cache.py](../tests/moe/test_prepared_expert_cache.py), [test_routing_profile_gpu.py](../tests/moe/test_routing_profile_gpu.py) |
| Deferred SM103 compilation | [compile_routing_profile.py](../scripts/compile_routing_profile.py), [preparation corpus](../scripts/_sm103_preparation_corpus.py) |

The companion changes exactly five files: its
`vllm/model_executor/layers/fused_moe/b12x_cache.py` CPU loader,
`vllm/model_executor/layers/quantization/modelopt.py` explicit selection hook,
`vllm/v1/worker/gpu_worker.py` preparation attachment,
`vllm/v1/worker/gpu/model_runner.py` phase handoff and
`tests/quantization/test_b12x_expert_cache.py`.

Documentation changes comprise this guide, the engineering ledger, the general
expert residency, integration and epoch guides, and the SM103 readiness report
and change summary. The commit diff provides the complete path inventory.
