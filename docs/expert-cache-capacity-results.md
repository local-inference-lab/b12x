# SM120 resident-capacity results, September 2026

The unchanged adaptive reference improves throughput at all three constrained
capacities tested. More resident memory improves throughput more than adaptation
at the preceding capacity. The complete checkpoint also fits all-resident with
the fixed KV, graph and safety reservations, and that same-recipe control is much
faster. These are deliberately constrained expert budgets, not evidence that
this checkpoint requires offloading on the entire card.

The experiment changes planning, profile construction, diagnostics and acceptance
tooling only. Cache policy, BF16 router arithmetic, whole-K W4A16 execution,
transport, lifecycle and the companion engine are unchanged. See the
[capacity guide](expert-cache-capacity.md) for commands and the
[reference guide](expert-cache-reference.md) for the supported environment.

## Sources, calibration and controls

Initial live heads were b12x `9cbb75cc3f1f65b794ac0c4a44a3da93c72f6460`
and vLLM `c3efbdf25b9fdcd9adf5ae3888ced9c7e0a906ca`. The inspected default
branches were b12x master `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`
and vLLM main `47ccf6c57d92f03630ebcbad3809450545825488`.
Both working trees were initially clean. Neither default branch nor the companion
was changed.

All new timed serving and GPU qualification use the frozen b12x export
`d28ee2561bbc553385014f6004c906e356d34cdb` with companion `c3efbdf`.
Later tooling commits add observed engine-storage accounting (`2d2c1e7`) and
static-route replay (`3c534d0`, with the actual receipt-schema correction at
`06f0efc3e86254b68f236b5686f5a3d37089892b`); they do not change measured runtime code.
The complete source-built companion wheel is reused without rebuilding. Its
SHA-256 is `9b4b73f9c57cad647b42fa535790dfc566141d21ee0ba1ed6eb46732fd5ebc99`.
Acceptance verifies packaged files, matching companion source, installed files
and loaded native libraries. A version suffix alone is not the build identity.

The environment is Torch 2.13.0, CUDA 13.3, CUTLASS DSL 4.6.2, Triton 3.7.1,
driver 580.173.02 and the explicitly recorded CUDA bindings 13.0.3 override.
The selected RTX PRO 4000 Blackwell UUID is
`GPU-47363510-b87a-13a5-4824-2542e97df76c`. Ripper's serving link is **PCIe Gen4
x16**, with NUMA node 1. Runs are serialized with the existing GPU lock and
compute-process exclusion. Raw telemetry records both GPUs, clocks, power and
negotiated link state; the second GPU is not used for concurrent experiments.
All 1 Hz samples inside timed traffic report Gen4 x16 and a 13,365 MHz memory
clock. SM clocks vary from 2,047 to 2,482 MHz and power from 31.17 to 148.32 W
across phases and capacities; clocks are not artificially locked. The second
GPU remains at 2 MiB and zero utilization throughout the recorded matrix.

The supplied Qwen3-30B-A3B-NVFP4 checkpoint fingerprint is
`bdde460712efe6ea50175eac186c27aa983348baf9faf2ab28a917ed93182609`.
The recipe remains `nvfp4_w4a16_whole_k_weighted_bf16_ordered_sum`, using actual
router IDs and weights. No checkpoint is downloaded or redistributed.

All capacity profiles use one immutable general-calibration count set: eight
separate prompts, 128 generated tokens each, recorded by the preceding
source-built calibration run. Its original profile ID is
`cf39df99c86608b4d780b482d8cbc0488ab7a3b82a1e1b1e7e0e809a8237d17a`;
the canonical-count digest is
`22332f815514d32bc7d2f64dbedfa30e4b3c6eab80b6db8ef01900301b4edc11`.
The planner verifies the original artifact, receipt and checkpoint/recipe/layer
identity, then calls the existing profile constructor with each admitted count.
Rankings and expert-ID tie breaking are identical across capacities. Every new
artifact records its construction inputs and has a separate validated hash.
Runtime adaptation never modifies it. Evaluation routes do not choose initial
residents. The finite calibration boundary does not claim convergence.

The C4 evaluation is `expert_health_chat_code.jsonl`: 16 held-out requests,
256 output tokens each, controlled admission in groups of four. Context 2,048,
prepared token capacity 64, graph sizes 1 and 4, 2 GiB KV, 0.5 GiB graph reserve,
1 GiB device safety and 1 GiB host safety remain fixed. Static has no observer.
Adaptive uses health probes every 16 delivered tokens, cold threshold 0.15,
maximum interval 1,024, experimental decayed LFU, 32 pairs / 128 MiB globally
and two prepared pairs per layer. Anchor recovery, history and specialist
protection are disabled. These settings are controls, not production defaults.

Each constrained capacity has three fresh-engine pairs in alternating order:
static/adaptive, adaptive/static, static/adaptive. The all-resident cache has
three fresh static trials. An engine trial is the replication unit; tokens are
not independent performance samples.

## Memory and admission

The envelope contains expert slots, private MoE workspaces and metadata. It is
not a payload percentage. The existing fair incremental allocator determines
counts and reserves adaptive storage before constructing both arms' identical
initial placement. Static still allocates no adaptive observer.

| Expert envelope | Actual resident experts, each of 48 layers | Resident payload | Payload resident | Static device use at serving end | Adaptive device use | Static Torch peak | Static device free |
|---|---|---:|---:|---:|---:|---:|---:|
| 5 GiB | 33 in 35 layers; 32 in 13 | 3.883 GiB | 25.6% | 10.016 GiB | 10.020 GiB | 9.550 GiB | 13.408 GiB |
| 8 GiB | 58 in every layer | 6.882 GiB | 45.3% | 13.023 GiB | 13.027 GiB | 12.548 GiB | 10.400 GiB |
| 13 GiB | 101 in 7 layers; 100 in 41 | 11.883 GiB | 78.2% | 18.023 GiB | 18.027 GiB | 17.549 GiB | 5.400 GiB |
| 17 GiB | 128 in every layer | 15.188 GiB | 100% | 21.324 GiB | Unsupported | 20.854 GiB | 2.100 GiB |

Values are means across trials. Device use is CUDA total minus free, including
storage outside Torch's live allocator; Torch peak is a separate measurement.
CUDA reports 23.424 GiB usable on this nominal 24 GB card. Adaptive Torch peaks
are approximately 0.00014 GiB above their static counterparts. These readings
are not interchangeable with NVML framebuffer accounting or a continuous
high-water mark for every native allocation.

Each point retains 15.188 GiB of mapped/pinned canonical backing **plus**
15.188 GiB of CPU checkpoint sources, including the all-resident cache. Private
MoE workspaces are approximately 1.116 GiB. Dense loading uses 1.670 GiB;
update journals, maps, counter/health storage and reserves are explicitly charged
in the machine-readable plan. No hypothetical shared-workspace savings or bounded
pageable staging are used. Host admission is a fixed 40 GiB envelope.

The final preflight also charges the reference's observed 0.492 GiB of additional
engine/native/pool storage beyond pre-preparation reservations. It adds only that
unaccounted increment and preserves the safety reserve. Total reserved device
bytes are approximately 11.030, 14.029, 19.029 and 22.334 GiB, respectively.
All four counts remain feasible; the all-resident plan leaves another 1.089 GiB
beyond the already reserved safety margin. The corresponding earlier timed
profiles are retained unchanged. Live loader admission remains authoritative.

A 1 GiB expert envelope is rejected before materialization because it cannot
fit one slot per layer plus the actual private workspaces. This is capacity
evidence, not a passed GPU test. A focused host test also verifies rejection of
all-resident adaptive configuration: the adapter requires at least one mutable
nonresident layer. No physical all-resident adaptive run is claimed, and no
observer diagnostic bypasses that contract.

## Serving performance

| Envelope | Static tok/s, mean ± sample SD | Adaptive tok/s, mean ± sample SD | Absolute gain | Adaptive / static − 1 | Adaptive / all-resident |
|---|---:|---:|---:|---:|---:|
| 5 GiB | 50.205 ± 0.021 | 64.267 ± 0.029 | 14.062 | +28.01% | 20.0% |
| 8 GiB | 85.456 ± 0.068 | 121.062 ± 0.073 | 35.606 | +41.67% | 37.7% |
| 13 GiB | 171.279 ± 0.133 | 197.321 ± 0.131 | 26.042 | +15.20% | 61.5% |
| 17 GiB, all resident | 320.968 ± 0.078 | Unsupported | — | — | — |

| Envelope | General static | General adaptive | Code static | Code adaptive |
|---|---:|---:|---:|---:|
| 5 GiB | 74.518 | 78.216 | 37.886 | 54.720 |
| 8 GiB | 172.907 | 172.096 | 56.822 | 93.607 |
| 13 GiB | 301.046 | 298.389 | 119.999 | 147.951 |
| 17 GiB | 327.393 | — | 316.956 | — |

Interval values are mean output tok/s. The fixture's general phase is not
necessarily a healthy-cache phase at low capacity: adaptation helps that phase
at 5 GiB. At 8 and 13 GiB its general-phase cost is about 0.47% and 0.88%.
Code-phase improvements account for the overall gains. No tested constrained
point loses overall; these four points do not locate a universal capacity cutoff.

Raw overall samples, in pair order:

| Envelope | Static | Adaptive |
|---|---|---|
| 5 GiB | 50.180, 50.220, 50.214 | 64.300, 64.256, 64.245 |
| 8 GiB | 85.415, 85.534, 85.419 | 121.000, 121.142, 121.044 |
| 13 GiB | 171.343, 171.127, 171.367 | 197.388, 197.405, 197.170 |
| 17 GiB | 320.903, 320.944, 321.055 | — |

The 17 GiB arm is legitimate full-model, all-resident execution through the
cache representation with the same arithmetic and reservations. It retains
backing and both-tier workspace, so it does not isolate cache-wrapper overhead
against an ordinary native whole-K engine path. That matched ordinary serving
path is not currently configured. The ordinary ModelOpt W4A4 smoke is a different
recipe and only a regression gate here, not a throughput or quality comparison.

Static at 8 GiB exceeds adaptive at 5 GiB; static at 13 GiB exceeds adaptive
at 8 GiB. Adaptation is useful when these expert budgets are imposed, but it does
not substitute for resident capacity. Freed VRAM might support other work; this
experiment does not measure additional KV, context or concurrency benefits.
For this checkpoint, fixture and fixed reservations, all-resident execution is
the measured performance choice when its device allocation is acceptable. Under
an imposed smaller expert envelope, the explicit adaptive configuration improves
the tested general-to-code sequence at each admitted point. That recommendation
does not extend to another workload, model geometry or use of the freed memory.

## Routing evidence

A separate observation-only engine records canonical counters at controlled
request boundaries. It performs no policy updates or movements, retains
generation-zero maps, matches the timed output IDs and closes cleanly. Each
validated initial placement is scored against that same captured route trace;
the following fractions are counterfactual routing diagnostics, not observers
inserted into timed static serving.

| Placement | General cold selections | Code cold selections | Combined cold selections |
|---|---:|---:|---:|
| 5 GiB general profile | 24.22% | 61.13% | 42.68% |
| 8 GiB general profile | 6.14% | 39.93% | 23.04% |
| 13 GiB general profile | 0.61% | 15.53% | 8.07% |
| 17 GiB all resident | 0% | 0% | 0% |
| 8 GiB mixed profile | 14.03% | 24.83% | 19.43% |

The diagnostic contains 1,562,112 recorded decode selections after subtracting
its pre-traffic baseline, across all four request groups. It does not combine
prefill with decode. At 13 GiB, general traffic is nearly covered but code still
has meaningful cold demand; 78.2% resident payload is not an all-demand-covered
configuration.

Adaptive maintenance receipts report cold fractions of 25.65%, 11.01% and 3.75%
at 5, 8 and 13 GiB, respectively. Their observed selection counts are 1,562,112,
1,560,192 and 1,223,808. These are complete policy windows, with no fabricated
final snapshot: especially at 13 GiB, an unobserved tail remains after the last
maintenance. They must not be presented as identically covered whole-run
comparisons against the static replay. Mixed-profile adaptive windows report
12.19–12.31% cold with 1,555,584–1,560,192 observations. Raw windows retain their
counts and boundaries.

## Initial-profile sensitivity

At 8 GiB only, a separate calibration uses eight independently authored prompts:
two each from general, code, math and multilingual traffic, with 128 generated
tokens each. Its 1,024-token effort matches general calibration and remains
disjoint from evaluation. This is a broad four-domain mixture, not an optimized
profile for the evaluation's equal general/code mix. The resulting profile ID is
`b717f184335bb13c0dbe58ff1ab5d4de4eba311f56c3763eac78ddfe9c45e761`.
It has the same 58 residents per layer and shares 2,169 of 2,784 resident experts
(77.9%) with the general-calibrated placement.

| Initial profile | Overall static | Overall adaptive | Adaptive gain | General static / adaptive | Code static / adaptive |
|---|---:|---:|---:|---:|---:|
| General | 85.456 ± 0.068 | 121.062 ± 0.073 | +41.67% | 172.907 / 172.096 | 56.822 / 93.607 |
| Four-domain mixture | 89.210 ± 0.032 | 113.049 ± 0.197 | +26.72% | 113.554 / 135.393 | 73.580 / 97.269 |

Values are output tok/s, mean ± sample SD where shown. Mixed-profile static
samples are 89.188, 89.246 and 89.196; adaptive samples are 113.276, 112.917 and
112.955. Its paired gains are 27.01%, 26.52% and 26.64%.

The mixed profile improves static overall throughput by 4.39% and code throughput
by 29.49%, while sacrificing general throughput. Adaptation still helps both
intervals from this placement; it does not disappear with broader calibration.
However, mixed-profile adaptive overall throughput is 6.62% below general-profile
adaptive. Initial coverage and the path of subsequent adaptation matter, not just
the count of resident bytes. This one small calibration sample does not establish
an optimal representative-mixture profile or a universal adaptive gain.

The three mixed adaptive trials perform 37, 35 and 35 maintenance operations,
with 1,184, 1,120 and 1,120 promotions. They copy 2.929, 2.770 and 2.770 GiB;
blocked scheduling totals 2.974, 2.804 and 2.805 seconds. Each includes one
maximum-interval trigger, with the remainder pressure-triggered. Exact outputs
remain identical despite different observation boundaries. The separately timed
calibration generates its 1,024 tokens in 27.09 seconds, excluding startup,
profile construction and shutdown.

## Control cost and latency

| Envelope | Maintenance / run | Promotions / run | API copy GiB / run | Blocked scheduling, mean | Final control tail, mean | Median proposed / selected / pair-cap skipped |
|---|---:|---:|---:|---:|---:|---|
| 5 GiB | 123 | 3,936 | 9.736 | 11.666 s | 39.41 ms | 95 / 32 / 63 |
| 8 GiB | 34 | 1,088 | 2.691 | 2.812 s | 4.65 ms | 96 / 32 / 64 |
| 13 GiB | 4 | 128 | 0.317 | 0.221 s | 6.46 ms | 95.5 / 32 / 63.5 |

All three trials at a capacity take the same movement decisions. Every
non-baseline maintenance operation saturates the pair cap; none is byte-cap
limited. Median proposing-layer count is 48. This establishes an admissible
proposal backlog, not that accepting every proposal would improve throughput.
The primary sweep does not retune movement capacity.

At 5 GiB all 123 operations are pressure-triggered. At 8 GiB one is triggered
by the maximum interval and 33 by pressure. At 13 GiB two are maximum-interval
operations and two are pressure-triggered. These wins cannot be attributed
entirely to health detection latency.

Blocked scheduling includes draining useful work already submitted. It is not
all idle overhead. Worker policy/apply stages are nested in the engine operation;
they are not summed or added again to measured serving wall time. Overall
throughput includes the final pending-control tail. Startup, calibration and
explicit shutdown are outside serving time.

| Envelope | TTFT p50, static / adaptive | TTFT p95 | Delivery gap p50 | Delivery gap p95 | Delivery gap p99 |
|---|---:|---:|---:|---:|---:|
| 5 GiB | 492.06 / 436.10 | 676.40 / 577.38 | 73.14 / 57.65 | 126.19 / 97.56 | 135.97 / 120.41 |
| 8 GiB | 347.77 / 287.35 | 459.81 / 431.00 | 39.07 / 27.33 | 88.93 / 58.16 | 99.36 / 84.57 |
| 13 GiB | 249.12 / 238.76 | 271.22 / 266.61 | 18.26 / 16.35 | 39.15 / 32.50 | 43.02 / 36.26 |
| 17 GiB | 230.72 / — | 232.73 / — | 11.56 / — | 12.46 / — | 12.71 / — |
| 8 GiB, mixed calibration | 319.32 / 307.02 | 330.68 / 357.26 | 39.50 / 30.90 | 75.98 / 55.68 | 83.57 / 83.73 |

Values are milliseconds, averaged across each trial's quantile. Client delivery
gaps are not GPU iteration timings. Raw request events preserve per-request
decode rates and interval distributions. The mixed-profile adaptive arm has
worse p95 TTFT despite higher overall throughput; its p99 delivery gap is nearly
unchanged. Aggregate throughput does not conceal that latency tradeoff.

## Lifecycle and qualification

All 27 timed trials pass the existing serving acceptance, generating 110,592
tokens and completing 18,880 promotions. Every trial has output-token SHA-256
`b610b6c6e4efd384474e475165debbc1143df5c3653195c13078ad952b3fb40e`.
The gate checks exact paired output IDs, expected generations, unchanged
graph/cache addresses within each engine and immutable profile artifacts.
Each of the five serving scopes also passes an ordinary non-cache graph smoke.

All capacity arms retain the CPU-source loading contract: the loader's recorded
device peak is 1.670 GiB, rather than first materializing the complete routed
checkpoint on GPU. Resource receipts preserve repeated preparation checkpoints,
graph/KV readiness, serving and release; no intermediate stage is discarded merely
because it shares a label with another checkpoint.

Every completed worker explicitly releases mapped backing, CPU expert sources,
graph owners and pending health reads. Normal timed shutdown is approximately
9.9–10.0 seconds, with no forced termination or ignored destructor exception.
The observed 459 MiB of live Torch storage before worker process exit is the
same process-lifetime engine residual documented by reference qualification,
not newly attributed to a cache leak. These runs preserve the prior cancellation
and bounded-lifecycle implementation; they do not relabel old cancellation tests
as new fault injection.

| Envelope | Configuration to first request, static / adaptive | Explicit shutdown, static / adaptive |
|---|---:|---:|
| 5 GiB | 77.92 / 78.72 s | 9.92 / 9.96 s |
| 8 GiB | 75.75 / 76.25 s | 9.91 / 9.96 s |
| 13 GiB | 78.83 / 79.49 s | 9.97 / 10.02 s |
| 17 GiB | 73.79 / — s | 10.00 / — s |
| 8 GiB, mixed calibration | 75.54 / 76.26 s | 9.92 / 9.98 s |

These means describe fresh engine processes with the retained compiler cache,
not a cold build from an empty artifact cache. Capacity-specific general profiles
reuse verified calibration counts and do not incur another model calibration run.

The bounded gates pass:

- Independent GitHub-hosted acceptance at `06f0efc3`: **1,097 passed, 66 visible
  skips**, covering host contracts and documented physical/compile exclusions.
  [Actual hosted run](https://github.com/local-inference-lab/b12x/actions/runs/35555121150).
  The timed source `d28ee256` separately passed 1,094 host tests with the same
  66 skips in its [own run](https://github.com/local-inference-lab/b12x/actions/runs/35550584691).
- The existing portable/SM120 tier at `d28ee256`: **41 passed, zero skips**,
  including actual checkpoint arithmetic, pending-read ownership and reduced
  25%, 75% and 100% resident cases. Capacity extremes compare native prepared
  whole-K outputs under changing logical routes and fixed graph/storage addresses,
  with replay allocation checks.
- Targeted capacity-extreme memcheck and synccheck at the same source:
  **three passed each, zero sanitizer errors**. Read-only pytest-cache warnings
  remain in the logs; no warning filter makes the gates pass.
- Companion loader/maintenance/lifecycle tests from `c3efbdf`, imported against
  its verified installed source-built wheel: **18 passed**. Tests are copied
  outside the unbuilt source checkout to prevent that checkout from shadowing
  installed native artifacts; their source hashes are retained.
- The completed observation-only route diagnostic has exact matching outputs,
  immutable maps, no maintenance and released owners. Corrected preflight at
  `3c534d0` preserves every measured placement and admits all four points after
  charging the observed additional engine allocation.

The first route-analysis attempt failed because the parser expected a `result`
field while the existing harness emits `routing_boundary.receipt`. The parser
and host fixture are corrected at `06f0efc3`; the failed analysis and intermediate
test logs remain in the bundle. No timed serving result is replaced by that fix.
Development preflight/fixture failures and expected admission rejections also
remain separate from qualified runs. There are no failed timed arms hidden from
the reported samples.

## Native target and interpretation

No explicitly configured, authorized B300/GB300 endpoint was accessible when the
track was selected. The existing [SM103 qualification](sm103-qualification.md)
launcher is unchanged. There is no new native execution acceptance, Grace TMA
evidence or B300 performance claim. The native MXFP4/MXFP8 full-model source and
preparation bridge remains behind physical operator prerequisites.

The controlled capacity/initial-placement comparisons are policy and
whole-system evidence for this supported model. Their absolute miss/copy costs
are specific to ripper's Gen4 x16 topology. No Gen5 or Grace result is extrapolated.
The previous source-bound 42.82% reference gain remains historical; the new
capacity tooling did not cause a runtime optimization.

Physical B300 correctness remains the higher-priority next hardware gate. If
that hardware remains unavailable, the next useful generalization is a supported
checkpoint whose complete expert payload cannot fit under honest fixed serving
reservations. It should reuse this capacity/calibration procedure rather than add
another policy variable.

## Receipts

Raw evidence is retained at
`ripper:/home/jasonc/b12x-capacity-results-20260920`, with a local copy under
`/home/jasonc/b12x-capacity-evidence-20260920`. These are evidence locations, not
required reproduction paths. The source-built wheel and build remain under
`ripper:/models/b12x-reference-build-20260920`.

The bundle includes exact launchers and commands, frozen source exports,
calibration counts and profile artifacts, artifact verification, source-pair
acceptance, output IDs, graph/storage checks, all resource checkpoints, raw
timings, topology/telemetry, rejected plans and independent host-CI artifacts.
Diagnostic runs are separate from timed arms. Historical evidence is unchanged.
