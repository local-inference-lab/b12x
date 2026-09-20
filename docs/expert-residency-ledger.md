# Expert residency engineering ledger

This ledger separates implementation evidence, compiler evidence, portable GPU
correctness, and deferred SM103 qualification. Raw receipts are retained outside
the repository at `/home/jasonc/b12x-residency-evidence-20260918` on the development
host and `/home/jasonc/b12x-residency-20260918` on the portable GPU host.

## Model-wide epoch control, September 19, 2026

The implementation starts from `6d3cf32322a2566ad6c8dec9f5a0a9e3967c2fc8` on
`work/sm103-bringup`. Live master is
`0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`; the branch is 79 commits ahead and
zero behind before this change. No rebase or master modification is needed.
The companion audit confirms vLLM main at `47ccf6c57d92f03630ebcbad3809450545825488`,
the PreparationSession branch at `ef1aeaf080879865febd27a92c5644233d157987`,
the Spark lane source at `76061de4bff2adc741cb25018ca79991263228be`, and the retained
SM103 companion at `f6c6ac72c3`. SGLang main remains
`c662a9fd6e35c2548c45c09ece8d3f47846d61f2`. No companion engine checkout or live
serving lane is changed.

Implemented: layer-scoped experimental decayed LFU, explicit canonical/exclusive
map transitions, subset acknowledgement, a globally bounded model epoch, one
counter-slab snapshot, prepared SM103 binding adaptation, and a vLLM
pause/worker-extension protocol. Rank-wide preflight precedes copies; partial
rank/layer failure requires coordinated reload. Static serving defaults and
numerical kernels remain unchanged. The
[epoch guide](expert-residency-epochs.md) specifies configuration, memory,
ownership, diagnostics and the missing loader/backend integration.

Final implementation/test files are frozen in `source-03.tar.gz`, SHA256
`9f73c4ac64dcf09e490d8f08deb7c70d6d3edb4a055c82ca07223c51c596b05e`.
Raw evidence is retained at
`/home/jasonc/b12x-model-epoch-evidence-20260919` and
`ripper:/home/jasonc/b12x-model-epoch-results-20260919`. Source manifests bind
every implementation/test file; documentation is completed after measurement.
Earlier source-01/source-02 logs remain attached to their own archives.

| Validation | Exact result and limit |
| --- | --- |
| Focused host suite | 199 passed, 10 GPU skips in 3.04 s; `host-final-02.log`. Includes per-layer and global budgets, full/partial acknowledgements, canonical maps, stable/shifted traffic, disabled path, stale generation, rank/layer failure, cancellation, JSON round-trip and actual engine parallel-config checks. |
| Portable SM120 suite | 20 passed, 2 native-SM103 skips in 12.54 s; `gpu-final.log`. Full two-layer cases use one graph for 16 policy windows, plus bounded cases and existing counter/metadata tests. Payload identity, exact counts, addresses and zero replay allocations pass. This is not vLLM model execution. |
| Bounded memcheck | One two-layer int64 case passed; zero reported errors. Full payload checks remain in the unsanitized full case. |
| Bounded synccheck | One two-layer int64 case passed; zero reported errors. This does not qualify SM103 TMA. |
| SM103 counter cross-compilation | Two sampling queries, four int32/int64 programs; CUDA remains uninitialized. No GPU operation, preparation declaration or core compute specialization is added by the coordinator. The historical 85-declaration/241-program census is not rerun or relabeled. |
| Checkpoint storage audit | Header hashes and exact serialized byte totals in `checkpoint-memory.json`. Research payload arithmetic requires 31.641 GiB resident plus 63.281 GiB canonical backing at 256/512 across 48 layers, before other model reservations. This is not whole-model admission. |
| Serving and B300 | Deferred. No throughput, TTFT, ITL, full-model quality, scheduler-pause benchmark or physical SM103 result is claimed. |

The portable GPU is RTX PRO 4000 Blackwell UUID
`GPU-cc109c01-9756-d0db-21ea-f1825d3f963f`, driver 580.173.02, CUDA 13.3,
Torch 2.13.0, CUTLASS DSL 4.6.2, Triton 3.7.1+gitf797708c.nv26.7,
cuda-bindings 13.0.3. The receipt records PCIe Gen4 x16 and a 145 W power limit.
These correctness runs use ordinary dynamic clocks and make no timing comparison.
The final counter compiler manifest binds package SHA256
`744f8280e711989eae7f1ff4677a58e7d57636e6646c335f8a651d236e847b33`
and records the separate host toolchain: Torch 2.14.0, CUTLASS DSL 4.6.2 and
Triton 3.8.0. Its four programs compile with CUDA uninitialized; this is not a
physical SM103 result.

Retained failures and rejected approaches:

- The full epoch test under memcheck reaches the explicit 180 s timeout without
  a test result (`memcheck-01.log`, exit 124). The bounded scale-reader case
  completes under memcheck and synccheck; it does not erase that timeout.
- The first host admission assertion expected 760 bytes from categories that
  sum to 660. The test oracle is corrected; admission arithmetic is unchanged.
  The failure is recorded in `initial-host-failure.txt`.
- Ruff passes for all added Python files. A broader check reports the unchanged
  `RoutingProfileConfig` import and non-strict candidate/victim `zip` already
  present at the starting commit. `lint-baseline.log` reproduces both findings
  on that source; they are not silently reported as a passing repository lint.
- Cumulative counts cannot implement the ordering-dependent offline LRU policy.
  LRU remains trace research; recent-frequency remains the default and decayed
  LFU is explicitly selected.
- The vLLM public async pause includes a fixed 20 ms sleep. The adapter uses the
  supported boundary and includes its cost; no private scheduler bypass or
  unmeasured four-token serving pause is introduced.
- Research SM120 fills are not imported into a production quantization method.
  Public prepared storage, CPU-source checkpoint loading, real phase hooks and
  whole-model memory admission must exist before a serving A/B is valid.
- An individually resumable copy error is not model-wide rollback. Distributed
  automatic recovery is deferred; all ranks must reload after partial failure.

The next integration gate is a CPU-source loader and a public SM120 prepared
cache backend with preserved numerical recipe. Then wire counter bindings and
runtime registration through the maintained PreparationSession lifecycle and
measure learned-static versus adaptive serving, including all pauses. Timeline
evidence must precede asynchronous replacement, shared scratch admission or
policy-default changes. Earlier 141/181-fixture and held-out replay receipts
remain immutable.

## Source and upstream evidence

The working branch is rebased onto master
`8783519a3e42c22c0f395669ca4b20c69439c974`. The pre-rebase branch is preserved as
`backup/sm103-before-residency-20260918` at
`339829140837dd9ce604fde2f82db8b7ceeaddd1`. Master is not modified.

The recipe review uses J-M-Recipes revision
`ed5a932956d2a60bd38191ae9de5abe064c7586f`, including the actual
[hot-expert hook](https://github.com/J-M-Recipes/recipes/blob/ed5a932956d2a60bd38191ae9de5abe064c7586f/recipes/dgx-station-gb300/deepseek-v4.1-flash-vllm-uva-dspark/results/2026-09-17-e2b-pin-hot-experts-v15/hook/pin_hot_experts_hook.py),
[confirmation results](https://github.com/J-M-Recipes/recipes/tree/ed5a932956d2a60bd38191ae9de5abe064c7586f/recipes/dgx-station-gb300/deepseek-v4.1-flash-vllm-uva-dspark/results/2026-09-17-e2c-v15-confirm),
[adaptation and alternate-workload results](https://github.com/J-M-Recipes/recipes/tree/ed5a932956d2a60bd38191ae9de5abe064c7586f/recipes/dgx-station-gb300/deepseek-v4.1-flash-vllm-uva-dspark/results/2026-09-17-e4-e4b-e5-placement-adaptation),
and the earlier placement, staged-copy, ATS, and VMM investigations under that
recipe. The hook obtains BF16 unfinalized expert outputs and performs an ordered
FP32 FMA reduction. Separate tier finalization is not numerically equivalent.
The confirmation contains both throughput variation and prompt disagreement;
it is not a full quality proof. Instrumented online adaptation costs and
workload-specific placement tradeoffs support a static baseline with no default
online telemetry. Those upstream timings are not b12x measurements.

The checkpoint configuration is pinned to
[DeepSeek-V4.1-Flash revision df42c109](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/config.json).
The activation and weight audit also checks the
[vLLM MXFP4 expert adapter](https://github.com/vllm-project/vllm/blob/8a5cf5438728180210aea43897d6a717de4b46ec/vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py)
and [FlashInfer's routed API](https://github.com/flashinfer-ai/flashinfer/blob/01045366e15df38e19e76050592f82e49b4cff64/flashinfer/fused_moe/core.py).
Downloaded files match these pinned revisions byte-for-byte; SHA256 values are
in `upstream-source-manifest.json`. Native MXFP4 weights do not imply A4
activations. The implementation declares A8 explicitly and preserves packed
checkpoint values.

## Implemented files

| Files | Responsibility |
| --- | --- |
| `b12x/moe/fused_moe/residency.py` | Immutable placement, hash/version, budgets/accounting, per-layer trace profiles |
| `b12x/moe/fused_moe/_residency_tuning.py` | Registered typed query, configuration, eligibility, numerical contract |
| `b12x/moe/fused_moe/_residency_storage.py` | Slab layout, bounded expert copies, exact scale swizzle, mapped-host ownership |
| `b12x/moe/fused_moe/_residency_preparation.py` | Compile jobs, retained programs, materialization, admission, fixed workspace and bindings |
| `b12x/moe/_shared/kernels/sm103/residency.py` | CuTe route compaction, MXFP8 boundaries, native routed GEMMs, ordered FMA finalizer |
| `b12x/gemm/_shared/sm103_blockscaled.py` | Mixed E4M3/E2M1 tcgen05 operands and compact route/count guards |
| `b12x/moe/fused_moe/api.py` | Hierarchical variant through public plan/bind/run; stale-binding rejection |
| `b12x/moe/fused_moe/__init__.py` | Public contract exports and API metadata |
| `b12x/moe/fused_moe/planning.py` | Explicit source-native MXFP4 A8 weight declaration |
| `b12x/moe/fused_moe/weights.py` | Checkpoint/layer identity in source bundles |
| `b12x/moe/_shared/execution.py` | Typed native MXFP4 preparation transform/layout |
| `b12x/preparation/catalog.py` | Authoritative residency variant registration |
| `b12x/attention/sparse_mla/_tuning.py` | Rebase fix: preserve native pooled-selection eligibility |
| `scripts/build_expert_residency_profile.py` | Offline JSONL trace-to-profile command |
| `scripts/_sm103_preparation_corpus.py` | Production compile-factory coverage for the residency declaration |
| `scripts/compile_sm103.py` | Parameterized residency cross-compilation and resource receipts |
| `benchmarks/moe/expert_residency.py` | Physical-only operator qualification, all-HBM control, graph/stage timings |
| `tests/moe/test_expert_residency.py` | Host admission, identity, lifecycle, profile, and storage-copy gates |
| `tests/moe/test_residency_kernels.py` | Portable numerical, graph, allocator, and mapped-host gates |
| `tests/moe/test_sm103_residency.py` | Deferred public SM103/Grace operator and graph qualification |
| `tests/preparation/test_device_reclaim.py` | Rebase test correction: enable cache selection for the cached-restart assertion |
| `docs/expert-residency.md` | Public contract, ownership, limits, exact qualification commands |
| `docs/expert-residency-ledger.md` | Evidence and rejected-experiment record |

Rebase conflict resolutions also preserve master's DSA operand/page geometry,
sparse-MLA pool fields, fused-MoE tuning changes, and MXFP8 allocation-counter
checks. Typed query/config/candidate versions preserve both upstream and SM103
contract changes. These resolutions are in the rebased commits, separately from
the residency implementation.

## Validation evidence

This section records the static-residency baseline at `78a8704f`. Automation
validation is recorded separately below; its totals must not be added to these
counts as if they were one run.

The baseline package source SHA256 is
`a3e65748153cb354026e3053fdb255cdd3693a6ca6d01baa16c64e0b59ef5bd9`.
Compiler receipts bind the actual package hash independently of the Git base.

| Gate and command | Result |
| --- | --- |
| `python -m pytest tests/preparation tests/architecture tests/moe/test_expert_residency.py tests/moe/test_sm103_residency.py tests/moe/test_fused_moe_variant_selection.py -q` | **930 passed, 63 skipped**; `host-regression-final.log` |
| `compute-sanitizer --tool memcheck --error-exitcode 99 python -m pytest tests/moe/test_residency_kernels.py -q` | **8 passed, 0 sanitizer errors**; `portable-memcheck-layouts.log` |
| `python -m pytest tests/moe/test_fused_moe.py tests/attention/test_glm_next_mla.py -k "run_w4a16_replays or pooled_selection" -x -q` | **2 passed, 45 deselected** on SM120; `rebase-portable-regression.log` |
| `python scripts/compile_sm103_prepared.py --output-dir /evidence/prepared-final --workers 2` | **83 declarations, 239 distinct programs compiled**, no CUDA context; `prepared-final/manifest.json` |
| Production-size residency compile, H=5120, I=2304, E=384, HBM=295, capacity=128, top-k=6, clamp=10 | **12 entry points cross-compiled**, PTX/cubin/SASS/resources hash-bound; `production-final/manifest.json` |
| All-HBM W31 and all-Grace W13 compile, H=I=256, E=4, capacity=8, top-k=3 | **10 entry points each**; empty tier omitted; `all-hot-compile/manifest.json`, `all-cold-compile/manifest.json` |
| Profile CLI smoke with different per-layer distributions | Correct distinct placements and cold fractions; `profile-smoke/` |
| Physical benchmark on a host without SM103 | Rejected before timing, failed receipt retained; `no-sm103-qualification.json` |

Portable GPU: `ripper`, RTX PRO 4000 Blackwell, SM120,
`GPU-47363510-b87a-13a5-4824-2542e97df76c`, driver `580.173.02`,
Torch `2.13.0+cu130`, Torch CUDA build `13.3`, CUTLASS DSL `4.6.2`, Triton `3.7.1`.
Sanitizer uses isolated `cuda-bindings==13.0.3` and Compute Sanitizer `13.4.57`.
The shared serving environment is unchanged. No serving service is stopped.
The final portable suite covers both gate orders, invalid/sentinel and oversized
int64 IDs, duplicate routes, all-hot/all-cold selections, multiple live M/top-k,
changed graph inputs, stable pointers, exact allocator counters, explicit FMA
and rank-order adversaries, and mapped-host reads. It does not execute SM103
expert GEMMs on SM120.

Compiler host: Torch `2.14`, CUTLASS DSL `4.6.2`, Triton `3.8`. Exact package
versions and compiler/disassembler versions are in the manifests. Production
geometry resource results:

| Program | Registers | Static SMEM | Dynamic launch SMEM | Stack/local bytes |
| --- | ---: | ---: | ---: | ---: |
| Partition, either ID width | 18 | 0 | 0 | 0 |
| Ordered finalize, either ID width | 32 | 0 | 0 | 0 |
| Input quantization | 46 | 0 | 0 | 0 |
| Activation quantization | 48 | 0 | 0 | 0 |
| Hot/cold FC1 | 142 | 1,024 | 51,328 | 0 |
| Hot/cold FC2 | 140 | 1,024 | 51,328 | 0 |

Native mixed blockscaled MMA and TMEM completion waits are present in the
compiler artifacts. There are no local-load/store spill instructions. Allocated
registers and launch storage are compiler evidence; occupancy, utilization,
latency, and bandwidth are unmeasured.

For that single-layer placement, owned expert storage is **5,546,188,800 HBM
bytes** and **1,673,256,960 mapped-host bytes**. Capacity scratch is **51,851,520
bytes**, and the map is **3,072 bytes**. These are layout calculations, not an
observed model-wide memory footprint.

## Failed or rejected experiments

| Experiment/approach | Disposition and retained evidence |
| --- | --- |
| Treat dense MXFP4 support as routed support | Rejected by the capability audit. The routed native backend was missing. |
| Infer A4 from FP4 checkpoint storage | Rejected after checking activation contracts. A8 is declared independently. |
| Requantize weights to NVFP4 | Rejected; source FP4/E8M0 bytes must retain their numerical meaning. |
| Finalize each tier and add | Rejected by portable BF16-rounding adversaries and the upstream operator evidence. |
| Separate FP32 multiply/add or regroup route ranks | Rejected by explicit residual/cancellation tests; finalization uses ordered `fma.rn.f32`. |
| High-level Torch route masking and online adaptation | Rejected for the baseline; upstream reports measurable instrumentation costs. |
| VMM-backed shared-VA expert mapping | Not adopted; upstream ATS/VMM investigations show worse operand bandwidth for that allocation strategy. The existing mapped-host allocator is reused. |
| CuTe early return inside the compact-count guard | Compilation rejected staged early exit. Replaced with a CTA-uniform body guard. `compile-first/` retains the partial artifacts; the error is recorded here. |
| Bind an unprepared residency plan through generic lazy preparation | Host purity gate exposed first-use default preparation. Residency bind/run now require an explicit prepared state; no runtime compilation escape. `host-first.log` retains the failure. |
| Source-native H=128 A8 declaration | Existing public W4A8 planner rejects H not divisible by 256. The prototype retains that public constraint; no hidden geometry bypass. |
| Broad host suite after rebase | Initial result: 925 passed, 63 skipped, 1 failed. The reclaim test requested autotuning disabled but expected a cached tuning selection. Its cached-restart session now enables selection, matching the existing disabled/default contract. Both failure receipts are retained. |
| Sanitizer with shared cuda-bindings 13.4.1 on a CUDA 13.0 driver | 82 API probe errors, all version/`cuGetProcAddress_v2` failures; no invalid-memory report. Not counted as a clean sanitizer run. Isolated CUDA 13.0.3 bindings produce zero errors. `portable-memcheck.log` and both successful reruns remain available. |
| Execute a benchmark without physical SM103 | Fails closed and writes a failed receipt. No substitute SM120 MoE timing is reported. |

## Deferred qualification and optimization order

Physical B300/GB300 is unavailable. The six parameterized complete-operator tests
are skipped, including real HBM/Grace split parity, TMA legality on Grace pages,
and public-plan graph replay. No checkpoint requests, model quality claims,
B300 timings, overlap measurements, C2C traffic, occupancy, or power results are
reported. The vLLM companion branch is not changed, and no compatibility APIs or
sitecustomize hooks are introduced.

Optimization priority is provisional; no B300 latency evidence exists to rank
speedups. The available compiler/structural evidence supports this order:

1. Run the physical correctness and sanitizer gates, then the per-stage harness.
   Establish whether coherent mapped operands are legal and competitive for TMA.
2. Reduce tiny-M projection waste and route-major intermediates. Each selected
   route still occupies a 128-row tile, 512 TMEM columns, and 140–142 registers.
   Shared input quantization and smaller/grouped schedules address visible work.
3. Measure the serial partition scan and empty-tier launch cost. Parallel
   compaction, prepared variants, or conditional graph nodes require evidence
   that their bookkeeping costs are lower.
4. Evaluate bounded cold staging and overlap only after the exact baseline is
   qualified. Any copy/stream/event path must preserve ownership and final rank
   order. Upstream split-versus-staging results motivate measurement, not a
   portable assumption about B300 TMA behavior.
5. Measure optional counters on the real B300 route before enabling persistent
   monitoring. Static usage-aware profiles remain the baseline. Portable
   counter evidence is recorded below and does not establish negligible cost.


## Automatic residency evidence

The automatic lifecycle extends branch HEAD `042d2734`; the master base remains
`8783519a3e42c22c0f395669ca4b20c69439c974`. The package SHA256 is
`aac5c587557754e1d3f4f59da6941f15bdb7cf41e715a05205288543f08c77c7`.
Raw receipts are retained under
`/home/jasonc/b12x-auto-residency-evidence-20260918` on the development host.
GPU receipts retain their source hashes and physical UUIDs. The
[automatic SM103 guide](expert-residency-automatic.md) is the canonical lifecycle
and integration specification.

| Files | Responsibility |
| --- | --- |
| `b12x/moe/fused_moe/automatic.py` | Typed model/configuration/budget contracts, greedy placement, schema-2 artifacts/store, convergence/drift controller, bootstrap and static-plan bridge |
| `b12x/moe/fused_moe/_routing_profile_tuning.py` | Registered counter query/configuration, capacities, phase and TP ownership |
| `b12x/moe/fused_moe/routing_profile.py` | Real preparation, retained programs, stable counter slab, bindings, quiescent reset/snapshot/control |
| `b12x/moe/_shared/kernels/routing_profile.py` | CuTe selection counters, periodic sampling and sticky uint64 overflow detection |
| `b12x/integration/vllm/expert_residency.py` | Explicit engine/worker hooks; no patch installation or placement math in vLLM |
| `b12x/_lib/architecture.py`, `b12x/preparation/catalog.py` | Exact kernel admission and component registration |
| `b12x/moe/fused_moe/api.py`, `b12x/moe/fused_moe/__init__.py` | Public configuration and profiling exports |
| `b12x/moe/fused_moe/residency.py` | Verified schema-2 artifacts can supply ordinary static layer plans |
| `b12x/testing/vllm_routing_trace.py` | Explicit diagnostic labels and unique experts per captured call |
| `scripts/_sm103_preparation_corpus.py` | Counter declaration in the production compiler corpus |
| `scripts/compile_routing_profile.py` | Configurable calibration/monitor counter cross-compilation and hashed artifacts |
| `scripts/inspect_expert_residency_profile.py` | Integrity/identity inspection and per-layer placement/memory summary |
| `benchmarks/moe/routing_profile.py` | Off/on/sampled portable or physical profiler diagnostic with exact oracles and raw receipts |
| `benchmarks/moe/expert_residency.py` | Model-profile geometry/recipe/capacity checks before physical qualification |
| `tests/moe/test_automatic_residency.py` | Model-wide budgeting, deterministic placement, convergence, store validity, lifecycle, TP and drift gates |
| `tests/moe/test_routing_profile_gpu.py` | Counter correctness, replay, ownership, sampling, concurrent atomics and worker lifecycle gates |
| `docs/expert-residency-automatic.md` | Detailed SM103 configuration, integration and operational guide |
| `docs/expert-residency.md`, `docs/expert-residency-ledger.md`, `docs/sm103-readiness-report.md`, `docs/sm103-change-summary.md` | Public contract, evidence and readiness reconciliation |

Host regression command:

```bash
python -m pytest tests/moe/test_automatic_residency.py \
  tests/moe/test_expert_residency.py tests/moe/test_sm103_residency.py \
  tests/moe/test_fused_moe_variant_selection.py tests/preparation tests/architecture -q
```

Result: **977 passed, 63 skipped** (`host-acceptance.log`). The 46 automatic-lifecycle
host tests are included in that run. Temporary files
use an evidence-local `TMPDIR` because the shared `/tmp` quota was exhausted.

The full production preparation compiler corpus passes with **84 declarations,
241 distinct programs**, including 235 native CuTe exports, without initializing
CUDA (`prepared-acceptance/manifest.json`). The new counter entries are int32 and
int64 IDs at E=384, M capacity=128 and top-k capacity=8. Resource receipts report
14/13 registers, 1,024 static SMEM bytes, 4 dynamic SMEM bytes and no stack/local bytes; PTX includes
uint64 atomic add and the sticky overflow OR. This is compilation evidence, not
SM103 runtime or occupancy evidence.

The configurable counter compiler additionally passes **four SM103 entry points**
for int32/int64 IDs at sampling intervals 1 and 128 without CUDA initialization
(`counter-acceptance/manifest.json`). The production-size 12-entry-point residency
compile is retained in `production-acceptance/manifest.json`.

GPU validation of the final package uses RTX PRO 4000 Blackwell SM120,
`GPU-cc109c01-9756-d0db-21ea-f1825d3f963f`, driver `580.173.02`, Torch `2.13.0`,
Torch CUDA build `13.3`, CUTLASS DSL `4.6.2`, Triton `3.7.1+gitf797708c.nv26.7` and isolated
CUDA bindings `13.0.3`. Compute Sanitizer reports:

- **5 passed, zero memcheck errors** (`gpu-acceptance/gpu-memcheck-acceptance.log`).
- **5 passed, zero synccheck errors** (`gpu-acceptance/gpu-synccheck-acceptance.log`).

These gates cover exact counters, duplicate/invalid/oversized IDs, both ID widths,
changing live M/top-k, frozen compilation, capture/replay, stable pointers, zero
allocator events, control/reset/overflow, aliases, sampled phases, replicated TP
ownership, concurrent producers, and the worker's replay-to-profile-to-next-startup
lifecycle. The lifecycle test uses SM103 host deployment metadata with portable
counters; it executes no SM103 expert kernel on SM120.

The preceding combined portable run also passed **13 tests with zero memcheck
errors**, including the unchanged residency quantization/finalizer/mapped-host
kernels (`gpu-initial/gpu-memcheck-first.log`). Its snapshot precedes the host-side
phase and counter-alias guards; the final five-test runs validate those guards.

Profile inspector smoke passes (`inspector-smoke.log`). An initial evidence
locator glob matched no pytest directory and raised `ValueError`; correcting the
truncated fixture-directory prefix allowed the inspector command to run. No
profile-format failure was suppressed.

### Portable profiler timing

`gpu-acceptance/profiler-uniform-acceptance.json` records E=384, M=1 through 128,
top-k=1/6/8, eight alternating samples of 1,024 device operations captured within
each timing graph. Exact route/counter oracles and allocator checks pass before
timing. The table shows the top-k=6 subset, in microseconds:

| M | Partition, profiling off | Partition + count every call | Partition + sample every 128 calls |
| ---: | ---: | ---: | ---: |
| 1 | 1.458 | 3.098 | 2.461 |
| 2 | 2.067 | 3.710 | 3.071 |
| 4 | 3.552 | 5.199 | 4.556 |
| 8 | 6.479 | 8.133 | 7.482 |
| 16 | 12.380 | 14.038 | 13.383 |
| 32 | 24.184 | 26.163 | 25.194 |
| 64 | 47.670 | 49.982 | 48.682 |
| 128 | 83.600 | 86.854 | 84.611 |

The ratio direction is **profiled / off**: at M=1, every-call counters cost
about **2.125x** the partition-only stage; at M=128, about **1.039x**. Sampling
still pays approximately a microsecond for the observer launch/control path.
This is a portable component diagnostic, not full MoE, C1 tok/s, verifier or B300
performance. It is not formal release tuning evidence.

Per-shape before/after snapshots remain P1 with no active software power cap.
For this table, SM clocks match within each pair: 2,055 MHz through M=16 and
2,460 MHz thereafter; power limit is 145 W. Clocks differ across shapes, so the
table does not isolate a pure scaling law. All raw samples and full GPU snapshots
are retained, together with package/harness hashes, worktree, command and UUID.

`profiler-contention-acceptance.json` repeats top-k=6 with every selection targeting
expert zero. At M=128, partition alone is 72.925 us, every-call counting is
76.298 us and sampling is 73.949 us. The incremental count cost is comparable to
the uniform diagnostic; it does not justify complex warp aggregation ahead of
launch reduction. The contention M=16 before/after SM clocks changed and that row
is retained as a diagnostic, not a controlled comparison.

### Automation failures and rejected approaches

| Attempt or design | Disposition |
| --- | --- |
| First host artifact tests under shared `/tmp` | 15 failures, 21 passes; writes failed with disk quota exceeded. `host-first.log` is retained. Evidence-local temporary storage removes that environmental failure. |
| Minor-noise convergence test with a 0.015 bound | One failure exposed a fixture whose adjacent cold fractions differed by 0.02. The declared noise tolerance is 0.025; a separate large-change fixture still resets convergence. `host-second.log` is retained. |
| Counter compiler factory without architecture admission | Host corpus rejected the exact module on SM103. Narrow module admission was added after direct SM103 cross-compilation; no blanket architecture bypass. `host-regression-first.log` retains 953 passes, 57 skips and the failure. |
| Python loop replay for microsecond timing | The first SM120 diagnostic is retained as `profiler-uniform-first.json`; it can include host enqueue gaps. The benchmark captures repeated device operations in one graph before timing, removing that ambiguity. First-run timings are not used as final kernel-latency evidence. |
| Global free HBM passed independently to every layer | Rejected: permits aggregate overcommit and repeats KV reserves. The model allocator apportions one envelope and exact per-plan budgets. |
| Mutate slabs or invoke a worker restart from b12x | Rejected: storage/graph lifetime and engine request coordination are external. The implemented transition is a typed restart-required result. |
| Automatically enable rich trace or persistent monitoring | Rejected: rich tracing is diagnostic and the separate counter launch has measurable cost. Off is the default; monitor is explicit. |
| Fuse instrumentation into every router specialization | Deferred: adjacent counters cover external and native selected IDs without changing non-profiled route code. Physical full-MoE evidence is required before choosing a fused variant. |
| Spend hypothetical shared-workspace savings on hot experts | Rejected: no lane/stream/output lifetime contract exists for cross-plan sharing. The estimator reports potential savings; admission charges private storage. |

The layout audit reports 51,851,520 scratch bytes per qualification layer,
2,074,060,800 for 40 private workspaces, and a hypothetical one-lane saving of
2,022,209,280 bytes. These are exact layout calculations, not allocator savings.
No shared arena or concurrent aliasing was implemented.

Physical B300, Grace TMA, complete-model calibration, full-MoE profiling overhead
and companion vLLM serving remain unqualified. No checkpoint quality, C1 tok/s,
C2C bandwidth or overlap result is inferred from counters or cross-compilation.


### Automation follow-up order

1. Qualify physical SM103 HBM/Grace execution and wire the companion engine's
   PreparationSession, phase, polling and restart hooks before end-to-end claims.
2. Prototype an optional fused route/counter variant and parallel compaction.
   Portable data shows a material extra launch at tiny M and serial-partition
   growth at larger M. Keep the uninstrumented variant unchanged and compare on
   B300 before adopting either optimization.
3. Define an execution-lane workspace arena contract. The layout audit exposes
   about 1.88 GiB of potential savings for the qualification model, but graph,
   stream and output ownership must be explicit before admission can spend it.
4. Revisit counter aggregation or unique-expert-per-step scoring only if physical
   contention or traffic evidence warrants it. These diagnostics do not show a
   large enough contention penalty to prioritize a more complex counter kernel.

Tiny-M tcgen05 scheduling and HBM/Grace overlap remain the underlying backend's
physical-measurement work; counter results cannot rank their whole-model gains.

## Residency policy review evidence

The review starts at `2de9d31ca6784916cba8867321db691a47fbe887` and rebases its
70 commits onto master `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`. The rebase
completes without conflicts; `git range-diff` reports the same patches for all
70 commits. The prior branch is retained locally as
`archive/sm103-before-review-20260918`. Master includes exact stable QSA winner
construction and the sparse-MLA TP3 16+8 head-partition fix.

The policy implementation binds package SHA256
`6b1c98b0af787cfd0900d728732e9724c6a71bc4c4e2c57086355d00127f7f09`.
Raw receipts and source archive are retained at
`/home/jasonc/b12x-residency-review-20260918`; `completion.json` identifies the
committed source, commands, file hashes and receipt hashes. Historical timing
and qualification sections above continue to describe their original sources.
No preceding timing measurement is attributed to this package.

### Review dispositions

| Feedback | Result and rationale |
| --- | --- |
| Bootstrap fairness | Implemented observation-free fractional coverage. Identical layers differ by at most one hot row, subject to bounds/budget. Synthetic counts are absent from bootstrap plans. Real counts still drive unequal learned placement. |
| Hard-limit activation | Implemented default `activation="converged"` and deliberate `best_available` acceptance. Bounded experiments remain saveable. Latest and converged indexes are separate, so an experiment cannot replace the accepted cache entry. Pins do not bypass acceptance. |
| Joint HBM/Grace constraints | Implemented a bounded exact byte-feasibility fallback after greedy density placement. It preserves bounds, identity and deterministic ties; it does not claim optimal varying-size selection score. Oversized searches report unknown feasibility explicitly. The allocator identity is `selection_density_joint_v2`. |
| Master reconciliation | Rebased onto `0f3a8cbf`; source-bound host, full compiler and portable GPU validation repeated. No master changes were discarded. |
| Companion integration | Reviewed retained SM103 source `f6c6ac72c3` and maintained preparation source `ef1aeaf080`. The maintained source already uses PreparationSession; CPU checkpoint ownership, SM103 component admission and lifecycle wiring remain missing. The [integration audit](expert-residency-integration.md) gives concrete integration points. No CLI or companion source change is claimed. |
| Profiler overhead/fusion | Separate optional counters retained. Off has no observer; sampled monitoring still launches. An optional prepared fused variant remains possible without changing the unprofiled router. Physical full-MoE evidence is required before implementation. |
| Serial partition | Retained deterministic scan and original route-index semantics. Portable growth motivates measuring this stage on B300; no speculative replacement. |
| Shared scratch | Ownership design documented for lanes, streams, graphs, outputs, speculative branches and TP. Private workspaces remain fully admitted; no hypothetical saving is spent. |
| Physical B300 priority | No B300 available. SM103 execution, Grace-backed TMA and full-model quality remain unqualified. |
| Baseline/stage comparisons | Existing all-HBM vs hierarchical stages retained. Repetitions moved inside timing graphs to remove Python enqueue gaps. FlashInfer equivalence is a deferred physical comparison, not an implemented harness feature. |
| Shared FC1 quantization | Deferred until physical quantization/stage evidence. Source-native recipe and route-major numerical boundaries unchanged. |
| Sparse cold launches | Compact-count body guards retained. Grid/conditional/persistent scheduling and overlap require physical evidence. |
| Monitor | Remains opt-in, static and advisory; no throughput forecast. Completed worker polls return the saved result without further counter reads. |
| Unique-expert scoring | Selection frequency retained. Rich-trace uniqueness and versioned statistics remain the research path. |
| Readiness/CI | Canonical docs reconciled with the master base and policy source. GitHub reports zero statuses/check runs on the reviewed `2de9d31c` source. PR CI remains a separate infrastructure gate; no workflow was added to the backend change. |

### Changed files

| Files | Change |
| --- | --- |
| `b12x/moe/fused_moe/_residency_allocation.py` | Balanced bootstrap, density ranking and exact bounded joint-byte feasibility repair |
| `b12x/moe/fused_moe/automatic.py` | Activation acceptance, converged/latest cache publication, allocator identity and observation-free bootstrap |
| `b12x/integration/vllm/expert_residency.py` | Terminal poll idempotence and completed-measurement reset rejection |
| `benchmarks/moe/expert_residency.py` | Device-batched stage/full-operator timing graphs |
| `tests/moe/test_automatic_residency.py` | Bootstrap, hard-limit/pin/cache semantics, real-slab adversary, exhaustive small-budget oracle and search-bound tests |
| `tests/moe/test_routing_profile_gpu.py` | Verify seven device repetitions per replay and three timed samples produce exactly 21 observations |
| `docs/expert-residency-integration.md` | Companion source audit, loader/control-plane sequence, workspace ownership and physical optimization gates |
| `docs/expert-residency-automatic.md`, `docs/expert-residency.md` | Present-state policy, cache, budgeting and qualification contracts |
| `docs/sm103-readiness-report.md`, `docs/sm103-change-summary.md`, `docs/expert-residency-ledger.md` | Master/source identity, validation scope, feature/fix map and retained failures |

### Fresh validation

Host acceptance command:

```bash
python -m pytest tests/moe/test_automatic_residency.py \
  tests/moe/test_expert_residency.py tests/moe/test_sm103_residency.py \
  tests/moe/test_fused_moe_variant_selection.py tests/preparation tests/architecture \
  tests/attention/test_qsa_contract.py tests/attention/test_qsa_program_keys.py \
  tests/attention/test_qsa_stable_selection.py \
  tests/attention/test_compressed_sparse_mla_v41.py -q
```

Result: **1,001 passed, 225 skipped**, 41.90 seconds (`host-acceptance.log`). This
includes 63 automatic-residency host tests. One test compares both learned and
bootstrap allocation against exhaustive feasible counts for 400 deterministic
small problems with differing sizes, bounds and tier budgets. GPU-only cases are
skipped on the host and are not counted as passes.

The same package cross-compiles **84 declarations / 241 distinct programs**, with
235 required/native CuTe exports and CUDA uninitialized (`prepared/manifest.json`,
`prepared/cases.jsonl`). The counter compiler emits **four entry points** for both
ID widths at sampling intervals 1 and 128 (`counters/manifest.json`). Production
residency emits **12 entry points** for H=5120, I=2304, E=384, HBM=295, capacity=128,
top-k=6 and clamp=10 (`production/manifest.json`). The reproduction commands are
in the [automatic guide](expert-residency-automatic.md) and
[physical runbook](expert-residency.md#qualification-commands).

Resource receipts retain 142/140 registers for FC1/FC2, 1,024 static and 51,328
dynamic shared-memory bytes, and zero stack/local bytes. Every-call counters use
14/13 registers for int32/int64 IDs; sampling every 128 calls uses 14 for both
widths. All use 1,024 static plus 4 dynamic shared-memory bytes and no stack/local
bytes. No compiler result establishes measured occupancy
or performance.

Portable validation runs on RTX PRO 4000 Blackwell SM120, driver `580.173.02`,
Torch `2.13.0`, Torch CUDA build `13.3`, CUTLASS DSL `4.6.2`, Triton
`3.7.1+gitf797708c.nv26.7` and isolated CUDA bindings `13.0.3`:

| Command following `python -m pytest` | Result | Receipt and physical GPU |
| --- | --- | --- |
| `tests/moe/test_routing_profile_gpu.py tests/moe/test_sm103_residency.py -q`, under `compute-sanitizer --tool memcheck --error-exitcode 99` | **6 passed, 6 skipped, zero errors**, 356.45 s | `gpu-memcheck.log`; `GPU-cc109c01-9756-d0db-21ea-f1825d3f963f` |
| Same tests under `compute-sanitizer --tool synccheck --error-exitcode 99` | **6 passed, 6 skipped, zero errors**, 120.26 s | `gpu-synccheck.log`; `GPU-47363510-b87a-13a5-4824-2542e97df76c` |
| `tests/moe/test_residency_kernels.py -q` | **8 passed**, three compiler optimization warnings, 6.07 s | `gpu-arithmetic.log`; `GPU-47363510-b87a-13a5-4824-2542e97df76c` |
| `tests/attention/test_qsa_stable_selection.py tests/attention/test_compressed_sparse_mla_v41.py -q` | **49 passed**, eleven compiler optimization warnings, 46.00 s | `gpu-master-regression.log`; `GPU-47363510-b87a-13a5-4824-2542e97df76c` |

The six skipped cases require physical SM103. The passing counter tests validate
replay, changing routes, both ID widths, overflow, TP ownership, allocator
stability, concurrent producers and restart/profile reuse. The arithmetic suite
validates portable quantization/partition/finalization and mapped-host access;
it executes no native SM103 expert MMA. The timing-graph test checks repetition
accounting, not B300 latency. No performance benchmark was repeated or claimed.

### Retained failures and limits

- `host-focused-first.log`: **61 passed, one failure**. A fixture reduced expert
  count with `dataclasses.replace` but retained the previous `maximum_hot=4`;
  admission correctly rejected the inconsistent geometry. The fixture now
  declares matching bounds. No implementation guard was relaxed.
- An inspector smoke-test locator used an untruncated pytest-directory name and
  found no artifact (`StopIteration`). A dedicated converged fixture supplies
  the inspector input; `inspector.log` records its successful validation. The
  collection failure is recorded in `audit-notes.txt`.
- Runtime identity and the historical profiler JSON identify the Torch CUDA
  build as 13.3, distinct from cuda-bindings 13.0.3. The preceding ledger
  section's 13.0 Torch-build label is corrected from that raw receipt.
- A general selection-score knapsack solver was rejected as unnecessary for
  tier admission. The exact fallback solves byte feasibility only, with explicit
  resource bounds. Uniform-cost placement keeps its ordinary greedy path.
- A companion flag layered after GPU weight loading was rejected because it
  cannot load a checkpoint larger than HBM. The source ownership prerequisite
  is documented before CLI/control-loop wiring.

Next backend work is physical all-HBM/all-Grace/mixed SM103 qualification and an
equivalent FlashInfer baseline, followed by stage-directed optimization. The
portable launch/partition evidence and exact scratch calculation justify
investigation, not changing kernels or spending arena savings in advance.

## Quiescent slot exchange evidence

This evidence records fixed-address replacement atop branch revision
`2a45657f949ef6f06587bd403c6faf1c2f237239`, already based on master
`0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`. Fetching both refs on September 18,
2026 found no additional master or working-branch changes. Master was not edited.
The tested package SHA256 is
`dbc81a149394350c3b3db79aa0172b9b397211230c51e0df5c93e9cf04846034`.

Raw evidence is retained outside the repository at
`/home/jasonc/b12x-slot-exchange-evidence-20260918/`. `source-manifest.json`
binds individual implementation/test/tooling files and the package hash;
`source-final.tar.gz` preserves the tested export. Compiler manifests record the
starting commit, dirty source paths and the authoritative package hash. The GPU
export is `/home/jasonc/b12x-slot-exchange-20260918` on ripper;
`gpu-identity.log` independently confirms the same package hash there.

### Result and rationale

The existing map is sufficient for fixed-address replacement at a quiescent
boundary. The implementation retains canonical router IDs and tensor-major
slabs, journals all four fields of both members of every pair, synchronizes
payload completion before map publication, and restores both payloads and map on
failure. A failed rollback poisons state and requires the engine to keep raw
graph submissions stopped. Snapshot identity includes the preparation instance
as well as generation. Static plans allocate no journal and retain the same
programs and captured operation sequence.

The [slot contract](expert-residency-slots.md) documents engine ownership,
nonresumable failure, TP coordination, exact byte costs and deferred native
qualification. Placement policy, automatic-profile activation, routing counters,
quantization and ordered-FMA semantics are unchanged. This is a mechanism for
explicit paused epochs, not a running adaptive cache or a vLLM serving feature.

### Files changed

| File | Purpose |
| --- | --- |
| `b12x/moe/fused_moe/residency.py` | Typed update capacity and explicit host-journal admission |
| `b12x/moe/fused_moe/_residency_updates.py` | Generation snapshots, CUDA transfers, quiescent batch transaction, rollback and poison state |
| `b12x/moe/fused_moe/_residency_storage.py` | Aligned journal/map memory formula |
| `b12x/moe/fused_moe/_residency_tuning.py` | Query schema 2 and declared pair capacity |
| `b12x/moe/fused_moe/_residency_preparation.py` | Journal materialization/ownership and eager health guards |
| `b12x/moe/fused_moe/api.py` | Optional declaration argument and exchange API exports |
| `b12x/moe/fused_moe/__init__.py` | Lazy public API metadata |
| `scripts/_sm103_preparation_corpus.py` | Update-enabled declaration with unchanged kernel corpus |
| `tests/moe/test_residency_updates.py` | Host fault injection, stale requests, capacity/accounting and lifecycle guards |
| `tests/moe/test_residency_updates_gpu.py` | Same-graph byte probes, stream drain, rollback and allocator/pointer invariants |
| `tests/moe/test_expert_residency.py` | Shared declaration fixture supports explicit updates |
| `tests/moe/test_sm103_residency.py` | Physical-only same-graph native parity against freshly prepared placement |
| `docs/expert-residency-slots.md` | Full SM103 slot contract, examples and deferred commands |
| `docs/expert-residency.md` | Storage/admission overview and exchange entry point |
| `docs/expert-residency-automatic.md` | Static activation remains independent of exchange |
| `docs/expert-residency-integration.md` | Engine/rank pause ownership |
| `docs/sm103-readiness-report.md` | Source-bound qualification status and totals |
| `docs/sm103-change-summary.md` | Feature and failure-handling map |
| `docs/expert-residency-ledger.md` | This evidence and rejected alternatives |

### Host and compiler gates

```bash
python -m pytest tests/moe/test_residency_updates.py \
  tests/moe/test_automatic_residency.py tests/moe/test_expert_residency.py \
  tests/moe/test_sm103_residency.py tests/moe/test_fused_moe_variant_selection.py \
  tests/preparation tests/architecture -q
python scripts/compile_sm103_prepared.py --output-dir RECEIPTS/prepared --workers 2
python scripts/compile_sm103.py --component residency \
  --hidden 5120 --intermediate 2304 --experts 384 --hot-experts 295 \
  --capacity 128 --top-k 6 --swiglu-limit 10 \
  --nvdisasm /path/to/nvdisasm --cuobjdump /path/to/cuobjdump \
  --output-dir RECEIPTS/production
```

Host result: **1,021 passed, 66 skipped**, 30.77 seconds (`host-final.log`). The
suite includes completion failures at every synchronization boundary, copy
failures after destination writes, rollback failure, externally corrupted maps,
stale generation/preparation, duplicate/invalid pairs, declaration purity and
host-budget rejection. The previous review's additional QSA/MLA suite was not
included in this invocation; its results remain tied to its own source.

Offline compilation: **85 declarations, 241 distinct programs, 235 native CuTe
exports**, no failures and CUDA uninitialized (`prepared/manifest.json` and
`prepared/cases.jsonl`). The update-enabled declaration reuses the same compiled
program identities as its static equivalent. The package was unchanged during
compilation. Production geometry compiles **12 entries**. FC1/FC2 retain 142/140
registers, 1,024 static plus 51,328 dynamic SMEM bytes and zero stack/local memory.
No new copy kernel is compiled. CUDA runtime transfers implement the transaction.

Local compiler packages: Torch 2.14.0, CUTLASS DSL 4.6.2 and Triton 3.8.0. Exact
commands, disassembler versions, object hashes, PTX, SASS and resource reports
are retained in `production/`. These results establish compilation and resource
usage, not measured occupancy, B300 legality or performance.

### Portable graph and sanitizer gates

The export runs in container image
`sha256:955e088a85b5378b00275842bc839eea8cb04ca0782ed79eaa3a967d11fd22e5`,
with `PYTHONPATH=/workspace:/cuda-bindings`. It uses Torch 2.13.0, Torch CUDA build
13.3, CUTLASS DSL 4.6.2, Triton 3.7.1 and isolated cuda-bindings 13.0.3. Both
physical GPUs are RTX PRO 4000 Blackwell SM120 in default compute mode, driver
580.173.02. No service was stopped and no architecture capability was spoofed.

```bash
python -m pytest tests/moe/test_residency_updates_gpu.py \
  tests/moe/test_residency_kernels.py tests/moe/test_routing_profile_gpu.py \
  tests/moe/test_sm103_residency.py -q
compute-sanitizer --tool memcheck --error-exitcode 91 \
  python -m pytest tests/moe/test_residency_updates_gpu.py -q
compute-sanitizer --tool synccheck --error-exitcode 91 \
  python -m pytest tests/moe/test_residency_updates_gpu.py -q
```

`gpu-final.log`: **20 passed, 9 skipped**, four existing static-loop compiler
warnings, 12.25 seconds on `GPU-cc109c01-9756-d0db-21ea-f1825d3f963f`. The nine
skips require physical SM103. Six passing cases use the actual partitioner and a
test-only CuTe byte reader, inspect every payload field, exercise HBM/mapped-host
copies and reuse the same graph after repeated slot exchange. They check both ID
widths, invalid/duplicate routes, completed side-stream consumers, injected
publication rollback, stable addresses, frozen resolution and unchanged Torch
allocator allocation/free counters during replay. Eight other tests cover
portable residency arithmetic; six cover prepared routing counters.

`synccheck.log`: **6 passed, zero errors**, 9.11 seconds on
`GPU-cc109c01-9756-d0db-21ea-f1825d3f963f`. `memcheck.log`: **6 passed, zero errors**,
312.91 seconds on `GPU-47363510-b87a-13a5-4824-2542e97df76c`.

No latency benchmark or adaptive-benefit measurement was performed. Test run
durations are validation wall times. Physical SM103 all-HBM/all-Grace/mixed
operator parity, TMA legality, repeated same-graph exchange and sanitizers remain
explicit deferred gates in the slot guide.

### Retained failures and rejected alternatives

- `host-first.log`: **94 passed, one failure**. The capacity type was imported
  into the API but omitted from lazy `META.entry_points`. Registration was fixed;
  `host-second.log` records **95 passed** before adding completion-failure cases.
- `gpu-first.log`: **5 passed, one failure** in a replay allocator check. Delayed
  Python collection of an earlier test's graph/buffers changed allocator counts
  inside the measurement window. Explicit collection before the window removes
  unrelated frees; no allocator assertion was relaxed. `gpu-second.log` records
  **6 passed**. Final tests additionally verify side-stream draining.
- A documentation patch targeted a nonmatching heading and applied no changes.
  It was corrected using the file's actual heading; source and test receipts
  were unaffected.
- Router-row swapping was rejected: the existing map already preserves physical
  addresses while keeping router, profile and checkpoint identity canonical.
- Full canonical Grace backing was deferred: the qualification example would
  add 206.612 GiB across 40 layers for hot-expert copies. The explicitly budgeted
  journal preserves the existing exclusive-tier capacity contract.
- Map-only rollback was rejected because fixed slots may already contain
  replacement payloads. Both sides are journaled before the first overwrite.
- Atomic map encoding, spare-slot retirement and concurrent replacement were
  deferred. A proven engine pause removes torn-read exposure without adding
  device-side protocol work. The pause requirement cannot be inferred from
  pointer stability or a device synchronization alone.

GitHub exposes zero status contexts and zero check runs for starting source
`2a45657f` (`github-starting-status.json`, `github-starting-checks.json`). Local
source-bound validation does not replace an independent PR CI gate.

The next evidence is physical native static correctness and same-graph exchange,
then full pause/copy/map timings and a controlled changing-workload comparison.
Only those measurements can justify spare backing, event overlap or an adaptive
policy. Existing measured tiny-M counter/partition cost and calculated scratch
savings remain separate optimization questions.

## Grace-served cache policy evidence

This evidence records a host-only recent-frequency controller atop branch
`94639562d1b9ff54545c0a69aadad0bd0c9ff7d3`, based on master
`0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`. Live fetch on September 18, 2026
found both refs unchanged. The tested package SHA256 is
`12217a30f50be813bbbf7c5b95253b1af12fde5d29ae41d7f101c5700d9f587a`.
Master was not edited.

Raw evidence lives at `/home/jasonc/b12x-cache-policy-evidence-20260918/`.
`source-manifest.json` hashes the implementation, test and tooling files;
`source-final.tar.gz` preserves the tested export. The remote worktree is
`/home/jasonc/b12x-cache-policy-20260918` on ripper. `gpu-identity.log` verifies
its package hash independently. Compiler manifests retain the starting commit,
working-tree edits and package hash. Test durations below are validation wall
times, not cache-performance measurements.

### Implementation and boundaries

Existing canonical counters are sufficient to classify misses while one
placement generation remains fixed. The controller differences cumulative
snapshots at an engine pause, ranks cold candidates and hot victims by recent
counts, and proposes disjoint pairs. Explicit count, score-margin, hot-residency
age and batch limits constrain replacement. Empty polls do not age the guard.
An acknowledgement accepts only unchanged placement or the exact committed pair
permutation; rollback/decline does not claim promotion. Post-promotion hit counts
exclude the observations that motivated promotion.

No routing kernel, slot-copy mechanism, slab layout, preparation contract,
numerical boundary or static-profile rule changed. Normal static serving does
not instantiate this controller or add an observer. Policy decisions remain
host-side; engine pause, snapshot, exchange, TP coordination and resume remain
explicit. Counts cannot establish per-iteration unique touches or time to first
reuse. Sampled observations are not extrapolated into total traffic or throughput.

The [cache guide](expert-residency-cache.md) documents API composition, diagnostics,
backing-store tradeoffs, exact native test commands and the required three-arm
serving experiment. Native policy-loop tests exist but require physical B300.
No serving-engine wiring or adaptive throughput harness is claimed.

### Files changed

| File | Purpose |
| --- | --- |
| `b12x/moe/fused_moe/residency_cache.py` | Typed explicit policy, generation-bound counter windows, decisions, acknowledgements and diagnostics |
| `b12x/moe/fused_moe/api.py` | Public policy exports |
| `b12x/moe/fused_moe/__init__.py` | Lazy API metadata for those exports |
| `tests/moe/test_residency_cache.py` | Host decisions, safeguards, lifecycle rejection, rank/phase/sampling and rollback acknowledgement |
| `tests/moe/test_residency_cache_gpu.py` | Prepared-counter policy loop through unchanged graphs; portable byte oracle and physical-only native all-HBM parity |
| `docs/expert-residency-cache.md` | SM103 policy contract, examples, diagnostics and physical experiment |
| `docs/expert-residency.md` | Overview and capability boundary |
| `docs/expert-residency-slots.md` | Exchange remains independent; link to the separate policy |
| `docs/expert-residency-automatic.md` | Automatic static activation remains separate |
| `docs/expert-residency-integration.md` | Engine-owned observation/exchange/acknowledgement boundary |
| `docs/sm103-readiness-report.md` | Fresh source-bound tests and focused compilation, with historical census identity |
| `docs/sm103-change-summary.md` | Feature/fix map and qualification limits |
| `docs/expert-residency-ledger.md` | This evidence and rejected alternatives |

### Host and compiler validation

```bash
python -m pytest tests/moe/test_residency_cache.py \
  tests/moe/test_residency_updates.py tests/moe/test_automatic_residency.py \
  tests/moe/test_expert_residency.py tests/moe/test_sm103_residency.py \
  tests/moe/test_fused_moe_variant_selection.py tests/preparation tests/architecture -q
python scripts/compile_sm103_prepared.py --output-dir RECEIPTS/prepared \
  --case moe:residency --case moe:residency_updates \
  --case moe:routing_profile --workers 2
```

`host-final.log`: **1,037 passed, 66 skipped**, 30.81 seconds, including 16 policy
cases. Focused intermediate receipts retain 14 passes (`host-first.log`) and 40
policy/exchange passes (`host-second.log`) before additional independence and
acknowledgement cases. No pytest failure occurred in this change.

`prepared/manifest.json` and `prepared/cases.jsonl`: **3 declarations, 14 distinct
programs, 14 native CuTe exports**, zero failures, CUDA uninitialized, source
unchanged. Local packages are Torch 2.14.0, CUTLASS DSL 4.6.2 and Triton 3.8.0.
The full inventory remains 85 declarations/241 programs, whose preceding full
receipt belongs to `94639562` and package `dbc81a14…`. Host-only policy code adds
no GPU program; a full production resource census was not repeated or attributed
to the policy source.

### Portable and sanitizer validation

The remote image is
`sha256:955e088a85b5378b00275842bc839eea8cb04ca0782ed79eaa3a967d11fd22e5`.
Tests use Torch 2.13.0, Torch CUDA build 13.3, CUTLASS DSL 4.6.2, Triton 3.7.1 and
isolated cuda-bindings 13.0.3 through `PYTHONPATH=/workspace:/cuda-bindings`.
Both GPUs are physical RTX PRO 4000 Blackwell SM120, driver 580.173.02, default
compute mode. No service was stopped or SM103 capability spoofed.

```bash
python -m pytest tests/moe/test_residency_cache_gpu.py \
  tests/moe/test_residency_updates_gpu.py tests/moe/test_residency_kernels.py \
  tests/moe/test_routing_profile_gpu.py tests/moe/test_sm103_residency.py -q
compute-sanitizer --tool memcheck --error-exitcode 91 \
  python -m pytest tests/moe/test_residency_cache_gpu.py -q
compute-sanitizer --tool synccheck --error-exitcode 91 \
  python -m pytest tests/moe/test_residency_cache_gpu.py -q
```

| Receipt | Result | Physical GPU |
| --- | --- | --- |
| `gpu-first.log` | 2 passed, 2 SM103 skips, 6.38 s | `GPU-cc109c01-9756-d0db-21ea-f1825d3f963f` |
| `gpu-final.log` | **22 passed, 11 SM103 skips**, 13.57 s | `GPU-cc109c01-9756-d0db-21ea-f1825d3f963f` |
| `memcheck.log` | **2 passed, 2 SM103 skips, zero errors**, 125.19 s | `GPU-47363510-b87a-13a5-4824-2542e97df76c` |
| `synccheck.log` | **2 passed, 2 SM103 skips, zero errors**, 48.61 s | `GPU-cc109c01-9756-d0db-21ea-f1825d3f963f` |

Compiler warnings concern the existing 128-iteration static scale-padding loop.
The passing policy cases test both ID widths, duplicate and invalid routes,
seven traffic windows, three promotions, observed post-promotion HBM reuse,
canonical byte outputs, stable pointers, frozen resolution and unchanged Torch
allocator counters. They combine the actual prepared counter program, production
partitioner and existing slot transaction with a test-only byte reader. Native
SM103 expert arithmetic and Grace-backed TMA are not exercised by that reader.

### Rejected scope and retained failures

- Three documentation patches used incomplete line contexts and applied no
  changes. The full existing line was used on retry. Source/test artifacts were
  unaffected; no guard or validation was weakened.
- A miss-specific GPU counter/bitset was unnecessary for windows with constant
  placement. Counter deltas plus the existing host map supply the observation
  without changing production kernels. Per-step touch statistics remain a rich
  trace research question.
- Immediate promotion on every cold access was rejected as the default. All
  policy thresholds are explicit, positive score gain is required and admission
  remains bounded. No threshold is advertised as a measured production choice.
- Asynchronous promotion/spare-slot retirement was deferred. An HBM spare alone
  does not protect the reused Grace row in the exclusive-tier model. Both tiers'
  readers and publication need a complete lifetime protocol.
- Canonical Grace backing was not added. Its additional hot-expert copies would
  materially change memory admission; eviction-copy savings require measurement.
- A CPU synthetic hit-rate improvement was not reported as performance evidence.
  B300 native correctness precedes full pause/copy measurements; a serving
  experiment additionally requires the engine's pause and rank callbacks.

Next evidence, in order: native static and same-graph policy correctness on B300;
full-MoE observer overhead and actual exchange pauses/C2C traffic; then balanced
static, observed-static and adaptive lanes on fixed and shifting traffic. Only
those measurements can justify a cost model, spare backing or concurrent work.

## Shared residency extraction evidence

This extraction starts from `181e234b5320eae67f7e1096672129d01c14ff09`, with
master base `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`. A fresh fetch on
September 18, 2026 found no missing master commits; master was not edited.
The tested package SHA256 is
`2400c738ec87ea2ac71e21a426b1de4f315c66e6e05a642578c1828f677e60d1`.
The local checkout and remote exported package hashes match.

### Result and boundaries

The shared `b12x.moe.residency` namespace owns canonical placement, cumulative
observation contracts, generation snapshots and recent-frequency host policy.
Its import and an actual decision/acknowledgement succeed with Torch, CUDA,
CUTLASS, Triton, preparation and fused-MoE imports explicitly blocked. A five-expert
host test exercises arbitrary payload accounting without an MXFP4 geometry.

The SM103 fused-MoE constructor remains a thin adapter with unchanged arguments.
It validates native layer/counter metadata and supplies successful payload/map
copy accounting. Shared types retain aliases at existing public paths. Two
fixtures generated from the starting source verify unchanged schema-1/schema-2
profile payloads and hashes; strict native recipe and integrity rejection remain.
Preparation, private storage, counters, kernels, transactions and the static
default are unchanged. Backend capability descriptors do not qualify hardware.

The [subsystem guide](expert-residency-subsystem.md) specifies extension and
ownership contracts. Native automatic profile derivation and HBM/Grace admission
remain backend-specific: their formulas describe actual native storage and
private scratch. Merely renaming these limits would not support other physical
memory topologies or prevent shared-pool double counting.

### Files changed

| File | Purpose |
| --- | --- |
| `b12x/moe/residency/__init__.py` | Shared public host namespace |
| `b12x/moe/residency/contracts.py` | Placement, observations, generations, exchange guarantees and copy accounting |
| `b12x/moe/residency/policy.py` | Extracted recent-frequency policy with backend-supplied accounting |
| `b12x/moe/__init__.py` | Lazy shared namespace without GPU-op registration |
| `b12x/moe/fused_moe/residency_cache.py` | Compatible SM103 policy adapter |
| `b12x/moe/fused_moe/residency.py` | Shared placement view and update-capacity alias; unchanged profile fields |
| `b12x/moe/fused_moe/automatic.py` | Shared observation type aliases |
| `b12x/moe/fused_moe/_residency_updates.py` | Shared snapshot/error aliases; transaction unchanged |
| `b12x/moe/fused_moe/_routing_profile_tuning.py` | Shared phase vocabulary |
| `b12x/moe/fused_moe/routing_profile.py` | Shared observation types |
| `b12x/moe/fused_moe/api.py` | Explicit exports for existing update/cache entry points |
| `tests/moe/test_shared_residency.py` | Backend independence, capability rejection, accounting and compatibility tests |
| `tests/moe/fixtures/residency-schema1.json` | Pre-extraction layer artifact |
| `tests/moe/fixtures/residency-schema2.json` | Pre-extraction automatic artifact with fixed timestamp |
| `docs/expert-residency-subsystem.md` | Shared architecture, backend requirements and compatibility |
| `docs/expert-residency-cache.md` | Shared policy/native adapter distinction |
| `docs/expert-residency.md` | Shared contract entry point |
| `docs/sm103-readiness-report.md` | Source-bound gates and known registry failure |
| `docs/sm103-change-summary.md` | Extraction behavior and validation |
| `docs/expert-residency-ledger.md` | Evidence and rejected scope |

### Validation commands and receipts

Raw receipts are outside the repository at
`/home/jasonc/b12x-residency-extraction-evidence-20260918/`.
`source-final.json`, `source-manifest.json`, `source-final.tar.gz`, compiler manifests and
`portable-source-toolchain-final.json` identify the tested sources. The remote export
is `/home/jasonc/b12x-residency-extraction-20260918` on ripper. Final source
manifests include documentation edits made after the package was frozen.

```bash
TMPDIR=/home/jasonc/b12x-residency-extraction-evidence-20260918/tmp \
  .venv/bin/python -m pytest tests/moe/test_shared_residency.py \
  tests/moe/test_residency_cache.py tests/moe/test_residency_updates.py \
  tests/moe/test_automatic_residency.py tests/moe/test_expert_residency.py \
  tests/moe/test_sm103_residency.py tests/moe/test_fused_moe_variant_selection.py \
  tests/preparation tests/architecture -q
.venv/bin/python scripts/compile_sm103_prepared.py --output-dir RECEIPTS/prepared-final \
  --case moe:residency --case moe:residency_updates \
  --case moe:routing_profile --workers 2
python -m pytest tests/moe/test_residency_cache_gpu.py \
  tests/moe/test_residency_updates_gpu.py tests/moe/test_routing_profile_gpu.py \
  tests/moe/test_residency_kernels.py tests/moe/test_sm103_residency.py -q
compute-sanitizer --tool memcheck --error-exitcode 91 \
  python -m pytest tests/moe/test_residency_cache_gpu.py -q
compute-sanitizer --tool synccheck --error-exitcode 91 \
  python -m pytest tests/moe/test_residency_cache_gpu.py -q
```

| Receipt | Result and scope |
| --- | --- |
| `host-accepted-final.log` | **1,052 passed, 66 skipped**, including 15 shared-subsystem tests; 31.99 s |
| `prepared-final/manifest.json` | **3 declarations, 14 distinct programs and 14 native SM103 exports**, zero failures, CUDA uninitialized, source unchanged |
| `portable-final.log` | **22 passed, 11 SM103 skips**, 13.35 s; prepared counters, partitioner, mapped-host byte probes, slot exchanges and unchanged graph replay |
| `memcheck-final.log` | **2 passed, 2 SM103 skips, zero errors**, 123.88 s |
| `synccheck-final.log` | **2 passed, 2 SM103 skips, zero errors**, 48.06 s |
| `host-initial.log`, `registry-baseline.log` | Separate registry gate has **5 failures** from MXFP6 metadata/registration, reproducing at untouched starting HEAD; baseline has 4 registry passes |

Local compilation uses Torch 2.14.0, CUTLASS DSL 4.6.2 and Triton 3.8.0. The
remote image identity is
`sha256:955e088a85b5378b00275842bc839eea8cb04ca0782ed79eaa3a967d11fd22e5`:
Torch 2.13.0, Torch CUDA 13.3, CUTLASS DSL 4.6.2,
Triton `3.7.1+gitf797708c.nv26.7`, cuda-bindings 13.0.3. Remote commands mount
that isolated binding directory at `/cuda-bindings` and use
`PYTHONPATH=/workspace:/cuda-bindings`.

Both physical devices are RTX PRO 4000 Blackwell SM120 in default compute mode,
driver 580.173.02. Portable/synccheck use
`GPU-cc109c01-9756-d0db-21ea-f1825d3f963f`; memcheck uses
`GPU-47363510-b87a-13a5-4824-2542e97df76c`. No service was stopped. Durations
above are test wall times, not performance measurements. Existing compiler
warnings concern the 128-iteration scale-padding loop.

The full 85-declaration/241-program/235-native-export resource census remains
historical evidence for `94639562`. No kernel, tuning query or candidate set was
added; the extraction rechecks the three composing declarations. No resource or
performance improvement is claimed. Physical B300, Grace-backed TMA, native
operator parity and end-to-end serving remain deferred; exact physical commands
remain in the cache guide and qualification runbook.

### Retained failures and rejected scope

- A final whitespace check removed two trailing blank lines from the shared
  contract module. Host, focused compiler, portable and sanitizer gates were
  repeated on the final package hash. Earlier passing receipts remain under
  their unsuffixed filenames and bind package `ea200747…`; they are not
  substituted for final-source evidence.

- Initial fixture generation hit the host `/tmp` disk quota while writing an
  atomic profile. The external evidence directory supplied `TMPDIR` for the
  successful retry; `fixture-initial-failure.txt` records the failure.
- Two early test runs used an incorrect error-message assertion for corruption.
  A workload edit was rejected at identity validation, and a corrupted hash was
  rejected with `integrity mismatch`. The final test changes the hash and checks
  the existing integrity error; validation was not weakened. `shared-initial.log`
  and `host-final.log` preserve both failed assertions.
- The repository-wide registry tests fail at untouched `181e234b` because
  `quantization.mxfp6` is registered without the expected public `api.py`/`META`
  facade. Extraction does not alter that subsystem. The independent shared import
  test passes; registry repair remains a separate integration gate.
- A generic allocator, speculative N-tier model, staged-miss backend, concurrent
  exchange and automatic policy selection were not introduced. The implemented
  backend is exclusive HBM/Grace, and another physical topology needs real
  admission and execution contracts.
- Copy accounting remains backend-supplied; no transfer-cost model or predicted
  throughput is inferred from recent counts. Router rows remain immutable.
- Moving the native geometry-aware profile optimizer merely for namespace
  symmetry was rejected. Its schema and hardware validation remain intact.
- `github-start-status.json` and `github-start-checks.json` record zero statuses
  and zero check runs for `181e234b`. Local receipts do not replace independent CI.

The next evidence priorities remain physical SM103 correctness and Grace TMA
legality, complete-MoE profiling/pause measurements, then engine integration and
workload-shift experiments. No additional backend optimization is justified by
this host-code extraction alone.

## SM120 mapped-host cache proof of concept

Status: **research-only native operator experiment**, validated on two physical
RTX PRO 4000 Blackwell cards on `ripper` on 2026-09-18. The implementation is
[`benchmarks/moe/sm120_residency_poc.py`](../benchmarks/moe/sm120_residency_poc.py);
the [SM120 guide](expert-residency-sm120-poc.md) specifies its scope and commands.
The working source is based on `7569c94a491ec319fabebfd2a41c578ce2d8cab0`, already
containing master `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`.

### Changes and contracts

- Explicit source-native ModelOpt A16 preparation uses the existing native
  W4A16 representation. The default uniform-A16 MMA packing is unchanged.
  An existing planner rejection prevented that explicit choice and was removed;
  the weight-plan test now checks both default and explicit behavior.
- A benchmark composes two ordinary prepared native operators, exact-size
  mapped PCIe host storage, six-field rollback journals, prepared counters and
  the shared host cache controller. It asserts retained source pointers, freezes
  resolution, exchanges slots at a serialized pause and replays captured graphs.
- A CuTe metadata helper derives tier maps and validates int64 IDs before
  narrowing. A CuTe finalizer sums already weighted BF16 route outputs in original
  order in FP32. It does not add the separately rounded tier outputs.
- No SM103 kernel, tuning candidate set, preparation registration, automatic
  placement algorithm or serving integration was added. The historical
  85-declaration/241-program SM103 census remains tied to its recorded source.

### Final source and results

Raw artifacts reside outside the repository at
`/home/jasonc/b12x-sm120-residency-evidence-20260918/`, with remote copies at
`ripper:/home/jasonc/b12x-sm120-residency-results-20260918/`.
The tested `source-final.tar.gz` has SHA256
`37edb8645b6994af775f25ae2f137fef6b25bbac17398d7fbea6e9beb796244c`.
`source-files-final.json` hashes package, experiment and focused test sources;
package SHA256 is
`3d8a44435651093068ee418c2e6d7893a19be305d527c5b53f0653e18baab9b7`.
Documentation evidence annotations were added after this export; implementation
and tests are identical to the exported files.

| Receipt | Result |
| --- | --- |
| `host-final.log` | **170 passed, 6 skipped**, 6.41 s: MoE preparation corpus, SM103 contracts/defaults, native weight planning, shared contracts, policy, transactions and experiment loader tests |
| `remote/gpu-final.log` | **16 passed, 2 deselected**, 16.98 s: focused experiment and NVFP4 preparation tests; the two large AUTO decode cases were deliberately excluded |
| `remote/gpu-final.log`, targeted memcheck | Ordered-reduction helper: **1 passed, 4 deselected, zero errors**, 6.75 s; this is not a complete native-MoE sanitizer result |
| `remote/checkpoint-final.json` and `.log` | **30 replay cases passed** at M=1,2,4 under capacity 4, including 18 policy windows, nine exchanges and 12 route-boundary cases |
| `lint-final.log` | New experiment/support/test modules pass Ruff; whitespace checks pass |

The full checkpoint-layer run uses E=512, H=2560, I=640, top-k=10 and an
experimental 256/256 split. Each tier slab is **707,790,848 bytes**; the rollback
allocation is **5,538,304 bytes**. The separate all-VRAM reference and private
operator scratch consume additional VRAM. Loaded checkpoint fields hash to
`05384d5b0bbe71843464786f15391673847f5eaa9fa08e2a2e8309ab80c6c90e`.
All graph-captured pointers remained stable and all 18 measured policy windows
had zero replay allocation/free events. Canonical router identity is unchanged.

Repeated-expert policy windows matched the all-VRAM native control bitwise.
Eleven of twelve route-boundary cases also matched bitwise. The remaining mixed
case had maximum absolute error **3.814697265625e-6**, relative L2
**0.0005540843121707439** and cosine **0.9999998211860657**. Diagnostic probes
localized the difference before reduction: FC1 activation relative L2
0.0001273119 and weighted FC2 relative L2 0.0002973982. The experimental finalizer
matched the independent original-order FP32 sum of those BF16 rows exactly.
Route-dependent native GEMM accumulation is consistent with this evidence;
bitwise equivalence of arbitrary split and unsplit GEMMs is not claimed.

GPU tests and targeted memcheck use
`GPU-47363510-b87a-13a5-4824-2542e97df76c`; the full checkpoint layer uses
`GPU-cc109c01-9756-d0db-21ea-f1825d3f963f`. Both are SM120, default compute mode,
driver 580.173.02. No service was stopped. The container image is
`sha256:955e088a85b5378b00275842bc839eea8cb04ca0782ed79eaa3a967d11fd22e5`,
with Torch 2.13.0, CUDA 13.3, CUTLASS DSL 4.6.2, Triton
`3.7.1+gitf797708c.nv26.7` and cuda-bindings 13.0.3. Commands mount the isolated
binding directory at `/cuda-bindings` with `PYTHONPATH=/workspace:/cuda-bindings`.

Final GPU commands inside that environment:

```bash
python -m pytest tests/moe/test_sm120_residency_poc.py \
  tests/moe/test_nvfp4_auto.py -k 'not auto_native_decode' -q
compute-sanitizer --tool memcheck --error-exitcode 99 python -m pytest \
  tests/moe/test_sm120_residency_poc.py -k ordered_sum -q
python -m benchmarks.moe.sm120_residency_poc \
  --checkpoint /models/Qwen3.8-Flash-Next-NVFP4 \
  --experts 512 --hot-experts 256 --live 1 2 4 \
  --source-revision 7569c94+export-37edb864 --receipt /results/checkpoint-final.json
```

### Retained failures and deferred work

- Checkpoint discovery initially assumed a safetensors index. The export has
  none; the implemented loader enumerates shard headers and checks missing and
  duplicate fields. Source tensors retain their original quantized bytes.
- `checkpoint-01.log` records the explicit-native-A16 planner rejection.
- `gpu-03.log` records a shell-quoting error in the test selector; `gpu-03b.log`
  contains the corrected command's result.
- `checkpoint-03.log` preserves the failed bitwise multi-expert comparison.
  `diagnose-03.log`, `diagnose-loop-03.log` and their scripts distinguish native
  activation/FC2 differences from exact ordered reduction. The acceptance gate
  records numerical errors rather than describing all comparisons as exact.
- `memcheck-03.log` preserves an incomplete full native-MoE memcheck attempt,
  stopped at the bounded experiment time budget. It produced no completed
  sanitizer verdict. Full native memcheck/synccheck remain deferred.
- `lint-04.log` retains the formatting and strict-zip findings before final
  cleanup; final source was exported and revalidated afterward.
- Automatic full-model budgeting, calibration/store wiring, TP, vLLM integration,
  asynchronous copies, spare slots and policy tuning are outside this proof of
  concept. No throughput benchmark, PCIe bandwidth claim or adaptive benefit is
  inferred from correctness tests or control-plane wall times.

Physical SM103 static correctness and Grace-backed TMA remain the first backend
qualification gates. This experiment shows that the shared policy and fixed-slot
transaction compose with a different native recipe and memory topology. Any
further SM120 work should first measure complete-operator PCIe miss cost and
exchange pause cost on shifting traffic before adding serving integration.

## SM120 residency spectrum evidence

Status: **recorded single-layer diagnostics**, collected 2026-09-19 UTC
(2026-09-18 America/New_York). The [spectrum guide and results](expert-residency-sm120-spectrum.md)
specify the method, numerical gates, selected tables, interpretation limits and
measured follow-up priorities. Production kernels, residency mechanisms and
policy defaults are unchanged.

The runner is `benchmarks/moe/sm120_residency_spectrum.py`; route-fixture tests
are in `tests/moe/test_sm120_residency_spectrum.py`. Source export
`source-01.tar.gz` is based on `f7d1c5329314d6e30393211b3f9b8643dac0eb8b`, SHA256
`5b174e0cc54e716ac217469cc53da9696e0aeb8c565628d63d88b374d63751dd`.
Each run's manifest hashes the package and all SM120 residency experiment
modules. Every manifest was checked against the final unchanged implementation.
Documentation annotations were added after measurement.

The complete evidence bundle is
`/home/jasonc/b12x-sm120-spectrum-evidence-20260918/spectrum-records.tar.gz`, SHA256
`de5f37cde469081f9fe5749d87749e73ae9a2cf9aa1e37b651869ff58f53dd67`.
Its contents include the frozen source export, exact launch commands, per-case
raw samples, source/device/toolchain manifests, all stdout logs, derived
`latency.csv`/`policy.csv`, a summary generator and receipt hashes. Original
remote receipts remain at
`ripper:/home/jasonc/b12x-sm120-spectrum-results-20260918/`.

| Receipt or directory | Result |
| --- | --- |
| `host-final.log` | **40 passed, 3 CUDA skips**, 2.81 s; routing fixtures, loader, shared contracts and cache policy |
| `remote/second-card.log` | **12 passed**, 16.30 s; fixture tests plus native graph exchange, independent reference and ordered-reduction tests on physical SM120 |
| `remote/main-01/` | **74 fixtures passed**: 38 latency cases and 36 policy cases |
| `remote/hot-128/`, `remote/hot-384/` | **12 fixtures passed each**, covering resident budgets below and above the half-resident control |
| `remote/topk-2/`, `remote/topk-6/` | **12 fixtures passed each**, preserving checkpoint H/I and source quantization |
| `remote/reuse-512/` | **10 fixtures passed**: longer reuse windows and latency controls |
| `remote/second-card/` | **9 latency fixtures passed** on the other physical RTX PRO 4000 |
| `remote/pilot-01/` | **17 preliminary fixtures passed**, retained separately from the seven recorded sweeps |
| `lint-final.log` | Runner and test module pass Ruff; whitespace checks pass |

The seven recorded sweeps total **141 fixtures**, including **202 cache-condition
latency comparisons** and **40 policy fixtures**. Each latency condition measures
all-VRAM, static and profiled operators plus five isolated stages. Policy
fixtures execute three static/adaptive pairs of eight epochs. Numerical,
counter-delta, frozen-program, pointer and replay-allocator gates pass. No failed
benchmark fixture was removed. The earlier proof-of-concept's incomplete native
sanitizer attempt remains incomplete; this suite does not replace it.

All runs load the same 512-expert layer from
`/models/Qwen3.8-Flash-Next-NVFP4`: H=2560, I=640, native NVFP4 weights and BF16
activations. Loaded fields hash to
`05384d5b0bbe71843464786f15391673847f5eaa9fa08e2a2e8309ab80c6c90e`.
Routing/activation fixtures are synthetic. Top-k=2/6/10, live M=1–128,
resident counts 128/256/384, warm/scrubbed cache conditions, and policy periods
1/16/128/512 vary explicitly. The prepared capacity remains 128 in all recorded
sweeps. The queried L2 is 50,331,648 bytes; the scrub writes 100,663,296 bytes.

Main and supplemental measurements use
`GPU-cc109c01-9756-d0db-21ea-f1825d3f963f`; the second-card check uses
`GPU-47363510-b87a-13a5-4824-2542e97df76c`. Both report RTX PRO 4000 Blackwell,
SM120, default compute mode, driver 580.173.02 and 145 W power limit. Runs are
serialized across cards to avoid competing PCIe traffic. No service was stopped.
The source runs in container image
`sha256:955e088a85b5378b00275842bc839eea8cb04ca0782ed79eaa3a967d11fd22e5`,
Torch 2.13.0/CUDA 13.3, CUTLASS DSL 4.6.2, Triton
`3.7.1+gitf797708c.nv26.7`, cuda-bindings 13.0.3. The isolated binding mount and
`PYTHONPATH=/workspace:/cuda-bindings` are preserved in launch receipts.

Measurements use default dynamic clocks. Main/supplemental snapshots are P1,
memory clock 13,365 MHz, SM clocks 2,047–2,475 MHz, and throttle masks `0x0` or
software-power-cap `0x4`. These are exploratory data, not release tuning
acceptance. Small profiler deltas are unresolved under these conditions.

Recorded limitations and rejected extrapolations:

- A sub-selection cold fraction cannot be represented in one tiny-M invocation.
  Actual integer counts and fractions are retained; a one-cold C1/top-k=10 case
  is labeled 10%, never 1.5625%.
- Three warmup replays and dynamic clocks do not isolate a few microseconds of
  instrumentation overhead. Some profiled medians are lower than static medians;
  those raw results are retained and are not described as free profiling.
- The roughly 46 ms exchange pause and positive results after sufficient reuse
  concern the journaled PCIe prototype. No PCIe byte counter, DMA-only rate,
  complete-model throughput, learned-static-profile comparison or B300 behavior
  was measured.
- Cache scrubbing is a controlled proxy. It does not prove a particular physical
  weight-fetch count; raw route hits and unique expert touches remain distinct.
- Per-step exchange can lose dramatically, and never-reused promotions lose at
  every tested period. Both outcomes remain in the published tables and receipts.
- No kernel, migration-policy or serving-engine optimization was added during
  the sweep. The next work is to isolate empty-cold and exchange costs, then
  validate promising conditions with real routing traces.

## SM120 empty-tier and exchange cost evidence

The targeted cost work inspects branch `d4eb33e86740b01285d4558eb1e9b941933cb6a0`
and master `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`; master is already an
ancestor, so no rebase is needed. The [cost report](expert-residency-sm120-costs.md)
documents the implemented correction, copy directions, activation of explicit
cacheable allocations, complete spectrum and scope limits.

The final implementation export is `qualified-source-03.tar.gz`, SHA256
`4a715986f3d9963e3f05155e456a284ffa3279c157c250ce0e6fd52c8be6a200`.
All eight spectrum manifests match the unchanged package and experiment source
files. Documentation annotations follow measurement. No preparation declaration,
component registration, cache policy or production storage default is added.

Evidence resides in `/home/jasonc/b12x-sm120-cost-evidence-20260918/`, with original
remote files at `ripper:/home/jasonc/b12x-sm120-cost-results-20260918/`. Retained
files include source exports, exact shell commands, Torch profiler traces,
Nsight Compute reports/SASS, per-copy API/event times, topology, test output,
raw spectrum JSONL, derived CSV and source-integrity checks. Earlier
`b12x-sm120-spectrum-evidence-20260918` receipts remain unchanged.
The complete bundle is `cost-records.tar.gz` in the local evidence directory,
SHA256 `778b5f9c7b3a08b04b4c16f729f4b51a34966b51c1b290035b69cbaaed339065`.

| Receipt | Result and attribution |
| --- | --- |
| `diagnostic-01/`, `empty-cold-ncu.ncu-rep`, `empty-cold-sass.csv` | Baseline unused LUT staging dominates the empty cold kernel. The first diagnostic export and separate NCU source export retain the exact instrumented wrappers. |
| `fixed-write_combined-write_combined/`, `fixed-write_combined-cached/`, `fixed-cached-write_combined/`, `fixed-cached-cached/` | Four allocation combinations on `diagnostic-source-02.tar.gz`; changing only one allocation leaves one slow CPU-read pass. |
| `remote/costs-cached-final/`, `remote/costs-wc-final/` | Six uninstrumented and two instrumented exchanges per mode on the final implementation. Median pauses: 0.860 ms cacheable, 45.926 ms write-combined. Twelve alternating samples per contiguous transport arm retain explicit staging and completion timing. |
| `remote/spectrum-main/` | 74 fixtures pass: 38 latency and 36 policy. |
| `remote/spectrum-hot-128/`, `remote/spectrum-hot-384/`, `remote/spectrum-topk-2/`, `remote/spectrum-topk-6/` | 12 fixtures pass per sweep. |
| `remote/spectrum-reuse-512/` | 10 fixtures pass, including four longer-reuse policy cases. |
| `remote/spectrum-second-card/` | Nine latency fixtures pass on the second physical card. |
| `remote/spectrum-break-even/` | 40 additional fixtures pass: periods 2/4/8 for all four workloads at M=1/8/32, plus four latency controls. |
| `host-final.log` | 82 passed, 11 CUDA skips, 3.05 s. Includes transaction faults, shared contracts, policy, loader, native-layout planning and LUT-staging guards. |
| `remote/gpu-tests.log` | 23 passed, 21.77 s. Includes both allocation modes, int32/int64 IDs, native numerical/graph tests, rollback after three failure points and source-format staging guards. |
| `host-initial.log` | Two test-fixture failures retained: the Trellis constructor correctly rejected the fixture's NVFP4 scale format. The fixture now supplies E4M3 K32 for Trellis and all six staging cases pass. |
| `remote/memcheck.log`, `remote/synccheck.log`, corresponding `.exit` files | Both native policy tests reach the 240-second limit, exit 124. Memcheck emits no completed-test marker; synccheck emits one without a final summary. Both gates remain incomplete; no clean sanitizer result is claimed. |
| `remote/empty-cold-fixed.ncu-rep`, `.txt`, `-sass.csv`, `remote/ncu-fixed/` | Corrected empty kernel: 7.94 µs under Nsight, 147 registers/thread, 54,272 dynamic SMEM bytes, zero reported spilling, one CTA/SM and 16.67% theoretical occupancy. The unused table-load prologue is absent. Baseline Nsight time is 191.81 µs; graph timing is reported separately. |

The spectrum totals **181 passed fixtures**: 105 latency cases with warm/scrubbed
conditions and 76 policy cases. At M=1 the corrected static all-hot graph costs
63.9 µs, compared with the historical 234.9 µs and a 50.0 µs all-VRAM control.
The repeated-cold policy's period-four wall ratio is 0.849 with 14 subsequent
VRAM selections per promotion; period two has ratio 1.056 with seven. These
ratios compare adaptive/static within the same physical device. Rotating traffic
earns no later hits and remains unfavorable. All raw outcomes are retained.

The environment uses the same real checkpoint field hash, Torch 2.13.0/CUDA
13.3, CUTLASS DSL 4.6.2, Triton `3.7.1+gitf797708c.nv26.7`, cuda-bindings 13.0.3,
container image and driver 580.173.02 as the preceding spectrum. GPU UUID
`GPU-cc109c01-9756-d0db-21ea-f1825d3f963f` runs the main spectrum and diagnostics;
`GPU-47363510-b87a-13a5-4824-2542e97df76c` runs the second-card and shorter-period
supplement. Runs are serialized across cards. Both negotiate Gen4 ×16 under
load, share NUMA node zero and retain the 145 W power limit. The host exposes
one NUMA node, so remote-node allocation testing is unavailable. The control
thread permits CPUs 0–63; no CPU affinity pin or production service change is
made. Per-round GPU mode snapshots are retained. Dynamic-clock results remain
diagnostic, with Nsight instrumentation reported separately.

Rejected or deferred changes:

- Conditional graphs and host-selected hot-only graphs are unnecessary to fix
  the measured unused-LUT defect. Residual empty-tier work remains visible.
- Changing only journal or backing memory leaves an expensive write-combined
  CPU read. Both allocation choices are explicit; defaults remain compatible.
- Removing rollback is unnecessary to obtain the measured exchange reduction.
  The generic transaction remains unchanged.
- A contiguous one-way H2D probe is not a canonical-cache transaction. No new
  promotion API, complete promotion latency or break-even claim is derived from
  that lower bound. Field copies, recovery and a pageable-backing miss service
  remain required work before such an API can be qualified.
- The 146→147 register increase is preserved in resource accounting. The
  unneeded 4 KiB shared-memory region is removed; launch geometry is unchanged.
- No cache admission constants are tuned from the synthetic break-even curves.
  Profiler/control overhead remains material on already-hot tiny-M traffic.
- PCIe results do not choose a Grace allocation policy or qualify SM103 TMA.

## Routing locality and canonical-fill evidence

Status: **research-only; no production policy, kernel or SM103 storage change**.
The [experiment report](expert-cache-evolution.md) describes offline real-routing
analysis and a separate recoverable canonical-fill benchmark. The branch parent
is `7768529ac34d4cad46a4ca54052734d06f1052a7`; fetched master remains
`0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`, already included in the branch.
No master commit or speculative rebase is performed.

Raw records reside at
`/home/jasonc/b12x-cache-evolution-evidence-20260918/` and
`ripper:/home/jasonc/b12x-cache-evolution-results-20260918/`.
The final physical source is `source-03.tar.gz`, SHA256
`fb2c4b6e40165702633f7ddfe98f0d6a2e7dc0627160d6215326a80af9b0b060`.
It contains the seeded runners and equal per-boundary validation work. The
all-layer analyzer used `source-02.tar.gz`, SHA256
`434f19ceb38d92d6824e09a5267b53b9695d2be528269f7fed8a86b5f431b636`;
its sidecar records the exact source hashes before the CLI gained automatic
source-hash emission. The replay module and production controller are unchanged
between those exports. The physical receipts retain all 546 file hashes. The physical implementation
and production policy still match those hashes. A subsequent offline-only LFU
fix treats an unseen frequency as zero (rather than LRU's -1 sentinel) when an
explicit one-observation admission threshold is used; 864 repeated recorded LFU
fixtures verify unchanged results with the measured two-observation threshold.

| Receipt | Result and scope |
| --- | --- |
| `routing-capture/` | Eighteen authored C1 nonspeculative requests across six workload labels, 48 layers, native vLLM exported IDs. Six training requests and twelve held-out requests supply 1,650,240 total selections. Responses, prompts, checkpoint/source identity, script backups, restoration and health checks are retained. |
| `capture-checkpoint-fields.json` | Capture host and SM120 host agree on the native layer-zero field hash; its authoritative full digest is recorded below. |
| `replay-first/`, `replay-equivalence.json` | The initial analysis is interrupted after redundant future-use computation proves slow. All 841 completed records are exactly equal to the corresponding amortized-analysis records. No policy semantics change. |
| `replay-second/`, `summary.json` | 48 locality records and 3,744 offline policy fixtures complete. Budgets 128/256/384; learned/positional starts; windows 4/16/128. |
| `workload-{agent,chat,code,math,multilingual,prose}/` | 216 additional offline comparisons on layers 0/12/24/47, with workload-specific training, held-out evaluation and retained per-window/miss details. |
| `fills-qualified/` | Twenty alternating uninstrumented transactions per arm, plus two instrumented transactions, on GPU `GPU-cc109c01-9756-d0db-21ea-f1825d3f963f`. Complete median wall times: exchange 1.244 ms, pinned fill 0.538 ms, pageable fill 0.601 ms, staged fill 0.784 ms. Native numerical and same-graph checks pass after every transaction. |
| `fills-peer/` | Twenty transactions per arm plus two instrumented samples on GPU `GPU-47363510-b87a-13a5-4824-2542e97df76c`, source-02. Medians 1.268/0.626/0.673/0.846 ms. Unseeded supplementary run; no cross-card performance ratio asserted. |
| `trace-native-qualified/` | 2,292 held-out layer-zero invocations, 256 resident experts, window 16, 142 promotions. Static cold selections 7,040 versus adaptive 7,323. Operator-event plus transaction-wall totals: static 1,127.54 ms, exchange 1,320.30 ms, canonical fill 1,205.76 ms. Ratios adaptive/static: 1.171 and 1.069; both lose. Synthetic activations and excluded engine/control/input-copy cost preclude serving claims. |
| `fills-01/`, `fills-02/`, `trace-native-01/` | Preliminary complete-transaction runs remain preserved. The preliminary route replay used unseeded activations and unequal validation frequency; its timing is not the headline comparison. The seeded, equally validated run confirms the unfavorable adaptive outcome. |
| `host-first.log`, `host-second.log`, `host-final.log`, `host-reviewed.log` | Respectively 33/66/107/108 passed, with 2/3/3/3 CUDA skips. Reviewed suite: 3.42 s. Final suite covers shared policy/exchange regressions, locality, censoring, train/test separation, import integrity and canonical recovery faults. |
| `gpu-first.log`, `gpu-second.log`, `gpu-qualified.log` | Respectively 23/32/32 passed. Final run: 31.39 seconds, including 11 GPU cases and 21 host cases. Tests include int32/int64 routes, changed live counts, changed inputs, generation/map checks, stable pointers, no Torch replay allocator events, retained programs and recovery after submitted writes. |
| `memcheck-small.log`, `synccheck-small.log`, `.exit` files | One bounded native graph/fill test passes under each tool: 39.73/18.61 seconds, exit zero, zero reported errors. The transaction/test files match final source. Earlier full-suite timeouts remain separate incomplete gates. |
| `recovery/`, `bench-recovery.py` | Twenty-four submitted-write fault cases restore the victim, map, generation and native graph output. Four samples per transport/point. Pinned/pageable/staged median failed-transaction-plus-recovery wall times: 0.887/0.844/1.023 ms after first payload write; 0.854/0.898/0.996 ms after publication. Benign injected exceptions do not establish recoverability or latency of real CUDA device faults. |
| `lfu-reviewed-{0,1,2,3}/`, `lfu-equivalence.json` | All 864 LFU comparisons repeat exactly after correcting the non-default unseen-frequency score margin. Locality records also match. |
| `host-topology.txt`, `hardware-after.txt`, runner manifests | PCIe Gen4 ×16, one NUMA node, CPU affinity 0–63, dynamic clocks and 145 W power limits. Both GPUs return idle; no ripper service is stopped. No multi-NUMA comparison, SM103 run or overlap measurement is available. |

The authoritative checkpoint field SHA256 in `capture-checkpoint-fields.json`
and every physical manifest is
`05384d5b0bbe71843464786f15391673847f5eaa9fa08e2a2e8309ab80c6c90e`.
This identifies the 4,608 native layer-zero source fields, not the complete
checkpoint or a quality evaluation. The environment/container matches the
preceding SM120 cost report. GPU runs are serialized across physical cards.

Decisions and rejected extensions:

- Real-routing locality is sufficiently nonuniform to retain per-layer analysis.
  At half residency, learned static initialization reduces held-out cold
  selections from 48.84% positional to 22.16%. The mixed-workload blend is an
  explicit experiment, not an automatic production profile merge.
- Earned promotion hits do not establish net benefit. Layer zero earns 1,706
  later hits but loses another 1,989 resident selections through eviction.
  Thirty-five of 86 completed promotion lifetimes have zero hits. Static wins
  even with the cheaper fill; the policy is not retuned against these prompts.
- Canonical fill remains a benchmark primitive. It has a complete recovery
  protocol but consumes canonical backing for hot experts; the shared exclusive
  map contract and a pageable/mmap miss service still need explicit design.
- Pinned canonical copies are faster here, but the experiment does not justify
  pinning entire checkpoints. The bounded staged transport is measured while a
  separate mapped canonical layer continues to serve misses. In allocation
  manifests, `extra_pageable_source_bytes` records bytes referenced by the fill
  source; all arms share the loader's one pageable source dictionary, so those
  entries must not be summed as additional allocations.
- No spare slot, asynchronous overwrite, hot-first kernel, fused profiler,
  production LRU/LFU implementation or fused-MoE rewrite is added. The traces
  lack GPU scheduling timestamps and cannot establish overlap opportunity.
- Accepted-token exports cannot recover rejected speculative work or concurrent
  invocation grouping. The importer requires the explicit C1/no-speculation
  contract instead of fabricating these dimensions.
- The corpus cannot resolve a 512-invocation within-request horizon. Censored
  observations remain unavailable rather than being labeled non-reuse.
- Historical 141/181-fixture spectra and failed sanitizer runs remain unchanged.
  There is no new SM103 compiler census or physical B300 performance claim.

The reviewed offline-source archive is `source-04-reviewed.tar.gz`, SHA256
`ee207cfc3721acce3f87572dea9053c96a614b877b6c005cdb27b3816ac1b622`.
The 864 LFU fixtures and 48 repeated locality records match exactly after the
non-default score correction. The physical source-03 receipts remain tied to
their original hashes; their transaction, kernel, production-policy and native
replay implementations are unchanged by the offline comparison correction.

The immutable evidence bundle is `evidence.tar.gz`, SHA256
`e36af6f16eb618ee763baf6542571570dcf529834d4dce6cfe5c4c92c751ceb9`.
All 166 member files in its receipt manifest were hash-verified after packaging.

## Held-out policy and matched-backing replay evidence

Status: **research-only; completed SM120 diagnostics**. The
[policy evaluation report](expert-cache-policy-evaluation.md) records longer
checkpoint-derived routing traces and additional-layer physical replay without
changing shared policies, kernels, preparation contracts or serving defaults.
Master `0f3a8cbf` remains an ancestor of the branch; no reconciliation is needed.

The corpus contains 24 authored requests across six workload classes. Twelve
train the initial placement and twelve are held out, totaling 12,276 training
and 11,447 held-out decode invocations per layer. One code response ends early
without completing its requested implementation. It is retained with proper
horizon censoring; this is routing evidence, not a quality evaluation. Native
vLLM export supplies canonical IDs under C1/no-speculation. Physical replay uses
native checkpoint weights, synthetic activations and uniform route weights.

The 2,592 offline policy fixtures cover all 48 layers, budgets 128/256/384,
windows 4/16 and learned/positional starts. At 256 resident experts, learned
static placement has a 20.14% cold-selection rate versus 48.21% positional.
Decayed LFU/window-4 reduces cold selections to 8.29%, LRU/window-4 to 8.38%,
and cumulative LFU/window-16 to 17.40%. Recent-frequency/window-16 increases
the rate to 21.91%. The evaluation does not tune policy constants against these
held-out requests or install alternative policies in production.

Eight paired physical trials cover layers 12/24/47 and a second-GPU layer-24
repeat. The static and adaptive arms both retain canonical backing to hold
cold-row geometry fixed. This is an explicit benchmark option; the original
exclusive static control remains the CLI default. All trials include complete
fill transaction wall time. Decayed LFU/window-4 reduces the operator-plus-fill
total by 25.8–35.3%; LFU/window-16 by 8.4–22.3%. The layer-24 LFU ratio is
0.7890 on GPU 1 and 0.7966 on GPU 0, adaptive/static. These are single-layer
diagnostics, excluding observation, policy computation and serving scheduling.

The shorter physical prose-to-code schedule also shows why scope matters:
recent-frequency improves layer 24 by 7.6% here despite losing cold selections
on the complete six-class schedule. Layer-12 LFU wins in aggregate while losing
its code interval. These unfavorable outcomes remain in the report.

Evidence roots:

- Administration copy: `/home/jasonc/b12x-cache-policy-evidence-20260919/`.
- Physical host: `ripper:/home/jasonc/b12x-cache-policy-results-20260919/`.
- `protocol.json` and `protocol-amendment.json`: selection fixed before physical
  results, including matched canonical backing and the second-GPU repeat.
- `routing-capture/`: exact prompts, responses, import manifest, original lane
  scripts, capture logs and restoration checks. `physical-trace.json` records
  its parent trace and selected requests.
- `offline-{0,1,2,3}/`: 48 locality records and 2,592 replay fixtures.
  `shifts.jsonl` retains the 15 selected-layer workload-shift analyses.
- `physical-*/`: eight source-bound records and all 32,736 raw timed replays.
  The 964 promotions retain generation, pair and complete wall-time records.
- `audit.json`: 1,960 passing numerical checks, 32 zero-allocation measurements,
  exact ordered reduction, lifetime pointer checks, source hashes and cold-count
  agreement. Maximum relative L2 is 0.00120905; minimum cosine is 0.999999225.
- `host.log`: 53 passed, 3 CUDA skips. `gpu.log`: 41 passed on SM120.
  `host-source02.log`: 9 focused schedule tests passed.
- `topology.txt`, `environment.txt`, `hardware-after.txt`: physical topology,
  container/toolchain identity and both GPUs released idle after testing.

The offline and initial test archive `source-01.tar.gz` has SHA256
`56e363514b3df7dcc4880d591d805337bf3f590979dddfa383bf86ae8b90bba3`.
The physical archive `source-02.tar.gz` has SHA256
`9f3ca9c044247933414cff5b9a8b5926dcb1d785367a7216fe79b518fe16c391`.
The latter adds the optional matched static-backing selector. Native kernels,
fill transactions, shared policy and offline analyzer are identical between
archives. Per-file hashes bind every physical and offline receipt.

Decisions, limitations and retained failures:

- The initial summary command raced an in-progress physical record before its
  replay file existed. `summary-first-failure.txt` retains the failure. The
  aggregator skips unfinished records; the acceptance audit independently
  requires all eight experiments to pass. No benchmark is dropped.
- Full canonical backing in both arms removes a geometry confound from the
  policy comparison. It does not establish that canonical backing fits a
  complete model or that pinning the full checkpoint is desirable.
- In-loop fills have 0.320–0.385 ms medians under these conditions. The fill
  implementation is unchanged; this is not an optimization claim against the
  preceding standalone transport measurements.
- Per-promotion zero-hit rates and earned hits are retained, but individual
  causal profit is not invented from a constant cost per route. Policy-level
  totals include both eviction harm and transaction time.
- LRU is close to decayed LFU offline but remains physically unmeasured.
  No production LRU/decayed-LFU controller, asynchronous fill, spare slot or
  fused-MoE change is added.
- Captured invocation order has no GPU scheduling timestamps. Overlap and
  complete-model performance remain unmeasured. Physical B300 qualification is
  still required; SM120 PCIe results do not qualify Grace-backed TMA.
- Historical 141/181-fixture spectra, bounded sanitizer successes and full-suite
  sanitizer timeouts retain their original sources. No compiler census or
  sanitizer rerun is claimed for benchmark-only edits.

The immutable `evidence.tar.gz` bundle has SHA256
`8a940b27a23fcf47cc334209ac41c339c5c1a8e51494c47940673d01c52d8e79`.
Its 164 receipt files are verified both inside the archive and against the
administration copy. No raw benchmark evidence is added to the repository.


## Prepared SM120 serving cache, September 19, 2026

Status: **implemented and experimentally qualified for single-rank serving**.
The [serving contract](expert-cache-serving.md) describes CPU checkpoint loading,
model-wide admission, the registered `moe.expert_cache` preparation backend,
canonical fills and the opt-in vLLM V2 integration. SM103 retains its distinct
MXFP8/MXFP4 HBM/Grace transport and physical gates.

Source and environment:

- b12x base: `52a12b46b890915fc15d95250bdeb3976691ae68`, on
  `work/sm103-bringup`. Master `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`
  remains an ancestor; the branch is 80 commits ahead and zero behind before
  these changes. No rebase or master write is necessary.
- Companion: `codex/b12x-expert-cache`, based on maintained PreparationSession
  branch `ef1aeaf080879865febd27a92c5644233d157987`. Inspected vLLM main is
  `47ccf6c57d92f03630ebcbad3809450545825488`; the older SM103 companion is not
  used. The companion adds a CPU ModelOpt loader, preparation registration,
  explicit phase handoff and three loader tests. Final companion commit is
  `1d1f870bd6`; `companion-source-11.json` and its archive bind its files.
- Measured source archive: `source-09.tar.gz`; b12x package SHA256
  `5bf175baabcb3b4d0678ce819a5e854e782fb554b016c8b11d71d3220a852da2`.
  `companion-source-09.json`, its patch and `companion-files-09.tar.gz` bind all
  five companion files, including added files absent from a tracked-only patch.
- Final code archive: `source-11.tar.gz`; package SHA256
  `2b90cf1b36809849be8449c579d42f3b614a3950489f598094134428d3daa94c`.
  It adds three public names to `api.__all__` and rejects non-CPU optional
  source metadata/retained tensor owners. b12x kernels, execution preparation,
  policy and benchmark implementations are byte-identical to measured source.
  `source-difference-09-11.json` records the comparison. Final host/GPU tests
  and the compiler census are rerun on source-11. The nine-run serving matrix
  and sanitizer receipts remain bound to source-09. Companion source-11 adds
  non-dictionary configuration opt-out guards and uses the maintained Torch
  accelerator aliases for device selection and preparation cleanup.
- Physical serving GPU: `GPU-47363510-b87a-13a5-4824-2542e97df76c`, RTX PRO 4000
  Blackwell, 24 GB, driver 580.173.02, PCIe Gen4 x16, dynamic clocks, 145 W cap.
  GPU `GPU-cc109c01-9756-d0db-21ea-f1825d3f963f` runs correctness/sanitizers.
  The host has one NUMA node and CPU affinity 0–63. Some separate-GPU sanitizer
  work overlaps serving; both GPUs share host CPU/memory resources.
- Runtime image: `jovian-judgement-qwen38-sm120-amd64-4c1f7b2-a45d3f7-c3:latest`,
  Torch 2.13.0, CUDA 13.3, CUTLASS DSL 4.6.2, Triton 3.7.1 and CUDA bindings
  13.0.3 overlay. Maintained Python source uses the image's compiled vLLM
  extensions, not a matching complete rebuild. Full-decode CUDA graphs with
  mode 0 and FlashInfer attention work; Inductor is not qualified.
- Checkpoint: `nvidia/Qwen3-30B-A3B-NVFP4`, revision
  `2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3`, 16.85 GiB on disk. Complete
  local content fingerprint:
  `bdde460712efe6ea50175eac186c27aa983348baf9faf2ab28a917ed93182609`.
  All 6,144 gate/up global-scale pairs match without reconciliation. Routed
  activations explicitly use W4A16; this is not A4 equivalence or model quality
  qualification. Dense weights follow ordinary engine preparation.

Evidence roots are `/home/jasonc/b12x-serving-evidence-20260919/` and
`ripper:/home/jasonc/b12x-serving-results-20260919/`. The administration copy's
`physical/` directory retains the complete remote receipts, including failures.
`protocol-09.json`, `protocol-amendment-09.json`, `run-serving.sh`, raw JSONL,
per-run telemetry and `serving-summary-09.json` specify the experiment. No raw
checkpoint, compiler or serving receipts are committed to the repository.

The learned profile has 58 resident experts in every one of 48 layers. Its hash
is `20006a3fa40680307aa6d1f72816ded8fe91c4d7297c44410e585031a28fc4fa`.
Eight authored general requests provide calibration; sixteen separate evaluation
requests contain eight stable general prompts followed by eight code prompts.
Every request generates 128 greedy tokens with actual runtime gate weights.
The model-wide cache envelope is 8 GiB, including private scratch/maps/counters.
KV is explicitly BF16 with 2 GiB reserved. Context is 2,048; prepared capacity
is 64 tokens. Both arms use the same profile and graph geometry.

| Concurrency / arm | Output tok/s | TTFT median ms | Delivery gap median / p99 ms | Promotions |
| --- | ---: | ---: | ---: | ---: |
| C1 learned static | 52.91 | 147.22 | 14.35 / 45.63 | 0 |
| C1 adaptive, 32-token trigger | 52.31 | 130.20 | 11.40 / 149.28 | 928 |
| C4 learned static | 79.43 | 270.67 | 50.60 / 96.35 | 0 |
| C4 adaptive, 32-token trigger | 72.72 | 284.15 | 40.00 / 202.83 | 736 |
| C8 learned static | 102.85 | 500.21 | 61.22 / 140.85 | 0 |
| C8 adaptive, 32-token trigger | 90.15 | 503.09 | 76.76 / 254.92 | 576 |
| C1 adaptive, 128-token trigger | 54.08 | 142.82 | 13.46 / 45.60 | 240 |
| C4 adaptive, 128-token trigger | 78.33 | 281.91 | 47.00 / 176.85 | 224 |
| C1 observation/control, zero movement | 44.23 | 149.20 | 15.11 / 137.02 | 0 |

Each cell is one complete serving run, not a statistically established winner.
Triggers count delivered output tokens across requests, not scheduler iterations.
The 128-token runs are explicitly exploratory follow-ups to the frequent-epoch
results. Ratios must compare matching concurrency: frequent adaptive/static
throughput is 0.9887, 0.9155 and 0.8766 at C1/C4/C8. Longer epochs yield 1.0222
and 0.9862 at C1/C4. The small C1 difference is inconclusive under dynamic clocks.
No production default changes.

| Adaptive trigger / concurrency | Epochs | Median / p95 pause ms | Total pause s | Copy bytes |
| --- | ---: | ---: | ---: | ---: |
| 32 / C1 | 58 | 153.94 / 254.55 | 10.060 | 2,464,746,752 |
| 32 / C4 | 46 | 208.08 / 307.31 | 10.170 | 1,954,823,936 |
| 32 / C8 | 36 | 252.13 / 356.81 | 9.541 | 1,529,864,704 |
| 128 / C1 | 15 | 148.46 / 186.02 | 2.284 | 637,439,872 |
| 128 / C4 | 14 | 205.88 / 277.71 | 3.030 | 594,937,600 |

Copy-byte accounting includes payload and required map traffic. Raw receipts
retain layer transaction durations and all control stages. At frequent C1,
median pause/drain is 46.63 ms, begin/snapshot RPC 21.51 ms, policy 9.07 ms,
preflight RPC 9.31 ms, apply RPC 24.06 ms, acknowledge RPC 4.01 ms and resume
1.22 ms. Stage medians do not sum to the median complete pause. The supported
AsyncLLM pause includes a 20 ms output-settling delay. JSON receipt generation,
request scheduling and the final pending epoch remain in serving wall time.
GPU iteration latency and isolated D2H/publication DMA are not measured here.

The no-movement C1 control observes 24.11% cold selections, versus 15.77% for
frequent adaptation and 20.92% for longer epochs. It preserves the static
placement and outputs but pays observation/control overhead; its throughput is
not used as the uninstrumented static baseline. Observations cover completed
pure-decode windows and exclude the final partial window. Window samples wholly
inside the general interval give 7.16% static versus 7.45% frequent adaptive cold
selections; wholly inside the code interval, 41.24% versus 23.84%. Crossing
windows are excluded from those phase-specific figures.

Stable general traffic loses throughput under adaptation. At C1 the interval
rates are 91.95 static, 66.83 frequent and 83.30 longer-epoch adaptive tok/s.
The code interval rates are 37.14, 42.99 and 40.06. The frequent C1 run does not
recover its initial stable-interval deficit before completion. No future
break-even point or universal workload benefit is inferred.

All nine completed source-09 evaluation runs generate 18,432 tokens. Every
adaptive and instrumented control matches its same-concurrency static output
IDs exactly. All 2,704 serving promotions preserve the captured graph objects
and slab/map/workspace pointers. The component tests independently check zero
Torch allocator events during replay; complete-engine allocation tracing is
not claimed. Repeated fresh worker startups/cleanup pass the serving protocol,
with the image's forced-shutdown warnings retained.

Final integration validation uses source-11 and companion `1d1f870bd6` on GPU
`GPU-cc109c01-9756-d0db-21ea-f1825d3f963f`, with the same profile and evaluation
requests. `final-static-c1-11.jsonl` records 53.07 tok/s; the 128-token adaptive
arm records 54.42 tok/s. Median TTFT is 145.09 versus 140.78 ms, and p99 delivery
gap is 45.64 versus 45.38 ms. Fifteen epochs perform 240 promotions with a
149.29 ms median pause and 2.292 s total pause. All 4,096 tokens match exactly
between arms and also match source-09 C1 outputs. All 48 layers retain graph
objects and cache pointers. The second GPU confirms the final integration, but
one additional dynamic-clock pair does not establish a general performance win.
`serving-summary-11.json` and `serving-audit-11.json` retain the separate results.
Across both source-bound sets, 22,528 tokens and 2,944 promotions pass.

| Validation receipt | Result |
| --- | --- |
| `host-11.log` | 281 passed, 5 physical-GPU skips, 8.88 s. Admission, identity/profile validation, calibration, shared policy, epoch faults and source ownership. |
| `physical/prepared-cache-11.log` | 24 passed, 2 DSL warnings, 43.92 s. Native cache graphs, production H=2048/I=768 at capacity 64, int32/int64 invalid/duplicate routes, changed inputs/live M, fills, exact reference outputs, stable pointers, frozen resolution, zero replay allocations, bounded counters and model-wide epochs. |
| `physical/vllm-loader-11.log` | 3 passed, 14 existing Torch warnings, 8.88 s. CPU allocation under an ambient meta device, retained byte ownership, rejection of unequal gate/up global scales and non-dictionary configuration opt-out. |
| `physical/prepared-cache-memcheck-09.log` | Two production-geometry graph cases passed in 327.17 s, zero sanitizer errors; 11 unrelated cases deselected. |
| `physical/prepared-cache-synccheck-09.log` | Same two cases passed in 147.03 s, zero sanitizer errors. |
| `sm103-full-11/manifest.json` | 86 declarations, 244 distinct programs: 238 native CuTe exports and 6 supporting Triton programs; source unchanged, CUDA uninitialized. Compiler environment Torch 2.14.0, CUTLASS 4.6.2, Triton 3.8.0. Offline evidence only. |
| `sm120-native-09/` | Six exact cache objects from the measured/tested source retain manifest-verified object SHA256s and extracted CUDA ELF hashes. Whole-K native kernels use 144 allocated registers, 1,024 B static SMEM, 54,272 B dynamic launch SMEM, zero stack and zero local memory. Achieved occupancy is not measured. |
| `preparation-09.log` | 130 passed, 33 skipped, two pre-existing stale internal-API tests fail because they omit required `decode_config`. Both failures reproduce on pristine `52a12b46` in `baseline-execution-52a.log`. |
| `preparation-06.log`, `baseline-registry-52a.log` | Five MXFP6 registry/META failures reproduce on pristine `52a12b46`. They are not concealed by the focused suite. |

The checkpoint-backed graph test hashes layer-12 source fields as
`4821ecbebad0a22b8aedc642a6efe9441f23b83673243af9eb1505d9085cd26b`.
It uses all 128 experts and live M=1/2/4/64, comparing cache output with a native
all-resident whole-K control exactly before and after repeated fills.

Failures and decisions retained:

- Source-07 serving diverges after promotions. Production-sized randomized
  tests expose one-BF16-ULP differences that H=I=128 misses. Split-K grouping
  depends on route packing. Whole-K scheduling fixes exact same-input parity
  without changing expert bytes or BF16 boundaries. It is an explicit numerical
  recipe; the native split-K result is not claimed bitwise equivalent. Decode
  query schema becomes 9. Source-07 timings and failed probes remain diagnostic.
- The initial native oracle omitted its identity map, allowing invalid routes
  to read stale native scratch. The oracle is corrected; raw failed runs and
  debug variants remain retained.
- Initial loader attempts lack `use_global_sf` and then miss vLLM's existing
  ModelOpt class-name dispatch. The dedicated subclass supplies both contracts.
  It does not patch loader functions or rename logical expert IDs.
- The mixed engine build's Inductor run fails on a missing unrelated DeepSeek
  extension overload. A BF16 attention attempt selects unsupported-PTX FA
  fallback. Full-decode graphs with mode 0 and explicit FlashInfer attention
  are the measured configuration. Neither failure is called a passing gate.
- A sanitizer selector initially matches no tests. Two later collection
  commands use nonexistent filenames. Their logs remain; explicit corrected
  invocations supply the accepted test totals. No sanitizer timeout occurs in
  the bounded source-09 runs; historical full-suite timeouts remain unchanged.
- Companion precommit initially rejects formatting, unguarded dictionary access
  on vLLM's alternate configuration type and forbidden Torch CUDA API names.
  Explicit opt-out guards and maintained accelerator aliases fix those cases.
  All required companion hooks, including mypy and import/API checks, pass on
  commit `1d1f870bd6`; failed hook logs remain in the evidence root.
- Direct cuobjdump on CUTLASS host objects reports no device code. The existing
  migration evidence extractor obtains the exact embedded CUDA ELF after
  object-hash verification; both failed attempts and successful resource dumps
  are retained.
- Full mapped canonical backing plus retained CPU sources costs about 30.38 GiB
  of host payload for this model. This is admitted on the test host, not a
  recommendation to pin arbitrary full checkpoints. Bounded pageable/mmap
  staging requires a separate miss-service design. TP/DP/PP>1, EP, speculation,
  LoRA and modified numerical recipes fail closed in the serving loader.
- No faster unsafe pause, concurrent replacement, production decayed-LFU
  default, spare-slot requirement or fused-MoE rewrite is added. The measured
  priority is reducing epoch coordination/serialization cost and avoiding
  unproductive stable-workload epochs. Physical B300 correctness/TMA gates
  precede any transfer of this serving experiment to SM103.
- GitHub exposes zero status contexts and zero check runs for the inspected
  base source. Local evidence does not replace a matching engine build or PR CI.

Both physical GPUs return idle with 2 MiB allocated after the experiments.
The pre-existing homeassistant container remains running; no household service
is stopped. Historical 141-fixture and 181-fixture spectra and the held-out
single-layer receipts are unchanged and are not relabeled as serving evidence.


## Scheduler-owned maintenance and serving control costs

Status: **experimental single-rank control; SM103 physical qualification unchanged**.
The source begins at b12x `b067db404e8b2dd22855482eaa11bff68c631342` on
`work/sm103-bringup` and companion vLLM
`1d1f870bd617a4905637a30fcb552859b9fb2ded` on `codex/b12x-expert-cache`.
The inspected master is `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`;
the companion main is `47ccf6c57d92f03630ebcbad3809450545825488`.
Raw receipts live outside the repository at
`/home/jasonc/b12x-control-evidence-20260919`; the remote collection directory is
`/home/jasonc/b12x-control-results-20260919` on ripper. Source archives,
companion patches/file hashes, commands, telemetry and failed runs are retained.

The implementation keeps the existing scheduler/device drain, moves the
single-rank policy into one worker RPC and avoids the administrative frontend's
20 ms output-settling delay. A configured cold-fraction gate declines movement
without discarding decayed history. Immutable unchanged map generations avoid
repeated structural validation. The shared all-rank protocol, static opt-out,
expert-copy transaction and SM103 storage model remain intact.

The frozen library package is
`667fad34f8bb0ba5ad048154e2cfd6281ebe206a19b1b8e32a69fd34e89e0d6b`.
`source-06.json` binds the engine files; `source-07.json` retains the same library
and engine hashes with the additional research-only cadence harness. The
`sm103-full-06/manifest.json` compile receipt passes 86 declarations and 244
programs, including 238 native CuTe exports, with CUDA uninitialized and the
source unchanged. This is compile evidence, not B300 execution evidence.

The serving matrix uses the same Qwen3-30B-A3B-NVFP4 checkpoint, learned
58/128-per-layer profile, 16 authored requests, 128 generated tokens per request,
8 GiB expert envelope, 2 GiB BF16 KV and full-decode graph configuration as the
prepared-cache qualification. Every arm starts a fresh worker. Static has no
observer; counters-only has no epochs. Fixed external 32/128-token epochs,
conditional worker-local maintenance and explicit healthy-check backoff are
separate arms. Conditional experiments use a declared 0.15 cold-fraction gate,
not a production default. The backoff experiment uses 32 through 256 delivered
tokens and does not claim fixed four-iteration observation semantics.

GPU 0 is RTX PRO 4000 Blackwell
`GPU-47363510-b87a-13a5-4824-2542e97df76c`; GPU 1 is
`GPU-cc109c01-9756-d0db-21ea-f1825d3f963f`. Driver 580.173.02, CUDA 13.3,
Torch 2.13.0, CUTLASS DSL 4.6.2 and Triton 3.7.1 identify the physical image.
The workstation has one NUMA node and negotiated PCIe Gen4 x16. Dynamic
clocks/power and CPU affinity are retained in receipts. The complete engine
build is paused during timing collection. Image binary extensions plus modified
maintained Python sources remain an explicit experimental build identity until
separately verified against a complete source-matched build.

Failures and rejected approaches retained:

- `stalled-maintenance-02.json` and its raw serving log retain an engine that
  completed its first maintenance transaction but did not resume requests.
  An idle callback resumed existing work, then the input loop blocked waiting
  for another client message. Rechecking `has_work()` after callbacks fixes the
  missing wakeup; a regression test requires progress without another message.
- `whole-k-03`, `whole-k8-03`, `whole-debug-03` and `whole-routes-03` expose a
  one-token packed W4A16 shortcut that reads only the first route in each
  expert block. A repeated expert leaves its later route unwritten. The mapped
  variant clears that row to zero; the unmapped variant can expose stale scratch.
  Adding an identity map merely hides the stale value and is not a fix.
  Restricting the shortcut to top-1 restores the declared duplicate-route
  contract. A production-sized independent top-1 oracle verifies repeated routes
  under the same captured graph. The serving capacity-64 path does not select
  this one-token compile specialization; prior serving receipts remain intact.
- `gpu-tests-05.log` retains an incorrect test filename. The corrected suite in
  `gpu-tests-05b.log` passes 24 cases and rejects the first independent-oracle
  declaration because an all-resident cache cannot reserve eviction pairs.
  The test declares no updates for that all-resident oracle; `duplicate-test-06`
  passes. Neither collection failure nor invalid test declaration is a GPU pass.
- `matching-build-03.log` retains the first build launch failure: the copied host
  `uv` executable requires a missing jemalloc library. The isolated build uses
  the official standalone wheel instead. Build logs and artifacts remain under
  `/models/b12x-control-build-20260919` on ripper.
- No always-on pressure atomic, router fusion, device policy, asynchronous fill,
  spare slot, storage rewrite or production cadence is introduced. Measurements
  distinguish counter overhead, host maintenance, movement and deterministic
  compute cost before choosing further work.

### Complete control cost

The fixed-cadence serving table in the
[maintenance guide](expert-cache-maintenance.md#serving-evidence) is computed
from `static`, `observe`, `external32`, `external128` and `maintenance` receipts
ending in `-06.jsonl`. Each concurrency has one shared output-token hash across
those arms. The additional `maintenance-repeat-c1-06` and
`static-repeat-c1-06` receipts give 63.00 and 52.82 generated tokens/s, compared
with 63.02 and 52.84 in the first pair. All rates include control work.

Median wall-time stages for the C1 external 32-token arm:

| Stage | ms |
| --- | ---: |
| Pause/drain roundtrip, including output settling | 45.58 |
| Administrative output settling alone | 20.17 |
| Begin/snapshot RPC | 21.74 |
| Client policy | 9.08 |
| Preflight RPC | 9.25 |
| Apply RPC | 24.43 |
| Acknowledge RPC | 4.25 |
| Resume RPC | 1.23 |
| Pause-to-resume interval | 152.27 |
| Complete client control call | 162.37 |

The broad stages do not account for all Python serialization/bookkeeping, and
medians are not additive. Worker snapshot work is about 2 ms, substantially less
than its RPC roundtrip. Administrative output settling is one cost, not the
entire explanation. Worker-local control removes repeated map-bearing RPCs and
retains only compact diagnostic replies.

Conditional-maintenance checks separate no movement from completed promotions:

| Concurrency / check | Count | Engine interval median ms | Drain median ms | Worker RPC median ms | Policy median ms | Fill-batch median ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 / no movement | 32 | 23.23 | 8.66 | 14.28 | 7.10 | 0.002 |
| C1 / promotion | 28 | 42.87 | 15.39 | 27.88 | 8.91 | 7.03 |
| C4 / no movement | 25 | 41.21 | 27.19 | 14.17 | 6.85 | 0.002 |
| C4 / promotion | 26 | 83.09 | 54.57 | 28.52 | 8.80 | 7.03 |
| C8 / no movement | 21 | 54.81 | 40.55 | 14.61 | 6.84 | 0.002 |
| C8 / promotion | 21 | 115.09 | 85.86 | 28.35 | 8.98 | 6.94 |

Drain includes the remaining useful execution of a submitted iteration. The
worker's no-movement cost remains approximately 14 ms across concurrency; the
larger engine interval is not all wasted CPU control time. C1 counter snapshot
medians are 0.044 ms drain, 0.058 ms D2H and 1.908 ms host decoding. Pressure
classification takes 2.179 ms; layer policy 5.958 ms and ranking 0.215 ms.
Frontend lock waiting is 0.003 ms. These are nested host intervals, not isolated
device-event measurements.

Client latency remains visible rather than being absorbed into aggregate rates:

| Arm | TTFT p50 / p95 ms | Delivery gap p50 / p95 / p99 ms |
| --- | ---: | ---: |
| C1 static | 147.73 / 195.04 | 14.30 / 36.04 / 46.22 |
| C1 conditional | 132.35 / 180.17 | 11.74 / 33.59 / 43.05 |
| C4 static | 269.79 / 512.60 | 51.19 / 82.88 / 96.83 |
| C4 conditional | 258.59 / 509.56 | 38.35 / 79.10 / 102.81 |
| C8 static | 502.06 / 938.89 | 61.50 / 133.63 / 140.78 |
| C8 conditional | 505.40 / 940.95 | 66.25 / 119.95 / 142.01 |

Delivery gaps are client event intervals, not GPU iteration durations; raw
receipts retain coalesced-token counts and intervals overlapping maintenance.
Tail latency is not uniformly better. Conditional C1/C4/C8 perform 448/416/336
promotions, account for 1,189,871,104 / 1,104,880,896 / 892,412,544 copy bytes,
and record 69,508 / 68,458 / 60,732 later selections of promoted experts.
Their observed decode cold fractions are 16.05% / 16.60% / 16.51%; static
has no observer, so no static cold fraction is inferred. Engine-interval p95
is 60.11 / 99.25 / 122.40 ms. The final summary retains request decode rates,
all workload splits, skipped proposals, promotion lifetimes and raw epoch series.

The 382 C1 layer-fill transactions have a 0.398 ms median complete transaction.
Median stages are 0.015 ms initial drain, 0.031/0.013 ms map-read enqueue/wait,
0.119 ms validation/encoding, 0.162/0.016 ms payload enqueue/wait and
0.025/0.012 ms map-publication enqueue/wait. Enqueue can include synchronous
work; these intervals are not transport bandwidth measurements. Copy mechanics
are unchanged. The evidence does not prioritize further DMA optimization.

The 32–256-token backoff experiment retains separate `backoff-c*-07` receipts.
C1 and C8 match static tokens and reduce stable penalties to about 2%, with
slower transition recovery than fixed checks. C4 generates different token IDs
for requests 4–7 before any promotion; its first movement is recorded only at
1,309 delivered tokens. The whole stable interval retains generation zero.
The run's 87.31 overall / 134.75 stable / 64.60 code tokens/s are retained as an
**unqualified comparison**, not evidence of a matched-output speedup. Changed
batch scheduling is a hypothesis; no unmeasured causal explanation is asserted.
Backoff remains an explicit research harness option.

The ungated worker-local C1 arm, `maintenance-ungated-c1-07`, matches output IDs
and produces 63.08 tokens/s overall, 84.45 stable and 50.43 after transition.
It performs 960 promotions versus 448 with the gate, but its whole-run rate is
effectively the same. The narrower engine boundary explains the main measured
gain; this corpus does not establish an additional aggregate throughput win
from the pressure threshold. The gate reduces movement and stable-workload loss.

### Deterministic scheduling cost

`whole-k-08.json` binds 16 physical comparisons on GPU 1: M=1 through 128,
top-k=2/8, checkpoint layer 12, E=128/H=2048/I=768. Each arm retains 12 balanced
samples of 100 CUDA graph replays, with warm constant activations/routes.
The output is finite/nonzero; the prepared cache matches native whole-K exactly
at every shape. Timed replay records zero Torch allocator events under frozen
kernel resolution. `whole-k-08-telemetry.csv` retains clock/power samples.

Median complete operator microseconds, lower is better:

| M | top-k=2 preferred | whole-K | all-resident cache | top-k=8 preferred | whole-K | all-resident cache |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12.29 | 24.21 | 34.81 | 26.64 | 38.93 | 51.22 |
| 2 | 16.42 | 24.57 | 34.83 | 53.66 | 64.49 | 73.69 |
| 4 | 26.63 | 38.94 | 49.22 | 144.77 | 168.84 | 179.35 |
| 8 | 55.00 | 65.52 | 73.70 | 285.03 | 280.07 | 297.14 |
| 16 | 160.62 | 158.97 | 166.90 | 450.26 | 448.39 | 468.52 |
| 32 | 264.74 | 258.32 | 276.11 | 574.81 | 571.16 | 594.70 |
| 64 | 429.09 | 423.53 | 442.94 | 659.23 | 655.63 | 690.09 |
| 128 | 569.86 | 565.90 | 594.59 | 1117.56 | 1103.79 | 1166.67 |

The preferred path selects `moe.w4a16.small_m_direct` through M=8 at top-k=2
and M=4 at top-k=8. Whole-K uses the packed native path. The small-M ratio
therefore includes route preparation and execution-structure differences; it
does not isolate split-K arithmetic alone. Preferred/whole-K cosine is at least
0.999986, but small-M results differ in many BF16 elements. No numerical mode
is substituted in serving. Larger shapes are close in this warm diagnostic;
small percentage differences under dynamic clocks are not tuning winners.

`sm120-native-08/identity.json` and `resources.json` bind 22 captured native
objects to verified manifests/cubins and exact driver resource/occupancy queries.
At top-k=8/M=1, the direct kernel uses 127 registers, 512 threads/CTA and 5,312
static SMEM bytes. Whole-K uses 144 registers, 256 threads/CTA, 1,024 static plus
54,272 dynamic SMEM bytes. Both permit one CTA/SM. At M=128, preferred/whole-K
use 152/150 registers, 128 threads/CTA and 1,024 + 27,648 SMEM bytes, permitting
three CTAs/SM. These are resource bounds, not measured achieved occupancy.
The audited compute kernels have no local loads/stores or driver local memory.
This targeted census covers observed CuTe programs, not every Triton metadata
program or an Nsight tensor-utilization trace.

### Focused correctness

The host suite passes 86 tests with six GPU-only skips (`host-08.log`). The
physical cache/counter/model-epoch suite passes 25 tests (`gpu-tests-09.log`).
The duplicate-route independent oracle passes targeted memcheck and synccheck,
one test each, zero errors (`duplicate-*-09.log`). Seven companion engine tests
pass at the final Python source, covering deferred drain, failure, cancellation,
opt-out, competing administrative control and idle-callback progress.

`gpu-tests-08.log` retains an allocator-accounting failure: a counter replay
window observed 1,536 bytes freed after preceding graph tests, with 24 other
cases passing. The counter test now collects retired Python owner cycles and
synchronizes before taking its replay baseline. It retains strict allocation
and free-event equality during replay. Only the test boundary changes; the
frozen library hash is unchanged in `source-09.json`.

The ordinary non-cache smoke in `ordinary-smoke-mixed-02` loads the complete
checkpoint through the ordinary GPU loader, selects FlashInfer CUTLASS, captures
decode graphs and generates 16 tokens with empty additional configuration.
It is an opt-out regression check, not an arithmetic comparison with the cache's
declared W4A16 recipe. The first smoke launcher omitted Python's multiprocessing
main guard; `ordinary-smoke-mixed.log` retains that bootstrap error. The guarded
launcher passes without an engine source change.

The serving image ID is
`sha256:697f1be219540b9a5bdcd020fdd549dd0f0e848011b6630d654f43cb1782908a`.
`mixed-loaded-binaries.json` records the actual imported package path/version
and native-library hashes. The source-build attempt in `matching-build-04.log`
was deliberately interrupted after timing collection to increase compile
parallelism; `matching-build-05.log` resumes the same source and object tree.
No precompiled engine extension is admitted to that build.

### Source-matched engine qualification

The complete CUDA/C++/Rust build succeeds in `matching-build-06.log` with
precompiled engine and Rust artifacts disabled. The source-bound wheel is
`vllm-0.0.0.dev0+expertcache.1d1f870b.cu133-cp312-cp312-linux_x86_64.whl`,
SHA256 `2dacf96bc6f6f046857e077c451ba24516b4f9bc4872c97a99a45709bd3a305a`,
retained under `/models/b12x-control-build-20260919/wheels` on ripper.
`matching-wheel-identity.json` verifies the three modified engine Python files
against `source-08.json` and records all 11 native-library hashes. The libraries
loaded from `/build/installed/vllm` match those hashes exactly.

`matching-unit-tests-03.log` passes ten engine/loader tests. The source-matched
adaptive smoke, `matching-cache-smoke.jsonl`, generates 2,048 tokens through
60 checks and 448 promotions. All graph/cache addresses remain unchanged, and
token IDs match the mixed-build static C1 reference exactly. The source-matched
ordinary smoke generates the same 16 tokens as the ordinary mixed-build smoke,
with no expert-cache additional configuration. These are separate correctness
checks; they do not relabel the mixed-build C1/C4/C8 timing matrix.

Build/environment failures remain visible. `matching-build-05.log` fails because
DeepGEMM's DeepJIT dependency needs `elfutils/libdwfl.h`; installing `libdw-dev`
and `libelf-dev` in the isolated build container resolves it. The initial wheel
check therefore has no artifact to load. `matching-unit-tests-02.log` then
records a runtime collection failure for missing `cbor2` in the build-only
virtual environment. Runtime qualification uses the image's complete dependency
environment with the source-built wheel first on the import path. It does not
fall back to the image's engine extensions.

The separate registry suite still reports four passes and five MXFP6 metadata
failures in both `registry-final.log` and a pristine `b067db4` checkout
(`registry-base-b067.log`). The focused maintenance gates do not clear that
repository-wide gate. Both repositories expose zero GitHub check runs/status
contexts for the inspected starting commits; independent PR CI remains required.

### Remaining work ranked by this evidence

1. Establish batching-sensitive output repeatability for C4 healthy backoff and
   repeat the full paired matrix on the source-built wheel. The failed backoff
   comparison prevents treating cadence adaptation as qualified across lanes.
2. Reduce the approximately 14 ms worker cost of a healthy check, especially
   counter decoding, repeated observation validation and host policy history.
   A compact health observation needs an explicit consistency/overhead contract
   before it can replace the complete quiescent snapshot.
3. Evaluate cadence/short-history tradeoffs on longer stable and transition
   traffic. Backoff reduces stable cost but delays reaction; a delivered-token
   interval is not a fixed scheduler-iteration window. Device counter banks or
   decay remain unimplemented until the loss from coarse history is measured.
4. Investigate a placement-invariant small-M native schedule. The preferred
   direct path is materially faster at small M, but its numerical grouping is
   different. Ordered finalization and movement-independent arithmetic remain
   required; no compute recipe is changed for timing.
5. Retain bounded host staging and physical B300 qualification as separate
   gates. Copies take hundreds of microseconds per layer transaction, while
   healthy host control takes milliseconds. Concurrent replacement and further
   DMA work are not the measured priority.

The timed matrix contains 43,008 generated tokens and 6,384 promotions across
21 fresh workers. One C4 backoff arm fails cross-arm token equality; the other
20 arms account for 40,960 matched tokens and 6,080 promotions. Earlier receipts,
the failing arm and the separate source-built smoke retain their own identities.
No adaptive default, distributed local-maintenance path, model-quality result
or physical SM103 performance claim follows from these tests.

## C4 cadence causality and controlled admission (2026-09-20 UTC)

The inspected public heads are b12x `6343ecc9b479425149fbc689b9b0732eed406303`
and vLLM `3e45b530e58186046383e7294e611c2f6bf5cfb8`. Master/main remain
`0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68` and
`47ccf6c57d92f03630ebcbad3809450545825488`. The b12x branch contains master;
this investigation requires no rebase or companion engine change.

Evidence root: `/home/jasonc/b12x-c4-evidence-20260920`; remote receipts:
`ripper:/home/jasonc/b12x-c4-results-20260920`. Frozen sources use
`/home/jasonc/b12x-c4-source-NN` on ripper. Original September 19 receipts are
unchanged. The [cadence guide](expert-cache-cadence.md) specifies diagnostic
scope, the isolated cause and the comparison contract.

### Causality receipts

- `original-backoff-c4-r1.jsonl` reproduces the retained failure before code
  changes: requests 4/5/6/7 first differ at token indices 36/12/5/14, and all
  sixteen output sequences match the original failing receipt. No promotion
  precedes the stable-interval divergence.
- `noop-fixed-c4-r1.jsonl` and `noop-backoff-c4-r1.jsonl` use global movement
  budget zero. Both match static. These non-reproductions are retained.
- CPU-traced streamed-admission runs also match static. Their prefill grouping
  varies; tracing perturbs admission timing. They are not timing evidence.
- `matched-static-together-c4.jsonl` uses the complete source-built engine,
  static placement, no observer and no maintenance. Group admission alone
  reproduces the original four changed output sequences.
- `matched-native-{streamed,together}-c4.jsonl` exercises ordinary non-cache
  ModelOpt/FlashInfer CUTLASS serving. All sixteen requests change output under
  the admission control. This is the ordinary A4 recipe, not an A4/A16 parity
  assertion. It demonstrates a scheduling-sensitive numerical property outside
  the cache path.
- `router-probe-{streamed,together}-c4-trace.json[.pt]` retains exact model
  inputs, positions, slot mappings, attention metadata, selected module
  activations, actual expert IDs/weights, full logits and residency map hashes.
  Source archive 05 adds the targeted router probes. Unchanged maps and pointers
  are verified within each static run.
- `router-replay-layers01.json` reproduces the recorded router outputs with
  checkpoint weights and the ordinary Torch BF16 linear operation. Fixed-shape
  repetitions are exact. Disabling reduced-precision BF16 reductions removes
  the observed cross-shape difference and matches BF16-rounded FP64 dots in
  this diagnostic. No serving numerical default is changed.
- `shape-invariance-02.json` checks checkpoint layer 12. The separate
  `actual-route-shapes-layer1.json` uses the actual affected serving activation,
  IDs and route weights. Both pass 48 exact row comparisons across M=1..128,
  duplicate routes and reversal, using native whole-K and prepared cache
  execution. The actual row also exactly matches mixed-residency serving.
- `matched-noop-backoff-together-c4` matches static group admission across all
  516 CPU execution signatures, all 42 selected device records and all 2,048
  output IDs. No promotions occur. The device comparison includes full logits,
  not only greedy choices.

The first component difference is the ordinary BF16 gate projection in a
57-row versus 15-row mixed prefill/decode batch. Layer 0 attention and routed
MoE output remain exact. Layer 1 has identical MoE inputs and selected IDs but
route-weight differences up to 0.0012212172, then 988 changed MoE output elements
(maximum absolute difference 0.00048828125). Router rounding propagates to
later greedy choices. The original untraced receipt has no batch metadata;
controlled reproduction establishes a sufficient causal mechanism without
inventing missing historical observations.

### Health diagnostics and rejected changes

`idle-floor-and-layer-pressure-c4.jsonl` uses source archive 06 and zero movement.
Median scheduler-only RPC is 0.684 ms; an existing 8-byte counter read is
0.0157 ms; a full 51,472-byte snapshot is 0.812 ms (D2H 0.0164 ms, host decoding
0.763 ms); complete idle no-op maintenance is 6.79 ms. These are lower-bound
idle diagnostics. They do not implement a compact health signal or include a
busy scheduler's remaining submitted iteration.

The paired trace records per-layer pressure without an additional device
snapshot. Healthy global windows at 3.3–13.8% cold already have 1–17 layers above
15%; positive unprotected score gaps occur in 45–48 layers. A worst-layer or
positive-gap-only trigger would cause unnecessary work on this corpus, so it
is not adopted. After the workload transition, 47–48 layers exceed 15% and the
no-movement global cold fraction rises to roughly 36–51%. Unique step-touch
statistics cannot be inferred from selection counts.

Natural request-group boundaries provide idle control opportunities, but this
corpus has only four groups at C4 and two at C8. Waiting exclusively for those
boundaries would detect drift only after a large fraction of code traffic has
completed. The idle measurement quantifies possible control savings; it does
not establish a generally responsive boundary-only policy. Device summaries,
rolling counter banks and automatic trigger tuning remain deferred.

The complete source-built wheel and its eleven loaded native libraries retain
SHA256 identity `2dacf96bc6f6f046857e077c451ba24516b4f9bc4872c97a99a45709bd3a305a`
for the wheel. `source-built-runtime-binaries.json` records the imported package
and individual library hashes. Traced runs and source-built timing runs are
separate. The original reproduction alone uses the retained mixed-build
source-07 environment.

Failures are retained. The first focused GPU bundle omitted `scripts/`, causing
one helper-import failure after twenty tests passed; the source archive is
corrected for the rerun. Some completed ordinary and cache workers print the
existing ignored `AsyncLLM.__del__` interpreter-teardown `TypeError` after
normal shutdown. Those logs remain visible; no shutdown fix is attributed to
this cache investigation. The known MXFP6 registry failures and independent PR
CI gate remain separate and unresolved.

### Controlled source-built cadence matrix

`qualified-{static,observe,maintenance,backoff}-c{1,4,8}.jsonl` uses archive 05,
the complete source-built companion wheel, no trace observer, and acknowledged
group admission in all arms. `qualified-matrix-summary.json` binds every raw
receipt by SHA256 and retains latency distributions, reaction timestamps,
movement, pressure and stage timings. `analyze_matrix.py` checks token equality
within each concurrency group and unchanged graphs/pointers. The twelve runs
complete 24,576 tokens and 2,016 promotions.

Overall static/fixed/backoff tok/s is 52.37/62.01/59.44 at C1,
79.60/91.83/87.08 at C4 and 100.18/107.06/103.15 at C8. Counters-only is
51.99/79.48/99.97. Backoff stable overhead is approximately 2.2%/1.3%/1.9%,
but first code-pressure detection takes 5.78/5.23/5.04 seconds versus fixed
0.99/0.81/1.13 seconds. The stable/transition split and unfavorable latency
tails are retained in [cadence qualification](expert-cache-cadence.md).
Controlled group admission changes the fixture relative to historical streamed
admission; absolute rates are not pooled across those fixtures. Dynamic clocks,
one corpus and some concurrent diagnostic CPU activity limit small-delta claims.

Final focused validation is 64 host tests passed, 21 prepared-cache/counter GPU
tests passed in 55.12 seconds, and 10 source-built companion tests passed in
8.92 seconds. The earlier missing-script bundle failure is not deleted. GPU
validation archive 07 is distinct from timing archive 05; final CLI/doc cleanup
does not overwrite either source identity. No core kernel, preparation contract,
engine default or production cadence changes. The additional evidence supports
configurable experimental cadence, not a universal adaptive default.

## Read-only routing health experiment (2026-09-20 UTC)

The starting sources are b12x `173d0746f80d317ee8b2ac9d4d463716fe065849`,
containing master `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`, and companion
`3e45b530e58186046383e7294e611c2f6bf5cfb8` with main
`47ccf6c57d92f03630ebcbad3809450545825488`. No companion engine, expert transport,
BF16 router or W4A16 whole-K compute change is needed. Evidence is retained in
`/home/jasonc/b12x-health-evidence-20260920` and on `ripper` under
`/home/jasonc/b12x-health-results-20260920`.
The only companion-main commit absent from its maintained branch removes
inherited GitHub Actions workflows. It changes no serving code and is left
separate from this experiment; the source-built companion remains unchanged.

The implemented experiment prepares a CuTe reduction over existing canonical
counters and current device maps, plus an independent per-expert health
baseline. A two-part worker utility submits a 2,304-byte asynchronous readback
for the qualification geometry and polls its completion event. It never pauses
the scheduler or advances policy history. Full maintenance remains the only
mutation protocol and rebases health after map publication. Static opt-out
allocates neither the reduction state nor its pinned result slot.

Source archive 04, SHA256
`604368dc0ad51a713ecd84cf8454a97a015d2191b9b98e216a1d4260a4389b81`, binds the
serving matrix. The complete source-built wheel remains
`2dacf96bc6f6f046857e077c451ba24516b4f9bc4872c97a99a45709bd3a305a`.
New held-out fixtures contain 48 independently authored prompts across
chat-to-code, prose-to-math and English-to-multilingual traffic. Their text does
not overlap the retained training or evaluation fixtures. Arms use controlled
admission and retain exact output equality as the numerical gate.

The original-fixture C4 health smoke matches every static output ID. It reports
approximately 4.10 microseconds for reduction and 6.62 microseconds for the
compact copy. Median client probe response is about 73.4 milliseconds while
serving continues; that is not a scheduler pause. Worker submission/poll costs
are recorded separately in the matrix. The first held-out C1 health arm uses
29 probes and two full maintenance operations during general traffic, detects
code pressure at 1.22 seconds, and reaches 68.08 tok/s versus fixed 67.61 and
backoff 63.74. This individual result does not select a universal cadence.

Initial failures are preserved. CuTe unsigned arithmetic required explicit
casts at control-flow joins, and shared allocation required a CuTe layout rather
than a shape tuple. The first SM103 compile also failed its module admission
check; the portable metadata kernel now has an explicit architecture admission
and corpus declaration. The admitted health case cross-compiles four declared
programs without initializing CUDA. A metadata census reports 87 declarations
and 245 distinct programs; this does not relabel the historical full compile
receipt as covering the modified source.

Targeted SM120 memcheck and synccheck for the initial exact-count/graph health
case both report zero errors. The expanded cache/counter suite reports 21 passes
and one skipped checkpoint case because that invocation omitted the checkpoint
mount/environment. The separate serving runs use the real checkpoint. The
ignored interpreter-shutdown `AsyncLLM.__del__` exception remains visible in
completed run logs; this experiment does not change engine teardown.

### Completed health matrix and duration experiments

The [health results](expert-cache-health-results.md) retain all five arms at
C1/C4/C8 for three independent corpora. All 45 matrix runs pass exact paired
outputs across 92,160 tokens and 8,512 promotions. Health reduces stable full
checks from 226 to twelve; stable throughput ranges from -1.3% to +1.8% of
static. Pressure response is 0.63–2.11 seconds. Health does not consistently
beat fixed checks, and prose-to-math C8 favors backoff. No losing arm is removed.

The duration and probe-interval additions bring the serving total to 56 runs,
115,712 tokens and 10,912 promotions. The short code regime exposes a backoff
loss: detection arrives near the end and complete-run cost is about 265 ms
worse than static. Health repays the preceding stable cost near four seconds
across the tested code prefixes; this is not a universal dwell threshold.
Sixteen-token probes outperform longer intervals in the one additional C4
fixture, partly with an early maximum-interval snapshot. No default changes.

Final validation passes 132 focused host tests, 99 preparation/admission tests,
25 prepared-cache/counter tests including the checkpoint, four health cases
under each of memcheck and synccheck with zero errors, and ten source-built
companion tests. The ordinary non-cache V2 smoke generates 512 tokens through
two captured graphs without cache configuration or worker extension. Its two
failed diagnostic drivers remain: missing multiprocessing entry guard, then a
callable RPC rejected by default serialization. The successful driver inspects
loaded libraries externally and changes no engine security setting. Existing
lint diagnostics are retained; new health code and tests add none.

The final SM103 health compile binds package
`a895d1e279365bbb72d6e72b84930879f9940422a5828a9ad4156b2bb47f5c97` and exports
four objects. The health reduction reports 32 registers, no stack/local memory,
1 KiB static shared memory and 6 KiB dynamic shared allocation. Source archives
04 and 06 remain the serving and GPU-validation identities, respectively;
final test formatting preserves the tested AST. No physical SM103 execution,
Gen5 transport result or independent GitHub CI acceptance is claimed.

## Deferred routing-history experiment (2026-09-20 UTC)

The inspected and rechecked sources are b12x
`f6daf48fb8484f68e62dee9d40212a3546a8cf4b`, containing master
`0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`, and companion
`3e45b530e58186046383e7294e611c2f6bf5cfb8`. Companion main remains
`47ccf6c57d92f03630ebcbad3809450545825488`; its absent CI-removal commit is
unrelated to serving. The companion source and complete source-built wheel are
unchanged. The [history contract](expert-cache-history.md) specifies the optional
counter ring; the [history results](expert-cache-history-results.md) retain the
measurements, source identities and interpretation.

Before implementation, nine immutable-source 16-token health repeats cover
three transitions at C1/C4/C8. All pass exact paired outputs and fixed-address
checks. Overall rates exceed their retained health-32 references, with stable
changes ranging from -1.65% to +3.17% of historical static. More probes also cause
24 stable maintenance operations versus twelve in the earlier health matrix.
No universal cadence is selected from these single runs.

The retained prose-to-math C8 comparison has two distinct differences: health
uses a much larger first observation window, and executes sixteen promotion
epochs versus fixed control's 21. Both repeatedly exhaust the 16-pair budget.
The implementation therefore changes history alone. It uses prepared D2D copies
and the existing coordinator's `allow_movement=False` path; movement budgets,
BF16 router arithmetic, W4A16 whole-K execution and transport remain fixed.
Wrap coalesces omitted boundaries explicitly while retaining every selection.

The depth sweep at 0/2/4/8/16 remains within a 0.5% overall range. Retained history
changes eleven of the first sixteen selected pairs in a same-endpoint replay,
and those choices have more subsequent candidate-minus-victim demand. However,
only the first maintenance can replay multiple cuts: sustained pressure already
triggers maintenance at every subsequent probe. Additional historical detail
does not recreate missed movement opportunities. A device replacement policy,
larger ring default, new completion-output engine hook and transport changes
are not justified by that result.

Failures remain in `/home/jasonc/b12x-history-evidence-20260920` and the matching
`b12x-history-results-20260920` directory on `ripper`. Source archive 02 computed
full-interval pressure/accounting after advancing deferred policy baselines,
which incorrectly limited those totals to the final window. Its no-history and
depth-4 receipts are retained separately. Archive 03 computes pressure before
replay; a regression checks all 196 selections in a multi-cut example. The
recorded health arms use an external pressure gate, but the earlier accounting
receipts are not qualification evidence for the corrected contract.

The first build verifier used a missing container mount path; the corrected
verifier checks the wheel and eleven installed native libraries. An offline
diagnostic initially overwrote its receipt-path variable while iterating
placements; both the initial output and corrected analysis remain available.
The existing ignored `AsyncLLM.__del__` shutdown exception and optional-extension
warnings remain visible in completed serving logs.

The frozen runtime digest is
`8244104e55c8f8e3df8ee983d990c979b4e87fbe4a6f49ec9ebc8d29f4ba6f16`.
Focused validation passes 275 host tests, including full-interval pressure
gating before replay when the final subwindow is entirely cold. Thirty SM120
prepared-counter/cache tests pass, including checkpoint execution. Each of
memcheck and synccheck passes
five history depths with zero errors. The SM103 health/history preparation case
cross-compiles four required native programs from that exact package. History
adds no kernel: the metadata inventory remains 87 declarations/245 programs.
These checks do not qualify Grace TMA, physical B300 execution or Gen5 transport.

The completed serving qualification contains sixty runs, 130,560 generated
tokens and 18,400 promotions, with exact paired outputs and unchanged graph/cache
addresses. The two source-02 accounting-bug receipts remain excluded. Ten
source-built companion tests pass, and ordinary non-cache graph serving produces
512 tokens without the cache extension. All sampled request intervals retain
PCIe Gen4 x16, P1 and throttle mask `0x0`; dynamic clocks still limit small
single-run comparisons.

The fixed-budget history matrix shows no consistent throughput gain. Measured
complete observation windows expand from 48 decode tokens under fixed control
to 64 under health-32 in the targeted math-C8 case. Sixteen-token probes restore
more movement opportunities. A separate history-disabled control increases only
the global allowance from 16 pairs/64 MiB to 32 pairs/128 MiB: math throughput
rises from 60.01 to 64.57 tok/s, with the same sixteen maintenance operations and
twice the promotions. This supports movement capacity as a stronger limitation
than history depth in that case; it does not change the default budget.

Mixed traffic improves overall throughput but exposes a return-to-general loss:
static reaches 185.42 tok/s, health 143.70 and history 143.00. Most return checks
remain below the unchanged routing-pressure threshold. History cannot correct a
movement decision that the health gate declines. Short math regimes also show
late, small payback margins, and history delays measured payback in the retained
duration controls. The result report preserves these losses, completed and
right-censored promotion lifetimes, raw timing distributions and fixture hashes.
Larger history defaults, a completion-output engine hook, asynchronous fills and
arithmetic changes remain deferred.
