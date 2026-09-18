# Expert residency engineering ledger

This ledger separates implementation evidence, compiler evidence, portable GPU
correctness, and deferred SM103 qualification. Raw receipts are retained outside
the repository at `/home/jasonc/b12x-residency-evidence-20260918` on the development
host and `/home/jasonc/b12x-residency-20260918` on the portable GPU host.

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
