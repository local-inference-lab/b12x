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
