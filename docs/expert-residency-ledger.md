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

The final package source SHA256 is
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
Torch `2.13.0+cu130`, CUDA `13.0`, CUTLASS DSL `4.6.2`, Triton `3.7.1`.
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
5. Add out-of-band telemetry only after demonstrating negligible overhead on
   the real route. Static usage-aware profiles remain the baseline.
