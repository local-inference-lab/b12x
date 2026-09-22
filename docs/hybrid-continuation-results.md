# Hybrid inference continuation

Status: **implemented, with bounded SM120 qualification below**. Whole-stack
distributed zero-error sanitizer acceptance remains **unqualified**. This
experiment preserves the immutable
b12x qualification point `70f62acd825343d9c1f89128be69c928dafd6a61` and companion
vLLM `7e3471fc0feb58a264433fc78ddf0a30ad3228a1`. The continuation branch is
`work/sm103-hybrid-continuation`. Results in
[the frozen report](hybrid-inference-results.md) remain attached to their
original source exports. No production policy or numerical defaults change.

The compact real-checkpoint fixture passes memcheck and synccheck. The full
512-expert TP2 single-layer fixture completes all checks under memcheck, with
only explicitly attributed NCCL initialization diagnostics; it remains a failed
whole-stack run. Matched prefill controls identify cold routed GEMMs as the main
execution lead, while larger prepared capacity reduces the measured C1 TTFT
from 12.78 to 7.95 s. TP2 exchange analysis finds both eviction harm and
substantial maintenance cost even with zero movement.

## Source and execution environment

Both requested working branches and remote heads matched the qualification
points and had clean worktrees at inspection. The inspected b12x master is
`b631ac19d2abcbec85947f423c844087fdb1fb84`, with merge base
`0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`. Companion main is
`47ccf6c57d92f03630ebcbad3809450545825488`, with merge base
`e12b91b032daed2afc34d77cca20902cef957b3c`. Neither qualification branch is
rebased or rewritten. The companion source and complete wheel are unchanged.

The evidence bundle is `/home/jasonc/b12x-hybrid-continuation-20260922` on ripper.
It contains immutable source exports, commands, rank-local logs, sanitizer
summaries, completion receipts and failed attempts. The complete companion
wheel SHA-256 is
`0222927943a77df15db945c62c856c258414d5411bfc7e3444711ff91f79f0a2`;
workers verify loaded Python and native libraries against its artifact manifest.

Physical tests use the two SM120 RTX PRO 4000 GPUs, UUIDs
`GPU-47363510-b87a-13a5-4824-2542e97df76c` and
`GPU-cc109c01-9756-d0db-21ea-f1825d3f963f`. The container image is
`sha256:697f1be219540b9a5bdcd020fdd549dd0f0e848011b6630d654f43cb1782908a`,
with driver 580.173.02, CUDA 13.3.73, CUTLASS DSL 4.6.2 and Compute Sanitizer
2026.2.1.0, build 38334959. The container loads the CUDA compatibility driver
`libcuda.so.610.43.02` (SHA-256
`0f77bc6df671f3933088cc2bef2fee7f68c96eebddc2cc569f973b422e396909`);
580.173.02 is the host kernel driver. Existing checkpoint files and profile identities are
reused; no model download occurs. GPU work is serialized with the existing
qualification lock and rejects an overlapping compute process.

The model is `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4`, immutable revision
`8fb2682f136cf94d932a498f18cb1e428832a912`, with complete checkpoint fingerprint
`f22fdcef6ae16e9a85415e35ec55069ba7ef7eab8220f48343747fc2adb4ec2e`.
The unchanged routed recipe is BF16 activation, native NVFP4 whole-K W4A16 and
weighted ordered BF16 output reduction. The ordinary W4A4 control remains a
separate deployment recipe. All physical results here are SM120 on ripper's
PCIe Gen4 x16 links; none establish Gen5 or Grace performance.

## Sanitizer isolation

`scripts/qualify_hybrid_sanitizer.py` executes one bounded ladder level. Each
rank has its own sanitizer command, application log, tool log, loaded-library
hashes and completion receipt. The parent requires all rank exit codes,
completion receipts and zero-error summaries. A timeout is incomplete even when
the partial log contains no error. Filtered runs have component scope and can
never satisfy whole-program acceptance. Known NCCL diagnostics are classified,
not suppressed or converted into passes.

The real-layer tests retain actual checkpoint weights, actual shared MLP and
sigmoid gate, independent routed arithmetic, resident/mapped/mixed placements,
reordered and duplicate routes, same-graph promotion, fixed pointers and replay
allocation checks. The optional CPU oracle preserves BF16 arithmetic boundaries;
a real-weight test compares it with the CUDA oracle. Validation-only output
comparisons execute on the CPU with their existing tolerances. Test-only TP
route agreement and barriers use the engine's CPU control group. Model
reductions continue through the production GPU collective and captured graph.

The compact mode retains 16 real experts selected by the checkpoint router and
an explicit compact-to-canonical ID table. Its full hidden/intermediate geometry
and top-k=10 are retained. It is not a 512-expert sanitizer claim.

| Control | Result | Elapsed | Scope |
| --- | --- | ---: | --- |
| CUDA arithmetic and captured graph | Zero-error memcheck, completed | 6.08 s | Whole program |
| NCCL 2.31.2 initialization only | 80 diagnostics per rank; both complete | 37.28 s | Failed whole program |
| NCCL 2.30.7, consistent preload | 40 diagnostics per rank; both complete | 29.22 s | Failed whole program |
| PyNCCL collective | Both ranks complete; initialization diagnostics | 39.60 s | Failed whole program; one summary truncated by the initial launcher |
| Production TP collective and captured graph | Both ranks complete; initialization diagnostics | 197.93 s | Failed whole program; one summary truncated by the initial launcher |
| Compact checkpoint, CPU oracle, CUDA assertions | Deadline in the second route case | 600.37 s | Incomplete memcheck |
| Compact checkpoint, CPU assertions | Deadline in the third route case | 600.37 s | Incomplete memcheck |
| Compact checkpoint, CPU assertions | Zero-error synccheck, completed | 244.31 s | Whole program, compact geometry |
| Compact checkpoint, CPU assertions, 900 s deadline | Zero-error memcheck, completed | 614.43 s | Whole program, compact geometry |
| Full TP2 layer, filtered kernels | Deadline after loading and route agreement, during preparation | 900.32 s | Incomplete component memcheck |
| Full TP2 layer, one resident expert | Rank 0 import subprocess abort; peers retired | 235.19 s | Failed whole program |
| Full TP2 layer, one resident, explicit existing driver path | Both ranks complete; exactly 120 NCCL probe diagnostics per rank | 644.74 s | Failed whole program; no unclassified diagnostics |
| Compact checkpoint, observed-expert promotion | Zero-error memcheck, completed | 612.19 s | Whole program, compact geometry |
| Compact checkpoint, observed-expert promotion | Zero-error synccheck, completed | 223.31 s | Whole program, compact geometry |
| Full TP2 layer, one resident, observed-expert promotion | Both ranks complete; exactly 120 NCCL probe diagnostics per rank | 637.16 s | Failed whole program; no unclassified diagnostics |

The launcher at `ad37bc3e6618f1c3b258ee4ff4f0f90c02dbc266` waits for all
instrumented rank processes to flush their summaries before returning failure.
Earlier truncated logs remain in the bundle. The isolated NCCL 2.30.7 attempt
that changed only `VLLM_NCCL_SO_PATH` failed communicator destruction while Torch
also loaded the default library. It is retained as a mixed-library failure;
the consistent-preload result is the version comparison.

The one-resident attempt fails before model initialization: a Triton import
launches `/sbin/ldconfig -p`, which aborts with `malloc(): unaligned fastbin
chunk detected`. Its zero-error sanitizer summary does not qualify execution: no
rank completes the checkpoint fixture. The remaining rank cannot complete
rendezvous and is explicitly retired. The launcher now recognizes an exited
application without a completion receipt and retires its peers, while allowing
completed ranks with sanitizer errors to flush all summaries. Host tests cover
that distinction. This import failure remains separate from NCCL image probes.
A CUDA-plus-subprocess control completes 20 `ldconfig` invocations under
memcheck, so the abort is not reproduced by that minimal fixture. Selecting the
already loaded compatibility-driver directory through `TRITON_LIBCUDA_PATH`
avoids the discovery subprocess in the completed TP2 retry. It changes neither
the loaded driver identity nor sanitizer error handling.

The completed distributed fixture covers all 512 logical experts, one resident
slot, real routed/shared arithmetic, route permutations, a captured production
collective and explicit release. Every reported error is an attributed NCCL
initialization probe. This is bounded production-component evidence, while the
whole-stack result remains failed. It does not qualify the deferred full
512/256/1-placement, three-layer distributed sanitizer matrix.

Inspection also finds that the historical real-layer promotion selected a fixed
expert without requiring it to occur in the graph's routes. The continuation
fixture requires an observed cold candidate. Historical arithmetic and sanitizer
receipts remain unchanged; their post-promotion assertion alone does not prove
that the graph read the moved expert. Qualification of the strengthened fixture
is reported separately.

At `64a02486f681655c3ff4b7d5d5bd3517aaa3b9af`, the strengthened TP1 suite passes
all five real-checkpoint tests in 29.67 s, including layers 0, 24 and 47. The
CPU/CUDA arithmetic-reference comparison passes within its declared tolerance.
This unsanitized result is separate from the instrumented gates below.

The strengthened compact fixture also completes memcheck at `64a02486` in
612.19 s with exit 0, a zero-error summary and an explicit release receipt. Synccheck also passes in 223.31 s with a zero-error summary and
explicit release. Both runs promote compact expert 9 with one observed route hit; the canonical checkpoint
ID table remains in the rank log. This is the instrumented same-graph
observed-promotion gate, separate from the historical fixed-candidate result.

The first multi-layer TP2 wrapper completes layer 0, including observed-expert
promotion and release, then stalls on both ranks while recreating the
distributed group for layer 24 in the same worker processes. Its stack dumps,
last-progress records and explicit termination remain in
`active-promotion-tp2-real`. It is a failed harness run, not three-layer
qualification or a normal-shutdown pass. The wrapper at `d51fe898` requires one
layer per fresh distributed job and rejects a multi-layer TP invocation before
CUDA initialization. A host test covers that ownership boundary. The individual
layer fixtures still exercise all three placements within their one process
group; no production lifecycle behavior changes.

This invocation attempted in-process process-group reconstruction, beyond the
fresh-worker engine lifecycle in the frozen serving qualification. PyTorch
[documents this reconstruction as unsupported/untested without external synchronization](https://docs.pytorch.org/docs/stable/distributed#reinitialization).
The observed stall is consistent with that boundary; its precise internal
race is not claimed to be fixed. Independent layer jobs preserve every
arithmetic, placement, graph and release assertion.

The three fresh jobs at `d51fe898` pass layers 0, 24 and 47 on both physical
ranks, with 512, 256 and one resident expert per layer. Maximum absolute error
after the TP reduction is respectively 0.00003052, 0.00004578 and 0.00007629
against the independent recipe reference. Both ranks promote the same observed
logical IDs, respectively 502, 506 and 501. Each selected expert has an observed
route hit in the replay fixture. Exact placement parity, actual shared-expert
composition, allocation-free replay and explicit release pass at every
placement. These complete, unsanitized jobs are separate from the instrumented
single-layer gate.

The strengthened TP2 memcheck at `d51fe898` completes in 637.16 s. Both ranks
perform the observed logical expert 502 promotion and same-graph replay, pass
arithmetic/shared-composition assertions, and emit release and completion
receipts. Each rank exits 99 because its sanitizer reports exactly 120 NCCL
initialization diagnostics: 20 symmetric-kernel compilation messages, 20 JIT
information messages and 80 unavailable-image API errors. All are accounted
for; no unclassified diagnostic is present. The acceptance status is **failed**,
not a filtered pass or a zero-error distributed claim. The completed real-layer
execution provides evidence that these initialization probes did not conceal a
later failure in this bounded fixture.

### NCCL diagnostic attribution

The loaded NCCL 2.31.2 library has SHA-256
`d028ea782ce1798e6ad751d1e14f4b4516a8211a6289579a92f3bcde5e634a79`.
Its ELF inventory contains 56 SM120 and 20 `sm_100f` cubins. All 20 failing FP8
symmetric-kernel symbols from the isolated initialization log occur exclusively
in the `sm_100f` cubins. These are unavailable images on the tested SM120 GPUs.
`nccl-image-inventory.json` retains the exact symbol-to-image mapping.

The diagnostics originate in `ncclInitKernelsForDevice`, before application
collectives. The corresponding
[upstream initialization loop](https://github.com/NVIDIA/nccl/blob/7b83616df3ae082a1f32bb74c27458bfe8153a13/src/enqueue/enqueue.cc)
probes function attributes and skips unavailable images. Both consistently
loaded NCCL versions complete initialization and destruction despite these
tool diagnostics. This supports unavailable-kernel probing as their cause;
it does not establish zero-error whole-stack acceptance or excuse unrelated
memory errors. The classifier requires the NCCL initialization stack, known
headings and exact agreement with the final error count. Any unrecognized
diagnostic remains visible and prevents probe-only classification.

## TP2 exchange value

`benchmarks/moe/analyze_tp_residency_value.py` consumes the frozen per-rank
maintenance receipts. It verifies matching checkpoint/profile identities,
canonical counters, maps and generations, and reconstructs each committed
logical exchange. Only complete counter differences are analyzed. A pair's
observation ends when its promoted expert is evicted or its victim is restored.
Final transactions without a subsequent complete snapshot remain unobserved;
they are not counted as zero-hit promotions.

The first active epoch recovers 3,318 / 3,318 / 3,392 cold selections in its
immediate following window across the three retained engines. The second epoch
is poorly timed for subsequent demand: 26 / 26 / 25 of its 32 promoted experts
have no hits in the retained following windows. Those exchanges have net
selection benefits of -171 / -172 / -2,286 after counting demand for their
evicted experts. In the third engine, the third epoch is useful again, with
5,079 net avoided cold selections in the following retained window.

This is evidence of uneven movement value and eviction harm, despite a full
96-proposal backlog at every epoch. It does not support raising the pair cap.
The second epoch completes 0.774 / 0.776 / -0.049 s relative to first code
admission. Its preceding observation window mostly describes general traffic.
This aligns the low-value exchanges with the regime boundary, rather than a
copy-capacity shortage. Request labels are retrospective diagnostics only.
Selection differences are routing counterfactuals, not predicted time saved.
The final unobserved tail prevents complete-run lifetime claims.

Every recorded moving epoch proposes 96 pairs and selects 32 across 48
proposing layers, skipping 64 because of the pair cap and none because of the
byte cap. Each rank copies approximately 28.5 MB per epoch, well below the
128-MiB per-rank limit. The report preserves every promoted/victim ID, pair
lifetime endpoint, positive routing-benefit window and number of layer
generations with observed promoted-expert hits. A hit is evidence of use, not
proof of movement payback.

| Retained trial / epoch | Promoted hits | Demand for evicted experts | Net avoided cold selections | Engine maintenance wall, ms |
| --- | ---: | ---: | ---: | ---: |
| 1 / 1 | 3,374 | 913 | 2,461 | 165.73 |
| 1 / 2 | 38 | 209 | -171 | 162.73 |
| 2 / 1 | 3,374 | 911 | 2,463 | 159.34 |
| 2 / 2 | 37 | 209 | -172 | 158.26 |
| 3 / 1 | 3,597 | 1,059 | 2,538 | 135.88 |
| 3 / 2 | 63 | 2,349 | -2,286 | 132.22 |
| 3 / 3 | 5,135 | 56 | 5,079 | 132.35 |

The final epoch of each trial has no following complete snapshot and is omitted
from this lifetime table. Pair lifetimes can span overlapping windows; summing
this table would not yield complete-run cold savings against the initial map.

The retained adaptive engines take 0.843 / 0.788 / 1.069 s longer than their
paired static controls. Their complete maintenance operations occupy
0.492 / 0.476 / 0.531 s. Those durations already include scheduler drain; drain
is not added again or described as entirely idle. These observations implicate
both control cost and uneven movement value. They do not isolate the remaining
wall-time difference causally. All 128 health assessments per engine are
healthy. The maximum-interval trigger follows from those assessments and the
configured control logic; the historical maintenance rows do not explicitly
record that trigger field.

## Prefill investigation

`benchmarks/moe/hybrid_prefill_operator.py` compares complete prepared W4A16
cache graphs under all-resident and all-selected-experts-cold placement. It
holds checkpoint bytes, recorded activations, logical IDs, route weights and
numerical recipe constant. It checks exact tier equality, an independent
arithmetic reference, fixed pointers and no replay allocation before timing.
Alternating CUDA-event samples and separate profiler traces retain different
measurement scopes. Lengths beyond the retained 64-row fixture repeat those
rows and are explicitly operator-scaling controls, not natural prompt fixtures.

The source at `8d67d792` passes the controls at each tested capacity. These are
six alternating CUDA-event samples from one process, with 20 graph replays per
sample, rather than independent serving-engine trials:

| Rows | Resident graph, ms | Mapped graph, ms | Mapped / resident | Mapped μs / row |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0534 | 0.8106 | 15.19× | 810.65 |
| 4 | 0.1276 | 2.9115 | 22.81× | 727.87 |
| 16 | 0.3300 | 7.6313 | 23.13× | 476.95 |
| 64 | 0.7934 | 21.1451 | 26.65× | 330.39 |
| 128 | 0.8650 | 24.1072 | 27.87× | 188.34 |
| 256 | 1.0948 | 34.3469 | 31.37× | 134.17 |

The actual 64-row fixture selects 227 distinct experts in 640 routes: median
two rows per active expert, mean 2.82, with 70 experts selected only once.
Repeating it four times retains the same 227 experts and raises the mean to
11.28 rows per expert. `prefill-route-occupancy.json` retains all canonical
counts. This concentration change helps explain why the repeated-row operator
control exposes reuse opportunity; it must not be mistaken for the expert
working set of 256 independently routed natural tokens.

At 64 rows, the separate mapped profiler trace attributes 21.197 ms to the
fused routed kernel and about 0.023 ms to packing, remapping and output
reduction. The matched control contains no scheduler, attention, shared MLP or
TP collective, so these cannot explain that operator-level gap. FC1, activation
and FC2 share the persistent fused kernel; its duration does not separately
measure those phases. The larger repeated-row control improves reuse, but is
not evidence of natural-prompt throughput by itself.

A separate Nsight Compute diagnostic captures both fused launches in each tier
range. The mapped range contains an empty resident launch before its active cold
launch; the initial one-launch diagnostic measured that empty launch and is
retained without attributing its counters to cold execution. The corrected
two-launch capture measures the active mapped kernel at 20.97 ms and the active
resident kernel at 0.764 ms. Both use 144 registers per thread, 54.27 KB dynamic
shared memory, 256 threads per block and one persistent block per SM, with no
reported local-memory spills. Resident execution reaches 84.12% of device DRAM
throughput. Mapped execution reports 51.12% long-scoreboard and 42.93% barrier
warp stalls. Those are sampled warp-state fractions, not fractions of end-to-end
wall time; the reported device-memory throughput is not PCIe bandwidth.

An additional diagnostic enables `CUTE_DSL_LINEINFO=1`, which remains part of
the raw compilation identity. It passes the same operator arithmetic/graph
checks and retains separate objects and profiler output. Source-correlated SASS
PC samples locate 69.59% in FC1/setup, 1.57% at FC1 completion/activation entry,
28.80% in FC2/epilogue, 0.023% in the activation body and 0.016% in its following
barrier/output-clear region. The analysis records the reviewed instruction
boundaries and source-file hash in `prefill-pc-attribution.json`. Both GEMMs
inline the same helper, so a Python helper name alone cannot distinguish them.
The profiler-only harness at `8d67d792` verifies arithmetic and captured
execution, but skips the timing loop that checks allocation counters and
pointers. Its raw `no_replay_allocation` and `fixed_pointers` flags therefore
overstate that diagnostic mode's checks; those fields are not acceptance
evidence. The unprofiled timed operator runs do execute both assertions.
`5332361c` makes an explicit frozen replay check unconditional before either
mode and retains this limitation of the preceding diagnostic receipts. Its
real 64-row diagnostic rerun passes exact tier equality, the independent
arithmetic reference, frozen kernel resolution, fixed pointers, zero replay
allocations and explicit preparation release. No profiler timing is relabeled
from that verification run.

These samples locate the expensive phases; multiplying their fractions by
kernel duration would not establish separate FC1/FC2 wall times. The complete
mapped kernel takes 21.39 ms in this distinct diagnostic.

Profiling uses a separate container with `CAP_PERFMON`, following NVIDIA's
[documented container profiling permission](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters).
The failed attempt without that capability remains in the evidence bundle. No
host driver permission or system limit changes. Profiler replays are separate
from the CUDA-event timing samples and headline serving.

The isolated operator excludes TP communication and shared/dense execution.
The retained whole-model TP2 phase traces include NCCL waiting as well as
collective execution; their rank-local elapsed time is not a measurement of
network transfer cost. This pass does not claim that the TP1 FC1 attribution
quantifies every TP2 wait or the ordinary W4A4 backend's internal arithmetic
cost. The matched operator identifies a W4A16 cold-service deficit independent
of those components, while the serving chunk controls measure scheduling
effects separately.

For serving chunk controls, the existing capacity planner admits one common
capacity-256 placement per topology, reconstructed from the original calibration
counts. TP1 has 165–166 residents per layer (32.23% of payload); TP2 has
374–375 (73.11%). Every arm retains 2 GiB KV, 512 MiB graph reservation and
1 GiB device safety. The workspace increase is charged before choosing resident
counts. These placements are distinct from the frozen maximum-capacity decode
profiles. No evaluation counts influence their rankings.

The capacity-256 profile identities are
`b71639e0c49e74a1af0bf11fc336b42ae75217f7ec7c48a7614923cfa52a41c1`
for TP1 and
`f109ad7839b271a383e4bc0a7d025dd14628e13ba42fdd0aebdc4b4eb1491aa3`
for TP2. The expert envelopes are 16,339,686,188 bytes and 18,093,537,228 bytes
per rank respectively. They include private preparation storage; these numbers
are not resident payload sizes. The capacity-plan receipts retain the complete
payload, workspace, host-source, backing and reservation arithmetic.

The retained natural fixture has 1,522 prompt tokens per request and four greedy
output tokens. The existing phase summarizer reports the maximum TP-rank model
duration, never the sum of rank durations. A mixed iteration contributes neither
its tokens nor its duration to pure-prefill throughput.

| TP / concurrency | Capacity / per-request chunk limit | Pure prompt tokens | Pure model prompt tok/s | Client TTFT, s |
| --- | --- | ---: | ---: | --- |
| TP1 / C1 | 64 / engine default | 1,522 | 119.73 | 12.781 |
| TP1 / C1 | 256 / engine default | 1,522 | 192.51 | 7.948 |
| TP1 / C4 | 64 / engine default | 5,521 | 115.80 | 13.089, 25.913, 38.780, 51.147 |
| TP1 / C4 | 256 / engine default | 3,793 | 180.27 | 8.657, 16.630, 24.614, 31.781 |
| TP1 / C4 | 256 / 64 | 6,088 | 203.34 | 30.023, 30.023, 30.023, 30.023 |
| TP2 / C4 | 64 / engine default | 5,521 | 356.83 | 4.315, 8.624, 12.935, 17.252 |
| TP2 / C4 | 256 / engine default | 3,793 | 745.47 | 1.932, 3.813, 5.700, 7.571 |

These are one fresh engine per cell, with diagnostic CUDA events enabled, not a
repeated throughput qualification. Capacity 64 uses 24 prompt iterations per
request; 256 uses six. The default C4 arms have nine mixed iterations, so their
pure-prefill rates cover different subsets. The C1 comparison has no mixed
iterations and covers identical prompt tokens. Its pure model duration falls
from 12.712 s to 7.906 s; client TTFT falls by 37.82%. Matching output IDs do not
make cross-recipe W4A4/W4A16 comparisons arithmetic equivalence claims.

Limiting each request to 64 tokens within a 256-token iteration admits all four
requests concurrently. It improves group completion time slightly over the
default 256-token scheduling, but delays the first response from 8.657 s to
30.023 s. Aggregate prompt throughput and individual TTFT therefore favor
different scheduling choices. The explicit harness option
`--prefill-chunk-tokens` leaves the engine default unchanged when omitted.

The capacity increase also reduces the admitted resident fraction because its
workspace is larger. The common placement isolates chunk execution in this
comparison; it does not establish that capacity 256 is a universally better
decode configuration. No cache policy, weight arithmetic or production kernel
changes are included.

One fresh ordinary selective-UVA control uses the same capacity 256, C1 prompt,
2-GiB KV reservation and context 2048. FlashInfer CUTLASS W4A4 processes all
1,522 prompt tokens in 3.286 s of pure model time: **463.21 prompt tok/s**, with
**3.328 s TTFT** and clean shutdown. Its six chunk sizes match W4A16 exactly.
The W4A16 control measures 192.51 prompt tok/s and 7.948 s TTFT. Thus larger
chunks help both paths and do not close the deployment gap. The ordinary arm
retains its 26-GiB selective routed-parameter offload configuration; it has no
learned expert map. Its four output IDs happen to match, but this remains a
cross-recipe, differently arranged residency comparison, not a W4A4/W4A16
arithmetic-equivalence or isolated kernel-efficiency result.

## Compact decode regression

The source export at `8d67d792` repeats one fresh static/adaptive engine pair
per topology using the frozen decode profiles and 4,096 generated tokens per
arm. Both pairs preserve exact output IDs, fixed cache/graph addresses and
explicit release of every rank's mapped backing, CPU sources, graphs and
pending health owners. The package and companion are unchanged; these are
source-bound regression samples, not performance improvements from new kernels.
Explicit shutdown takes 15.86/16.04 s for TP1 static/adaptive and 13.84/14.13 s
for TP2 static/adaptive. Every release receipt reports zero mapped bytes, CPU
expert-source bytes, graph owners and pending health work. No forced kill or
ignored `AsyncLLM` destructor exception occurs in these completed serving runs.
Shutdown is outside serving throughput and retained separately.

| Topology / mode | Overall tok/s | General tok/s | Code tok/s | Promotions | Engine maintenance wall, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| TP1 static | 59.32 | 61.45 | 57.40 | 0 | 0 |
| TP1 adaptive | 69.32 | 78.66 | 62.06 | 3,104 | 11,680.52 |
| TP2 static | 201.64 | 203.21 | 200.98 | 0 | 0 |
| TP2 adaptive | 200.71 | 204.10 | 198.36 | 96 | 469.61 |

TP1 adaptation gains 16.86% in this pair. TP2 loses 0.46%; its static sample is
slower than the frozen 208.41 tok/s mean, so the smaller deficit is not evidence
that a runtime change fixed adaptation. Both TP2 arms have P1 serving samples,
Gen4 x16 links, approximately 2,450/2,437 MHz mean SM clocks by GPU and similar
power. Those observations do not identify the source of trial variation. The
frozen three-pair result remains the repeated comparison.

| Arm | TTFT p50 / p95, ms | Delivery gap p50 / p95 / p99, ms |
| --- | --- | --- |
| TP1 static | 517.65 / 588.97 | 65.21 / 83.56 / 89.62 |
| TP1 adaptive | 535.43 / 661.85 | 48.06 / 118.19 / 140.94 |
| TP2 static | 271.35 / 372.81 | 18.40 / 20.94 / 22.17 |
| TP2 adaptive | 274.67 / 376.77 | 18.04 / 20.85 / 22.35 |

TP1 performs 97 moving epochs and 5,511,889,152 API copy bytes. TP2 performs
three and 171,214,336 aggregate bytes across both ranks. Its complete observation
windows contain 6.08% cold selections. Static has no routing observer, and no
static cold rate is inferred. Serving wall includes the final control tail;
maintenance durations are nested within it and are not added again.

The separate TP2 zero-movement diagnostic keeps health and maximum-interval
maintenance enabled, changing only the explicit pair cap to zero. It measures
199.17 overall / 202.36 general / 196.99 code tok/s, with exact paired output
IDs, zero copies, zero promotions and clean release. Its three
`maximum_interval` operations occupy 422.15 ms, compared with 469.61 ms for
the moving arm. Removing copies and publication saves only 47.46 ms of those
measured operations in this pair of trials; approximately 90% of the moving
arm's maintenance duration remains. This is evidence that snapshot/policy/RPC
work matters independently of copy volume. It does not causally partition the
entire serving slowdown, and the modest throughput difference between these
single trials is not a policy-selection result.

All 128 TP2 health assessments remain healthy in each adaptive diagnostic.
The moving arm also records `maximum_interval` for all three operations.
TP1 records `pressure` for all 97 moving epochs. Trigger reasons are therefore
explicit in the continuation receipts, rather than inferred from the historical
configuration. No threshold, cadence or policy change is implemented.

## Default-branch applicability

The NVFP4 register-layout/grid work in `b294e69d` changes b12x's dynamic W4A4
path. The qualified cache explicitly selects whole-K W4A16, while the ordinary
W4A4 deployment control uses the maintained engine's FlashInfer backend. Its
Qwen TP2 measurements therefore do not directly qualify this cache path.

`5a53424d` adds a preparation-result rejection count for a different companion
consumer; the maintained companion does not access that field. `2f3475f0`
fixes GDN binding diagnostics and KDA recovery. The qualified Qwen3-Next lane
uses the engine's CUDA GDN path, rather than b12x GDN/KDA. `a32e45ae` adds live
packed-attention lengths and a Kimi MLA tile; it does not change the selected
Next80 attention path. No performance gain from these commits is attributed to
the frozen hybrid measurements.

The additional `6debca95` work changes repacked W4A8 execution, which is not the
selected recipe. `79a29c1b` changes autotuning trial reuse; these source-bound
reference runs disable b12x autotuning. `f818b3ab` changes tuning-cache device
identity, rather than the measured execution path. These commits are not merged
just to create a new runtime source. The continuation changes qualification
and analysis tooling while the package under `b12x/` remains identical to the
frozen qualification.

## Native SM103 boundary

No explicitly configured and authorized Station/B300 endpoint is available.
Physical SM103 remains unqualified. The existing
[native qualification launcher and ordered gates](sm103-qualification.md)
remain the execution path. No SM120, software TP or sanitizer result in this
report qualifies tcgen05/TMEM, Grace-backed TMA or native HBM/Grace residency.
The native MXFP4/MXFP8 checkpoint adapter remains dependent on those physical
gates; the SM120 ModelOpt loader is not that adapter.

## Reproduction and acceptance scope

Use the verified companion environment described in
[the hybrid reference](hybrid-inference.md), with the complete pinned Next80
checkpoint, immutable profile and artifact manifest supplied explicitly. Each
sanitizer output directory must be new. Set `B12X_TEST_NEXT80_CHECKPOINT` to the
checkpoint directory and `B12X_ACCEPTANCE_BUILD_MANIFEST` to the verified build
manifest. The ladder records actual loaded CUDA/NCCL libraries; changing a
library path does not establish that the process loaded only that library.

```bash
python scripts/qualify_hybrid_sanitizer.py \
  --stage cuda --ranks 1 --tool memcheck \
  --sanitizer "$COMPUTE_SANITIZER" --deadline 60 --output "$CUDA_RECEIPT"

python scripts/qualify_hybrid_sanitizer.py \
  --stage compact --ranks 1 --oracle-device cpu --tool memcheck \
  --sanitizer "$COMPUTE_SANITIZER" --deadline 900 --output "$COMPACT_RECEIPT"

python scripts/qualify_hybrid_sanitizer.py \
  --stage tp-layer --ranks 2 --layers 0 --resident-counts 1 \
  --oracle-device cpu --tool memcheck --sanitizer "$COMPUTE_SANITIZER" \
  --deadline 900 --output "$TP_RECEIPT"
```

The intermediate stages are `nccl-init`, `collective` and `production` with two
ranks. `layer --ranks 1 --layers 0 24 47` selects full real checkpoint layers;
the TP stage requires a separate invocation for each layer, for example
`tp-layer --ranks 2 --layers 24`, retaining all three default placements.
Run `synccheck` as a separate invocation. A `--kernel-filter` changes the scope
to component instrumentation and cannot produce `whole_program_pass`.
The explicit `TRITON_LIBCUDA_PATH` workaround in this bundle names the already
verified loaded driver directory; it is not a portable driver recommendation.

The matched operator control is reproducible with the retained tensor fixture:

```bash
python benchmarks/moe/hybrid_prefill_operator.py \
  --checkpoint "$CHECKPOINT" --routes "$RETAINED_ROUTES" \
  --rows 1 4 16 64 128 256 --samples 6 --replays 20 --output "$OPERATOR_RECEIPT"

python benchmarks/moe/analyze_tp_residency_value.py \
  "$ADAPTIVE_RECEIPT_1" "$ADAPTIVE_RECEIPT_2" "$ADAPTIVE_RECEIPT_3" \
  --output "$MARGINAL_VALUE_REPORT"
```

`prefill-operator/report.json` binds the retained route tensor hash, checkpoint,
recipe, compiled program keys and raw alternating samples. The object inventory
`operator-object-index.json` preserves exact CuTe objects and verifies them
against their compilation manifests. Profiler and serving receipts retain their
separate source exports. The bundle's `serve.sh` and per-run command manifests
record the complete serving arguments, including capacity, profile, context,
KV reservation, admission and movement settings. The external
`continuation-evidence-index.json` links completed sanitizer, serving and layer
receipts by SHA-256. `receipt-interpretation-notes.json` preserves the diagnostic
flag limitation without modifying its original receipts.

The focused host invocation is:

```bash
python -m pytest -q \
  tests/architecture/test_hybrid_sanitizer.py \
  tests/architecture/test_tp_residency_value.py \
  tests/architecture/test_prefill_controls.py \
  tests/architecture/test_checkpoint_promotion.py \
  tests/moe/test_expert_cache_capacity.py
```

Physical arithmetic qualification uses
`python -m pytest tests/moe/test_next80_checkpoint.py -q` with the supplied
checkpoint. The TP jobs use the same source-matched environment:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/comm/sanitizer_worker.py --stage tp-layer --layers 0 \
  --oracle-device cpu --output "$TP_LAYER_RECEIPT"
```

Repeat that fresh-job command with layers 24 and 47 and distinct output paths.
The portable GPU suite is the existing `scripts/qualify_expert_cache.py --tier
gpu` path. Its retained command runs `test_prepared_expert_cache.py`,
`test_routing_profile_gpu.py` and `test_residency_epoch_gpu.py`, with source,
build manifest and GPU UUID supplied explicitly. The Qwen3-30 serving runner
uses one C4 pair with 64 generated tokens per request. Every exact command and
its exit code remain in the acceptance bundle; completed receipts do not stand
in for executing the tests.

Focused host tests pass 28 cases. Independent host acceptance on
[`5332361c`](https://github.com/local-inference-lab/b12x/actions/runs/35792378638)
passes 1,215 tests, with 67 explicitly retained GPU skips, zero failures and zero
errors. Those skips provide no physical qualification. The companion remains
unchanged at `7e3471fc`; there is no Python overlay or native-wheel rebuild claim.

Portable physical SM120 acceptance at `64a02486` passes all 44 prepared-cache,
routing-health and model-wide epoch tests, with no skips. The Qwen3-30 regression
on the same source passes ordinary serving and a static/adaptive C4 pair with
1,024 generated tokens per cache arm, exact paired IDs, fixed addresses and
clean explicit release. The adaptive arm performs 384 promotions. This is a
bounded regression, not a replacement for its historical performance matrix.

## Remaining limits and next experiment

The whole-stack distributed zero-error sanitizer gate remains unqualified.
NCCL initialization diagnostics are explained and counted, but not waived. The
full three-layer distributed instrumented matrix, native SM103/Grace execution,
and native MXFP4/MXFP8 checkpoint integration remain deferred. No physical
N=3/N=4 or B300 claim follows from these results. The failed in-process TP test
wrapper and every timeout remain explicit failures.

The strongest execution lead is mapped W4A16 prompt processing. At matched
routes and graph shape, the cold graph is 26.65 times slower than the resident
graph at 64 rows; packing/reduction takes only about 23 microseconds. PC
samples place most stalled work inside FC1 and FC2. Larger prepared capacity
reduces natural-prompt TTFT substantially, but the matched-capacity ordinary
W4A4 deployment remains faster. Neither activation handling nor cache policy
is supported as the first optimization target by these controls.

The next concrete engineering task is a prepared W4A16 prefill execution
prototype that improves cold weight-tile reuse or latency hiding, starting
with FC1. Use the retained routes, identical BF16 boundaries, both resident
and mapped controls, and the capacity-256 natural prompt as acceptance inputs.
Separate direct CUDA timing from serving, charge any extra workspace before
placement, and retain the decode control. The current evidence identifies
where to prototype; it does not establish a particular tile/pipeline design
or a PCIe bandwidth ceiling. No kernel-speedup claim is made in this pass.

TP2 policy work should first test whether avoiding low-value full maintenance
can repay control cost while preserving useful epochs. Backlog alone is
insufficient, and the final unobserved demand tail must not become synthetic
zero-hit evidence. TP1's measured adaptive benefit remains intact.

## Implementation commits

These commits are descendants of the preserved qualification point. The
companion has no continuation changes. Documentation updates retain the tested
source identity for each result rather than assigning measurements to this
report commit.

| Commit | Change |
| --- | --- |
| `c112d096` | Record bounded rank-local sanitizer isolation gates |
| `8d2e804f` | Retain real-layer sanitizer progress and support a host arithmetic oracle |
| `e9636e0e` | Separate NCCL probe attribution from sanitizer acceptance |
| `240ce05e` | Evaluate checkpoint assertions without validation-only CUDA launches |
| `ad37bc3e` | Wait for every sanitizer rank and retain libraries on failure |
| `05f5d08e` | Analyze TP exchange value from complete routing windows |
| `32ff528a` | Measure matched resident and mapped W4A16 prompt operators |
| `50de3e9d` | Retain complete eviction endpoints in marginal-value analysis |
| `cd473e3a` | Account for explicit prepared capacity in placement projections |
| `d5b040bf` | Separate external profiler replays and unobserved routing demand |
| `c62bf9c6` | Declare placement scope for distributed checkpoint sanitizer controls |
| `8d67d792` | Expose bounded engine prefill chunking in the serving harness |
| `85e7a7fe` | Retire sanitizer peers when an application cannot complete |
| `18189ae4` | Retain request admission times beside offline exchange value |
| `56c304bb` | Keep checkpoint diagnostic paths outside kernel compile identity |
| `64a02486` | Require real-layer graph promotions to move observed experts |
| `d51fe898` | Require a fresh distributed job for each checkpoint layer |
| `5332361c` | Check replay invariants before profiler-only operator runs |
