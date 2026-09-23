# Prepared W4A16 cold-prefill experiments

Status: research-only, negative performance result. Two prepared mapped
execution variants preserve the qualified arithmetic after retaining the K128
accumulation geometry. Neither improves the primary real 64-row fixture. The
qualified resident, decode and serving selections remain the defaults. No
companion changes or additional checkpoint downloads are required.
Three broader W4A16 regression cases fail on both the unchanged baseline and
the continuation source; their results remain open below.

## Sources and physical scope

The baseline is continuation commit
`5247940a91ede8c71e4ef3c01df0cea487253439`. The six-shape timing arms use
`02ef4e0cf991f1a9636d8ad30e5fe6d474eaa0ef`. Preparation finalizer hardening is
separate from those measurements; its commits and validation appear below.
The immutable qualification at b12x `70f62acd` and companion vLLM
`7e3471fc0feb58a264433fc78ddf0a30ad3228a1` remains unchanged. No master merge is
part of this experiment.

The checkpoint is `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4`, revision
`8fb2682f136cf94d932a498f18cb1e428832a912`, fingerprint
`f22fdcef6ae16e9a85415e35ec55069ba7ef7eab8220f48343747fc2adb4ec2e`.
The retained layer-0 fixture has SHA256
`59c96c71ddec22ad41def9f5d3713328674a966330effd4bc0a2a40825bfcb09`.
It contains actual BF16 inputs, logical top-10 IDs and float32 route weights.
The layer has 512 experts, H=2048 and I=512.

Physical execution uses ripper GPU
`GPU-47363510-b87a-13a5-4824-2542e97df76c`, an RTX PRO 4000 Blackwell SM120
with 70 SMs, and PCIe Gen4 x16 under load. The image is
`sha256:697f1be219540b9a5bdcd020fdd549dd0f0e848011b6630d654f43cb1782908a`.
The source-built companion wheel SHA256 is
`0222927943a77df15db945c62c856c258414d5411bfc7e3444711ff91f79f0a2`;
its artifact manifest SHA256 is
`1bd9dd3f172994fee9f4ce6ae4fc4259931f6e9d04f16ad24bd77bf62ae1c87d`.
The environment uses Torch 2.13.0/CUDA 13.3, CUTLASS DSL 4.6.2,
NVCC 13.3.73, host driver 580.173.02 and the image's CUDA compatibility
library. Nsight Compute is 2026.2.1; Compute Sanitizer is 2026.2.1.

Raw evidence is retained outside the repository in
`/home/jasonc/b12x-cold-prefill-20260922` on ripper and the administration host.
Each launch retains its source export, Python-file hashes, artifact identity,
command, exit status, timestamps, GPU telemetry and logs. Operator reports
retain compiled-program identities, pointer/allocation checks and all samples.
The GPU lock serializes jobs and rejects unrelated compute processes.
These are SM120/PCIe measurements; they qualify no SM103, Grace or TP2 path.

## Route occupancy and inferred requests

At 64 rows, 640 routes touch 227 experts. Median occupancy is two routes per
expert, mean 2.82, maximum 19; 70 experts occur once. Eight-row route blocks
produce 237 expert blocks. Only ten blocks repeat an already requested expert.
Each FC1 expert contains 1,048,576 packed bytes, 131,072 scale bytes and one
four-byte global scale. FC2 contains 524,288 packed bytes, 65,536 scale bytes
and one four-byte global scale.

The whole-K schedule traverses each weight tile once per route block. The
following estimates count packed weights and block scales, excluding global
scalar loads and activation/metadata traffic. They are logical request bytes,
not PCIe measurements. `weight_schedule()` retains each expert's route count,
block count and FC1/FC2 tile requests in the report.

| Rows | Experts | Route blocks | FC1 tile requests (repeated) | FC2 tile requests (repeated) | Unique MiB | Scheduled MiB | Routes / unique MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10 | 10 | 1,280 (0) | 640 (0) | 16.875 | 16.875 | 0.593 |
| 4 | 35 | 35 | 4,480 (0) | 2,240 (0) | 59.0625 | 59.0625 | 0.677 |
| 16 | 92 | 92 | 11,776 (0) | 5,888 (0) | 155.250 | 155.250 | 1.031 |
| 64 | 227 | 237 | 30,336 (1,280) | 15,168 (640) | 383.0625 | 399.9375 | 1.671 |
| 128 | 227 | 277 | 35,456 (6,400) | 17,728 (3,200) | 383.0625 | 467.4375 | 3.341 |
| 256 | 227 | 386 | 49,408 (20,352) | 24,704 (10,176) | 383.0625 | 651.375 | 6.683 |

Rows 128 and 256 repeat the retained 64 rows. Their increasing reuse does not
represent a natural longer prompt. At 64 rows, perfect reuse across repeated
expert blocks could remove only 4.22% of scheduled weight/scale bytes. This is
not a throughput bound: it omits instruction, transaction and synchronization
cost. Reuse inside each block already occurs through shared memory; a second
expert sort cannot eliminate the predominantly unique weight demand.

## Hardware counters

The `traffic-ncu` and `pipeline3-ncu` jobs profile the same six launches: active
resident plus empty mapped, empty resident plus active mapped, and empty
resident plus active prototype. Kernel IDs 0, 3 and 5 are the active controls.
Event timing and profiler replay are separate runs.

| Counter, real 64 rows | Resident | Mapped control | Narrow two-CTA | Full-width three-stage |
| --- | ---: | ---: | ---: | ---: |
| Threads / CTA | 256 | 256 | 128 | 256 |
| Grid CTAs / planned CTAs per SM | 70 / 1 | 70 / 1 | 140 / 2 | 140 / 2 |
| Registers / thread | 144 | 144 | 145 (152 allocated) | 124 |
| Dynamic SMEM / CTA, bytes | 54,272 | 54,272 | 35,840 | 40,960 |
| Driver SMEM / CTA, bytes | 1,024 | 1,024 | 1,024 | 1,024 |
| Achieved active-warps occupancy | 16.86% | 17.36% | 16.79% | 33.89% |
| Reported local spill requests | 0 | 0 | 0 | 0 |
| System-memory fill sectors | 1 | 13,110,475 | 14,569,303 | 13,110,480 |
| System-memory fill bytes, 32 B/sector | 32 | 419,535,200 | 466,217,696 | 419,535,360 |
| PCIe read-byte counter, MB | 0 | 518.637 | 571.187 | 518.637 |
| PCIe write-byte counter, MB | 0.0179 | 123.895 | 131.237 | 123.896 |

The mapped control's inferred schedule is 419,364,864 weight/scale bytes. Its
measured system-memory sectors are close to that value. Narrowing N raises
measured sector demand by 11.13% despite unchanged logical weight bytes. Global
L1 tag requests rise from 6,691,387 to 7,385,652. These counters count requests,
not independent logical experts or per-lane load instructions. Per-instruction
address reuse was not measured. The whole-kernel counters cannot divide actual
transactions between FC1 and FC2; that division remains a schedule estimate.

Source-address analysis explains almost all of the narrow variant's extra
traffic. `_stage_modelopt_scales()` loads 16-byte vectors for N128, but eight-byte
vectors for N64. The two N64 tiles need different halves of the same sectors.
Without reuse between those tiles, inferred scale-sector demand doubles from
46,596,096 to 93,192,192 bytes. The predicted increment, 46,596,096 bytes, is
close to the measured 46,682,496-byte increment. The remaining 86,400 bytes are
not attributed. `weight_schedule()` now reports this sector estimate separately
from logical scale bytes; its unit test checks both representations. This is
address analysis supported by whole-kernel counters, not isolated FC1 PCIe
measurement.

SM120 reports this traffic through `syslts`, rather than the ordinary `lts`
system-aperture counters. The active mapped control records 13,110,474 system
read sectors and the same number of misses. The narrow variant records
14,569,285 reads and 14,569,256 misses. Thus the aggregate L2 hit rate reported
elsewhere in the profile is not evidence that mapped weights enjoy L2 reuse.
The measured PCIe counters are reported separately, at the tool's 512-byte
counting granularity; their sum is not substituted for checkpoint payload.
Metric meanings follow the [NVIDIA profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).

Sampled stall fractions change from 52.90% long scoreboard / 42.88% barrier in
the mapped control to 69.98% / 19.30% in the narrow variant. Two smaller CTAs
reduce barrier waiting but do not increase active-warps occupancy and increase
system-memory traffic. The three-stage variant doubles active-warps occupancy,
keeps measured traffic almost identical, and still loses event-timed performance.
Its sampled fractions are 53.34% long scoreboard / 25.41% barrier. More resident
CTAs are therefore insufficient; neither these fractions nor PC attribution are
independent phase wall times.

The earlier source-correlated 69.59% FC1/setup sample share remains the reason
to investigate FC1 first. These experiments do not establish separate FC1/FC2
wall times, a Gen4 bandwidth ceiling, or the benefit of a split pipeline.

## Prepared prototypes and arithmetic

`ExpertCacheConfig(cold_prefill=...)` is an explicit preparation pin. Automatic
candidate enumeration and the default remain `fused`. Both experimental values
use the existing canonical mapped source, whole-K BF16 W4A16 recipe, SiLU,
actual output route weights and deterministic ordered BF16 reduction. They
reject incompatible source/recipe/layout controls. They compile before capture;
live row counts are not added to compiler identities.

* `two_cta`: retain K128, narrow N128 to N64, use 128 threads and two cooperative
  CTAs per SM. FC1 and FC2 must share the fused CTA geometry.
* `two_cta_pipeline3`: retain K128/N128 and 256 threads; use three shared-memory
  stages instead of four so two CTAs fit. Stage count is an explicit compiled
  property of the existing GEMM helper, including its cache identity.

The mapped preparation owns both the baseline and experimental executable.
Rows below 16 use the baseline; resident execution always uses the baseline.
The threshold selects precompiled launches and introduces no graph-replay host
work. Neither variant adds an intermediate, staging allocation or background
copy system. The shared source helpers, FC1/SiLU/FC2 boundaries and ordered
output combination remain intact.

The rejected K64/N128 attempt at `cf8bc04a` changed accumulation order. The real
64-row check found 211/131,072 differing elements, maximum absolute difference
0.0001220703125. Timing stopped before any performance claim. The correction
retains K128 and changes only N or pipeline depth. Both corrected variants pass
exact resident/mapped equality for every timed shape and the existing independent
CPU oracle for the first four rows (one row in the M=1 case), with the established
atol=0.001, rtol=0.03. Finite and nonzero checks also pass. Exact tier equality
is not relaxed to those oracle tolerances.

## Operator timings

Values below are median [minimum, maximum] milliseconds from six alternating
arm-order samples, 20 graph replays per sample, following three warm replays.
Ratios are candidate time divided by control time: values above one are slower.
M=1/4 prototype rows exercise the unchanged decode launch, not a new kernel.

### `two-cta-all-rows`

| Rows | Resident ms [min,max] | Mapped ms [min,max] | Prototype ms [min,max] | Prototype / mapped | Prototype / resident |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0532 [0.0532, 0.0537] | 0.8109 [0.8108, 0.8111] | 0.8105 [0.8103, 0.8105] | 0.9995× | 15.22× |
| 4 | 0.1289 [0.1289, 0.1291] | 2.9106 [2.9097, 2.9110] | 2.9118 [2.9109, 2.9137] | 1.0004× | 22.58× |
| 16 | 0.3319 [0.3314, 0.3323] | 7.6246 [7.6223, 7.6278] | 8.6822 [8.6754, 8.7052] | 1.1387× | 26.16× |
| 64 | 0.8006 [0.8002, 0.8009] | 21.1018 [20.9461, 21.1817] | 22.2817 [22.2478, 22.3130] | 1.0559× | 27.83× |
| 128 | 0.8701 [0.8688, 0.8713] | 24.0466 [23.9954, 24.0769] | 25.5829 [25.5722, 25.5907] | 1.0639× | 29.40× |
| 256 | 1.1188 [1.1137, 1.1197] | 34.5258 [34.0615, 34.6993] | 35.4792 [35.4078, 35.4914] | 1.0276× | 31.71× |

### `pipeline3-all-rows`

| Rows | Resident ms [min,max] | Mapped ms [min,max] | Prototype ms [min,max] | Prototype / mapped | Prototype / resident |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0532 [0.0532, 0.0537] | 0.8113 [0.8109, 0.8113] | 0.8107 [0.8106, 0.8108] | 0.9993× | 15.22× |
| 4 | 0.1292 [0.1289, 0.1296] | 2.8977 [2.8960, 2.8998] | 2.9135 [2.9119, 2.9150] | 1.0055× | 22.55× |
| 16 | 0.3321 [0.3316, 0.3327] | 7.6355 [7.6278, 7.6716] | 7.7657 [7.7624, 7.7763] | 1.0171× | 23.38× |
| 64 | 0.8014 [0.8002, 0.8018] | 21.1609 [21.0898, 21.3122] | 22.6726 [22.5395, 22.8002] | 1.0714× | 28.29× |
| 128 | 0.8716 [0.8703, 0.8732] | 24.0500 [23.9901, 24.1135] | 24.9679 [24.9559, 25.0180] | 1.0382× | 28.65× |
| 256 | 1.1214 [1.1183, 1.1232] | 34.5436 [34.4856, 34.6557] | 34.0019 [33.9547, 34.0344] | 0.9843× | 30.32× |

## Memory and serving disposition

The variants add zero tensor workspace and zero pinned backing. The existing
per-tier FC1 and activation buffers remain in the planner. For the one-resident
operator arm, both control and prototype report:

| Capacity | Resident slab | Canonical backing | CPU source | Device workspace | Device metadata |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1,769,984 B | 905,973,760 B | 905,973,760 B | 25,889,684 B | 8,208 B |
| 256 | 1,769,984 B | 905,973,760 B | 905,973,760 B | 48,374,036 B | 8,208 B |

Shared-memory staging is on-chip CTA storage, not an added resident-expert
allocation. The prototype retains another compiled executable; CUDA module and
runtime storage are distinct from tensor-workspace accounting. No claim of zero
additional total CUDA memory follows from the unchanged workspace formula.

Because the tensor admission formula is identical, these prototypes do not
subtract any tensor bytes from the capacity-256 placement. The existing plan
uses a 16,339,686,188-byte expert envelope, 165/166 resident experts per layer
and 32.230655% logical payload residency, with fixed 2-GiB KV, 512-MiB graph and
1-GiB safety reservations. The profile identity remains
`b71639e0c49e74a1af0bf11fc336b42ae75217f7ec7c48a7614923cfa52a41c1`.
No prototype serving arm is admitted or timed: neither passes the required
real-64 operator performance gate. Thus there is no deployment speedup, no
prototype TTFT result and no unmeasured claim that extra module storage fits a
full-model deployment.

The retained W4A16 capacity-256 baseline is 192.51 pure prompt tokens/s and
7.948 s TTFT for the natural 1,522-token C1 prompt. The retained W4A4
selective-UVA result, 463.21 prompt tokens/s and 3.328 s TTFT, remains a
cross-recipe deployment comparison. Neither is relabeled as a result from the
prototype source.

Default serving regression controls at `e2d50cd7` reuse the complete source-built
companion and unchanged profiles. Each row below is one fresh engine, not a new
replicated performance estimate:

| Control | Retained continuation | New default control | Checks |
| --- | ---: | ---: | --- |
| Natural C1, 1,522 prompt tokens, capacity 256: pure prompt tokens/s | 192.51 | 192.07 | Six chunks: 256/256/256/256/256/242 |
| Same request: client TTFT | 7.948 s | 7.967 s | Exact four output IDs |
| Static C4 decode, 4,096 generated tokens, capacity 64 | 59.318 generated tokens/s | 59.264 generated tokens/s | Exact IDs for all 16 requests |

The new decode control measures 61.44 general and 57.31 code tokens/s over their
respective request intervals. Client TTFT p50/p95 is 514.34/581.17 ms; delivery
gap p50/p95/p99 is 65.42/83.41/89.88 ms. These controls show no material change
at this sampling depth. The complete serving wall interval includes the final
control tail; it is not reconstructed by adding stage timers.

Both runs preserve the complete prepared status, including all cache and graph
pointers and generation zero. Shutdown takes 15.85/15.89 s respectively and
finishes normally. Mapped bytes, retained CPU expert-source bytes, graph owners
and health storage are zero after explicit worker release. The residual
548,416,000 Torch-allocated bytes belong to process-lifetime engine/runtime
state, not those cache owners. No forced kill or ignored destructor exception
appears.

The natural-prompt control observes 43,486,740,480 mapped bytes and
43,487,133,696 retained CPU-source bytes. After graph preparation it has
23,450,003,968 Torch-allocated bytes and 1,103,691,776 CUDA-free bytes. During
serving the Torch peak reaches 23,488,987,136 bytes; CUDA free at completion is
1,040,777,216 bytes. This is slightly below the planner's nominal 1-GiB safety
reservation after runtime/pool growth; the reservation is not a guarantee of
that much measured free memory. No extra resident experts or prototype modules
were admitted against that remaining space. Full resource checkpoints are in
the `default-natural-c1-resources.jsonl` and `default-decode-c4-resources.jsonl`
receipts.

## Capture finalization failure and bounded fix

Sequential reconstruction exposed `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED` in
three uninstrumented attempts. The CUDA API log in `reconstruct-controls.log`
identifies `cuLibraryUnload` returning capture-unsupported before the following
kernel launch fails. A standalone three-stage process passed. Diagnostic
wrapping changed the collection timing and sometimes passed too; those passes
were not used to dismiss the repeatable failure.

CuTe `JitModule` finalization unloads its CUDA library. Python's automatic cyclic
collection can run while a prepared call builds argument descriptors inside
capture. A diagnostic that defers cyclic collection completed four consecutive
reconstructions of each variant. `PreparationSession.capture()` now serializes
its bounded capture scopes, disables automatic cyclic collection for that scope,
restores the caller's prior GC state, and collects retired cyclic owners after
capture when GC was enabled. Nested scopes and exceptions preserve that state.
This adds no explicit CUDA synchronization, replay work or suppression of CUDA
errors; library unload itself may synchronize in the driver. Live programs remain retained by the existing preparation owners.

Enter `session.capture()` before `torch.cuda.graph()` and exit after graph
capture ends. The guard is specific to capture; it does not disable collection
throughout serving. Host tests cover cyclic finalization, exceptional exit,
nesting with disabled collection, and overlapping capture threads. GPU tests
cover live row counts 1/2/4/16/63/64, repeated/invalid routes, mutated inputs,
promotion, rollback, fixed pointers and allocation-free replay with frozen
kernel resolution.

## Validation and preserved failures

The existing host acceptance runner at `5ff70d7f` passes 1,222 tests and skips
69 with explicit reasons. Independent host CI also
[passes on `e2d50cd7`](https://github.com/local-inference-lab/b12x/actions/runs/35809602648).
This later commit adds two physical production-geometry cases, which remain
skips in CPU-only CI. These are host results, not GPU acceptance.

The focused physical regression at `5ff70d7f` passes 49 tests and skips two:

```sh
python -m pytest -xq -p no:cacheprovider \
  tests/moe/test_cold_prefill.py \
  tests/moe/test_prepared_expert_cache.py \
  tests/moe/test_w4a16_tile_selection.py \
  tests/moe/test_w4a16_reference.py \
  tests/moe/test_w4a16_route_pack_warmup.py \
  tests/preparation/test_capture_finalizers.py
```

The real-checkpoint six-shape operator runs cover exact tier equality and the
CPU oracle separately from these synthetic tests. Neither path constitutes
new full-model adaptive, TP2 or distributed sanitizer acceptance.

Unfiltered synccheck at `e2d50cd7` completes four graph/promotion cases with
zero reported errors, exit code zero, in 411.58 s. These cover both experiments
at H128/I128 and H2048/I512, with E64, capacity 64 and live tails. No NCCL or
kernel exclusion filter is involved.

The first memcheck attempt is **failed**, not a zero-error pass: it exits 11
after 382.07 s despite a zero-error summary. CUDA-GDB was attached during that
run to inspect progress. NVIDIA documents that Compute Sanitizer cannot run
with other CUDA developer tools in its
[known limitations](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html#known-limitations).
This makes the attempt unsuitable for isolating a kernel defect or claiming
acceptance. Its logs and native stacks are retained; the later clean attempt
uses no debugger.

That clean retry completes both H2048/I512 variants: two tests pass in 520.89 s,
with exit code zero and a zero-error memcheck summary (522.58 s including the
wrapper). It uses the same 900-s
deadline and no kernel filter. Python's 90-s faulthandler progress dumps are
retained; they do not stop the tests. The completion receipt, rather than an
intermediate summary or stack dump, decides acceptance.

The committed capture guard completes four reconstructions of each variant
(eight preparations total) in one process. Every cycle captures live shapes,
replays changed data, promotes and rolls back before release. The separate
`real64-release` diagnostic at `e2d50cd7` also passes exact real-checkpoint tier
equality, four-row CPU reference, fixed pointers, no replay allocation and
explicit prepared/mapped-owner release. Neither run contributes timing samples
to the operator table.

The additional W4A16 suite is **not green**. Running the following complete
selection from both an untouched `5247940a` archive and the `e2d50cd7` archive
produces the same three failures, 19 passes and 226 deselections:

```sh
python -m pytest -q -rs -p no:cacheprovider tests/moe/test_w4a16_e2e.py \
  -k 'modelopt_nvfp4 or preplanned_capacity or moe_matches_oracle'
```

| Failed case | Baseline and continuation result |
| --- | --- |
| `test_tp_moe_w4a16_modelopt_nvfp4_uses_normal_nvfp4_scale_contract[relu2]` | Identical oracle mismatch: cosine 0.0316647, max absolute error 0.151123 |
| Same scale-contract case, `[silu]` | Identical oracle mismatch: cosine 0.0327251, max absolute error 0.0563965 |
| `test_w4a16_preplanned_capacity_launch_accepts_smaller_live_m` | Prepared packed call lacks route-pack programs |

The last test supplies a compiled fused launch but omits the route-pack launch
required by the current low-level contract. The scale-case cause is not isolated
in this pass. No tolerance, skip or safety guard is changed. The initial `-x`
attempt (17 passes, one failure), complete baseline attempt (36.04 s) and complete
continuation attempt (36.07 s) all remain in the bundle. These failures predate
this prototype by direct reproduction; that does not turn them into passes.
The prepared cache and real-checkpoint equality controls above are distinct
paths and remain passing.

Other retained failures are the K64 arithmetic rejection, three graph-capture
failures before the finalizer fix, a host attempt whose source changed during
collection, and an archive-only host run whose checkout-identity test requires
`.git`. The latter passes from the actual checkout. None is silently converted
to a passing receipt.

## Commits

All changes descend from `5247940a` on `work/sm103-hybrid-continuation`:

| Commit | Change |
| --- | --- |
| `cf8bc04a` | Explicit cold-prefill configuration, traffic tooling and initial prototype |
| `4fece866` | Retain launch geometry before arithmetic rejection |
| `17373620` | Preserve the qualified K traversal |
| `02ef4e0c` | Three-stage, full-width concurrency control; timing source |
| `ef2e7d89` | Workspace and mapped-owner release checks |
| `5ff70d7f` | Defer cyclic compiler finalizers across capture, with host tests |
| `e2d50cd7` | Exercise pipeline wraparound at production expert dimensions |
| `64e7b683` | Separate logical scale bytes from inferred source-sector requests |

The companion branch and its wheel remain unchanged. These commits preserve
the frozen qualification hashes and do not select an experimental launch by
default.

## Reproduction

Use the verified source-built environment and explicit checkpoint/fixture paths.
The two timing commands differ only in the explicit prepared experiment:

```sh
python benchmarks/moe/hybrid_prefill_operator.py \
  --checkpoint "$CHECKPOINT" --routes "$RETAINED_ROUTES" \
  --rows 1 4 16 64 128 256 --samples 6 --replays 20 \
  --cold-prefill two_cta --output "$EVIDENCE/narrow"

python benchmarks/moe/hybrid_prefill_operator.py \
  --checkpoint "$CHECKPOINT" --routes "$RETAINED_ROUTES" \
  --rows 1 4 16 64 128 256 --samples 6 --replays 20 \
  --cold-prefill two_cta_pipeline3 --output "$EVIDENCE/pipeline3"
```

Omit `--cold-prefill` for the existing two-arm control. `--diagnostic-only`
establishes arithmetic, graph and allocation checks before NVTX replay without
producing event-timed samples. The retained profiling command adds
`--nvtx-include 'regex:w4a16_.*_m64/'`, selects six
`W4A16FusedMoeKernel` launches, and records SpeedOfLight,
MemoryWorkloadAnalysis, LaunchStats, Occupancy, WarpStateStats, SourceCounters,
PCIe byte counters and explicit SYSLTS system-memory reads/fills. The complete
command is in `traffic-ncu-command.json` or `pipeline3-ncu-command.json`.

## Next execution experiment

Do not select either concurrency variant for serving. Keep the full-width
four-stage path. Expert-local reuse has little byte-saving opportunity on the
primary fixture, and simply doubling CTA residency did not repay its costs.

The next bounded experiment is a separately prepared FC1 launch with the same
K128 arithmetic and original transaction geometry, using the already planned
FC1/activation intermediates. It can isolate FC1 latency and resource cost from
the FC2 cooperative barrier and register requirements before considering bounded
staging. Measure its real-64 result first; only a meaningful gain should proceed
to capacity-256 serving with a complete memory charge. No speedup from splitting
or staging is established by this report.
