# SM120 residency cost diagnosis

Status: **implemented fixes; research-only operator measurements**. The native
NVFP4 path stages a lookup table only when its payload is Trellis. The SM120
residency harness can explicitly allocate cacheable pinned backing and journals.
The journaled exchange protocol, static serving defaults, canonical expert IDs,
numerical recipe and graph addresses are unchanged.

This report concerns PCIe RTX PRO 4000 Blackwell execution. It does not qualify
SM103 kernels, Grace-backed TMA, complete-model quality or serving throughput.
The [spectrum report](expert-residency-sm120-spectrum.md) retains the measured
baseline at `d4eb33e8`. The [engineering ledger](expert-residency-ledger.md)
identifies raw receipts and their source exports.

## Empty-tier work

The fused W4A16 constructor previously enabled a 4 KiB SQG lookup-table copy
based on the default codebook name alone. NVFP4 and packed FP4 configurations
also carry that name, although they never consume a Trellis lookup table.
Their unused LUT argument aliases weight-scale storage. Consequently, the
prologue copied 4 KiB from that placeholder in every CTA before inspecting route
counts. With 70 CTAs, the cold operation issued 286,720 bytes of unnecessary
mapped-host loads even when the packed route count was zero.

The correction requires `weight_layout == "trellis_t256"` as well as the SQG
codebook before staging. This preserves Trellis staging and removes the unused
loads from other weight formats. It also removes a potential out-of-bounds
read when an otherwise-unused scale placeholder contains less than 4 KiB.

The baseline Nsight Compute report identifies the prologue's
`LDG.E.128.CONSTANT` followed by `STS.128` as the principal long-scoreboard stall.
The recorded packed route count is zero. The fused-kernel trace median is
164.4 µs before the guard and approximately 6.0 µs afterward at M=1.
The complete empty cold operation falls from approximately 167 µs to 12 µs in
the targeted CUDA-event samples.

An empty tier still performs route-count initialization, histogram, prefix,
route initialization and sorting, followed by the native fused operation and
tier reduction. The fused operation retains grid barriers, activation over live
route capacity, and FC2 output clearing. These preserve ordinary native MoE
output semantics. No host route readback, runtime compilation, conditional graph
node or recapture is introduced.

The remaining empty-tier work is much smaller. A conditional graph or alternate
prepared graph family would require device-side condition production and a
measured dispatch comparison. That complexity is deferred. Selecting a graph
using CPU-known synthetic routes would not qualify an engine with GPU routing.

## Exchange cost and memory type

An expert in the qualification layer occupies 2,764,808 bytes across six fields:
W13, W2, their block scales and two FP32 global scales. A reversible cross-tier
exchange copies that payload four times and copies the 4,096-byte map in both
directions. It retains the original mutually exclusive tier representation.

The baseline's two host-to-host passes dominate the pause. CUDA's default copy
API recognizes mapped pointers as host memory; those copies are synchronous
with the host, as specified by the
[CUDA copy contract](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html).
Both backing and journal allocations use write-combined pages. Reading
those pages on the CPU is expensive, consistent with NVIDIA's
[write-combined memory guidance](https://docs.nvidia.com/cuda/archive/10.2/cuda-c-programming-guide/#write-combining-memory).
This is unrelated to a measured PCIe line-rate limit.

The targeted baseline trace records approximately 22.6 ms for mapped backing
to journal and another 22.5 ms for journal to mapped backing. Each direction
moves one expert payload, approximately 0.123 GB/s of effective host-copy
bandwidth. VRAM-to-journal and journal-to-VRAM event intervals total about
0.21 ms each across six copy submissions. The instrumented transaction's five
synchronization calls together take about 45 µs; map validation and Python work
are smaller than the two slow host passes.

`MappedHostAllocation(..., write_combined=False)` selects cacheable pinned
pages while retaining exact allocation size and CPU/CUDA aliases. Existing
callers retain the write-combined default. The research harness exposes separate
`--backing-memory cached` and `--journal-memory cached` options; both are required
to remove both expensive CPU-read directions. Selecting only one leaves a
roughly 24–26 ms exchange. Selecting both retains full journaling/rollback and
reduces the preliminary uninstrumented transactions to approximately 0.9–1.2 ms.
SM103 preparation does not automatically select this PCIe experiment's memory
policy.

The source-bound allocation comparison records these uninstrumented wall times:

| Backing pages | Journal pages | Median exchange |
| --- | --- | ---: |
| Write-combined | Write-combined | 45.93 ms |
| Write-combined | Cacheable | 25.60 ms |
| Cacheable | Write-combined | 25.61 ms |
| Cacheable | Cacheable | 0.860 ms |

The matching-page results use six uninstrumented samples each; the intermediate
allocation combinations use two samples each. The cacheable range is
0.841–1.151 ms. No transaction safety step is removed.

## Copy instrumentation

`benchmarks/moe/sm120_residency_costs.py` observes the existing transfer object.
Its receipt separates initial draining, map readback, each field's journal copy,
journal completion, replacement copies, replacement completion, map publication
and publication completion. Each copy identifies its logical direction and byte
count. Gaps between transfer calls retain host validation/bookkeeping time.

CUDA events and host API wall time have different meanings. An asynchronous
H2D/D2H submission's API time is not its completion time. An event interval
around a synchronous host copy includes idle stream time while the CPU copies.
Per-field event recording adds measurable overhead; uninstrumented transactions
are recorded separately. Event intervals must not be added to API wall time as
if they were independent costs. No result is labeled PCIe peak, C2C bandwidth
or raw DRAM bandwidth.

Instrumented medians across two transactions per allocation mode are below.
Payload rows each aggregate six field copies and 2,764,808 bytes. Each map
direction transfers 4,096 bytes. Rates use summed event intervals, including
submission gaps; they are effective copy diagnostics.

| Direction | Write-combined API wall / event interval | Cacheable API wall / event interval | Effective event rate, WC / cached |
| --- | ---: | ---: | ---: |
| VRAM → pinned journal | 93 / 221 µs | 90 / 197 µs | 12.48 / 14.01 GB/s |
| Mapped host → pinned journal | 22,648 / 22,722 µs | 229 / 297 µs | 0.122 / 9.31 GB/s |
| Pinned journal → VRAM | 89 / 213 µs | 87 / 188 µs | 12.96 / 14.74 GB/s |
| Pinned journal → mapped host | 22,645 / 22,716 µs | 227 / 296 µs | 0.122 / 9.35 GB/s |
| Device map → host | 18 / 30 µs | 19 / 31 µs | — |
| Host map → device | 54 / 64 µs | 16 / 27 µs | — |

The initial drain and four completion waits are individually 6.5–8.7 µs in
these instrumented receipts. Recorded gaps total 437 µs for write-combined and
308 µs for cacheable storage. They include view construction, map validation,
bookkeeping and some tracing work. Instrumented totals are 46.74 ms and 1.72 ms;
event creation/recording adds roughly 0.8–0.9 ms relative to the uninstrumented
medians. This instrumentation cannot assign an exact unperturbed Python cost to
every statement. Separately, the spectrum's snapshot/decision/acknowledgment
work, excluding exchange, costs about 0.28–0.30 ms per M=1/8/32 period-16
boundary. None of that work executes in graph replay.

The transport probe additionally compares pageable H2D, cacheable pinned H2D,
write-combined pinned H2D, and pageable-to-bounded-pinned-staging-to-H2D. It uses
one contiguous payload-sized buffer and checks every byte. These are transport
lower bounds: actual tensor-major expert slots require separate field copies,
map publication and recovery. The probe does not implement or time a canonical
backing cache transaction.

For the 2,764,808-byte contiguous transport probe, twelve alternating samples
per arm give these medians. The source row is warm ordinary RAM; file I/O,
page-fault service and complete-model memory pressure are excluded.

| Source path | Completion wall | H2D event interval | Effective wall rate |
| --- | ---: | ---: | ---: |
| Pageable RAM → VRAM | 185 µs | 166 µs | 14.9 GB/s |
| Cacheable pinned → VRAM | 136 µs | 117 µs | 20.3 GB/s |
| Write-combined pinned → VRAM | 136 µs | 117 µs | 20.3 GB/s |
| Pageable → bounded cacheable staging → VRAM | 268 µs | 115 µs | 10.3 GB/s |

Explicit CPU staging costs about 133 µs and is included only in completion wall
time. The probe allocates two payload-sized staging buffers for the competing
arms, totaling 5,529,616 pinned bytes. It does not pin a checkpoint.

## Cache-specific promotion

A canonical backing store can avoid reading the victim back from VRAM. It must
retain a verified recoverable copy of every evicted expert. Once a destination
slot is overwritten, restoring only the old map is unsafe; recovery must restore
the victim payload before allowing any graph replay.

That is a separate storage contract from exclusive HBM/Grace residency. A PCIe
backend could retain ordinary RAM or checkpoint views and use bounded pinned
staging. It would also need a backend-specific miss service for nonresident
experts; a pageable pointer cannot replace the current mapped-host operand.
Keeping the whole checkpoint pinned is not assumed.

The allocation correction makes the existing safe exchange substantially
cheaper without adding another transaction implementation. One-way promotion is
therefore left as a transport experiment, with no new public cache API or claim
about its complete break-even point. Admission thresholds, residency windows and
score margins remain experimental, unchanged policy controls. Observed later
VRAM selections per promotion remain route-count diagnostics, not measured
weight-fetch savings or a predictive benefit model.

## Spectrum and amortization

All **141 original fixtures pass** with the corrected kernel and explicit
cacheable backing/journals on the two physical RTX PRO 4000s. Another **40
fixtures pass** for periods 2/4/8, totaling 181 fixtures: 105 latency cases with
warm/scrubbed conditions and 76 policy cases. The original 141-fixture receipts
remain separate and unchanged.

Selected warm graph medians are below. Historical and corrected sweeps use
default dynamic clocks; small differences are not formal tuning acceptance.

| Live M | Cold selections | Historical static | Corrected static | Corrected all-VRAM |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0% | 234.9 µs | 63.9 µs | 50.0 µs |
| 1 | 10% | 340.1 µs | 190.9 µs | 49.9 µs |
| 1 | 100% | 1,593.0 µs | 1,410.7 µs | 49.9 µs |
| 8 | 0% | 612.5 µs | 441.1 µs | 420.8 µs |
| 128 | 0% | 1,596.3 µs | 1,459.8 µs | 1,400.6 µs |
| 128 | 1.5625% | 4,498.0 µs | 4,257.8 µs | 1,454.1 µs |
| 128 | 100% | 38,455.6 µs | 37,157.0 µs | 1,355.4 µs |

Host-direct execution remains expensive as cold participation grows. The fixes
support studying it as a miss service; they do not establish a competitive
steady host execution tier on PCIe.

For repeated-cold M=1, the ratio is **adaptive amortized wall / static wall**;
above one is slower. Wall time includes counters, snapshots, decisions and
exchange pauses. Each arm has eight epochs and at most one promotion per epoch.

| Replays per boundary | Historical ratio | Corrected ratio | Later VRAM selections per promotion |
| ---: | ---: | ---: | ---: |
| 2 | — | 1.056 | 7 |
| 4 | — | 0.849 | 14 |
| 8 | — | 0.747 | 28 |
| 16 | 2.506 | 0.709 | 56 |
| 128 | 0.917 | 0.661 | 448 |
| 512 | 0.761 | 0.652 | 1,792 |

The tested M=1 break-even lies between periods two and four for this synthetic
workload. The three paired period-four ratios are 0.848–0.850; period two is
1.054–1.065. This supports the observed reuse requirement for the corrected
reversible exchange, not a universal 14-hit policy threshold. The shorter-period
supplement uses the second card; each ratio pairs arms on that same device.

Shape and reuse both matter. M=8 repeated-cold period four has ratio 0.859 and
112 later route selections per promotion. M=32 already benefits at period one
with ratio 0.887 and 112 selections per promotion. Routes repeat the same top-k
set across tokens, so a route selection is not an independent expert fetch.
M=1 period one performs no promotions because the unchanged admission policy
requires two observations. Its result measures profiling/control overhead.

Rotating/no-reuse traffic earns **zero later hits** and remains slower: M=1
ratios are 1.414/1.210/1.106 at periods 2/4/8. At period 512 the measured 1.005
ratio is close to dynamic-clock noise and provides no evidence of benefit.
Already-hot M=1 traffic also loses with instrumentation: ratios 3.915 at period
one and 1.296 at period 16. Normal static graphs contain no counter node.

These results do not identify the complete break-even point of a one-way
canonical-backing cache. Only its transport lower bounds are measured. A future
backend must measure recovery, publication, staging and miss service together.

## Hardware and validation scope

Both cards report PCIe Gen4 ×16 while active on the single-NUMA-node
Threadripper PRO 5975WX host. The control process permits CPUs 0–63 and is not
pinned to a specific core. There is no remote NUMA node to compare. Receipts
retain affinity, topology, negotiated link state, physical UUIDs, driver,
toolchain, clocks, power, throttle state, source hashes and raw samples.

The focused host suites pass **82 tests with 11 CUDA skips**. Physical SM120
tests pass **23 tests**, including both ID widths, both host memory modes,
independent numerical references, invalid IDs, changed routes, repeated same-
graph exchanges and allocation checks. Injected failures after journal,
replacement and map writes restore payload and mapping before replay.

Native memcheck and synccheck attempts each reach their 240-second limit
(exit 124). Memcheck emits no completed-test marker; synccheck emits one before
the limit, with no final test/sanitizer summary. Both receipts are retained as
**incomplete**, with no zero-error acceptance claim. The 181-fixture and ordinary
GPU test results do not replace these gates.

The targeted compiler/resource evidence keeps the same 70-CTA, 256-thread
geometry. Dynamic shared memory falls from 58,368 to 54,272 bytes. Registers
increase from 146 to 147 per thread; this positive delta is retained rather than
hidden by the latency result. Nsight reports zero local/shared spilling and the
same one-CTA-per-SM limit and 16.67% theoretical occupancy. Under Nsight's
profiling conditions the empty fused kernel falls from 191.81 to 7.94 µs; these
instrumented times remain separate from the uninstrumented graph results.
Full branch compiler census and physical SM103
qualification are not repeated or claimed by this targeted SM120 change.

The most useful remaining experiments, in order, are real routing traces and
inter-layer scheduling; a measured canonical-backing miss-service transaction;
then residual empty-tier dispatch and redundant reductions if they remain
material in a complete layer schedule. Hot-first overlap and concurrent cache
replacement require their own lifetime contract and measurements.

## Reproduction

The cost runner refuses an existing output directory and retains failed runs.
Use the source-bound container command in the evidence bundle for the recorded
Torch/CUDA/CUTLASS combination.

```bash
python -m benchmarks.moe.sm120_residency_costs \
  --checkpoint /models/Qwen3.8-Flash-Next-NVFP4 \
  --backing-memory cached --journal-memory cached \
  --output /tmp/sm120-costs-cached

python -m benchmarks.moe.sm120_residency_spectrum \
  --checkpoint /models/Qwen3.8-Flash-Next-NVFP4 \
  --live 1 2 4 8 16 32 64 128 --cold-fractions 0 .015625 .05 .25 1 \
  --policy-live 1 8 32 --periods 1 16 128 --epochs 8 \
  --rounds 3 --repeats 8 --backing-memory cached --journal-memory cached \
  --source-revision "$(git rev-parse HEAD)" --output /tmp/sm120-spectrum-cached

ncu --profile-from-start off \
  --kernel-name 'regex:.*W4A16FusedMoeKernel.*' --launch-count 1 --set full \
  --export /tmp/sm120-empty-cold \
  python -m benchmarks.moe.sm120_residency_costs \
  --checkpoint /models/Qwen3.8-Flash-Next-NVFP4 \
  --output /tmp/sm120-ncu --ncu
```

The original spectrum and its write-combined exchange remain reproducible at
`d4eb33e8` and in the retained source archive. The corrected source still permits
write-combined storage for an independent allocation comparison. It does not
offer a production switch to restore the erroneous unused LUT reads.

## Changed implementation files

| File | Responsibility |
| --- | --- |
| [`kernel.py`](../b12x/moe/_shared/kernels/w4a16/kernel.py) | Stage the SQG table only for Trellis payloads. |
| [`disk_table.py`](../b12x/sequence/_shared/disk_table.py) | Allow explicit cacheable mapped allocations; preserve the write-combined default. |
| [`sm120_residency_poc.py`](../benchmarks/moe/sm120_residency_poc.py) | Pass explicit backing/journal memory choices during preparation. |
| [`sm120_residency_spectrum.py`](../benchmarks/moe/sm120_residency_spectrum.py) | Expose and record those choices while retaining existing workloads and policy. |
| [`sm120_residency_costs.py`](../benchmarks/moe/sm120_residency_costs.py) | Collect kernel traces, copy-direction timings, topology and transport bounds. |
| [`test_w4a16_lookup_staging.py`](../tests/moe/test_w4a16_lookup_staging.py) | Verify source-format admission and shared-memory accounting for the LUT. |
| [`test_sm120_residency_poc.py`](../tests/moe/test_sm120_residency_poc.py) | Exercise both allocation modes and rollback at journal/replacement/publication failures. |

Documentation changes comprise this report, the SM120 prototype and spectrum
guides, the engineering ledger, and the general SM103 readiness/change guides.
The generic exchange implementation and shared cache policy are unchanged.
