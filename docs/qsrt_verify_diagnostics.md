# QSRT verify decode and phase diagnostics

Status: research-only. Decoder GPU oracles are qualified; fused native-MoE,
phase-record GPU correctness and latency qualification remain pending.

Kimi-K3 verify uses four input rows, top-k 16 routes, hidden width 3584,
and per-rank intermediate widths 256 or 384. Its native 2-bit SQG-XOR-Cheb-T12
weights are decoded inside the fused MoE kernel. The model-facing input is
BF16, while the full-rotation GEMM fragments and intermediate activations
are FP16. The optional BF16 fragment path is a separate supported conversion.

`B12X_SQG_XOR_CHEB_T12_DIRECT_PAIRS=1` changes the direct shared-memory
LUT decoder's byte packing. Eight identical table lookups yield four
independent 16-bit byte pairs rather than two 32-bit words. Each pair feeds
one native FP8x2 conversion. This removes two dependent byte-merge operations
per eight weights without changing window extraction, table bytes, conversion
type, MMA order, or split-K reduction order. The setting is off by default
and participates in the GEMM and fused compile identities. It applies only
when the direct shared-memory table path is selected.

`benchmarks/probe_k3_verify_resources.py` compiles this four-row contract with
CTA sizes 256 or 512 on a host without GPU access. The 512-thread option
already exists; the probe qualifies its resource footprint instead of choosing
it as a serving default. More threads can reduce per-thread static code while
increasing CTA resource use. Static instruction counts are not speedups.

Example inside an isolated container containing CUDA 13.3.73, Torch 2.13.0,
and CUTLASS DSL 4.6.2, launched with `--runtime runc`, no GPU device mapping,
and empty `CUDA_VISIBLE_DEVICES`:

```sh
/opt/venv/bin/python benchmarks/probe_k3_verify_resources.py \
  --width 384 --threads 256 --pairs 1 --source-revision <commit> \
  --output /evidence/w384-t256-pairs
```

The probe records source hashes, compile arguments, tool versions, the exact
wrapper object and extracted cubin, resource output, and SASS. Compilation
stops before device resource queries or direct-LUT allocation. Compare pairs
0/1 at both widths and both CTA sizes under the same compiler. Shared layout
bytes reported from source are unrounded; they are not a CUDA occupancy query.

## Static compiler measurements

The measurements below use source revision `1414b128`, the toolchain above,
and profiling disabled. The four baseline CUDA `.text` sections are byte-identical
to the archived production-contract objects. Full wrapper objects retain
separate hashes. Artifact identities and measurements are in
`validation/performance/kimi_k3_verify_direct_pairs_offline_20260908.json`.

| Width | CTA threads | SASS instructions, baseline → pairs | Registers/thread, baseline → pairs |
| --- | --- | --- | --- |
| 256 | 256 | 6104 → 5976 | 120 → 121 |
| 384 | 256 | 6128 → 6000 | 121 → 119 |
| 256 | 512 | 4728 → 4664 | 100 → 103 |
| 384 | 512 | 4768 → 4704 | 98 → 103 |

LUT byte-load, FP8-to-FP16 conversion and HMMA counts remain unchanged within
each pair. No local loads or stores appear. All three register increases
remain flagged for GPU measurement; the source planner still selects one CTA
per SM. These observations support testing the option, not enabling it by
default. With width 384, CTA256 and pairs enabled, diagnostic instrumentation
adds 56 static instructions and two registers; its durations require a control.

## Per-CTA phase records

`B12X_W4A16_PHASE_PROFILE=1` enables diagnostic records for single-tier,
full-rotation coupled-Hadamard direct-route decode with at most eight rows
and one resident CTA per SM. Mixed-tier kernels remain uninstrumented. Set
the option before planning workspaces and compiling kernels. Reprepare a pool
when the mode changes; captured graphs must be discarded before changing
their workspace or kernel contract.

The planner preserves the split-K locks and two grid counters and appends
ten uint64 fields per CTA at a 16-byte-aligned offset. SMS188 needs 4516
int32 elements instead of 754. Compiled launches validate the required
capacity before execution. Thread zero owns each record, and every launch
initializes its cumulative counters. No recording atomics or extra shared
memory are required.

| Fields | Meaning |
| --- | --- |
| `body_start_ns` | After LUT staging and counter initialization |
| `rotation_arrive_ns`, `rotation_release_ns` | Barrier after input rotation |
| `fc1_arrive_ns`, `fc1_release_ns` | Barrier after FC1 |
| `activation_arrive_ns`, `activation_release_ns` | Barrier before FC2 |
| `fc2_end_ns` | After an additional diagnostic CTA sync at FC2 completion |
| `fc1_lock_wait_ns` | Sum of reduction-turn acquire-loop durations |
| `fc1_lock_wait_count` | Number of acquire-loop calls, including immediately successful calls |

Barrier timestamps are inside their surrounding CTA synchronizations: after
the first sync and before the arrival atomic, then after publication/polling
and before the last sync. Their difference includes atomic/fence/polling cost.
The FC1 counter excludes accumulator combination and the following CTA sync.
Timing instrumentation perturbs execution and must be compared with an
uninstrumented control.

Only the actual launch grid rows are valid. A workspace reused by multiple
layers retains only the last launch, so use an isolated extent. Synchronize
its stream, save `workspace.cpu().numpy().tofile(path)`, then run:

```sh
.venv/bin/python benchmarks/analyze_w4a16_phase_profile.py workspace.bin \
  --sms 188 --ctas 188 --output phases.json
```

The reader rejects missing records and distinguishes arrival spread from
release latency. Per-CTA intervals overlap and must not be added to estimate
step latency. The body interval excludes LUT staging and the separate route
sum kernel.

## Qualification

CPU tests cover compile identities, workspace boundaries, pool re-preparation,
and the record reader. The GPU oracle in
`tests/moe/test_trellis_direct_lut_decode.py` enumerates every 16-bit state in
each window position for 2/3/4-bit packing and both FP16/BF16 converters.
It compares against independent CPU table indexing and the 32-bit decoder.
All nine GPU decoder cases passed, as recorded in
`validation/performance/kimi_k3_verify_gpu_correctness_20260908.json`.

`benchmarks/benchmark_qsrt_verify_variants.py` prepares one native weight extent
and shares it between CTA256/512 and pair off/on variants. It checks mutated
inputs and records interleaved timings, compiled objects and optional phase
records. This harness has passed static checks but has not executed on a GPU.

Before adoption, require native-payload output equality, eager and graph
replay, fixed workspace addresses, and decode-boundary/prefill checks. Time
the exact compiled objects with balanced ordering and recorded GPU power,
clock and throttle state. Keep TP9/DCP9, K=3, and the 1,048,576-token context
budget fixed for serving comparisons. Inspect a short rank-0 verify trace
and default `llm_decode_bench.py` results before claiming end-to-end benefit.
