# Kimi QSRT extent diagnostics

Status: **implemented instrumentation; qualified numerical cases;
research-only performance**. The 2026-09-08 record in
`kimi_qsrt_extent_diagnostics.json` compares 256- and 512-thread launches on
actual Kimi-K3 QSRT two-bit expert extents. It uses layer 1, TP9 logical ranks
0 and 4, extent widths 384 and 256, four row counts (1, 2, 4, 8), and seeded
uniform routing. Both arms retain the same atoms, scales, rotations, routing,
and BF16 output contract.

All 16 cases return finite values and bit-identical eager/graph output. The
two launch widths produce matching output digests for each source extent and
row count. Four-row profiler records attest grid `[188, 1, 1]`, block widths
256 or 512, and 101,376 bytes of shared memory. Register counts depend on the
extent and thread width and are included in the record.

Three of the four replay intervals show service activity on the same GPU.
The rank-0 512-thread interval has unchanged counters and idle endpoint gauges,
but its 256-thread comparison overlaps a request. All timing samples are
retained as **unqualified diagnostics** and must not be used to claim a gain or
regression for 512 threads. The numerical comparisons do not
qualify full-model arithmetic, TP9 collectives, or a serving performance gain.
The target remains on 256-thread execution.

## Measurement contract

`benchmarks/benchmark_qsrt_tp9_extent.py` retains individual replay samples,
GPU operating state before and after execution, and FP32 diagnostic norms and
errors. `--profile-m4` records a graph replay trace for the four-row case and
extracts the extent kernel's grid, block, register count, and shared memory.
The trace is separate from the timed graph replay.

`--server-metrics-url` optionally samples vLLM activity around the complete
measurement interval. Use the endpoint for the service sharing the measured
GPU. Running or waiting requests, or changes in observed
counters, mark `timing_eligible=false`. Missing required counters yield
`timing_eligible=null`. `true` means only that this endpoint showed no activity;
it does not prove that other GPU processes were idle. Use an exclusive idle
GPU for a performance decision. Metrics acquisition failures abort a requested
activity check instead of silently claiming an uncontended measurement.

Output allocation uses the dtype configured by
`B12X_W4A16_TOPK_SUM_OUTPUT`. PR #339 revision `ab269462fcfc` already includes
that caller repair. The evidence instrumentation preserves it; it does not
introduce another output-allocation fix. The measured source is B12X
`0bf9f177b237`, whose MoE kernel sources match that revision. A Python binding
error-message difference does not change the launch arithmetic.

## Reproduction

Use a source checkout with the same native QSRT payload available locally.
Substitute the model path and an output filename; record the source revision,
GPU state, and environment for each arm. Apply the per-run environment from
the JSON record. The environment must also expose the vLLM QSRT checkpoint
reader `vllm.model_executor.layers.quantization.kquant_qsrt_atoms_v2`; the
recorded loader revision is `fa6ea71c01fd`.

```bash
B12X_W4A16_TOPK_SUM_OUTPUT=bf16 \
B12X_W4A16_M8_CTA_THREADS=256 \
python benchmarks/benchmark_qsrt_tp9_extent.py \
  --model /path/to/model --layer 1 --rank 0 --widths 384 \
  --block-m 8 --m-values 1,2,4,8 --time-iters 100 --profile-m4 \
  --server-metrics-url http://127.0.0.1:8000/metrics \
  --output extent-256.json
```

Repeat with `B12X_W4A16_M8_CTA_THREADS=512`, and use logical rank 4 with
width 256 for the shorter source extent. The CLI validates model metadata and
the requested routing geometry. The native two-bit reconstruction and coupled
rotations remain unchanged.

The routing and overlap-classification CPU suite passes eight tests. It covers
activity appearing at the interval boundary, counters changing between idle
boundaries, and unavailable activity observations.
