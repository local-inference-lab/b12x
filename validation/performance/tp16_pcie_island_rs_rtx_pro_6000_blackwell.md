# TP16 PCIe equal-quarter all-reduce qualification

Status: **qualified**.

## Purpose

This record compares the B12X hierarchical and equal-quarter BF16 all-reduce
implementations on one TP16 PCIe topology. The comparison measures rank-maximum
CUDA-graph replay latency while a production-shaped Kimi-K3 DCP IPC pool is
resident. It does not measure end-to-end model throughput.

## Conditions

- Source repository: `local-inference-lab/b12x`
- Source revision: `01e9bab08ee1b29fface47ee1258dadbe831a560`
- Source tree: `b096b95eb5035b4fa6cad6a6787a580de1280db5`
- Measured worktree: `/mnt/luke/worktrees/b12x-pr220-source-qualified-20260816`
- Worktree state before and after timing: clean and unchanged
- Container image: `voipmonitor/vllm:kimi-k3-tp16-vllm2ddc210-b12x3bce5d8-cu133-torch213-20260816-r1 (image ID sha256:09c00dba1db141c3141a15848293064bc67ac1ff8cc64d3219f413f23f26d4ec)`
- CUDA runtime: `13.3`
- PyTorch: `2.13.0`
- Driver: `610.57.04`
- Physical GPUs: 16 × `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`
- Required GPU mode: P1, Default compute mode, active throttle mask
  `0x0000000000000000`
- Warm measurement: 100 alternating warmup
  pairs, 2000 graph replays per sample,
  20 samples per implementation
- Receipt: `validation/performance/tp16_pcie_island_rs_rtx_pro_6000_blackwell.json`

Both implementations use the same source revision. Each captured CUDA graph is
replayed and correctness-validated before timed samples. Warm samples alternate
AB and BA order with equal position counts. The receipt records every cold and
warm rank-maximum sample, the per-sample order, source hashes, compile manifests
and objects, PTXAS identity, physical GPU UUIDs, clocks, modes, and correctness
results.

## Results

| BF16 elements | Automatic dispatch | Hierarchical µs | Equal-quarter µs | Hierarchical/equal-quarter |
| ---: | :--- | ---: | ---: | ---: |
| 7,168 | hierarchical | 18.075 | 17.913 | 1.009× |
| 14,336 | equal_quarter | 21.812 | 18.453 | 1.182× |
| 14,338 | hierarchical | 23.893 | 42.014 | 0.569× |
| 28,672 | equal_quarter | 29.907 | 20.621 | 1.450× |
| 57,344 | equal_quarter | 46.169 | 25.650 | 1.800× |

The ratio is hierarchical median latency divided by equal-quarter median
latency. Values above one mean equal-quarter is faster.

## Correctness and qualification gates

Every eager result, captured result, and post-measurement result preserves the
input, is bit-identical across all 16 ranks, and matches the FP32 accumulation
reference at `rtol=0.02` and `atol=0.125`. Failed qualification checks:
none.

## Reproduction

```bash
/opt/venv/bin/python3 /mnt/luke/worktrees/b12x-pr220-source-qualified-20260816/benchmarks/benchmark_pcie_island_rs.py --output validation/performance/tp16_pcie_island_rs_rtx_pro_6000_blackwell.json --report validation/performance/tp16_pcie_island_rs_rtx_pro_6000_blackwell.md --source-revision 01e9bab08ee1b29fface47ee1258dadbe831a560 --source-tree b096b95eb5035b4fa6cad6a6787a580de1280db5 --expected-pci-bus-islands '0x03,0x04,0x23,0x24|0x43,0x44,0x63,0x64|0x83,0x84,0xA3,0xA4|0xC3,0xC4,0xE3,0xE4' --warmup 100 --iterations 2000 --samples 20 --required-active-throttle-mask 0x0
```

The command requires exactly 16 visible GPUs in the PCI bus order declared by
`--expected-pci-bus-islands`, an empty `B12X_COMPILE_CACHE_DIR`, and a clean
checkout at the recorded source revision.

## Limitations

Status applies only to the recorded TP16 topology, source revision, container,
driver, GPU mode, tensor shapes, BF16 data type, and CUDA-graph execution. The
receipt does not qualify other topology orders, GPU modes, message sizes,
dtypes, or end-to-end serving workloads.
