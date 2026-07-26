# PCIe Calibration Probe

`sparkinfer.comm.pcie.overlap_probe` is a standalone, vLLM-independent
calibrator for GLM-shaped TP and DCP traffic. It measures real SparkInfer and
NCCL collectives on the deployed rank topology instead of inferring policy
from PCIe link labels alone.

## Numeric contract

Automatic policy is lossless-only. The TP comparison uses SparkInfer
`DmaAllReduce` with `dma_wire_mode=0`, which transports BF16 without wire
quantization, and validates its result against NCCL before timing it.

Compressed FP8, INT8, and MXFP8 modes (`ag`, `ring`, `a2a`, `i8*`, and `mx*`)
are deliberately rejected by the calibrator. They remain explicit deployment
choices and can never be enabled by the generated policy.

## Measurements

The probe reports the median of the slowest rank after warmup and records the
physical rank-to-GPU mapping. It calibrates three independent decisions:

1. NCCL versus lossless BF16 SparkInfer DMA over a TP payload ladder. DMA is
   selected only at the start of a sustained winning tail.
2. One-layer CKV prefetch by timing isolated and concurrent TP/CKV collectives
   in both launch orders. Any material regression or pathological contention
   disables prefetch without disabling other DCP optimizations.
3. Query split using the real paged FP8 indexer plus exact FP32/int32 candidate
   exchange, owner top-k merge, and final TP index gather. The split and
   replicated results are checked for equality at every requested context.

## Run

Run from a SparkInfer checkout inside an image that contains its build
dependencies:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc-per-node=8 \
  -m sparkinfer.comm.pcie.overlap_probe \
  --tp-size 8 \
  --dcp-size 4 \
  --context-tokens 8192,65536,131072 \
  --output /tmp/pcie-calibration.json
```

The `policy` object contains:

- `tp_allreduce.dma_min_bytes`: first payload where lossless DMA wins
  consistently; use NCCL below it. A value of `0` means that DMA never won
  the measured ladder and the deployment should disable this backend.
- `dcp_ckv_prefetch_depth`: `1` when overlap is beneficial and non-harmful
  across the measured ladder, otherwise `0`.
- `dcp_query_split`: `1` only when the complete exact split phase wins.
- `dcp_query_split_min_context_tokens`: first context in a sustained winning
  tail; vLLM keeps the unsplit path below this crossover.
- `compressed_dma_requires_explicit_opt_in`: always `true`.

The JSON also contains all raw timing summaries and the exact physical PCI
addresses used by each rank. Calibration should be rerun after changing GPU
placement, CPU/NUMA binding, PCIe generation caps, NCCL, or collective code.

## Validation snapshot

The probe was validated on 8x RTX PRO 6000 Blackwell with TP8/DCP4. All three
correctness gates passed before timing: DMA matched NCCL exactly, the TP/CKV
collectives matched their analytic patterns, and split top-k matched the
replicated oracle at every context.

| Topology | CKV overlap at 8k / 64k / 128k | Prefetch | Query split |
| --- | ---: | ---: | ---: |
| Adjacent GPU order | +3.1% / +3.1% / +3.8% | `1` | `1` |
| Interleaved root complexes | -59.5% / -428.6% / -723.8% | `0` | `1` |

On the adjacent topology, NCCL won through 6 MiB and lossless BF16 DMA won at
24 MiB and 96 MiB, so the measured crossover was 24 MiB. The complete
query-split phase was 23% faster at 8k and 58-60% faster at 64k-128k. At 8k,
each shard has only 1024 local candidates, so both exact paths move 64 MiB per
rank; at 64k and above the owner path reduces per-rank top-k communication from
128 MiB to 72 MiB. The interleaved run reproduces the CKV-prefetch collapse
while correctly retaining the unrelated query-split optimization.

The percentages above use `(serialized - concurrent) / serialized` for CKV
overlap and `(full - split) / full` for query split, so positive values are
faster. Decisions use the pessimistic median from both launch orders.

### Reproducible evidence

- Probe source: commit `f3fb62c8af5bf6759e28a067b4e6fc8732a9e497`, worktree
  `/root/vllm/worktrees/sparkinfer-pcie-calibration-pr-20260726`.
- Runtime: PyTorch `2.12.0+cu132`; CUDA `13.2`; nine measured samples after
  three warmup iterations for every mode and point.
- [Adjacent GPU 0-7 raw JSON](evidence/pcie_calibration/20260726-adjacent-tp8-dcp4.json)
- [Interleaved even-GPU raw JSON](evidence/pcie_calibration/20260726-interleaved-tp8-dcp4.json)

Adjacent command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc-per-node=8 \
  -m sparkinfer.comm.pcie.overlap_probe \
  --tp-size 8 --dcp-size 4 \
  --context-tokens 8192,65536,131072 \
  --output adjacent-gpu0-7-f3fb62c.json
```

Interleaved command:

```bash
CUDA_VISIBLE_DEVICES=0,2,4,6,8,10,12,14 \
torchrun --standalone --nproc-per-node=8 \
  -m sparkinfer.comm.pcie.overlap_probe \
  --tp-size 8 --dcp-size 4 \
  --context-tokens 8192,65536,131072 \
  --output interleaved-gpu-even-f3fb62c.json
```

The linked artifacts contain physical rank-to-PCI mappings, all raw timing
samples, medians and extrema, exact commands, source identity, correctness
results, byte counts, and final policy. This is the audit source for the table;
the console rendering is only a summary.
