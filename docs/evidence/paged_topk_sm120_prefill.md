# SM120 paged top-k prefill qualification

## Status and operation contract

Status: **qualified** for the paged SM120 top-k selector used by GLM-5.3 C4
prefill.

The selector admits the largest shared-memory candidate capacity supported by
the device, omits the final score write when the caller requests indices only,
and initializes padding candidates before an exact overflow rescan. The public
indexer interfaces, logical-index output contract, page layout, top-k value,
and supported input layouts are unchanged.

For a row with fewer live candidates than `topk`, the result contains every
live candidate at most once and fills remaining entries with `-1`. CUDA graph
replay may change live lengths and logits without retaining a stale padded
winner.

## Source and hardware

- Comparison revision: `fc1d4b68f7a5b0cfdb88bf06abccd869f5c589d5`,
  tree `7fcb6afb69b04b6981060012e943e125e58168b4`.
- Qualified implementation revision:
  `5a24588f1a6846a03fb4ca65debc868a5efee7d8`, tree
  `b7ef58e13e4057810bdd6b64170e7c93e711a300`.
- Runtime image:
  `local/vllm:glm53-flash-nvfp4-dcp4-dflash2-flashkda-bt4096-20260830-r3`.
- Physical GPU 1: `GPU-3747d42a-c416-459c-cf88-bc2e84f471b1`, NVIDIA RTX PRO
  6000 Blackwell Workstation Edition, driver 610.57.04, PCIe Gen5 x16.
- Performance measurements used CUDA graph replay with L2 flushing, 10
  warmups, and 30 timed iterations.

## Correctness

The focused suite was:

```bash
CUDA_VISIBLE_DEVICES=1 CUTE_DSL_ARCH=sm_120a \
  /opt/venv/bin/python -B -m pytest -q \
  tests/attention/test_attention_dsa_indexer_api.py \
  tests/attention/test_paged_prefill_topk_long_context.py \
  -k "row_topk or paged_prefill_topk"
```

Result: `15 passed, 29 deselected`.

A production-geometry analytic run used 4,080 query rows, 64 replicated
indexer heads, an 8,192-token visible cache, top-k 512, and a shared paged
table. Every selected index matched the exact FP32 top-k membership and the
maximum valid-score error was 0.000488.

## Performance

The comparison revision always materializes final scores. The qualified
revision used the serving contract in which sparse attention consumes only
selected indices:

```bash
/opt/venv/bin/python -B benchmarks/benchmark_paged_indexer.py \
  --rows 4080 \
  --global-heads 64 \
  --tp-size 4 \
  --page-table-width 128 \
  --seq-len 8192 \
  --mode supertile-topk \
  --route paged-tiled \
  --topk 512 \
  --indices-only \
  --warmup 10 \
  --iters 30
```

The comparison command omitted `--indices-only` because that revision does not
implement the indices-only contract.

| Revision | Median | Minimum |
| --- | ---: | ---: |
| Comparison | 1,074.18 us | 1,072.13 us |
| Qualified | 1,053.70 us | 1,052.22 us |

The fixed-work latency reduction is 1.91%. The equivalent throughput ratio is
`1074.18 / 1053.70 = 1.019436`, or 1.94% higher selector throughput.

Comparison raw samples, microseconds:

```text
1073.15, 1073.73, 1075.78, 1074.75, 1075.17, 1073.73, 1074.94,
1073.15, 1074.72, 1073.79, 1073.70, 1074.85, 1077.79, 1074.18,
1077.15, 1075.84, 1074.18, 1073.15, 1076.22, 1073.15, 1074.18,
1074.18, 1075.20, 1074.18, 1075.20, 1072.13, 1074.18, 1075.20,
1074.18, 1075.20
```

Qualified raw samples, microseconds:

```text
1052.67, 1052.22, 1052.70, 1054.40, 1052.74, 1054.30, 1053.79,
1055.52, 1053.60, 1055.36, 1053.60, 1054.37, 1056.70, 1053.70,
1054.69, 1054.50, 1052.67, 1052.67, 1053.70, 1054.72, 1053.70,
1052.67, 1053.70, 1053.70, 1054.72, 1053.70, 1052.67, 1053.70,
1052.67, 1052.67
```

The operation result does not by itself establish an end-to-end serving gain.
The release-level benchmark must include page-table preparation, C4 scoring,
communication, attention, and all remaining model work.
