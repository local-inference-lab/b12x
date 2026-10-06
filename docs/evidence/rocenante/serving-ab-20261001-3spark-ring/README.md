# Serving A/B on a switchless three-Spark ring: RoCEnante vs NCCL, 2026-10-01

Purpose: show that, with the per-peer rail routing in #457, RoCEnante comes up and serves on a ring
topology where it previously failed setup, and compare decode against NCCL with nothing else changed.

Measured configuration, identical for both arms except one environment variable:

- Nodes: three DGX Spark GB10 (`sm_121`), one GPU each, cabled in a ring with no switch (each node's
  QSFP port 0 to the next node's port 1). All four ConnectX-7 functions per node, RoCE v2 GID index 3,
  six point-to-point /30 subnets (two per cable). Kubernetes, host networking, one pod per node.
- Serving: vLLM fork `local-inference-lab/vllm` at `ec49d5781a` with the b12x backends, b12x
  `0d6600e6` with `b12x/comm/roce` from this PR's head `0790f115` laid over it (`comm/roce` is
  identical at `0d6600e6` and the PR base `a321f9a6`, so the overlay is exactly the PR's change). One
  image for both arms, `sha256:ca51aa5681ecd86ebc1961148f586c2591e3056c1582ca74691a7fab2741565f`
  (label `local-inference.b12x.commit=0790f115...`).
- Model and engine settings: DeepSeek-V4.1-Flash (official checkpoint), tensor parallelism of 3
  across the three nodes, max_model_len 8192, max_num_seqs 2, max_num_batched_tokens 1024, FP8 KV,
  6 GiB KV cache, Engram tables on disk, DSpark/MTP off, prefix caching on, CUDA graphs
  `FULL_AND_PIECEWISE` with capture sizes 1 and 2. Full spec: `inferenceservice-roce.yaml`.
- NCCL 2.30.7 with `NCCL_IB_HCA` listing all four functions and `NCCL_IB_SUBNET_AWARE_ROUTING=1`
  (required for NCCL on this topology). `B12X_ROCE_HCA` unset, so b12x uses the same four devices.
- Arm ROCE: `VLLM_ENABLE_ROCE_ALLREDUCE=1` (all-reduce cap 2 MB, all-gather cap 16 MB). Startup
  (`roce-startup.txt`): `Using ['B12X_ROCENANTE', 'PYNCCL'] all-reduce backends ... for group 'tp:0'`,
  `RoCEnante all-gather is live`, `RoCEnante all-reduce is live ... NCCL remains the fallback above 2MB`.
- Arm NCCL: the same spec with `VLLM_ENABLE_ROCE_ALLREDUCE=0`. Startup (`nccl-startup.txt`):
  `Using ['PYNCCL'] all-reduce backends ... for group 'tp:0'`.
- Both arms: the expert-parallel group `ep:0` uses `['PYNCCL']`; only `tp:0` collectives move to
  RoCEnante.
- Correctness state: a temperature-0 request ("Write a Python one-liner that reverses a string.")
  returns a correct `[::-1]` answer under both arms (`*-sanity.json`). The wording differs between
  arms: the all-reduce summation order differs, so bf16 rounding and the greedy token path diverge.
  The published `run-ab.sh` stops an arm before timing if deployment fails, the startup log does not
  show the arm's `tp:0` backend, or the sanity request fails or lacks `[::-1]`. Those gates were added
  after the recorded run; its outputs satisfy them (`*-startup.txt`, `*-sanity.json`), checked after
  the run. Collective correctness against NCCL is the standalone receipt's gate, which passed before
  and after timing at every size.
- Client: `decode_ab.py` (in this directory) against the service on rank 0, sequenced by `run-ab.sh`:
  per concurrency (1, 2), one warmup round and five measured rounds of identical greedy streaming
  requests, `ignore_eos`, 512 tokens each. Decode rate per request = (completion_tokens - 1) /
  (last token time - first token time). The ROCE arm ran 18:23 to 18:29 UTC and the NCCL arm 18:29 to
  18:35 UTC, back to back on the same nodes. Raw per-request samples are in `ROCE-decode.json` and
  `NCCL-decode.json`.

| Node | Role | GPU | UUID | Driver |
|---|---|---|---|---|
| ahazidgx1 | vLLM TP rank 0 | NVIDIA GB10 | `68e485bd-85d2-71a4-ddc6-72cd760ca94d` | 580.173.02 |
| ahazidgx2 | vLLM TP rank 1 | NVIDIA GB10 | `e9ec5df4-fed8-bc47-bb4c-037e131cebff` | 580.173.02 |
| ahazidgx3 | vLLM TP rank 2 | NVIDIA GB10 | `1001b145-0a7c-e65e-de6a-82976ac90456` | 580.173.02 |

## Result (per-stream decode tok/s, median of the measured requests)

| cell | ROCE | NCCL | delta | ROCE range | NCCL range | TTFT ROCE / NCCL |
|---|---|---|---|---|---|---|
| c=1 (5 requests) | 33.68 | 33.91 | -0.7% | 32.55-34.54 | 33.87-34.04 | 0.224 / 0.226 s |
| c=2 (10 requests) | 27.36 | 27.58 | -0.8% | 26.75-27.86 | 27.33-27.84 | 0.363 / 0.336 s |

Reading: decode is at parity on this configuration; the differences are inside the ROCE arm's
spread. The standalone receipt (`../20261001-3spark-ring-bf16-standalone.json`) is a proxy: it times
isolated collectives (2.6-5.2x faster in graph replay) and does not measure the `tp:0` collective
inside a serving step. Why that speedup does not show up in decode here is not measured. Two
untested candidates: at one or two sequences the all-reduce may be a small share of the step, and
the MoE's `ep:0` group stays on NCCL in both arms. This configuration is capped at two sequences.

An earlier single-deployment run of the same code (image built from the PR's first commit,
`earlier-run-*-c1.jsonl`, 17:44 to 17:52 UTC) measured ROCE 36.24 (35.74-36.30) vs NCCL 34.04
(33.81-34.17) tok/s at c=1. It did not reproduce here: the NCCL arm is stable across both sessions
while the ROCE arm varies between deployments, so no end-to-end decode gain is claimed.
