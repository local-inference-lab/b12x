# RoCEnante: one-shot RoCE collectives for multi-node DGX Spark TP

`b12x.comm.roce` (RoCEnante) is an all-reduce and all-gather runtime for tensor
parallelism across DGX Spark nodes joined by their ConnectX-7 200 GbE ports, one
GPU per node. It is a runtime API: a serving integration routes eligible
collectives to it and keeps everything else on its own backend. The vLLM adapter
in local-inference-lab/vllm#597 (`vllm/distributed/device_communicators/b12x_roce_all_reduce.py`,
base branch `dev/jovian-judgement`) does that for decode, which takes NCCL out of
the decode path for the models measured below.

Status: implemented and qualified on four DGX Spark (GB10, `sm121a`) nodes with
the tests and receipts named in this page. Other GPUs, hosts with GPUDirect RDMA,
and integrations other than that adapter are unsupported.

## Why it works on GB10 without GPUDirect RDMA

The GB10 is an integrated GPU with unified memory: pinned host memory is directly
addressable by GPU kernels at full bandwidth, and the NIC can register it with a
plain `ibv_reg_mr`. So peers RDMA-write into pinned slots that the local kernel
reads in place. No dmabuf or `nvidia_peermem` support is needed (neither is
available on the DGX Spark driver stack; NCCL therefore stays host-staged there).

Each Spark's single cabled QSFP port is exposed as two PCIe Gen5 x4 functions
(`rocep1s0f0`, `roceP2p1s0f0`). The runtime spreads rank pairs across both.

## Protocol

One pinned region per rank: `recv[src][slot]`, `flag[src][slot]`, `send[slot]`,
and a control record. One kernel launch per collective:

1. stage the input into `send[seq & 1]`;
2. the last block to finish staging publishes `nbytes` (per slot) and `seq` to
   the control record, which a C proxy thread (`_roce_proxy.c`, libibverbs)
   polls; the proxy posts one RDMA write of the payload and one 4-byte write of
   `seq` per peer on the same reliable QP, so the flag cannot land before the
   payload. The doorbell holds only the newest `seq`, and a rank's kernel for
   op N finishes on the peers' payloads alone, so op N+1 can ring before the
   proxy has seen op N; the proxy posts every sequence between the last one it
   posted and the doorbell (at most two are ever pending). After a run of polls
   without a doorbell the thread requests a short sleep between polls (the OS
   decides the actual delay) so an idle runtime does not hold a core; the
   catch-up keeps the protocol correct however long the thread is away;
3. wait on `flag[peer][seq & 1] == seq` for every peer (bounded; a timeout
   records the missing peer and the host raises instead of hanging);
4. all-reduce: sum the local input and every peer slot in fixed rank order, so
   all ranks produce bit-identical output; all-gather: strided copy that writes
   the concatenated layout directly (dim 0 or last dim);
5. the last block to finish advances a device-resident epoch, which makes `seq` a
   runtime value and keeps CUDA-graph replay correct.

Two slots suffice because a peer cannot start op k+2 before finishing op k+1,
which needs our op k+1 data, which we only post after our op k kernel completed.
Staging and tail arrivals use separate counters (a block with nothing to stage can
pass the wait before a slower block has staged).

## Interface

`AllReduce.from_exchange_group(exchange_group=<gloo group>, device=..., max_size=,
max_gather_bytes=)` mirrors `comm.pcie.AllReduce`: `should_allreduce`,
`all_reduce`, `should_all_gather`, `all_gather(inp, dim=)`, `prepare(dtypes)`
(compile before graph capture), `for_stream`, `capture`, `check_health`, `close`.
Exchange setup over a CPU (gloo) group: using a torch NCCL group would create a
torch NCCL communicator costing about 3.4 GB of unified memory per rank.

Environment: `B12X_ROCE_HCA` (falls back to `NCCL_IB_HCA`), `B12X_ROCE_GID_INDEX`
(falls back to `NCCL_IB_GID_INDEX`, default 3), `B12X_ROCE_SPIN_LIMIT`,
`B12X_ROCE_CACHE_DIR` (where the proxy .so is built with the host C compiler).

Constraints: 2 to 16 ranks, one collective in flight per runtime (single stream),
integrated GPU with unified addressing, active RDMA devices.

## Results

Measured configuration: four DGX Spark GB10 nodes (`sm121a`, one GPU each,
unified memory), both ConnectX-7 functions per node, RoCE v2 GID index 3, bf16,
world size 4. Timing is CUDA events around one call, median over samples per
rank, then the slowest rank; graph replay is the decode path. The receipt
`docs/evidence/rocenante/20260902-4spark-bf16-standalone.json` (emitted by
`benchmarks/benchmark_roce_oneshot.py`) holds the command, source revision,
worktree state, per-rank GPU identity, correctness results, raw samples, the
executed arm order, and ratios with their direction.

| Collective | NCCL | RoCEnante, graph replay |
|---|---|---|
| all-reduce 48 KB (6-token decode step) | 59 us | 23 us |
| all-reduce 1.5 MB (192-token batch) | 877 us | 277 us |
| all-gather [6, 38720] logits shard | 331 us (incl. reshape copy) | 96 us |
| all-gather [96, 38720] | 1491 us | 1493 us |

Serving A/B, GLM-5.3-Flash TP4 with MTP5 on the same four nodes through the vLLM
adapter, RoCEnante versus NCCL for the same image and configuration
(`docs/evidence/rocenante/serving-ab-20260902/`: per-arm decode and prefill
result JSON from `llm_decode_bench.py`, the arm comparison tables, and the README
that names the image, compose overrides, and commands): per-step decode latency
lower in every cell of a 15-cell concurrency-by-context matrix (5 to 19%), c=16
throughput +15 to +23%, coding-peak c=1 +12.7%, prefill unchanged (its 64 MB
all-reduces stay above the size cap and remain on NCCL). With both collectives
routed, a decode-step profile shows zero NCCL kernels.

## vLLM integration

The adapter lives in the vLLM fork, local-inference-lab/vllm#597 on
`dev/jovian-judgement` (`vllm/distributed/device_communicators/b12x_roce_all_reduce.py`
plus hooks in `cuda_communicator.py` and three entries in `envs.py`, about 180
lines). It dispatches eligible all-reduces and all-gathers to the runtime;
enable with `VLLM_ENABLE_ROCE_ALLREDUCE=1`, bound with
`VLLM_ROCE_ALLREDUCE_MAX_SIZE` (2MB) and `VLLM_ROCE_ALLGATHER_MAX_SIZE` (16MB).
The backend appears as `B12X_ROCENANTE` in the communicator's dispatch list.
Status: qualified with the serving A/B above against that fork revision; the
adapter is supported once #597 merges, and no other integration is.

## Tests and benchmark

`tests/comm/test_roce_oneshot_gpu.py` (torchrun, 2+ nodes): NCCL parity,
bit-identical ranks, dtype/shape eligibility, dim-0/last-dim/unaligned gathers,
CUDA-graph replay mixing both collectives. `benchmarks/benchmark_roce_oneshot.py`
times both collectives against NCCL.

## Unsupported

Fused all-reduce + residual + RMSNorm (the PCIe runtime has it), all-to-all for
expert parallelism, GPU-initiated posting (needs GDR support the platform lacks),
more than one collective in flight per runtime, and hosts without an integrated
GPU or without active RDMA devices (`is_supported()` returns False there).
