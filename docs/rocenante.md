# RoCEnante: one-shot RoCE collectives for multi-node DGX Spark TP

`b12x.comm.roce` (RoCEnante) is an all-reduce and all-gather runtime for tensor
parallelism across DGX Spark nodes joined by their ConnectX-7 200 GbE ports, one
GPU per node. It replaces NCCL on the decode path.

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
2. the last block to finish staging publishes `nbytes` and `seq` to the control
   record, which a C proxy thread (`_roce_proxy.c`, libibverbs) polls; the proxy
   posts one RDMA write of the payload and one 4-byte write of `seq` per peer on
   the same reliable QP, so the flag cannot land before the payload;
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

## Results (four DGX Spark, bf16, median of the slowest rank)

| Collective | NCCL | RoCEnante, graph replay |
|---|---|---|
| all-reduce 48 KB (6-token decode step) | 59 us | 23 us |
| all-reduce 1.5 MB (192-token batch) | 877 us | 277 us |
| all-gather [6, 38720] logits shard | 331 us (incl. reshape copy) | 96 us |
| all-gather [96, 38720] | 1491 us | 1493 us |

Serving GLM-5.3-Flash (TP4, MTP5) with the vLLM shim: per-step decode latency
lower in every cell of a 15-cell matrix (5 to 19%), c=16 throughput +15 to +23%,
coding-peak c=1 +12.7%, prefill unchanged (its 64 MB all-reduces stay on NCCL).
With both collectives routed, a decode-step profile shows zero NCCL kernels.

## vLLM integration

A thin adapter (`b12x_roce_all_reduce.py`, about 180 lines with the communicator
hooks and env vars) dispatches eligible all-reduces and all-gathers to the
runtime; enable with `VLLM_ENABLE_ROCE_ALLREDUCE=1`, bound with
`VLLM_ROCE_ALLREDUCE_MAX_SIZE` (2MB) and `VLLM_ROCE_ALLGATHER_MAX_SIZE` (16MB).
The backend appears as `B12X_ROCENANTE` in the communicator's dispatch list.

## Tests and benchmark

`tests/comm/test_roce_oneshot_gpu.py` (torchrun, 2+ nodes): NCCL parity,
bit-identical ranks, dtype/shape eligibility, dim-0/last-dim/unaligned gathers,
CUDA-graph replay mixing both collectives. `benchmarks/benchmark_roce_oneshot.py`
times both collectives against NCCL.

## Not yet

Fused all-reduce + residual + RMSNorm (the PCIe runtime has it), all-to-all for
expert parallelism, GPU-initiated posting (needs GDR support the platform lacks).
