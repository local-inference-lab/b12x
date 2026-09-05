# RoCEnante elapsed-time failure bound

Issue: https://github.com/local-inference-lab/b12x/issues/313

Kernel/test source: `f43d071f9e0a5a8ffb1351bfc4dc09344feb5a21`, based on
`6f9153bb` (master). Worktree: `b12x-roce-fix`, branch
`fix/roce-bounded-wait`. Test date: 2026-09-04 Pacific / 2026-09-05 UTC.

## Environment

Four DGX Spark GB10 nodes, one GPU per node, `sm_121a`, 48 SMs. The tests used
isolated containers made from the serving image: CUDA toolkit 13.3.1,
PyTorch 2.13.0+cu130, CUTLASS DSL 4.7.0, driver 580.173.02. Python imported
the candidate source via `PYTHONPATH=/work`. The final four-rank run had GLM
stopped and host swap disabled on every node.

These are correctness and failure-latency checks, not healthy-path throughput
benchmarks. No performance improvement is claimed.

## Finding and change

The old wait counted polls of pinned system memory. It did not measure time.
A missing-flag probe of the unpatched kernel at the production setting of
50,000,000 polls failed its 60-second subprocess deadline. This does not prove
an infinite loop, or identify the exact kernel active during the original
serving incident.

The candidate retains the poll limit and also checks a 64-bit GPU nanosecond
timer. `B12X_ROCE_TIMEOUT_MS` defaults to 20,000 ms and participates in the
cross-rank configuration check. All-reduce and all-gather both pass it as a
runtime scalar, including captured graph launches. The output operand is
assigned only after the wait's inputs have been consumed.

Both kernels also use a CTA-wide vote before the outer poison branch. Another
block may publish poison while a block starts; independently sampled mutable
values must not decide which threads participate in block barriers. This is
defensive synchronization hardening, not a proven reconstruction of the
original incident.

## Single-GPU kernel checks

Command, inside the isolated container with the candidate on `PYTHONPATH`:

```sh
python3 -m pytest -q tests/comm/test_roce_wait_gpu.py -x
```

Result:

```text
24 passed in 58.89s
```

Coverage: both collectives; matching and stale peer flags; poll budgets of 1
and `UINT32_MAX`; a 10 ms time budget; graph replay and poisoned no-op replay;
32/512 threads; 1/8/256 blocks; sequences 1, 24963333, `0x7fffffff`, and
`0xffffffff`. Each case runs 100 iterations with a parent-process deadline.
Faults must retain the epoch and output sentinel and record the failed peer.
The timer cases also assert that the wait does not expire prematurely.

A first targeted non-instrumented run with `UINT32_MAX`, 512 threads and
8 blocks printed `first replay: 0.010026s` before completing successfully.

## Four-rank transport checks

One process per node, `RANK=0..3`, `LOCAL_RANK=0`, `WORLD_SIZE=4`, a common
`MASTER_ADDR` pointing at rank 0, and a dedicated `MASTER_PORT`:

```sh
B12X_ROCE_HCA=rocep1s0f0 NCCL_IB_HCA=rocep1s0f0 NCCL_IB_GID_INDEX=3 \
NCCL_SOCKET_IFNAME=enp1s0f0np0 \
timeout -k 10 180 python3 -m pytest -q -x tests/comm/test_roce_oneshot_gpu.py
```

Final run, swap disabled and serving stopped:

```text
rank 0: 74 passed, 2 skipped in 10.31s
rank 1: 74 passed, 2 skipped in 10.37s
rank 2: 74 passed, 2 skipped in 10.10s
rank 3: 74 passed, 2 skipped in 10.09s
```

The new asymmetric test captures the real transport kernels but substitutes
a pinned flag region on rank 0 with one peer flag deliberately stale. Rank 0
must finish with an error within the test's wall-clock bound; the other three
ranks must finish that operation with correct output. The next replay must
leave every rank poisoned. Both all-reduce and all-gather pass, using a
200 ms time budget and `UINT32_MAX` polls.

Existing proxy-stop, missed-doorbell recovery, alternating-stream/grid,
mixed-collective graph, padded-gather, and NCCL-oracle tests also pass.
Skips are the dual-HCA-only case and one shard exceeding the configured
gather capacity. The fixture destroys its owned process group on teardown.

An initial torchrun-parent launch ran out of memory while serving occupied
the nodes; the direct process launch avoids the extra parent. A separate
dual-HCA attempt failed during NCCL setup because `roceP2p1s0f0` has no usable
GID at index 3 (`::`). No dual-rail qualification is claimed.

## Synchronization checker

```sh
ROCE_TEST_ITERATIONS=1 timeout -k 5 55 \
compute-sanitizer --tool synccheck --error-exitcode 99 \
python3 tests/comm/test_roce_wait_gpu.py reduce 1 512 8
```

```text
first replay: 0.090685s
PASS kind='reduce' fault=True threads=512 blocks=8
========= ERROR SUMMARY: 0 errors
```

The 32-thread/one-block variant also reports zero errors. These instrumented
latencies include sanitizer overhead. An earlier unpatched, heavily
oversubscribed sanitizer attempt did not finish within its process deadline;
it is not counted as a successful check.

## Remaining limits

This repairs the measured peer-wait failure bound. It does not establish why
the original serving run stopped making progress, eliminate every possible
lost-flag trigger, or recover a GPU/driver that stops executing instructions.
No multi-hour GLM canary, worker/QP-kill qualification, or dual-rail acceptance
has been completed. The serving recipe still routes collectives to PYNCCL;
issue #313 should remain open for the original-trigger investigation.
