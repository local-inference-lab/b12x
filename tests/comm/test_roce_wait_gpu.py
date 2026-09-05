"""Single-GPU fault injection for the real RoCE kernels; no NIC is needed.

Run each case in a subprocess: a regression must fail at the parent deadline,
even if CUDA synchronization never returns. The pinned transport region is
populated directly to inject matching or stale peer flags.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch


def _worker(kind: str, fault: bool, threads: int, blocks: int) -> None:
    from b12x.comm.roce import _allgather_cute, _oneshot_cute

    world, slots, flag_stride = 4, 2, 64
    nbytes = 8192
    spin_limit = int(os.environ.get("ROCE_TEST_SPIN_LIMIT", "4294967295"))
    inp = torch.ones(nbytes // 2, device="cuda", dtype=torch.bfloat16)
    out = torch.full(
        (inp.numel() * (world if kind == "gather" else 1),),
        -17,
        device="cuda",
        dtype=inp.dtype,
    )
    recv = torch.ones(world * slots * inp.numel(), dtype=inp.dtype, pin_memory=True)
    send = torch.zeros(slots * nbytes, dtype=torch.uint8, pin_memory=True)
    flags = torch.zeros(
        world * slots * flag_stride // 4, dtype=torch.int32, pin_memory=True
    )
    ctrl = torch.zeros(16, dtype=torch.int32, pin_memory=True)
    counters = torch.zeros(4, dtype=torch.int32, device="cuda")
    if kind == "reduce":
        launch = _oneshot_cute.get_launcher(
            "bfloat16", world, 0, threads, slots, flag_stride, 1, 0
        )
    else:
        launch = _allgather_cute.get_launcher(
            world, 0, threads, slots, flag_stride, 1, 0
        )

    def run():
        args = [inp.data_ptr(), out.data_ptr(), nbytes // 16, nbytes]
        if kind == "gather":
            args.append(nbytes // 16)
        launch(
            *args,
            recv.data_ptr(),
            flags.data_ptr(),
            send.data_ptr(),
            ctrl.data_ptr(),
            nbytes,
            counters.data_ptr(),
            counters.data_ptr() + 4,
            counters.data_ptr() + 8,
            counters.data_ptr() + 12,
            spin_limit,
            10_000_000,
            blocks,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    torch.cuda.synchronize()
    for iteration in range(int(os.environ.get("ROCE_TEST_ITERATIONS", "100"))):
        # Include the incident's epoch and unsigned sequence boundaries.
        seq = (1, 24963333, 0x7FFFFFFF, 0xFFFFFFFF)[iteration % 4]
        signed = seq if seq < 2**31 else seq - 2**32
        counters.zero_()
        counters[0] = seq - 1 if seq - 1 < 2**31 else seq - 1 - 2**32
        ctrl.zero_()
        flags.fill_(signed)
        if fault:
            flags[(3 * slots + (seq & 1)) * flag_stride // 4] = 0
        out.fill_(-17)
        torch.cuda.synchronize()
        started = time.monotonic()
        graph.replay()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        if iteration == 0:
            print(f"first replay: {elapsed:.6f}s", flush=True)
        assert elapsed < 2, "device wait exceeded wall-clock bound"
        if fault:
            if spin_limit == 0xFFFFFFFF:
                assert elapsed >= 0.008, "wait expired before its 10 ms budget"
            assert ctrl[2].item() == signed
            assert ctrl[3].item() == 3
            assert counters[3].item() == signed
            assert counters[0].item() == (
                seq - 1 if seq - 1 < 2**31 else seq - 1 - 2**32
            )
            assert torch.all(out == -17)
            graph.replay()  # a poisoned captured launch must also finish
            torch.cuda.synchronize()
            assert torch.all(out == -17)
        else:
            assert ctrl[2].item() == 0
            assert counters[0].item() == signed
            assert torch.all(out == (world if kind == "reduce" else 1))
    print(f"PASS {kind=} {fault=} {threads=} {blocks=}", flush=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["reduce", "gather"])
@pytest.mark.parametrize("fault", [False, True])
@pytest.mark.parametrize("threads,blocks", [(32, 1), (512, 8), (512, 256)])
@pytest.mark.parametrize("spin_limit", [1, 0xFFFFFFFF])
def test_collective_wait_is_bounded(kind, fault, threads, blocks, spin_limit):
    env = dict(os.environ)
    env["ROCE_TEST_SPIN_LIMIT"] = str(spin_limit)
    root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            kind,
            str(int(fault)),
            str(threads),
            str(blocks),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


if __name__ == "__main__":
    _worker(sys.argv[1], bool(int(sys.argv[2])), int(sys.argv[3]), int(sys.argv[4]))
