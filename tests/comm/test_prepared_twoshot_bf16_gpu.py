"""Qualify prepared BF16 two-shot collectives on four CUDA peer-access GPUs.

Run with ``python -m torch.distributed.run --standalone --nproc-per-node=4
tests/comm/test_prepared_twoshot_bf16_gpu.py``. Every operation is prepared
through its production factory before eager execution and CUDA graph replay.
Replay consumes changed inputs into stable caller-owned outputs without a
PyTorch allocation. All-reduce and reduce-scatter use an FP32-sum oracle.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from b12x.comm.pcie import PCIeTwoShotBF16
from b12x.comm.pcie._twoshot_preparation import (
    plan as make_plan,
    prepared_call,
    query_from_runtime,
)
from b12x.preparation import PreparationSession


def _check(operation, rank, device):
    world, rows, width = 4, 16, 4096
    pool = PCIeTwoShotBF16.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        max_rows=rows,
        row_elems=width,
    )
    input_rows = rows // world if operation == "all_gather" else rows
    output_rows = rows // world if operation == "reduce_scatter" else rows
    source = torch.empty((input_rows, width), dtype=torch.bfloat16, device=device)
    source.fill_(rank + 1)
    output = torch.empty((output_rows, width), dtype=source.dtype, device=device)
    call = {"inp" if operation == "all_reduce" else "payload": source, "out": output}
    query = query_from_runtime(pool, surface=f"PCIeTwoShotBF16.{operation}", call=call)
    plan = make_plan(query, runtime=pool)
    request = plan.request(
        name=f"bf16_{operation}",
        prepare_call=lambda state: prepared_call(state, payload=source, out=output),
    )
    graph = torch.cuda.CUDAGraph()
    try:
        with PreparationSession(
            device=device, autotune=False, compile_workers=2
        ) as session:
            session.prepare((request,))
            launch = getattr(pool, operation)
            for _ in range(3):
                assert launch(source, out=output, plan=plan) is output
            torch.cuda.synchronize(device)
            dist.barrier()
            with session.capture(), pool.capture(plan=plan), torch.cuda.graph(graph):
                assert launch(source, out=output, plan=plan) is output
            address = output.data_ptr()
            for step in range(3):
                torch.manual_seed(1000 + 10 * step + rank)
                source.copy_(torch.randn_like(source))
                gathered = [torch.empty_like(source) for _ in range(world)]
                dist.all_gather(gathered, source)
                if operation == "all_gather":
                    expected = torch.cat(gathered)
                else:
                    expected = torch.stack(gathered).float().sum(dim=0)
                    if operation == "reduce_scatter":
                        expected = expected.chunk(world)[rank]
                output.fill_(float("nan"))
                torch.cuda.synchronize(device)
                dist.barrier()
                allocations = torch.cuda.memory_stats(device)[
                    "allocation.all.allocated"
                ]
                graph.replay()
                torch.cuda.synchronize(device)
                assert (
                    torch.cuda.memory_stats(device)["allocation.all.allocated"]
                    == allocations
                )
                assert output.data_ptr() == address
                assert torch.isfinite(output).all()
                if operation == "all_gather":
                    assert torch.equal(output, expected)
                else:
                    bound = expected.abs() * 2.0**-8 + 1e-5
                    assert ((output.float() - expected).abs() <= bound).all()
                dist.barrier()
            graph.reset()
        if rank == 0:
            print(
                f"PASS {operation}: preparation, eager, 3 graph replays, FP32 oracle",
                flush=True,
            )
    finally:
        graph.reset()
        pool.close()


def main():
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    try:
        assert dist.get_world_size() == 4
        for operation in ("all_reduce", "reduce_scatter", "all_gather"):
            _check(operation, rank, device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
