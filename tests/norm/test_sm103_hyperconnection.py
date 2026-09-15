"""Prepared CuTe HyperConnection primitives, live capacity, and graph replay."""
from __future__ import annotations

import pytest
import torch

from b12x.norm import hyperconnection as hc
from b12x.preparation import FrozenMapping, require_prepared
from b12x._lib.runtime_control import kernel_resolution_guard
from tests.conftest import require_sm103_or_sm12x


@pytest.mark.parametrize("operation,hidden", [(op, h) for op in
    ("grouped_rmsnorm", "scaled_silu", "gate_mean", "combine", "combine_norm", "add", "sigmoid")
    for h in (128, 2560) if op != "combine_norm" or h == 2560])
@torch.inference_mode()
def test_prepared_cute_primitives_reuse_capacity_and_replay(operation, hidden):
    require_sm103_or_sm12x()
    torch._dynamo.reset()
    device = torch.device("cuda", torch.cuda.current_device())
    capacity, streams, lowrank = 17, 4, 320
    torch.manual_seed(103)
    caps = hc.Caps(device=device, max_tokens=capacity, hidden_size=hidden, streams=streams, lowrank=lowrank)
    plan = hc.plan(caps, invocation=FrozenMapping({"operation": operation}),
        override=hc.HyperConnectionConfig(backend="cutedsl_full", reduction_block_h=1 << (hidden - 1).bit_length(),
                                          pointwise_block=256, reduction_num_warps=4))
    state = require_prepared(plan, "norm.hyperconnection", device)
    launcher = state.launch
    x = torch.randn(capacity, streams * hidden, device=device, dtype=torch.bfloat16) / 4
    weight = torch.randn(streams * hidden, device=device, dtype=torch.bfloat16) / 16
    gates = torch.randn_like(x)
    down = torch.randn(capacity, lowrank, device=device, dtype=torch.bfloat16)
    block = torch.randn(capacity, hidden, device=device, dtype=torch.bfloat16)
    injection = torch.randn(capacity, streams, device=device, dtype=torch.bfloat16)
    outputs = dict(normalized=torch.empty_like(x), bottleneck=torch.empty_like(down), block_input=torch.empty_like(block))

    def run(rows):
        binding = hc.bind(plan, tokens=rows, **outputs)
        if operation == "grouped_rmsnorm":
            return hc.run_grouped_rmsnorm(x[:rows], weight, eps=1e-6, binding=binding)
        if operation == "scaled_silu":
            return hc.run_scaled_silu(down[:rows], binding=binding)
        if operation == "gate_mean":
            return hc.run_gate_mean(x[:rows], gates[:rows], binding=binding)
        if operation == "combine":
            return hc.run_combine(x[:rows], block[:rows], injection[:rows], plan=plan)
        if operation == "combine_norm":
            return hc.run_combine_norm(x[:rows], block[:rows], injection[:rows], weight, eps=1e-6, plan=plan)
        if operation == "add":
            return hc.run_add(x[:rows], gates[:rows], out=outputs["normalized"][:rows], plan=plan)
        return hc.run_sigmoid(x[:rows], out=outputs["normalized"][:rows], plan=plan)

    def expected(rows):
        if operation == "grouped_rmsnorm":
            return hc.reference.grouped_rmsnorm(x[:rows], weight, streams=streams, eps=1e-6)
        if operation == "scaled_silu":
            return hc.reference.scaled_silu(down[:rows], streams=streams)
        if operation == "gate_mean":
            return hc.reference.gate_mean(x[:rows], gates[:rows], streams=streams)
        if operation == "combine":
            return hc.reference.combine(x[:rows], block[:rows], injection[:rows], streams=streams)
        if operation == "combine_norm":
            return hc.reference.combine_norm(x[:rows], block[:rows], injection[:rows], weight, streams=streams, eps=1e-6)
        if operation == "add":
            return x[:rows] + gates[:rows]
        return torch.sigmoid(x[:rows].float()).bfloat16()

    def check(actual, rows):
        torch.testing.assert_close(actual, expected(rows), rtol=0.01, atol=0.016)
        tensors = actual if isinstance(actual, tuple) else (actual,)
        assert all(torch.isfinite(t).all() and (not rows or t.count_nonzero()) for t in tensors)

    check(run(capacity), capacity)
    with kernel_resolution_guard("HyperConnection retains each prepared operation"):
        for rows in (1, 3, capacity, 0):
            check(run(rows), rows)
        compiled = torch.compile(lambda: run(3), backend="eager", fullgraph=True)
        check(compiled(), 3)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run(3)
        pointers = tuple(t.data_ptr() for t in outputs.values())
        x.mul_(0.5)
        down.neg_()
        gates.neg_()
        before = torch.cuda.memory_stats(device)
        graph.replay()
        torch.cuda.synchronize(device)
        after = torch.cuda.memory_stats(device)
        assert before["allocation.all.allocated"] == after["allocation.all.allocated"]
        assert pointers == tuple(t.data_ptr() for t in outputs.values())
        check(actual, 3)
        assert state.launch is launcher
