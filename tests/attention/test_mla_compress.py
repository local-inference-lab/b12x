"""CSA boundaries, carry identity, poison, graph replay and Int64 state offsets."""
from __future__ import annotations

import pytest
import torch

from b12x.attention import mla_compress as op
from b12x.attention.mla_compress.reference import streaming_reference
from b12x.policy import PolicyContext, PolicyMode
from b12x._lib.runtime_control import (
    freeze_kernel_resolution, unfreeze_kernel_resolution, kernel_resolution_frozen,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _binding(ratio=2, states=8):
    device = torch.device("cuda", torch.cuda.current_device())
    if not op.is_supported(device):
        pytest.skip("SM12x CuTe required")
    cap = op.Caps(device=device, max_tokens=12, max_requests=4, max_states=states, ratio=ratio)
    plan = op.plan(cap, policy=PolicyContext.for_device(device, mode=PolicyMode.HEURISTIC_ONLY))
    kwargs = dict(
        values=torch.empty((12, 512), dtype=torch.float32 if ratio == 2 else torch.bfloat16, device=device),
        weight=torch.linspace(-1.1, 1.3, 512, device=device),
        query_start_loc=torch.zeros(5, dtype=torch.int32, device=device),
        positions=torch.zeros(4, dtype=torch.int64, device=device),
        state_ids=torch.full((4,), -1, dtype=torch.int64, device=device),
        destination_slots=torch.arange(12, dtype=torch.int64, device=device) + 2**35,
        live_counts=torch.zeros(2, dtype=torch.int32, device=device),
        out=torch.empty((12, 512), dtype=torch.bfloat16, device=device),
        emitted=torch.empty(12, dtype=torch.bool, device=device),
        emitted_slots=torch.empty(12, dtype=torch.int64, device=device),
    )
    if ratio == 2:
        kwargs.update(
            gates=torch.empty((12, 512), device=device),
            pending_values=torch.empty((states, 512), device=device),
            pending_gates=torch.empty((states, 512), device=device),
            pending_position=torch.full((states,), -1, dtype=torch.int64, device=device),
        )
        if states < 1024:
            kwargs["pending_values"].fill_(float("nan"))
            kwargs["pending_gates"].fill_(float("nan"))
    before = torch.cuda.memory_allocated(device)
    bound = op.bind(plan, **kwargs)
    assert torch.cuda.memory_allocated(device) == before
    return bound


def _prepare(b, lengths, positions, ids, seed):
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    n = starts[-1]
    b.query_start_loc.copy_(torch.tensor(starts + [n] * (5 - len(starts)), dtype=torch.int32, device=b.values.device))
    b.positions.copy_(torch.tensor(positions + [-1] * (4 - len(positions)), device=b.values.device))
    b.state_ids.copy_(torch.tensor(ids + [-1] * (4 - len(ids)), device=b.values.device))
    b.live_counts.copy_(torch.tensor([n, len(lengths)], dtype=torch.int32, device=b.values.device))
    gen = torch.Generator().manual_seed(seed)
    values = torch.randn((12, 512), generator=gen).to(b.values.dtype)
    b.values.copy_(values)
    gates = None
    if b.gates is not None:
        gates = torch.randn((12, 512), generator=gen) * 30
        # A channelwise gate alternation catches accidentally scalar pooling.
        gates[:, ::2] *= -1
        b.gates.copy_(gates)
        b.gates[n:].fill_(float("nan"))
    b.values[n:].fill_(float("nan"))
    return dict(values=values, gates=gates, weight=b.weight.cpu(), starts=starts,
                positions=positions, state_ids=ids, slots=b.destination_slots.cpu(),
                ratio=b.plan.caps.ratio)


def _assert_result(b, expected):
    torch.testing.assert_close(b.out.cpu(), expected[0], rtol=0.012, atol=0.016)
    torch.testing.assert_close(b.emitted.cpu(), expected[1], rtol=0, atol=0)
    torch.testing.assert_close(b.emitted_slots.cpu(), expected[2], rtol=0, atol=0)
    assert torch.isfinite(b.out).all()


def test_pair_chunk_decode_reorder_and_poison():
    b = _binding()
    state = {}
    calls = [
        ([3, 2, 0], [0, 0, 77], [5, 2, 1]),
        # First request consumes old carry AND writes new carry in this call.
        ([4, 1, 1], [3, 2, 0], [5, 2, 7]),
        # Reorder requests, complete three independent old pairs.
        ([1, 1, 1, 2], [1, 7, 3, 900], [7, 5, 2, -1]),
        # Empty rows retain state. Missing/stale carry must not be consumed.
        ([0, 1, 1], [8, 11, 0], [5, 7, 2]),
        ([0, 0], [8, 12], [5, 7]),
        ([], [], []),
    ]
    for seed, (lengths, positions, ids) in enumerate(calls):
        kwargs = _prepare(b, lengths, positions, ids, seed)
        expected = streaming_reference(**kwargs, state=state)
        b.out.fill_(float("nan"))
        b.emitted.fill_(True)
        b.emitted_slots.fill_(42)
        op.run(b)
        _assert_result(b, expected)
        for sid in set(i for i in ids if i >= 0):
            tag = b.pending_position[sid].item()
            if sid in state:
                assert tag == state[sid][0]
                torch.testing.assert_close(b.pending_values[sid].cpu(), state[sid][1], rtol=0, atol=0)
                torch.testing.assert_close(b.pending_gates[sid].cpu(), state[sid][2], rtol=0, atol=0)
            else:
                assert tag == -1


@pytest.mark.parametrize("ratio", [1, 2])
def test_graph_dynamic_counts_and_stable_output(ratio):
    b = _binding(ratio)
    _prepare(b, [2], [0], [1], 4)
    op.run(b)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.run(b)
    state = {}
    frozen = kernel_resolution_frozen()
    freeze_kernel_resolution("CSA fixed-capacity graph regression")
    try:
        for seed, (lengths, positions, ids) in enumerate([
            ([3, 1], [0, 0], [4, 2]),
            ([1, 2, 1], [1, 3, 700], [2, 4, -1]),
            ([1], [5], [4]),
            ([], [], []),
        ]):
            kwargs = _prepare(b, lengths, positions, ids, seed + 20)
            expected = streaming_reference(**kwargs, state=state)
            b.out.fill_(float("nan"))
            before = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == before
            _assert_result(b, expected)
    finally:
        if not frozen:
            unfreeze_kernel_resolution()


def test_high_state_id_crosses_int32_element_offset():
    sid = (2**31 // 512) + 1
    required = (sid + 2) * (512 * 4 * 2 + 8)
    free, _ = torch.cuda.mem_get_info()
    if free < required + 2**30:
        pytest.skip("high-state addressing probe requires 18 GiB free")
    b = _binding(states=sid + 2)
    state = {}
    for seed, position in enumerate((0, 1)):
        kwargs = _prepare(b, [1], [position], [sid], seed + 50)
        expected = streaming_reference(**kwargs, state=state)
        op.run(b)
        _assert_result(b, expected)
    assert b.pending_position[sid].item() == -1


def test_binding_rejects_mutable_state_alias():
    from dataclasses import fields

    b = _binding()
    kwargs = {field.name: getattr(b, field.name) for field in fields(b)
              if field.name not in {"plan", "_pointers"}}
    kwargs["pending_gates"] = b.pending_values
    with pytest.raises(ValueError, match="overlap"):
        op.bind(b.plan, **kwargs)
