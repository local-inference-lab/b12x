"""Portable fixed-address byte probes; these do not execute SM103 expert MMA."""
from functools import lru_cache
import gc

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import pytest
import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.moe import fused_moe as moe
from b12x.moe.fused_moe._residency_storage import materialize_tier
from b12x.moe.fused_moe._residency_updates import materialize_updates, ResidencyUpdateError
from tests.moe.test_expert_residency import declaration, placement
from tests.moe.test_residency_kernels import invoke, programs


class ReadSlotPayload:
    """Read every byte using the production partitioner's compact row metadata."""
    def __init__(self, width): self.width = width

    @cute.jit
    def __call__(self, source: cute.Pointer, local: cute.Pointer, indices: cute.Pointer,
                 count: cute.Pointer, output: cute.Pointer, live: cutlass.Int32, stream: cuda.CUstream):
        self.kernel(source, local, indices, count, output).launch(
            grid=(cute.ceil_div(self.width, 256), live, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, source: cute.Pointer, local: cute.Pointer, indices: cute.Pointer,
               count: cute.Pointer, output: cute.Pointer):
        block, route, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        col = block*256+tid
        if route < count[0] and col < self.width:
            row = cutlass.Int64(local[cutlass.Int64(route)])
            original = cutlass.Int64(indices[cutlass.Int64(route)])
            output[original*self.width+col] = source[row*self.width+col]


@lru_cache(None)
def reader(width):
    from b12x._lib.architecture import architecture_for
    from b12x.moe._shared.kernels.sm103.launch import pointer
    return cute.compile(ReadSlotPayload(width), pointer(cutlass.Uint8), pointer(cutlass.Int32),
        pointer(cutlass.Int32), pointer(cutlass.Int32), pointer(cutlass.Uint8), cutlass.Int32(1),
        cuda.CUstream(0), options=f"--gpu-arch={architecture_for(torch.cuda.get_device_capability()).compilation_target}")


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("hot", [(0, 1, 2, 3), (0, 2), ()])
def test_same_graph_reads_exchanged_payloads_without_pointer_changes(dtype, hot, monkeypatch):
    from tests.conftest import require_sm103_or_sm12x
    require_sm103_or_sm12x()
    device = torch.device("cuda", torch.cuda.current_device())
    profile = placement(hot=hot, cold=tuple(e for e in range(4) if e not in hot))
    plan, weights = declaration(profile=profile, max_tokens=2, updates=moe.ResidencyUpdateCapacity(max_pairs=2))
    generator = torch.Generator().manual_seed(912)
    for tensor in (weights.w13, weights.w2, weights.w13_block_scales, weights.w2_block_scales):
        tensor.random_(0, 256, generator=generator)
    tiers, graph, updates = [], None, None
    try:
        for tier, ids in enumerate((profile.hbm_expert_ids, profile.grace_expert_ids)):
            tiers.append(materialize_tier(ids, weights, plan.query, device, grace=tier == 1))
        mapping = torch.tensor(profile.expert_map, dtype=torch.int32, device=device)
        updates = materialize_updates(plan.query, tuple(tiers), mapping, profile, device)
        original = {expert: {name: field[row].cpu().clone() for name, field in tiers[tier].fields.items()}
                    for expert, (tier, row) in enumerate(profile.expert_map)}
        partition = programs(dtype)[0]
        # programs() prepares PartitionRoutes with capacity 32.
        local = torch.empty(2, 32, device=device, dtype=torch.int32)
        indices = torch.empty_like(local)
        counts = torch.empty(8, device=device, dtype=torch.int32)
        ids = torch.tensor([[0, 1, 2], [3, 0, -1]], dtype=dtype, device=device)
        outputs = {name: torch.empty(6, value.numel(), dtype=torch.uint8, device=device)
                   for name, value in original[0].items()}
        calls = []
        for index, tier in enumerate(tiers):
            if tier is not None:
                for name, field in tier.fields.items():
                    calls.append((reader(outputs[name].shape[1]),
                        (field, local[index], indices[index], counts[index*4:], outputs[name])))
        def run():
            invoke(partition, (ids, mapping, local, indices, counts), (ids.numel(),))
            for program, args in calls: invoke(program, args, (ids.numel(),))
        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): run()
        buffers = [mapping, local, indices, counts, ids, *outputs.values(), updates.owner.host_view,
                   *(t.slab for t in tiers if t)]
        pointers = tuple(t.data_ptr() for t in buffers)
        def verify():
            for out in outputs.values(): out.fill_(165)
            # Collect earlier test graphs before measuring this graph. Their
            # delayed Python-cycle cleanup can otherwise count unrelated frees.
            gc.collect()
            before = torch.cuda.memory_stats()
            with kernel_resolution_guard("fixed-slot graph replay"):
                graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for key in ("allocation.all.allocated", "allocation.all.freed", "allocated_bytes.all.current"):
                assert before[key] == after[key], key
            for name, out in outputs.items():
                expected = torch.full_like(out.cpu(), 165)
                for route, expert in enumerate(ids.cpu().flatten().tolist()):
                    if 0 <= expert < 4: expected[route].copy_(original[expert][name].flatten())
                torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
            assert pointers == tuple(t.data_ptr() for t in buffers)
        verify()
        side = torch.cuda.Stream(device=device)
        drained = torch.cuda.Event()
        for _ in range(3):
            snapshot = updates.snapshot()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                graph.replay()
                drained.record()
            with kernel_resolution_guard("quiescent copies resolve no kernels"):
                updates.exchange(((0, 1), (2, 3)), expected=snapshot, quiescent=True)
            assert drained.query()
            verify()
            ids.copy_(torch.tensor([[3, 2, 1], [0, 3, 2**40 if dtype == torch.int64 else -1]], dtype=dtype))
            verify()
        # Fail after enqueueing publication. Rollback must restore data as well
        # as the map before this same graph is allowed to run again.
        snapshot = updates.snapshot()
        copy, failed = updates.transfer.copy, False
        def fault(destination, source):
            nonlocal failed
            copy(destination, source)
            if destination.data_ptr() == mapping.data_ptr() and not failed:
                failed = True
                raise OSError("injected publication failure")
        monkeypatch.setattr(updates.transfer, "copy", fault)
        with pytest.raises(ResidencyUpdateError) as caught:
            updates.exchange(((0, 1),), expected=snapshot, quiescent=True)
        assert caught.value.resumable and updates.snapshot() == snapshot
        verify()
    finally:
        torch.cuda.synchronize()
        if graph is not None: graph.reset()
        if updates is not None: updates.owner.close()
        for tier in tiers:
            if tier is not None and tier.owner is not None: tier.owner.close()
