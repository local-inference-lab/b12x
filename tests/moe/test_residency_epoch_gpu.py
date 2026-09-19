"""Two-layer same-graph epoch contract using portable byte payload readers.

Exercises native partition/counter kernels and real mapped-host transactions.
The async engine harness supplies a pause boundary; it is not vLLM serving or
SM103 expert MMA/TMA qualification.
"""
import asyncio
import gc
from types import SimpleNamespace

import pytest
import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.integration.vllm.residency_epoch import (
    ResidencyEpochRuntime, ResidencyEpochWorkerExtension, ResidencyLayerBinding,
    VllmResidencyEpochs,
)
from b12x.moe import fused_moe as moe, residency as r
from b12x.moe.fused_moe._residency_storage import materialize_tier
from b12x.moe.fused_moe._residency_updates import materialize_updates
from b12x.preparation import PreparationSession, PreparedCall
from tests.moe.test_expert_residency import declaration, placement
from tests.moe.test_residency_kernels import invoke, programs
from tests.moe.test_residency_updates_gpu import reader
from tests.moe.test_vllm_residency_epoch import Engine, memory


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("bounded", [False, True], ids=["full", "bounded"])
def test_model_epoch_same_graph_payloads_counters_and_pointers(dtype, bounded, tmp_path):
    from tests.conftest import require_sm103_or_sm12x
    require_sm103_or_sm12x()
    device = torch.device("cuda", torch.cuda.current_device())
    query = moe.RoutingProfileQuery(layers=(("a", 4), ("b", 4)), max_tokens=2, max_top_k=3)
    counter = moe.plan_routing_profile(query)
    ids = {name: torch.zeros(2, 3, dtype=dtype, device=device) for name in ("a", "b")}
    resources, graph = [], None
    with PreparationSession(device=device, autotune=False, compile_workers=0, cache_dir=tmp_path) as session:
        def prime(state):
            observers = [state.bind(layer=n, phase="decode", topk_ids=v) for n, v in ids.items()]
            return PreparedCall(run=lambda: [o.run() for o in observers], owners=(state, *observers))
        session.prepare((counter.request(name="model-counters", prepare_call=prime),))
        controls = moe.routing_profile_state(counter)
        try:
            bindings, operations = {}, []
            for layer_index, name in enumerate(ids):
                plan, weights = declaration(max_tokens=2, updates=moe.ResidencyUpdateCapacity(max_pairs=1))
                for expert in range(4):
                    for value in (weights.w13, weights.w2, weights.w13_block_scales, weights.w2_block_scales):
                        value[expert].fill_(expert*23 + layer_index*7 + 1)
                tiers = [materialize_tier(experts, weights, plan.query, device, grace=tier == 1)
                         for tier, experts in enumerate(((0, 2), (1, 3)))]
                mapping = torch.tensor(placement().expert_map, dtype=torch.int32, device=device)
                updates = materialize_updates(plan.query, tuple(tiers), mapping, placement(), device)
                original = {e: {key: value[row].cpu().clone() for key, value in tiers[tier].fields.items()}
                            for e, (tier, row) in enumerate(placement().expert_map)}
                local = torch.empty(2, 32, device=device, dtype=torch.int32)
                indices = torch.empty_like(local)
                counts = torch.empty(8, device=device, dtype=torch.int32)
                # The bounded sanitizer case probes scales and the complete
                # control protocol. The full case reads every payload byte.
                output = {key: torch.empty(6, value.numel(), dtype=torch.uint8, device=device)
                          for key, value in original[0].items() if not bounded or key == "s2"}
                partition = programs(dtype)[0]
                calls = [(reader(output[key].shape[1]), (value, local[tier], indices[tier], counts[tier*4:], output[key]))
                         for tier, storage in enumerate(tiers) for key, value in storage.fields.items() if key in output]
                observer = moe.bind_routing_profile(counter, layer=name, phase="decode", topk_ids=ids[name])
                buffers = [mapping, local, indices, counts, ids[name], controls.storage,
                           *output.values(), *(t.slab for t in tiers)]
                def execute(partition=partition, layer_ids=ids[name], mapping=mapping, local=local,
                            indices=indices, counts=counts, calls=calls, observer=observer):
                    invoke(partition, (layer_ids, mapping, local, indices, counts), (6,))
                    for program, args in calls:
                        invoke(program, args, (6,))
                    observer.run()
                operations.append(execute)
                bindings[name] = ResidencyLayerBinding(observations=r.RoutingObservationSpec(
                    layer=name, experts=4, phase="decode", max_top_k=3),
                    exchange=r.ResidencyExchangeSpec(backend="portable_byte_exchange",
                        direct_backing_execution=True, fixed_address_quiescent_exchange=True,
                        payload_copy_bytes_per_pair=4*sum(v.numel() for v in original[0].values()),
                        map_copy_bytes_per_transaction=64), max_pairs=1,
                    snapshot=updates.snapshot, apply=updates.exchange,
                    pointers=lambda buffers=buffers: tuple(v.data_ptr() for v in buffers),
                    validate=updates.require_healthy)
                resources.append((name, tiers, updates, original, output))
            def execute():
                for operation in operations:
                    operation()
            execute()
            torch.cuda.synchronize()
            session.freeze()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                execute()
            controls.reset(quiescent=True)
            engine = Engine(ranks=1)
            runtime = ResidencyEpochRuntime(rank=0, owner_rank=0, bindings=bindings,
                snapshot_counters=controls.snapshot, checkpoint_id="portable-byte-checkpoint",
                initial_profile_id="portable-static-profile", memory=memory())
            worker = ResidencyEpochWorkerExtension()
            worker.model_runner = SimpleNamespace(b12x_residency_runtime=runtime)
            engine.workers = [worker]
            controller = VllmResidencyEpochs(engine,
                configs={n: r.ResidencyCacheConfig(max_pairs=1, minimum_cold_selections=2,
                    minimum_score_gain=1, minimum_residency_windows=1, scoring="decayed_lfu") for n in ids},
                budget=r.ResidencyEpochBudget(max_pairs=1, max_copy_bytes=2**30))

            async def run():
                await controller.run()
                promotions = 0
                windows = 2 if bounded else 16
                for window in range(windows):
                    for index, name in enumerate(ids):
                        expert = (window//2 + index) % 4
                        invalid = 2**40 if dtype == torch.int64 else -1
                        values = [[expert, expert, -1], [expert, invalid, expert]]
                        if window % 2:
                            values[1] = [-1, -1, -1]
                        ids[name].copy_(torch.tensor(values, dtype=dtype))
                    for _, _, _, _, output in resources:
                        for value in output.values():
                            value.fill_(165)
                    gc.collect()
                    before = torch.cuda.memory_stats()
                    with kernel_resolution_guard("model epoch graph replay"):
                        graph.replay()
                        graph.replay()
                    torch.cuda.synchronize()
                    after = torch.cuda.memory_stats()
                    for key in ("allocation.all.allocated", "allocation.all.freed", "allocated_bytes.all.current"):
                        assert before[key] == after[key], key
                    for name, _, _, original, output in resources:
                        for key, value in output.items():
                            expected = torch.full_like(value.cpu(), 165)
                            for route, expert in enumerate(ids[name].cpu().flatten().tolist()):
                                if 0 <= expert < 4:
                                    expected[route].copy_(original[expert][key].flatten())
                            torch.testing.assert_close(value.cpu(), expected, rtol=0, atol=0)
                    with kernel_resolution_guard("model epoch uses retained programs"):
                        receipt = await controller.run()
                    assert receipt["decision"]["selected_pairs"] <= 1
                    for layer in receipt["decision"]["layers"]:
                        assert sum(layer["decision"]["counts"]) == (4 if window % 2 else 8)
                    promotions += receipt["decision"]["selected_pairs"]
                assert promotions >= (1 if bounded else 6)
                assert engine.events.count("pause") == engine.events.count("resume") == windows + 1
            asyncio.run(run())
        finally:
            torch.cuda.synchronize()
            if graph is not None:
                graph.reset()
            for _, tiers, updates, _, _ in resources:
                updates.owner.close()
                for tier in tiers:
                    if tier.owner is not None:
                        tier.owner.close()
