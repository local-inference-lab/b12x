"""Observation -> decision -> exchange -> same-graph execution qualification."""
import gc

import pytest
import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.moe import fused_moe as moe
from b12x.moe.fused_moe._residency_storage import materialize_tier
from b12x.moe.fused_moe._residency_updates import materialize_updates
from b12x.preparation import PreparationSession, PreparedCall
from tests.moe.test_expert_residency import declaration, placement
from tests.moe.test_residency_updates_gpu import reader
from tests.moe.test_residency_kernels import invoke, programs
from tests.moe.test_sm103_residency import prepare, randomize, require_device


@pytest.mark.parametrize("native", [False, True], ids=["portable_bytes", "native_sm103"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_miss_served_then_promoted_without_recapture(native, dtype, tmp_path):
    if native:
        device = require_device(grace=True)
    else:
        from tests.conftest import require_sm103_or_sm12x
        require_sm103_or_sm12x()
        device = torch.device("cuda", torch.cuda.current_device())
    profile = placement()
    plan, weights = declaration(max_tokens=2, updates=moe.ResidencyUpdateCapacity(max_pairs=1))
    randomize(weights)
    query = moe.RoutingProfileQuery(layers=((profile.layer, 4),), max_tokens=2, max_top_k=3)
    counter = moe.plan_routing_profile(query)
    ids = torch.zeros(2, 3, dtype=dtype, device=device)
    a = torch.randn(2, 256, dtype=torch.bfloat16, device=device)*.1
    route_weights = torch.full((2, 3), 1/3, device=device)
    tiers, updates, graph = [], None, None
    with PreparationSession(device=device, autotune=False, compile_workers=0, cache_dir=tmp_path) as session:
        def counter_call(state):
            binding = state.bind(layer=profile.layer, phase="decode", topk_ids=ids)
            return PreparedCall(run=binding.run, owners=(state, binding))
        session.prepare((counter.request(name="counter", prepare_call=counter_call),))
        controls = moe.routing_profile_state(counter)
        observer = moe.bind_routing_profile(counter, layer=profile.layer, phase="decode", topk_ids=ids)
        try:
            if native:
                prepare(session, plan, a, ids, route_weights, "cache")
                control, control_weights = declaration(max_tokens=2, profile=placement(hot=(0, 1, 2, 3), cold=()))
                randomize(control_weights)
                prepare(session, control, a, ids, route_weights, "control")
                binding = moe.bind(plan, a=a, topk_ids=ids, topk_weights=route_weights)
                baseline = moe.bind(control, a=a, topk_ids=ids, topk_weights=route_weights)
                state = plan.prepared.state
                slot_snapshot = lambda: moe.residency_slot_snapshot(plan)
                exchange = lambda d: moe.exchange_expert_slots(plan, d.pairs, expected=d.expected, quiescent=True)
                local, indices, counts = (state.workspace[n] for n in ("local_ids", "indices", "counts"))
                def execute(): moe.run(binding=binding)
                def check():
                    torch.testing.assert_close(binding.output, moe.run(binding=baseline), atol=0, rtol=0)
                    assert torch.count_nonzero(binding.output) and torch.isfinite(binding.output).all()
                buffers = [state.mapping, state.slab, *(t.slab for t in state.tiers), a, route_weights]
            else:
                for tier, expert_ids in enumerate((profile.hbm_expert_ids, profile.grace_expert_ids)):
                    tiers.append(materialize_tier(expert_ids, weights, plan.query, device, grace=tier == 1))
                mapping = torch.tensor(profile.expert_map, dtype=torch.int32, device=device)
                updates = materialize_updates(plan.query, tuple(tiers), mapping, profile, device)
                original = {e: {name: value[row].cpu().clone() for name, value in tiers[tier].fields.items()}
                            for e, (tier, row) in enumerate(profile.expert_map)}
                local = torch.empty(2, 32, device=device, dtype=torch.int32)
                indices = torch.empty_like(local)
                counts = torch.empty(8, device=device, dtype=torch.int32)
                outputs = {name: torch.empty(6, value.numel(), dtype=torch.uint8, device=device)
                           for name, value in original[0].items()}
                partition = programs(dtype)[0]
                calls = [(reader(outputs[name].shape[1]), (value, local[tier], indices[tier], counts[tier*4:], outputs[name]))
                         for tier, storage in enumerate(tiers) for name, value in storage.fields.items()]
                def execute():
                    invoke(partition, (ids, mapping, local, indices, counts), (6,))
                    for program, args in calls: invoke(program, args, (6,))
                def check():
                    for name, output in outputs.items():
                        actual = output.cpu()
                        for route, expert in enumerate(ids.cpu().flatten().tolist()):
                            if 0 <= expert < 4:
                                torch.testing.assert_close(actual[route], original[expert][name].flatten(), atol=0, rtol=0)
                slot_snapshot = updates.snapshot
                exchange = lambda d: updates.exchange(d.pairs, expected=d.expected, quiescent=True)
                buffers = [mapping, local, indices, counts, *outputs.values(), *(t.slab for t in tiers)]
            def run():
                execute()  # A cold selection is serviced before any policy decision.
                observer.run()
            run()
            torch.cuda.synchronize()
            session.freeze()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): run()
            controls.reset(quiescent=True)  # Discard warmup/capture observations.
            policy = moe.ResidencyCacheController(
                spec=moe.ResidencyLayerSpec(layer=profile.layer, experts=4, hidden=256,
                    intermediate=256, max_tokens=2, max_top_k=3),
                config=moe.ResidencyCacheConfig(max_pairs=1, minimum_cold_selections=2,
                    minimum_score_gain=2, minimum_residency_windows=2),
                counter_query=query, slots=slot_snapshot(), baseline=controls.snapshot(quiescent=True))
            buffers += [ids, controls.storage]
            pointers = tuple(x.data_ptr() for x in buffers)
            exchanges = 0
            # Two windows establish a cold prior; later windows change traffic.
            for expert in (1, 1, 1, 3, 3, 0, 0):
                ids.copy_(torch.tensor([[expert, expert, expert], [expert, -1, 2**40 if dtype == torch.int64 else -1]], dtype=dtype))
                a.mul_(.875)
                before_slots = slot_snapshot()
                gc.collect()
                before = torch.cuda.memory_stats()
                with kernel_resolution_guard("observed cache graph replay"):
                    graph.replay()
                    graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for key in ("allocation.all.allocated", "allocation.all.freed", "allocated_bytes.all.current"):
                    assert before[key] == after[key], key
                check()
                assert counts[before_slots.expert_map[expert][0]*4].cpu().item() == 4
                decision = policy.observe(controls.snapshot(quiescent=True), slots=before_slots)
                assert decision.counts[expert] == 8
                assert decision.cold_selections == (8 if before_slots.expert_map[expert][0] else 0)
                result = before_slots
                if decision.pairs:
                    with kernel_resolution_guard("cache exchange reuses prepared storage"):
                        result = exchange(decision)
                    assert result.expert_map[expert][0] == 0
                    exchanges += 1
                outcome = policy.finish(decision, slots=result)
                assert pointers == tuple(x.data_ptr() for x in buffers)
            assert exchanges == 3 and outcome.promotions == 3
            assert outcome.observed_hits_after_promotion >= 24
            assert profile == placement()  # Runtime generations do not rewrite the prior.
        finally:
            torch.cuda.synchronize()
            if graph is not None: graph.reset()
            if updates is not None: updates.owner.close()
            for tier in tiers:
                if tier.owner is not None: tier.owner.close()
