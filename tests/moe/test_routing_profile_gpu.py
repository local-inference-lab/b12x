"""Physical Blackwell gates for prepared counters and allocator-free replay."""
from dataclasses import replace
import gc

import pytest
import torch

from b12x.moe.fused_moe.routing_profile import (
    RoutingProfileQuery, bind_routing_profile, plan_routing_profile, routing_profile_state,
)
from b12x.preparation import PreparationSession, PreparedCall


def prepare(query):
    from tests.conftest import require_sm103_or_sm12x
    require_sm103_or_sm12x()
    plan = plan_routing_profile(query)
    ids = torch.zeros((query.max_tokens, query.max_top_k), dtype=torch.int64, device="cuda")
    def prime(state):
        calls = [state.bind(layer=layer, phase=phase, topk_ids=ids)
                 for layer, _ in query.layers for phase in query.phases]
        def run():
            if query.runtime_token_limit:
                state.set_token_limit(query.max_tokens)
            for binding in calls: binding.run()
        return PreparedCall(run=run, output=state.storage, owners=(state, *calls))
    session = PreparationSession(device="cuda:0", autotune=False, compile_workers=0)
    result = session.prepare((plan.request(name="counter", prepare_call=prime),))
    session.freeze()
    return plan, session, result


def test_engine_phase_limit_changes_under_same_graph_without_host_reads():
    from b12x._lib.runtime_control import kernel_resolution_guard
    query = RoutingProfileQuery(layers=(("layer", 4),), max_tokens=4,
        max_top_k=2, runtime_token_limit=True)
    plan, session, result = prepare(query)
    graph = None
    try:
        state = routing_profile_state(plan)
        ids = torch.tensor([[0,0], [1,2], [3,3], [3,3]], device="cuda", dtype=torch.int64)
        observer = bind_routing_profile(plan, layer="layer", phase="decode", topk_ids=ids)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            observer.run()
        state.reset(quiescent=True)
        pointer = state.storage.data_ptr()
        for live in (0, 1, 2, 0):
            before = torch.cuda.memory_stats()
            with kernel_resolution_guard("runtime profile extent"):
                state.set_token_limit(live)
                graph.replay()
            torch.cuda.synchronize()
            assert before["allocation.all.allocated"] == torch.cuda.memory_stats()["allocation.all.allocated"]
        row = state.snapshot(quiescent=True).layers[0]
        assert row.counts == (4, 1, 1, 0)
        assert row.calls == 2 and row.tokens == 3
        assert state.storage.data_ptr() == pointer
    finally:
        if graph is not None:
            graph.reset()
        result.close()
        session.close()


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_counter_exact_graph_capacity_controls_and_overflow(dtype, monkeypatch):
    query = RoutingProfileQuery(layers=(("a", 4), ("b", 7)), phases=("decode", "verify"), max_tokens=128, max_top_k=8)
    plan, session, result = prepare(query)
    try:
        state = routing_profile_state(plan)
        pointers = (state.storage.data_ptr(), *(x.data_ptr() for x in state.rows.values()))
        with pytest.raises(ValueError, match="overlap"):
            state.bind(layer="a", phase="decode", topk_ids=state.storage[:16].view(dtype).view(1, -1))
        import cutlass.cute as cute
        def forbidden(*a, **k): pytest.fail("runtime resolved or allocated a program")
        monkeypatch.setattr(cute, "compile", forbidden)
        from b12x.moe.fused_moe import routing_profile as implementation
        monkeypatch.setattr(implementation, "compile_programs", forbidden)
        for m, k in ((1, 1), (1, 6), (2, 3), (4, 8), (8, 3), (16, 8), (32, 6), (64, 8), (128, 6)):
            ids = torch.empty((m, k), dtype=dtype, device="cuda")
            binding = bind_routing_profile(plan, layer="a", phase="decode", topk_ids=ids)
            ids.zero_()
            binding.run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): binding.run()
            try:
                for fill in (0, 3, -1, 2**40 if dtype == torch.int64 else 2**30):
                    state.reset(quiescent=True)
                    state.set_enabled(True, quiescent=True)
                    ids.fill_(fill)
                    # Retire unrelated graph-owner cycles before measuring replay.
                    gc.collect()
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_stats()
                    for _ in range(3): graph.replay()
                    torch.cuda.synchronize()
                    after = torch.cuda.memory_stats()
                    for key in ("allocated_bytes.all.current", "allocation.all.allocated", "allocation.all.freed"):
                        assert before[key] == after[key]
                    row = state.snapshot(quiescent=True).layers[0]
                    assert row.counts == tuple(3*m*k if e == fill else 0 for e in range(4))
                    assert row.calls == row.sampled_calls == 3
                    assert row.tokens == row.sampled_tokens == 3*m
                    state.set_enabled(False, quiescent=True)
                    graph.replay()
                    assert state.snapshot(quiescent=True).layers[0] == row
                state.reset(quiescent=True)
                state.set_enabled(True, quiescent=True)
                ids.copy_((torch.arange(m*k, device="cuda").reshape(m, k) % 7 - 1).to(dtype))
                graph.replay()
                expected = tuple(int((ids.cpu() == e).sum()) for e in range(4))
                assert state.snapshot(quiescent=True).layers[0].counts == expected
                assert pointers == (state.storage.data_ptr(), *(x.data_ptr() for x in state.rows.values()))
            finally:
                graph.reset()
        state.reset(quiescent=True)
        ids = torch.zeros((1, 1), device="cuda", dtype=dtype)
        binding = bind_routing_profile(plan, layer="a", phase="decode", topk_ids=ids)
        # Assign all-one bits through an int64 view because Torch scalar fills
        # reject Python values larger than signed int64 on some toolchains.
        state.rows["a", "decode"].view(torch.int64)[0] = -1
        binding.run()
        with pytest.raises(OverflowError): state.snapshot(quiescent=True)
        state.reset(quiescent=True)
        assert state.snapshot(quiescent=True).layers[0].counts == (0, 0, 0, 0)
        with pytest.raises(RuntimeError, match="pause graph"): state.snapshot()
    finally:
        result.close()
        session.close()
    with pytest.raises(RuntimeError, match="released"): binding.run()


def test_sampling_layer_phase_and_tp_ownership():
    query = RoutingProfileQuery(layers=(("a", 4),), phases=("decode", "prefill", "verify"),
        max_tokens=16, max_top_k=3, sample_every=3, rank=0, tp_size=2)
    plan, session, result = prepare(query)
    try:
        state = routing_profile_state(plan)
        state.reset(quiescent=True)
        ids = torch.tensor([[0, 0, 3], [1, -1, 2**40]], device="cuda", dtype=torch.int64)
        bindings = [bind_routing_profile(plan, layer="a", phase=p, topk_ids=ids) for p in query.phases]
        for _ in range(7): bindings[0].run()
        for _ in range(2): bindings[1].run()
        rows = state.snapshot(quiescent=True).layers
        assert rows[0].counts == (6, 3, 0, 3) and rows[0].sampled_calls == 3 and rows[0].calls == 7
        assert rows[1].counts == (2, 1, 0, 1) and rows[1].sampled_calls == 1 and rows[1].calls == 2
        assert rows[2].counts == (0, 0, 0, 0)
    finally:
        result.close()
        session.close()
    plan, session, result = prepare(replace(query, rank=1))
    try:
        state = routing_profile_state(plan)
        assert state.storage is None and not state.programs
        bind_routing_profile(plan, layer="a", phase="decode", topk_ids=ids).run()
        assert state.snapshot(quiescent=True).layers == ()
    finally:
        result.close()
        session.close()


def test_concurrent_counter_producers_are_atomic():
    plan, session, result = prepare(RoutingProfileQuery(layers=(("a", 4),), max_tokens=128, max_top_k=8))
    try:
        state = routing_profile_state(plan)
        ids = torch.zeros((128, 8), device="cuda", dtype=torch.int32)
        binding = bind_routing_profile(plan, layer="a", phase="decode", topk_ids=ids)
        state.reset(quiescent=True)
        streams = [torch.cuda.Stream() for _ in range(2)]
        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(10): binding.run()
        row = state.snapshot(quiescent=True).layers[0]
        assert row.counts == (20*128*8, 0, 0, 0) and row.calls == 20
    finally:
        result.close()
        session.close()


def test_worker_calibration_replay_to_static_artifact(tmp_path):
    from b12x.integration.vllm.expert_residency import ExpertResidencyWorker
    from b12x.moe.fused_moe.automatic import (
        AutomaticResidencyConfig, ModelExpertMemoryBudget, ResidencyCalibrationConfig,
        ResidencyController, ResidencyHardware, ResidencyLayerSpec, ResidencyModelSpec, ResidencyProfileStore,
    )
    from tests.conftest import require_sm103_or_sm12x
    require_sm103_or_sm12x()
    spec = ResidencyLayerSpec(layer="layer", experts=4, hidden=256, intermediate=256, max_tokens=8, max_top_k=3)
    model = ResidencyModelSpec(checkpoint_fingerprint="portable-counter-test", layers=(spec,))
    cfg = AutomaticResidencyConfig(mode="auto", workload="portable-counter-test",
        calibration=ResidencyCalibrationConfig(minimum_observations=12, convergence_window=12, stable_windows=2))
    memory = spec.memory(1)
    budget = ModelExpertMemoryBudget(hbm_bytes=memory.hbm_total_bytes+1024, grace_bytes=memory.grace_expert_bytes)
    def controller():
        # Host deployment metadata exercises decisions only. This test runs no
        # SM103 expert GEMM and allocates no Grace expert slabs on SM120.
        return ResidencyController(config=cfg, model=model, budget=budget,
            hardware=ResidencyHardware(compute_capability=(10, 3), grace_coherent=True),
            store=ResidencyProfileStore(tmp_path))
    worker = ExpertResidencyWorker(controller())
    assert worker.startup().state == "calibrating"
    with PreparationSession(device="cuda:0", autotune=False, compile_workers=0) as session:
        session.prepare((worker.preparation_request(),))
        session.freeze()
        ids = torch.tensor([[0, 0, 1], [0, 2, -1]], dtype=torch.int64, device="cuda")
        observer = worker.bind_routes(layer="layer", phase="decode", topk_ids=ids)
        assert worker.bind_routes(layer="layer", phase="prefill", topk_ids=ids) is None
        observer.run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): observer.run()
        try:
            worker.begin_calibration(quiescent=True)
            for window in range(3):
                for _ in range(3): graph.replay()
                progress = worker.poll(request_count=(window+1)*3, token_count=(window+1)*6, quiescent=True)
            assert progress.state == "restart_required"
            assert progress.profile.placements[0].hbm_expert_ids == (0,)
            assert worker.controller.active is None
            after = worker.snapshot_counters(quiescent=True)
            graph.replay()
            assert worker.snapshot_counters(quiescent=True) == after
        finally:
            graph.reset()
    next_worker = ExpertResidencyWorker(controller())
    assert next_worker.startup().state == "ready"
    assert next_worker.preparation_request() is None


def test_operator_timing_graph_batches_device_repetitions():
    """Timing graphs must contain their repetitions rather than Python gaps."""
    from benchmarks.moe.expert_residency import capture, time_graph
    plan, session, _ = prepare(RoutingProfileQuery(layers=(("a", 4),), max_tokens=1, max_top_k=1))
    graph = None
    try:
        ids = torch.zeros((1, 1), dtype=torch.int64, device="cuda")
        binding = bind_routing_profile(plan, layer="a", phase="decode", topk_ids=ids)
        graph = capture(binding.run, repetitions=7)
        state = routing_profile_state(plan)
        state.reset(quiescent=True)
        state.set_enabled(True, quiescent=True)
        timing = time_graph(graph, iterations=7, samples=3)
        snapshot = state.snapshot(quiescent=True).layers[0]
        assert snapshot.counts == (21, 0, 0, 0) and snapshot.calls == 21
        assert len(timing["raw_us"]) == 3 and timing["median_us"] > 0
    finally:
        if graph is not None: graph.reset()
        session.close()
