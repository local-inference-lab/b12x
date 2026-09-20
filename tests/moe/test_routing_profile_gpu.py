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
            for binding in calls:
                binding.run()
            if state.health is not None:
                state.health.rebase(("prime",))
            if state.history is not None:
                state.history.rebase(("prime",))
                for _ in range(state.history.depth):
                    state.history.checkpoint(("prime",))

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


@pytest.mark.parametrize("experts", [7, 128, 385])
def test_health_reduction_exact_read_only_reset_generation_and_graph(experts):
    from b12x._lib.runtime_control import kernel_resolution_guard

    query = RoutingProfileQuery(
        layers=(("a", 4), ("b", experts)),
        max_tokens=8,
        max_top_k=4,
        health_summary=True,
    )
    plan, session, result = prepare(query)
    try:
        state = routing_profile_state(plan)
        state.reset(quiescent=True)
        health = state.health
        maps = {
            n: torch.tensor(
                [[e % 2, e] for e in range(count)], device="cuda", dtype=torch.int32
            )
            for n, count in query.layers
        }
        health.bind_maps(maps)
        health.rebase((0, 0))
        pointers = (
            state.storage.data_ptr(),
            health.previous.data_ptr(),
            health.output.data_ptr(),
            health.host.data_ptr(),
        )
        ids = torch.tensor(
            [[1, 1, -1, 2**40], [0, 3, 2, 1]], device="cuda", dtype=torch.int64
        )
        bindings = [
            bind_routing_profile(plan, layer=n, phase="decode", topk_ids=ids)
            for n, _ in query.layers
        ]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for binding in bindings:
                binding.run()

        def read(generation=(0, 0)):
            with kernel_resolution_guard("prepared health"):
                health.start(generation)
            health.done.synchronize()
            return health.poll(generation)

        assert read()["selections"] == 0
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        for _ in range(3):
            graph.replay()
        summary = read()
        assert before == torch.cuda.memory_stats()["allocation.all.allocated"]
        assert summary["selections"] == 36 and summary["cold_selections"] == 24
        assert all(
            r
            == dict(
                selections=18,
                cold_selections=12,
                cold_experts_once=0,
                cold_experts_repeated=2,
                repeated_cold_selections=10,
            )
            for r in summary["layers"].values()
        )
        snapshot = state.snapshot(quiescent=True)
        assert read()["selections"] == 0
        assert state.snapshot(quiescent=True) == snapshot
        health.start((0, 0))
        health.done.synchronize()
        with pytest.raises(RuntimeError, match="stale health"):
            health.poll((1, 0))
        health.poll((0, 0))
        maps["a"][:, 0].zero_()
        with pytest.raises(RuntimeError, match="stale health"):
            read((1, 0))
        health.rebase((1, 0))
        graph.replay()
        assert read((1, 0))["cold_selections"] == 4
        state.reset(quiescent=True)
        with pytest.raises(RuntimeError, match="stale health"):
            read((1, 0))
        health.rebase((1, 0))
        graph.replay()
        read((1, 0))
        state.rows["a", "decode"].zero_()  # Unannounced reset must also fail closed.
        with pytest.raises(ValueError, match="invalid health"):
            read((1, 0))
        assert pointers == (
            state.storage.data_ptr(),
            health.previous.data_ptr(),
            health.output.data_ptr(),
            health.host.data_ptr(),
        )
        graph.reset()
    finally:
        result.close()
        session.close()


def test_health_unsigned_totals_and_sticky_overflow_fail_closed():
    q = RoutingProfileQuery(
        layers=(("a", 4),), max_tokens=1, max_top_k=1, health_summary=True
    )
    plan, session, result = prepare(q)
    try:
        s = routing_profile_state(plan)
        s.reset(quiescent=True)
        h = s.health
        h.rebase((0,))
        s.rows["a", "decode"].view(torch.int64)[:3].fill_(2**62)
        h.start((0,))
        h.done.synchronize()
        assert h.poll((0,))["selections"] == 3 * 2**62
        s.reset(quiescent=True)
        h.rebase((0,))
        s.rows["a", "decode"].view(torch.int64)[:4].fill_(2**62)
        h.start((0,))
        h.done.synchronize()
        with pytest.raises(ValueError, match="invalid health"):
            h.poll((0,))
        s.reset(quiescent=True)
        h.rebase((0,))
        s.rows["a", "decode"].view(torch.int64)[8] = 1
        h.start((0,))
        h.done.synchronize()
        with pytest.raises(ValueError, match="invalid health"):
            h.poll((0,))
    finally:
        result.close()
        session.close()


@pytest.mark.parametrize("depth", [1, 2, 4, 8, 16])
def test_history_wrap_preserves_canonical_counts_without_replay_allocations(depth):
    from b12x._lib.runtime_control import kernel_resolution_guard
    query = RoutingProfileQuery(layers=(("a", 4), ("b", 7)), max_tokens=128,
                                max_top_k=4, history_depth=depth)
    plan, session, result = prepare(query)
    graph = None
    try:
        state = routing_profile_state(plan)
        history = state.history
        state.reset(quiescent=True)
        generation = (("a", "prepared-a", 0), ("b", "prepared-b", 0))
        history.rebase(generation)
        assert history.read(generation, quiescent=True)[0] == ()
        ids = torch.tensor([[0, 0, -1, 2**40], [3, 6, 3, -1]],
                           device="cuda", dtype=torch.int64)
        bindings = [bind_routing_profile(plan, layer=n, phase="decode", topk_ids=ids)
                    for n in ("a", "b")]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for binding in bindings:
                binding.run()
        state.reset(quiescent=True)
        history.rebase(generation)
        pointers = (state.storage.data_ptr(), history.storage.data_ptr(), history.host.data_ptr())
        gc.collect()
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()
        with kernel_resolution_guard("routing history checkpoints"):
            for _ in range(depth+3):
                graph.replay()
                history.checkpoint(generation)
        torch.cuda.synchronize()
        after = torch.cuda.memory_stats()
        for key in ("allocation.all.allocated", "allocation.all.freed", "allocated_bytes.all.current"):
            assert before[key] == after[key]
        snapshots, receipt = history.read(generation, quiescent=True)
        assert receipt["coalesced_checkpoints"] == 3
        assert receipt["retained"] == depth
        assert receipt["readback_bytes"] == depth*query.storage_bytes
        for count, snap in zip(range(4, depth+4), snapshots, strict=True):
            assert snap.layers[0].counts == (2*count, 0, 0, 2*count)
            assert snap.layers[1].counts == (2*count, 0, 0, 2*count, 0, 0, count)
            assert snap.layers[0].calls == count
        with pytest.raises(RuntimeError, match="quiescence"):
            history.read(generation)
        with pytest.raises(RuntimeError, match="stale"):
            history.checkpoint((("a", "foreign", 0), ("b", "prepared-b", 0)))
        newer = (("a", "prepared-a", 1), ("b", "prepared-b", 0))
        history.rebase(newer)
        assert history.read(newer, quiescent=True)[0] == ()
        ids.fill_(-1)
        graph.replay()
        history.checkpoint(newer)
        torch.cuda.synchronize()
        current, _ = history.read(newer, quiescent=True)
        assert current[0].layers[0].counts == snapshots[-1].layers[0].counts
        assert pointers == (state.storage.data_ptr(), history.storage.data_ptr(), history.host.data_ptr())
        state.reset(quiescent=True)
        with pytest.raises(RuntimeError, match="stale"):
            history.checkpoint(newer)
        history.rebase(newer)
        state.rows["a", "decode"][8] = 1  # Sticky counter-overflow flag.
        history.checkpoint(newer)
        torch.cuda.synchronize()
        with pytest.raises(OverflowError):
            history.read(newer, quiescent=True)
    finally:
        if graph is not None:
            graph.reset()
        result.close()
        session.close()
