"""Prepared launchers remain valid after compiler caches release their copies."""

import gc

import torch

from b12x._lib import compiler
from b12x.preparation import PreparationSession, PreparedCall
from ..conftest import require_b12x


def test_graph_replay_survives_compiler_cache_eviction(monkeypatch):
    from b12x.gemm import bf16_gemv

    device = require_b12x()
    source = torch.randn(2, 2048, device=device, dtype=torch.bfloat16)
    weight = torch.randn(96, 2048, device=device, dtype=torch.bfloat16)
    plan = bf16_gemv.plan(bf16_gemv.query_from_call(source, weight))
    request = plan.request(
        name="gemv",
        prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(source, weight)
        ),
    )

    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        session.prepare((request,))
        graph = torch.cuda.CUDAGraph()
        try:
            with session.capture(), torch.cuda.graph(graph):
                output = bf16_gemv.mm(source, weight, plan=plan)

            compiler.clear_compile_cache()
            gc.collect()

            def forbidden(*args, **kwargs):
                raise AssertionError(
                    "prepared graph replay reached compiler or loader"
                )

            monkeypatch.setattr(compiler, "compile", forbidden)
            monkeypatch.setattr(
                compiler, "_load_cute_compile_from_disk", forbidden
            )
            changed = torch.randn_like(source)
            source.copy_(changed)
            output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(device)

            expected = changed.float() @ weight.float().T
            assert torch.isfinite(output).all()
            assert torch.count_nonzero(output) > 0
            torch.testing.assert_close(
                output, expected.bfloat16(), rtol=2e-2, atol=2e-2
            )
        finally:
            graph.reset()


def test_candidate_graphs_preserve_outputs_without_global_allocator_flushes(monkeypatch):
    from b12x.gemm import bf16_gemv
    from b12x.preparation import _measurement

    device = require_b12x()
    weight = torch.randn(96, 2048, device=device, dtype=torch.bfloat16)
    sources = [torch.ones(rows, 2048, device=device, dtype=torch.bfloat16) for rows in (2, 3)]
    plans = [bf16_gemv.plan(bf16_gemv.query_from_call(source, weight)) for source in sources]
    requests = [
        plan.request(
            name=f"candidate-{index}",
            prepare_call=lambda state, source=source: PreparedCall(
                run=lambda: state.run(source, weight),
            ),
        )
        for index, (source, plan) in enumerate(zip(sources, plans))
    ]
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare(requests)
        calls = [
            PreparedCall(
                run=lambda source=source, plan=plan: bf16_gemv.mm(source, weight, plan=plan),
                produce=lambda source=source: source.add_(0.125),
                owners=(source, weight),
            )
            for source, plan in zip(sources, plans)
        ]

        def forbidden(*_args, **_kwargs):
            raise AssertionError("candidate graph construction flushed the process allocator")

        with monkeypatch.context() as guards:
            guards.setattr(torch.cuda, "empty_cache", forbidden)
            guards.setattr(torch._C, "_host_emptyCache", forbidden)
            race = _measurement._prepare_race(calls, device_ordinal=torch.cuda.current_device(), samples=8)
        try:
            pointers = [call.output.data_ptr() for call in calls]
            assert len(set(pointers)) == len(calls)
            for initial in (1.0, 3.0):
                for source, call in zip(sources, calls):
                    source.fill_(initial)
                    with torch.inference_mode():
                        call.output.fill_(float("nan"))
                torch.cuda.synchronize(device)
                allocated = torch.cuda.memory_allocated(device)
                for timer in race.timers:
                    timer.replay()
                torch.cuda.synchronize(device)
                assert torch.cuda.memory_allocated(device) == allocated
                assert [call.output.data_ptr() for call in calls] == pointers
                for source, call, timer in zip(sources, calls, race.timers):
                    torch.testing.assert_close(source, torch.full_like(source, initial + 1.0))
                    expected = source.float() @ weight.float().T
                    assert torch.isfinite(call.output).all()
                    assert torch.count_nonzero(call.output) > 0
                    torch.testing.assert_close(call.output, expected.bfloat16(), rtol=2e-2, atol=2e-2)
                    assert all(value > 0 for value in timer.samples())
        finally:
            race.close()


def test_retired_candidate_pools_do_not_accumulate_or_release_live_graphs(monkeypatch):
    from b12x.preparation import _measurement
    from b12x.preparation._memory import release_graph_pool_cache

    device = require_b12x()
    ordinal = torch.cuda.current_device()
    source = torch.ones(1024, 1024, device=device)
    live_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(live_graph):
        live_output = source + 11

    # Keep a reusable default-pool allocation larger than candidate temporaries.
    cached = torch.empty(16 << 20, device=device)
    cached_pointer = cached.data_ptr()
    del cached

    def forbidden(*_args, **_kwargs):
        raise AssertionError("candidate cleanup flushed the process allocator")

    reserved = []
    try:
        with monkeypatch.context() as guards:
            guards.setattr(torch.cuda, "empty_cache", forbidden)
            guards.setattr(torch._C, "_host_emptyCache", forbidden)
            for batch in range(8):
                calls = [PreparedCall(run=source.clone, produce=lambda: source.add_(0.125))
                         for _ in range(2)]
                race = _measurement._prepare_race(calls, device_ordinal=ordinal, samples=8)
                try:
                    for timer in race.timers:
                        timer.replay()
                    torch.cuda.synchronize(device)
                    for call in calls:
                        assert torch.isfinite(call.output).all()
                        call.output = None
                finally:
                    race.close()
                reserved.append(torch.cuda.memory_reserved(device))
                del race, calls
                source.fill_(batch)
                live_graph.replay()
                torch.cuda.synchronize(device)
                torch.testing.assert_close(live_output, torch.full_like(source, batch + 11))

            cached = torch.empty(16 << 20, device=device)
            assert cached.data_ptr() == cached_pointer
            assert len(set(reserved[2:])) == 1, reserved
    finally:
        pool = live_graph.pool()
        live_graph.reset()
        del live_output
        release_graph_pool_cache(pool)


def test_candidate_pool_reuse_preserves_carried_outputs_and_bounds_residency(monkeypatch):
    from b12x.preparation import _measurement
    from b12x.preparation._memory import release_graph_pool_cache
    from b12x.preparation.types import _prime

    device = require_b12x()
    ordinal = torch.cuda.current_device()
    source = torch.ones(1024, 1024, device=device)
    stream = torch.cuda.Stream(device=device)
    pools, reserved = [], []
    champion = None
    live_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(live_graph):
        live_output = source + 11

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a candidate batch evicted allocator storage")

    try:
        with monkeypatch.context() as guards:
            guards.setattr(torch.cuda, "empty_cache", forbidden)
            guards.setattr(torch._C, "_host_emptyCache", forbidden)
            guards.setattr(_measurement, "_prime", forbidden)
            guards.setattr("b12x.preparation._memory.release_graph_pool_cache", forbidden)
            for batch in range(8):
                calls = [] if champion is None else [champion]
                while len(calls) < 2:
                    call = PreparedCall(run=source.clone, produce=lambda: source.add_(0.125))
                    _prime(call)
                    calls.append(call)
                race = _measurement._prepare_race(
                    calls, device_ordinal=ordinal, samples=2, primed=True,
                    graph_pools=pools, capture_stream=stream,
                )
                pointers = [call.output.data_ptr() for call in calls]
                assert len(set(pointers)) == 2
                source.fill_(batch)
                with torch.inference_mode():
                    for call in calls:
                        call.output.fill_(float("nan"))
                allocated = torch.cuda.memory_allocated(device)
                order = (0, 1) if batch % 2 == 0 else (1, 0)
                for index in order:
                    race.timers[index].replay()
                torch.cuda.current_stream(device).synchronize()
                assert torch.cuda.memory_allocated(device) == allocated
                assert [call.output.data_ptr() for call in calls] == pointers
                for position, index in enumerate(order):
                    torch.testing.assert_close(calls[index].output, torch.full_like(source, batch + 0.25 * (position + 1)))
                race.close()
                assert len(pools) == 2
                champion = calls[batch % 2]
                calls[1 - batch % 2].output = None
                del race, calls, call
                reserved.append(torch.cuda.memory_reserved(device))
                source.fill_(batch)
                live_graph.replay()
                torch.cuda.current_stream(device).synchronize()
                torch.testing.assert_close(live_output, torch.full_like(source, batch + 11))
            assert len(set(reserved[3:])) == 1, reserved
    finally:
        if champion is not None:
            champion.output = None
        for pool in pools:
            pool.close()
        pool = live_graph.pool()
        live_graph.reset()
        del live_output
        release_graph_pool_cache(pool)
