"""Representative activation-producing races owned by preparation."""
from __future__ import annotations

import gc
import math
import statistics
from contextlib import contextmanager
from dataclasses import dataclass, field

from b12x._lib.compile_plan import (
    ProgramKey, forbid_lowering, observe_programs, record_program, retain_compiled_programs,
)
from .types import PreparedCall, _prime, _close_all

# Surviving candidates complete three scored rounds within the leader margin.
SURVIVOR_ROUNDS = 3
ELIMINATION_MARGIN = 1.10
DEFAULT_SAMPLES = 2
ROUND_BUDGET_US = 256.0


class _ParentCompilations:
    """Scoped actual-compilation counters and program keys, not launch tracing."""

    def __init__(self, *, cache_only):
        self.cache_only = cache_only
        self.cute = self.triton = 0

    def __enter__(self):
        from triton import knobs
        from b12x._lib import compiler
        self.compiler, self.knobs = compiler, knobs
        self.before = int(compiler.compile_cache_info()["compile_misses"])
        self.original_compile = compiler._call_cute_compile
        self.previous_listener = knobs.compilation.listener
        self.observation = observe_programs()
        self.programs = self.observation.__enter__()

        def listener(*args, **kwargs):
            metadata = kwargs.get("metadata", args[1] if len(args) > 1 else {})
            record_program(ProgramKey("triton", metadata["hash"], metadata.get("name", "")))
            if not kwargs.get("cache_hit", args[4] if len(args) > 4 else False):
                self.triton += 1
            if self.previous_listener is not None:
                self.previous_listener(*args, **kwargs)

        def reject_compile(*_args, **_kwargs):
            name = getattr(_kwargs.get("compile_spec"), "kernel_id", "unknown")
            raise RuntimeError(
                f"no-compilation phase encountered an unplanned CuTe program: {name} {_kwargs.get('cache_key')}"
            )

        knobs.compilation.listener = listener
        if self.cache_only:
            compiler._call_cute_compile = reject_compile
        return self

    def check(self):
        self.cute = int(self.compiler.compile_cache_info()["compile_misses"]) - self.before
        if self.cache_only and (self.cute or self.triton):
            raise RuntimeError(f"no-compilation phase compiled CuTe={self.cute}, Triton={self.triton}")

    def __exit__(self, kind, value, traceback):
        self.cute = int(self.compiler.compile_cache_info()["compile_misses"]) - self.before
        self.knobs.compilation.listener = self.previous_listener
        if self.cache_only:
            self.compiler._call_cute_compile = self.original_compile
        self.observation.__exit__(kind, value, traceback)


@contextmanager
def no_compilation():
    """Permit already-built object loads, but no new CuTe/Triton compilation."""
    with _ParentCompilations(cache_only=True) as observed, forbid_lowering():
        yield observed
        observed.check()


class _GraphPool:
    """Keep CUDA and pinned-host allocator pools owned between candidate graphs."""

    def __init__(self, stream):
        import torch

        # A bare pool token owns neither allocator's reference count. This graph
        # holds both counts without allocating sample storage or being replayed.
        self.graph = torch.cuda.CUDAGraph(keep_graph=True)
        self.event = torch.cuda.Event(external=True)
        with torch.cuda.stream(stream):
            self.graph.capture_begin()
            try:
                self.event.record()
            finally:
                self.graph.capture_end()
        self.id = self.graph.pool()

    def close(self):
        from ._memory import release_graph_pool_cache

        self.graph.reset()
        release_graph_pool_cache(self.id)


class _TimedCall:
    def __init__(self, call, eviction, samples, capture_stream, *, pool=None):
        import torch
        self.call, self.eviction = call, eviction
        self.events = tuple((torch.cuda.Event(enable_timing=True, external=True),
                             torch.cuda.Event(enable_timing=True, external=True)) for _ in range(samples))
        self.graph = None
        with retain_compiled_programs() as self._retained:
            if call.capture_safe:
                self.graph = torch.cuda.CUDAGraph()
                # A discarded CuTe executor can own a cyclic reference to its CUDA
                # library. Finalizing that cycle during capture invalidates it.
                collecting = gc.isenabled()
                gc.disable()
                try:
                    with torch.cuda.stream(capture_stream):
                        self.graph.capture_begin(pool=pool)
                        try:
                            self._invoke()
                        finally:
                            self.graph.capture_end()
                finally:
                    if collecting:
                        gc.enable()

    def _invoke(self):
        from .types import call_scope

        with call_scope():
            for start, end in self.events:
                self.eviction()
                if self.call.reset is not None:
                    self.call.reset()
                if self.call.produce is not None:
                    self.call.produce()
                start.record()
                self.call.invoke()
                end.record()

    def replay(self):
        if self.graph is None:
            self._invoke()
        else:
            self.graph.replay()

    def samples(self):
        return tuple(start.elapsed_time(end) * 1000.0 for start, end in self.events)

    def close(self):
        if self.graph is not None:
            self.graph.reset()
        # CUDA graph nodes do not own their compiled libraries. Drop these
        # references only after destroying the graph executable.
        self._retained = None


@dataclass
class PreparedRace:
    timers: tuple[_TimedCall, ...]
    eviction: object
    sample_count: int
    completed_rounds: int = 0
    planned_rounds: int = 0
    active_count: int = 0
    latest_round_us: tuple[float, ...] = ()
    release_pools: bool = True
    _closed: bool = field(default=False, init=False)
    pool_ids: tuple = field(default=(), init=False)

    def close(self):
        if self._closed:
            return
        self._closed = True
        from ._memory import release_graph_pool_cache

        self.pool_ids = tuple(timer.graph.pool() for timer in self.timers if timer.graph is not None)
        closers = [timer.close for timer in self.timers]
        if self.release_pools:
            closers.extend(lambda pool=pool: release_graph_pool_cache(pool) for pool in self.pool_ids)
        _close_all(closers)


@dataclass(frozen=True)
class RaceMeasurements:
    latencies_us: tuple[float, ...]
    overlapped_samples: int



def _l2_flush_fn(device: object, *, enabled: bool):
    if not enabled:
        return None
    import torch

    properties = torch.cuda.get_device_properties(device)
    flush_bytes = max(2 * int(properties.L2_cache_size), 64 << 20)
    buffer = torch.ones(
        (flush_bytes + 3) // 4,
        dtype=torch.float32,
        device=device,
    )
    reduction = torch.empty((), dtype=torch.float32, device=device)

    def flush() -> None:
        torch.sum(buffer, dim=0, out=reduction)

    return flush


def prepare_race_steps(
    calls, *, device_ordinal, samples=DEFAULT_SAMPLES, primed=False, graph_pools=None,
    capture_stream=None, eviction=None,
):
    """Capture a batch, optionally reusing pools after their graphs are destroyed.

    Each position owns a separate pool: simultaneously live candidate graphs
    must not share storage because the race changes their replay order.
    """
    import torch
    if not calls or type(samples) is not int or samples <= 0:
        raise ValueError("a race requires candidates and positive samples")
    if any(call.produce is None for call in calls):
        raise ValueError("candidate races require an activation-producing context")
    timers = []
    completed = False
    try:
        with torch.cuda.device(device_ordinal), no_compilation():
            if eviction is None:
                eviction = _l2_flush_fn(torch.device("cuda", device_ordinal), enabled=True)
            if capture_stream is None:
                capture_stream = torch.cuda.Stream(device=device_ordinal)
            try:
                eviction()
                if not primed:
                    for call in calls:
                        _prime(call)
            finally:
                torch.cuda.current_stream(device_ordinal).synchronize()
        yield
        for index, call in enumerate(calls):
            with torch.cuda.device(device_ordinal), no_compilation():
                pool = None
                if graph_pools is not None and call.capture_safe:
                    while len(graph_pools) <= index:
                        graph_pools.append(_GraphPool(capture_stream))
                    pool = graph_pools[index].id
                timers.append(_TimedCall(call, eviction, samples, capture_stream, pool=pool))
            yield
        completed = True
        return PreparedRace(tuple(timers), eviction, samples, release_pools=graph_pools is None)
    finally:
        if not completed:
            PreparedRace(tuple(timers), None, samples, release_pools=graph_pools is None).close()


def _replay_timers(timers, *, device_ordinal, sample_count=0, compilation_active=None):
    import torch

    overlaps = 0
    with torch.cuda.device(device_ordinal), no_compilation():
        try:
            for timer in timers:
                if compilation_active is not None and compilation_active():
                    overlaps += sample_count
                timer.replay()
        finally:
            torch.cuda.current_stream(device_ordinal).synchronize()
    return overlaps


def measure_race_steps(
    prepared, *, device_ordinal, rounds=7, compilation_active=None,
    eliminate=False, champion=False,
):
    """Balanced comparison; cancellation discards this generator's result.

    A selection race sets ``eliminate``: a timer whose best round so far trails
    the leader by more than ELIMINATION_MARGIN stops being re-timed and keeps the
    median of the rounds it completed, survivors run at most SURVIVOR_ROUNDS
    rounds, and ``champion`` exempts timer 0, which carries the previous batch's
    winner. Left unset, every timer completes every round.
    """
    if type(rounds) is not int or rounds <= 0:
        raise ValueError("race rounds must be positive")
    if eliminate:
        rounds = min(rounds, SURVIVOR_ROUNDS)
    prepared.planned_rounds = rounds
    prepared.active_count = len(prepared.timers)
    values = [[] for _ in prepared.timers]
    active = list(range(len(prepared.timers)))
    overlaps = 0
    repeats = [1] * len(prepared.timers)
    for turn in range(rounds):
        order = list(active)
        if turn % 2:
            order.reverse()
        offset = (turn // 2) % len(order)
        order = order[offset:] + order[:offset]
        totals = [0.0] * len(prepared.timers)
        repetition = 0
        while repetition < max(repeats[index] for index in order):
            indices = tuple(
                index for index in (order if repetition % 2 == 0 else reversed(order))
                if repetition < repeats[index]
            )
            overlaps += _replay_timers(
                tuple(prepared.timers[index] for index in indices),
                device_ordinal=device_ordinal, sample_count=prepared.sample_count,
                compilation_active=compilation_active,
            )
            for index in indices:
                latency = statistics.fmean(prepared.timers[index].samples())
                if not math.isfinite(latency) or latency <= 0:
                    raise RuntimeError("candidate race produced an invalid latency")
                totals[index] += latency
                if turn == 0 and repetition == 0 and prepared.timers[index].call.capture_safe:
                    # The first scored replay also sizes the remaining work.
                    repeats[index] = max(1, math.ceil(ROUND_BUDGET_US / (prepared.sample_count * latency)))
            repetition += 1
            yield
        for index in order:
            values[index].append(totals[index] / repeats[index])
        # Timers that sat out this round report no latency, so a reader of the
        # round feed does not mistake a stale entry for a fresh measurement.
        timed = frozenset(order)
        prepared.latest_round_us = tuple(
            series[-1] if index in timed else math.nan
            for index, series in enumerate(values)
        )
        prepared.completed_rounds = turn + 1
        if eliminate:
            best = [min(series) for series in values]
            leader = min(best[index] for index in active)
            # A timer outside the leader's margin stops being re-timed and keeps
            # the median of the rounds it completed; the champion at position 0
            # is re-timed against every batch.
            active = [
                index for index in active
                if best[index] <= ELIMINATION_MARGIN * leader or (champion and index == 0)
            ]
            prepared.active_count = len(active)
    latencies = tuple(statistics.median(series) for series in values)
    if any(not math.isfinite(value) or value <= 0 for value in latencies):
        raise RuntimeError("candidate race produced an invalid latency")
    return RaceMeasurements(latencies, overlaps)


def _consume(steps):
    while True:
        try:
            next(steps)
        except StopIteration as finished:
            return finished.value


def _prepare_race(calls, *, device_ordinal, samples=DEFAULT_SAMPLES, primed=False, graph_pools=None, capture_stream=None):
    return _consume(prepare_race_steps(
        calls, device_ordinal=device_ordinal, samples=samples, primed=primed, graph_pools=graph_pools,
        capture_stream=capture_stream,
    ))


def _measure_race(prepared, *, device_ordinal, rounds=7, compilation_active=None):
    return _consume(measure_race_steps(
        prepared, device_ordinal=device_ordinal, rounds=rounds,
        compilation_active=compilation_active,
    ))
