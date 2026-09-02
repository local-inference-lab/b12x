"""Generic layer-stack harness shared by the in-place profile generators.

A candidate is ranked by its own device time while it runs as one slot of a
small synthetic transformer-like stack: the previous kernel leaves its input
L2-warm, its weights stream cold like every layer's, its output feeds the next
kernel, and launches overlap as in a real step. Timing events recorded inside
the captured graph bracket the tested slot; the whole-stack replay time is
kept as secondary evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .timing import _bounded_repetitions, _median_of_group_medians

CONTEXT_LAYERS = 3
CONTEXT_MARGIN = 0.02
BF16_INPUT_RECIPES = frozenset({"mxfp8", "tensor_fp8"})


def _context_hidden(in_features: int, out_features: int) -> int:
    """Hidden width of the synthetic layer around a (K, N) GEMM.

    An up-like GEMM (K <= N) widens the hidden state, so H = K; a down-like
    GEMM narrows it back, so H = N. Either way the stack keeps one square
    H x H neighbour and one H <-> K/N neighbour, like attention-out + MLP.
    """
    hidden = in_features if in_features <= out_features else out_features
    return max(256, -(-hidden // 128) * 128)


def layer_budget(
    device,
    per_layer_bytes: int,
    *,
    requested: int = CONTEXT_LAYERS,
    fraction: float = 0.6,
) -> int:
    """Layers that fit ``fraction`` of the free device memory, at least one."""
    import torch

    free, _total = torch.cuda.mem_get_info(device)
    affordable = int(free * fraction) // max(int(per_layer_bytes), 1)
    return max(1, min(int(requested), affordable))


class MxFp8Filler:
    """An MXFP8 GEMM neighbour on the built-in plan (K -> N for ``tokens`` rows)."""

    def __init__(
        self, *, in_features: int, out_features: int, tokens: int, device, generator
    ):
        import torch

        from b12x.gemm import dense_linear
        from b12x.policy import PolicyContext, PolicyMode

        from .gemm import _DenseLinearOperands

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.tokens = int(tokens)
        self.operands = _DenseLinearOperands(
            recipe="mxfp8",
            in_features=self.in_features,
            out_features=self.out_features,
            device=device,
            generator=generator,
            with_reference=False,
        )
        caps = dense_linear.Caps(
            device=device,
            recipe="mxfp8",
            in_features=self.in_features,
            out_features=self.out_features,
            max_tokens=self.tokens,
            output_dtype=torch.bfloat16,
        )
        heuristic = PolicyContext.for_device(device, mode=PolicyMode.HEURISTIC_ONLY)
        self.plan = dense_linear.plan(caps, policy=heuristic)
        self._mm = dense_linear.mm

    def __call__(self, source):
        return self._mm(
            source, self.operands.packed, plan=self.plan, expected_m=self.tokens
        )


def rmsnorm(x, weight, eps: float = 1e-6):
    import torch

    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps)).to(torch.bfloat16) * weight


class GenericLayerStack:
    """``filler_in -> produce -> [tested slot] -> consume -> filler_out -> norm``.

    Subclasses implement ``produce(layer, activation)`` (write the tested
    op's inputs from the neighbour's output), ``tested(layer, slot)`` (run the
    candidate and return its output) and ``consume(layer, output)`` (turn the
    output into the ``[tokens, out_width]`` activation the closing neighbour
    reads). ``events`` brackets only the tested slot.
    """

    def __init__(
        self,
        *,
        hidden: int,
        in_width: int,
        out_width: int,
        tokens: int,
        layers: int,
        device,
        generator,
    ) -> None:
        import torch

        self.hidden = int(hidden)
        self.tokens = int(tokens)
        self.layers = int(layers)
        self.device = device
        self.fillers_in = [
            MxFp8Filler(
                in_features=self.hidden,
                out_features=int(in_width),
                tokens=self.tokens,
                device=device,
                generator=generator,
            )
            for _ in range(self.layers)
        ]
        self.fillers_out = [
            MxFp8Filler(
                in_features=int(out_width),
                out_features=self.hidden,
                tokens=self.tokens,
                device=device,
                generator=generator,
            )
            for _ in range(self.layers)
        ]
        self.norm_weight = torch.ones(self.hidden, dtype=torch.bfloat16, device=device)
        self.hidden_state = torch.randn(
            (self.tokens, self.hidden),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )

    def produce(self, layer: int, activation):
        raise NotImplementedError

    def tested(self, layer: int, slot):
        raise NotImplementedError

    def consume(self, layer: int, output):
        return output

    def run(self, slot, events=None):
        h = self.hidden_state
        for layer in range(self.layers):
            a = self.fillers_in[layer](h)
            self.produce(layer, a)
            if events is not None:
                events[layer][0].record()
            out = self.tested(layer, slot)
            if events is not None:
                events[layer][1].record()
            b = self.fillers_out[layer](self.consume(layer, out))
            h = rmsnorm(h + b, self.norm_weight)
        return h


def _context_race(*, context, slot, settings, device, flush):
    """Time the tested op inside the captured layer stack.

    Returns the per-layer in-place op time (the ranking score), the whole
    stack replay time (secondary evidence), finiteness and the repetitions.
    """
    import torch

    for _ in range(settings.warmup):
        context.run(slot)
    torch.cuda.synchronize(device)
    events = [
        (
            torch.cuda.Event(enable_timing=True, external=True),
            torch.cuda.Event(enable_timing=True, external=True),
        )
        for _ in range(context.layers)
    ]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = context.run(slot, events)
    graph.replay()
    torch.cuda.synchronize(device)
    finite = bool(torch.isfinite(output).all().item())

    def sample():
        if flush is not None:
            flush()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize(device)
        op_us = sum(a.elapsed_time(b) for a, b in events) * 1_000.0 / len(events)
        return op_us, float(start.elapsed_time(end)) * 1_000.0

    pilot_op, pilot_stack = sample()
    repetitions = _bounded_repetitions(settings, pilot_us=pilot_stack)
    samples = [sample() for _ in range(settings.groups * repetitions)]
    op_latency = _median_of_group_medians(
        tuple(item[0] for item in samples),
        groups=settings.groups,
        repetitions=repetitions,
    )
    stack_latency = _median_of_group_medians(
        tuple(item[1] for item in samples),
        groups=settings.groups,
        repetitions=repetitions,
    )
    return _ContextRace(
        op_us=op_latency,
        stack_us=stack_latency,
        finite=finite,
        repetitions=repetitions,
        sample=sample,
        graph=graph,
    )


def _keep_race(races, candidate, race, baseline_config):
    """Retain the graphs of the built-in plan and the running leader only."""
    config = candidate.config.to_dict()
    is_baseline = baseline_config is not None and config == baseline_config
    leader = races.get("__leader__")
    if leader is None or race.op_us < leader[1]:
        old = leader[0] if leader else None
        races["__leader__"] = (candidate.candidate_id, race.op_us)
        races[candidate.candidate_id] = race
        if old is not None and old != races.get("__baseline__"):
            races.pop(old, None)
    elif is_baseline:
        races[candidate.candidate_id] = race
    if is_baseline:
        races["__baseline__"] = candidate.candidate_id


@dataclass
class _ContextRace:
    op_us: float
    stack_us: float
    finite: bool
    repetitions: int
    sample: object
    graph: object


CONFIRM_ROUNDS = 8


def _confirm_winner(
    measurements, races, *, baseline_config, passes=None, on_sample=None
):
    """Re-time the leader and the built-in plan interleaved.

    Two sweeps seconds apart can disagree by more than the margin; measuring
    the two contenders alternately in one pass removes that drift before the
    reducer applies the margin. Both latencies are replaced by the confirmed
    medians; the first-pass values stay in the metrics. ``on_sample`` is told
    which candidate is about to replay, so a crash guard can blame it.
    """
    import statistics

    if passes is None:

        def passes(m):
            return m.correct and m.latency_us is not None

    passing = [m for m in measurements if passes(m)]
    leader = races.get("__leader__")
    if len(passing) < 2 or baseline_config is None or leader is None:
        return measurements
    best = next((m for m in passing if m.candidate.candidate_id == leader[0]), None)
    baseline = next(
        (m for m in passing if m.candidate.config.to_dict() == baseline_config), None
    )
    if best is None:
        return measurements
    if baseline is None or baseline is best:
        return measurements
    best_race = races.get(best.candidate.candidate_id)
    base_race = races.get(baseline.candidate.candidate_id)
    if best_race is None or base_race is None:
        return measurements
    best_samples, base_samples = [], []
    for _ in range(CONFIRM_ROUNDS):
        if on_sample is not None:
            on_sample(best.candidate.candidate_id)
        best_samples.append(best_race.sample()[0])
        if on_sample is not None:
            on_sample(baseline.candidate.candidate_id)
        base_samples.append(base_race.sample()[0])
    confirmed = {
        best.candidate.candidate_id: statistics.median(best_samples),
        baseline.candidate.candidate_id: statistics.median(base_samples),
    }
    updated = []
    for m in measurements:
        value = confirmed.get(m.candidate.candidate_id)
        if value is None:
            updated.append(m)
            continue
        metrics = dict(m.metrics or {})
        metrics["first_pass_us"] = m.latency_us
        metrics["confirmed"] = True
        updated.append(replace(m, latency_us=value, metrics=metrics))
    return updated


__all__ = [
    "BF16_INPUT_RECIPES",
    "CONFIRM_ROUNDS",
    "CONTEXT_LAYERS",
    "CONTEXT_MARGIN",
    "GenericLayerStack",
    "MxFp8Filler",
    "_ContextRace",
    "_confirm_winner",
    "_context_hidden",
    "_context_race",
    "_keep_race",
    "layer_budget",
    "rmsnorm",
]
