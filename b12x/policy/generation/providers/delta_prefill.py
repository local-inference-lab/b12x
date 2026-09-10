"""Measured, resumable policy generators for the shared delta-rule prefill kernels."""

from __future__ import annotations

import statistics
from itertools import product

from b12x.policy.generation.delta_prefill_cases import (
    GDN_PREFILL_CASES, KDA_PREFILL_CASES, PrefillCase, check_binding,
    make_binding, make_inputs, oracle, run_binding,
)
from b12x.policy.generation.sweep import DiscreteSweepGenerator, SweepCandidate, SweepCase, SweepMeasurement
from b12x.sequence._shared.delta_prefill.workspace import (
    K_SPLIT_CHOICES, STAGE_CHOICES, V_SPLIT_CHOICES, default_window_tiles, tiles_capacity,
)


def prefill_cases(recipe):
    cases = GDN_PREFILL_CASES if recipe == "gdn" else KDA_PREFILL_CASES
    result = []
    for case in cases:
        geometry = ({"key_heads": case.key_heads, "value_heads": case.value_heads}
                    if recipe == "gdn" else {"heads": case.value_heads})
        for checkpoint in (False, True):
            query = dict(geometry, head_dim=128, model_dtype="bfloat16", state_dtype="float32",
                         qk_l2norm=True, checkpoint_export=checkpoint,
                         max_tokens=case.tokens, max_seqs=len(case.lengths))
            result.append(SweepCase.create(
                group_id=f"{recipe}-h{case.value_heads}", query=query,
                metadata={"recipe": recipe, "lengths": list(case.lengths)},
                scenario="checkpoint" if checkpoint else "final-state", label=case.name,
            ))
    return tuple(result)


def prefill_candidates(case):
    is_gdn = "value_heads" in case.query
    heads = int(case.query["value_heads"] if "value_heads" in case.query else case.query["heads"])
    window = default_window_tiles(heads, int(case.query["max_tokens"]), int(case.query["max_seqs"]))
    tile_capacity = -(-int(case.query["max_tokens"]) // 16) + int(case.query["max_seqs"])
    windows = sorted({max(1, window // 2), window, min(tile_capacity, 2 * window)})
    sequential = tuple(SweepCandidate.create({
        "backend": "cutedsl", "v_split": v, "k_split": k, "stages": stages, "window_tiles": selected_window,
        **({"algorithm": "sequential", "segment_tokens": 256} if is_gdn else {}),
    }) for v, k, stages, selected_window in product(V_SPLIT_CHOICES, K_SPLIT_CHOICES, STAGE_CHOICES, windows)
        if 2 * v * k + 32 <= 1024)
    parallel = ()
    if is_gdn:
        tokens, seqs = int(case.query["max_tokens"]), int(case.query["max_seqs"])
        parallel = tuple(SweepCandidate.create({
            "backend": "cutedsl", "v_split": v, "k_split": k, "stages": stages,
            "window_tiles": tiles_capacity(tokens, (tokens + segment - 1) // segment + seqs),
            "algorithm": "chunk_parallel", "segment_tokens": segment,
        }) for v, k, stages, segment in product(V_SPLIT_CHOICES, K_SPLIT_CHOICES, STAGE_CHOICES,
                                               (128, 256, 512, 1024))
            if 2 * v * k + 32 <= 1024 and (tokens + segment - 1) // segment + seqs <= 4096)
    return sequential + parallel


class PrefillBenchmarkSession:
    def __init__(self, context):
        self.context = context

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def candidates(self, case):
        return prefill_candidates(case)

    def measure(self, case, candidates):
        import torch
        from benchmarks.common import nvidia_smi_gpu_mode_snapshot
        from .gpu_workers import _l2_flush_fn

        if not set(c.candidate_id for c in candidates) <= set(c.candidate_id for c in self.candidates(case)):
            raise ValueError("unknown delta-prefill candidate")
        recipe = str(case.metadata["recipe"])
        heads = int(case.query["value_heads"] if recipe == "gdn" else case.query["heads"])
        key_heads = int(case.query["key_heads"]) if recipe == "gdn" else heads
        shape = PrefillCase(recipe, key_heads, heads, tuple(map(int, case.metadata["lengths"])))
        device = torch.device("cuda", self.context.device_ordinal)
        settings = self.context.settings
        tensors = make_inputs(shape, device=device, seed=settings.seed)
        if case.query["checkpoint_export"]:
            tensors["checkpoint_offsets"][0] = 16
        initial = tensors["recurrent_state"].clone()
        expected, state = oracle(shape, tensors)
        flush = _l2_flush_fn(device, enabled=settings.cold_l2)
        active, failures = [], []
        with torch.cuda.device(device):
            for candidate in candidates:
                try:
                    binding = make_binding(shape, tensors, checkpoint_export=bool(case.query["checkpoint_export"]),
                                           config=candidate.config.to_dict())
                    for _ in range(settings.warmup):
                        binding.recurrent_state.copy_(initial)
                        run_binding(recipe, binding)
                    graph = torch.cuda.CUDAGraph()
                    binding.recurrent_state.copy_(initial)
                    with torch.cuda.graph(graph):
                        run_binding(recipe, binding)
                    binding.recurrent_state.copy_(initial)
                    binding.output.fill_(float("nan"))
                    binding.scratch.fill_(0xFF)
                    addresses = tuple(x.data_ptr() for x in (binding.output, binding.scratch, binding.recurrent_state))
                    allocated = torch.cuda.memory_allocated(device)
                    graph.replay()
                    torch.cuda.synchronize(device)
                    delta = torch.cuda.memory_allocated(device)-allocated
                    metrics = check_binding(shape, binding, expected, state, initial)
                    if delta != 0:
                        raise AssertionError(f"graph replay allocated {delta} bytes")
                    if addresses != tuple(x.data_ptr() for x in (binding.output, binding.scratch, binding.recurrent_state)):
                        raise AssertionError("graph replay changed bound addresses")
                    metrics.update(stable_addresses=True, replay_allocation_bytes=delta)
                    active.append((candidate, binding, graph, metrics, []))
                except (AssertionError, ValueError) as exc:
                    failures.append(SweepMeasurement(candidate=candidate, latency_us=None, correct=False,
                                                     error=f"{type(exc).__name__}: {exc}"))
            timing_mode_before = nvidia_smi_gpu_mode_snapshot()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for iteration in range(settings.groups * settings.repetitions):
                order = active if iteration % 2 == 0 else list(reversed(active))
                for candidate, binding, graph, metrics, samples in order:
                    if flush is not None:
                        flush()
                    binding.recurrent_state.copy_(initial)
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end) * 1000)
            timing_mode_after = nvidia_smi_gpu_mode_snapshot()
        measured = {
            c.candidate_id: SweepMeasurement(candidate=c, latency_us=statistics.median(samples), correct=True,
                                            metrics=dict(metrics, samples_us=samples, cold_l2=settings.cold_l2,
                                                         gpu_mode_before=timing_mode_before,
                                                         gpu_mode_after=timing_mode_after))
            for c, _, _, metrics, samples in active
        }
        measured.update((m.candidate.candidate_id, m) for m in failures)
        return tuple(measured[c.candidate_id] for c in candidates)


class PrefillBenchmarkFactory:
    def __call__(self, group_id, cases, context):
        return PrefillBenchmarkSession(context)


class _PrefillGenerator(DiscreteSweepGenerator):
    def __init__(self, recipe, *, cases=None, benchmark_factory=None):
        corpus = prefill_cases(recipe) if cases is None else tuple(cases)
        super().__init__(
            component_id=f"sequence.{recipe}_prefill", query_schema_version=1,
            config_schema_version=2 if recipe == "gdn" else 1,
            query_fields=tuple(sorted(corpus[0].query)), range_fields=frozenset(), cases=corpus,
            benchmark_factory=benchmark_factory or PrefillBenchmarkFactory(),
            coverage={"recipe": recipe, "qualification": "GPU production plans",
                      "candidate_count_min": min(len(prefill_candidates(case)) for case in corpus),
                      "candidate_count_max": max(len(prefill_candidates(case)) for case in corpus)},
            candidate_contract_version=5,
        )


class GdnPrefillGenerator(_PrefillGenerator):
    def __init__(self, **kwargs):
        super().__init__("gdn", **kwargs)


class KdaPrefillGenerator(_PrefillGenerator):
    def __init__(self, **kwargs):
        super().__init__("kda", **kwargs)
