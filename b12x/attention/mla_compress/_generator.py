"""Offline GPU qualification of the actual planned CSA streaming callpath.

No embedded measurements: the fixed configuration is emitted only after every
stable case ID executes, matches its mathematical oracle, and graph-replays
without allocations. Timings include the two-chunk metadata-copy transaction.
"""
from __future__ import annotations

from b12x.policy.generation.measured import GpuProbeMeasurement, MeasuredPolicyGenerator
from ._policy import MLA_COMPRESS_POLICY, MlaCompressQuery

_CASES = ((1, 8, 3), (2, 8, 3), (1, 128, 127), (2, 128, 127))


class _Probe:
    case_ids = tuple(f"csa{ratio}-capacity{capacity}-chunks{length}x2"
                     for ratio, capacity, length in _CASES)
    case_count = len(case_ids)
    description = "CSA prefill/chunk carry, ordinary RMS, emission metadata and fixed-address graph replay"

    def __call__(self, context):
        import torch

        from b12x.policy import PolicyContext, PolicyMode
        from b12x.policy.generation.providers.gpu_workers import (
            _cuda_event_samples_us, _l2_flush_fn, _median_of_group_medians,
        )
        from .api import Caps, bind, plan, run
        from .reference import streaming_reference

        device = torch.device("cuda", context.device_ordinal)
        settings = context.settings
        flush = _l2_flush_fn(device, enabled=settings.cold_l2)
        measurements = []
        for index, (ratio, capacity, length) in enumerate(_CASES):
            p = plan(Caps(device=device, max_tokens=capacity, max_requests=1,
                          max_states=2, ratio=ratio),
                     policy=PolicyContext.for_device(device, mode=PolicyMode.HEURISTIC_ONLY))
            generator = torch.Generator().manual_seed(settings.seed + index)
            host_values = torch.randn((capacity, 512), generator=generator).to(
                torch.float32 if ratio == 2 else torch.bfloat16)
            host_gates = torch.randn((capacity, 512), generator=generator) * 12
            host_weight = torch.linspace(0.2, 1.5, 512)
            host_slots = torch.arange(capacity, dtype=torch.int64) + 2**35
            kwargs = dict(
                values=host_values.to(device), weight=host_weight.to(device),
                query_start_loc=torch.tensor([0, length], dtype=torch.int32, device=device),
                positions=torch.zeros(1, dtype=torch.int64, device=device),
                state_ids=torch.ones(1, dtype=torch.int64, device=device),
                destination_slots=host_slots.to(device),
                live_counts=torch.tensor([length, 1], dtype=torch.int32, device=device),
                out=torch.empty((capacity, 512), dtype=torch.bfloat16, device=device),
                emitted=torch.empty(capacity, dtype=torch.bool, device=device),
                emitted_slots=torch.empty(capacity, dtype=torch.int64, device=device),
            )
            if ratio == 2:
                kwargs.update(gates=host_gates.to(device),
                              pending_values=torch.empty((2, 512), device=device),
                              pending_gates=torch.empty((2, 512), device=device),
                              pending_position=torch.full((2,), -1, dtype=torch.int64, device=device))
            b = bind(p, **kwargs)
            first_position = torch.zeros(1, dtype=torch.int64, device=device)
            next_position = torch.full((1,), length, dtype=torch.int64, device=device)
            state = {}
            oracle = dict(values=host_values, gates=host_gates if ratio == 2 else None,
                          weight=host_weight, starts=[0, length], state_ids=[1],
                          slots=host_slots, ratio=ratio, state=state)
            streaming_reference(**oracle, positions=[0])
            expected = streaming_reference(**oracle, positions=[length])
            expected = tuple(t.to(device) for t in expected)

            def transaction():
                b.positions.copy_(first_position)
                run(b)
                b.positions.copy_(next_position)
                run(b)

            for _ in range(max(1, settings.warmup)):
                transaction()
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                transaction()
            b.out.fill_(float("nan"))
            b.emitted.fill_(True)
            b.emitted_slots.fill_(-7)
            graph.replay()
            torch.cuda.synchronize(device)
            torch.testing.assert_close(b.out, expected[0], rtol=0.012, atol=0.016)
            torch.testing.assert_close(b.emitted, expected[1], rtol=0, atol=0)
            torch.testing.assert_close(b.emitted_slots, expected[2], rtol=0, atol=0)
            if ratio == 2 and b.pending_position[1].item() != -1:
                raise AssertionError("complete two-chunk transaction retained stale CSA carry")
            before = torch.cuda.memory_allocated(device)
            samples = _cuda_event_samples_us(
                graph.replay, count=settings.groups * settings.repetitions,
                device=device, flush=flush)
            after = torch.cuda.memory_allocated(device)
            measurements.append(GpuProbeMeasurement(
                label=self.case_ids[index],
                latency_us=_median_of_group_medians(samples, groups=settings.groups,
                                                    repetitions=settings.repetitions),
                correct=after == before,
                metrics={"replay_allocation_bytes": after - before,
                         "transaction_chunks": 2, "tokens_per_chunk": length,
                         "normalized_pre_rope": True, "emission_metadata_exact": True},
            ))
        return tuple(measurements)


class MlaCompressGenerator(MeasuredPolicyGenerator):
    def __init__(self):
        super().__init__(
            policy=MLA_COMPRESS_POLICY,
            queries=tuple(MlaCompressQuery(ratio=ratio, max_tokens=capacity,
                                          max_requests=1, max_states=2)
                          for ratio, capacity, _ in _CASES),
            encode_config=lambda config: config.to_dict(),
            probe=_Probe(),
        )
