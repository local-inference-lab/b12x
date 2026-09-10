"""Fixed-backend Engram provider; emits profiles only after native GPU probes."""
from __future__ import annotations

from b12x.policy.generation.measured import GpuProbeMeasurement, MeasuredPolicyGenerator


class _EngramProbe:
    case_ids = ("layer1-decode-t3", "layer14-odd-prefill-t7")
    case_count = len(case_ids)
    description = "Engram packed DEAD hashing and FP8/E8M0 row-shard lookup graph qualification"

    def __call__(self, context):
        import torch
        from benchmarks.benchmark_ple import _time_case
        from b12x.sequence import engram
        from b12x.sequence.engram.reference import hash_reference
        from .gpu_workers import _l2_flush_fn

        device = torch.device("cuda", context.device_ordinal)
        flush = _l2_flush_fn(device, enabled=context.settings.cold_l2)
        results = []
        for label, layer, tokens in zip(self.case_ids, (1, 14), (3, 7), strict=True):
            geometry = engram.build_geometry(base_table_size=101, compressed_vocab_size=32)
            p = engram.plan(engram.Caps(device=device, max_tokens=tokens,
                max_seqs=2, max_requests=2, vocab_size=32, layer_id=layer,
                tp_size=2, tp_rank=1), token_map=list(range(32)), geometry=geometry)
            def tensor(values, dtype):
                return torch.tensor(values, dtype=dtype, device=device)
            raw = list(range(1, tokens + 1))
            mask = [i != 1 for i in range(tokens)]
            starts = [0, 1, tokens]
            history = [[-1, -1, -1], [9, -1, 11]]
            spec = p.scratch_specs()[0]
            b = engram.bind(p, scratch=torch.empty(spec.shape, dtype=spec.dtype, device=device),
                token_ids=tensor(raw, torch.int64), token_mask=tensor(mask, torch.bool),
                query_start_loc=tensor(starts, torch.int32), request_slots=tensor([1, 0], torch.int32),
                committed_history=tensor(history, torch.int64), num_seqs=tensor([2], torch.int32),
                num_tokens=tensor([tokens], torch.int32),
                hash_ids=torch.empty((tokens, 24), dtype=torch.int64, device=device))
            row = torch.arange(p.shard_rows, device=device) + p.shard_start
            weight = (row[:, None] % 7 + 1).expand(-1, 256).to(torch.float8_e4m3fn).contiguous()
            scales = torch.arange(123, 131, device=device, dtype=torch.uint8).expand(p.shard_rows, -1).contiguous()
            lookup = engram.bind_lookup(p, weight=weight, scales=scales,
                hash_ids=b.hash_ids, num_tokens=b.num_tokens,
                out=torch.empty((tokens, 6144), dtype=torch.bfloat16, device=device))
            expected_hash = hash_reference(raw, mask, starts, [1, 0], history,
                                            list(range(32)), geometry, layer).to(device)
            values = ((expected_hash % 7 + 1).float()[..., None] *
                      torch.pow(2.0, torch.arange(-4, 4, device=device).repeat_interleave(32)))
            owned = (expected_hash >= p.shard_start) & (expected_hash < p.shard_end) & (expected_hash < p.table_rows)
            expected = torch.where(owned[..., None], values, 0).to(torch.bfloat16).reshape(tokens, 6144)
            history_before = b.committed_history.clone()
            def launch():
                engram.run(b)
                return engram.run_lookup(lookup)
            def validate():
                torch.testing.assert_close(b.hash_ids, expected_hash, rtol=0, atol=0)
                torch.testing.assert_close(lookup.out, expected, rtol=0, atol=0)
                torch.testing.assert_close(b.committed_history, history_before, rtol=0, atol=0)
                if int(b.error_code) != 0:
                    raise AssertionError("Engram metadata rejected valid packed probe")
                return {"status": "passed"}
            timing, correctness, graph_contract = _time_case(
                launch=launch, validate=validate,
                address_tensors={"hash_ids": b.hash_ids, "out": lookup.out,
                                 "scratch": b.scratch, "history": b.committed_history,
                                 "weight": weight, "scales": scales},
                prepare=None, mode="graph", warmup=context.settings.warmup,
                samples=max(1, context.settings.groups * context.settings.repetitions),
                l2_flush=flush, device=device)
            results.append(GpuProbeMeasurement(label=label,
                latency_us=float(timing["kernel"]["median"]),
                correct=correctness.get("status") == "passed" and graph_contract is not None,
                metrics={"layer": layer, "tokens": tokens, "tp_size": 2}))
        return tuple(results)


class EngramGenerator(MeasuredPolicyGenerator):
    def __init__(self):
        from b12x.sequence.engram._policy import ENGRAM_POLICY, EngramQuery
        from b12x.sequence.engram.geometry import build_geometry
        geometry = build_geometry()
        queries = tuple(EngramQuery(max_tokens=tokens, max_seqs=3, max_requests=4,
            vocab_size=129280, compressed_vocab_size=99092, layer_id=layer,
            table_rows=rows, tp_size=8)
            for layer, rows, tokens in zip(geometry.layer_ids, geometry.num_embeddings, (3, 7), strict=True))
        super().__init__(policy=ENGRAM_POLICY, queries=queries,
                         encode_config=lambda config: config.to_dict(), probe=_EngramProbe())


__all__ = ["EngramGenerator"]
