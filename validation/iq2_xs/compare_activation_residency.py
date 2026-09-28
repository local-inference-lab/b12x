"""Compare recorded dense plans with cold and freshly produced activations."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.gemm import blockscaled
from b12x.preparation import PreparationSession, PreparedCall, require_prepared
from benchmarks.checkpoint_dense import checkpoint_cases, check_output, load_weight, oracle
from benchmarks.common import make_l2_flush_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    records = []
    for path in (args.baseline, args.candidate):
        records.append(next(row for row in map(json.loads, path.read_text().splitlines())
                            if row.get("kind") == "case" and row["rows"] == 512
                            and row["role"] == "backbone.layers.*.mixer.out_proj.weight"))
    assert records[0]["query"] == records[1]["query"]
    index, cases = checkpoint_cases(args.model_path)
    case = next(case for case in cases if case["weight"] == records[0]["weight"])
    weight, decoded, multiplier, hashes = load_weight(args.model_path, index, case)
    assert hashes == records[0]["payload_sha256"] == records[1]["payload_sha256"]
    torch.manual_seed(554)
    source = torch.randn((512, case["k"]), device="cuda", dtype=torch.bfloat16) * .25
    produced = source.clone()
    expected = oracle(source, decoded, multiplier)
    flush = make_l2_flush_fn(enabled=True)
    evidence = {"records": records, "payload_sha256": hashes,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "metric": "microseconds; lower is better", "results": []}
    with PreparationSession(device=source.device, autotune=False, compile_workers=0) as session:
        plans = [blockscaled.plan(blockscaled.BlockscaledQuery(**row["query"]),
                 override=blockscaled.BlockscaledConfig(**row["config"])) for row in records]

        def prepare(state):
            return PreparedCall(run=lambda: state.run(source, weight.values, weight.scale_mma,
                                                     weight.global_scale),
                                produce=lambda: source.copy_(produced))

        session.prepare(tuple(plan.request(name=f"projection-{i}", prepare_call=prepare)
                              for i, plan in enumerate(plans)))
        session.freeze()
        graphs = []
        outputs = []
        with kernel_resolution_guard("activation residency comparison"):
            for plan in plans:
                state = require_prepared(plan, "gemm.blockscaled_precision", source.device)
                assert state.required_workspace == 0
                for _ in range(20):
                    blockscaled.mm(source, weight, plan=plan)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = blockscaled.mm(source, weight, plan=plan)
                graph.replay()
                torch.cuda.synchronize()
                check_output(output, expected)
                graphs.append(graph)
                outputs.append(output)
            for mode in ("cold", "produced", "filled"):
                samples = [[], []]
                for order in ((0, 1), (1, 0), (0, 1), (1, 0)):
                    for arm in order:
                        events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                                  for _ in range(25)]
                        for start, end in events:
                            flush()
                            if mode == "produced":
                                source.copy_(produced)
                            elif mode == "filled":
                                source.fill_(0.125)
                            start.record()
                            graphs[arm].replay()
                            end.record()
                        torch.cuda.synchronize()
                        samples[arm].extend(start.elapsed_time(end) * 1000 for start, end in events)
                result = {"mode": mode, "samples_us": samples,
                          "median_us": [statistics.median(s) for s in samples]}
                evidence["results"].append(result)
                print(json.dumps({key: value for key, value in result.items() if key != "samples_us"}), flush=True)
            from b12x.preparation._measurement import _prepare_race, _measure_race
            states = [require_prepared(plan, "gemm.blockscaled_precision", source.device) for plan in plans]
            calls = [prepare(state) for state in states]
            for call in calls:
                call.produce = lambda: source.fill_(0.125)
            race = _prepare_race(calls, device_ordinal=source.device.index, samples=100)
            try:
                measured = _measure_race(race, device_ordinal=source.device.index, rounds=3)
                result = {"mode": "production-race-filled", "latencies_us": measured.latencies_us}
                evidence["results"].append(result)
                print(json.dumps(result), flush=True)
            finally:
                race.close()
            for graph in graphs:
                graph.reset()
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
