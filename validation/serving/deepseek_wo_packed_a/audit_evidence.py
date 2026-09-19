"""Check serving samples and execute the recorded client's KV admission rule.

The budget flag is a client-side admission bound, not a server KV allocation
command. This audit checks its actual implementation on the qualified cells;
it does not assert that arbitrary cache capacities or contexts are equivalent.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def audit(benchmark: Path) -> dict:
    """Validate both exports and their context-zero admission decisions."""
    directory = Path(__file__).parent
    source = benchmark.read_bytes()
    tree = ast.parse(source)
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_should_skip"
    ]
    assert len(nodes) == 1, "Expected the recorded client's single admission rule"
    function = nodes[0]
    code = compile(ast.Module(body=[function], type_ignores=[]), str(benchmark), "exec")
    records = []
    for relative in ("../deepseek_wo_layout/results.json", "results.json"):
        evidence = json.loads((directory / relative).read_text())
        assert hashlib.sha256(source).hexdigest() == evidence["benchmark"]["sha256"]
        for name, arm in evidence["arms"].items():
            assert arm["methodology"]["prefill"]["present"] is True
            assert arm["methodology"]["prefill"]["mode"] == "standalone"
            assert len(arm["prefill"]) == len(arm["decode"]) == 5
            budget = arm["logical_kv_capacity_tokens"]
            namespace = {
                "args": SimpleNamespace(max_total_tokens=budget, max_tokens=8192),
                "max_run": 8,
            }
            exec(code, namespace)
            decisions = []
            for concurrency in (1, 8):
                skipped = namespace["_should_skip"](0, concurrency)
                assert not skipped
                assert concurrency * 8192 < budget
                decisions.append(
                    {"context": 0, "concurrency": concurrency, "skipped": skipped}
                )
            for run in arm["decode"]:
                assert {cell["concurrency"] for cell in run["cells"]} == {1, 8}
                for cell in run["cells"]:
                    assert cell["context_tokens"] == 0
                    for flag in (
                        "capacity_limited",
                        "underfilled",
                        "loop_detected",
                        "num_errors",
                        "failure_reason",
                        "warmup_timed_out",
                    ):
                        assert not cell[flag], (relative, name, run["file"], flag)
            records.append(
                {
                    "evidence": relative,
                    "arm": name,
                    "budget": budget,
                    "admission": decisions,
                }
            )
    return {
        "benchmark_sha256": hashlib.sha256(source).hexdigest(),
        "admission_function_source": ast.get_source_segment(source.decode(), function),
        "records": records,
        "conclusion": "All four arms execute the same C1/C8 context-zero cells; no audited KV budget limits these requests. Each arm contains five completed standalone prefill windows.",
    }


def main() -> None:
    """Write the reproducible admission and sample audit as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.benchmark), indent=2))


if __name__ == "__main__":
    main()
