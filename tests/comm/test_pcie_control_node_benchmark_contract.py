from __future__ import annotations

import ast
import math
import os
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from typing import Sequence

import pytest


_ROOT = Path(__file__).resolve().parents[2]


def test_benchmark_prepares_distinct_stable_eager_and_graph_channels() -> None:
    source = (
        _ROOT / "benchmarks" / "benchmark_pcie_oneshot_control_node.py"
    ).read_text(encoding="utf-8")

    assert 'eager_channel_id = "eager:control-node-benchmark"' in source
    assert 'graph_channel_id = "graph:control-node-benchmark"' in source
    assert "pool.prepare_channels((eager_channel_id, graph_channel_id))" in source
    assert "pool.for_stream(stream, channel_id=eager_channel_id)" in source
    assert "pool.capture(stream, channel_id=graph_channel_id)" in source


def _load_functions(relative_path: str, names: set[str]) -> dict[str, object]:
    path = _ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    assert {node.name for node in selected} == names
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {
        "CORRECTNESS_KEYS": {"verdict"},
        "math": math,
        "METRICS": ("mean_us", "p50_us", "p95_us"),
        "os": os,
        "Path": Path,
        "Sequence": Sequence,
        "signal": signal,
        "subprocess": subprocess,
        "suppress": suppress,
    }
    # Execute only the selected, dependency-injected helpers from the benchmark.
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    "relative_path",
    [
        "benchmarks/benchmark_pcie_oneshot_control_node.py",
        "benchmarks/run_pcie_oneshot_control_node_ab.py",
    ],
)
def test_benchmark_correctness_contract_requires_an_explicit_pass(
    relative_path: str,
) -> None:
    functions = _load_functions(relative_path, {"_validate_correctness"})
    validate = functions["_validate_correctness"]

    validate({"verdict": "pass"})
    with pytest.raises(ValueError, match="correctness did not pass"):
        validate({"verdict": "fail"})
    with pytest.raises(ValueError, match="correctness schema mismatch"):
        validate({"verdict": "pass", "details": "unexpected"})


def test_distributed_critical_path_uses_aligned_iteration_maxima() -> None:
    worker = _load_functions(
        "benchmarks/benchmark_pcie_oneshot_control_node.py",
        {"_summary", "_distributed_critical_path"},
    )
    per_rank = [
        {
            "eager": {"cold_us": 11.0, "samples_us": [10.0, 1.0, 10.0, 1.0]},
            "graph": {"cold_us": 7.0, "samples_us": [6.0, 2.0, 6.0, 2.0]},
        },
        {
            "eager": {"cold_us": 12.0, "samples_us": [1.0, 10.0, 1.0, 10.0]},
            "graph": {"cold_us": 8.0, "samples_us": [2.0, 6.0, 2.0, 6.0]},
        },
    ]

    result = worker["_distributed_critical_path"](per_rank)

    # The old max(mean(rank)) aggregation reports 5.5us / 4us. The actual
    # collective critical path changes rank each iteration and is 10us / 6us.
    assert max(sum(row["eager"]["samples_us"]) / 4 for row in per_rank) == 5.5
    assert result["eager"]["samples_us"] == [10.0, 10.0, 10.0, 10.0]
    assert result["eager"]["mean_us"] == 10.0
    assert result["eager"]["cold_us"] == 12.0
    assert result["graph"]["samples_us"] == [6.0, 6.0, 6.0, 6.0]
    assert result["graph"]["mean_us"] == 6.0
    assert result["graph"]["cold_us"] == 8.0


def test_controller_paired_deltas_use_distributed_critical_path() -> None:
    controller = _load_functions(
        "benchmarks/run_pcie_oneshot_control_node_ab.py",
        {"_paired_deltas"},
    )

    def record(variant: str, eager: float, graph: float) -> dict[str, object]:
        def mode(value: float) -> dict[str, float | list[float]]:
            return {
                "cold_us": value,
                "mean_us": value,
                "p50_us": value,
                "p95_us": value,
                "min_us": value,
                "max_us": value,
                "samples_us": [value],
            }

        return {
            "identity": {"variant": variant},
            "distributed_critical_path": {
                "eager": mode(eager),
                "graph": mode(graph),
            },
        }

    deltas = controller["_paired_deltas"](
        [
            record("baseline", 10.0, 20.0),
            record("candidate", 12.0, 18.0),
        ]
    )

    assert deltas[0]["metrics"]["eager"]["mean_us"]["delta_us"] == 2.0
    assert deltas[0]["metrics"]["graph"]["p95_us"]["delta_us"] == -2.0


def test_controller_timeout_terminates_the_command_process_group() -> None:
    controller = _load_functions(
        "benchmarks/run_pcie_oneshot_control_node_ab.py",
        {"_run"},
    )

    with pytest.raises(TimeoutError, match="process group was terminated"):
        controller["_run"](
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout=0.05,
        )
