from __future__ import annotations

from dataclasses import fields
import inspect

import pytest
import torch

from benchmarks import benchmark_gdn_decode as benchmark


def test_cli_refuses_to_overwrite_existing_evidence(tmp_path) -> None:
    output = tmp_path / "gdn.json"
    output.write_text("preserved\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        benchmark.main(["--json", str(output)])

    assert output.read_text(encoding="utf-8") == "preserved\n"


def test_benchmark_suite_covers_sharded_qwen_head_geometries() -> None:
    by_name = {case.name: case for case in benchmark.QWEN38_GDN_CASES}
    assert set(by_name) == {
        "qk16-v48-decode-bs1",
        "qk8-v24-decode-bs1",
        "qk8-v24-decode-bs4",
        "qk8-v24-spec2-bs4",
        "qk8-v24-spec4-bs1",
        "qk8-v24-spec4-uneven",
        "qk8-v24-spec4-bs4",
        "qk4-v12-decode-bs1",
        "qk2-v6-decode-bs1",
    }
    assert {(case.key_heads, case.value_heads) for case in by_name.values()} == {
        (16, 48),
        (8, 24),
        (4, 12),
        (2, 6),
    }
    assert all(case.state_dtype == torch.float32 for case in by_name.values())


def test_speculative_cases_preserve_sequential_request_geometry() -> None:
    by_name = {case.name: case for case in benchmark.QWEN38_GDN_CASES}
    case = by_name["qk8-v24-spec4-uneven"]
    assert case.query_lengths == (4, 2, 1, 3)
    assert case.tokens == 10
    assert case.sequences == 4
    assert case.columns == 4
    assert by_name["qk8-v24-spec2-bs4"].query_lengths == (2, 2, 2, 2)
    assert by_name["qk8-v24-spec4-bs4"].query_lengths == (4, 4, 4, 4)


def test_planned_capacity_is_independent_of_live_metadata() -> None:
    by_name = {case.name: case for case in benchmark.QWEN38_GDN_CASES}
    for name in (
        "qk16-v48-decode-bs1",
        "qk8-v24-decode-bs4",
        "qk8-v24-spec4-bs1",
    ):
        assert benchmark.resolve_capacity(
            by_name[name], capacity_seqs=None, capacity_columns=None
        ) == (4, 4, 16)

    assert benchmark.resolve_capacity(
        by_name["qk8-v24-decode-bs4"], capacity_seqs=32, capacity_columns=4
    ) == (32, 4, 128)
    with pytest.raises(ValueError, match="exceeds planned capacity"):
        benchmark.resolve_capacity(
            by_name["qk8-v24-spec4-bs1"], capacity_seqs=4, capacity_columns=3
        )


def test_case_selection_is_ordered_and_rejects_unknown_names() -> None:
    selected = benchmark.select_cases(
        "qk8-v24-spec4-bs4,qk16-v48-decode-bs1"
    )
    assert [case.name for case in selected] == [
        "qk8-v24-spec4-bs4",
        "qk16-v48-decode-bs1",
    ]
    with pytest.raises(ValueError, match="unknown cases"):
        benchmark.select_cases("glm-kda")


def test_percentile_uses_nearest_rank_upper_sample() -> None:
    assert benchmark._percentile([1.0, 2.0, 3.0, 4.0], 0.9) == 4.0


def test_graph_contract_poison_precedes_replay_and_state_output_gate() -> None:
    source = inspect.getsource(benchmark._bench_graph)

    output_poison = source.index('binding.output.fill_(float("nan"))')
    scratch_poison = source.index("binding.scratch.fill_(0xFF)")
    replay = source.index("graph.replay()", scratch_poison)
    replay_gate = source.index("replay_correctness = _check_current_result")
    assert output_poison < scratch_poison < replay < replay_gate
    assert "graph_replay_after_output_poison" in {
        field.name for field in fields(benchmark.CaseReport)
    }
    assert "graph_replay_after_scratch_poison" in {
        field.name for field in fields(benchmark.CaseReport)
    }


def test_eager_and_graph_flush_before_restoring_recurrent_state() -> None:
    eager_source = inspect.getsource(benchmark._bench_eager)
    graph_source = inspect.getsource(benchmark.bench_cuda_graph)

    assert eager_source.index("l2_flush()") < eager_source.index(
        "binding.recurrent_state.copy_(buffers.initial_state)",
        eager_source.index("for _ in range(iterations)"),
    )
    assert graph_source.index("l2_flush()") < graph_source.index("prepare()")


def test_persisted_provenance_covers_reproducibility_and_metric_direction() -> None:
    source = inspect.getsource(benchmark.main)

    for field in (
        '"command"',
        '"cwd"',
        '"git"',
        '"device"',
        '"gpu_mode_before"',
        '"gpu_mode_after"',
        '"metric_direction"',
    ):
        assert field in source
    assert '"lower_is_better"' in source
