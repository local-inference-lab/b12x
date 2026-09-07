from __future__ import annotations

import inspect

import pytest
import torch

from benchmarks import benchmark_hyperconnection as benchmark
from b12x.norm import hyperconnection as hc


def test_emitter_refuses_to_overwrite_existing_evidence(tmp_path) -> None:
    output = tmp_path / "hyperconnection.jsonl"
    benchmark._Emitter(output)
    output.write_text("preserved\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        benchmark._Emitter(output)

    assert output.read_text(encoding="utf-8") == "preserved\n"


def test_profiles_fix_qwen38_flash_next_geometry() -> None:
    assert benchmark.DEFAULT_PROFILE_TOKENS == (1, 4, 16, 128, 512, 2048)
    for tokens in benchmark.DEFAULT_PROFILE_TOKENS:
        profile = benchmark.Profile(tokens)
        assert (profile.streams, profile.hidden_size, profile.lowrank) == (
            4,
            2560,
            320,
        )
        assert profile.label == f"t{tokens}_s4_h2560_r320"

    with pytest.raises(ValueError, match="fixed at S=4, H=2560, R=320"):
        benchmark.Profile(1, hidden_size=4096)


def test_public_plan_bind_lifecycle_has_exact_capacity_shapes() -> None:
    plan, binding = benchmark.build_plan_binding(device="cpu", tokens=3)

    assert isinstance(plan, hc.Plan)
    assert isinstance(binding, hc.Binding)
    assert plan.caps == hc.Caps(
        device="cpu",
        max_tokens=3,
        hidden_size=2560,
        streams=4,
        lowrank=320,
        dtype=torch.bfloat16,
    )
    assert binding.tokens == 3
    assert binding.normalized.shape == (3, 4 * 2560)
    assert binding.bottleneck.shape == (3, 320)
    assert binding.block_input.shape == (3, 2560)
    assert plan.scratch_specs() == ()


def test_cli_filters_cover_every_public_runtime_entry_point() -> None:
    assert benchmark.OPERATORS == (
        "grouped_rmsnorm",
        "scaled_silu",
        "gate_mean",
        "combine",
        "combine_norm",
        "full_chain",
    )
    assert (
        benchmark.parse_name_filter(
            "all",
            choices=benchmark.OPERATORS,
            label="operators",
        )
        == benchmark.OPERATORS
    )
    assert benchmark.parse_name_filter(
        "combine_norm,grouped_rmsnorm,combine_norm",
        choices=benchmark.OPERATORS,
        label="operators",
    ) == ("combine_norm", "grouped_rmsnorm")
    assert benchmark.parse_name_filter(
        "graph,eager",
        choices=benchmark.MODES,
        label="modes",
    ) == ("graph", "eager")
    assert benchmark.parse_token_filter("1,16,16,512") == (1, 16, 512)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("", "must not be empty"),
        ("all,combine", "cannot be combined"),
        ("torch", "unknown operators"),
    ],
)
def test_cli_operator_filter_rejects_ambiguous_or_unknown_values(
    value: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark.parse_name_filter(
            value,
            choices=benchmark.OPERATORS,
            label="operators",
        )


@pytest.mark.parametrize("value", ["", "0", "-1", "1,two"])
def test_cli_token_filter_rejects_invalid_profiles(value: str) -> None:
    with pytest.raises(ValueError):
        benchmark.parse_token_filter(value)


def test_parser_defaults_run_all_profiles_operators_and_modes() -> None:
    parser = benchmark.build_parser()
    args, tokens, operators, modes = benchmark._parse_args(parser, [])

    assert tokens == benchmark.DEFAULT_PROFILE_TOKENS
    assert operators == benchmark.OPERATORS
    assert modes == benchmark.MODES
    assert args.warmup > 0
    assert args.samples > 0


def test_graph_contract_poison_precedes_replay_and_reference_gate() -> None:
    source = inspect.getsource(benchmark._graph_samples_us)

    poison = source.index('tensor.fill_(float("nan"))')
    replay = source.index("graph.replay()")
    replay_gate = source.index('"graph_replay_after_output_poison": True')
    assert poison < replay < replay_gate
    assert '"replay_after_output_poison": True' in source
    assert "replay_allocation_delta_bytes" in source
