from __future__ import annotations

import inspect

import pytest
import torch

from benchmarks import benchmark_ple as benchmark
from b12x.sequence import ple, ple_embedding, ple_hash


def test_emitter_refuses_to_overwrite_existing_evidence(tmp_path) -> None:
    output = tmp_path / "ple.jsonl"
    benchmark._Emitter(output)
    output.write_text("preserved\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        benchmark._Emitter(output)

    assert output.read_text(encoding="utf-8") == "preserved\n"


def test_packed_profiles_cover_decode_speculative_and_prefill() -> None:
    assert [
        (profile.name, profile.phase, profile.query_lengths)
        for profile in benchmark.PACKED_PROFILES
    ] == [
        ("decode-t1-bs1", "decode", (1,)),
        ("spec-t4-bs1", "speculative", (4,)),
        ("prefill-t128-bs1", "prefill", (128,)),
        ("prefill-t512-bs4", "prefill", (128, 128, 128, 128)),
    ]
    assert benchmark.PACKED_PROFILES[-1].tokens == 512
    assert benchmark.PACKED_PROFILES[-1].sequences == 4


def test_layer_profiles_fix_qwen38_flash_next_geometry_and_all_public_runs() -> None:
    assert (benchmark.STREAMS, benchmark.HIDDEN_SIZE) == (4, 2560)
    assert (benchmark.KERNEL_SIZE, benchmark.DILATION) == (4, 3)
    assert benchmark.MAX_SPECULATIVE_TOKENS == 4
    assert {profile.mode for profile in benchmark.LAYER_PROFILES} == {
        "decode",
        "prefill",
        "mixed",
    }
    mixed = next(
        profile for profile in benchmark.LAYER_PROFILES if profile.mode == "mixed"
    )
    assert mixed.query_lengths == (128, 4, 1)
    assert mixed.request_is_prefill == (True, False, False)


def test_hash_builder_uses_production_geometry_and_public_lifecycle() -> None:
    profile = benchmark.PackedProfile("unit", "decode", (2,))
    case = benchmark.build_hash_case(profile, device="cpu", seed=11)

    assert isinstance(case.binding, ple_hash.Binding)
    assert case.binding.plan.caps.base_table_size == 20_000_000
    assert case.binding.plan.caps.head_count == 16
    assert case.binding.out.shape == (2, 16)
    torch.testing.assert_close(case.expected, case.expected.clone(), rtol=0, atol=0)
    with pytest.raises(ValueError, match="GPU run requires CUDA"):
        case.launch()


@pytest.mark.parametrize("quant_mode", benchmark.QUANT_MODES)
def test_embedding_plan_preserves_qwen_compute_geometry_for_scaled_storage(
    quant_mode: str,
) -> None:
    plan = benchmark.build_embedding_plan(
        benchmark.PACKED_PROFILES[0],
        device="cpu",
        quant_mode=quant_mode,
        geometry=benchmark.EMBEDDING_GEOMETRIES["storage-scaled"],
        table_memory="device",
    )

    assert isinstance(plan, ple_embedding.Plan)
    assert plan.head_count == 16
    assert plan.head_dim == 160
    assert plan.output_shape == (1, 2560)
    assert plan.caps.tp_size == 4
    assert plan.caps.tp_rank == 0
    assert plan.caps.base_table_size == benchmark.STORAGE_SCALED_BASE_TABLE_SIZE


def test_embedding_builder_uses_public_storage_and_bind_contract_on_cpu() -> None:
    profile = benchmark.PackedProfile("unit", "decode", (2,))
    geometry = benchmark.EmbeddingGeometry(
        name="unit-storage",
        base_table_size=5,
        storage_scope="test-only storage geometry",
        production_storage=False,
    )
    case = benchmark.build_embedding_case(
        profile,
        device="cpu",
        seed=13,
        quant_mode="bf16",
        geometry=geometry,
        table_memory="device",
    )
    try:
        assert isinstance(case.binding, ple_embedding.Binding)
        assert isinstance(case.storage, ple_embedding.TableStorage)
        assert case.binding.plan.head_count == 16
        assert case.binding.plan.head_dim == 160
        assert case.binding.out.shape == (2, 2560)
        assert bool(torch.isfinite(case.expected.float()).all())
        assert int(torch.count_nonzero(case.expected)) > 0
        with pytest.raises(ValueError, match="GPU run requires CUDA"):
            case.launch()
    finally:
        case.close()


@pytest.mark.parametrize("quant_mode", benchmark.QUANT_MODES)
def test_embedding_fixture_distinguishes_live_rows_and_zeros_inactive_rows(
    quant_mode: str,
) -> None:
    profile = benchmark.PackedProfile("unit", "decode", (2,))
    geometry = benchmark.EmbeddingGeometry(
        name="unit-storage",
        base_table_size=5,
        storage_scope="test-only storage geometry",
        production_storage=False,
    )
    case = benchmark.build_embedding_case(
        profile,
        device="cpu",
        seed=13,
        quant_mode=quant_mode,
        geometry=geometry,
        table_memory="device",
    )
    try:
        weight = case.storage.weight_load_view
        nonzero_rows = torch.count_nonzero(weight, dim=1) != 0
        initialized = int(nonzero_rows.sum().item())
        distinct = torch.unique(weight[nonzero_rows].float(), dim=0)

        assert initialized == case.initialized_local_rows
        assert initialized > 1
        assert distinct.shape[0] == initialized
        assert bool(torch.any(~nonzero_rows))
    finally:
        case.close()


@pytest.mark.parametrize("mode", ["decode", "prefill", "mixed"])
def test_layer_builder_uses_public_plan_and_bind_at_exact_geometry(mode: str) -> None:
    profile = next(
        profile for profile in benchmark.LAYER_PROFILES if profile.mode == mode
    )
    if profile.tokens > 4:
        profile = benchmark.LayerProfile(
            name=f"unit-{mode}",
            mode=mode,
            query_lengths=(1, 1) if mode == "mixed" else (1,),
            state_is_fresh=(True, False) if mode == "mixed" else (False,),
            num_accepted_tokens=(0, 1)
            if mode == "mixed"
            else ((0,) if mode == "prefill" else (1,)),
            request_is_prefill=(True, False) if mode == "mixed" else None,
        )
    case = benchmark.build_layer_case(
        profile,
        device="cpu",
        seed=17,
        compute_reference=False,
    )

    assert isinstance(case.binding, ple.Binding)
    caps = case.binding.plan.caps
    assert (caps.streams, caps.hidden_size) == (4, 2560)
    assert (caps.kernel_size, caps.dilation) == (4, 3)
    assert caps.max_speculative_tokens == 4
    assert case.binding.plan.state_length == 9
    assert case.binding.plan.state_capacity == 13
    assert case.binding.out.shape == (profile.tokens, 4, 2560)
    with pytest.raises(ValueError, match="GPU run requires CUDA"):
        case.launch()


def test_storage_contract_labels_scaled_results_and_reports_exact_bytes() -> None:
    scaled = benchmark._embedding_storage_contract(
        benchmark.EMBEDDING_GEOMETRIES["storage-scaled"], "bf16"
    )
    production = benchmark._embedding_storage_contract(
        benchmark.EMBEDDING_GEOMETRIES["production"], "bf16"
    )

    assert scaled["production_storage"] is False
    assert "not production evidence" in str(scaled["storage_scope"])
    assert production["production_storage"] is True
    assert production["table_vocab_size"] == 320_001_446
    assert production["padded_vocab_size"] == 320_001_536
    assert production["weight_shape"] == [80_000_384, 160]
    assert production["persistent_nbytes"] == 80_000_384 * 160 * 2
    assert int(production["persistent_nbytes"]) > int(scaled["persistent_nbytes"])


def test_default_cli_is_runnable_and_excludes_mapped_host_and_production() -> None:
    parser = benchmark.build_parser()
    args = benchmark._parse_args(parser, [])

    assert args.selected_apis == benchmark.APIS
    assert args.selected_quant_modes == benchmark.QUANT_MODES
    assert args.selected_table_memories == ("device",)
    assert args.embedding_geometry == "storage-scaled"
    assert args.allow_production_table is False
    assert args.selected_modes == benchmark.TIMING_MODES


def test_mapped_host_is_an_explicit_separate_filter() -> None:
    parser = benchmark.build_parser()
    args = benchmark._parse_args(
        parser,
        [
            "--apis",
            "embedding",
            "--table-memory",
            "mapped_host",
            "--quant-modes",
            "nvfp4_group16",
        ],
    )

    assert args.selected_apis == ("embedding",)
    assert args.selected_table_memories == ("mapped_host",)
    assert args.selected_quant_modes == ("nvfp4_group16",)


def test_production_storage_requires_opt_in_but_can_be_listed_safely() -> None:
    parser = benchmark.build_parser()
    with pytest.raises(SystemExit):
        benchmark._parse_args(
            parser,
            ["--apis", "embedding", "--embedding-geometry", "production"],
        )

    listed = benchmark._parse_args(
        parser,
        [
            "--apis",
            "embedding",
            "--embedding-geometry",
            "production",
            "--list-profiles",
        ],
    )
    assert listed.list_profiles is True
    report = benchmark._profile_listing(listed)
    assert report["embedding_storage"][0]["production_storage"] is True

    opted_in = benchmark._parse_args(
        parser,
        [
            "--apis",
            "embedding",
            "--embedding-geometry",
            "production",
            "--allow-production-table",
        ],
    )
    assert opted_in.allow_production_table is True


def test_filters_preserve_requested_order_and_reject_ambiguous_values() -> None:
    assert benchmark._parse_filter(
        "graph,eager,graph",
        choices=benchmark.TIMING_MODES,
        label="timing modes",
    ) == ("graph", "eager")
    with pytest.raises(ValueError, match="cannot be combined"):
        benchmark._parse_filter("all,hash", choices=benchmark.APIS, label="apis")
    with pytest.raises(ValueError, match="unknown apis"):
        benchmark._parse_filter("glm", choices=benchmark.APIS, label="apis")


def test_timing_summary_preserves_every_raw_sample() -> None:
    summary = benchmark._summary([5.0, 1.0, 4.0, 2.0, 3.0])
    assert summary["median"] == 3.0
    assert summary["minimum"] == 1.0
    assert summary["maximum"] == 5.0
    assert summary["raw_samples_us"] == [5.0, 1.0, 4.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="at least one"):
        benchmark._summary([])


def test_graph_contract_poison_precedes_replay_and_reference_gate() -> None:
    source = inspect.getsource(benchmark._time_case)

    output_poison = source.index('address_tensors["out"].fill_(float("nan"))')
    scratch_poison = source.index('address_tensors["scratch"].fill_(0xFF)')
    replay = source.index("graph.replay()", scratch_poison)
    replay_gate = source.index("correctness = validate()", replay)
    assert output_poison < scratch_poison < replay < replay_gate
    assert '"replay_after_output_poison": True' in source
    assert '"replay_after_scratch_poison": True' in source
    assert "replay_allocation_delta_bytes" in source
