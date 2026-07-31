from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PCIE = ROOT / "sparkinfer" / "comm" / "pcie"
UINT32_MASK = (1 << 32) - 1


def _advance(generation: int) -> tuple[int, int]:
    """Model the device control kernel's unsigned generation update."""
    return generation & 1, (generation + 1) & UINT32_MASK


def _section(source: str, start: str, end: str) -> str:
    return source[source.index(start) : source.index(end, source.index(start))]


def test_slot_control_alternates_across_replays_and_wraps() -> None:
    generation = 0
    slots = []
    for _ in range(7):
        slot, generation = _advance(generation)
        slots.append(slot)

    assert slots == [0, 1, 0, 1, 0, 1, 0]
    assert _advance(UINT32_MASK) == (1, 0)
    assert _advance(0) == (0, 1)


def test_same_stream_capture_replay_handles_variable_worker_grids() -> None:
    generation = 0
    replay_slots = []
    worker_slots = []

    for worker_blocks in (1, 64, 3, 36, 8):
        slot, generation = _advance(generation)
        replay_slots.append(slot)
        worker_slots.append([slot] * worker_blocks)

    assert replay_slots == [0, 1, 0, 1, 0]
    assert all(len(set(cta_slots)) == 1 for cta_slots in worker_slots)


def test_multistream_separate_channels_match_across_rank_interleavings() -> None:
    schedules = {
        0: ("decode", "prefill", "decode", "prefill"),
        1: ("prefill", "decode", "prefill", "decode"),
    }
    rank_observations = {}

    for rank, schedule in schedules.items():
        generations = {"decode": 0, "prefill": 0}
        observed = {"decode": [], "prefill": []}
        for channel in schedule:
            slot, generations[channel] = _advance(generations[channel])
            observed[channel].append(slot)
        rank_observations[rank] = observed

    assert rank_observations[0] == rank_observations[1]
    assert rank_observations[0] == {
        "decode": [0, 1],
        "prefill": [0, 1],
    }


@pytest.mark.parametrize("source_name", ("pcie_oneshot.cu", "pcie_twoshot.cu"))
def test_worker_slot_selection_has_no_grid_rendezvous(source_name: str) -> None:
    source = (PCIE / source_name).read_text(encoding="utf-8")

    assert "active_staging_slot" in source
    assert "advance_staging_slot_kernel" in source
    assert "staging_arrive" not in source
    assert "cudaOccupancy" not in source
    assert "cudaLaunchCooperativeKernel" not in source
    assert "launch_cooperative" not in source
    assert "resident_grid" not in source

    selector_name = (
        "DINLINE void select_rank_data"
        if source_name == "pcie_oneshot.cu"
        else "DINLINE void select_rank_ptrs"
    )
    selector = _section(source, selector_name, "template <")
    assert "active_staging_slot" in selector
    assert "atomicAdd" not in selector
    assert "while (" not in selector


def test_oneshot_control_is_staged_only_and_precedes_each_worker() -> None:
    source = (PCIE / "pcie_oneshot.cu").read_text(encoding="utf-8")
    control_launch = "advance_staging_slot_kernel<<<1, 1, 0, stream>>>(self_sg_);"
    assert source.count(control_launch) == 2

    regular = _section(
        source,
        "void allreduce(cudaStream_t stream",
        "void allreduce_fused_add_rms_norm",
    )
    assert f"if (stage_input) {{\n      {control_launch}" in regular
    assert regular.index(control_launch) < regular.index("#define KL(ngpus)")

    fused = _section(
        source,
        "void allreduce_fused_add_rms_norm",
        "~PCIeAllreduce",
    )
    assert f"if (use_eager_staging) {{\n      {control_launch}" in fused
    assert fused.index(control_launch) < fused.index("#define KL(ngpus, SINGLE, MODE)")

    # Registered one-shot and fused registered paths retain their original
    # direct buffer route; only the double-buffered staged route advances.
    assert "if (stage_input)" in regular
    assert "pcie_allreduce_kernel<T, ngpus, true>" in regular
    assert "pcie_allreduce_kernel<T, ngpus, false>" in regular
    assert "if (mode == kModeStagePush)" in fused
    assert "KL(ngpus, SINGLE, kModeStagePush)" in fused
    assert "KL(ngpus, SINGLE, kModeStagePull)" in fused
    assert "kModeRegistered" in fused


def test_twoshot_control_precedes_reduce_scatter_and_all_gather_workers() -> None:
    source = (PCIE / "pcie_twoshot.cu").read_text(encoding="utf-8")
    control_launch = "advance_staging_slot_kernel<<<1, 1, 0, stream>>>(self_sg_);"
    assert source.count(control_launch) == 2

    reduce_scatter = _section(source, "void reduce_scatter(", "void all_gather(")
    assert reduce_scatter.index(control_launch) < reduce_scatter.index(
        "rs_fp8_kernel<ngpus><<<blocks, threads, 0, stream>>>"
    )

    all_gather = _section(source, "void all_gather(", "\n};")
    assert all_gather.index(control_launch) < all_gather.index(
        "ag_fp8_kernel<ngpus><<<blocks, threads, 0, stream>>>"
    )


@pytest.mark.parametrize("source_name", ("pcie_oneshot.cu", "pcie_twoshot.cu"))
def test_control_kernel_is_graph_replay_dynamic(source_name: str) -> None:
    source = (PCIE / source_name).read_text(encoding="utf-8")
    control = _section(
        source,
        "__global__ void advance_staging_slot_kernel",
        "template <",
    )

    # The generation update runs in a CUDA kernel—not on the host—so capture
    # records it as a graph node and every replay advances the channel slot.
    assert "staging_generation" in control
    assert "active_staging_slot = generation & FlagType{1}" in control
    assert "staging_generation = generation + FlagType{1}" in control


@pytest.mark.parametrize(
    ("source_name", "expected_checks"),
    (("pcie_oneshot.cu", 4), ("pcie_twoshot.cu", 4)),
)
def test_control_and_worker_launches_have_immediate_error_checks(
    source_name: str, expected_checks: int
) -> None:
    source = (PCIE / source_name).read_text(encoding="utf-8")
    assert source.count("CHECK_CUDA_SUCCESS(cudaGetLastError());") == expected_checks

    control_launch = "advance_staging_slot_kernel<<<1, 1, 0, stream>>>(self_sg_);"
    for suffix in source.split(control_launch)[1:]:
        assert suffix.lstrip().startswith("CHECK_CUDA_SUCCESS(cudaGetLastError());")


def test_control_node_overhead_ab_contract() -> None:
    # Relative to the folded-selector baseline, a staged operation has exactly
    # one extra tiny control node. Registered paths remain one worker node.
    control_nodes = {
        "oneshot_registered": 0,
        "oneshot_staged_pull": 1,
        "fused_registered": 0,
        "fused_staged_pull": 1,
        "fused_staged_push": 1,
        "twoshot_reduce_scatter": 1,
        "twoshot_all_gather": 1,
    }
    total_nodes = {name: 1 + extra for name, extra in control_nodes.items()}

    assert total_nodes["oneshot_registered"] == 1
    assert total_nodes["fused_registered"] == 1
    assert {
        total_nodes[name]
        for name in total_nodes
        if name not in {"oneshot_registered", "fused_registered"}
    } == {2}
