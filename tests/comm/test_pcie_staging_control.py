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
    assert "cudaLaunchCooperativeKernel" not in source
    assert "launch_cooperative" not in source

    selector_name = (
        "DINLINE void select_rank_data"
        if source_name == "pcie_oneshot.cu"
        else "DINLINE void select_rank_ptrs"
    )
    selector = _section(source, selector_name, "template <")
    assert "active_staging_slot" in selector
    assert "atomicAdd" not in selector
    assert "while (" not in selector


def test_fused_rms_residency_is_runtime_derived_and_capture_cached() -> None:
    source = (PCIE / "pcie_oneshot.cu").read_text(encoding="utf-8")

    assert "cudaOccupancyMaxActiveBlocksPerMultiprocessor" in source
    assert "cudaDevAttrMultiProcessorCount" in source
    assert "fused_resident_capacity_" in source
    assert "cap_fused_ctas_per_row" in source
    assert "SPARKINFER_PCIE_TEST_RESIDENT_GRID_CAPACITY" in source
    assert "kMaxBlocks <= SM count" not in source

    # Model the native clamp for reduced/MIG-like capacities. A capacity below
    # the requested multi-CTA grid must select the barrier-free single-CTA path.
    def cap(requested: int, rows: int, resident: int) -> int:
        return max(1, min(requested, max(1, resident // rows)))

    assert cap(3, 1, 1) == 1
    assert cap(3, 1, 2) == 2
    assert cap(3, 2, 2) == 1
    assert cap(3, 2, 8) == 3


@pytest.mark.parametrize(
    ("source_name", "python_name", "constant_name", "required_sms"),
    (
        ("pcie_oneshot.cu", "pcie_oneshot.py", "ONESHOT_REQUIRED_SMS", 36),
        ("pcie_twoshot.cu", "pcie_twoshot.py", "TWOSHOT_REQUIRED_SMS", 64),
        ("pcie_dcp_a2a.cu", "pcie_dcp_a2a.py", "DCP_A2A_REQUIRED_SMS", 64),
    ),
)
def test_peer_waiting_grid_is_fully_resident_or_rejected_before_ipc(
    source_name: str,
    python_name: str,
    constant_name: str,
    required_sms: int,
) -> None:
    source = (PCIE / source_name).read_text(encoding="utf-8")
    wrapper = (PCIE / python_name).read_text(encoding="utf-8")

    assert f"constexpr int kMaxBlocks = {required_sms};" in source
    assert source.count("__global__ void __launch_bounds__(512, 1)") == 2
    assert f"{constant_name} = {required_sms}" in wrapper
    assert "_require_full_grid_residency(" in wrapper

    # All peer-waiting worker launches use no dynamic shared memory. Combined
    # with launch_bounds(..., 1), one CTA is resident per visible SM; requiring
    # kMaxBlocks SMs makes every possible worker CTA simultaneously resident.
    worker_launches = [
        line
        for line in source.splitlines()
        if "<<<blocks, threads, 0, stream>>>" in line
    ]
    assert worker_launches
    assert all(", 0, stream>>>" in line for line in worker_launches)


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


def test_twoshot_validates_launch_geometry_before_grid_division() -> None:
    source = (PCIE / "pcie_twoshot.cu").read_text(encoding="utf-8")
    validator = _section(source, "void check_launch_config(", "void reduce_scatter(")

    assert "threads < world_size_" in validator
    assert "threads > 1024" in validator
    assert "threads % 32 != 0" in validator
    assert "block_limit <= 0" in validator
    assert "block_limit > kMaxBlocks" in validator

    reduce_scatter = _section(source, "void reduce_scatter(", "void all_gather(")
    all_gather = _section(source, "void all_gather(", "\n};")
    for operation in (reduce_scatter, all_gather):
        assert operation.index("check_launch_config(threads, block_limit);") < (
            operation.index("(shard_packs + threads - 1) / threads")
        )


def test_dcp_control_selects_one_slot_per_operation_before_workers() -> None:
    source = (PCIE / "pcie_dcp_a2a.cu").read_text(encoding="utf-8")
    control_launch = "advance_staging_slot_kernel<<<1, 1, 0, stream>>>(self_signal_);"
    assert source.count(control_launch) == 2

    selector = _section(source, "DINLINE void select_staging", "// pre_sync")
    assert "active_staging_slot" in selector
    assert "self_counter[blockIdx.x]" not in selector
    assert "atomicAdd" not in selector
    assert "while (" not in selector

    reduce_scatter = _section(
        source, "void run(cudaStream_t stream", "void all_gather_heads("
    )
    gather = _section(source, "void all_gather_heads(", "\n};")
    assert reduce_scatter.index(control_launch) < reduce_scatter.index(
        "#define LAUNCH(world)"
    )
    assert gather.index(control_launch) < gather.index("#define LAUNCH(world)")
    assert "SPARKINFER_PCIE_DCP_TEST_POST_BARRIER_DELAY_CYCLES" in source
    assert "test_post_barrier_delay(rank, delayed_rank, delay_cycles);" in source


@pytest.mark.parametrize(
    "source_name", ("pcie_oneshot.cu", "pcie_twoshot.cu", "pcie_dcp_a2a.cu")
)
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
    assert source.count("CHECK_CUDA_SUCCESS(cudaGetLastError());") >= expected_checks

    control_launch = "advance_staging_slot_kernel<<<1, 1, 0, stream>>>(self_sg_);"
    for suffix in source.split(control_launch)[1:]:
        assert suffix.lstrip().startswith("CHECK_CUDA_SUCCESS(cudaGetLastError());")


@pytest.mark.parametrize(
    "source_name",
    ("pcie_oneshot.cu", "pcie_twoshot.cu", "pcie_dcp_a2a.cu"),
)
def test_each_staged_collective_has_one_control_launch(source_name: str) -> None:
    source = (PCIE / source_name).read_text(encoding="utf-8")
    control_launch = "advance_staging_slot_kernel<<<1, 1, 0, stream>>>"

    # Each source implements exactly two staged collectives. Registered
    # one-shot paths remain control-free; their conditional gating is covered
    # by the neighboring source and GPU graph tests.
    assert source.count(control_launch) == 2
