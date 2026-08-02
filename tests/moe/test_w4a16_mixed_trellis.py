from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch

pytest.importorskip("cutlass")

from sparkinfer.moe._shared.kernels.w4a16.host import (
    make_w4a16_packed_buffers,
    max_packed_route_slots,
)
from sparkinfer.moe._shared.kernels.w4a16.kernel import (
    W4A16FusedMoeKernel,
    W4A16TopKSumKernel,
    run_w4a16_moe,
)
from sparkinfer.moe._shared.kernels.w4a16.mixed_trellis import (
    W4A16MixedTrellisKernel,
    build_tiered_maps,
    combine_trellis_rotations,
    compile_mixed_trellis,
    make_mixed_trellis_buffers,
    run_mixed_trellis,
)
from sparkinfer.moe._shared.kernels.w4a16.prepare import (
    prepare_trellis256_moe_weights,
)


def _sm12x_available() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return major == 12 and minor in (0, 1)


def test_mixed_kernel_tracks_shared_moe_body_contract() -> None:
    """Keep the direct CuTe call aligned with the shared driver's ABI."""

    tree = ast.parse(textwrap.dedent(inspect.getsource(W4A16MixedTrellisKernel.kernel)))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_moe_body"
    ]
    assert len(calls) == 1

    driver_parameters = inspect.signature(W4A16FusedMoeKernel._moe_body).parameters
    assert len(calls[0].args) + len(calls[0].keywords) == len(driver_parameters) - 1
    assert [ast.unparse(arg) for arg in calls[0].args[-10:]] == [
        "descriptor_map",
        "Int32(self.total_experts)",
        "Int32(self.total_experts)",
        "smem_base",
        "tid",
        "cta",
        "grid_x",
        "active_m",
        "fc1_emit",
        "fc2_emit",
    ]


def test_mixed_runtime_tracks_topk_sum_contract() -> None:
    """Keep the direct compiled top-k launch aligned with its runtime ABI."""

    tree = ast.parse(textwrap.dedent(inspect.getsource(run_mixed_trellis)))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "compiled"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "topk_sum"
    ]
    assert len(calls) == 1

    sum_parameters = inspect.signature(W4A16TopKSumKernel.__call__).parameters
    assert len(calls[0].args) + len(calls[0].keywords) == len(sum_parameters) - 1
    assert [ast.unparse(arg) for arg in calls[0].args[-4:]] == [
        "Int32(launch.topk_sum.num_experts)",
        "Int32(launch.topk_sum.route_num_experts)",
        "m",
        "stream",
    ]


def _prepared(
    *,
    experts: int,
    hidden: int,
    intermediate: int,
    bits: int,
    seed: int,
    device: torch.device,
    tile_config: tuple[int, int, int, int] = (128, 128, 128, 128),
):
    generator = torch.Generator(device=device).manual_seed(seed)

    def scales(shape: tuple[int, ...]) -> torch.Tensor:
        return (
            0.875 + 0.25 * torch.rand(shape, generator=generator, device=device)
        ).to(torch.float16)

    return prepare_trellis256_moe_weights(
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_experts=experts,
        activation="silu",
        fc1_tile_n=tile_config[1],
        fc2_tile_n=tile_config[3],
        device=device,
        seed=seed,
        params_dtype=torch.float16,
        w13_layout="trellis3_t256_proj",
        trellis_bits=bits,
        gate_suh=scales((experts, hidden)),
        up_suh=scales((experts, hidden)),
        intermediate_rotations=scales((experts, 3 * intermediate)),
        down_svh=scales((experts, hidden)),
        tile_config=tile_config,
    )


def _serial_tier(
    x: torch.Tensor,
    prepared,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
    block_size_m: int = 8,
) -> torch.Tensor:
    m, topk = int(topk_ids.shape[0]), int(topk_ids.shape[1])
    buffers = make_w4a16_packed_buffers(
        prepared,
        m=m,
        topk=topk,
        dtype=torch.float16,
        device=x.device,
        route_num_experts=int(expert_map.numel()),
        full_rotation=True,
        block_size_m=block_size_m,
    )
    assert buffers.rotation_a_gate is not None
    assert buffers.rotation_a_up is not None
    return run_w4a16_moe(
        x,
        prepared,
        topk_weights,
        topk_ids,
        activation="silu",
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output,
        fc1_c_tmp=buffers.fc1_c_tmp,
        fc2_c_tmp=buffers.fc2_c_tmp,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
        expert_counts=buffers.expert_counts,
        expert_map=expert_map,
        output_expert_map=expert_map,
        route_block_size_m=block_size_m,
        intermediate_rotation_scales=prepared.intermediate_rotations,
        full_rotation=True,
        suh_gate_table=prepared.gate_suh,
        suh_up_table=prepared.up_suh,
        svh_table=prepared.down_svh,
        rotation_a_gate=buffers.rotation_a_gate,
        rotation_a_up=buffers.rotation_a_up,
    )


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize("route_ids_dtype", [torch.int32, torch.int64])
def test_mixed_k3_k4_matches_serial_and_captures(
    route_ids_dtype: torch.dtype,
) -> None:
    torch.manual_seed(20260730)
    device = torch.device("cuda", torch.cuda.current_device())
    m, hidden, intermediate, topk = 2, 128, 128, 2
    tier0 = _prepared(
        experts=2,
        hidden=hidden,
        intermediate=intermediate,
        bits=3,
        seed=301,
        device=device,
    )
    tier1 = _prepared(
        experts=2,
        hidden=hidden,
        intermediate=intermediate,
        bits=4,
        seed=401,
        device=device,
    )
    x = (torch.randn((m, hidden), device=device) * 1.0e-3).to(torch.bfloat16)
    # Global expert ids deliberately interleave K3 and K4 tiers. The combined
    # namespace remains tier ordered so weight and rotation tables stay dense.
    topk_ids = torch.tensor([[0, 1], [3, 2]], dtype=route_ids_dtype, device=device)
    topk_weights = torch.tensor(
        [[0.65, 0.35], [0.2, 0.8]], dtype=torch.float32, device=device
    )
    map0 = torch.tensor([1, -1, 0, -1], dtype=torch.int32, device=device)
    map1 = torch.tensor([-1, 1, -1, 0], dtype=torch.int32, device=device)
    serial = _serial_tier(x, tier0, topk_weights, topk_ids, map0)
    serial = serial + _serial_tier(x, tier1, topk_weights, topk_ids, map1)
    torch.cuda.synchronize(device)

    props = torch.cuda.get_device_properties(device)
    launch = compile_mixed_trellis(
        size_m=m,
        hidden_size=hidden,
        intermediate_size=intermediate,
        tier0_num_experts=2,
        tier1_num_experts=2,
        top_k=topk,
        max_m_blocks=8,
        sms=int(props.multi_processor_count),
        max_shared_mem=int(props.shared_memory_per_block_optin),
        force_tile_config=(128, 128, 128, 128),
        route_ids_dtype=route_ids_dtype,
    )
    assert launch.local_memory_bytes == 0
    global_to_combined, descriptor = build_tiered_maps((2, 0), (3, 1), device=device)
    rotations = combine_trellis_rotations(tier0, tier1)
    buffers = make_mixed_trellis_buffers(
        launch, device=device, sms=int(props.multi_processor_count)
    )
    assert buffers.fc2.data_ptr() == buffers.rotation_gate.data_ptr()

    eager = run_mixed_trellis(
        x,
        tier0,
        tier1,
        topk_weights,
        topk_ids,
        global_to_combined,
        descriptor,
        rotations,
        launch,
        buffers,
    )
    torch.cuda.synchronize(device)
    eager = eager.clone()
    assert not torch.isnan(eager).any()
    relative = (eager - serial).norm() / serial.norm().clamp_min(1.0e-12)
    assert float(relative) < 4.0e-3

    repeat = run_mixed_trellis(
        x,
        tier0,
        tier1,
        topk_weights,
        topk_ids,
        global_to_combined,
        descriptor,
        rotations,
        launch,
        buffers,
    )
    torch.cuda.synchronize(device)
    assert torch.equal(repeat, eager)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run_mixed_trellis(
            x,
            tier0,
            tier1,
            topk_weights,
            topk_ids,
            global_to_combined,
            descriptor,
            rotations,
            launch,
            buffers,
        )
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(captured, eager)

    # An unmapped global expert must contribute zero without changing any
    # mapped route. This is the contract needed by expert-parallel placement.
    skipped_map1 = map1.clone()
    skipped_map1[3] = -1
    skipped_serial = _serial_tier(
        x, tier0, topk_weights, topk_ids, map0
    ) + _serial_tier(x, tier1, topk_weights, topk_ids, skipped_map1)
    skipped_global_to_combined = global_to_combined.clone()
    skipped_global_to_combined[3] = -1
    skipped = run_mixed_trellis(
        x,
        tier0,
        tier1,
        topk_weights,
        topk_ids,
        skipped_global_to_combined,
        descriptor,
        rotations,
        launch,
        buffers,
    )
    torch.cuda.synchronize(device)
    skipped_relative = (
        skipped - skipped_serial
    ).norm() / skipped_serial.norm().clamp_min(1.0e-12)
    assert float(skipped_relative) < 4.0e-3


def test_build_tiered_maps_rejects_invalid_partitions() -> None:
    with pytest.raises(ValueError, match="disjoint partition"):
        build_tiered_maps((0, 1), (1, 2), device=torch.device("cpu"))
    with pytest.raises(ValueError, match="disjoint partition"):
        build_tiered_maps((0, 4), (1, 2), device=torch.device("cpu"))


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize("candidate_block_size", [32, 64])
def test_one_grid_large_blocks_avoid_serial_prefill_drift(
    candidate_block_size: int,
) -> None:
    """Large route blocks with paired-M8 FC2 preserve one-grid arithmetic."""

    torch.manual_seed(20260801)
    device = torch.device("cuda", torch.cuda.current_device())
    m, hidden, intermediate, topk = 64, 512, 256, 8
    tile_config = (128, 128, 32, 512)
    tier0_experts, tier1_experts = 6, 2

    prepared_tiers = tuple(
        _prepared(
            experts=experts,
            hidden=hidden,
            intermediate=intermediate,
            bits=bits,
            seed=seed,
            device=device,
            tile_config=tile_config,
        )
        for experts, bits, seed in (
            (tier0_experts, 3, 301),
            (tier1_experts, 4, 401),
        )
    )

    x = (torch.randn((m, hidden), device=device) * 1.0e-3).to(torch.bfloat16)
    topk_ids = (
        torch.tensor([0, 6, 1, 7, 2, 3, 4, 5], dtype=torch.int32, device=device)
        .expand(m, -1)
        .contiguous()
    )
    topk_weights = torch.softmax(
        torch.randn((m, topk), dtype=torch.float32, device=device), dim=-1
    )
    props = torch.cuda.get_device_properties(device)
    global_to_combined, descriptor = build_tiered_maps(
        range(tier0_experts), range(tier0_experts, 8), device=device
    )

    def one_grid(
        block_size_m: int,
    ) -> tuple[torch.Tensor, object, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        route_slots = max_packed_route_slots(m * topk, block_size_m, 8)
        launch = compile_mixed_trellis(
            size_m=m,
            hidden_size=hidden,
            intermediate_size=intermediate,
            tier0_num_experts=tier0_experts,
            tier1_num_experts=tier1_experts,
            top_k=topk,
            max_m_blocks=(route_slots + block_size_m - 1) // block_size_m,
            moe_block_size=block_size_m,
            sms=int(props.multi_processor_count),
            max_shared_mem=int(props.shared_memory_per_block_optin),
            force_tile_config=tile_config,
        )
        buffers = make_mixed_trellis_buffers(
            launch, device=device, sms=int(props.multi_processor_count)
        )
        output = run_mixed_trellis(
            x,
            prepared_tiers[0],
            prepared_tiers[1],
            topk_weights,
            topk_ids,
            global_to_combined,
            descriptor,
            combine_trellis_rotations(*prepared_tiers),
            launch,
            buffers,
        ).clone()
        phase_outputs = (
            buffers.fc1.clone(),
            buffers.activated.clone(),
            buffers.fc2.clone(),
        )
        return output, launch, phase_outputs

    reference, reference_launch, reference_phases = one_grid(8)
    candidate, candidate_launch, candidate_phases = one_grid(candidate_block_size)
    torch.cuda.synchronize(device)

    phase_equal = tuple(
        torch.equal(candidate_phase, reference_phase)
        for candidate_phase, reference_phase in zip(
            candidate_phases, reference_phases, strict=True
        )
    )
    geometry = (
        f"packed_block={candidate_launch.moe_block_size} "
        f"fc2_subtile={candidate_launch.fc2_moe_block_size} "
        f"fc2_schedule_factor={candidate_launch.fc2_schedule_route_block_factor} "
        f"regs={candidate_launch.registers_per_thread} "
        f"local={candidate_launch.local_memory_bytes} "
        f"smem={candidate_launch.shared_memory_bytes}"
    )
    assert reference_launch.moe_block_size == 8
    assert reference_launch.fc2_moe_block_size == 8
    assert reference_launch.fc2_schedule_route_block_factor == 1
    assert candidate_launch.moe_block_size == candidate_block_size
    assert candidate_launch.fc2_moe_block_size == 8
    assert candidate_launch.fc2_schedule_route_block_factor == 2
    assert candidate_launch.fc2_paired_m8_routes is True
    assert phase_equal == (True, True, True), geometry
    assert torch.equal(candidate, reference), geometry
    assert candidate_launch.local_memory_bytes == 0
    assert candidate_launch.shared_memory_bytes <= int(
        props.shared_memory_per_block_optin
    )


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
def test_glm52_large_m_mixed_k3_k4_matches_serial() -> None:
    """Cover the production prefill shape that exposed lost FC1 reductions."""

    torch.manual_seed(20260730)
    device = torch.device("cuda", torch.cuda.current_device())
    m, hidden, intermediate, topk = 3072, 6144, 512, 8
    tile_config = (128, 128, 32, 512)
    tier0_experts, tier1_experts = 192, 64
    tier0 = _prepared(
        experts=tier0_experts,
        hidden=hidden,
        intermediate=intermediate,
        bits=3,
        seed=301,
        device=device,
        tile_config=tile_config,
    )
    tier1 = _prepared(
        experts=tier1_experts,
        hidden=hidden,
        intermediate=intermediate,
        bits=4,
        seed=401,
        device=device,
        tile_config=tile_config,
    )
    x = (torch.randn((m, hidden), device=device) * 1.0e-3).to(torch.bfloat16)
    route_row = torch.tensor(
        [0, 1, 2, 3, 4, 5, tier0_experts, tier0_experts + 1],
        dtype=torch.int32,
        device=device,
    )
    topk_ids = route_row.expand(m, -1).contiguous()
    topk_weights = torch.softmax(
        torch.randn((m, topk), dtype=torch.float32, device=device), dim=-1
    )
    map0 = torch.cat(
        (
            torch.arange(tier0_experts, dtype=torch.int32, device=device),
            torch.full((tier1_experts,), -1, dtype=torch.int32, device=device),
        )
    )
    map1 = torch.cat(
        (
            torch.full((tier0_experts,), -1, dtype=torch.int32, device=device),
            torch.arange(tier1_experts, dtype=torch.int32, device=device),
        )
    )
    serial = _serial_tier(x, tier0, topk_weights, topk_ids, map0).clone()
    serial.add_(_serial_tier(x, tier1, topk_weights, topk_ids, map1))

    props = torch.cuda.get_device_properties(device)
    route_slots = max_packed_route_slots(m * topk, 8, 256)
    launch = compile_mixed_trellis(
        size_m=m,
        hidden_size=hidden,
        intermediate_size=intermediate,
        tier0_num_experts=tier0_experts,
        tier1_num_experts=tier1_experts,
        top_k=topk,
        max_m_blocks=(route_slots + 7) // 8,
        sms=int(props.multi_processor_count),
        max_shared_mem=int(props.shared_memory_per_block_optin),
        force_tile_config=tile_config,
    )
    assert launch.moe_block_size == 8
    global_to_combined, descriptor = build_tiered_maps(
        range(tier0_experts), range(tier0_experts, 256), device=device
    )
    buffers = make_mixed_trellis_buffers(
        launch, device=device, sms=int(props.multi_processor_count)
    )
    mixed = run_mixed_trellis(
        x,
        tier0,
        tier1,
        topk_weights,
        topk_ids,
        global_to_combined,
        descriptor,
        combine_trellis_rotations(tier0, tier1),
        launch,
        buffers,
    )
    torch.cuda.synchronize(device)

    relative = (serial - mixed).norm() / serial.norm().clamp_min(1.0e-12)
    # The serial oracle sums two independently reduced tier outputs, while the
    # mixed grid reduces them in one schedule. Normal rounding stays below
    # 1e-4; the unsafe 64x256 FC1 geometry was three orders larger (~8e-3).
    assert float(relative) < 1.0e-4
