from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace

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
    MixedTrellisRotations,
    W4A16MixedTrellisKernel,
    _validate_mixed_trellis_tier_storage,
    build_ordered_maps,
    build_tiered_maps,
    combine_trellis_rotations,
    compile_mixed_trellis,
    make_mixed_trellis_buffers,
    run_mixed_trellis,
)
from sparkinfer.moe._shared.kernels.w4a16.prepare import (
    prepare_trellis256_moe_weights,
)


def _mixed_cache_key(tier0_experts: int, tier1_experts: int) -> tuple[object, ...]:
    """Build the key without constructing CUDA-backed kernels.

    The tier subkeys are stubbed, so this proves only that the outer mixed key
    does not read expert counts directly. The GPU ABBA test covers real child
    keys and verifies that they resolve to one compiled object.
    """

    kernel = object.__new__(W4A16MixedTrellisKernel)
    kernel.driver = SimpleNamespace(__cache_key__=("driver",))
    kernel.tier0 = SimpleNamespace(
        __cache_key__=("dynamic-k3",), num_experts=tier0_experts
    )
    kernel.tier1 = SimpleNamespace(
        __cache_key__=("dynamic-k4",), num_experts=tier1_experts
    )
    kernel.blocks_per_sm = 1
    kernel.shared_words = 1
    return kernel.__cache_key__


def test_mixed_kernel_cache_key_is_tier_partition_agnostic() -> None:
    # The exact K3/K4 partition is checkpoint data, not launch geometry. One
    # compiled object must serve every 256-expert GLM-5.2 mixed layout.
    assert _mixed_cache_key(206, 50) == _mixed_cache_key(160, 96)
    assert _mixed_cache_key(192, 64) == _mixed_cache_key(206, 50)
    assert _mixed_cache_key(96, 32) == _mixed_cache_key(80, 16)


def test_mixed_kernel_uses_runtime_expert_bounds() -> None:
    emit_source = textwrap.dedent(
        inspect.getsource(W4A16MixedTrellisKernel._emit_tier_tile)
    )
    kernel_source = textwrap.dedent(inspect.getsource(W4A16MixedTrellisKernel.kernel))
    call_parameters = inspect.signature(W4A16MixedTrellisKernel.__call__).parameters

    assert "tier0_num_experts" in call_parameters
    assert "tier1_num_experts" in call_parameters
    assert "self.tier0.num_experts" not in emit_source
    assert "self.tier1.num_experts" not in emit_source
    assert "self.total_experts" not in emit_source
    assert "self.total_experts" not in kernel_source
    assert "local_expert < tier0_num_experts" in emit_source
    assert "local_expert < tier1_num_experts" in emit_source


def test_mixed_runtime_rejects_invalid_raw_tier_storage() -> None:
    device = torch.device("cpu")
    experts, hidden, intermediate, bits = 2, 128, 128, 3
    tier = SimpleNamespace(
        num_experts=experts,
        w13=torch.empty(
            experts * (hidden // 16) * ((2 * intermediate) // 16) * (8 * bits),
            dtype=torch.int32,
        ),
        w2=torch.empty(
            experts * (intermediate // 16) * (hidden // 16) * (8 * bits),
            dtype=torch.int32,
        ),
        w13_scale=torch.empty(4, dtype=torch.uint8),
        w2_scale=torch.empty(4, dtype=torch.uint8),
        w13_global_scale=torch.empty(experts, dtype=torch.float32),
        w2_global_scale=torch.empty(experts, dtype=torch.float32),
    )

    def validate(candidate) -> None:
        _validate_mixed_trellis_tier_storage(
            name="tier0",
            tier=candidate,
            expected_experts=experts,
            bits=bits,
            hidden_size=hidden,
            intermediate_size=intermediate,
            device=device,
        )

    validate(tier)
    for field, replacement in (
        ("w13", tier.w13[:-1]),
        ("w2", tier.w2.to(torch.int16)),
        ("w13_scale", tier.w13_scale[:3]),
        ("w2_global_scale", tier.w2_global_scale[:1]),
    ):
        candidate = SimpleNamespace(**{**vars(tier), field: replacement})
        with pytest.raises(ValueError, match=rf"tier0\.{field}"):
            validate(candidate)


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
        "total_experts",
        "total_experts",
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
    shared_h: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
):
    generator = torch.Generator(device=device).manual_seed(seed)

    def scales(shape: tuple[int, ...]) -> torch.Tensor:
        return (
            0.875 + 0.25 * torch.rand(shape, generator=generator, device=device)
        ).to(torch.float16)

    if shared_h is None:
        gate_suh = scales((experts, hidden))
        up_suh = scales((experts, hidden))
        intermediate_rotations = scales((experts, 3 * intermediate))
        down_svh = scales((experts, hidden))
    else:
        gate_suh, up_suh, down_svh = shared_h
        intermediate_rotations = scales((experts, 3 * intermediate))
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
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=intermediate_rotations,
        down_svh=down_svh,
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

    misaligned_x = torch.empty(m * hidden + 1, dtype=torch.bfloat16, device=device)[
        1:
    ].view(m, hidden)
    assert misaligned_x.is_contiguous()
    assert misaligned_x.data_ptr() % 16 != 0
    with pytest.raises(ValueError, match=r"input.*16-byte alignment"):
        run_mixed_trellis(
            misaligned_x,
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

    misaligned_intermediate = torch.empty(
        rotations.intermediate.numel() + 1, dtype=torch.float16, device=device
    )[1:]
    assert misaligned_intermediate.is_contiguous()
    assert misaligned_intermediate.data_ptr() % 16 != 0
    misaligned_rotations = type(rotations)(
        intermediate=misaligned_intermediate,
        gate_suh=rotations.gate_suh,
        up_suh=rotations.up_suh,
        down_svh=rotations.down_svh,
    )
    with pytest.raises(ValueError, match=r"intermediate rotations.*16-byte alignment"):
        run_mixed_trellis(
            x,
            tier0,
            tier1,
            topk_weights,
            topk_ids,
            global_to_combined,
            descriptor,
            misaligned_rotations,
            launch,
            buffers,
        )

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


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
def test_mixed_k3_k4_shared_h_matches_expanded_and_captures() -> None:
    """A physical one-row H rotation must stay broadcast through mixed K3/K4."""

    torch.manual_seed(20260804)
    device = torch.device("cuda", torch.cuda.current_device())
    m, hidden, intermediate, topk = 2, 128, 128, 2
    generator = torch.Generator(device=device).manual_seed(20260804)

    def shared_row() -> torch.Tensor:
        return (
            0.875
            + 0.25 * torch.rand((1, hidden), generator=generator, device=device)
        ).to(torch.float16)

    shared_h = (shared_row(), shared_row(), shared_row())
    tier0 = _prepared(
        experts=2,
        hidden=hidden,
        intermediate=intermediate,
        bits=3,
        seed=301,
        device=device,
        shared_h=shared_h,
    )
    tier1 = _prepared(
        experts=2,
        hidden=hidden,
        intermediate=intermediate,
        bits=4,
        seed=401,
        device=device,
        shared_h=shared_h,
    )
    x = (torch.randn((m, hidden), device=device) * 1.0e-3).to(torch.bfloat16)
    topk_ids = torch.tensor([[0, 1], [3, 2]], dtype=torch.int32, device=device)
    topk_weights = torch.tensor(
        [[0.65, 0.35], [0.2, 0.8]], dtype=torch.float32, device=device
    )
    map0 = torch.tensor([1, -1, 0, -1], dtype=torch.int32, device=device)
    map1 = torch.tensor([-1, 1, -1, 0], dtype=torch.int32, device=device)
    serial = _serial_tier(x, tier0, topk_weights, topk_ids, map0)
    serial.add_(_serial_tier(x, tier1, topk_weights, topk_ids, map1))

    intermediate_rotations = torch.cat(
        (tier0.intermediate_rotations, tier1.intermediate_rotations), dim=0
    ).contiguous()
    broadcast_rotations = MixedTrellisRotations(
        intermediate=intermediate_rotations,
        gate_suh=shared_h[0],
        up_suh=shared_h[1],
        down_svh=shared_h[2],
    )
    total_experts = int(tier0.num_experts + tier1.num_experts)
    expanded_rotations = MixedTrellisRotations(
        intermediate=intermediate_rotations,
        gate_suh=shared_h[0].expand(total_experts, -1).contiguous(),
        up_suh=shared_h[1].expand(total_experts, -1).contiguous(),
        down_svh=shared_h[2].expand(total_experts, -1).contiguous(),
    )

    props = torch.cuda.get_device_properties(device)

    def compile_launch(*, broadcast: bool):
        return compile_mixed_trellis(
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
            broadcast_suh=broadcast,
            broadcast_svh=broadcast,
        )

    broadcast_launch = compile_launch(broadcast=True)
    expanded_launch = compile_launch(broadcast=False)
    assert broadcast_launch.broadcast_suh is True
    assert broadcast_launch.broadcast_svh is True
    assert expanded_launch.broadcast_suh is False
    assert expanded_launch.broadcast_svh is False
    assert broadcast_launch.compiled is not expanded_launch.compiled

    global_to_combined, descriptor = build_tiered_maps((2, 0), (3, 1), device=device)
    broadcast_buffers = make_mixed_trellis_buffers(
        broadcast_launch, device=device, sms=int(props.multi_processor_count)
    )
    expanded_buffers = make_mixed_trellis_buffers(
        expanded_launch, device=device, sms=int(props.multi_processor_count)
    )
    with pytest.raises(ValueError, match=r"gate SUH.*512 elements"):
        run_mixed_trellis(
            x,
            tier0,
            tier1,
            topk_weights,
            topk_ids,
            global_to_combined,
            descriptor,
            broadcast_rotations,
            expanded_launch,
            expanded_buffers,
        )
    with pytest.raises(ValueError, match=r"gate SUH.*128 elements"):
        run_mixed_trellis(
            x,
            tier0,
            tier1,
            topk_weights,
            topk_ids,
            global_to_combined,
            descriptor,
            expanded_rotations,
            broadcast_launch,
            broadcast_buffers,
        )

    broadcast = run_mixed_trellis(
        x,
        tier0,
        tier1,
        topk_weights,
        topk_ids,
        global_to_combined,
        descriptor,
        broadcast_rotations,
        broadcast_launch,
        broadcast_buffers,
    ).clone()
    expanded = run_mixed_trellis(
        x,
        tier0,
        tier1,
        topk_weights,
        topk_ids,
        global_to_combined,
        descriptor,
        expanded_rotations,
        expanded_launch,
        expanded_buffers,
    ).clone()
    torch.cuda.synchronize(device)
    assert torch.equal(broadcast, expanded)
    relative = (broadcast - serial).norm() / serial.norm().clamp_min(1.0e-12)
    assert float(relative) < 4.0e-3

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
            broadcast_rotations,
            broadcast_launch,
            broadcast_buffers,
        )
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(captured, broadcast)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize(
    ("layouts", "max_m_blocks"),
    (
        (((206, 50), (160, 96)), 8),
        (((160, 96), (206, 50)), 9),
        (((80, 16), (96, 32)), 10),
    ),
    ids=("206-50_then_160-96", "160-96_then_206-50", "total-96_then_total-128"),
)
def test_mixed_runtime_partition_reuses_one_compiled_object_abba(
    layouts: tuple[tuple[int, int], tuple[int, int]], max_m_blocks: int
) -> None:
    """One compiled kernel must safely serve both production split families."""

    torch.manual_seed(20260803)
    device = torch.device("cuda", torch.cuda.current_device())
    m, hidden, intermediate, topk = 2, 256, 128, 2
    props = torch.cuda.get_device_properties(device)

    def prepare_partition(
        tier0_experts: int, tier1_experts: int, seed: int
    ) -> tuple[object, object]:
        return (
            _prepared(
                experts=tier0_experts,
                hidden=hidden,
                intermediate=intermediate,
                bits=3,
                seed=seed,
                device=device,
            ),
            _prepared(
                experts=tier1_experts,
                hidden=hidden,
                intermediate=intermediate,
                bits=4,
                seed=seed + 100,
                device=device,
            ),
        )

    def serial_partition(
        x: torch.Tensor,
        tiers: tuple[object, object],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        tier0_experts = int(tiers[0].num_experts)
        tier1_experts = int(tiers[1].num_experts)
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
        return _serial_tier(x, tiers[0], topk_weights, topk_ids, map0) + _serial_tier(
            x, tiers[1], topk_weights, topk_ids, map1
        )

    tiers_a = prepare_partition(*layouts[0], seed=301)
    tiers_b = prepare_partition(*layouts[1], seed=501)
    x = (torch.randn((m, hidden), device=device) * 1.0e-3).to(torch.bfloat16)
    topk_weights = torch.tensor(
        [[0.65, 0.35], [0.2, 0.8]], dtype=torch.float32, device=device
    )

    # Exercise the highest local id in both tiers. Stale compile-time bounds
    # would either skip these routes or read beyond the compact tier tensor.
    def boundary_routes(layout: tuple[int, int]) -> torch.Tensor:
        tier0_experts, tier1_experts = layout
        total_experts = tier0_experts + tier1_experts
        return torch.tensor(
            [[tier0_experts - 1, total_experts - 1], [0, tier0_experts]],
            dtype=torch.int32,
            device=device,
        )

    topk_ids_a = boundary_routes(layouts[0])
    topk_ids_b = boundary_routes(layouts[1])

    def compile_partition(
        tier0_experts: int, tier1_experts: int, *, sms: int | None = None
    ):
        return compile_mixed_trellis(
            size_m=m,
            hidden_size=hidden,
            intermediate_size=intermediate,
            tier0_num_experts=tier0_experts,
            tier1_num_experts=tier1_experts,
            top_k=topk,
            max_m_blocks=max_m_blocks,
            sms=int(props.multi_processor_count if sms is None else sms),
            max_shared_mem=int(props.shared_memory_per_block_optin),
            force_tile_config=(128, 128, 128, 128),
        )

    launch_a = compile_partition(*layouts[0])
    launch_b = compile_partition(*layouts[1])
    assert launch_a.compiled is launch_b.compiled
    assert (launch_a.tier0_num_experts, launch_a.tier1_num_experts) == layouts[0]
    assert (launch_b.tier0_num_experts, launch_b.tier1_num_experts) == layouts[1]
    alternate_sms = max(int(props.multi_processor_count) - 1, 1)
    launch_alt_sms = compile_partition(*layouts[1], sms=alternate_sms)
    launch_restored_sms = compile_partition(
        *layouts[0], sms=int(props.multi_processor_count)
    )
    assert launch_alt_sms.compiled is launch_a.compiled
    assert launch_restored_sms.compiled is launch_a.compiled
    assert launch_alt_sms.sms == alternate_sms
    assert launch_restored_sms.sms == int(props.multi_processor_count)

    def run_partition(
        tiers: tuple[object, object],
        topk_ids: torch.Tensor,
        launch,
        buffers,
        global_to_combined: torch.Tensor,
        descriptor: torch.Tensor,
        rotations,
    ) -> torch.Tensor:
        return run_mixed_trellis(
            x,
            tiers[0],
            tiers[1],
            topk_weights,
            topk_ids,
            global_to_combined,
            descriptor,
            rotations,
            launch,
            buffers,
        )

    maps_a = build_ordered_maps(*layouts[0], device=device)
    maps_b = build_ordered_maps(*layouts[1], device=device)
    rotations_a = combine_trellis_rotations(*tiers_a)
    rotations_b = combine_trellis_rotations(*tiers_b)
    buffers_a = make_mixed_trellis_buffers(
        launch_a, device=device, sms=int(props.multi_processor_count)
    )
    buffers_b = make_mixed_trellis_buffers(
        launch_b, device=device, sms=int(props.multi_processor_count)
    )
    serial_a = serial_partition(x, tiers_a, topk_weights, topk_ids_a)
    serial_b = serial_partition(x, tiers_b, topk_weights, topk_ids_b)

    output_a1 = run_partition(
        tiers_a, topk_ids_a, launch_a, buffers_a, *maps_a, rotations_a
    ).clone()
    output_b1 = run_partition(
        tiers_b, topk_ids_b, launch_b, buffers_b, *maps_b, rotations_b
    ).clone()
    output_b2 = run_partition(
        tiers_b, topk_ids_b, launch_b, buffers_b, *maps_b, rotations_b
    ).clone()
    output_a2 = run_partition(
        tiers_a, topk_ids_a, launch_a, buffers_a, *maps_a, rotations_a
    ).clone()
    torch.cuda.synchronize(device)

    for actual, expected in (
        (output_a1, serial_a),
        (output_b1, serial_b),
        (output_b2, serial_b),
        (output_a2, serial_a),
    ):
        assert not torch.isnan(actual).any()
        relative = (actual - expected).norm() / expected.norm().clamp_min(1.0e-12)
        assert float(relative) < 4.0e-3
    assert torch.equal(output_a1, output_a2)
    assert torch.equal(output_b1, output_b2)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_b = run_partition(
            tiers_b, topk_ids_b, launch_b, buffers_b, *maps_b, rotations_b
        )
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(captured_b, output_b1)


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
