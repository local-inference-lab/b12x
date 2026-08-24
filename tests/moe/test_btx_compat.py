"""Equivalence of lifted frozen QSRT containers with the BTX load path.

Each test packs one rank extent in a frozen container layout, lifts it to
an in-memory BTX extent, and requires byte-identical prepared tensors
against a synthetically written BTX checkpoint carrying the same plane
words. Byte equality against the frozen containers' dedicated readers was
established when the lift landed beside them; those readers are removed.
"""

from __future__ import annotations

import pytest
import torch

from b12x.moe._shared.btx_schema import matrix_atom_bytes, rate_code
from b12x.moe._shared.kernels.w4a16.btx import prepare_btx_moe_weights
from b12x.moe._shared.kernels.w4a16.btx_compat import (
    lift_qsrt_atoms_v1_extent,
    lift_qsrt_atoms_v2_extent,
    lift_qsrt_atoms_v3_extent,
)
from b12x.moe._shared.kernels.w4a16.btx_synth import (
    BtxSynthConfig,
    synth_layer_payloads,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _rand_plane(generator, hidden_tiles, bits):
    return torch.randint(
        -(1 << 15),
        1 << 15,
        (hidden_tiles, 16 * bits),
        dtype=torch.int16,
        generator=generator,
    )


def _span(generator):
    raw = torch.rand((32,), generator=generator, dtype=torch.float32)
    return (0.5 + raw).to(torch.float16)


@requires_cuda
def test_v1_lift_matches_synth_btx(tmp_path) -> None:
    from b12x.moe._shared.kernels.w4a16.btx import read_btx_layer
    from b12x.moe._shared.kernels.w4a16.btx_synth import write_btx_checkpoint

    hidden, experts, layer_index, first_atom_slot = 256, 3, 1, 0
    expert_ids = torch.tensor([3, 7, 11], dtype=torch.int32)
    format_codes = torch.tensor([0x21, 0x02, 0x11], dtype=torch.uint8)
    physical_pair = first_atom_slot // 8
    rotation = (5 * expert_ids.to(torch.int64) + layer_index) % 12
    logical_pair = (physical_pair - rotation) % 12
    fc1_codes = torch.where(
        logical_pair < (format_codes.to(torch.int64) >> 4),
        rate_code(2, 4),
        rate_code(3, 3),
    ).to(torch.uint8)
    fc2_codes = torch.where(
        logical_pair < (format_codes.to(torch.int64) & 0xF),
        rate_code(2, 4),
        rate_code(3, 3),
    ).to(torch.uint8)

    config = BtxSynthConfig(
        codebook="sqg_e4m3",
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=256,
        moe_layer_indices=(layer_index,),
        bits=None,
        rate_tables={
            layer_index: (
                fc1_codes.reshape(1, -1).clone(),
                fc2_codes.reshape(1, -1).clone(),
            )
        },
        extent_alignment_slots=8,
        seed=41,
    )
    manifest = write_btx_checkpoint(tmp_path, config)
    btx_layer = read_btx_layer(
        tmp_path, manifest, layer_index, first_slot=0, slot_count=8
    )
    device = _device()
    from_synth = prepare_btx_moe_weights(
        btx_layer, activation="situ", device=device
    )

    payloads = synth_layer_payloads(config, layer_index)
    matrix_bytes = matrix_atom_bytes(hidden, 3, 3)
    bundle = 3 * matrix_bytes + 3 * 64
    payload = torch.zeros((8, experts, bundle), dtype=torch.uint8)
    for slot in range(8):
        for expert in range(experts):
            cursor = 0
            for matrix in range(3):
                low, high = payloads.planes[(expert, slot, matrix)]
                for plane in (low, high):
                    raw = plane.contiguous().view(torch.uint8).reshape(-1)
                    payload[slot, expert, cursor : cursor + raw.numel()] = raw
                    cursor += raw.numel()
            for matrix in range(3):
                payload[slot, expert, cursor : cursor + 64] = (
                    payloads.rotations[slot, expert, matrix]
                    .contiguous()
                    .view(torch.uint8)
                )
                cursor += 64

    lifted_layer = lift_qsrt_atoms_v1_extent(
        payload,
        first_atom_slot=first_atom_slot,
        layer_index=layer_index,
        expert_ids=expert_ids,
        format_codes=format_codes,
        hidden_size=hidden,
        global_intermediate_size=256,
        gate_suh=payloads.gate_suh,
        up_suh=payloads.up_suh,
        down_svh=payloads.down_svh,
    )
    lifted = prepare_btx_moe_weights(
        lifted_layer, activation="situ", device=device
    )

    assert torch.equal(lifted.w13, from_synth.w13)
    assert torch.equal(lifted.w2, from_synth.w2)
    assert torch.equal(
        lifted.intermediate_rotations, from_synth.intermediate_rotations
    )
    assert lifted.fc1_trellis_pair_kind == from_synth.fc1_trellis_pair_kind
    assert torch.equal(
        lifted.fc1_trellis_pair_modes, from_synth.fc1_trellis_pair_modes
    )
    assert torch.equal(
        lifted.fc2_trellis_pair_modes, from_synth.fc2_trellis_pair_modes
    )


@requires_cuda
@pytest.mark.parametrize(
    ("bits", "profile", "seed", "per_expert_input_rotations"),
    (
        (2, "k2_coupled_h512_h128", 21, False),
        (3, "k3_coupled_h512_h128", 22, False),
        (3, "k3_coupled_h512_h128", 23, True),
    ),
)
def test_v2_pure_uniform_lift_matches_synth_btx(
    tmp_path,
    bits: int,
    profile: str,
    seed: int,
    per_expert_input_rotations: bool,
) -> None:
    from b12x.moe._shared.kernels.w4a16.btx import read_btx_layer
    from b12x.moe._shared.kernels.w4a16.btx_synth import write_btx_checkpoint

    hidden, global_i, experts, slots = 512, 512, 2, 8
    config = BtxSynthConfig(
        codebook="sqg_e4m3",
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=global_i,
        moe_layer_indices=(1,),
        bits=bits,
        coupled=True,
        pre_block=512,
        post_block=128,
        per_expert_input_rotations=per_expert_input_rotations,
        extent_alignment_slots=4,
        extent_barriers=(8,),
        seed=seed,
    )
    manifest = write_btx_checkpoint(tmp_path, config)
    btx_layer = read_btx_layer(
        tmp_path, manifest, 1, first_slot=0, slot_count=slots
    )
    device = _device()
    from_synth = prepare_btx_moe_weights(
        btx_layer,
        activation="situ",
        device=device,
        tile_config=(128, 128, 128, 128),
    )

    payloads = synth_layer_payloads(config, 1)
    section = matrix_atom_bytes(hidden, bits, bits)
    bundle = 3 * section + 3 * 64
    payload = torch.zeros((slots, experts * bundle), dtype=torch.uint8)
    for slot in range(slots):
        cursor = 0
        for expert in range(experts):
            for matrix in range(3):
                low, high = payloads.planes[(expert, slot, matrix)]
                for plane in (low, high):
                    raw = plane.contiguous().view(torch.uint8).reshape(-1)
                    payload[slot, cursor : cursor + raw.numel()] = raw
                    cursor += raw.numel()
            for matrix in range(3):
                payload[slot, cursor : cursor + 64] = (
                    payloads.rotations[slot, expert, matrix]
                    .contiguous()
                    .view(torch.uint8)
                )
                cursor += 64

    lifted_layer = lift_qsrt_atoms_v2_extent(
        payload,
        profile=profile,
        first_atom_slot=0,
        layer_index=1,
        hidden_size=hidden,
        global_intermediate_size=global_i,
        num_experts=experts,
        gate_suh=payloads.gate_suh,
        up_suh=payloads.up_suh,
        down_svh=payloads.down_svh,
        rotation_draws=payloads.rotation_draws,
    )
    assert (
        lifted_layer.manifest.hadamard.per_expert_input_rotations
        is per_expert_input_rotations
    )
    lifted = prepare_btx_moe_weights(
        lifted_layer,
        activation="situ",
        device=device,
        tile_config=(128, 128, 128, 128),
    )

    assert torch.equal(lifted.w13, from_synth.w13)
    assert torch.equal(lifted.w2, from_synth.w2)
    assert torch.equal(
        lifted.intermediate_rotations, from_synth.intermediate_rotations
    )
    assert torch.equal(lifted.gate_suh, from_synth.gate_suh)
    assert lifted.coupled_hadamard and from_synth.coupled_hadamard


@pytest.mark.parametrize("first_slot", (4, 12))
def test_v3_compact_scales_expand_only_the_local_extent(first_slot: int) -> None:
    hidden, global_i, experts = 512, 512, 2
    slots = 4
    section = matrix_atom_bytes(hidden, 3, 3)
    payload = torch.zeros(
        (slots, experts * 3 * section), dtype=torch.uint8
    )

    def signs(length: int, offset: int) -> torch.Tensor:
        values = torch.ones(length, dtype=torch.float16)
        values[offset::2] = -1
        return values

    hidden_signs = [signs(hidden, offset) for offset in (0, 1, 0)]
    intermediate_signs = [signs(global_i, offset) for offset in (0, 1, 0)]
    layer_magnitudes = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float16)
    expert_magnitudes = torch.tensor(
        [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], dtype=torch.float16
    )
    lifted = lift_qsrt_atoms_v3_extent(
        payload,
        first_atom_slot=first_slot,
        layer_index=44,
        hidden_size=hidden,
        global_intermediate_size=global_i,
        num_experts=experts,
        gate_suh_signs=hidden_signs[0],
        up_suh_signs=hidden_signs[1],
        down_svh_signs=hidden_signs[2],
        gate_svh_signs=intermediate_signs[0],
        up_svh_signs=intermediate_signs[1],
        down_suh_signs=intermediate_signs[2],
        layer_scale_magnitudes=layer_magnitudes,
        expert_scale_magnitudes=expert_magnitudes,
        rotation_draws=torch.zeros(experts, dtype=torch.uint8),
    )

    def scale_atoms(values: torch.Tensor) -> torch.Tensor:
        physical_to_logical = torch.tensor([0, 3, 1, 2])
        logical_to_physical = torch.argsort(physical_to_logical)
        return (
            values.reshape(4, 128)
            .index_select(0, logical_to_physical)
            .reshape(16, 32)
        )

    gate_atoms, up_atoms, down_atoms = [
        scale_atoms(values) for values in intermediate_signs
    ]
    upstream = torch.cat((gate_atoms, up_atoms), dim=0)
    indices = torch.arange(first_slot, first_slot + slots)
    expected_signs = torch.stack(
        (upstream[2 * indices], upstream[2 * indices + 1], down_atoms[indices]),
        dim=1,
    )
    upstream_magnitude = expert_magnitudes[0 if first_slot < 8 else 1]
    expected_magnitudes = torch.stack(
        (upstream_magnitude, upstream_magnitude, expert_magnitudes[2]), dim=1
    )
    expected = expected_magnitudes[None, :, :, None] * expected_signs[:, None]
    assert torch.equal(lifted.rotations, expected)
    assert torch.equal(lifted.gate_suh, hidden_signs[0] * 2)
    assert torch.equal(lifted.up_suh, hidden_signs[1] * 3)
    assert torch.equal(lifted.down_svh, hidden_signs[2] * 4)
    assert lifted.atoms.shape == payload.shape
    assert lifted.first_slot == first_slot
    assert lifted.slot_count == slots
    assert lifted.manifest.rates.bits == 3
    assert not lifted.manifest.hadamard.per_expert_input_rotations


@requires_cuda
def test_v3_compact_lift_prepares_identically_to_embedded_v2() -> None:
    hidden, global_i, experts, slots = 512, 512, 2, 8
    generator = torch.Generator().manual_seed(71)
    section = matrix_atom_bytes(hidden, 3, 3)
    words = torch.randint(
        0,
        256,
        (slots, experts, 3 * section),
        dtype=torch.uint8,
        generator=generator,
    )

    def signs(length: int, offset: int) -> torch.Tensor:
        values = torch.ones(length, dtype=torch.float16)
        values[offset::2] = -1
        return values

    hidden_signs = [signs(hidden, offset) for offset in (0, 1, 0)]
    intermediate_signs = [signs(global_i, offset) for offset in (0, 1, 0)]
    layer_magnitudes = torch.tensor([0.5, 0.75, 1.25], dtype=torch.float16)
    expert_magnitudes = torch.tensor(
        [[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]], dtype=torch.float16
    )
    side_tables = [
        hidden_signs[index] * layer_magnitudes[index] for index in range(3)
    ]
    local_channels = slots * 32
    def scale_atoms(values: torch.Tensor) -> torch.Tensor:
        physical_to_logical = torch.tensor([0, 3, 1, 2])
        logical_to_physical = torch.argsort(physical_to_logical)
        return (
            values.reshape(4, 128)
            .index_select(0, logical_to_physical)
            .reshape(16, 32)
        )

    gate_atoms, up_atoms, down_atoms = [
        scale_atoms(values) for values in intermediate_signs
    ]
    upstream = torch.cat((gate_atoms, up_atoms), dim=0)
    indices = torch.arange(slots)
    rotation_signs = torch.stack(
        (upstream[2 * indices], upstream[2 * indices + 1], down_atoms[indices]),
        dim=1,
    )
    rotation_magnitudes = torch.stack(
        (expert_magnitudes[0], expert_magnitudes[0], expert_magnitudes[2]),
        dim=1,
    )
    rotations = (
        rotation_magnitudes[None, :, :, None] * rotation_signs[:, None]
    ).contiguous()
    embedded = torch.cat(
        (words, rotations.view(torch.uint8).reshape(slots, experts, 3 * 64)),
        dim=2,
    ).reshape(slots, -1)
    draws = torch.zeros(experts, dtype=torch.uint8)

    v2 = lift_qsrt_atoms_v2_extent(
        embedded,
        profile="k3_coupled_h512_h128",
        first_atom_slot=0,
        layer_index=44,
        hidden_size=hidden,
        global_intermediate_size=global_i,
        num_experts=experts,
        gate_suh=side_tables[0],
        up_suh=side_tables[1],
        down_svh=side_tables[2],
        rotation_draws=draws,
    )
    v3 = lift_qsrt_atoms_v3_extent(
        words.reshape(slots, -1),
        first_atom_slot=0,
        layer_index=44,
        hidden_size=hidden,
        global_intermediate_size=global_i,
        num_experts=experts,
        gate_suh_signs=hidden_signs[0],
        up_suh_signs=hidden_signs[1],
        down_svh_signs=hidden_signs[2],
        gate_svh_signs=intermediate_signs[0],
        up_svh_signs=intermediate_signs[1],
        down_suh_signs=intermediate_signs[2],
        layer_scale_magnitudes=layer_magnitudes,
        expert_scale_magnitudes=expert_magnitudes,
        rotation_draws=draws,
    )
    from_v2 = prepare_btx_moe_weights(
        v2,
        activation="situ",
        device=_device(),
        tile_config=(128, 128, 128, 128),
    )
    from_v3 = prepare_btx_moe_weights(
        v3,
        activation="situ",
        device=_device(),
        tile_config=(128, 128, 128, 128),
    )
    assert torch.equal(from_v3.w13, from_v2.w13)
    assert torch.equal(from_v3.w2, from_v2.w2)
    assert torch.equal(from_v3.gate_suh, from_v2.gate_suh)
    assert torch.equal(from_v3.up_suh, from_v2.up_suh)
    assert torch.equal(from_v3.down_svh, from_v2.down_svh)
    assert torch.equal(
        from_v3.intermediate_rotations, from_v2.intermediate_rotations
    )


@requires_cuda
def test_v2_fixed_high_rate_lift_matches_synth_btx(tmp_path) -> None:
    from b12x.moe._shared.kernels.w4a16.btx import read_btx_layer
    from b12x.moe._shared.kernels.w4a16.btx_synth import write_btx_checkpoint

    hidden, experts, layer_index = 256, 3, 1
    hidden_tiles = hidden // 16
    first_atom_slot = 0
    physical_pair = 0
    expert_ids = torch.arange(experts, dtype=torch.int64)
    rotation = (5 * expert_ids + layer_index) % 12
    base_pair = (physical_pair - rotation) % 12
    modes = (base_pair == 0) | (base_pair == 6)
    fc_codes = torch.where(modes, rate_code(4, 3), rate_code(3, 3)).to(
        torch.uint8
    )
    config = BtxSynthConfig(
        codebook="sqg_e4m3",
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=256,
        moe_layer_indices=(layer_index,),
        bits=None,
        rate_tables={
            layer_index: (
                fc_codes.reshape(1, -1).clone(),
                fc_codes.reshape(1, -1).clone(),
            )
        },
        extent_alignment_slots=8,
        seed=33,
    )
    manifest = write_btx_checkpoint(tmp_path, config)
    btx_layer = read_btx_layer(
        tmp_path, manifest, layer_index, first_slot=0, slot_count=8
    )
    device = _device()
    from_synth = prepare_btx_moe_weights(
        btx_layer, activation="situ", device=device
    )

    # Pack the identical planes as a grouped-by-rate-class atoms-v2 row set.
    payloads = synth_layer_payloads(config, layer_index)
    p33_ids = torch.nonzero(~modes, as_tuple=False).flatten().tolist()
    p43_ids = torch.nonzero(modes, as_tuple=False).flatten().tolist()
    rows = []
    for slot in range(8):
        chunks = []
        for expert in p33_ids + p43_ids:
            for matrix in range(3):
                low, high = payloads.planes[(expert, slot, matrix)]
                chunks.append(low.contiguous().view(torch.uint8).reshape(-1))
                chunks.append(high.contiguous().view(torch.uint8).reshape(-1))
            for matrix in range(3):
                chunks.append(
                    payloads.rotations[slot, expert, matrix]
                    .contiguous()
                    .view(torch.uint8)
                )
        rows.append(torch.cat(chunks))
    payload = torch.stack(rows)

    lifted_layer = lift_qsrt_atoms_v2_extent(
        payload,
        profile="k3x22_k4x2",
        first_atom_slot=first_atom_slot,
        layer_index=layer_index,
        hidden_size=hidden,
        global_intermediate_size=256,
        num_experts=experts,
        gate_suh=payloads.gate_suh,
        up_suh=payloads.up_suh,
        down_svh=payloads.down_svh,
    )
    lifted = prepare_btx_moe_weights(
        lifted_layer, activation="situ", device=device
    )

    assert torch.equal(lifted.w13, from_synth.w13)
    assert torch.equal(lifted.w2, from_synth.w2)
    assert torch.equal(
        lifted.intermediate_rotations, from_synth.intermediate_rotations
    )
    assert lifted.fc1_trellis_pair_kind == "P33_P43"
    modes_tensor = lifted.fc1_trellis_pair_modes
    assert modes_tensor is not None
    assert torch.equal(
        (modes_tensor & 1).to(torch.bool).cpu(), modes
    )


def test_coupled_high_rate_profile_has_no_lift() -> None:
    with pytest.raises(ValueError, match="no BTX lift"):
        lift_qsrt_atoms_v2_extent(
            torch.zeros((8, 16), dtype=torch.uint8),
            profile="k3x22_k4x2_coupled_h512_h128",
            first_atom_slot=0,
            layer_index=1,
            hidden_size=3584,
            global_intermediate_size=3072,
            num_experts=896,
            gate_suh=torch.ones(3584, dtype=torch.float16),
            up_suh=torch.ones(3584, dtype=torch.float16),
            down_svh=torch.ones(3584, dtype=torch.float16),
        )
