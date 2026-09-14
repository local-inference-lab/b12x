"""BTX preparation equivalence and fail-closed reader tests.

Independent-assembly checks reconstruct the runtime word order with
naive loops so the shared restore helper is never compared against
itself; frozen-container equivalence lives in test_btx_compat.
"""

from __future__ import annotations

import json

import pytest
import torch

from b12x.moe._shared.btx_schema import (
    BTX_MANIFEST_FILENAME,
    matrix_atom_bytes,
    rate_code,
)
from b12x.moe._shared.kernels.w4a16.btx import (
    prepare_btx_moe_weights,
    read_btx_layer,
    read_btx_manifest,
)
from b12x.moe._shared.kernels.w4a16.btx_synth import (
    BtxSynthConfig,
    synth_layer_payloads,
    write_btx_checkpoint,
)
from b12x.moe._shared.kernels.w4a16.prepare import (
    prepare_trellis256_moe_weights,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _plane_pair(payloads, expert: int, slot: int, matrix: int):
    return payloads.planes[(expert, slot, matrix)]


def _naive_fc1_words(low_planes, high_planes) -> torch.Tensor:
    """FC1 runtime order via explicit loops: per K16 tile, all low-plane
    windows atom-major, then all high-plane windows."""

    hidden_tiles = low_planes[0].shape[0]
    rows = []
    for kt in range(hidden_tiles):
        low = torch.cat([plane[kt] for plane in low_planes])
        high = torch.cat([plane[kt] for plane in high_planes])
        rows.append(torch.cat((low, high)))
    return torch.cat(rows)


def _naive_fc2_words(low_planes, high_planes) -> torch.Tensor:
    """FC2 runtime order: the K-major low planes, then the high planes."""

    low = torch.cat([plane.reshape(-1) for plane in low_planes])
    high = torch.cat([plane.reshape(-1) for plane in high_planes])
    return torch.cat((low, high))


@requires_cuda
@pytest.mark.parametrize("global_i,fc1_tile_n", ((512, 256), (1152, 128)))
def test_btx_uniform_mcg_matches_direct_binder(
    tmp_path, global_i: int, fc1_tile_n: int
) -> None:
    hidden, experts, bits = 256, 3, 3
    slots = global_i // 32
    config = BtxSynthConfig(
        codebook="mcg",
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=global_i,
        moe_layer_indices=(0,),
        bits=bits,
        per_expert_input_rotations=True,
        extent_alignment_slots=4,
        seed=5,
    )
    manifest = write_btx_checkpoint(tmp_path, config)
    layer = read_btx_layer(
        tmp_path, manifest, 0, first_slot=0, slot_count=slots
    )
    device = _device()
    btx_prepared = prepare_btx_moe_weights(
        layer, activation="silu", device=device
    )

    payloads = synth_layer_payloads(config, 0)
    hidden_tiles = hidden // 16
    w13 = torch.empty(
        (2, experts, hidden_tiles, 2 * slots, 16 * bits), dtype=torch.int16
    )
    w2 = torch.empty(
        (experts, 2 * slots, hidden_tiles, 16 * bits), dtype=torch.int16
    )
    for expert in range(experts):
        for slot in range(slots):
            for matrix in range(2):
                low, high = _plane_pair(payloads, expert, slot, matrix)
                w13[matrix, expert, :, 2 * slot] = low
                w13[matrix, expert, :, 2 * slot + 1] = high
            low, high = _plane_pair(payloads, expert, slot, 2)
            w2[expert, 2 * slot] = low
            w2[expert, 2 * slot + 1] = high

    rotations = torch.cat(
        [
            payloads.rotations[:, :, matrix, :]
            .permute(1, 0, 2)
            .reshape(experts, global_i)
            for matrix in range(3)
        ],
        dim=1,
    ).to(device)
    binder_prepared = prepare_trellis256_moe_weights(
        w13.to(device),
        w2.to(device),
        hidden_size=hidden,
        intermediate_size=global_i,
        num_experts=experts,
        activation="silu",
        fc1_tile_n=fc1_tile_n,
        fc2_tile_n=fc1_tile_n,
        params_dtype=torch.float16,
        w13_layout="trellis_t256_proj",
        trellis_bits=bits,
        codebook="mcg",
        gate_suh=payloads.gate_suh.to(device),
        up_suh=payloads.up_suh.to(device),
        intermediate_rotations=rotations,
        down_svh=payloads.down_svh.to(device),
        tile_config=(64, fc1_tile_n, 64, fc1_tile_n),
    )

    assert torch.equal(btx_prepared.w13, binder_prepared.w13)
    assert torch.equal(btx_prepared.w2, binder_prepared.w2)
    assert torch.equal(
        btx_prepared.intermediate_rotations,
        binder_prepared.intermediate_rotations,
    )
    assert btx_prepared.trellis_codebook == "mcg"
    assert btx_prepared.source_format == "btx"


def _per_expert_config(kinds: dict[int, tuple[int, int]]) -> BtxSynthConfig:
    experts = len(kinds)
    fc1 = torch.tensor(
        [[rate_code(*kinds[e]) for e in range(experts)]], dtype=torch.uint8
    )
    fc2 = torch.full_like(fc1, rate_code(3, 3))
    return BtxSynthConfig(
        codebook="sqg_e4m3",
        num_experts=experts,
        hidden_size=256,
        intermediate_size=256,
        moe_layer_indices=(0,),
        bits=None,
        rate_tables={0: (fc1, fc2)},
        extent_alignment_slots=8,
        seed=3,
    )


@requires_cuda
@pytest.mark.parametrize(
    "high_rates, expected_kind",
    [((2, 4), "PDYNAMIC"), ((4, 3), "P33_P43")],
)
def test_btx_per_expert_pair_matches_naive_assembly(
    tmp_path, high_rates, expected_kind
) -> None:
    if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        pytest.skip("SM12x coalesced pair preparation; SM103 uses compressed atom rows")
    kinds = {0: (3, 3), 1: high_rates, 2: (3, 3)}
    config = _per_expert_config(kinds)
    manifest = write_btx_checkpoint(tmp_path, config)
    layer = read_btx_layer(tmp_path, manifest, 0, first_slot=0, slot_count=8)
    device = _device()
    prepared = prepare_btx_moe_weights(
        layer, activation="situ", device=device
    )
    assert prepared.fc1_trellis_pair_kind == expected_kind
    assert prepared.fc2_trellis_pair_kind == expected_kind

    payloads = synth_layer_payloads(config, 0)
    experts = config.num_experts

    def _matrix_planes(expert: int, matrix: int):
        lows, highs = [], []
        for slot in range(8):
            low, high = _plane_pair(payloads, expert, slot, matrix)
            lows.append(low)
            highs.append(high)
        return lows, highs

    expected_gate = [
        _naive_fc1_words(*_matrix_planes(e, 0)) for e in range(experts)
    ]
    expected_up = [
        _naive_fc1_words(*_matrix_planes(e, 1)) for e in range(experts)
    ]
    expected_down = [
        _naive_fc2_words(*_matrix_planes(e, 2)) for e in range(experts)
    ]
    expected_w13 = torch.cat(expected_gate + expected_up).to(device)
    expected_w2 = torch.cat(expected_down).to(device)
    assert torch.equal(prepared.w13.view(torch.int16), expected_w13)
    assert torch.equal(prepared.w2.view(torch.int16), expected_w2)

    # Rotation rows follow the pair runtime's record-major channel order:
    # each atom's first 16 channels belong to the low record, its last 16
    # to the high record.
    expected_rotations = []
    for matrix in range(3):
        planes = payloads.rotations[:, :, matrix, :].reshape(8, experts, 2, 16)
        low = planes[:, :, 0, :].permute(1, 0, 2).reshape(experts, -1)
        high = planes[:, :, 1, :].permute(1, 0, 2).reshape(experts, -1)
        expected_rotations.append(torch.cat((low, high), dim=1))
    assert torch.equal(
        prepared.intermediate_rotations,
        torch.cat(expected_rotations, dim=1).to(device),
    )

    modes = prepared.fc1_trellis_pair_modes
    assert modes is not None
    if expected_kind == "PDYNAMIC":
        assert modes.tolist() == [0, 1, 0]
        fc2_modes = prepared.fc2_trellis_pair_modes
        assert fc2_modes is not None and fc2_modes.tolist() == [0, 0, 0]
    else:
        # Descriptors: gap-free u32 offsets over the expert sections with
        # bit zero selecting the high-rate record pair.
        lengths = [section.numel() // 2 for section in expected_gate]
        offsets = [0, lengths[0], lengths[0] + lengths[1]]
        assert modes.tolist() == [
            (offsets[0] << 1) | 0,
            (offsets[1] << 1) | 1,
            (offsets[2] << 1) | 0,
        ]


def test_btx_reader_fails_closed(tmp_path) -> None:
    config = BtxSynthConfig(
        codebook="sqg_e4m3",
        num_experts=2,
        hidden_size=128,
        intermediate_size=256,
        moe_layer_indices=(0,),
        bits=2,
        extent_alignment_slots=4,
        seed=1,
    )
    manifest = write_btx_checkpoint(tmp_path, config)
    read_btx_layer(tmp_path, manifest, 0, first_slot=0, slot_count=8)

    with pytest.raises(ValueError, match="align"):
        read_btx_layer(tmp_path, manifest, 0, first_slot=1, slot_count=4)
    with pytest.raises(ValueError, match="does not declare layer"):
        read_btx_layer(tmp_path, manifest, 7, first_slot=0, slot_count=4)

    # Manifest/metadata disagreement: tamper the manifest codebook (the
    # tampered value must still parse) and require the metadata cross-check
    # to reject the layer.
    data = json.loads((tmp_path / BTX_MANIFEST_FILENAME).read_text())
    data["codebook"] = "mcg"
    data["codebook_seed"] = 0xCBAC1FED
    data["rates"]["bits"] = 3
    from b12x.moe._shared.btx_schema import BtxManifest

    tampered = BtxManifest.from_dict(data)
    with pytest.raises(ValueError, match="metadata 'codebook'"):
        read_btx_layer(tmp_path, tampered, 0, first_slot=0, slot_count=4)

    # sha verification.
    ref = manifest.layers[0]
    path = tmp_path / ref.file
    blob = bytearray(path.read_bytes())
    blob[-1] ^= 0xFF
    path.write_bytes(bytes(blob))
    with pytest.raises(ValueError, match="sha256 mismatch"):
        read_btx_layer(
            tmp_path, manifest, 0, first_slot=0, slot_count=4, verify_sha=True
        )


@requires_cuda
def test_btx_uniform_padding_must_be_zero(tmp_path) -> None:
    config = BtxSynthConfig(
        codebook="sqg_e4m3",
        num_experts=2,
        hidden_size=128,
        intermediate_size=256,
        moe_layer_indices=(0,),
        bits=2,
        extent_alignment_slots=4,
        seed=2,
    )
    manifest = write_btx_checkpoint(tmp_path, config)
    layer = read_btx_layer(tmp_path, manifest, 0, first_slot=0, slot_count=8)
    corrupted = layer.atoms.clone()
    corrupted[0, -1] = 1
    from dataclasses import replace

    with pytest.raises(ValueError, match="padding must be zero"):
        prepare_btx_moe_weights(
            replace(layer, atoms=corrupted),
            activation="situ",
            device=_device(),
        )
