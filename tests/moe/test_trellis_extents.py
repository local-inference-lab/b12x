"""Global coupled draw coordinates and canonical/BTX rank-extent parity."""

from dataclasses import replace
import hashlib
from types import SimpleNamespace

import pytest
import torch

from b12x.moe import fused_moe
from b12x.moe.fused_moe.trellis import (
    _coupled_input_scales,
    _coupled_rows,
    _validate_extent,
)
from tests.moe.test_trellis_config import _k3_config


def bundle(**metadata):
    scales = fused_moe.ScaleFactors(torch.ones(1))
    return fused_moe.TrellisWeights(
        atoms=torch.empty(4, 1, dtype=torch.uint8),
        rate=torch.tensor([0x33], dtype=torch.uint8),
        input_scales=scales,
        intermediate_scales=scales,
        output_scales=scales,
        **metadata,
    )


@pytest.mark.parametrize(
    "metadata,error",
    [
        ({"global_intermediate_size": True}, TypeError),
        ({"global_intermediate_size": 0}, ValueError),
        ({"global_intermediate_size": 511}, ValueError),
        ({"intermediate_offset": False}, TypeError),
        ({"intermediate_offset": -32}, ValueError),
        ({"intermediate_offset": 1}, ValueError),
        ({"intermediate_offset": 128}, ValueError),
    ],
)
def test_extent_metadata_rejects_ambiguous_coordinates(metadata, error):
    with pytest.raises(error):
        bundle(**metadata)


@pytest.mark.parametrize(
    "global_size,offset,width", [(512, 512, 128), (544, 0, 128), (512, 32, 128)]
)
def test_coupled_extent_checks_bounds_and_transform_blocks(global_size, offset, width):
    weights = bundle(global_intermediate_size=global_size, intermediate_offset=offset)
    with pytest.raises(ValueError):
        _validate_extent(
            fused_moe.TrellisConfig.from_dict(_k3_config()), weights, width
        )


def test_frozen_global_draw_bytes_and_extent_slices():
    from b12x.moe._shared.kernels.w4a16.btx import _coupled_rotation_rows

    draws = torch.arange(8, dtype=torch.uint8)
    whole = _coupled_rows(
        torch.zeros(8, 3 * 512, dtype=torch.float16),
        draws,
        intermediate_size=512,
        device=torch.device("cpu"),
        global_intermediate_size=512,
    )[:, 3 * 512 :]
    # Frozen encoder rotation_signs bytes: model revision and source hash are
    # recorded in docs/moe-execution-model.md. The residual draw is always zero.
    assert hashlib.sha256(whole.numpy().tobytes()).hexdigest() == (
        "1c8d4453d21b51eab567a73606fc4009e939b5d0c544731814498c82f0da09fa"
    )
    for offset in (0, 128, 256, 384):
        values = torch.zeros(8, 3 * 128, dtype=torch.float16)
        local = _coupled_rows(
            values,
            draws,
            intermediate_size=128,
            device=torch.device("cpu"),
            global_intermediate_size=512,
            intermediate_offset=offset,
        )
        expected = torch.cat(
            (
                whole[:, 2 * offset : 2 * (offset + 128)],
                whole[:, 1024 + offset : 1024 + offset + 128],
            ),
            dim=1,
        )
        torch.testing.assert_close(local[:, 384:], expected, atol=0, rtol=0)
        layer = SimpleNamespace(
            manifest=SimpleNamespace(
                hadamard=SimpleNamespace(pre_block=512, post_block=128),
                geometry=SimpleNamespace(
                    num_experts=8, intermediate_size=512, atom_channels=32
                ),
            ),
            local_intermediate_size=128,
            first_slot=offset // 32,
            rotation_draws=draws,
        )
        torch.testing.assert_close(
            local,
            _coupled_rotation_rows(layer, values, torch.device("cpu")),
            atol=0,
            rtol=0,
        )


@pytest.mark.parametrize("offset,which", [(0, 0), (128, 0), (256, 1), (384, 1)])
def test_coupled_input_scale_selection_uses_global_extent(offset, which):
    gate, up = torch.ones(1, 512), torch.full((1, 512), 2.0)
    weights = bundle(global_intermediate_size=512, intermediate_offset=offset)
    selected = _coupled_input_scales(weights, gate, up, 128)
    assert all(t.data_ptr() == (gate, up)[which].data_ptr() for t in selected)
    with pytest.raises(ValueError, match="explicit global extent"):
        _coupled_input_scales(bundle(), gate, up, 128)
    assert _coupled_input_scales(
        bundle(global_intermediate_size=512), gate, up, 512
    ) == (gate, up)
    assert _coupled_input_scales(
        bundle(global_intermediate_size=512), gate, gate, 512
    ) == (gate, gate)


@pytest.mark.parametrize("codebook", ["sqg_e4m3", "mcg"])
@pytest.mark.parametrize(
    "global_width,width,offset",
    [(1024, 256, 0), (1024, 256, 768), (384, 384, 0), (384, 256, 128)],
)
@pytest.mark.parametrize("per_expert", [False, True])
def test_canonical_coupled_extent_matches_btx(
    tmp_path, codebook, global_width, width, offset, per_expert
):
    if not torch.cuda.is_available():
        pytest.skip("canonical weight preparation requires CUDA")
    from b12x.moe._shared.kernels.w4a16.btx import (
        read_btx_layer,
        prepare_btx_moe_weights,
    )
    from b12x.moe._shared.kernels.w4a16.btx_synth import (
        BtxSynthConfig,
        write_btx_checkpoint,
    )
    from b12x.moe.fused_moe._sm103_trellis import _mixed_contract
    from tests._reference.trellis_reference import moe_reference

    experts, hidden = 8, 512
    config = BtxSynthConfig(
        codebook=codebook,
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=global_width,
        moe_layer_indices=(0,),
        bits=3,
        coupled=True,
        pre_block=512,
        post_block=128,
        per_expert_input_rotations=per_expert,
        extent_alignment_slots=4,
        extent_barriers=(16,) if global_width == 1024 else (),
        seed=39,
    )
    manifest = write_btx_checkpoint(tmp_path, config)
    layer = read_btx_layer(
        tmp_path, manifest, 0, first_slot=offset // 32, slot_count=width // 32
    )
    layer = replace(layer, rotation_draws=torch.arange(experts, dtype=torch.uint8))
    expected = prepare_btx_moe_weights(layer, activation="situ", device="cuda")

    declaration = _k3_config()
    declaration["codebook"] = codebook
    for name in declaration["scale"]:
        declaration["scale"][name] = dict(
            vectors="per_expert"
            if per_expert or name == "intermediate_scales"
            else "per_layer",
            gains="none",
        )
    plan = fused_moe.plan_weights(
        source=fused_moe.TrellisConfig.from_dict(declaration),
        activation=fused_moe.ActivationSpec(
            mode="a16", nonlinearity="situ", io_dtype=torch.bfloat16
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=experts, hidden_size=hidden, intermediate_size=width
        ),
    )
    weights = fused_moe.TrellisWeights(
        atoms=layer.atoms.cuda(),
        rate=torch.tensor([0x33], dtype=torch.uint8, device="cuda"),
        input_scales=fused_moe.ScaleFactors(
            torch.stack(
                (layer.gate_suh, layer.up_suh), dim=1 if per_expert else 0
            ).cuda()
        ),
        intermediate_scales=fused_moe.ScaleFactors(
            layer.rotations.permute(1, 2, 0, 3).reshape(experts, 3, width).cuda()
        ),
        output_scales=fused_moe.ScaleFactors(layer.down_svh.cuda()),
        expert_transform_draws=layer.rotation_draws.cuda(),
        global_intermediate_size=global_width,
        intermediate_offset=offset,
    )
    actual = fused_moe.prepare_weights(
        plan=plan, weights=weights
    )._impl.representation_for("w4a16")
    if codebook == "mcg":
        state, _, _ = _mixed_contract(
            SimpleNamespace(
                weight_E=experts, k=hidden, n=width, device=expected.w13.device
            ),
            actual,
        )
        payload = actual.tiers[0]
    else:
        state, payload = actual.trellis, actual
    crosses = 2 * offset < global_width < 2 * (offset + width)
    assert (state.gate_suh.data_ptr() == state.up_suh.data_ptr()) is not crosses
    assert (
        state.input_scale_split
        == expected.trellis.input_scale_split
        == (global_width // 2 - offset if crosses else None)
    )
    for name in ("w13", "w2"):
        torch.testing.assert_close(
            getattr(payload, name).reshape(-1),
            getattr(expected, name).reshape(-1),
            atol=0,
            rtol=0,
        )
    for name in ("gate_suh", "up_suh", "down_svh", "intermediate_rotations"):
        torch.testing.assert_close(
            getattr(state, name), getattr(expected.trellis, name), atol=0, rtol=0
        )
    torch.manual_seed(390)
    source = torch.randn(4, hidden, device="cuda", dtype=torch.bfloat16) * 0.01
    ids = torch.arange(experts, device="cuda").view(4, 2)
    routing = torch.rand(4, 2, device="cuda")
    oracle = moe_reference(source, expected, ids, routing, activation_kind="situ")
    assert torch.isfinite(oracle).all() and torch.count_nonzero(oracle)
    torch.testing.assert_close(
        moe_reference(source, actual, ids, routing, activation_kind="situ"),
        oracle,
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("codebook", ["sqg_e4m3", "mcg"])
def test_uniform_btx_partial_pair_extent_roundtrip(tmp_path, codebook):
    from b12x.moe._shared.kernels.w4a16.btx import read_btx_layer
    from b12x.moe._shared.kernels.w4a16.btx_synth import (
        BtxSynthConfig,
        write_btx_checkpoint,
    )

    config = BtxSynthConfig(
        codebook=codebook,
        num_experts=2,
        hidden_size=512,
        intermediate_size=384,
        moe_layer_indices=(0,),
        bits=3,
        coupled=True,
        pre_block=512,
        post_block=128,
    )
    manifest = write_btx_checkpoint(tmp_path, config)
    whole = read_btx_layer(tmp_path, manifest, 0, first_slot=0, slot_count=12)
    tail = read_btx_layer(tmp_path, manifest, 0, first_slot=4, slot_count=8)
    assert whole.local_intermediate_size == 384 and tail.local_intermediate_size == 256
    torch.testing.assert_close(tail.atoms, whole.atoms[4:], atol=0, rtol=0)
    torch.testing.assert_close(tail.rotations, whole.rotations[4:], atol=0, rtol=0)
