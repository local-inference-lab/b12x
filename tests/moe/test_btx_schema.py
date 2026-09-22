"""Fail-closed validation tests for the btx-atoms-v1 container schema."""

from __future__ import annotations

import copy
import json

import pytest
import torch

from b12x.moe._shared.btx_schema import (
    ATOM_CHANNELS,
    BTX_MANIFEST_FILENAME,
    BtxManifest,
    bundle_bytes,
    matrix_atom_bytes,
    rate_code,
)
from b12x.moe._shared.kernels.w4a16.btx_synth import (
    BtxSynthConfig,
    assemble_atoms_rows,
    synth_layer_payloads,
    write_btx_checkpoint,
)


def _uniform_manifest_dict(
    *,
    codebook: str = "sqg_e4m3",
    bits: int = 2,
    coupled: bool = False,
) -> dict:
    hadamard: dict = {
        "coupled": coupled,
        "per_expert_input_rotations": False,
    }
    if coupled:
        hadamard["pre_block"] = 512
        hadamard["post_block"] = 128
    data: dict = {
        "kind": "btx-manifest",
        "schema": "btx-atoms-v1",
        "codebook": codebook,
        "geometry": {
            "num_experts": 8,
            "hidden_size": 512,
            "intermediate_size": 512,
            "atom_channels": 32,
            "atom_slots": 16,
            "moe_layer_indices": [1, 2],
        },
        "rates": {"structure": "uniform", "bits": bits},
        "hadamard": hadamard,
        "layout": {
            "atom_row_alignment": 4096,
            "extent_alignment_slots": 4,
            "extent_barriers": [8],
        },
        "layers": {
            "1": {"file": "btx-layer-00001.safetensors", "sha256": "0" * 64},
            "2": {"file": "btx-layer-00002.safetensors", "sha256": "0" * 64},
        },
    }
    if codebook == "mcg":
        data["codebook_seed"] = 0xCBAC1FED
    return data


def _per_expert_manifest_dict() -> dict:
    data = _uniform_manifest_dict(bits=3)
    data["rates"] = {
        "structure": "per_expert_pair",
        "pair_kinds": ["P33", "P43"],
    }
    return data


def test_uniform_manifest_parses() -> None:
    manifest = BtxManifest.from_dict(_uniform_manifest_dict())
    assert manifest.codebook == "sqg_e4m3"
    assert manifest.rates.uniform_code() == rate_code(2, 2)
    assert manifest.geometry.atom_slots == 16
    assert manifest.layout.extent_barriers == (8,)


@pytest.mark.parametrize("bits", [2, 3])
def test_canonical_weight_plan_preserves_btx_transform_and_rate(bits: int) -> None:
    from b12x.moe import fused_moe

    manifest = BtxManifest.from_dict(_uniform_manifest_dict(bits=bits, coupled=True))
    plan = fused_moe.plan_weights(
        source=fused_moe.BtxSource(manifest=manifest),
        activation=fused_moe.ActivationSpec(
            mode="a16", nonlinearity="silu", io_dtype=torch.bfloat16
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=8, hidden_size=512, intermediate_size=256
        ),
    )
    assert plan.prepared_format.weights is fused_moe.WeightEncoding.TRELLIS
    assert plan.prepared_format.packing is fused_moe.WeightPacking.TRELLIS_NATIVE
    assert plan._impl.trellis_bits == bits
    assert plan._impl.coupled_hadamard_blocks == (512, 128)
    assert plan.geometry.intermediate_size == 256
    assert plan.source.manifest.geometry.intermediate_size == 512


@pytest.mark.parametrize(
    ("mode", "experts", "intermediate", "message"),
    [
        ("a8", 8, 256, "A16 activations"),
        ("a16", 9, 256, "must match the manifest"),
        ("a16", 8, 1024, "exceeds the source width"),
    ],
)
def test_canonical_btx_plan_rejects_incompatible_contract(
    mode: str, experts: int, intermediate: int, message: str
) -> None:
    from b12x.moe import fused_moe

    manifest = BtxManifest.from_dict(_uniform_manifest_dict(coupled=True))
    with pytest.raises(ValueError, match=message):
        fused_moe.plan_weights(
            source=fused_moe.BtxSource(manifest=manifest),
            activation=fused_moe.ActivationSpec(
                mode=mode, nonlinearity="silu", io_dtype=torch.bfloat16
            ),
            geometry=fused_moe.MoEGeometry(
                num_experts=experts, hidden_size=512, intermediate_size=intermediate
            ),
        )


def test_per_expert_manifest_parses() -> None:
    manifest = BtxManifest.from_dict(_per_expert_manifest_dict())
    assert manifest.rates.pair_kinds == frozenset({"P33", "P43"})
    assert manifest.rates.uniform_code() is None


def test_mcg_manifest_requires_seed() -> None:
    data = _uniform_manifest_dict(codebook="mcg", bits=3)
    BtxManifest.from_dict(data)
    missing = copy.deepcopy(data)
    del missing["codebook_seed"]
    with pytest.raises(ValueError, match="codebook_seed"):
        BtxManifest.from_dict(missing)
    wrong = copy.deepcopy(data)
    wrong["codebook_seed"] = 1
    with pytest.raises(ValueError, match="codebook_seed"):
        BtxManifest.from_dict(wrong)


def test_sqg_manifest_forbids_seed() -> None:
    data = _uniform_manifest_dict()
    data["codebook_seed"] = 0xCBAC1FED
    with pytest.raises(ValueError, match="only for mcg"):
        BtxManifest.from_dict(data)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda d: d.update(kind="qsrt-manifest"), "kind"),
        (lambda d: d.update(schema="btx-atoms-v2"), "schema"),
        (lambda d: d.update(codebook="sqg_xor_cheb_t12"), "codebook"),
        (lambda d: d.update(profile="k2"), "unknown keys"),
        (lambda d: d["geometry"].update(atom_slots=15), "atom_slots"),
        (lambda d: d["geometry"].update(atom_channels=16), "32 channels"),
        (lambda d: d["geometry"].update(hidden_size=250), "multiple of 16"),
        (lambda d: d["geometry"].update(rotation_multiplier=5), "unknown keys"),
        (lambda d: d["geometry"].update(moe_layer_indices=[1, 1]), "distinct"),
        (lambda d: d["rates"].update(bits=7), "2..6"),
        (lambda d: d["rates"].update(structure="profile"), "structure"),
        (
            lambda d: d["rates"].update(pair_kinds=["P33"]),
            "uniform BTX rates declare bits and no pair_kinds",
        ),
        (lambda d: d["hadamard"].update(pre_block=512), "coupled"),
        (lambda d: d["layout"].update(extent_alignment_slots=5), "divisor"),
        (lambda d: d["layout"].update(extent_barriers=[0]), "interior"),
        (lambda d: d["layers"].pop("2"), "moe_layer_indices"),
        (
            lambda d: d["layers"]["1"].update(sha256="ff"),
            "hex sha256",
        ),
    ],
)
def test_manifest_fails_closed(mutate, match) -> None:
    data = _uniform_manifest_dict()
    mutate(data)
    with pytest.raises(ValueError, match=match):
        BtxManifest.from_dict(data)


def test_sqg_bit_ranges_fail_closed() -> None:
    with pytest.raises(ValueError, match="sqg_e4m3"):
        BtxManifest.from_dict(_uniform_manifest_dict(bits=5))
    with pytest.raises(ValueError, match="sqg_fp16"):
        BtxManifest.from_dict(
            _uniform_manifest_dict(codebook="sqg_fp16", bits=3)
        )
    BtxManifest.from_dict(_uniform_manifest_dict(codebook="sqg_fp16", bits=6))


def test_per_expert_rates_fail_closed() -> None:
    data = _per_expert_manifest_dict()
    data["rates"]["pair_kinds"] = ["P33", "P35"]
    with pytest.raises(ValueError, match="pair_kinds"):
        BtxManifest.from_dict(data)
    data = _per_expert_manifest_dict()
    data["rates"]["bits"] = 3
    with pytest.raises(ValueError, match="no bits"):
        BtxManifest.from_dict(data)


def test_coupled_manifest_block_rules() -> None:
    manifest = BtxManifest.from_dict(_uniform_manifest_dict(coupled=True))
    assert manifest.hadamard.pre_block == 512
    assert manifest.hadamard.post_block == 128
    data = _uniform_manifest_dict(coupled=True)
    del data["hadamard"]["post_block"]
    with pytest.raises(ValueError, match="pre_block and post_block"):
        BtxManifest.from_dict(data)
    data = _uniform_manifest_dict(coupled=True)
    data["hadamard"]["post_block"] = 48
    with pytest.raises(ValueError, match="multiple of 32"):
        BtxManifest.from_dict(data)


def test_extent_validation() -> None:
    manifest = BtxManifest.from_dict(_uniform_manifest_dict())
    manifest.validate_extent(0, 8)
    manifest.validate_extent(8, 8)
    manifest.validate_extent(12, 4)
    with pytest.raises(ValueError, match="align"):
        manifest.validate_extent(2, 4)
    with pytest.raises(ValueError, match="align"):
        manifest.validate_extent(0, 6)
    with pytest.raises(ValueError, match="exceeds"):
        manifest.validate_extent(12, 8)
    with pytest.raises(ValueError, match="barrier"):
        manifest.validate_extent(4, 8)


def test_bundle_byte_arithmetic_matches_frozen_sizes() -> None:
    # Kimi-K3 geometry: H=3584 reproduces the frozen per-atom matrix sizes.
    assert matrix_atom_bytes(3584, 2, 2) == 28_672
    assert matrix_atom_bytes(3584, 3, 3) == 43_008
    assert matrix_atom_bytes(3584, 4, 3) == 50_176
    assert matrix_atom_bytes(3584, 4, 4) == 57_344
    assert bundle_bytes(3584, rate_code(3, 3), rate_code(3, 3)) == 129_024


def _small_config(**overrides) -> BtxSynthConfig:
    defaults = dict(
        codebook="sqg_e4m3",
        num_experts=4,
        hidden_size=128,
        intermediate_size=512,
        moe_layer_indices=(1,),
        bits=2,
        atom_row_alignment=4096,
        extent_alignment_slots=4,
        seed=7,
    )
    defaults.update(overrides)
    return BtxSynthConfig(**defaults)


def test_synth_rows_are_expert_major_and_zero_padded() -> None:
    config = _small_config()
    payloads = synth_layer_payloads(config, 1)
    atoms = assemble_atoms_rows(config, 1, payloads)
    assert atoms.dtype == torch.uint8
    assert atoms.shape[0] == config.atom_slots
    assert atoms.shape[1] % config.atom_row_alignment == 0
    per_bundle = bundle_bytes(
        config.hidden_size, rate_code(2, 2), rate_code(2, 2)
    )
    payload = per_bundle * config.num_experts
    assert bool(torch.all(atoms[:, payload:] == 0))
    # Expert 1's gate low plane sits after expert 0's complete bundle.
    low, _high = payloads.planes[(1, 0, 0)]
    raw = low.contiguous().view(torch.uint8).reshape(-1)
    assert torch.equal(atoms[0, per_bundle : per_bundle + raw.numel()], raw)


def test_write_btx_checkpoint_round_trips(tmp_path) -> None:
    config = _small_config()
    manifest = write_btx_checkpoint(tmp_path, config)
    assert manifest.rates.uniform_code() == rate_code(2, 2)
    on_disk = json.loads((tmp_path / BTX_MANIFEST_FILENAME).read_text())
    reparsed = BtxManifest.from_dict(on_disk)
    assert reparsed == manifest

    from safetensors import safe_open

    ref = manifest.layers[1]
    with safe_open(str(tmp_path / ref.file), framework="pt") as handle:
        metadata = handle.metadata()
        names = set(handle.keys())
        rotations = handle.get_tensor("rotations")
    assert metadata["schema"] == "btx-atoms-v1"
    assert metadata["codebook"] == "sqg_e4m3"
    assert metadata["layer"] == "1"
    assert names == {"atoms", "rotations", "gate_suh", "up_suh", "down_svh"}
    assert rotations.shape == (
        config.atom_slots,
        config.num_experts,
        3,
        ATOM_CHANNELS,
    )


def test_write_btx_checkpoint_coupled_and_per_expert(tmp_path) -> None:
    pairs = 512 // ATOM_CHANNELS // 8
    fc1 = torch.full((pairs, 4), rate_code(3, 3), dtype=torch.uint8)
    fc1[0, 1] = rate_code(4, 3)
    fc2 = torch.full((pairs, 4), rate_code(3, 3), dtype=torch.uint8)
    config = _small_config(
        bits=None,
        rate_tables={1: (fc1, fc2)},
        extent_alignment_slots=8,
    )
    manifest = write_btx_checkpoint(tmp_path / "per-expert", config)
    assert manifest.rates.pair_kinds == frozenset({"P33", "P43"})

    coupled = _small_config(
        coupled=True, pre_block=512, post_block=128, seed=9, hidden_size=512
    )
    manifest = write_btx_checkpoint(tmp_path / "coupled", coupled)
    assert manifest.hadamard.coupled

    from safetensors import safe_open

    with safe_open(
        str(tmp_path / "coupled" / manifest.layers[1].file), framework="pt"
    ) as handle:
        assert "rotation_draws" in set(handle.keys())


def test_synth_rejects_unknown_rate_codes() -> None:
    pairs = 2
    bad = torch.full((pairs, 4), 0x53, dtype=torch.uint8)
    good = torch.full((pairs, 4), rate_code(3, 3), dtype=torch.uint8)
    config = _small_config(bits=None, rate_tables={1: (bad, good)})
    with pytest.raises(ValueError, match="unknown rate code"):
        synth_layer_payloads(config, 1)
