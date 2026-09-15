"""Canonical BTX declarations reject incompatible metadata before CUDA work."""

from dataclasses import replace

import pytest
import torch

from b12x.moe import fused_moe
from tests._reference.trellis_atoms import btx_atom_fixture


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    raw, layer, _ = btx_atom_fixture(tmp_path_factory.mktemp("btx-public"))
    source = fused_moe.BtxSource(manifest=layer.manifest)
    activation = fused_moe.ActivationSpec(
        mode="a16",
        nonlinearity=raw.activation,
        io_dtype=torch.bfloat16,
    )
    geometry = fused_moe.MoEGeometry(
        num_experts=raw.num_experts,
        hidden_size=raw.hidden_size,
        intermediate_size=raw.intermediate_size,
    )
    return raw, layer, source, activation, geometry


def test_canonical_btx_preserves_native_plan(checkpoint):
    raw, _, source, activation, geometry = checkpoint
    plan = fused_moe.plan_weights(
        source=source, activation=activation, geometry=geometry
    )
    assert plan._impl == raw
    assert plan.prepared_format.packing is fused_moe.WeightPacking.TRELLIS_NATIVE
    assert plan.prepared_format.weights is fused_moe.WeightEncoding.TRELLIS


@pytest.mark.parametrize(
    "field,value",
    [("num_experts", 4), ("hidden_size", 1024), ("intermediate_size", 1024)],
)
def test_btx_geometry_mismatch(checkpoint, field, value):
    _, _, source, activation, geometry = checkpoint
    with pytest.raises(ValueError, match="geometry"):
        fused_moe.plan_weights(
            source=source,
            activation=activation,
            geometry=replace(geometry, **{field: value}),
        )


def test_btx_requires_a16_and_native_packing(checkpoint):
    _, _, source, activation, geometry = checkpoint
    with pytest.raises(ValueError, match="A16"):
        fused_moe.plan_weights(
            source=source, activation=replace(activation, mode="a8"), geometry=geometry
        )
    with pytest.raises(ValueError, match="packing"):
        fused_moe.plan_weights(
            source=source,
            activation=activation,
            geometry=geometry,
            constraints=fused_moe.WeightPlanConstraints(required_packing="mma_packed"),
        )


def test_btx_bundle_rejects_wrong_source_or_destination(checkpoint):
    _, layer, source, activation, geometry = checkpoint
    plan = fused_moe.plan_weights(
        source=source, activation=activation, geometry=geometry
    )
    with pytest.raises(TypeError, match="BtxManifest"):
        fused_moe.BtxSource(manifest={})
    with pytest.raises(TypeError, match="BtxLayer"):
        fused_moe.BtxWeights(layer=None, device="cuda")
    with pytest.raises(ValueError, match="CUDA destination"):
        fused_moe.BtxWeights(layer=layer, device="cpu")
    with pytest.raises(TypeError, match="BtxWeights"):
        fused_moe.prepare_weights(plan=plan, weights=None)
    different = replace(layer, manifest=replace(layer.manifest, codebook="mcg"))
    with pytest.raises(ValueError, match="manifest differs"):
        fused_moe.prepare_weights(
            plan=plan, weights=fused_moe.BtxWeights(layer=different, device="cuda")
        )
