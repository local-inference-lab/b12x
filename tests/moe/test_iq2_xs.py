"""Independent codec, compact-storage and public planning contracts."""

from __future__ import annotations

import hashlib

import pytest
import torch

from b12x._lib.quant.iq2_xs import iq2_xs_execution_lut_cpu
from b12x.moe import fused_moe as moe
from b12x.moe._shared.kernels.w4a16.iq2_xs import pack_iq2_xs_matrix
from b12x.testing.iq2_xs_reference import dequantize_blocks, descriptor_vectors


def blocks(e=2, n=544, k=768):
    g = torch.Generator().manual_seed(712)
    raw = torch.randint(0, 256, (e, n, k // 256, 74), dtype=torch.uint8, generator=g)
    bases = (torch.randn(e, n, k // 256, generator=g) * 0.01).half()
    raw[..., :2] = bases[..., None].view(torch.uint8)
    return raw


def unpack_planes(words, metadata, shape):
    e, n, kb, _ = shape
    raw = torch.empty(shape, dtype=torch.uint8)
    q = words.reshape(e, kb * 16, n, 1).view(torch.uint8)
    raw[..., 2:66] = q.permute(0, 2, 1, 3).reshape(e, n, kb, 64)
    raw[..., :2] = metadata[: e * kb * n * 2].reshape(e, kb, n, 2).permute(0, 2, 1, 3)
    raw[..., 66:] = (
        metadata[e * kb * n * 2 :]
        .reshape(e, kb, n // 16, 8, 16)
        .permute(0, 2, 4, 1, 3)
        .reshape(e, n, kb, 8)
    )
    return raw


def test_every_descriptor_matches_independent_table():
    actual, reference = iq2_xs_execution_lut_cpu(), descriptor_vectors()
    assert (
        hashlib.sha256(reference.numpy().tobytes()).hexdigest()
        == "516a77f2ff3be1f22250b0ef65df740495343ec90feb2e21cb7a04a22041deeb"
    )
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("swap", [False, True])
def test_compact_planes_preserve_source_bytes_and_projection_order(swap):
    source = blocks()
    original = source.clone()
    words, metadata = pack_iq2_xs_matrix(source, swap_halves=swap)
    assert words.numel() * words.element_size() + metadata.numel() == source.numel()
    assert (
        words.untyped_storage().nbytes() + metadata.untyped_storage().nbytes()
        == source.numel()
    )
    expected = torch.cat(source.chunk(2, 1)[::-1], 1) if swap else source
    reconstructed = unpack_planes(words, metadata, source.shape)
    assert torch.equal(reconstructed, expected)
    assert torch.equal(source, original)


@pytest.mark.parametrize(
    "base_bits", [0, 0x8000, 1, 0x8001, 0x3FF, 0x400, 0x3C01, 0x7BFF, 0xFBFF]
)
def test_scale_nibbles_and_fp16_edges_survive_packing(base_bits):
    source = blocks(e=1, n=32, k=256)
    source[..., 0], source[..., 1] = base_bits & 255, base_bits >> 8
    source[..., 66:] = (
        torch.arange(8, dtype=torch.uint8) * 2
        + (torch.arange(8, dtype=torch.uint8) * 2 + 1) * 16
    )
    words, metadata = pack_iq2_xs_matrix(source)
    decoded = dequantize_blocks(unpack_planes(words, metadata, source.shape)).bfloat16()
    reference = dequantize_blocks(source).bfloat16()
    assert torch.equal(decoded.view(torch.int16), reference.view(torch.int16))
    assert torch.isfinite(decoded).all()


@pytest.mark.parametrize("bits", [0x7C00, 0xFC00, 0x7E00])
def test_nonfinite_bases_rejected(bits):
    source = blocks(e=1, n=16, k=256)
    source[0, 0, 0, 0], source[0, 0, 0, 1] = bits & 255, bits >> 8
    with pytest.raises(ValueError, match="finite"):
        pack_iq2_xs_matrix(source)


def weight_plan(
    *, mode="a16", activation="silu", dtype=torch.bfloat16, h=2048, i=512, packing=None
):
    return moe.plan_weights(
        source=moe.PackedSource(
            format=moe.PackedSourceFormat.IQ2_XS, w13_layout=moe.W13Layout.W31
        ),
        activation=moe.ActivationSpec(
            mode=mode, nonlinearity=activation, io_dtype=dtype
        ),
        geometry=moe.MoEGeometry(num_experts=256, hidden_size=h, intermediate_size=i),
        constraints=moe.WeightPlanConstraints(required_packing=packing),
    )


@pytest.mark.parametrize("activation", ["silu", "relu2"])
@pytest.mark.parametrize("i", [256, 512])
def test_public_weight_plan(activation, i):
    plan = weight_plan(activation=activation, i=i)
    assert plan.prepared_format.weights is moe.WeightEncoding.IQ2_XS
    assert plan.prepared_format.scales is moe.ScaleEncoding.IQ2_XS
    assert plan.prepared_format.packing is moe.WeightPacking.IQ2_XS_COMPACT
    assert plan._impl.w4a16_weight_layout == "iq2_xs"
    assert plan._impl.w4a16_scale_format == "iq2_xs"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "a8"},
        {"mode": "a4"},
        {"dtype": torch.float16},
        {"activation": "swigluoai"},
        {"h": 384},
        {"i": 128},
        {"packing": "mma_packed"},
    ],
)
def test_unsupported_weight_contracts_fail_at_planning(kwargs):
    with pytest.raises((ValueError, NotImplementedError)):
        weight_plan(**kwargs)


@pytest.mark.parametrize(
    "source", [torch.zeros(2, 16, 1, 73, dtype=torch.uint8), torch.zeros(2, 16, 1, 74)]
)
def test_invalid_payload_rejected(source):
    with pytest.raises((TypeError, ValueError)):
        moe.IQ2XSWeights(source, source)
    with pytest.raises((TypeError, ValueError)):
        pack_iq2_xs_matrix(source)
