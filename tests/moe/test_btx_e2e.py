"""End-to-end BTX serving through the public fused-MoE surface.

Covers the production support matrix — uniform K2-K6 across the mcg,
sqg_e4m3, and sqg_fp16 codebooks with coupled and uncoupled Hadamard — plus
the supported non-production per-expert-pair structures. Every case drives
plan_weights -> prepare_weights(btx_layer) -> Caps/plan/bind/run against a
decoded torch reference, then requires CUDA-graph capture/replay equality.
"""

from __future__ import annotations

import pytest
import torch

from b12x.moe import fused_moe
from b12x.moe._shared.btx_schema import rate_code
from b12x.moe._shared.kernels.w4a16.btx import read_btx_layer
from b12x.moe._shared.kernels.w4a16.btx_synth import (
    BtxSynthConfig,
    synth_layer_payloads,
    write_btx_checkpoint,
)
from tests._reference.helpers import require_b12x
from tests._reference.trellis_moe import trellis_moe_reference
from tests.moe.test_fused_moe_trellis import (
    _reconstruct_native,
    _reference_coupled_decoded,
    _reference_full_rotation_decoded,
    _sm12x_available,
)

require_b12x()

requires_sm12x = pytest.mark.skipif(
    not _sm12x_available(), reason="requires an SM120/SM121 GPU"
)


def _device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _decoded_uniform_weights(config, payloads, device):
    """Decode per-expert matrices from planes with naive assembly."""

    hidden_tiles = config.hidden_size // 16
    slots = config.atom_slots
    bits = config.bits
    experts = config.num_experts
    gate = torch.empty(
        (2, experts, hidden_tiles, 2 * slots, 16 * bits), dtype=torch.int16
    )
    down = torch.empty(
        (experts, 2 * slots, hidden_tiles, 16 * bits), dtype=torch.int16
    )
    for expert in range(experts):
        for slot in range(slots):
            for matrix in range(2):
                low, high = payloads.planes[(expert, slot, matrix)]
                gate[matrix, expert, :, 2 * slot] = low
                gate[matrix, expert, :, 2 * slot + 1] = high
            low, high = payloads.planes[(expert, slot, 2)]
            down[expert, 2 * slot] = low
            down[expert, 2 * slot + 1] = high
    gate_w = torch.stack(
        [
            _reconstruct_native(gate[0, e], codebook=config.codebook)
            for e in range(experts)
        ]
    ).to(device)
    up_w = torch.stack(
        [
            _reconstruct_native(gate[1, e], codebook=config.codebook)
            for e in range(experts)
        ]
    ).to(device)
    down_w = torch.stack(
        [
            _reconstruct_native(down[e], codebook=config.codebook)
            for e in range(experts)
        ]
    ).to(device)
    return gate_w, up_w, down_w


def _serve(weights, *, x, ids, router_weights, quant_mode="w4a16"):
    output_dtype = torch.float32 if quant_mode == "w4a16" else x.dtype
    output = torch.zeros(
        (int(x.shape[0]), int(x.shape[1])), dtype=output_dtype, device=x.device
    )
    plan = fused_moe.plan(
        fused_moe.Caps(
            max_tokens=int(x.shape[0]),
            num_topk=int(ids.shape[1]),
            route_num_experts=int(weights.plan.num_experts),
            device=x.device,
            weight_plan=weights.plan,
            quant_mode=quant_mode,
        )
    )
    spec = plan.scratch_specs()[0]
    scratch = torch.zeros(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = fused_moe.bind(
        plan,
        scratch=scratch,
        a=x,
        experts=weights,
        topk_weights=router_weights,
        topk_ids=ids,
        output=output,
    )
    actual = fused_moe.run(binding=binding).clone()
    torch.cuda.synchronize(x.device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fused_moe.run(binding=binding)
    graph.replay()
    torch.cuda.synchronize(x.device)
    if quant_mode == "w4a16":
        assert torch.equal(captured, actual)
    else:
        # The dynamic W4A8 band reduces through atomic scatter, so eager
        # and replay accumulate in different orders; require closeness,
        # not bit equality.
        delta = (captured.float() - actual.float()).norm()
        assert float(delta / actual.float().norm().clamp_min(1e-9)) < 1e-2
    return actual


def _routing(m, experts, topk, device, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.stack(
        [
            torch.randperm(experts, generator=generator)[:topk]
            for _ in range(m)
        ]
    ).to(device=device, dtype=torch.int32)
    router_weights = torch.softmax(
        torch.randn(m, topk, generator=generator), dim=-1
    ).to(device=device, dtype=torch.float32)
    return ids, router_weights


@requires_sm12x
def test_btx_uniform_prepared_owner_preserves_public_plan_contract(tmp_path) -> None:
    """A native owner can cross a checkpoint adapter without repacking."""

    device = _device()
    experts, hidden, intermediate = 4, 256, 512
    config = BtxSynthConfig(
        codebook="mcg",
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        moe_layer_indices=(1,),
        bits=3,
        extent_alignment_slots=4,
        seed=20260817,
    )
    manifest = write_btx_checkpoint(tmp_path, config)
    layer = read_btx_layer(
        tmp_path,
        manifest,
        1,
        first_slot=0,
        slot_count=config.atom_slots,
    )
    plan = fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format="btx",
        activation="situ",
        params_dtype=torch.float16,
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        trellis_bits=3,
        trellis_codebook="mcg",
        trellis_tile_config=(64, 256, 64, 256),
    )
    from_atoms = fused_moe.prepare_weights(
        plan=plan,
        params_dtype=torch.float16,
        btx_layer=layer,
        btx_device=device,
    )
    native = from_atoms.representation_for("w4a16")

    adopted = fused_moe.adopt_btx_weights(
        plan=plan,
        prepared=native,
    )

    assert adopted.representation_for("w4a16") is native
    assert adopted.w1_fp4.data_ptr() == from_atoms.w1_fp4.data_ptr()
    assert adopted.w2_fp4.data_ptr() == from_atoms.w2_fp4.data_ptr()

    torch.manual_seed(20260817)
    x = (torch.randn((4, hidden), device=device) * 0.05).to(torch.float16)
    ids, router_weights = _routing(4, experts, 2, device, 20260817)
    original_output = _serve(
        from_atoms,
        x=x,
        ids=ids,
        router_weights=router_weights,
    )
    adopted_output = _serve(
        adopted,
        x=x,
        ids=ids,
        router_weights=router_weights,
    )
    assert torch.count_nonzero(original_output).item() > 0
    assert torch.equal(adopted_output, original_output)

    mismatched = fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format="btx",
        activation="situ",
        params_dtype=torch.float16,
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        trellis_bits=4,
        trellis_codebook="mcg",
        trellis_tile_config=(64, 256, 64, 256),
    )
    with pytest.raises(ValueError, match="codebook, bitrate, or tile"):
        fused_moe.adopt_btx_weights(
            plan=mismatched,
            prepared=native,
        )


@requires_sm12x
@pytest.mark.parametrize(
    ("codebook", "bits"),
    [
        ("sqg_e4m3", 3),
        ("sqg_e4m3", 4),
        ("mcg", 3),
        ("mcg", 6),
        ("sqg_fp16", 5),
        ("sqg_fp16", 6),
    ],
)
def test_btx_uniform_uncoupled_serving_matches_reference(
    tmp_path, codebook, bits
) -> None:
    device = _device()
    experts, topk, m, hidden, global_i = 4, 2, 4, 256, 512
    config = BtxSynthConfig(
        codebook=codebook,
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=global_i,
        moe_layer_indices=(1,),
        bits=bits,
        extent_alignment_slots=4,
        seed=100 + bits,
    )
    manifest = write_btx_checkpoint(tmp_path, config)
    layer = read_btx_layer(
        tmp_path, manifest, 1, first_slot=0, slot_count=config.atom_slots
    )
    weight_plan = fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format="btx",
        activation="situ",
        params_dtype=torch.float16,
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=global_i,
        trellis_bits=bits,
        trellis_codebook=codebook,
        trellis_tile_config=(64, 256, 64, 256),
    )
    weights = fused_moe.prepare_weights(
        plan=weight_plan,
        params_dtype=torch.float16,
        btx_layer=layer,
        btx_device=device,
    )

    torch.manual_seed(20260815)
    x = (torch.randn((m, hidden), device=device) * 0.05).to(torch.float16)
    ids, router_weights = _routing(m, experts, topk, device, 20260815)
    actual = _serve(weights, x=x, ids=ids, router_weights=router_weights)

    payloads = synth_layer_payloads(config, 1)
    gate_w, up_w, down_w = _decoded_uniform_weights(config, payloads, device)
    prepared = weights.representation_for("w4a16")
    reference = _reference_full_rotation_decoded(
        x,
        ids,
        router_weights,
        gate_w,
        up_w,
        down_w,
        prepared.gate_suh,
        prepared.up_suh,
        prepared.intermediate_rotations,
        prepared.down_svh,
        activation="situ",
    )
    relative_error = (actual - reference).norm() / reference.norm().clamp_min(
        1.0e-9
    )
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(), reference.flatten(), dim=0
    )
    assert float(relative_error) <= 2.0e-2, float(relative_error)
    assert float(cosine) >= 0.999, float(cosine)


@requires_sm12x
@pytest.mark.parametrize(
    ("bits", "tile_config"),
    [
        (2, (128, 128, 128, 128)),
        (3, (64, 256, 64, 256)),
        (4, (64, 256, 64, 256)),
    ],
)
def test_btx_uniform_coupled_serving_matches_reference(
    tmp_path, bits, tile_config
) -> None:
    device = _device()
    experts, topk, m, hidden, global_i = 4, 2, 4, 512, 512
    slots = 8
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
        extent_alignment_slots=4,
        extent_barriers=(8,),
        seed=200 + bits,
    )
    manifest = write_btx_checkpoint(tmp_path, config)
    layer = read_btx_layer(
        tmp_path, manifest, 1, first_slot=0, slot_count=slots
    )
    weight_plan = fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format="btx",
        activation="situ",
        params_dtype=torch.float16,
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=slots * 32,
        trellis_bits=bits,
        trellis_codebook="sqg_e4m3",
        coupled_hadamard=True,
        trellis_tile_config=tile_config,
    )
    weights = fused_moe.prepare_weights(
        plan=weight_plan,
        params_dtype=torch.float16,
        btx_layer=layer,
        btx_device=device,
    )

    torch.manual_seed(20260816)
    x = (torch.randn((m, hidden), device=device) * 0.05).to(torch.float16)
    ids, router_weights = _routing(m, experts, topk, device, 20260816)
    actual = _serve(weights, x=x, ids=ids, router_weights=router_weights)

    intermediate = slots * 32
    payloads = synth_layer_payloads(config, 1)
    local = BtxSynthConfig(
        codebook=config.codebook,
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        moe_layer_indices=(1,),
        bits=bits,
    )
    gate_w, up_w, down_w = _decoded_uniform_weights(local, payloads, device)
    prepared = weights.representation_for("w4a16")
    rotations = prepared.intermediate_rotations
    reference = _reference_coupled_decoded(
        x,
        ids,
        router_weights,
        gate_w,
        up_w,
        down_w,
        prepared.gate_suh,
        rotations[:, : 3 * intermediate],
        rotations[:, 3 * intermediate :],
        prepared.down_svh,
        fc1_pair_kind=None,
    )
    relative_error = (actual - reference).norm() / reference.norm().clamp_min(
        1.0e-9
    )
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(), reference.flatten(), dim=0
    )
    assert float(relative_error) <= 2.0e-2, float(relative_error)
    assert float(cosine) >= 0.999, float(cosine)


@requires_sm12x
@pytest.mark.parametrize("high_kind", ["P24", "P43"])
def test_btx_per_expert_pair_serving_matches_reference(
    tmp_path, high_kind
) -> None:
    device = _device()
    experts, topk, m, hidden = 3, 2, 4, 256
    high = (2, 4) if high_kind == "P24" else (4, 3)
    fc1 = torch.tensor(
        [[rate_code(3, 3), rate_code(*high), rate_code(3, 3)]],
        dtype=torch.uint8,
    )
    fc2 = torch.tensor(
        [[rate_code(*high), rate_code(3, 3), rate_code(3, 3)]],
        dtype=torch.uint8,
    )
    config = BtxSynthConfig(
        codebook="sqg_e4m3",
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=256,
        moe_layer_indices=(1,),
        bits=None,
        rate_tables={1: (fc1, fc2)},
        extent_alignment_slots=8,
        seed=300,
    )
    manifest = write_btx_checkpoint(tmp_path, config)
    layer = read_btx_layer(tmp_path, manifest, 1, first_slot=0, slot_count=8)
    weight_plan = fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format="btx",
        activation="situ",
        params_dtype=torch.float16,
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=256,
        trellis_bits=3,
        trellis_codebook="sqg_e4m3",
        trellis_rate_structure="per_expert_pair",
        trellis_pair_kinds=("P33", high_kind),
        trellis_tile_config=(64, 256, 64, 256),
    )
    weights = fused_moe.prepare_weights(
        plan=weight_plan,
        params_dtype=torch.float16,
        btx_layer=layer,
        btx_device=device,
    )

    torch.manual_seed(20260817)
    x = (torch.randn((m, hidden), device=device) * 0.05).to(torch.float16)
    ids, router_weights = _routing(m, experts, topk, device, 20260817)
    actual = _serve(weights, x=x, ids=ids, router_weights=router_weights)

    # Decoded weights in the pair runtime's record-major channel order.
    payloads = synth_layer_payloads(config, 1)

    def _decode_matrix(expert: int, matrix: int, *, fc1_axis: bool):
        # An FC1 plane is one N16 column (tile grid [ht, 1]); an FC2 plane
        # is one K16 row across all N16 columns (tile grid [1, ht]).
        lows, highs = [], []
        for slot in range(8):
            low, high = payloads.planes[(expert, slot, matrix)]
            shape = (hidden // 16, 1, -1) if fc1_axis else (1, hidden // 16, -1)
            lows.append(_reconstruct_native(
                low.reshape(shape), codebook="sqg_e4m3"
            ))
            highs.append(_reconstruct_native(
                high.reshape(shape), codebook="sqg_e4m3"
            ))
        return torch.cat(lows + highs, dim=1 if fc1_axis else 0)

    gate_w = torch.stack(
        [_decode_matrix(e, 0, fc1_axis=True) for e in range(experts)]
    ).to(device)
    up_w = torch.stack(
        [_decode_matrix(e, 1, fc1_axis=True) for e in range(experts)]
    ).to(device)
    down_w = torch.stack(
        [_decode_matrix(e, 2, fc1_axis=False) for e in range(experts)]
    ).to(device)
    prepared = weights.representation_for("w4a16")
    reference = _reference_full_rotation_decoded(
        x,
        ids,
        router_weights,
        gate_w,
        up_w,
        down_w,
        prepared.gate_suh,
        prepared.up_suh,
        prepared.intermediate_rotations,
        prepared.down_svh,
        activation="situ",
    )
    relative_error = (actual - reference).norm() / reference.norm().clamp_min(
        1.0e-9
    )
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(), reference.flatten(), dim=0
    )
    assert float(relative_error) <= 2.0e-2, float(relative_error)
    assert float(cosine) >= 0.999, float(cosine)


@requires_sm12x
@pytest.mark.parametrize("m", [1, 4, 16])
def test_btx_w4a8_mx_uniform_coupled_serving(tmp_path, m) -> None:
    device = _device()
    experts, topk, hidden, global_i = 32, 4, 1024, 512
    slots = 8
    config = BtxSynthConfig(
        codebook="sqg_e4m3",
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=global_i,
        moe_layer_indices=(1,),
        bits=2,
        coupled=True,
        pre_block=512,
        post_block=128,
        unit_hidden_rotations=True,
        extent_alignment_slots=4,
        extent_barriers=(8,),
        seed=400,
    )
    manifest = write_btx_checkpoint(tmp_path, config)
    layer = read_btx_layer(
        tmp_path, manifest, 1, first_slot=0, slot_count=slots
    )
    weight_plan = fused_moe.plan_weights(
        quant_modes="w4a8_mx",
        source_format="btx",
        activation="situ",
        params_dtype=torch.bfloat16,
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=slots * 32,
        trellis_bits=2,
        trellis_codebook="sqg_e4m3",
        coupled_hadamard=True,
        trellis_tile_config=(128, 128, 128, 128),
    )
    weights = fused_moe.prepare_weights(
        plan=weight_plan,
        params_dtype=torch.bfloat16,
        btx_layer=layer,
        btx_device=device,
    )

    torch.manual_seed(20260818)
    x = (torch.randn((m, hidden), device=device) * 1.0e-2).to(torch.bfloat16)
    ids, router_weights = _routing(m, experts, topk, device, 20260818)
    actual = _serve(
        weights,
        x=x,
        ids=ids,
        router_weights=router_weights,
        quant_mode="w4a8_mx",
    )

    intermediate = slots * 32
    payloads = synth_layer_payloads(config, 1)
    local = BtxSynthConfig(
        codebook=config.codebook,
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        moe_layer_indices=(1,),
        bits=2,
    )
    gate_w, up_w, down_w = _decoded_uniform_weights(local, payloads, device)
    prepared = weights.representation_for("w4a8_mx")
    reference = trellis_moe_reference(
        x.float(),
        torch.stack((gate_w.permute(0, 2, 1), up_w.permute(0, 2, 1))),
        down_w.permute(0, 2, 1),
        prepared.intermediate_rotations,
        ids,
        router_weights,
        coupled=True,
    )
    got = actual.float()
    assert torch.isfinite(got).all()
    cosine = torch.nn.functional.cosine_similarity(
        got.reshape(1, -1), reference.reshape(1, -1)
    ).item()
    assert cosine > 0.995, cosine
    rel = ((got - reference).norm() / reference.norm().clamp_min(1e-9)).item()
    assert rel < 0.12, rel
