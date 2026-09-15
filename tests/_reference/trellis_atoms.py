"""Synthetic canonical atom bytes and independently assembled projection matrices."""

import torch

from b12x.moe import fused_moe
from tests.moe.test_trellis_config import _glm_config, _k3_config
from tests._reference.trellis_decode import native_weight


def atom_fixture(
    *,
    codebook="mcg",
    group_size=32,
    granularity="per_expert_projection",
    coupled=False,
    experts=3,
    hidden=512,
    width=256,
    device="cpu",
):
    generator = torch.Generator().manual_seed(413)
    config = _glm_config() if codebook == "mcg" else _k3_config()
    config["codebook"] = codebook
    config["rate"] = {"granularity": granularity}
    if group_size is not None:
        config["rate"]["group_size"] = group_size
    config["transform"]["expert"] = (
        _k3_config()["transform"]["expert"] if coupled else {"kind": "none"}
    )
    for name in config["scale"]:
        config["scale"][name] = {"vectors": "per_expert", "gains": "none"}
    config = fused_moe.TrellisConfig.from_dict(config)
    groups = width // (group_size or width)
    palette = {
        "mcg": (0x42, 0x33, 0x25, 0x64, 0x56),
        "sqg_e4m3": (0x42, 0x33, 0x24, 0x43),
        "sqg_fp16": (0x65, 0x55, 0x56, 0x66),
    }[codebook]
    logical = torch.empty(groups, experts, 3, dtype=torch.uint8)
    for group in range(groups):
        for expert in range(experts):
            for projection in range(3):
                index = group
                if granularity in {"per_expert", "per_expert_projection"}:
                    index += expert
                if granularity == "per_expert_projection":
                    index += projection
                logical[group, expert, projection] = palette[index % len(palette)]
    if granularity in {"uniform", "per_layer"}:
        selected = (
            logical[:, 0, 0].contiguous()
            if group_size is not None
            else logical[:1, 0, 0]
        )
    elif granularity == "per_expert":
        selected = (
            logical[:, :, 0].T.contiguous()
            if group_size is not None
            else logical[0, :, 0].contiguous()
        )
    else:
        selected = (
            logical.permute(1, 2, 0).contiguous()
            if group_size is not None
            else logical[0].contiguous()
        )
    rows = []
    expected = {
        (p, e): torch.empty(
            (width, hidden) if p < 2 else (hidden, width), dtype=torch.float16
        )
        for e in range(experts)
        for p in range(3)
    }
    for slot in range(width // 32):
        group = slot * 32 // (group_size or width)
        pieces = []
        for expert in range(experts):
            for projection in range(3):
                code = int(logical[group, expert, projection])
                for plane, bits in enumerate((code & 15, code >> 4)):
                    record = torch.randint(
                        -32768,
                        32768,
                        (hidden // 16, 16 * bits),
                        dtype=torch.int16,
                        generator=generator,
                    )
                    pieces.append(record.flatten().view(torch.uint8))
                    shape = (
                        (1, hidden // 16, 1, 16 * bits)
                        if projection < 2
                        else (1, 1, hidden // 16, 16 * bits)
                    )
                    decoded = native_weight(record.reshape(shape), bits, codebook)[0]
                    begin = slot * 32 + plane * 16
                    if projection < 2:
                        expected[projection, expert][begin : begin + 16] = decoded
                    else:
                        expected[projection, expert][:, begin : begin + 16] = decoded
        rows.append(torch.cat(pieces))
    # Deliberate padding proves that offsets use the declared physical pitch.
    stride = max(row.numel() for row in rows) + 128
    atoms = torch.zeros(len(rows), stride, dtype=torch.uint8)
    for i, row in enumerate(rows):
        atoms[i, : row.numel()] = row

    def scales(shape):
        return fused_moe.ScaleFactors(
            (0.75 + 0.5 * torch.rand(shape, generator=generator)).half().to(device)
        )

    weights = fused_moe.TrellisWeights(
        atoms=atoms.to(device),
        rate=selected.to(device),
        input_scales=scales((experts, 2, hidden)),
        intermediate_scales=scales((experts, 3, width)),
        output_scales=scales((experts, hidden)),
        expert_transform_draws=torch.arange(
            experts, dtype=torch.uint8, device=device
        ).remainder(8)
        if coupled
        else None,
        global_intermediate_size=width if coupled else None,
    )
    plan = fused_moe.plan_weights(
        source=config,
        activation=fused_moe.ActivationSpec(
            mode="a16",
            nonlinearity="situ" if coupled else "silu",
            io_dtype=torch.bfloat16,
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=experts, hidden_size=hidden, intermediate_size=width
        ),
    )
    return plan, weights, logical, expected


def btx_atom_fixture(
    path, *, codebook="sqg_e4m3", coupled=False, experts=3,
    hidden=512, width=512, global_width=768, first_slot=0,
):
    from dataclasses import replace
    from b12x.moe._shared.kernels.w4a16.btx import read_btx_layer
    from b12x.moe._shared.kernels.w4a16.btx_synth import (
        BtxSynthConfig, synth_layer_payloads, write_btx_checkpoint,
    )

    palette = (0x22, 0x33, 0x24, 0x43, 0x44)
    fc1 = torch.empty(global_width // 256, experts, dtype=torch.uint8)
    fc2 = torch.empty_like(fc1)
    for pair in range(global_width // 256):
        for expert in range(experts):
            fc1[pair, expert] = palette[(pair + expert) % len(palette)]
            fc2[pair, expert] = palette[(pair + expert + 3) % len(palette)]
    config = BtxSynthConfig(
        codebook=codebook, num_experts=experts, hidden_size=hidden,
        intermediate_size=global_width, moe_layer_indices=(0,),
        rate_tables={0: (fc1, fc2)}, coupled=coupled,
        pre_block=512 if coupled else None, post_block=128 if coupled else None,
        per_expert_input_rotations=True, extent_alignment_slots=8, seed=819,
    )
    manifest = write_btx_checkpoint(path, config)
    layer = read_btx_layer(path, manifest, 0, first_slot=first_slot, slot_count=width // 32)
    # Safetensors CPU mappings need not share the CUDA allocator's alignment.
    layer = replace(
        layer, atoms=layer.atoms.clone(), gate_suh=layer.gate_suh.clone(),
        up_suh=layer.up_suh.clone(), down_svh=layer.down_svh.clone(),
    )
    from b12x.moe.fused_moe._impl import plan_b12x_fp4_moe_weights
    plan = plan_b12x_fp4_moe_weights(
        quant_modes="w4a16", source_format="btx",
        activation="situ" if coupled else "silu", params_dtype=torch.bfloat16,
        num_experts=experts, hidden_size=hidden, intermediate_size=width,
        trellis_codebook=codebook, trellis_bits=3,
        trellis_rate_granularity="per_expert_pair",
        trellis_pair_kinds=manifest.rates.pair_kinds, coupled_hadamard=coupled,
        coupled_hadamard_blocks=(512, 128) if coupled else None,
    )
    payloads = synth_layer_payloads(config, 0)
    expected = {}
    for expert in range(experts):
        for projection in range(3):
            records = []
            for pair in range(width // 256):
                for plane in range(2):
                    tiles = []
                    for atom in range(8):
                        slot = first_slot + pair * 8 + atom
                        record = payloads.planes[expert, slot, projection][plane]
                        bits = record.shape[-1] // 16
                        shape = (
                            (1, hidden // 16, 1, 16 * bits)
                            if projection < 2 else (1, 1, hidden // 16, 16 * bits)
                        )
                        tiles.append(native_weight(record.reshape(shape), bits, codebook)[0])
                    records.append(torch.cat(tiles, dim=0 if projection < 2 else 1))
            expected[projection, expert] = torch.cat(records, dim=0 if projection < 2 else 1)
    return plan, layer, expected
