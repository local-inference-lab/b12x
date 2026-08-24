"""End-to-end w4a8_mx serving of a private trellis kernel recipe.

The canonical b12x_trellis checkpoint contract does not expose this K2/W4A8
combination. This test keeps low-level kernel coverage without presenting a
second public checkpoint format.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from b12x.moe import fused_moe
from b12x.moe._shared.execution import PreparedWeightLayout
from b12x.moe._shared.kernels.w4a16.prepare import (
    PreparedW4A16MoeWeights,
    TrellisWeightState,
)
from b12x.moe.fused_moe._impl import (
    B12XFP4ExpertWeights,
    _PreparedWeightRepresentation,
    plan_b12x_fp4_moe_weights,
)
from tests._reference.helpers import require_b12x
from tests._reference.trellis_moe import (
    build_trellis_weight,
    trellis_moe_reference,
)

require_b12x()

_BITS = 2


def _make_expert_weights(
    *,
    E: int,
    K: int,
    n: int,
    coupled: bool,
    seed: int,
    device: torch.device,
) -> tuple[B12XFP4ExpertWeights, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    gate_payload, gate_w = build_trellis_weight(gen, E, n, K, _BITS, device)
    up_payload, up_w = build_trellis_weight(gen, E, n, K, _BITS, device)
    down_payload, down_w = build_trellis_weight(gen, E, K, n, _BITS, device)
    w13 = torch.stack((gate_payload, up_payload), dim=0).contiguous()
    rot_segments = 6 if coupled else 3
    rotations = (
        torch.rand((E, rot_segments * n), generator=gen, dtype=torch.float32)
        .add_(0.5)
        .to(device=device, dtype=torch.float16)
    )

    # The QSRT plan catalog admits K2 only as the coupled atoms-v2
    # profile; the ordinary boundary stays covered by the kernel-level
    # parity suites.
    assert coupled
    plan = plan_b12x_fp4_moe_weights(
        quant_modes="w4a8_mx",
        source_format="btx",
        activation="situ",
        params_dtype=torch.bfloat16,
        num_experts=E,
        hidden_size=K,
        intermediate_size=n,
        trellis_bits=_BITS,
        trellis_codebook="sqg_e4m3",
        trellis_tile_config=(128, 128, 128, 128),
        coupled_hadamard=True,
    )
    plan = replace(
        plan,
        checkpoint_config=fused_moe.PackedConfig(
            source_format="fp4_e8m0_k32"
        ),
    )
    dummy_scale = torch.zeros(4, dtype=torch.uint8, device=device)
    ones_e = torch.ones(E, dtype=torch.float32, device=device)
    value = PreparedW4A16MoeWeights(
        w13=w13.view(torch.int32),
        w13_scale=dummy_scale,
        w13_global_scale=ones_e,
        w2=down_payload.contiguous().view(torch.int32),
        w2_scale=dummy_scale,
        w2_global_scale=ones_e,
        workspace=torch.zeros(1, dtype=torch.int32, device=device),
        hidden_size=K,
        intermediate_size=n,
        num_experts=E,
        is_gated=True,
        params_dtype=torch.float16,
        fc1_tile_n=128,
        fc2_tile_n=128,
        source_format="btx",
        w13_layout="trellis_t256_proj",
        weight_layout="trellis_t256",
        scale_format="e4m3_k32",
        trellis=TrellisWeightState(
            codebook="sqg_e4m3",
            bits=_BITS,
            intermediate_rotations=rotations,
            coupled_hadamard=coupled,
        ),
    )
    scalar_one = torch.ones((), dtype=torch.float32, device=device)
    experts = B12XFP4ExpertWeights(
        plan=plan,
        a1_gscale=scalar_one,
        w1_fp4=value.w13,
        w1_blockscale=value.w13_scale,
        w1_alphas=value.w13_global_scale,
        a2_gscale=scalar_one,
        w2_fp4=value.w2,
        w2_blockscale=value.w2_scale,
        w2_alphas=value.w2_global_scale,
        representation=_PreparedWeightRepresentation(
            quant_mode="w4a8_mx",
            layout=PreparedWeightLayout.TRELLIS_NATIVE,
            value=value,
        ),
    )
    w13_ref = torch.stack((gate_w, up_w), dim=0)
    return experts, w13_ref, down_w, rotations


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
# m in {1, 4} lands in the micro band; m=16 exceeds it and exercises the
# dynamic backend's w4a8_trellis recipe through the same public surface.
@pytest.mark.parametrize("m", [1, 4, 16])
def test_w4a8_trellis_micro_serving_matches_reference(m: int) -> None:
    coupled = True
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(20260812)
    E, K, n, topk = 32, 1024, 256, 4
    experts, w13_ref, down_w, rotations = _make_expert_weights(
        E=E, K=K, n=n, coupled=coupled, seed=20260812, device=device
    )

    plan = fused_moe.plan(
        fused_moe.Caps(
            config=experts.plan.checkpoint_config,
            max_tokens=m,
            num_topk=topk,
            route_num_experts=E,
            device=device,
            weight_plan=experts.plan,
        )
    )
    spec = plan.scratch_specs()[0]
    scratch = torch.zeros(spec.shape, dtype=spec.dtype, device=spec.device)

    x = (torch.randn((m, K), device=device) * 1.0e-2).to(torch.bfloat16)
    ids = torch.stack(
        [torch.randperm(E, device=device)[:topk] for _ in range(m)]
    ).to(torch.int32)
    topk_weights = torch.softmax(
        torch.randn(m, topk, device=device), dim=-1
    ).float()

    binding = fused_moe.bind(
        plan,
        scratch=scratch,
        a=x,
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=ids,
    )
    out = fused_moe.run(binding=binding)
    torch.cuda.synchronize(device)

    oracle = trellis_moe_reference(
        x.float(),
        w13_ref,
        down_w,
        rotations,
        ids,
        topk_weights,
        coupled=coupled,
    )
    got = out.float()
    assert torch.isfinite(got).all()
    cosine = torch.nn.functional.cosine_similarity(
        got.reshape(1, -1), oracle.reshape(1, -1)
    ).item()
    assert cosine > 0.995, cosine
    rel = ((got - oracle).norm() / oracle.norm().clamp_min(1e-9)).item()
    assert rel < 0.12, rel
