"""Bit-identity of the full-rotation top-k sum's in-kernel bf16/fp16 store.

One CUDA device. The full-rotation W4A16 top-k route sum stores an fp32 row
by default and the caller rounds it to the model dtype; with the output
element set to bf16 or fp16 the kernel rounds the same fp32 value once in
its store. Every full-rotation variant (plain H128, coupled Hadamard with a
per-expert and with a broadcast output scale; mapped and unmapped route ids;
int32 and int64 ids; invalid routes present) must produce the bytes of the
separate cast.

Run with B12X_RUN_W4A16_TOPK_SUM_OUTPUT_TEST=1 (the compile takes seconds).
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("B12X_RUN_W4A16_TOPK_SUM_OUTPUT_TEST") != "1"
    or not torch.cuda.is_available(),
    reason="set B12X_RUN_W4A16_TOPK_SUM_OUTPUT_TEST=1 on a CUDA host",
)

TOPK = 16
HIDDEN = 1024
EXPERTS = 12
ROUTE_EXPERTS = 20


def _inputs(*, m: int, ids_dtype: torch.dtype, mapped: bool, broadcast: bool, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    device = torch.device("cuda", torch.cuda.current_device())
    fc2 = (torch.randn(m * TOPK, HIDDEN, generator=generator) * 2.0).to(
        torch.float16
    )
    weights = torch.rand(m * TOPK, generator=generator).to(torch.float32)
    route_count = ROUTE_EXPERTS if mapped else EXPERTS
    ids = torch.randint(0, route_count, (m * TOPK,), generator=generator)
    ids[::7] = -1  # invalid routes are skipped by the sum
    expert_map = None
    if mapped:
        expert_map = torch.full((ROUTE_EXPERTS + 1,), -1, dtype=torch.int32)
        perm = torch.randperm(ROUTE_EXPERTS, generator=generator)[:EXPERTS]
        expert_map[perm] = torch.arange(EXPERTS, dtype=torch.int32)
    svh_rows = 1 if broadcast else EXPERTS
    svh = (torch.randn(svh_rows, HIDDEN, generator=generator) * 0.5 + 1.0).to(
        torch.float16
    )
    return (
        fc2.to(device),
        weights.to(device),
        ids.to(dtype=ids_dtype, device=device),
        None if expert_map is None else expert_map.to(device),
        svh.reshape(-1).to(device) if broadcast else svh.to(device),
    )


def _launch(output: torch.Tensor, *, m: int, coupled: bool, tensors) -> None:
    from b12x.moe._shared.kernels.w4a16.kernel import _w4a16_topk_sum_launch_flat

    fc2, weights, ids, expert_map, svh = tensors
    _w4a16_topk_sum_launch_flat(
        fc2,
        output,
        m,
        TOPK,
        HIDDEN,
        "fp16",
        torch.cuda.current_stream().cuda_stream,
        full_rotation=True,
        coupled_hadamard=coupled,
        num_experts=EXPERTS,
        topk_weights=weights,
        route_expert_ids=ids,
        expert_map=expert_map,
        svh_table=svh,
    )


@pytest.mark.parametrize("m", [1, 4, 9])
@pytest.mark.parametrize("coupled", [False, True])
@pytest.mark.parametrize("broadcast", [False, True])
@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
def test_rotation_sum_in_kernel_rounding_matches_separate_cast(
    m: int, coupled: bool, broadcast: bool, mapped: bool, ids_dtype: torch.dtype
) -> None:
    if broadcast and not coupled:
        pytest.skip("the broadcast output scale is a coupled-Hadamard variant")
    seed = 20260908 + m * 10 + int(coupled) * 100 + int(broadcast) * 1000 + int(mapped)
    tensors = _inputs(m=m, ids_dtype=ids_dtype, mapped=mapped, broadcast=broadcast, seed=seed)
    device = tensors[0].device
    reference = torch.full((m, HIDDEN), float("nan"), dtype=torch.float32, device=device)
    _launch(reference, m=m, coupled=coupled, tensors=tensors)
    torch.cuda.synchronize()
    assert torch.isfinite(reference).all()
    for dtype in (torch.bfloat16, torch.float16):
        out = torch.full((m, HIDDEN), float("nan"), dtype=dtype, device=device)
        _launch(out, m=m, coupled=coupled, tensors=tensors)
        torch.cuda.synchronize()
        expected = reference.to(dtype)
        assert torch.equal(out.view(torch.int16), expected.view(torch.int16)), (
            f"{dtype} store differs from the separate cast "
            f"(m={m}, coupled={coupled}, broadcast={broadcast}, mapped={mapped})"
        )
        # A second launch through the same compiled variant is deterministic.
        again = torch.empty_like(out)
        _launch(again, m=m, coupled=coupled, tensors=tensors)
        torch.cuda.synchronize()
        assert torch.equal(again.view(torch.int16), out.view(torch.int16))


def test_run_w4a16_moe_rejects_an_output_dtype_off_the_setting(monkeypatch) -> None:
    from b12x.moe._shared.kernels.w4a16.host import (
        w4a16_topk_sum_rotation_output_torch_dtype,
    )

    monkeypatch.setenv("B12X_W4A16_TOPK_SUM_OUTPUT", "bf16")
    assert w4a16_topk_sum_rotation_output_torch_dtype() == torch.bfloat16
    monkeypatch.setenv("B12X_W4A16_TOPK_SUM_OUTPUT", "fp32")
    assert w4a16_topk_sum_rotation_output_torch_dtype() == torch.float32
