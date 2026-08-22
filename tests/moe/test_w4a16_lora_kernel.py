from __future__ import annotations

import pytest
import torch

from b12x.moe._shared.kernels.w4a16.lora_kernel import (
    run_w4a16_static_lora_projection,
)


def _projection_oracle(
    x: torch.Tensor,
    adapter_a: torch.Tensor,
    adapter_b: torch.Tensor,
    expert_ids: torch.Tensor,
    destination: torch.Tensor,
    *,
    scale: float,
    input_row_divisor: int,
    route_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    ranks: list[torch.Tensor] = []
    rows: list[torch.Tensor] = []
    for route, expert in enumerate(expert_ids.cpu().tolist()):
        input_row = route // input_row_divisor
        rank_value = (
            adapter_a[expert].float() @ x[input_row].float()
        ).to(torch.bfloat16)
        delta = adapter_b[expert].float() @ rank_value.float()
        multiplier = float(scale)
        if route_weights is not None:
            multiplier *= float(route_weights[route].item())
        ranks.append(rank_value)
        rows.append((destination[route].float() + delta * multiplier).to(torch.bfloat16))
    return torch.stack(rows), torch.stack(ranks)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("input_rows", "input_row_divisor", "with_route_weights"),
    [(7, 2, False), (14, 1, True)],
)
def test_w4a16_static_lora_projection_matches_oracle_eager_and_graph(
    input_rows: int,
    input_row_divisor: int,
    with_route_weights: bool,
) -> None:
    torch.manual_seed(20260821 + input_row_divisor)
    device = torch.device("cuda")
    experts, width, output_width = 5, 2113, 769
    routes = input_rows * input_row_divisor
    x = (torch.randn(input_rows, width, device=device) * 0.05).to(torch.bfloat16)
    adapter_a = (torch.randn(experts, 4, width, device=device) * 0.03).to(
        torch.bfloat16
    )
    adapter_b = (torch.randn(experts, output_width, 4, device=device) * 0.04).to(
        torch.bfloat16
    )
    expert_ids = torch.randint(0, experts, (routes,), device=device, dtype=torch.int32)
    base = (torch.randn(routes, output_width, device=device) * 0.1).to(torch.bfloat16)
    route_weights = (
        torch.softmax(torch.randn(routes, device=device), dim=0)
        if with_route_weights
        else None
    )
    rank_scratch = torch.empty(routes, 4, dtype=torch.bfloat16, device=device)
    destination = base.clone()
    expected, expected_rank = _projection_oracle(
        x,
        adapter_a,
        adapter_b,
        expert_ids,
        base,
        scale=0.375,
        input_row_divisor=input_row_divisor,
        route_weights=route_weights,
    )

    run_w4a16_static_lora_projection(
        x,
        adapter_a,
        adapter_b,
        expert_ids,
        destination,
        rank_scratch,
        scale=0.375,
        input_row_divisor=input_row_divisor,
        route_weights=route_weights,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(rank_scratch, expected_rank, rtol=0.0, atol=0.0078125)
    torch.testing.assert_close(destination, expected, rtol=0.0, atol=0.0078125)

    destination.copy_(base)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_w4a16_static_lora_projection(
            x,
            adapter_a,
            adapter_b,
            expert_ids,
            destination,
            rank_scratch,
            scale=0.375,
            input_row_divisor=input_row_divisor,
            route_weights=route_weights,
        )
    destination.copy_(base)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(rank_scratch, expected_rank, rtol=0.0, atol=0.0078125)
    torch.testing.assert_close(destination, expected, rtol=0.0, atol=0.0078125)

    # Prove replay reads the live adapter storage rather than retaining a
    # capture-time result or hidden packed copy.
    adapter_b.zero_()
    destination.copy_(base)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(destination, base)
