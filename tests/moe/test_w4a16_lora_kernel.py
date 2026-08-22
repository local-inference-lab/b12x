from __future__ import annotations

import pytest
import torch

from b12x.moe._shared.kernels.w4a16.lora_kernel import (
    run_w4a16_static_lora_output_sum,
    run_w4a16_static_lora_projection,
    run_w4a16_static_lora_split_w13_projection,
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
    token_row_divisor: int | None = None,
    token_lora_mapping: torch.Tensor | None = None,
    adapter_slot: int = 0,
    route_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    ranks: list[torch.Tensor] = []
    rows: list[torch.Tensor] = []
    token_divisor = (
        input_row_divisor if token_row_divisor is None else token_row_divisor
    )
    for route, expert in enumerate(expert_ids.cpu().tolist()):
        input_row = route // input_row_divisor
        active = token_lora_mapping is None or int(
            token_lora_mapping[route // token_divisor].item()
        ) == int(adapter_slot)
        rank_value = (
            (adapter_a[expert].float() @ x[input_row].float())
            if active
            else torch.zeros(4, dtype=torch.float32, device=x.device)
        ).to(torch.bfloat16)
        delta = adapter_b[expert].float() @ rank_value.float()
        multiplier = float(scale)
        if route_weights is not None:
            multiplier *= float(route_weights[route].item())
        ranks.append(rank_value)
        rows.append((destination[route].float() + delta * multiplier).to(torch.bfloat16))
    return torch.stack(rows), torch.stack(ranks)


def _output_sum_oracle(
    x: torch.Tensor,
    adapter_a: torch.Tensor,
    adapter_b: torch.Tensor,
    expert_ids: torch.Tensor,
    route_weights: torch.Tensor,
    destination: torch.Tensor,
    *,
    scale: float,
    topk: int,
    token_lora_mapping: torch.Tensor | None = None,
    adapter_slot: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    ranks: list[torch.Tensor] = []
    rows: list[torch.Tensor] = []
    for token in range(int(destination.shape[0])):
        delta = torch.zeros(
            int(destination.shape[1]), dtype=torch.float32, device=x.device
        )
        active = token_lora_mapping is None or int(
            token_lora_mapping[token].item()
        ) == int(adapter_slot)
        for route_in_token in range(topk):
            route = token * topk + route_in_token
            expert = int(expert_ids[route].item())
            rank_value = (
                (adapter_a[expert].float() @ x[route].float())
                if active
                else torch.zeros(4, dtype=torch.float32, device=x.device)
            ).to(torch.bfloat16)
            ranks.append(rank_value)
            delta += (
                adapter_b[expert].float()
                @ rank_value.float()
                * float(route_weights[route].item())
            )
        rows.append(
            (destination[token].float() + delta * float(scale)).to(torch.bfloat16)
        )
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_w4a16_static_lora_projection_graph_replays_live_token_mapping() -> None:
    torch.manual_seed(20260822)
    device = torch.device("cuda")
    experts, tokens, topk = 5, 6, 2
    width, output_width = 257, 193
    routes = tokens * topk
    # FC2 geometry: one activation row per route, but adapter selection remains
    # one entry per original token.
    x = (torch.randn(routes, width, device=device) * 0.05).to(torch.bfloat16)
    adapter_a = (torch.randn(experts, 4, width, device=device) * 0.03).to(
        torch.bfloat16
    )
    adapter_b = (torch.randn(experts, output_width, 4, device=device) * 0.04).to(
        torch.bfloat16
    )
    expert_ids = torch.randint(0, experts, (routes,), device=device, dtype=torch.int32)
    base = (torch.randn(routes, output_width, device=device) * 0.1).to(torch.bfloat16)
    rank_scratch = torch.empty(routes, 4, dtype=torch.bfloat16, device=device)
    destination = base.clone()
    token_mapping = torch.tensor(
        [0, -1, 0, -1, 0, -1],
        dtype=torch.int32,
        device=device,
    )

    def run() -> torch.Tensor:
        return run_w4a16_static_lora_projection(
            x,
            adapter_a,
            adapter_b,
            expert_ids,
            destination,
            rank_scratch,
            scale=0.5,
            input_row_divisor=1,
            token_row_divisor=topk,
            token_lora_mapping=token_mapping,
            adapter_slot=0,
        )

    expected, expected_rank = _projection_oracle(
        x,
        adapter_a,
        adapter_b,
        expert_ids,
        base,
        scale=0.5,
        input_row_divisor=1,
        token_row_divisor=topk,
        token_lora_mapping=token_mapping,
        adapter_slot=0,
        route_weights=None,
    )
    run()
    torch.cuda.synchronize()
    torch.testing.assert_close(rank_scratch, expected_rank, rtol=0.0, atol=0.0078125)
    torch.testing.assert_close(destination, expected, rtol=0.0, atol=0.0078125)

    graph = torch.cuda.CUDAGraph()
    destination.copy_(base)
    with torch.cuda.graph(graph):
        run()

    # Change only live metadata after capture. The same graph must now apply
    # LoRA to the complementary tokens.
    token_mapping.mul_(-1).sub_(1)
    destination.copy_(base)
    expected_flipped, expected_rank_flipped = _projection_oracle(
        x,
        adapter_a,
        adapter_b,
        expert_ids,
        base,
        scale=0.5,
        input_row_divisor=1,
        token_row_divisor=topk,
        token_lora_mapping=token_mapping,
        adapter_slot=0,
        route_weights=None,
    )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        rank_scratch,
        expected_rank_flipped,
        rtol=0.0,
        atol=0.0078125,
    )
    torch.testing.assert_close(
        destination,
        expected_flipped,
        rtol=0.0,
        atol=0.0078125,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_w4a16_static_lora_split_w13_matches_packed_oracle_and_graph() -> None:
    torch.manual_seed(20260824)
    device = torch.device("cuda")
    experts, tokens, topk = 5, 7, 2
    width, intermediate = 257, 193
    routes = tokens * topk
    x = (torch.randn(tokens, width, device=device) * 0.05).to(torch.bfloat16)
    adapter_a = (torch.randn(experts, 4, width, device=device) * 0.03).to(
        torch.bfloat16
    )
    gate_b = (torch.randn(experts, intermediate, 4, device=device) * 0.04).to(
        torch.bfloat16
    )
    up_b = (torch.randn(experts, intermediate, 4, device=device) * 0.04).to(
        torch.bfloat16
    )
    packed_b = torch.cat((gate_b, up_b), dim=1)
    expert_ids = torch.randint(0, experts, (routes,), device=device, dtype=torch.int32)
    base = (torch.randn(routes, 2 * intermediate, device=device) * 0.1).to(
        torch.bfloat16
    )
    rank_scratch = torch.empty(routes, 4, dtype=torch.bfloat16, device=device)
    destination = base.clone()
    token_mapping = torch.tensor(
        [0, -1, 0, -1, 0, -1, 0], dtype=torch.int32, device=device
    )
    expected, expected_rank = _projection_oracle(
        x,
        adapter_a,
        packed_b,
        expert_ids,
        base,
        scale=0.625,
        input_row_divisor=topk,
        token_lora_mapping=token_mapping,
        adapter_slot=0,
    )

    def run() -> torch.Tensor:
        return run_w4a16_static_lora_split_w13_projection(
            x,
            adapter_a,
            gate_b,
            up_b,
            expert_ids,
            destination,
            rank_scratch,
            scale=0.625,
            input_row_divisor=topk,
            token_lora_mapping=token_mapping,
            adapter_slot=0,
        )

    run()
    torch.cuda.synchronize()
    torch.testing.assert_close(rank_scratch, expected_rank, rtol=0.0, atol=0.0078125)
    torch.testing.assert_close(destination, expected, rtol=0.0, atol=0.0078125)

    destination.copy_(base)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    token_mapping.fill_(-1)
    destination.copy_(base)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(destination, base)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_w4a16_static_lora_output_sum_matches_oracle_and_live_graph() -> None:
    torch.manual_seed(20260823)
    device = torch.device("cuda")
    experts, tokens, topk = 5, 4, 3
    width, output_width = 257, 193
    routes = tokens * topk
    x = (torch.randn(routes, width, device=device) * 0.05).to(torch.bfloat16)
    adapter_a = (torch.randn(experts, 4, width, device=device) * 0.03).to(
        torch.bfloat16
    )
    adapter_b = (torch.randn(experts, output_width, 4, device=device) * 0.04).to(
        torch.bfloat16
    )
    expert_ids = torch.randint(0, experts, (routes,), device=device, dtype=torch.int32)
    route_weights = torch.softmax(
        torch.randn(tokens, topk, device=device, dtype=torch.float32), dim=-1
    ).view(-1)
    base = (torch.randn(tokens, output_width, device=device) * 0.1).to(torch.bfloat16)
    destination = base.clone()
    rank_scratch = torch.empty(routes, 4, dtype=torch.bfloat16, device=device)
    token_mapping = torch.tensor([0, -1, 0, -1], dtype=torch.int32, device=device)

    def run() -> torch.Tensor:
        return run_w4a16_static_lora_output_sum(
            x,
            adapter_a,
            adapter_b,
            expert_ids,
            route_weights,
            destination,
            rank_scratch,
            scale=0.375,
            topk=topk,
            token_lora_mapping=token_mapping,
            adapter_slot=0,
        )

    expected, expected_rank = _output_sum_oracle(
        x,
        adapter_a,
        adapter_b,
        expert_ids,
        route_weights,
        base,
        scale=0.375,
        topk=topk,
        token_lora_mapping=token_mapping,
        adapter_slot=0,
    )
    run()
    torch.cuda.synchronize()
    torch.testing.assert_close(rank_scratch, expected_rank, rtol=0.0, atol=0.0078125)
    torch.testing.assert_close(destination, expected, rtol=0.0, atol=0.0078125)

    destination.copy_(base)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    token_mapping.mul_(-1).sub_(1)
    destination.copy_(base)
    expected_flipped, expected_rank_flipped = _output_sum_oracle(
        x,
        adapter_a,
        adapter_b,
        expert_ids,
        route_weights,
        base,
        scale=0.375,
        topk=topk,
        token_lora_mapping=token_mapping,
        adapter_slot=0,
    )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        rank_scratch, expected_rank_flipped, rtol=0.0, atol=0.0078125
    )
    torch.testing.assert_close(
        destination, expected_flipped, rtol=0.0, atol=0.0078125
    )
