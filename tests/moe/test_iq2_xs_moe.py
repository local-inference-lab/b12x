"""Prepared IQ2_XS execution, mapped routes and graph reuse on CUDA."""

from pathlib import Path

import pytest

from b12x.moe.fused_moe import FC2Invocation, IQ2XSWeights, plan_fc2
from b12x.preparation import PreparationSession
from benchmarks.iq2_xs_checkpoint import IQ2XSLayer
from benchmarks.benchmark_iq2_xs_moe import prepare_experts, qualify_capacity
from tests._reference.helpers import require_b12x
from .test_iq2_xs import blocks


@pytest.mark.parametrize(
    "hidden_size,intermediate_size", [(256, 256), (2048, 512)]
)
@pytest.mark.parametrize(
    "activation,mapped,route,capacity,counts",
    [
        ("silu", False, "packed", 16, (1, 3, 8, 16)),
        ("silu", True, "packed", 16, (1, 3, 8, 16)),
        ("silu", True, "direct", 8, (8,)),
        ("relu2", False, "packed", 16, (1, 3, 8, 16)),
        ("relu2", True, "packed", 16, (1, 3, 8, 16)),
    ],
)
def test_prepared_execution(
    activation, mapped, route, capacity, counts, hidden_size, intermediate_size
):
    device = require_b12x()
    ids = (7, 0, 5) if mapped else tuple(range(8))
    w13 = blocks(
        e=len(ids), n=intermediate_size * (2 if activation == "silu" else 1),
        k=hidden_size,
    )
    w2 = blocks(e=len(ids), n=hidden_size, k=intermediate_size)
    layer = IQ2XSLayer(
        IQ2XSWeights(w13, w2), hidden_size, intermediate_size, 8, 2, ids,
        Path("synthetic-iq2-xs"), 0, 1, 0
    )
    experts, _ = prepare_experts(layer, device, activation=activation)
    with pytest.raises(NotImplementedError, match="standalone IQ2_XS FC2"):
        plan_fc2(experts=experts, invocation=FC2Invocation(max_routes=capacity * 2))
    with PreparationSession(
        device=device, autotune=False, compile_workers=1
    ) as session:
        results = qualify_capacity(
            layer,
            experts,
            session,
            capacity=capacity,
            counts=counts,
            route_mode=route,
            activation=activation,
            patterns=("balanced", "hot"),
            repeats=1,
            launches=1,
        )
    assert len(results) == len(counts) * 2
