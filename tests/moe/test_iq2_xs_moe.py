"""Prepared IQ2_XS execution, mapped routes and graph reuse on CUDA."""

from pathlib import Path

import pytest

from b12x.moe.fused_moe import IQ2XSWeights
from b12x.preparation import PreparationSession
from benchmarks.iq2_xs_checkpoint import IQ2XSLayer
from benchmarks.benchmark_iq2_xs_moe import prepare_experts, qualify_capacity
from tests._reference.helpers import require_b12x
from .test_iq2_xs import blocks


@pytest.mark.parametrize("activation,mapped,route,capacity,counts", [
    ("silu", False, "packed", 16, (1, 3, 8, 16)),
    ("silu", True, "packed", 16, (1, 3, 8, 16)),
    ("silu", True, "direct", 8, (8,)),
    ("relu2", False, "packed", 16, (1, 3, 8, 16)),
    ("relu2", True, "packed", 16, (1, 3, 8, 16)),
])
def test_prepared_execution(activation, mapped, route, capacity, counts):
    device = require_b12x()
    ids = (7, 0, 5) if mapped else tuple(range(8))
    w13 = blocks(e=len(ids), n=512 if activation == "silu" else 256, k=256)
    w2 = blocks(e=len(ids), n=256, k=256)
    layer = IQ2XSLayer(IQ2XSWeights(w13, w2), 256, 256, 8, 2, ids, Path("synthetic-iq2-xs"), 0, 1, 0)
    experts, _ = prepare_experts(layer, device, activation=activation)
    with PreparationSession(device=device, autotune=False, compile_workers=1) as session:
        results = qualify_capacity(
            layer, experts, session, capacity=capacity, counts=counts,
            route_mode=route, activation=activation, patterns=("balanced", "hot"),
            repeats=1, launches=1,
        )
    assert len(results) == len(counts) * 2
