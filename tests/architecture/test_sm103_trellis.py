"""Construction and capacity boundaries for the SM103 Trellis primitives."""

import pytest

from b12x._lib.architecture import UnsupportedArchitectureError
from b12x.moe._shared.kernels.sm103.trellis import (
    TrellisPipeline,
    ReconstructTrellisTiles,
)
from b12x.moe._shared.kernels.sm103.trellis_gemm import RoutedTrellisGemm


@pytest.mark.parametrize(
    "stage,n,k", [("gate", 2304, 5120), ("up", 2304, 5120), ("down", 5120, 2304)]
)
def test_weight_plan_projection_geometry(stage, n, k):
    pipeline = TrellisPipeline(5120, 2304, 384, "mcg", True)
    projection = pipeline.projection(projection=stage, bits=3, capacity=128)
    assert (projection.n, projection.k, projection.experts, projection.capacity) == (
        n,
        k,
        384,
        128,
    )
    with pytest.raises(UnsupportedArchitectureError, match="mixed-rate MoE"):
        pipeline.require_execution()


@pytest.mark.parametrize(
    "overrides",
    [
        {"n": 129},
        {"k": 65},
        {"n": 0},
        {"experts": 0},
        {"capacity": 0},
        {"capacity": 2**31},
        {"bits": 1},
        {"codebook": "sqg_e4m3", "bits": 5},
        {"codebook": "sqg_fp16", "bits": 3},
        {"codebook": "unknown"},
    ],
)
def test_projection_rejects_invalid_contract(overrides):
    kwargs = dict(n=144, k=80, experts=3, capacity=17, bits=3, codebook="mcg")
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        RoutedTrellisGemm(**kwargs)


def test_mcg_decoder_preserves_existing_checkpoint_codebook():
    pipeline = TrellisPipeline(5120, 2304, 384, "mcg", False)
    decoder = pipeline.reconstruction(projection="w2", bits=5)
    assert decoder.bits == 5 and decoder.codebook == "mcg"
    with pytest.raises(ValueError, match="positive capacity"):
        ReconstructTrellisTiles(3, 0)
    with pytest.raises(ValueError, match="gate, up, or down"):
        pipeline.projection(projection="w13", bits=3, capacity=1)
