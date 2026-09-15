"""SM103 mHC preparation retains precision and validated launch geometry."""
from dataclasses import replace
import pytest
from b12x.norm import mhc
from b12x.norm.mhc._tuning import TUNING
from b12x.preparation import FrozenMapping
from tests.preparation.test_sm103_contracts import IDENTITY as B300

@pytest.mark.parametrize("hidden", [4096, 5120, 7168])
@pytest.mark.parametrize("capacity", [1, 8, 17, 389, 4096])
def test_sm103_mhc_preparation_and_candidates(hidden, capacity):
    plan = mhc.plan(mhc.Caps(device="cpu", max_tokens=capacity, hidden_size=hidden),
                    invocation=FrozenMapping({"operation": "post_pre", "has_norm_weight": True,
                                              "norm_eps": 1e-6, "rms_eps": 1e-6,
                                              "hc_eps": 1e-6, "sinkhorn_iters": 20}))
    assert plan.prepared is None
    query = replace(plan.query, smem_limit=227 * 1024)
    selected = TUNING.configure(query, device=B300)
    assert selected.default.projection_split_fp32
    for _, config in TUNING.iterate(selected):
        TUNING.validate_config(query, config, B300)
        assert config.projection_split_fp32

@pytest.mark.parametrize("precision", [False, 1, "true"])
def test_sm103_mhc_precision_override_fails_closed(precision):
    plan = mhc.plan(mhc.Caps(device="cpu", max_tokens=389, hidden_size=5120),
                    invocation=FrozenMapping({"operation": "post_pre", "has_norm_weight": True,
                                              "norm_eps": 1e-6, "rms_eps": 1e-6,
                                              "hc_eps": 1e-6, "sinkhorn_iters": 20}))
    query = replace(plan.query, smem_limit=227 * 1024)
    config = TUNING.configure(query, device=B300).default
    with pytest.raises((ValueError, TypeError)):
        TUNING.configure(query, device=B300,
                         override=replace(config, projection_split_fp32=precision))
