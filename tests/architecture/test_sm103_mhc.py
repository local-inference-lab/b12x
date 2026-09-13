"""Host policy and capacity contracts for the SM103 mHC execution path."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from b12x._lib import gating
from b12x._lib.architecture import require_kernel_architecture
from b12x.norm import mhc
from b12x.norm.mhc._policy import MHC_POLICY, MhcQuery
from b12x.policy import DeviceIdentity, PolicyContext, PolicySource
from b12x.policy.generation.providers.norm_sequence import _MhcSession, _mhc_cases

B300 = DeviceIdentity(
    vendor="nvidia",
    product_name="NVIDIA B300",
    compute_capability=(10, 3),
    sm_count=148,
)


@pytest.mark.parametrize("hidden", [4096, 5120, 7168])
@pytest.mark.parametrize("capacity", [1, 8, 17, 389, 4096])
def test_sm103_mhc_public_plan_and_generator(hidden, capacity, monkeypatch):
    context = PolicyContext.for_identity(B300)
    monkeypatch.setattr(PolicyContext, "require_device", lambda *args: None)
    monkeypatch.setattr(gating, "get_compute_capability", lambda *args: (10, 3))
    monkeypatch.setattr(gating, "has_cutlass_dsl", lambda: True)
    assert mhc.is_supported(torch.device("cuda:0"))
    plan = mhc.plan(
        mhc.Caps(device="cuda:0", max_tokens=capacity, hidden_size=hidden),
        policy=context,
    )
    assert plan.policy_resolution.source is PolicySource.HEURISTIC
    assert plan.config.projection_split_fp32
    assert plan.config.backend == ("tf32_tma" if capacity >= 384 else "native")
    assert plan.scratch_specs()[0].nbytes >= capacity * (hidden // 64) * 25 * 4
    for module in ("b12x.norm.mhc._kernels", "b12x.norm.mhc._pre_prefill"):
        require_kernel_architecture(module, (10, 3))
    cases = [c for c in _mhc_cases() if c.query["hidden_size"] == hidden]
    for case in cases:
        candidates = _MhcSession(SimpleNamespace(device=B300)).candidates(case)
        assert candidates
        for candidate in candidates:
            config = MHC_POLICY.decode_profile(candidate.config)
            MHC_POLICY.validate_config(MhcQuery(**case.query.to_dict()), config, B300)


@pytest.mark.parametrize(
    "changes", [dict(hidden_size=16), dict(split_k=1), dict(max_tokens=65536)]
)
def test_sm103_mhc_rejects_unimplemented_geometry(changes):
    query = replace(
        MhcQuery(dtype="bfloat16", max_tokens=8, hidden_size=5120, split_k=80),
        **changes,
    )
    with pytest.raises(ValueError):
        PolicyContext.for_identity(B300).resolve(MHC_POLICY, query)


@pytest.mark.parametrize("precision", [False, 1, "true"])
def test_sm103_mhc_precision_override_fails_closed(precision):
    query = MhcQuery(dtype="bfloat16", max_tokens=389, hidden_size=5120, split_k=80)
    config = replace(MHC_POLICY.heuristic(query, B300), projection_split_fp32=precision)
    with pytest.raises(ValueError):
        PolicyContext.for_identity(B300).resolve(MHC_POLICY, query, override=config)


def test_plan_validates_environment_enabled_projection(monkeypatch):
    from b12x.norm.mhc._policy import schedule_for

    query = MhcQuery(dtype="bfloat16", max_tokens=8, hidden_size=4096, split_k=64)
    config = replace(MHC_POLICY.heuristic(query, B300), projection_tile_m=1)
    monkeypatch.setenv("B12X_MHC_PREFILL_TF32_MMA", "1")
    with pytest.raises(ValueError, match="projection_tile_m"):
        schedule_for(query, config, B300)


@pytest.mark.parametrize(
    "hidden,splits,paired", [(5120, 8, False), (7168, 8, False), (5120, 4, True)]
)
def test_plan_rejects_partial_decode_thread_blocks(hidden, splits, paired, monkeypatch):
    from b12x.norm.mhc._policy import schedule_for

    query = MhcQuery(
        dtype="bfloat16", max_tokens=8, hidden_size=hidden, split_k=hidden // 64
    )
    monkeypatch.setenv("B12X_MHC_DECODE_SPLITS", str(splits))
    monkeypatch.setenv("B12X_MHC_DECODE_BF16X2", str(int(paired)))
    with pytest.raises(ValueError, match="divisible by"):
        schedule_for(query, MHC_POLICY.heuristic(query, B300), B300)
