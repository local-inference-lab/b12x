"""Host capacity and policy contracts for per-stream FP8 MTP feedback."""

import math
from types import SimpleNamespace

import pytest
import torch

from b12x._lib.architecture import require_kernel_architecture
from b12x.policy import DeviceIdentity, PolicyContext, PolicySource
from b12x.policy.generation.providers.norm_sequence import (
    _MtpFeedbackSession,
    _mtp_feedback_cases,
)
from b12x.sequence import mtp_feedback as mtp
from b12x.sequence.mtp_feedback import _fp8
from b12x.sequence.mtp_feedback._policy import MTP_FEEDBACK_POLICY, MtpFeedbackQuery


B300 = DeviceIdentity(
    vendor="nvidia",
    product_name="NVIDIA B300",
    compute_capability=(10, 3),
    sm_count=148,
)


@pytest.mark.parametrize("hidden,streams", [(128, 1), (256, 3), (5120, 4)])
@pytest.mark.parametrize("capacity", [1, 17, 4096, 65537])
def test_fp8_feedback_public_plan_owns_all_stream_capacity(
    hidden, streams, capacity, monkeypatch
):
    calls = []

    def record(*args):
        calls.append(args)
        return object()

    monkeypatch.setattr(PolicyContext, "require_device", lambda *args: None)
    monkeypatch.setattr(_fp8, "compile_norm", record)
    monkeypatch.setattr(_fp8, "compile_projection", record)
    monkeypatch.setattr(_fp8, "compile_aux", lambda *args: (object(), object()))
    plan = mtp.plan(
        mtp.Caps(
            device="cuda:0",
            max_tokens=capacity,
            hidden_size=hidden,
            streams=streams,
            contract="rms_streams_fp8",
        ),
        policy=PolicyContext.for_identity(B300),
    )
    assert plan.policy_resolution.source is PolicySource.HEURISTIC
    assert (
        plan.output_shape(3) == (3, streams, hidden)
        if capacity >= 3
        else plan.output_shape() == (1, streams, hidden)
    )
    assert plan.state_projection_rows == capacity * streams
    assert calls[-1] == (hidden, hidden, 1, "bfloat16", True, True, 0, 148, "sm_103a")
    assert {call[3] for call in calls[:2]} == {torch.int32, torch.int64}
    layout = plan._backend_plan.layout
    end = 0
    for offset, shape, dtype in layout.values():
        assert offset % 1024 == 0
        assert offset >= end
        end = offset + math.prod(shape) * dtype.itemsize
    assert plan.scratch_specs()[0].nbytes >= end
    assert layout["state_norm"][0] == math.prod(layout["embedding_norm"][1]) * 2
    assert layout["state_quant"][1] == (capacity * streams, hidden)
    for module in ("b12x.sequence.mtp_feedback._fp8", "b12x._lib.fp8_gemm"):
        require_kernel_architecture(module, (10, 3))


@pytest.mark.parametrize(
    "changes",
    [
        {"hidden_size": 192},
        {"hidden_size": 16512},
        {"streams": 17},
        {"max_tokens": 2**31 // 4},
    ],
)
def test_fp8_feedback_rejects_unsupported_geometry(changes):
    kwargs = dict(
        device="cuda:0",
        hidden_size=128,
        streams=4,
        max_tokens=17,
        contract="rms_streams_fp8",
    )
    with pytest.raises(ValueError, match="FP8 stream feedback"):
        mtp.Caps(**(kwargs | changes))


def test_fp8_feedback_generator_qualifies_ordinary_stream_norms():
    cases = [
        case
        for case in _mtp_feedback_cases()
        if case.query["contract"] == "rms_streams_fp8"
    ]
    assert cases
    for case in cases:
        candidates = _MtpFeedbackSession(SimpleNamespace(device=B300)).candidates(case)
        assert candidates
        for candidate in candidates:
            config = MTP_FEEDBACK_POLICY.decode_profile(candidate.config)
            assert config.norm_block_s == 1
            MTP_FEEDBACK_POLICY.validate_config(
                MtpFeedbackQuery(**case.query.to_dict()), config, B300
            )
