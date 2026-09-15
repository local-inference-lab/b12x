"""MTP declarations budget every stream without compiling kernels."""
import math
import pytest
from b12x.sequence import mtp_feedback as mtp
from b12x.sequence.mtp_feedback import _fp8
from tests.preparation.test_sm103_contracts import IDENTITY as B300, DEVICE


@pytest.mark.parametrize("hidden,streams", [(128, 1), (256, 3), (5120, 4)])
@pytest.mark.parametrize("capacity", [1, 17, 4096, 65537])
def test_fp8_feedback_declaration_owns_stream_capacity(hidden, streams, capacity, monkeypatch):
    monkeypatch.setattr(_fp8, "compile_norm", lambda *a: pytest.fail("declaration compiled"))
    declaration = mtp.plan(mtp.Caps(device="cuda:0", max_tokens=capacity,
                                  hidden_size=hidden, streams=streams, contract="rms_streams_fp8"))
    configured = declaration.contract.configure(declaration.query, device=B300)
    assert declaration.prepared is None and configured.default.norm_block_s == 1
    metadata = _fp8.plan(mtp.Caps(device="cuda:0", max_tokens=capacity, hidden_size=hidden,
                                streams=streams, contract="rms_streams_fp8"),
                         configured.default, B300, compile_launches=False)
    assert metadata.state_projection_rows == capacity * streams
    end = 0
    for offset, shape, dtype in metadata._backend_plan.layout.values():
        assert offset % 1024 == 0 and offset >= end
        end = offset + math.prod(shape) * dtype.itemsize
    assert declaration._memory_requirements(configured.default, DEVICE).scratch_nbytes >= end
    assert metadata._backend_plan.layout["state_quant"][1] == (capacity * streams, hidden)
    for _, config in declaration.contract.iterate(configured):
        assert config.norm_block_s == 1


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
