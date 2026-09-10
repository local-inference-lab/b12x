"""GPU checks for the FlashInfer race's raw-projection preparation boundary."""

import pytest
import torch

from b12x.policy.generation.delta_prefill_cases import PrefillCase, assert_close, make_inputs, oracle
from ..conftest import require_b12x


@pytest.mark.parametrize("use_cp", ["auto", False])
def test_scalar_gate_preparation_retains_small_softplus(use_cp):
    pytest.importorskip("flashinfer")
    from benchmarks._gdn_prefill_flashinfer import FlashInferArm

    case = PrefillCase("gdn", 2, 6, (65,))
    tensors = make_inputs(case, device=require_b12x())
    tensors["raw_g"].fill_(-20)
    tensors["A_log"].fill_(20)
    tensors["dt_bias"].zero_()
    initial = tensors["recurrent_state"].clone()
    expected, pool = oracle(case, tensors)
    arm = FlashInferArm(case, tensors, use_cp=use_cp)
    arm()
    assert_close("output", tensors["output"], expected, ratio=1e-2)
    assert_close("final state", tensors["recurrent_state"][1], pool[1], ratio=5e-3)
    torch.testing.assert_close(tensors["recurrent_state"][[0, 2, 3]], initial[[0, 2, 3]], rtol=0, atol=0)
