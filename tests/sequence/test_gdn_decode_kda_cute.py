"""Qualify the CuTe KDA backend through the public planned API."""

import pytest
import torch

from b12x.policy import GDN_ATTENTION, PolicyContext, PolicyMode
from b12x.sequence import gdn_decode as gdn
from . import test_gdn_decode_kda as contract
from ..conftest import require_sm103_or_sm12x


@pytest.fixture(autouse=True)
def cute_backend(monkeypatch):
    monkeypatch.setattr(contract, "require_sm120", require_sm103_or_sm12x)
    original = gdn.plan

    def plan(caps, *, policy=None):
        policy = policy or PolicyContext.for_device(
            caps.device, mode=PolicyMode.HEURISTIC_ONLY
        )
        config = policy._overrides.get(GDN_ATTENTION)
        return original(
            caps,
            policy=policy.with_override(
                GDN_ATTENTION,
                gdn.GdnConfig(
                    backend="cutedsl",
                    recurrent_block_v=32
                    if config is None
                    else config.recurrent_block_v,
                ),
            ),
        )

    monkeypatch.setattr(gdn, "plan", plan)


test_matches_reference = contract.test_packed_kda_matches_reference
test_live_tensors = contract.test_kda_binds_live_tensors_without_device_validation
test_bound_capacity = contract.test_kda_validation_uses_bound_tensor_capacity
test_live_graph = contract.test_live_kda_cuda_graph_replays_without_validation
test_glm53_geometry = contract.test_glm53_tp8_head_geometry_matches_reference
test_rejected_draft = contract.test_kda_rejected_draft_restarts_from_accepted_checkpoint
test_transaction = contract.test_kda_duplicate_state_slot_is_transactional
test_null_graph = contract.test_kda_null_state_sentinel_is_graph_safe_and_immutable
test_graph_addresses = contract.test_kda_cuda_graph_replay_preserves_addresses
test_torch_compile = contract.test_kda_torch_compile_fullgraph_keeps_outer_op_opaque
test_big_slot = contract.test_kda_padded_state_slot_past_int32_element_boundary


@pytest.mark.parametrize("index_dtype", (torch.int32, torch.int64))
@pytest.mark.parametrize("parameter_dtype", (torch.bfloat16, torch.float32))
@pytest.mark.parametrize("qk_l2norm", (False, True))
def test_strides_and_parameter_types(index_dtype, parameter_dtype, qk_l2norm):
    device = require_sm103_or_sm12x()
    binding = contract._make_case(device=device, qk_l2norm=qk_l2norm)
    beta = torch.empty(
        (binding.raw_beta.shape[0], binding.raw_beta.shape[1] * 3),
        device=device,
        dtype=torch.bfloat16,
    )[:, ::3]
    beta.copy_(binding.raw_beta)
    binding = contract._rebind(
        binding,
        raw_beta=beta,
        state_indices=binding.state_indices.to(index_dtype),
        A_log=binding.A_log.to(parameter_dtype),
        dt_bias=binding.dt_bias.to(parameter_dtype),
        norm_weight=binding.norm_weight.to(parameter_dtype),
    )
    state = binding.recurrent_state.clone()
    expected = contract._reference(binding, state)
    gdn.run_kda(binding)
    torch.testing.assert_close(binding.output, expected, atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(binding.recurrent_state, state, atol=2e-5, rtol=1e-5)


def test_frozen_resolution_reuses_callables_and_graph_with_live_counts(monkeypatch):
    from b12x._lib.runtime_control import (
        freeze_kernel_resolution,
        unfreeze_kernel_resolution,
    )
    from b12x.sequence.gdn_decode import _cute_kda

    device = require_sm103_or_sm12x()
    binding = contract._make_case(device=device)
    initial = binding.recurrent_state.clone()
    gdn.run_kda(binding)
    compiled = {key: tuple(map(id, value)) for key, value in _cute_kda._CACHE.items()}
    assert compiled

    def refuse(*args, **kwargs):
        pytest.fail("live counts caused CuTe kernel compilation")

    monkeypatch.setattr(_cute_kda, "b12x_compile", refuse)
    freeze_kernel_resolution("CuTe KDA live-count qualification")
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gdn.run_kda(binding)
        for starts, sequences, tokens in (
            ([0, 3, 4], 2, 4),
            ([0, 2, 3], 2, 3),
            ([0, 1, 1], 1, 1),
            ([0, 0, 0], 0, 0),
        ):
            binding.query_start_loc.copy_(
                torch.tensor(starts, device=device, dtype=torch.int32)
            )
            binding.num_seqs.fill_(sequences)
            binding.num_tokens.fill_(tokens)
            binding.num_accepted_tokens.fill_(1)
            binding.raw_g.copy_(torch.randn_like(binding.raw_g))
            state = initial.clone()
            expected = contract._reference(binding, state)
            binding.recurrent_state.copy_(initial)
            gdn.run_kda(binding)
            assert {
                key: tuple(map(id, value)) for key, value in _cute_kda._CACHE.items()
            } == compiled
            binding.recurrent_state.copy_(initial)
            binding.output.fill_(float("nan"))
            before = torch.cuda.memory_stats(device)
            graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats(device)
            assert (
                before["allocation.all.allocated"] == after["allocation.all.allocated"]
            )
            assert binding.error_code.item() == 0
            torch.testing.assert_close(binding.output, expected, atol=2e-2, rtol=1e-2)
            torch.testing.assert_close(
                binding.recurrent_state, state, atol=2e-5, rtol=1e-5
            )
    finally:
        unfreeze_kernel_resolution()


def test_smaller_bound_views_reuse_frozen_callables(monkeypatch):
    from b12x._lib.runtime_control import (
        freeze_kernel_resolution,
        unfreeze_kernel_resolution,
    )
    from b12x.sequence.gdn_decode import _cute_kda

    device = require_sm103_or_sm12x()
    binding = contract._make_case(device=device)
    gdn.run_kda(binding)
    compiled = {key: tuple(map(id, value)) for key, value in _cute_kda._CACHE.items()}

    def refuse(*args, **kwargs):
        pytest.fail("bound tensor capacities caused CuTe kernel compilation")

    monkeypatch.setattr(_cute_kda, "b12x_compile", refuse)
    freeze_kernel_resolution("CuTe KDA bound-capacity qualification")
    try:
        for tokens, seqs, columns in ((3, 2, 2), (1, 1, 1)):
            smaller = contract._rebind(
                binding,
                mixed_qkv=binding.mixed_qkv[:tokens],
                raw_g=binding.raw_g[:tokens],
                raw_beta=binding.raw_beta[:tokens],
                z=binding.z[:tokens],
                output=binding.output[:tokens],
                query_start_loc=binding.query_start_loc[: seqs + 1],
                num_accepted_tokens=binding.num_accepted_tokens[:seqs],
                state_indices=binding.state_indices[:seqs, :columns],
            )
            starts = [0, 2, 3] if seqs == 2 else [0, 1]
            smaller.query_start_loc.copy_(
                torch.tensor(starts, dtype=torch.int32, device=device)
            )
            smaller.num_accepted_tokens.fill_(1)
            smaller.num_tokens.fill_(tokens)
            smaller.num_seqs.fill_(seqs)
            state = smaller.recurrent_state.clone()
            expected = contract._reference(smaller, state)
            gdn.run_kda(smaller)
            assert {
                key: tuple(map(id, value)) for key, value in _cute_kda._CACHE.items()
            } == compiled
            torch.testing.assert_close(smaller.output, expected, atol=2e-2, rtol=1e-2)
            torch.testing.assert_close(
                smaller.recurrent_state, state, atol=2e-5, rtol=1e-5
            )
    finally:
        unfreeze_kernel_resolution()
