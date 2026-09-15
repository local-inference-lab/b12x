"""Qualify CuTe KDA with the prepared API's trusted metadata contract."""
from dataclasses import replace
import pytest
from b12x.preparation import FrozenMapping
from b12x.sequence import gdn_decode as gdn
from . import test_gdn_decode_kda as contract
from ..conftest import require_sm103_or_sm12x

# Preserve the production API's numerical, graph, padded-stride, and large-slot
# corpus while selecting the portable CuTe implementation during preparation.
for _name, _test in vars(contract).items():
    if _name.startswith("test_") and callable(_test):
        globals()[_name] = _test
_prepared_scopes = contract._prepared_scopes

@pytest.fixture(autouse=True)
def cute_backend(monkeypatch):
    monkeypatch.setattr(contract, "require_sm120", require_sm103_or_sm12x)
    original = gdn.plan
    def plan(caps, *, invocation=FrozenMapping(), override=None):
        config = gdn.GdnConfig(backend="cutedsl", recurrent_block_v=32) if override is None else replace(override, backend="cutedsl")
        return original(caps, invocation=invocation, override=config)
    monkeypatch.setattr(gdn, "plan", plan)


import torch
@pytest.mark.parametrize("recurrent_block_v", [16, 32])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_kda_tiles_reuse_compiled_kernels_with_mutable_live_counts(
    recurrent_block_v: int,
    state_dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.sequence.gdn_decode import _cute_kda

    device = require_sm103_or_sm12x()
    binding = contract._make_case(
        device=device,
        state_dtype=state_dtype,
        recurrent_block_v=recurrent_block_v,
    )
    initial_state = binding.recurrent_state.clone()
    gdn.run_kda(binding)
    state = binding.plan.prepared.state
    def program_ids():
        return (id(state.recurrent), id(state.norm))
    compiled = program_ids()
    assert compiled
    def refuse_compile(*args, **kwargs):
        pytest.fail("KDA live counts triggered compilation after warmup")
    monkeypatch.setattr(_cute_kda, "b12x_compile", refuse_compile)
    addresses = tuple(
        tensor.data_ptr()
        for tensor in (
            binding.scratch,
            binding.mixed_qkv,
            binding.raw_g,
            binding.raw_beta,
            binding.recurrent_state,
            binding.output,
        )
    )
    with kernel_resolution_guard("KDA tile live-count qualification"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gdn.run_kda(binding)
        for starts, sequences, tokens in (([0, 3, 4], 2, 4), ([0, 2, 3], 2, 3), ([0, 1, 1], 1, 1)):
            binding.query_start_loc.copy_(
                torch.tensor(starts, dtype=torch.int32, device=device)
            )
            binding.num_seqs.fill_(sequences)
            binding.num_tokens.fill_(tokens)
            binding.num_accepted_tokens.fill_(1)
            binding.mixed_qkv.copy_(torch.randn_like(binding.mixed_qkv).mul_(0.2))
            binding.raw_g.copy_(torch.randn_like(binding.raw_g).mul_(0.2))
            binding.raw_beta.copy_(torch.randn_like(binding.raw_beta).mul_(0.2))
            state_reference = initial_state.clone()
            expected = contract._reference(binding, state_reference)
            binding.recurrent_state.copy_(initial_state)
            gdn.run_kda(binding)
            assert program_ids() == compiled
            binding.recurrent_state.copy_(initial_state)
            binding.output.fill_(float("nan"))
            torch.cuda.synchronize(device)
            before = torch.cuda.memory_stats(device)
            graph.replay()
            torch.cuda.synchronize(device)
            after = torch.cuda.memory_stats(device)
            for key in ("allocation.all.allocated", "allocated_bytes.all.allocated"):
                assert before[key] == after[key]
            assert bool(torch.isfinite(binding.output).all())
            assert int(torch.count_nonzero(binding.output[:tokens])) > 0
            torch.testing.assert_close(binding.output, expected, rtol=1e-2, atol=2e-2)
            torch.testing.assert_close(
                binding.recurrent_state,
                state_reference,
                rtol=1e-2 if state_dtype == torch.bfloat16 else 1e-5,
                atol=8e-3 if state_dtype == torch.bfloat16 else 2e-5,
            )
        assert tuple(
            tensor.data_ptr()
            for tensor in (
                binding.scratch,
                binding.mixed_qkv,
                binding.raw_g,
                binding.raw_beta,
                binding.recurrent_state,
                binding.output,
            )
        ) == addresses
    graph.reset()


def test_kda_binds_live_tensors_within_planned_capacity():
    device = require_sm103_or_sm12x()
    binding = contract._make_case(device=device, query_lengths=(1, 1), columns=3,
                                   max_tokens=6, tensor_tokens=2, tensor_columns=1,
                                   noncontiguous_beta=True)
    with pytest.raises(ValueError, match="strides"):
        contract._rebind(binding, mixed_qkv=contract._row_padded(binding.mixed_qkv))
    args = {name: getattr(binding, name) for name in (
        "mixed_qkv", "raw_g", "raw_beta", "z", "A_log", "dt_bias", "norm_weight",
        "recurrent_state", "query_start_loc", "num_accepted_tokens", "state_indices",
        "num_seqs", "num_tokens", "output",
    )}
    for name in ("mixed_qkv", "raw_g", "z", "output"):
        args[name] = contract._row_padded(args[name])
    binding = contract._prepare_binding(binding._state.caps, args)
    state = binding.recurrent_state.clone()
    expected = contract._reference(binding, state)
    gdn.run_kda(binding)
    torch.testing.assert_close(binding.output, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(binding.recurrent_state, state, rtol=1e-5, atol=2e-5)
