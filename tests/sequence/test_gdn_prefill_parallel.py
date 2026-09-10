"""Segment-parallel GDN qualification against the sequential contract oracle."""

from __future__ import annotations

import pytest
import torch

from b12x.policy.generation.delta_prefill_cases import (
    PrefillCase,
    check_binding,
    make_binding,
    make_inputs,
    oracle,
)
from b12x.sequence import gdn_prefill as gdn
from b12x.sequence._shared.delta_prefill.workspace import tiles_capacity

from ..conftest import require_b12x


def _config(tokens: int, seqs: int, segment_tokens: int, stages: int = 3):
    segments = (tokens + segment_tokens - 1) // segment_tokens + seqs
    return {
        "backend": "cutedsl",
        "v_split": 64,
        "k_split": 1,
        "stages": stages,
        "window_tiles": tiles_capacity(tokens, segments),
        "algorithm": "chunk_parallel",
        "segment_tokens": segment_tokens,
    }


@pytest.mark.parametrize("segment_tokens", [128, 256, 512, 1024])
@pytest.mark.parametrize("profile", ["random", "weak", "strong", "alternating"])
@pytest.mark.parametrize("stages", [2, 3])
def test_gpu_parallel_segment_boundaries_and_checkpoints(segment_tokens, profile, stages):
    case = PrefillCase("gdn", 1, 3, (2 * segment_tokens + 17, 0, 33))
    tensors = make_inputs(case, device=require_b12x())
    if profile == "weak":
        tensors["raw_g"].fill_(-12)
    elif profile == "strong":
        tensors["raw_g"].fill_(10)
        tensors["raw_g"][::16].fill_(1e10)
    elif profile == "alternating":
        tensors["k"][1:] = tensors["k"][:1].clone()
        tensors["k"][1::2].neg_()
        tensors["raw_g"].fill_(-12)
        tensors["raw_beta"].fill_(12)
    tensors["checkpoint_offsets"][0] = segment_tokens + 16
    tensors["checkpoint_offsets"][2] = 16
    initial = tensors["recurrent_state"].clone()
    expected, state = oracle(case, tensors)
    binding = make_binding(
        case, tensors, checkpoint_export=True,
        config=_config(tensors["q"].shape[0], len(case.lengths), segment_tokens, stages),
    )
    gdn.prewarm(binding)
    binding.scratch.fill_(0xFF)
    gdn.run(binding)
    check_binding(case, binding, expected, state, initial)


@pytest.mark.parametrize("v_split,k_split", [(16, 1), (32, 1), (128, 1), (64, 2), (32, 4)])
def test_gpu_parallel_dependency_groups_preserve_checkpoint_state(v_split, k_split):
    case = PrefillCase("gdn", 1, 3, (529, 0, 33))
    tensors = make_inputs(case, device=require_b12x())
    tensors["checkpoint_offsets"][0] = 384
    tensors["checkpoint_offsets"][2] = 16
    initial = tensors["recurrent_state"].clone()
    expected, state = oracle(case, tensors)
    config = {**_config(case.tokens, len(case.lengths), 256, stages=2),
              "v_split": v_split, "k_split": k_split}
    binding = make_binding(case, tensors, checkpoint_export=True, config=config)
    gdn.prewarm(binding)
    binding.scratch.fill_(0xFF)
    gdn.run(binding)
    assert binding.parallel.plan.reuse_outputs == (k_split == 1)
    check_binding(case, binding, expected, state, initial)


@pytest.mark.parametrize("stages", [2, 3])
def test_gpu_parallel_graph_reuses_live_worklists_and_storage(stages):
    from b12x._lib.runtime_control import (
        freeze_kernel_resolution,
        unfreeze_kernel_resolution,
    )

    device = require_b12x()
    capacity, seq_capacity = 1024, 4
    case = PrefillCase("gdn", 2, 6, (1,))
    tensors = make_inputs(case, device=device, max_tokens=capacity, max_seqs=seq_capacity)
    binding = make_binding(
        case, tensors, checkpoint_export=True,
        config=_config(capacity, seq_capacity, 256, stages),
    )
    gdn.prewarm(binding)
    gdn.run(binding)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gdn.run(binding)
    addresses = tuple(t.data_ptr() for t in (binding.output, binding.scratch, binding.recurrent_state))
    freeze_kernel_resolution("GDN segment-parallel runtime worklists")
    try:
        for lengths in ((1024,), (129, 257, 0, 33), (0, 0, 0), (1, 16)):
            live = PrefillCase("gdn", 2, 6, lengths)
            data = make_inputs(live, device=device, max_tokens=capacity, max_seqs=seq_capacity, seed=93)
            if lengths[0] >= 256:
                data["checkpoint_offsets"][0] = 256
            for name, tensor in tensors.items():
                tensor.copy_(data[name])
            initial = tensors["recurrent_state"].clone()
            expected, state = oracle(live, tensors)
            binding.scratch.fill_(0xFF)
            allocated = torch.cuda.memory_allocated(device)
            graph.replay()
            torch.cuda.synchronize(device)
            assert torch.cuda.memory_allocated(device) == allocated
            assert addresses == tuple(t.data_ptr() for t in (binding.output, binding.scratch, binding.recurrent_state))
            check_binding(live, binding, expected, state, initial)
    finally:
        unfreeze_kernel_resolution()


@pytest.mark.parametrize("name,parameters", [
    ("single_sequence", {"tokens": 32768}),
    ("serving_geometry_and_packed_views", {"key_heads": 2}),
    ("serving_geometry_and_packed_views", {"key_heads": 16}),
    ("padded_state_slot_offset_past_int32_boundary", {}),
    *[("bad_metadata_is_transactional", {"error": error})
      for error in ("duplicate", "length", "slot", "checkpoint")],
    *[("empty_null_inplace_and_checkpoint_slots", {"lengths": lengths})
      for lengths in ((), (0,), (0, 0), (0, 17, 33))],
    *[("int64_indices_strided_gates_and_immutable_inputs", {"validation": validation})
      for validation in ("transactional", "trusted")],
    *[("prefill_state_continues_through_public_decode", {"large_rate": large_rate})
      for large_rate in (False, True)],
])
def test_gpu_parallel_preserves_public_contract(name, parameters, monkeypatch):
    from . import test_gdn_prefill as contract

    bind = contract.make_binding

    def parallel_binding(case, tensors, **kwargs):
        tokens = kwargs.get("max_tokens", tensors["q"].shape[0])
        seqs = kwargs.get("max_seqs", tensors["cu_seqlens"].numel() - 1)
        return bind(case, tensors, config=_config(tokens, seqs, 256), **kwargs)

    monkeypatch.setattr(contract, "make_binding", parallel_binding)
    getattr(contract, "test_gpu_" + name)(**parameters)


@pytest.mark.parametrize("special", [0.25, float("nan"), float("inf"), 3.4e38])
@pytest.mark.parametrize("poison_local", [False, True])
def test_gpu_zero_transfer_boundary_matches_dense_product(special, poison_local):
    from b12x.sequence.gdn_prefill._parallel_kernels import compile_auxiliary

    case = PrefillCase("gdn", 1, 3, (3 * 256 + 17,))
    tensors = make_inputs(case, device=require_b12x())
    tensors["raw_g"].fill_(10)
    binding = make_binding(case, tensors, config=_config(tensors["q"].shape[0], 1, 256))
    gdn.prewarm(binding)
    gdn.run(binding)
    parallel = binding.parallel
    auxiliary = compile_auxiliary(binding)
    segments = parallel.plan.max_segments
    assert torch.equal(parallel.transfer_flags[:3] & 1, torch.ones_like(parallel.transfer_flags[:3]))
    tensors["recurrent_state"][0, 0, 0, 16] = special
    if poison_local:
        parallel.pool[2 + segments, 1, 5, 13] = float("nan")
        auxiliary.pack_transfer(binding)
    auxiliary.boundaries(binding)
    expected = parallel.pool[2 + 2 * segments:2 + 2 * segments + 4].clone()
    parallel.transfer_flags.zero_()
    auxiliary.boundaries(binding)
    actual = parallel.pool[2 + 2 * segments:2 + 2 * segments + 4]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("source", ["finite", "k", "raw_g", "raw_beta"])
@pytest.mark.parametrize("summary", ["separate", "joint", "output"])
def test_gpu_zero_transfer_summary_matches_unskipped_recurrence(source, summary):
    from b12x.sequence._shared.delta_prefill._cute_kernels import (
        _compile_recurrence, run_prepare, run_prologue,
    )
    from b12x.sequence._shared.delta_prefill.workspace import WorkspaceRecord
    from b12x.sequence.gdn_prefill._parallel_kernels import compile_auxiliary

    case = PrefillCase("gdn", 1, 3, (2 * 512 + 17,))
    tensors = make_inputs(case, device=require_b12x())
    tensors["raw_g"].fill_(10)
    if source != "finite":
        tensors[source][144, 0] = float("nan")
    binding = make_binding(case, tensors, config=_config(tensors["q"].shape[0], 1, 512))
    gdn.prewarm(binding)
    parallel = binding.parallel
    auxiliary = compile_auxiliary(binding)
    run_prologue(binding)
    auxiliary.partition(binding)
    run_prologue(parallel.output)
    run_prepare(parallel.output, lower_bound=0.0, scale=1 / 128**0.5, eps=1e-6)
    passes = ([(parallel.transfer, 3)] if summary == "joint" else
              [(parallel.transfer, 2), (parallel.local_state, 4 if summary == "output" else 1)])
    for active, mode in passes:
        _compile_recurrence(active, mode)[1](active, 0)
    slots = [2, 3, 2 + parallel.plan.max_segments, 3 + parallel.plan.max_segments]
    expected = parallel.pool[slots].clone()
    parallel.output.ws.view(torch.float32)[..., WorkspaceRecord.SUMMARY_FINITE // 4].zero_()
    for active, mode in passes:
        _compile_recurrence(active, mode)[1](active, 0)
    torch.testing.assert_close(parallel.pool[slots], expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("source", ["finite", "weak", "v", "k", "raw_g", "raw_beta", "state"])
@pytest.mark.parametrize("checkpoint", [16, 128, 256, 528])
def test_gpu_output_reuse_matches_forced_full_recurrence(source, checkpoint):
    from b12x.sequence._shared.delta_prefill._cute_kernels import (
        _compile_recurrence, run_prepare, run_prologue,
    )
    from b12x.sequence.gdn_prefill._parallel_kernels import compile_auxiliary

    case = PrefillCase("gdn", 1, 3, (2 * 512 + 17, 0, 33))
    tensors = make_inputs(case, device=require_b12x())
    if source == "weak":
        tensors["raw_g"].fill_(-12)
    elif source == "state":
        tensors["recurrent_state"][0, 0, 0, 0] = float("nan")
    elif source != "finite":
        tensors[source][144, 0] = float("nan")
    tensors["checkpoint_offsets"][0] = checkpoint
    binding = make_binding(case, tensors, checkpoint_export=True,
                           config=_config(case.tokens, len(case.lengths), 512))
    initial = tensors["recurrent_state"].clone()
    gdn.prewarm(binding)
    gdn.run(binding)
    expected_output = binding.output.clone()
    expected_state = binding.recurrent_state.clone()
    binding.recurrent_state.copy_(initial)
    binding.scratch.fill_(0xFF)
    p = binding.parallel
    aux = compile_auxiliary(binding)
    run_prologue(binding)
    aux.partition(binding)
    run_prologue(p.output)
    run_prepare(p.output, lower_bound=0.0, scale=128**-0.5, eps=1e-6)
    _compile_recurrence(p.transfer, 2)[1](p.transfer, 0)
    _compile_recurrence(p.local_state, 4)[1](p.local_state, 0)
    aux.pack_transfer(binding)
    aux.boundaries(binding)
    # One changed mantissa bit per element forces the complete correction pass.
    begin = 2 + 3 * p.plan.max_segments
    p.pool[begin:begin + p.plan.max_segments].view(torch.int32).bitwise_xor_(1)
    _compile_recurrence(p.output, 5)[1](p.output, 0)
    aux.commit(binding)
    torch.testing.assert_close(binding.output, expected_output, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(binding.recurrent_state, expected_state, rtol=0, atol=0, equal_nan=True)
