"""Scalar GDN algebra, public contracts, and real CuTe prefill qualification."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from b12x.policy.generation.delta_prefill_cases import (
    PrefillCase, assert_close, check_binding, make_binding, make_inputs, oracle, run_binding,
)
from b12x.sequence import gdn_prefill as gdn
from b12x.sequence.gdn_prefill.reference import chunk_mirror, prepare_chunk, recurrent_gdn


def _recurrent_args(tensors):
    return [tensors[n] for n in ("q", "k", "v", "raw_g", "raw_beta", "A_log", "dt_bias")]


@pytest.mark.parametrize("tokens", [1, 15, 16, 17, 63, 64, 65, 256])
@pytest.mark.parametrize("profile", ["random", "strong", "weak", "alternating"])
def test_chunk_algebra_matches_sequential_oracle(tokens, profile):
    case = PrefillCase("gdn", 1, 3, (tokens,))
    tensors = make_inputs(case, device="cpu")
    if profile == "strong":
        tensors["raw_g"].fill_(10)
    elif profile == "weak":
        tensors["raw_g"].fill_(-12)
    elif profile == "alternating":
        tensors["k"][1:] = tensors["k"][:1].clone()
        tensors["k"][1::2].neg_()
        tensors["raw_beta"].fill_(12)
    args = _recurrent_args(tensors)
    state = tensors["recurrent_state"][0]
    expected, final, _ = recurrent_gdn(*args, initial_state=state)
    actual, actual_final, _ = chunk_mirror(*args, initial_state=state)
    assert_close("mirror output", actual, expected, ratio=1e-2)
    assert_close("mirror state", actual_final, final, ratio=5e-3)


def test_strong_earlier_gate_does_not_cancel_later_decay():
    tensors = make_inputs(PrefillCase("gdn", 1, 3, (16,)), device="cpu")
    tensors["raw_g"].fill_(10)
    tensors["raw_g"][0].fill_(1e10)
    args = _recurrent_args(tensors)
    prep = prepare_chunk(args[0], args[1], args[3], args[4], args[5], args[6])
    assert all(torch.isfinite(x).all() for x in prep.values())
    state = tensors["recurrent_state"][0]
    expected, final, _ = recurrent_gdn(*args, initial_state=state)
    actual, actual_final, _ = chunk_mirror(*args, initial_state=state)
    assert_close("strong gate output", actual, expected, ratio=1e-2)
    assert_close("strong gate state", actual_final, final, ratio=5e-3)


def test_sequential_oracle_matches_decode_state_and_beta_rounding():
    from b12x.sequence.gdn_decode.reference import decode
    case = PrefillCase("gdn", 1, 3, (4,))
    tensors = make_inputs(case, device="cpu")
    _, expected, _ = recurrent_gdn(*_recurrent_args(tensors), initial_state=tensors["recurrent_state"][0])
    pool = torch.zeros(4, 3, 128, 128)
    pool[0] = tensors["recurrent_state"][0]
    mixed = torch.cat([tensors[n].flatten(1) for n in ("q", "k", "v")], dim=1)
    decode(mixed, tensors["raw_g"], tensors["raw_beta"], torch.ones_like(tensors["v"]),
           tensors["A_log"], tensors["dt_bias"], torch.ones(128, dtype=torch.bfloat16), pool,
           torch.tensor([0, 4], dtype=torch.int32), torch.tensor([1], dtype=torch.int32),
           torch.tensor([[0, 1, 2, 3]], dtype=torch.int32), 1, 4,
           key_heads=1, value_heads=3, gate_activation="silu")
    torch.testing.assert_close(pool[3], expected, rtol=1e-5, atol=2e-5)


def test_caps_accept_long_prefill_and_reject_unsupported_geometry():
    caps = gdn.Caps(device="cuda:0", max_tokens=32768, max_seqs=1,
                    max_state_slots=4, key_heads=2, value_heads=6)
    assert caps.tiles_capacity == 2049
    for kwargs, match in (({"value_heads": 4}, "three value"),
                           ({"state_dtype": torch.bfloat16}, "state_dtype"),
                           ({"chunk_tokens": 64}, "chunk_tokens"),
                           ({"head_dim": 64}, "head_dim")):
        with pytest.raises(ValueError, match=match):
            replace(caps, **kwargs)


def test_packed_oracle_checkpoints_and_null_initial():
    case = PrefillCase("gdn", 1, 3, (33, 0, 17))
    tensors = make_inputs(case, device="cpu")
    tensors["initial_state_indices"][0] = 9
    tensors["recurrent_state"][9].fill_(float("nan"))
    tensors["checkpoint_offsets"][0] = 16
    output, pool = oracle(case, tensors, null_state_index=9)
    assert torch.isfinite(output).all()
    expected, state, checkpoint = recurrent_gdn(
        *[x[:33] if i < 5 else x for i, x in enumerate(_recurrent_args(tensors))],
        initial_state=torch.zeros(3, 128, 128), checkpoint_offset=16,
    )
    torch.testing.assert_close(output[:33], expected)
    torch.testing.assert_close(pool[3], state)
    torch.testing.assert_close(pool[6], checkpoint)
    assert torch.isnan(pool[9]).all()


def _gpu():
    from ..conftest import require_b12x
    return require_b12x()


@pytest.mark.parametrize("tokens", [1, 15, 16, 17, 63, 64, 65, 1024, 4096, 16384, 32768])
def test_gpu_single_sequence(tokens):
    case = PrefillCase("gdn", 1, 3, (tokens,))
    tensors = make_inputs(case, device=_gpu())
    initial = tensors["recurrent_state"].clone()
    expected, expected_state = oracle(case, tensors)
    binding = make_binding(case, tensors)
    gdn.run(binding)
    check_binding(case, binding, expected, expected_state, initial)


@pytest.mark.parametrize("key_heads", [2, 4, 8, 16])
def test_gpu_serving_geometry_and_packed_views(key_heads):
    case = PrefillCase("gdn", key_heads, 3*key_heads, (128, 47, 1, 0))
    tensors = make_inputs(case, device=_gpu(), max_tokens=256, max_seqs=8)
    mixed = torch.cat([tensors[n].flatten(1) for n in ("q", "k", "v")], dim=1)
    kw = key_heads*128
    tensors["q"] = mixed[:, :kw].view(256, key_heads, 128)
    tensors["k"] = mixed[:, kw:2*kw].view(256, key_heads, 128)
    tensors["v"] = mixed[:, 2*kw:].view(256, 3*key_heads, 128)
    initial = tensors["recurrent_state"].clone()
    expected, state = oracle(case, tensors)
    binding = make_binding(case, tensors, max_tokens=1024, max_seqs=16)
    assert binding.q.data_ptr() == mixed.data_ptr()
    gdn.run(binding)
    check_binding(case, binding, expected, state, initial)
    torch.testing.assert_close(mixed[:, :kw], tensors["q"].flatten(1), rtol=0, atol=0)


@pytest.mark.parametrize("profile", ["strong", "weak", "cancel", "beta_zero", "beta_one", "repeated", "alternating", "large_rate"])
def test_gpu_gate_extremes(profile):
    case = PrefillCase("gdn", 1, 3, (257,))
    tensors = make_inputs(case, device=_gpu())
    if profile in ("strong", "cancel"):
        tensors["raw_g"].fill_(10)
        if profile == "cancel":
            tensors["raw_g"][::16].fill_(1e10)
    elif profile == "weak":
        tensors["raw_g"].fill_(-12)
    elif profile == "large_rate":
        tensors["raw_g"].fill_(-20)
        tensors["A_log"].fill_(20)
        tensors["dt_bias"].zero_()
    elif profile in ("repeated", "alternating"):
        tensors["k"][1:] = tensors["k"][:1].clone()
        if profile == "alternating":
            tensors["k"][1::2].neg_()
        tensors["raw_g"].fill_(-12)
        tensors["raw_beta"].fill_(12)
    else:
        tensors["raw_beta"].fill_(-40 if profile == "beta_zero" else 40)
    initial = tensors["recurrent_state"].clone()
    expected, state = oracle(case, tensors)
    binding = make_binding(case, tensors)
    gdn.run(binding)
    check_binding(case, binding, expected, state, initial)


def test_gpu_graph_reuses_live_metadata_and_kernel_resolution():
    from b12x._lib.runtime_control import freeze_kernel_resolution, unfreeze_kernel_resolution
    from b12x.sequence._shared.delta_prefill import _cute_kernels as kernels
    device = _gpu()
    case = PrefillCase("gdn", 1, 3, (1,))
    tensors = make_inputs(case, device=device, max_tokens=512, max_seqs=4)
    binding = make_binding(case, tensors, checkpoint_export=True,
                           config={"backend": "cutedsl", "v_split": 64, "k_split": 1, "stages": 3, "window_tiles": 3})
    gdn.prewarm(binding)
    gdn.run(binding)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gdn.run(binding)
    keys = (kernels._prepare_key(binding), kernels._recurrence_key(binding), kernels._prologue_key(binding))
    launchers = (kernels._PREPARE_CACHE[keys[0]], kernels._RECURRENCE_CACHE[keys[1]], kernels._PROLOGUE_CACHE[keys[2]])
    addresses = tuple(t.data_ptr() for t in (binding.output, binding.recurrent_state, binding.scratch))
    freeze_kernel_resolution("GDN prefill live-count reuse")
    try:
        for lengths in ((512,), (128, 33, 256, 0)):
            live = PrefillCase("gdn", 1, 3, lengths)
            data = make_inputs(live, device=device, max_tokens=512, max_seqs=4, seed=91)
            data["checkpoint_offsets"][0] = 16
            for name, tensor in tensors.items():
                tensor.copy_(data[name])
            initial = tensors["recurrent_state"].clone()
            expected, state = oracle(live, tensors)
            binding.output.fill_(float("nan"))
            binding.scratch.fill_(0xFF)
            allocated = torch.cuda.memory_allocated(device)
            graph.replay()
            torch.cuda.synchronize(device)
            assert torch.cuda.memory_allocated(device) == allocated
            check_binding(live, binding, expected, state, initial)
            assert addresses == tuple(t.data_ptr() for t in (binding.output, binding.recurrent_state, binding.scratch))
            binding.recurrent_state.copy_(initial)
            binding.output.fill_(float("nan"))
            gdn.run(binding, max_live_tokens=sum(lengths), max_live_seqs=len(lengths))
            check_binding(live, binding, expected, state, initial)
    finally:
        unfreeze_kernel_resolution()
    assert launchers == (kernels._PREPARE_CACHE[keys[0]], kernels._RECURRENCE_CACHE[keys[1]], kernels._PROLOGUE_CACHE[keys[2]])


@pytest.mark.parametrize("error", ["duplicate", "length", "slot", "checkpoint"])
def test_gpu_bad_metadata_is_transactional(error):
    case = PrefillCase("gdn", 1, 3, (32, 17))
    tensors = make_inputs(case, device=_gpu())
    binding = make_binding(case, tensors, checkpoint_export=True)
    initial = tensors["recurrent_state"].clone()
    if error == "duplicate":
        tensors["final_state_indices"][1] = tensors["final_state_indices"][0]
    elif error == "length":
        tensors["cu_seqlens"][1] = -1
    elif error == "slot":
        tensors["final_state_indices"][0] = 999
    else:
        tensors["checkpoint_offsets"][0] = 15
    gdn.run(binding)
    assert binding.error_code.item() != 0
    assert torch.isnan(binding.output).all()
    torch.testing.assert_close(binding.recurrent_state, initial, rtol=0, atol=0)


def test_gpu_alternating_recipes_keep_specialized_kernel_caches_separate():
    from b12x.sequence._shared.delta_prefill import _cute_kernels as kernels
    device = _gpu()
    bindings = []
    for case in (PrefillCase("gdn", 1, 3, (65,)), PrefillCase("kda", 3, 3, (65,))):
        tensors = make_inputs(case, device=device)
        initial = tensors["recurrent_state"].clone()
        expected, state = oracle(case, tensors)
        binding = make_binding(case, tensors)
        run_binding(case.recipe, binding)
        check_binding(case, binding, expected, state, initial)
        bindings.append((case, binding, expected, state, initial))
    a, b = [x[1] for x in bindings]
    assert kernels._prepare_key(a) != kernels._prepare_key(b)
    assert kernels._recurrence_key(a) != kernels._recurrence_key(b)
    for case, binding, expected, state, initial in bindings:
        binding.recurrent_state.copy_(initial)
        run_binding(case.recipe, binding)
        check_binding(case, binding, expected, state, initial)


def test_gpu_padded_state_slot_offset_past_int32_boundary():
    device = _gpu()
    case = PrefillCase("gdn", 1, 3, (17,))
    tensors = make_inputs(case, device=device)
    compact_initial = tensors["recurrent_state"].clone()
    expected, state = oracle(case, tensors)
    elements = 3 * 128 * 128
    stride = elements + 128
    high = (1 << 31) // stride + 1
    storage = torch.empty((high+3) * stride, dtype=torch.float32, device=device)
    pool = torch.as_strided(storage, (high+3, 3, 128, 128), (stride, 128*128, 128, 1))
    pool[high].copy_(compact_initial[0])
    pool[high+1].fill_(float("nan"))
    pool[high+2].fill_(5)
    tensors["recurrent_state"] = pool
    tensors["initial_state_indices"].fill_(high)
    tensors["final_state_indices"].fill_(high+1)
    tensors["checkpoint_state_indices"].fill_(high+2)
    binding = make_binding(case, tensors)
    gdn.run(binding)
    assert binding.error_code.item() == 0
    assert_close("large-slot output", binding.output, expected, ratio=1e-2)
    assert_close("large-slot state", pool[high+1], state[1], ratio=5e-3)
    torch.testing.assert_close(pool[high], compact_initial[0], rtol=0, atol=0)
    assert torch.all(pool[high+2] == 5)


@pytest.mark.parametrize("lengths", [(), (0,), (0, 0), (0, 17, 33)])
def test_gpu_empty_null_inplace_and_checkpoint_slots(lengths):
    case = PrefillCase("gdn", 1, 3, lengths)
    tensors = make_inputs(case, device=_gpu())
    null = tensors["recurrent_state"].shape[0]-1
    tensors["recurrent_state"][null].fill_(float("nan"))
    tensors["initial_state_indices"].fill_(null)
    if len(lengths) > 1:
        tensors["initial_state_indices"][1] = tensors["final_state_indices"][1]
    if lengths and lengths[-1] >= 16:
        tensors["checkpoint_offsets"][len(lengths)-1] = 16
    initial = tensors["recurrent_state"].clone()
    expected, state = oracle(case, tensors, null_state_index=null)
    binding = make_binding(case, tensors, checkpoint_export=True, null_state_index=null)
    gdn.run(binding)
    check_binding(case, binding, expected, state, initial)


@pytest.mark.parametrize("validation", ["transactional", "trusted"])
def test_gpu_int64_indices_strided_gates_and_immutable_inputs(validation):
    case = PrefillCase("gdn", 1, 3, (63, 128))
    tensors = make_inputs(case, device=_gpu())
    for name in ("initial_state_indices", "final_state_indices", "checkpoint_state_indices"):
        tensors[name] = tensors[name].to(torch.int64)
    for name in ("raw_g", "raw_beta"):
        if name == "raw_g":
            storage = torch.zeros(case.tokens, 2, 3, dtype=torch.bfloat16, device=tensors[name].device)
            view = storage[:, 0, :]
        else:
            storage = torch.zeros(case.tokens, 3, 2, dtype=torch.bfloat16, device=tensors[name].device)
            view = storage[:, :, 0]
        view.copy_(tensors[name])
        tensors[name] = view
    initial = tensors["recurrent_state"].clone()
    saved = {n: t.clone() for n, t in tensors.items() if n not in ("output", "recurrent_state")}
    expected, state = oracle(case, tensors, qk_l2norm=False)
    binding = make_binding(case, tensors, metadata_validation=validation, qk_l2norm=False)
    gdn.run(binding)
    check_binding(case, binding, expected, state, initial)
    for name, tensor in saved.items():
        torch.testing.assert_close(tensors[name], tensor, rtol=0, atol=0)


@pytest.mark.parametrize("large_rate", [False, True])
def test_gpu_prefill_state_continues_through_public_decode(large_rate):
    from b12x.sequence import gdn_decode
    from benchmarks.benchmark_gdn_decode import BenchmarkCase, CaseBuffers, _reference, build_case

    device = _gpu()
    case = PrefillCase("gdn", 2, 6, (1024,))
    tensors = make_inputs(case, device=device)
    if large_rate:
        tensors["raw_g"].fill_(-20)
        tensors["A_log"].fill_(20)
        tensors["dt_bias"].zero_()
    _, expected_state = oracle(case, tensors)
    prefill = make_binding(case, tensors)
    gdn.run(prefill)
    decode_buffers = build_case(BenchmarkCase("continuation", (4,), 2, 6), device=device, seed=19)
    decode = decode_buffers.binding
    decode.A_log.copy_(tensors["A_log"])
    decode.dt_bias.copy_(tensors["dt_bias"])
    if large_rate:
        decode.a.fill_(-20)
    reference_initial = decode.recurrent_state.clone()
    reference_initial[0].copy_(expected_state[1])
    expected_output, expected_pool = _reference(CaseBuffers(decode, reference_initial))
    decode.recurrent_state[0].copy_(prefill.recurrent_state[1])
    actual = gdn_decode.run(decode)
    assert decode.error_code.item() == 0
    assert_close("continued decode output", actual, expected_output, ratio=1e-2)
    assert_close("continued decode state", decode.recurrent_state[3], expected_pool[3], ratio=5e-3)
