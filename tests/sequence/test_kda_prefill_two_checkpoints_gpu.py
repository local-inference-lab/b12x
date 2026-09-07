"""GB10 GPU qualification tests for two recurrent checkpoint exports.

H16 and H32 correspond to GLM-5.3-Flash's 64 KDA heads at TP4 and TP2.
These are per-rank component tests, not multi-GPU serving qualification.
"""

from __future__ import annotations

import os

import pytest
import torch

from .test_kda_prefill import (
    HEAD_DIM,
    _run as run_op,
    assert_kda_close,
    make_binding,
    make_inputs,
    run_oracle,
)


def require_gb10():
    from ..conftest import require_b12x
    from b12x.policy import PolicyContext

    device = require_b12x()
    identity = PolicyContext.for_device(device).device
    if (
        identity is None
        or identity.vendor != "nvidia"
        or identity.product_name != "nvidia gb10"
        or identity.compute_capability != (12, 1)
        or identity.sm_count != 48
    ):
        pytest.skip(
            "two-checkpoint GPU qualification requires NVIDIA GB10 / SM121 / 48 SMs"
        )
    return device


def checkpoint_inputs(
    lengths, *, heads=1, offsets=None, seed=900, state_slots=16, capacity=2, device
):
    count = len(lengths)
    inputs = make_inputs(
        lengths=lengths,
        heads=heads,
        seed=seed,
        state_slots=state_slots,
        device=device,
    )
    inputs["checkpoint_slots"] = torch.arange(
        2 * count,
        (2 + capacity) * count,
        dtype=torch.int32,
        device=device,
    ).reshape(count, capacity)
    inputs["checkpoint_offsets"] = torch.tensor(
        offsets or [[16, length // 16 * 16] for length in lengths],
        dtype=torch.int32,
        device=device,
    )
    return inputs


def assert_checkpoint_oracle(binding, tensors, inputs):
    capacity = inputs["checkpoint_slots"].shape[1]
    expected_out, expected_pool = run_oracle(inputs, max_checkpoints=capacity)
    tokens = inputs["num_tokens"]
    assert binding.error_code.item() == 0
    assert_kda_close(
        "output", expected_out[:tokens], binding.output[:tokens], ratio=1e-2
    )
    writes = set(inputs["final"].tolist())
    for slots, offsets in zip(
        inputs["checkpoint_slots"].tolist(),
        inputs["checkpoint_offsets"].tolist(),
        strict=True,
    ):
        writes.update(
            slot for slot, offset in zip(slots, offsets, strict=True) if offset > 0
        )
    for slot in writes:
        assert_kda_close(
            f"state[{slot}]",
            expected_pool[slot],
            tensors["recurrent_state"][slot],
            ratio=5e-3,
        )
    untouched = sorted(set(range(inputs["pool"].shape[0])) - writes)
    torch.testing.assert_close(
        tensors["recurrent_state"][untouched], inputs["pool"][untouched], rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "heads", [1, 16, 32], ids=["minimal", "glm-tp4-heads", "glm-tp2-heads"]
)
@pytest.mark.parametrize("inplace", [False, True])
def test_two_checkpoint_gpu_independent_fp32_oracle(heads, inplace):
    device = require_gb10()
    inputs = checkpoint_inputs(
        [80, 64], offsets=[[64, 16], [16, 64]], heads=heads, device=device
    )
    if inplace:
        inputs["final"][0] = inputs["initial"][0]
    binding, tensors = make_binding(
        inputs,
        max_tokens=256,
        max_seqs=4,
        final_stride=3,
        checkpoint_export=True,
        max_checkpoints=2,
    )

    run_op(binding, inputs)
    torch.cuda.synchronize(device)
    assert_checkpoint_oracle(binding, tensors, inputs)


@pytest.mark.parametrize("heads", [16, 32], ids=["glm-tp4-heads", "glm-tp2-heads"])
def test_two_checkpoint_gpu_8k_matches_one_checkpoint_prefixes(heads):
    device = require_gb10()

    inputs = checkpoint_inputs(
        [8192], heads=heads, offsets=[[6144, 7168]], device=device
    )
    binding, tensors = make_binding(
        inputs,
        max_tokens=8192,
        max_seqs=1,
        checkpoint_export=True,
        max_checkpoints=2,
    )
    run_op(binding, inputs)
    torch.cuda.synchronize(device)
    assert binding.error_code.item() == 0
    for length, slot in ((8192, 1), (6144, 2), (7168, 3)):
        prefix = dict(inputs)
        for name in ("q", "k", "v", "raw_g", "raw_beta"):
            prefix[name] = inputs[name][:length]
        prefix.update(
            num_tokens=length,
            cu_seqlens=torch.tensor([0, length], dtype=torch.int32, device=device),
            checkpoint_slots=torch.tensor([0], dtype=torch.int32, device=device),
            checkpoint_offsets=torch.tensor([0], dtype=torch.int32, device=device),
        )
        one_checkpoint, one_checkpoint_tensors = make_binding(
            prefix, max_tokens=8192, max_seqs=1
        )
        run_op(one_checkpoint, prefix)
        torch.cuda.synchronize(device)
        assert one_checkpoint.error_code.item() == 0
        torch.testing.assert_close(
            tensors["recurrent_state"][slot],
            one_checkpoint_tensors["recurrent_state"][1],
            rtol=0,
            atol=0,
        )
        if length == 8192:
            torch.testing.assert_close(
                binding.output, one_checkpoint.output, rtol=0, atol=0
            )


def copy_live(tensors, live):
    for name in ("q", "k", "v", "raw_g", "raw_beta"):
        tensors[name].zero_()
        tensors[name][: live["num_tokens"]].copy_(live[name])
    for destination, source in (
        ("cu_seqlens", "cu_seqlens"),
        ("initial_state_indices", "initial"),
        ("final_state_indices", "final"),
        ("checkpoint_state_indices", "checkpoint_slots"),
        ("checkpoint_offsets", "checkpoint_offsets"),
    ):
        tensors[destination].zero_()
        tensors[destination][: live[source].shape[0]].copy_(live[source])
    for name in ("A_log", "dt_bias"):
        tensors[name].copy_(live[name])
    tensors["recurrent_state"].copy_(live["pool"])
    tensors["num_tokens"].fill_(live["num_tokens"])
    tensors["num_seqs"].fill_(live["num_seqs"])


def test_two_checkpoint_gpu_frozen_replay_changes_live_counts_and_offsets():
    device = require_gb10()
    from b12x._lib.runtime_control import (
        freeze_kernel_resolution,
        unfreeze_kernel_resolution,
    )
    from b12x.sequence.kda_prefill import _cute_kernels as kernels

    first = checkpoint_inputs([96], device=device)
    binding, tensors = make_binding(
        first, max_tokens=256, max_seqs=4, checkpoint_export=True, max_checkpoints=2
    )
    run_op(binding, first)
    torch.cuda.synchronize(device)
    launchers = (
        kernels._PROLOGUE_CACHE[kernels._prologue_key(binding)],
        kernels._PREPARE_CACHE[kernels._prepare_key(binding)],
        kernels._RECURRENCE_CACHE[kernels._recurrence_key(binding)],
    )
    addresses = tuple(t.data_ptr() for t in (*tensors.values(), binding.scratch))
    freeze_kernel_resolution("two-checkpoint GB10 qualification")
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_op(binding, first)
        for lengths, offsets in (
            ([128], [[112, 32]]),
            ([64, 80, 96], [[16, 48], [64, 32], [32, 80]]),
            ([48], [[32, 0]]),
        ):
            live = checkpoint_inputs(
                lengths, offsets=offsets, seed=sum(lengths), device=device
            )
            copy_live(tensors, live)
            binding.output.fill_(float("nan"))
            binding.scratch.fill_(0xFF)
            torch.cuda.synchronize(device)
            before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize(device)
            assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == before
            assert addresses == tuple(
                t.data_ptr() for t in (*tensors.values(), binding.scratch)
            )
            assert_checkpoint_oracle(binding, tensors, live)
            assert torch.isnan(binding.output[live["num_tokens"] :].float()).all()
    finally:
        unfreeze_kernel_resolution()
    assert launchers == (
        kernels._PROLOGUE_CACHE[kernels._prologue_key(binding)],
        kernels._PREPARE_CACHE[kernels._prepare_key(binding)],
        kernels._RECURRENCE_CACHE[kernels._recurrence_key(binding)],
    )


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate-slot",
        "initial-alias",
        "duplicate-offset",
        "unaligned",
        "out-of-range",
    ],
)
def test_two_checkpoint_gpu_invalid_metadata_preserves_state(fault):
    device = require_gb10()

    inputs = checkpoint_inputs([64], device=device)
    binding, tensors = make_binding(
        inputs, max_tokens=96, max_seqs=2, checkpoint_export=True, max_checkpoints=2
    )
    if fault == "duplicate-slot":
        tensors["checkpoint_state_indices"][0, 1] = tensors["checkpoint_state_indices"][
            0, 0
        ]
    elif fault == "initial-alias":
        tensors["checkpoint_state_indices"][0, 1] = 0
    elif fault == "duplicate-offset":
        tensors["checkpoint_offsets"][0, 1] = 16
    elif fault == "unaligned":
        tensors["checkpoint_offsets"][0, 1] = 17
    else:
        tensors["checkpoint_state_indices"][0, 1] = 16
    before = tensors["recurrent_state"].clone()
    binding.output.fill_(7)
    run_op(binding, inputs)
    torch.cuda.synchronize(device)
    assert binding.error_code.item() != 0
    assert torch.equal(tensors["recurrent_state"], before)
    # Transactional failure poisons the full bound output capacity.
    assert torch.isnan(binding.output.float()).all()


def test_two_checkpoint_gpu_high_pool_offsets():
    if os.environ.get("B12X_RUN_LARGE_POOL_TESTS") != "1":
        pytest.skip(
            "set B12X_RUN_LARGE_POOL_TESTS=1 on an idle GB10; requires over 8 GiB"
        )
    device = require_gb10()

    slot_stride = HEAD_DIM * HEAD_DIM + 2048
    high = (1 << 31) // slot_stride + 8
    storage_elements = (high + 4) * slot_stride
    free, _ = torch.cuda.mem_get_info(device)
    if free < storage_elements * 4 + (2 << 30):
        pytest.skip(
            "insufficient free memory for an 8 GiB high-offset pool plus scratch"
        )
    inputs = checkpoint_inputs([64], offsets=[[16, 48]], state_slots=4, device=device)
    compact, compact_tensors = make_binding(
        inputs, max_tokens=64, max_seqs=1, checkpoint_export=True, max_checkpoints=2
    )
    run_op(compact, inputs)
    torch.cuda.synchronize(device)
    assert compact.error_code.item() == 0
    storage = torch.empty(storage_elements, dtype=torch.float32, device=device)
    pool = torch.as_strided(
        storage,
        (high + 4, 1, HEAD_DIM, HEAD_DIM),
        (slot_stride, HEAD_DIM * HEAD_DIM, HEAD_DIM, 1),
    )
    for slot in range(4):
        pool[high + slot].copy_(inputs["pool"][slot])
    large_inputs = dict(inputs)
    large_inputs["initial"] = torch.tensor([high], dtype=torch.int64, device=device)
    large_inputs["final"] = torch.tensor([high + 1], dtype=torch.int64, device=device)
    large_inputs["checkpoint_slots"] = torch.tensor(
        [[high + 2, high + 3]], dtype=torch.int64, device=device
    )
    large, _ = make_binding(
        large_inputs,
        max_tokens=64,
        max_seqs=1,
        recurrent_state=pool,
        checkpoint_export=True,
        max_checkpoints=2,
    )
    run_op(large, large_inputs)
    torch.cuda.synchronize(device)
    assert large.error_code.item() == 0
    torch.testing.assert_close(large.output, compact.output, rtol=0, atol=0)
    for slot in range(4):
        torch.testing.assert_close(
            pool[high + slot], compact_tensors["recurrent_state"][slot], rtol=0, atol=0
        )


@pytest.mark.parametrize("heads", [1, 16], ids=["minimal", "glm-tp4-heads"])
@pytest.mark.parametrize("inplace", [False, True])
def test_four_checkpoint_gpu_independent_fp32_oracle(heads, inplace):
    device = require_gb10()
    inputs = checkpoint_inputs(
        [80, 96],
        heads=heads,
        capacity=4,
        offsets=[[64, 16, 48, 32], [16, 80, 48, 96]],
        device=device,
    )
    if inplace:
        inputs["final"][0] = inputs["initial"][0]
    binding, tensors = make_binding(
        inputs, max_tokens=256, max_seqs=4, checkpoint_export=True, max_checkpoints=4
    )
    run_op(binding, inputs)
    torch.cuda.synchronize(device)
    assert_checkpoint_oracle(binding, tensors, inputs)


def test_four_checkpoint_gpu_8k_matches_fine_and_coarse_prefixes():
    device = require_gb10()
    positions = (4096, 6144, 7168, 7680)
    inputs = checkpoint_inputs(
        [8192], heads=16, capacity=4, offsets=[list(positions)], device=device
    )
    binding, tensors = make_binding(
        inputs, max_tokens=8192, max_seqs=1, checkpoint_export=True, max_checkpoints=4
    )
    run_op(binding, inputs)
    torch.cuda.synchronize(device)
    assert binding.error_code.item() == 0
    for length, slot in ((8192, 1), *zip(positions, (2, 3, 4, 5), strict=True)):
        prefix = dict(inputs)
        for name in ("q", "k", "v", "raw_g", "raw_beta"):
            prefix[name] = inputs[name][:length]
        prefix.update(
            num_tokens=length,
            cu_seqlens=torch.tensor([0, length], dtype=torch.int32, device=device),
            checkpoint_slots=torch.tensor([0], dtype=torch.int32, device=device),
            checkpoint_offsets=torch.tensor([0], dtype=torch.int32, device=device),
        )
        single, single_tensors = make_binding(prefix, max_tokens=8192, max_seqs=1)
        run_op(single, prefix)
        torch.cuda.synchronize(device)
        assert single.error_code.item() == 0
        torch.testing.assert_close(
            tensors["recurrent_state"][slot],
            single_tensors["recurrent_state"][1],
            rtol=0,
            atol=0,
        )
        if length == 8192:
            torch.testing.assert_close(binding.output, single.output, rtol=0, atol=0)


def test_four_checkpoint_gpu_frozen_replay_changes_live_counts_and_offsets():
    device = require_gb10()
    from b12x._lib.runtime_control import (
        freeze_kernel_resolution,
        unfreeze_kernel_resolution,
    )
    from b12x.sequence.kda_prefill import _cute_kernels as kernels

    first = checkpoint_inputs(
        [96], capacity=4, state_slots=32, offsets=[[16, 32, 64, 80]], device=device
    )
    binding, tensors = make_binding(
        first, max_tokens=256, max_seqs=4, checkpoint_export=True, max_checkpoints=4
    )
    run_op(binding, first)
    torch.cuda.synchronize(device)
    launchers = (
        kernels._PROLOGUE_CACHE[kernels._prologue_key(binding)],
        kernels._PREPARE_CACHE[kernels._prepare_key(binding)],
        kernels._RECURRENCE_CACHE[kernels._recurrence_key(binding)],
    )
    addresses = tuple(t.data_ptr() for t in (*tensors.values(), binding.scratch))
    freeze_kernel_resolution("four-checkpoint GB10 qualification")
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_op(binding, first)
        for lengths, offsets in (
            ([128], [[112, 32, 64, 16]]),
            ([64, 80, 96], [[16, 32, 48, 64], [64, 32, 16, 80], [32, 80, 16, 64]]),
            ([48], [[32, 0, 16, 0]]),
        ):
            live = checkpoint_inputs(
                lengths,
                capacity=4,
                offsets=offsets,
                state_slots=32,
                seed=sum(lengths),
                device=device,
            )
            copy_live(tensors, live)
            binding.output.fill_(float("nan"))
            binding.scratch.fill_(0xFF)
            torch.cuda.synchronize(device)
            before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize(device)
            assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == before
            assert addresses == tuple(
                t.data_ptr() for t in (*tensors.values(), binding.scratch)
            )
            assert_checkpoint_oracle(binding, tensors, live)
            assert torch.isnan(binding.output[live["num_tokens"] :].float()).all()
    finally:
        unfreeze_kernel_resolution()
    assert launchers == (
        kernels._PROLOGUE_CACHE[kernels._prologue_key(binding)],
        kernels._PREPARE_CACHE[kernels._prepare_key(binding)],
        kernels._RECURRENCE_CACHE[kernels._recurrence_key(binding)],
    )


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate-slot",
        "initial-alias",
        "duplicate-offset",
        "unaligned",
        "out-of-range",
    ],
)
def test_four_checkpoint_gpu_invalid_fourth_metadata_preserves_state(fault):
    device = require_gb10()
    inputs = checkpoint_inputs(
        [80], capacity=4, offsets=[[16, 32, 48, 64]], device=device
    )
    binding, tensors = make_binding(
        inputs, max_tokens=96, max_seqs=2, checkpoint_export=True, max_checkpoints=4
    )
    if fault == "duplicate-slot":
        tensors["checkpoint_state_indices"][0, 3] = tensors["checkpoint_state_indices"][
            0, 0
        ]
    elif fault == "initial-alias":
        tensors["checkpoint_state_indices"][0, 3] = 0
    elif fault == "duplicate-offset":
        tensors["checkpoint_offsets"][0, 3] = 16
    elif fault == "unaligned":
        tensors["checkpoint_offsets"][0, 3] = 17
    else:
        tensors["checkpoint_state_indices"][0, 3] = 16
    before = tensors["recurrent_state"].clone()
    binding.output.fill_(7)
    run_op(binding, inputs)
    torch.cuda.synchronize(device)
    assert binding.error_code.item() != 0
    assert torch.equal(tensors["recurrent_state"], before)
    assert torch.isnan(binding.output.float()).all()


def test_four_checkpoint_gpu_high_pool_offsets():
    if os.environ.get("B12X_RUN_LARGE_POOL_TESTS") != "1":
        pytest.skip(
            "set B12X_RUN_LARGE_POOL_TESTS=1 on an idle GB10; requires over 8 GiB"
        )
    device = require_gb10()
    heads = 16
    slot_stride = heads * HEAD_DIM * HEAD_DIM + 2048
    high = (1 << 31) // slot_stride + 8
    storage_elements = (high + 6) * slot_stride
    free, _ = torch.cuda.mem_get_info(device)
    if free < storage_elements * 4 + (2 << 30):
        pytest.skip(
            "insufficient free memory for an 8 GiB high-offset pool plus scratch"
        )
    inputs = checkpoint_inputs(
        [80],
        heads=heads,
        capacity=4,
        state_slots=6,
        offsets=[[16, 32, 48, 64]],
        device=device,
    )
    compact, compact_tensors = make_binding(
        inputs, max_tokens=80, max_seqs=1, checkpoint_export=True, max_checkpoints=4
    )
    run_op(compact, inputs)
    torch.cuda.synchronize(device)
    assert compact.error_code.item() == 0
    storage = torch.empty(storage_elements, dtype=torch.float32, device=device)
    pool = torch.as_strided(
        storage,
        (high + 6, heads, HEAD_DIM, HEAD_DIM),
        (slot_stride, HEAD_DIM * HEAD_DIM, HEAD_DIM, 1),
    )
    for slot in range(6):
        pool[high + slot].copy_(inputs["pool"][slot])
    large_inputs = dict(inputs)
    large_inputs["initial"] = torch.tensor([high], dtype=torch.int64, device=device)
    large_inputs["final"] = torch.tensor([high + 1], dtype=torch.int64, device=device)
    large_inputs["checkpoint_slots"] = torch.tensor(
        [[high + i for i in (2, 3, 4, 5)]], dtype=torch.int64, device=device
    )
    large, _ = make_binding(
        large_inputs,
        max_tokens=80,
        max_seqs=1,
        recurrent_state=pool,
        checkpoint_export=True,
        max_checkpoints=4,
    )
    run_op(large, large_inputs)
    torch.cuda.synchronize(device)
    assert large.error_code.item() == 0
    torch.testing.assert_close(large.output, compact.output, rtol=0, atol=0)
    for slot in range(6):
        torch.testing.assert_close(
            pool[high + slot], compact_tensors["recurrent_state"][slot], rtol=0, atol=0
        )
