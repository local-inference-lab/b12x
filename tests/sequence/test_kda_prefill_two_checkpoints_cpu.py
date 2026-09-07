"""CPU contracts for bounded checkpoint export; no GPU qualification claims."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from b12x.policy import (
    ComponentProfile,
    DeviceIdentity,
    FrozenMapping,
    GpuProfile,
    InvalidPreplannedPolicyError,
    PolicyContext,
    PolicyMode,
    PolicySource,
    PreplannedPolicyNotFoundError,
    ProfileRegistry,
    ProfileRule,
)
from b12x.sequence.kda_prefill import _impl as impl
from b12x.sequence.kda_prefill._policy import (
    KDA_PREFILL_POLICY,
    KdaPrefillConfig,
    KdaPrefillQuery,
)
from b12x.sequence.kda_prefill.metadata import validate_metadata
from b12x.sequence.kda_prefill.reference import prefill_kda_chunk_mirror, recurrent_kda

from .test_kda_prefill import PURE_FP32, make_inputs, run_oracle


GB10 = DeviceIdentity(
    vendor="nvidia",
    product_name="NVIDIA GB10",
    compute_capability=(12, 1),
    sm_count=48,
)


def query(checkpoints: int = 2) -> KdaPrefillQuery:
    return KdaPrefillQuery(
        heads=16,
        head_dim=128,
        model_dtype="bfloat16",
        state_dtype="float32",
        qk_l2norm=True,
        checkpoint_export=True,
        max_tokens=8192,
        max_seqs=4,
        max_checkpoints=checkpoints,
    )


def metadata() -> dict:
    return dict(
        cu_seqlens=[0, 64, 160],
        initial_state_indices=[1, 5],
        final_state_indices=[2, 6],
        checkpoint_state_indices=[[3, 4], [7, 8]],
        checkpoint_offsets=[[16, 48], [32, 80]],
        num_seqs=2,
        num_tokens=160,
        token_capacity=192,
        seq_capacity=2,
        state_slots=10,
        null_state_index=0,
        max_checkpoints=2,
    )


@pytest.mark.parametrize(
    "device",
    [
        None,
        replace(GB10, product_name="Unknown SM121"),
        replace(GB10, compute_capability=(12, 0)),
        replace(GB10, sm_count=47),
    ],
)
def test_two_checkpoint_policy_rejects_unsupported_targets_even_with_override(device):
    context = PolicyContext.for_identity(device, mode=PolicyMode.HEURISTIC_ONLY)
    for override in (False, True):
        kwargs = {"override": KdaPrefillConfig()} if override else {}
        with pytest.raises(ValueError, match="supports only NVIDIA GB10"):
            context.resolve(KDA_PREFILL_POLICY, query(), **kwargs)
    # One-checkpoint planning accepts these device identities.
    assert (
        context.resolve(KDA_PREFILL_POLICY, query(1)).source is PolicySource.HEURISTIC
    )


def test_gb10_two_checkpoint_activation_is_explicit_and_not_qualification():
    context = PolicyContext.for_identity(GB10, registry=ProfileRegistry())
    one = context.resolve(KDA_PREFILL_POLICY, query(1))
    two = context.resolve(KDA_PREFILL_POLICY, query(2))
    assert one.config == two.config
    assert two.source is PolicySource.HEURISTIC
    assert two.profile_id is None and two.evidence is None
    assert context.resolve(KDA_PREFILL_POLICY, query(2)) is two
    strict = PolicyContext.for_identity(
        GB10,
        mode=PolicyMode.PREPLANNED_ONLY,
        registry=ProfileRegistry(),
    )
    with pytest.raises(PreplannedPolicyNotFoundError):
        strict.resolve(KDA_PREFILL_POLICY, query())


def test_query_schema_includes_planned_checkpoint_capacity_only():
    assert KDA_PREFILL_POLICY.query_schema_version == 3
    assert KDA_PREFILL_POLICY.config_schema_version == 1
    assert set(query().profile_fields()) == KDA_PREFILL_POLICY.query_fields
    assert query(1).profile_fields()["max_checkpoints"] == 1
    assert query(2).profile_fields()["max_checkpoints"] == 2
    assert {
        "num_tokens",
        "num_seqs",
        "checkpoint_offsets",
        "checkpoint_state_indices",
    }.isdisjoint(KDA_PREFILL_POLICY.query_fields)


@pytest.mark.parametrize("schema", [1, 2])
def test_schema1_and_schema2_profiles_rejected_by_schema3_query_contract(schema):
    registry = ProfileRegistry()
    registry.register(
        GpuProfile(
            profile_id="test.kda.schema1",
            targets=(GB10,),
            metadata=FrozenMapping(),
            components=(
                ComponentProfile(
                    component_id="sequence.kda_prefill",
                    query_schema_version=schema,
                    config_schema_version=1,
                    rules=(
                        ProfileRule.create(
                            name="synthetic-only",
                            exact={"heads": 16},
                            ranges={},
                            config=KdaPrefillConfig().to_dict(),
                            evidence="unit-test-not-gpu-evidence",
                        ),
                    ),
                ),
            ),
        )
    )
    context = PolicyContext.for_identity(GB10, registry=registry)
    with pytest.raises(InvalidPreplannedPolicyError, match="query schema mismatch"):
        context.resolve(KDA_PREFILL_POLICY, query())


@pytest.mark.parametrize("count", [0, 3, True, 2.0])
def test_caps_reject_invalid_checkpoint_capacity(count):
    with pytest.raises(ValueError, match="max_checkpoints"):
        impl.Caps(
            device="cuda:0",
            max_tokens=32,
            max_seqs=1,
            max_state_slots=8,
            heads=1,
            checkpoint_export=True,
            max_checkpoints=count,
        )


@pytest.mark.parametrize(
    "extra", [{"checkpoint_export": False}, {"metadata_validation": "trusted"}]
)
def test_two_checkpoint_caps_require_transactional_export(extra):
    kwargs = dict(checkpoint_export=True, metadata_validation="transactional")
    kwargs.update(extra)
    with pytest.raises(ValueError, match="multiple checkpoints require"):
        impl.Caps(
            device="cuda:0",
            max_tokens=32,
            max_seqs=1,
            max_state_slots=8,
            heads=1,
            max_checkpoints=2,
            **kwargs,
        )


@pytest.mark.parametrize("slot", [1, 2, 3, 5, 6, 7, 8])
def test_checkpoint_destinations_cannot_alias_other_owners(slot):
    args = metadata()
    args["checkpoint_state_indices"][0][1] = slot
    with pytest.raises(ValueError):
        validate_metadata(**args)


@pytest.mark.parametrize("offset", [17, 80, 16])
def test_second_checkpoint_rejects_unaligned_past_end_or_duplicate_offset(offset):
    args = metadata()
    args["checkpoint_offsets"][0][1] = offset
    with pytest.raises(ValueError):
        validate_metadata(**args)


@pytest.mark.parametrize("slot", [-1, 10])
def test_second_checkpoint_rejects_invalid_active_slot(slot):
    args = metadata()
    args["checkpoint_state_indices"][0][1] = slot
    with pytest.raises(IndexError):
        validate_metadata(**args)


def test_null_disabled_reversed_and_final_boundary_exports():
    args = metadata()
    args["final_state_indices"][0] = 1  # Own initial/final alias is legal.
    args["checkpoint_offsets"] = [[64, 16], [96, 32]]
    assert validate_metadata(**args) == [(0, 64), (64, 160)]
    args["checkpoint_state_indices"][0][1] = 0
    args["checkpoint_offsets"][0][1] = 64  # A null writer does not own an offset.
    validate_metadata(**args)
    args["checkpoint_offsets"][0][1] = 17
    with pytest.raises(ValueError, match="unaligned"):
        validate_metadata(**args)
    args["checkpoint_state_indices"][0][1] = 1
    for offset in (0, -1):
        args["checkpoint_offsets"][0][1] = offset
        validate_metadata(**args)


def test_inactive_metadata_is_ignored_but_live_counts_are_bounded():
    args = metadata()
    args.update(num_seqs=1, num_tokens=64)
    args["checkpoint_state_indices"][1] = [-999, -999]
    args["checkpoint_offsets"][1] = [17, 999]
    assert validate_metadata(**args) == [(0, 64)]
    args["num_seqs"] = 3
    with pytest.raises(ValueError, match="capacities"):
        validate_metadata(**args)


@pytest.mark.parametrize("inplace", [False, True])
def test_two_exports_equal_independent_prefix_recurrences_and_final_state(inplace):
    inputs = make_inputs(lengths=[64], heads=1, seed=830, state_slots=6)
    if inplace:
        inputs["final"][0] = inputs["initial"][0]
    one_checkpoint_output, one_checkpoint_pool = run_oracle(inputs)
    inputs["checkpoint_slots"] = torch.tensor([[3, 4]], dtype=torch.int32)
    inputs["checkpoint_offsets"] = torch.tensor([[48, 16]], dtype=torch.int32)
    output, pool = run_oracle(inputs, max_checkpoints=2)
    torch.testing.assert_close(output, one_checkpoint_output, rtol=0, atol=0)
    torch.testing.assert_close(
        pool[int(inputs["final"][0])],
        one_checkpoint_pool[int(inputs["final"][0])],
        rtol=0,
        atol=0,
    )
    for slot, offset in ((3, 48), (4, 16)):
        _, expected, _ = recurrent_kda(
            *(inputs[name][:offset] for name in ("q", "k", "v", "raw_g", "raw_beta")),
            inputs["A_log"],
            inputs["dt_bias"],
            lower_bound=-5.0,
            initial_state=inputs["pool"][int(inputs["initial"][0])],
        )
        torch.testing.assert_close(pool[slot], expected, rtol=0, atol=0)
    _, mirror_pool = run_oracle(
        inputs,
        fn=prefill_kda_chunk_mirror,
        max_checkpoints=2,
        policy=PURE_FP32,
    )
    for slot in (int(inputs["final"][0]), 3, 4):
        torch.testing.assert_close(mirror_pool[slot], pool[slot], rtol=2e-4, atol=2e-5)


def test_reference_rejection_is_transactional_for_pool_and_output():
    inputs = make_inputs(lengths=[64], heads=1, seed=831, state_slots=6)
    inputs["checkpoint_slots"] = torch.tensor([[3, 1]], dtype=torch.int32)
    inputs["checkpoint_offsets"] = torch.tensor([[16, 48]], dtype=torch.int32)
    before = inputs["pool"].clone()
    output = torch.full_like(inputs["q"], 7)
    from b12x.sequence.kda_prefill.reference import prefill_kda

    with pytest.raises(ValueError, match="duplicate"):
        prefill_kda(
            *(inputs[name] for name in ("q", "k", "v", "raw_g", "raw_beta")),
            inputs["A_log"],
            inputs["dt_bias"],
            inputs["pool"],
            inputs["cu_seqlens"],
            inputs["initial"],
            inputs["final"],
            inputs["checkpoint_slots"],
            inputs["checkpoint_offsets"],
            1,
            64,
            max_checkpoints=2,
            output=output,
        )
    assert torch.equal(inputs["pool"], before)
    assert torch.all(output == 7)


def test_plural_reference_zero_offset_returns_an_independent_initial_snapshot():
    inputs = make_inputs(lengths=[32], heads=1, seed=832)
    initial = inputs["pool"][0]
    _, _, snapshots = recurrent_kda(
        *(inputs[name] for name in ("q", "k", "v", "raw_g", "raw_beta")),
        inputs["A_log"],
        inputs["dt_bias"],
        lower_bound=-5.0,
        initial_state=initial,
        checkpoint_offsets=(0, 16),
    )
    assert isinstance(snapshots, dict) and set(snapshots) == {0, 16}
    assert torch.equal(snapshots[0], initial)
    assert snapshots[0].data_ptr() != initial.data_ptr()


@pytest.mark.parametrize("fault", ["shape", "dtype"])
def test_bind_reports_checkpoint_tensor_contract_before_cross_index_dtype(fault):
    from torch._subclasses.fake_tensor import FakeTensorMode

    caps = impl.Caps(
        device="cuda:0",
        max_tokens=32,
        max_seqs=1,
        max_state_slots=4,
        heads=1,
        checkpoint_export=True,
        max_checkpoints=2,
    )
    # Materialize only the static layout; fake CUDA tensors allocate no storage.
    plan = impl._materialize_plan(
        caps,
        v_split=64,
        k_split=1,
        stages=3,
        window_tiles=4,
        policy_resolution=None,
    )
    with FakeTensorMode():

        def tensor(shape, dtype=torch.bfloat16):
            return torch.empty(shape, device="cuda:0", dtype=dtype)

        inputs = {name: tensor((32, 1, 128)) for name in ("q", "k", "v", "raw_g")}
        checkpoint_indices = tensor(
            (2,) if fault == "shape" else (1, 2),
            torch.int16,
        )
        with pytest.raises(
            (ValueError, TypeError), match="checkpoint_state_indices must have"
        ):
            impl.bind(
                plan,
                scratch=tensor(plan.scratch_specs()[0].shape, torch.uint8),
                **inputs,
                raw_beta=tensor((32, 1)),
                A_log=tensor((1,), torch.float32),
                dt_bias=tensor((1, 128), torch.float32),
                recurrent_state=tensor((4, 1, 128, 128), torch.float32),
                cu_seqlens=tensor((2,), torch.int32),
                initial_state_indices=tensor((1,), torch.int64),
                final_state_indices=tensor((1,), torch.int64),
                checkpoint_state_indices=checkpoint_indices,
                checkpoint_offsets=tensor((1, 2), torch.int32),
                num_seqs=tensor((1,), torch.int32),
                num_tokens=tensor((1,), torch.int32),
                output=tensor((32, 1, 128)),
            )


@pytest.mark.parametrize("capacity", [2, 4])
def test_multi_checkpoint_caps_validate_matrix_width_before_live_scalar_dtype(capacity):
    from torch._subclasses.fake_tensor import FakeTensorMode

    caps = impl.Caps(
        device="cuda:0",
        max_tokens=64,
        max_seqs=2,
        max_state_slots=16,
        heads=1,
        checkpoint_export=True,
        max_checkpoints=capacity,
    )
    plan = impl._materialize_plan(
        caps, v_split=64, k_split=1, stages=3, window_tiles=4, policy_resolution=None
    )
    with FakeTensorMode():

        def tensor(shape, dtype=torch.bfloat16):
            return torch.empty(shape, device="cuda:0", dtype=dtype)

        # Fake tensors have no distinct device addresses. Stop before alias checks.
        with pytest.raises((ValueError, TypeError), match="num_tokens must have"):
            impl.bind(
                plan,
                scratch=tensor(plan.scratch_specs()[0].shape, torch.uint8),
                **{name: tensor((64, 1, 128)) for name in ("q", "k", "v", "raw_g")},
                raw_beta=tensor((64, 1)),
                A_log=tensor((1,), torch.float32),
                dt_bias=tensor((1, 128), torch.float32),
                recurrent_state=tensor((16, 1, 128, 128), torch.float32),
                cu_seqlens=tensor((3,), torch.int32),
                initial_state_indices=tensor((2,), torch.int32),
                final_state_indices=tensor((2,), torch.int32),
                checkpoint_state_indices=tensor((2, capacity), torch.int32),
                checkpoint_offsets=tensor((2, capacity), torch.int32),
                num_seqs=tensor((1,), torch.int32),
                num_tokens=tensor((1,), torch.int16),
                output=tensor((64, 1, 128)),
            )
    resolved = PolicyContext.for_identity(GB10, registry=ProfileRegistry()).resolve(
        KDA_PREFILL_POLICY, query(capacity)
    )
    assert resolved.source is PolicySource.HEURISTIC
    with pytest.raises(ValueError, match="supports only NVIDIA GB10"):
        PolicyContext.for_identity(None).resolve(KDA_PREFILL_POLICY, query(capacity))


@pytest.mark.parametrize("inplace", [False, True])
def test_four_exports_equal_independent_prefix_recurrences(inplace):
    inputs = make_inputs(lengths=[80], heads=1, seed=842, state_slots=8)
    if inplace:
        inputs["final"][0] = inputs["initial"][0]
    baseline_output, baseline_pool = run_oracle(inputs)
    inputs["checkpoint_slots"] = torch.tensor([[2, 3, 4, 5]], dtype=torch.int32)
    inputs["checkpoint_offsets"] = torch.tensor([[64, 16, 48, 32]], dtype=torch.int32)
    output, pool = run_oracle(inputs, max_checkpoints=4)
    torch.testing.assert_close(output, baseline_output, rtol=0, atol=0)
    torch.testing.assert_close(
        pool[int(inputs["final"][0])],
        baseline_pool[int(inputs["final"][0])],
        rtol=0,
        atol=0,
    )
    for slot, offset in zip((2, 3, 4, 5), (64, 16, 48, 32), strict=True):
        _, expected, _ = recurrent_kda(
            *(inputs[name][:offset] for name in ("q", "k", "v", "raw_g", "raw_beta")),
            inputs["A_log"],
            inputs["dt_bias"],
            lower_bound=-5.0,
            initial_state=inputs["pool"][int(inputs["initial"][0])],
        )
        torch.testing.assert_close(pool[slot], expected, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("fault", ["offset", "slot", "initial"])
def test_four_checkpoint_nonadjacent_aliases_are_rejected(fault):
    args = metadata()
    args.update(
        cu_seqlens=[0, 80],
        initial_state_indices=[0],
        final_state_indices=[1],
        checkpoint_state_indices=[[2, 3, 4, 5]],
        checkpoint_offsets=[[16, 32, 48, 64]],
        num_seqs=1,
        num_tokens=80,
        null_state_index=None,
        max_checkpoints=4,
    )
    if fault == "offset":
        args["checkpoint_offsets"][0][3] = 16
    elif fault == "slot":
        args["checkpoint_state_indices"][0][3] = 2
    else:
        args["checkpoint_state_indices"][0][3] = 0
    with pytest.raises(ValueError):
        validate_metadata(**args)
