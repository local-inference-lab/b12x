"""Synthetic policy and resume tests; GPU measurements belong to the generators."""

from __future__ import annotations

from dataclasses import replace
from typing import get_type_hints

import pytest

from b12x.policy import (
    ComponentProfile, DeviceIdentity, GpuProfile, InvalidPreplannedPolicyError,
    PolicyContext, PolicyMode, PolicySource, ProfileRegistry, ProfileRule,
)
from b12x.policy.generation import CheckpointStore, GenerationContext, GenerationSettings, SweepMeasurement
from b12x.policy.generation.progress import NullProgressReporter
from b12x.policy.generation.providers.delta_prefill import (
    GdnPrefillGenerator, KdaPrefillGenerator, prefill_candidates, prefill_cases,
)
from b12x.sequence.gdn_prefill._policy import GDN_PREFILL_POLICY, GdnPrefillConfig, GdnPrefillQuery
from b12x.sequence.kda_prefill._policy import KDA_PREFILL_POLICY, KdaPrefillConfig, KdaPrefillQuery
from b12x.sequence._shared.delta_prefill.workspace import (
    SM12X_SHARED_MEMORY_LIMIT, recurrence_shared_bytes,
)


DEVICE = DeviceIdentity(vendor="nvidia", product_name="Synthetic prefill GPU", compute_capability=(12, 1), sm_count=48)


@pytest.mark.parametrize("mode,expected", [(0, 32688), (1, 24496), (2, 24500),
                                          (3, 24500), (4, 32688), (5, 32692)])
def test_segment_recurrence_shared_memory_accounts_for_pipeline_and_termination(mode, expected):
    assert recurrence_shared_bytes(64, 1, 2, 2081, max_sequence_tiles=64,
                                   summary_mode=mode) == expected


def _contract(recipe):
    fields = prefill_cases(recipe)[0].query.to_dict()
    if recipe == "gdn":
        return GDN_PREFILL_POLICY, GdnPrefillQuery(**fields), GdnPrefillConfig
    return KDA_PREFILL_POLICY, KdaPrefillQuery(**fields), KdaPrefillConfig


def _registry(policy, query, config):
    registry = ProfileRegistry()
    registry.register(GpuProfile(profile_id="synthetic.prefill", targets=(DEVICE,), components=(
        ComponentProfile(component_id=policy.component_id, query_schema_version=policy.query_schema_version,
                         config_schema_version=policy.config_schema_version,
                         rules=(ProfileRule.create(name="synthetic-test-case", exact=query.profile_fields(), ranges={},
                                                   config=config, evidence="synthetic-unit-test"),)),)))
    registry.freeze()
    return registry


@pytest.mark.parametrize("recipe", ["gdn", "kda"])
def test_prefill_policy_precedence_unknown_devices_and_invalid_matches(recipe):
    policy, query, cls = _contract(recipe)
    measured = cls(v_split=32, stages=2)
    registry = _registry(policy, query, measured.to_dict())
    context = PolicyContext.for_identity(DEVICE, registry=registry)
    result = context.resolve(policy, query)
    assert result.source is PolicySource.PREPLANNED
    assert result.config == measured
    override = cls(v_split=16)
    configured = context.with_override(policy.component_id, override)
    assert configured.resolve(policy, query).config is override
    call = cls(v_split=128)
    assert configured.resolve(policy, query, override=call).config is call
    unknown = PolicyContext.for_identity(replace(DEVICE, product_name="Unknown GPU"), registry=registry)
    assert unknown.resolve(policy, query).source is PolicySource.HEURISTIC
    assert context.resolve(policy, replace(query, max_tokens=query.max_tokens+1)).source is PolicySource.HEURISTIC
    heuristic = PolicyContext.for_identity(DEVICE, registry=registry, mode=PolicyMode.HEURISTIC_ONLY)
    assert heuristic.resolve(policy, query).source is PolicySource.HEURISTIC
    for config in (dict(backend="unknown"), dict(backend="cutedsl", stages=99),
                   dict(backend="cutedsl", v_split=128, k_split=2, stages=4),
                   dict(backend="cutedsl", window_tiles=True), dict(backend="cutedsl", extraneous=1)):
        invalid = PolicyContext.for_identity(DEVICE, registry=_registry(policy, query, config))
        with pytest.raises(InvalidPreplannedPolicyError):
            invalid.resolve(policy, query)


@pytest.mark.parametrize("recipe,count", [("gdn", 80), ("kda", 60)])
def test_prefill_candidate_contract_covers_legal_geometry_and_static_queries(recipe, count):
    cases = prefill_cases(recipe)
    assert len(cases) == count
    assert len({c.case_id for c in cases}) == count
    assert tuple(c.case_id for c in cases) == tuple(c.case_id for c in prefill_cases(recipe))
    policy, _, cls = _contract(recipe)
    for case in cases:
        query_cls = GdnPrefillQuery if recipe == "gdn" else KdaPrefillQuery
        query = query_cls(**case.query.to_dict())
        assert set(query.profile_fields()) == policy.query_fields
        assert not {"num_tokens", "num_seqs", "lengths"} & set(case.query)
        candidates = prefill_candidates(case)
        sequential = [c for c in candidates if c.config.get("algorithm", "sequential") == "sequential"]
        parallel = [c for c in candidates if c.config.get("algorithm") == "chunk_parallel"]
        windows = {candidate.config["window_tiles"] for candidate in sequential}
        assert 1 <= len(windows) <= 3
        assert len(sequential) == 33 * len(windows)
        assert len(parallel) == (132 if recipe == "gdn" else 0)
        assert len(candidates) == len({c.candidate_id for c in candidates})
        if parallel:
            assert {c.config["segment_tokens"] for c in parallel} == {128, 256, 512, 1024}
        for candidate in candidates:
            config = cls.from_profile(candidate.config)
            segmented = getattr(config, "algorithm", "sequential") == "chunk_parallel"
            shared_bytes = recurrence_shared_bytes(
                config.v_split, config.k_split, config.stages, config.window_tiles,
                max_sequence_tiles=config.segment_tokens // 16 if segmented else 0,
                summary_mode=5 if segmented and config.k_split == 1 else 0,
                reuse_value_buffer=recipe == "gdn",
            )
            if shared_bytes > SM12X_SHARED_MEMORY_LIMIT:
                with pytest.raises(ValueError, match="shared-memory"):
                    policy.validate_config(query, config, DEVICE)
            else:
                policy.validate_config(query, config, DEVICE)


@pytest.mark.parametrize("recipe", ["gdn", "kda"])
def test_public_prefill_plan_uses_typed_resolution_without_tensor_allocation(recipe, monkeypatch):
    from b12x.policy.generation.delta_prefill_cases import prefill_op

    op = prefill_op(recipe)
    policy, query, cls = _contract(recipe)
    config = cls(v_split=32, stages=2)
    context = PolicyContext.for_identity(DEVICE, registry=_registry(policy, query, config.to_dict()))
    devices = []
    monkeypatch.setattr(PolicyContext, "require_device", lambda self, device: devices.append(str(device)))
    geometry = {"key_heads": query.key_heads, "value_heads": query.value_heads} if recipe == "gdn" else {"heads": query.heads}
    caps = op.Caps(device="cuda:0", max_tokens=query.max_tokens, max_seqs=query.max_seqs,
                   max_state_slots=4, **geometry)
    planned = op.plan(caps, policy=context)
    assert devices == ["cuda:0"]
    assert planned.caps is caps
    assert planned.policy_resolution.source is PolicySource.PREPLANNED
    assert planned.policy_resolution.config == config
    assert (planned.v_split, planned.k_split, planned.stages) == (32, 1, 2)
    assert planned.scratch_specs()[0].shape[0] > 0
    assert get_type_hints(op.Plan.__init__)["caps"] is op.Caps
    assert get_type_hints(op.Binding.__init__)["plan"] is op.Plan


class _SyntheticSession:
    def __init__(self, calls):
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def candidates(self, case):
        return prefill_candidates(case)

    def measure(self, case, candidates):
        self.calls.append(case.case_id)
        return tuple(SweepMeasurement(candidate=c, latency_us=float(i+1), correct=True)
                     for i, c in enumerate(candidates))


@pytest.mark.parametrize("recipe,cls", [("gdn", GdnPrefillGenerator), ("kda", KdaPrefillGenerator)])
def test_prefill_generator_resumes_qualified_candidates(recipe, cls, tmp_path):
    calls = []
    cases = prefill_cases(recipe)[:2]
    generator = cls(cases=cases, benchmark_factory=lambda *args: _SyntheticSession(calls))
    context = GenerationContext(device=DEVICE, device_ordinal=0, work_dir=tmp_path,
                                source_revision="synthetic-test", settings=GenerationSettings())
    checkpoints = CheckpointStore(tmp_path/"checkpoints")
    first = generator.generate(context, progress=NullProgressReporter(), checkpoints=checkpoints)
    resumed = generator.generate(context, progress=NullProgressReporter(), checkpoints=checkpoints)
    assert calls == [case.case_id for case in cases]
    assert first.component == resumed.component
