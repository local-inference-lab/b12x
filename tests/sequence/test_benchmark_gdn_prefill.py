"""CPU checks for prefill evidence boundaries, pairing, and artifact handling."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import gzip
import json

import pytest

from benchmarks import benchmark_gdn_decode
from benchmarks._gdn_prefill_race import balanced_order, policy_context_from_file, select_cases
from benchmarks.benchmark_delta_prefill_regression import compare_reports


def test_prefill_corpus_has_each_geometry_and_requested_length():
    cases = select_cases("all")
    assert len(cases) == 40
    assert {(c.key_heads, c.value_heads) for c in cases} == {(2, 6), (4, 12), (8, 24), (16, 48)}
    for heads in (2, 4, 8, 16):
        lengths = {c.lengths for c in cases if c.key_heads == heads}
        assert lengths == {(n,) for n in (16, 64, 256, 1024, 4096, 8192, 32768)} | {
            (1024,)*4, (512,)*8, (4096, 1024, 127, 1)}


def test_case_subset_preserves_requested_order_and_rejects_duplicates():
    first, last = select_cases("all")[0], select_cases("all")[-1]
    assert select_cases(f"{last.name},{first.name}") == (last, first)
    with pytest.raises(ValueError, match="duplicate"):
        select_cases(f"{first.name},{first.name}")
    with pytest.raises(ValueError, match="unknown"):
        select_cases("decode")


def test_arm_pairing_balances_each_position():
    arms = ("b12x", "flashinfer_auto", "flashinfer_no_cp")
    order = [balanced_order(arms, i) for i in range(100)]
    assert sum(x[0] == "b12x" for x in order) == 50
    assert sum(x[-1] == "b12x" for x in order) == 50
    assert all(set(row) == set(arms) for row in order)


def test_prefill_cli_refuses_to_overwrite_evidence_before_gpu_access(tmp_path):
    path = tmp_path/"race.json"
    path.write_text("retain")
    with pytest.raises(SystemExit):
        benchmark_gdn_decode.main(["--operation", "prefill", "--race", "flashinfer", "--json", str(path)])
    assert path.read_text() == "retain"


@pytest.mark.parametrize("args", [
    ["--race", "flashinfer"], ["--capacity-tokens", "1024"],
    ["--policy-profile", "profile.json"],
    ["--profile-replays", "1"], ["--operation", "prefill", "--profile-replays", "-1"],
    ["--operation", "prefill", "--capacity-columns", "4"],
    ["--operation", "prefill", "--cases", "missing"],
])
def test_incompatible_prefill_flags_fail_before_gpu_access(args):
    with pytest.raises(SystemExit):
        benchmark_gdn_decode.main(args)


def test_regression_gate_flags_each_case_and_rejects_different_evidence():
    baseline = {"provenance": dict(input_contract_sha256="abc", seed=42, device_uuid="gpu1", torch="2", cutlass="4"),
                "reports": [{"case": "short", "median_us": 100}, {"case": "long", "median_us": 1000}]}
    candidate = deepcopy(baseline)
    candidate["reports"][0]["median_us"] = 106
    candidate["reports"][1]["median_us"] = 950
    result = compare_reports(baseline, candidate)
    assert result["short"] == {"candidate_over_baseline": 1.06, "exceeds_5_percent": True}
    assert not result["long"]["exceeds_5_percent"]
    candidate["provenance"]["device_uuid"] = "gpu2"
    with pytest.raises(ValueError, match="device_uuid"):
        compare_reports(baseline, candidate)


@pytest.mark.parametrize("compressed", [False, True])
def test_explicit_profile_requires_matching_device_and_query(tmp_path, compressed):
    from b12x.policy import DeviceIdentity, PolicySource, PreplannedPolicyNotFoundError
    from b12x.policy.generation.providers.delta_prefill import prefill_cases
    from b12x.sequence.gdn_prefill._policy import GDN_PREFILL_POLICY, GdnPrefillQuery

    identity = DeviceIdentity(vendor="nvidia", product_name="Synthetic prefill GPU",
                              compute_capability=(12, 1), sm_count=48)
    query = GdnPrefillQuery(**prefill_cases("gdn")[0].query.to_dict())
    payload = {"profile": {
        "profile_id": "synthetic.prefill", "targets": [asdict(identity)],
        "components": [{"component_id": "sequence.gdn_prefill",
                        "query_schema_version": GDN_PREFILL_POLICY.query_schema_version,
                        "config_schema_version": GDN_PREFILL_POLICY.config_schema_version,
                        "rules": [{"name": "unit-test",
                        "exact": query.profile_fields(), "ranges": {}, "config": {"backend": "cutedsl"}}]}],
    }}
    path = tmp_path/("profile.json.gz" if compressed else "profile.json")
    raw = json.dumps(payload).encode()
    path.write_bytes(gzip.compress(raw) if compressed else raw)
    policy = policy_context_from_file(path, identity)
    assert policy.resolve(GDN_PREFILL_POLICY, query).source is PolicySource.PREPLANNED
    with pytest.raises(PreplannedPolicyNotFoundError):
        policy.resolve(GDN_PREFILL_POLICY, replace(query, max_tokens=query.max_tokens+1))
    with pytest.raises(ValueError, match="does not match device"):
        policy_context_from_file(path, replace(identity, sm_count=47))
