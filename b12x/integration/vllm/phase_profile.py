"""Validated static placement from separate, immutable calibration phases."""

from b12x.moe.fused_moe.residency import ExpertResidencyPlan, balanced_phase_profile


OBJECTIVE = {
    "name": "normalized_prefill_decode",
    "version": 1,
    "normalization": "per_layer_phase_total",
    "prefill_weight": [1, 2],
    "decode_weight": [1, 2],
    "tie_break": "ascending_logical_expert_id",
}


def placements(profile):
    rows = {}
    for row in profile.get("phase_counts", []):
        key = row["layer"], row["phase"]
        if key in rows or row["phase"] not in ("decode", "prefill"):
            raise ValueError("duplicate or unknown calibration phase")
        rows[key] = row["counts"]
    expected = {(n, p) for n in profile["placements"] for p in ("decode", "prefill")}
    if set(rows) != expected:
        raise ValueError("balanced profile requires every layer in both phases")
    result = {}
    for name, value in profile["placements"].items():
        prior = ExpertResidencyPlan.from_dict(value)
        result[name] = balanced_phase_profile(
            prefill_counts=rows[name, "prefill"],
            decode_counts=rows[name, "decode"],
            hot_count=len(prior.hbm_expert_ids),
            layer=prior.layer,
            model_fingerprint=prior.model_fingerprint,
            workload=prior.workload,
            provenance="equal normalized prefill/decode calibration; finite observations",
        ).to_dict()
    return result


def validate(profile):
    """A combined-phase placement is admitted only with its declared objective."""
    if profile.get("placement_objective") != OBJECTIVE:
        raise ValueError("unsupported phase placement objective")
    expected = {name: ExpertResidencyPlan.from_dict(value)
                for name, value in placements(profile).items()}
    actual = {name: ExpertResidencyPlan.from_dict(value)
              for name, value in profile["placements"].items()}
    if expected != actual:
        raise ValueError("phase placement differs from retained observations")
