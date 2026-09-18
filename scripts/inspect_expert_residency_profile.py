#!/usr/bin/env python3
"""Verify a residency artifact's integrity and print its placement and provenance."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from b12x.moe.fused_moe.automatic import ResidencyProfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--checkpoint-fingerprint")
    parser.add_argument("--workload")
    args = parser.parse_args()
    profile = ResidencyProfile.from_dict(json.loads(args.profile.read_text()))
    if args.checkpoint_fingerprint is not None and profile.model.checkpoint_fingerprint != args.checkpoint_fingerprint:
        parser.error("checkpoint fingerprint differs from the placement artifact")
    if args.workload is not None and profile.workload != args.workload:
        parser.error("workload differs from the placement artifact")
    print(json.dumps(dict(profile_hash=profile.profile_hash, checkpoint=profile.model.checkpoint_fingerprint,
        workload=profile.workload, phase=profile.phase, provenance=profile.provenance,
        algorithm=profile.algorithm, created_at=profile.created_at, converged=profile.converged,
        termination=profile.termination, expected_cold_fraction=profile.expected_cold_fraction,
        memory=profile.memory, workspace=profile.model.workspace_estimate(),
        layers=[dict(layer=p.layer, experts=p.total_experts, hot=len(p.hbm_expert_ids), cold=len(p.grace_expert_ids),
            observations=sum(p.selection_counts), expected_cold_fraction=p.expected_cold_fraction)
            for p in profile.placements]), indent=2))


if __name__ == "__main__":
    main()
