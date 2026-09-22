"""Construct one equal-weight phase profile without changing resident geometry."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from b12x.integration.vllm.expert_cache import digest
from b12x.integration.vllm.phase_profile import OBJECTIVE, placements, validate
from benchmarks.moe.expert_cache_capacity import calibration_counts


def build(profile, receipt):
    calibration_counts(profile, receipt)
    observed = next(r["result"] for r in receipt if r["kind"] == "profile")
    if isinstance(observed, list):
        observed = observed[0]
    if observed["snapshot"]["layers"] != profile.get("phase_counts"):
        raise ValueError("retained phase counts differ from calibration receipt")
    result = deepcopy(profile)
    result.pop("hash")
    result["placements"] = placements(profile)
    result["placement_objective"] = deepcopy(OBJECTIVE)
    result["construction"] = {
        "calibrated_profile_hash": profile["hash"],
        "method": "fixed equal normalized phase weights",
    }
    validate(result)
    result["hash"] = digest(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        json.loads(args.profile.read_text()),
        [json.loads(r) for r in args.receipt.read_text().splitlines()],
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
