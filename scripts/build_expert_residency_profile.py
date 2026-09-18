"""Build per-layer expert placements from a flat routing JSONL trace."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from b12x.moe.fused_moe.residency import profiles_from_trace, write_profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--geometry", type=Path, required=True,
                        help='JSON object mapping layer names to expert counts')
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--hot-count", type=int)
    budget.add_argument("--hot-bytes", type=int)
    parser.add_argument("--expert-bytes", type=int)
    parser.add_argument("--phase", choices=("all", "decode", "prefill"), default="decode")
    parser.add_argument("--model-fingerprint", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--provenance", required=True)
    args = parser.parse_args()
    with args.trace.open() as stream:
        profiles = profiles_from_trace((json.loads(line) for line in stream if line.strip()),
            experts_per_layer=json.loads(args.geometry.read_text()), hot_count=args.hot_count,
            hot_bytes=args.hot_bytes, expert_bytes=args.expert_bytes, phase=args.phase,
            model_fingerprint=args.model_fingerprint, workload=args.workload, provenance=args.provenance)
    write_profiles(args.output, profiles)
    print(json.dumps([{ "layer": p.layer, "profile_hash": p.profile_hash,
                      "hot_experts": len(p.hbm_expert_ids),
                      "expected_cold_fraction": p.expected_cold_fraction} for p in profiles], indent=2))


if __name__ == "__main__":
    main()
