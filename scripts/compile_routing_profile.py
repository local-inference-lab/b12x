#!/usr/bin/env python3
"""Cross-compile configurable SM103 routing counters without initializing CUDA."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--sample-every", type=int, nargs="+", default=[1, 128])
    parser.add_argument("--runtime-token-limit", action="store_true",
                        help="Include the prepared engine phase/extent setter")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import torch
    from b12x.moe.fused_moe.routing_profile import RoutingProfileQuery, _compile
    from scripts._sm103_source import package_source_sha256, source_identity
    manifest = dict(command=sys.argv, source=source_identity(ROOT), source_sha256=package_source_sha256(ROOT),
        toolchain={name: importlib.metadata.version(name) for name in ("torch", "nvidia-cutlass-dsl", "triton")},
        target="sm_103a", status="running", queries=[], programs=0)
    try:
        for sample in dict.fromkeys(args.sample_every):
            query = RoutingProfileQuery(layers=(("qualification", args.experts),), max_tokens=args.capacity,
                max_top_k=args.top_k, sample_every=sample,
                runtime_token_limit=args.runtime_token_limit)
            programs = _compile(query, target="sm_103a", offline_dir=args.output_dir/f"every-{sample}")
            manifest["queries"].append(asdict(query))
            manifest["programs"] += len(programs)
        if torch.cuda.is_initialized():
            raise RuntimeError("offline counter compilation initialized CUDA")
        if package_source_sha256(ROOT) != manifest["source_sha256"]:
            raise RuntimeError("package source changed during compilation")
        manifest["status"] = "passed"
    except BaseException:
        manifest["status"] = "failed"
        manifest["error"] = traceback.format_exc()
        raise
    finally:
        manifest["cuda_initialized"] = torch.cuda.is_initialized()
        manifest["artifacts"] = {str(p.relative_to(args.output_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.output_dir.rglob("*") if p.is_file() and p.suffix in (".ptx", ".cubin", ".mlir")}
        (args.output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
