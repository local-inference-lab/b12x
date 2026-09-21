#!/usr/bin/env python3
"""Audit checkpoint headers before allocating model storage or downloading weights.

Remote mode fetches only configuration, the index and bounded safetensors headers
from an explicit immutable revision. It never falls back to full-shard downloads.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from b12x.integration.vllm.checkpoint import (
    HEADER_LIMIT,
    audit,
    decode_json,
    read_json,
    shard_name,
)
from scripts._sm103_source import source_identity


def fetch(url, limit, byte_range=None):
    headers = (
        {}
        if byte_range is None
        else {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"}
    )
    with urlopen(Request(url, headers=headers), timeout=60) as response:
        if byte_range is not None:
            start, end = byte_range
            if response.status != 206 or not response.headers.get(
                "Content-Range", ""
            ).startswith(f"bytes {start}-{end}/"):
                raise ValueError("server did not honor bounded checkpoint range")
        data = response.read(limit + 1)
        if len(data) > limit or (
            byte_range is not None and len(data) != byte_range[1] - byte_range[0] + 1
        ):
            raise ValueError("checkpoint response exceeds limit or is truncated")
        return data, dict(response.headers)


def export_headers(repository, revision, output):
    if not re.fullmatch(
        r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository
    ) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(
            "metadata export requires a repository and immutable 40-digit revision"
        )
    output.mkdir(parents=True, exist_ok=False)
    (output / "headers").mkdir()
    base = f"https://huggingface.co/{repository}/resolve/{revision}"
    for name in ("config.json", "hf_quant_config.json", "model.safetensors.index.json"):
        data, _ = fetch(f"{base}/{name}", HEADER_LIMIT)
        decode_json(data)
        (output / name).write_bytes(data)
    records = []
    for name in sorted(
        set(read_json(output / "model.safetensors.index.json")["weight_map"].values())
    ):
        shard_name(name)
        prefix, headers = fetch(f"{base}/{name}?audit=length", 8, (0, 7))
        size = struct.unpack("<Q", prefix)[0]
        if not 2 <= size <= HEADER_LIMIT:
            raise ValueError("safetensors header exceeds bounded read")
        data, header_response = fetch(
            f"{base}/{name}?audit=header", size, (8, 7 + size)
        )

        # HTTP header names are case-insensitive; preserve the server's actual total.
        def total(h):
            return int(
                next(v for k, v in h.items() if k.lower() == "content-range").split(
                    "/"
                )[-1]
            )

        if total(headers) != total(header_response):
            raise ValueError("checkpoint shard size changed between range reads")
        decode_json(data)
        (output / "headers" / (name + ".json")).write_bytes(data)
        records.append(
            dict(
                shard=name,
                file_bytes=total(headers),
                header_bytes=size,
                header_sha256=hashlib.sha256(data).hexdigest(),
            )
        )
    manifest = dict(repo=repository, revision=revision, shards=records)
    (output / "header-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model",
        type=Path,
        help="Complete local checkpoint; validates global scale values",
    )
    group.add_argument(
        "--metadata", type=Path, help="Existing header export; no weight value claim"
    )
    group.add_argument("--repository", help="Fetch metadata only from Hugging Face")
    parser.add_argument("--revision")
    parser.add_argument(
        "--check-block-scales",
        action="store_true",
        help="Validate all native Qwen3-Next target block-scale values locally",
    )
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument(
        "--loader-coverage", type=Path,
        help="Validate an actual CPU-source loader receipt against local target tensors",
    )
    parser.add_argument("--device-bytes", type=int, required=True)
    parser.add_argument("--host-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--identity-output",
        type=Path,
        help="Hash local contents once and write a reusable immutable-file receipt",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("report path already exists")
    if args.identity_output and not args.model:
        parser.error("content identity requires complete local files")
    if args.check_block_scales and not args.model:
        parser.error("block scale values require complete local files")
    if args.loader_coverage and not args.model:
        parser.error("loader coverage requires complete local files")
    if min(args.device_bytes, args.host_bytes) <= 0:
        parser.error("memory capacities must be positive")
    if args.repository:
        if not args.revision or not args.metadata_output:
            parser.error("remote metadata requires --revision and --metadata-output")
        export_headers(args.repository, args.revision, args.metadata_output)
    directory = args.model or args.metadata or args.metadata_output
    result = audit(
        directory, headers_only=args.model is None, check_values=args.model is not None
    )
    if args.identity_output:
        from b12x.integration.vllm.checkpoint_identity import checkpoint_identity

        result["content_identity"] = checkpoint_identity(
            directory, output=args.identity_output
        )
    if args.check_block_scales:
        from b12x.integration.vllm.checkpoint import validate_block_scales

        result["block_scale_values"] = validate_block_scales(directory)
    if args.loader_coverage:
        from b12x.integration.vllm.checkpoint import validate_loader_coverage
        from b12x.testing.artifacts import sha256

        if args.loader_coverage.stat().st_size > 256 << 20:
            raise ValueError("loader coverage exceeds the bounded report size")
        with args.loader_coverage.open() as stream:
            result["loader_coverage"] = validate_loader_coverage(
                directory, json.load(stream)
            )
        result["loader_coverage_sha256"] = sha256(args.loader_coverage)
    result.update(
        source=source_identity(ROOT),
        command=sys.argv,
        supplied_device_bytes=args.device_bytes,
        supplied_host_bytes=args.host_bytes,
        routed_packed_alone_exceeds_device=result["routed_weight_lower_bound_bytes"]
        > args.device_bytes,
        expert_representations_exceed_host=result["host_expert_lower_bound_bytes"]
        > args.host_bytes,
        combined_host_lower_bound_exceeds_host=result.get(
            "host_with_ple_lower_bound_bytes", result["host_expert_lower_bound_bytes"]
        )
        > args.host_bytes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "required_target_bytes",
                    "host_expert_lower_bound_bytes",
                    "routed_packed_alone_exceeds_device",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
