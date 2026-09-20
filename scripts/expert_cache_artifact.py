#!/usr/bin/env python3
"""Bind a built companion wheel to source, then verify the installed copy."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from b12x.testing.artifacts import sha256, verify_installed, wheel_manifest
from scripts._sm103_source import source_identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wheel", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--build-log", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--verify-installed", action="store_true")
    args = p.parse_args()
    result = wheel_manifest(args.wheel, args.source)
    result.update(
        source=source_identity(args.source), build_log_sha256=sha256(args.build_log)
    )
    if args.verify_installed:
        result["installed"] = verify_installed(result)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)


if __name__ == "__main__":
    main()
