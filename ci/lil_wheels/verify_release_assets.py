#!/usr/bin/env python3
"""Verify immutable B12X wheel release assets before publication or promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--beta-tag", required=True)
    parser.add_argument("--promotion", action="store_true")
    args = parser.parse_args()

    manifest_path = args.directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "local-inference-b12x-wheel-release/v1"
    assert manifest["source"]["commit"] == args.source_commit
    assert manifest["release_tag"] == args.beta_tag
    for package in manifest["packages"]:
        wheel = args.directory / package["file"]
        if not wheel.is_file():
            wheel = args.directory / "wheels" / package["file"]
        assert wheel.is_file()
        assert sha256(wheel) == package["sha256"]
    if args.promotion:
        assert (args.directory / "stable-promotion.json").is_file()
    print("B12X release assets: PASS")


if __name__ == "__main__":
    main()
