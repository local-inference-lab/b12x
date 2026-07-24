#!/usr/bin/env python
"""Collect config.json from multiple FP6 checkpoint directories into one file.

Useful for comparing quantization coverage (exclude_modules, weight_format,
activation_format) across experimental exports without manually opening each dir.

Example (defaults match the known 27B dense FP6 variants on the main rig):

    python scripts/collect_fp6_configs.py \
        --out /tmp/fp6_configs_bundle.json

Custom directories:

    python scripts/collect_fp6_configs.py \
        --dirs /media/fmodels/TheHouseOfTheDude/Qwen3.6-27B-FP6 \
               /media/fmodels/TheHouseOfTheDude/Qwen3.6-27B-FP6-FRESH \
        --out fp6_configs_bundle.json

Human-readable summary only:

    python scripts/collect_fp6_configs.py --out /tmp/fp6_configs.txt --format text
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime, timezone

# Known 27B dense FP6 checkpoints on the main rig (Jun 2026).
DEFAULT_DIRS: tuple[str, ...] = (
    "/media/fmodels/TheHouseOfTheDude/qwen3-6_27B_dense_fp6",
    "/media/fmodels/TheHouseOfTheDude/qwen3-6_27B_dense_fp6_full",
    "/media/fmodels/TheHouseOfTheDude/qwen3-6_27B_dense_fp6_gr",
    "/media/fmodels/TheHouseOfTheDude/qwen3-6_27B_dense_fp6_la",
    "/media/fmodels/TheHouseOfTheDude/Qwen3.6-27B-FP6",
    "/media/fmodels/TheHouseOfTheDude/Qwen3.6-27B-FP6-DequantBF16",
    "/media/fmodels/TheHouseOfTheDude/Qwen3.6-27B-FP6-FRESH",
)


def _dir_size_gb(path: pathlib.Path) -> float | None:
    if not path.is_dir():
        return None
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return round(total / (1024**3), 2)


def _load_config(path: pathlib.Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _summarize_qc(qc: dict | None) -> dict:
    if not qc:
        return {}
    exclude = qc.get("exclude_modules") or []
    return {
        "quant_method": qc.get("quant_method"),
        "quant_algo": qc.get("quant_algo"),
        "weight_format": qc.get("weight_format"),
        "activation_format": qc.get("activation_format"),
        "group_size": qc.get("group_size"),
        "exclude_modules_count": len(exclude),
        "exclude_modules": exclude,
        "producer": qc.get("producer"),
    }


def collect(dirs: list[pathlib.Path]) -> dict:
    entries: list[dict] = []
    for d in dirs:
        cfg_path = d / "config.json"
        entry: dict = {
            "label": d.name,
            "path": str(d),
            "config_path": str(cfg_path),
            "exists": d.is_dir(),
            "config_exists": cfg_path.is_file(),
            "size_gb": _dir_size_gb(d),
        }
        if cfg_path.is_file():
            try:
                mtime = datetime.fromtimestamp(
                    cfg_path.stat().st_mtime, tz=timezone.utc
                ).isoformat()
            except OSError:
                mtime = None
            entry["config_mtime_utc"] = mtime
            try:
                full_cfg = _load_config(cfg_path)
                entry["config"] = full_cfg
                entry["quantization_config_summary"] = _summarize_qc(
                    full_cfg.get("quantization_config")
                )
            except (OSError, json.JSONDecodeError) as exc:
                entry["error"] = str(exc)
        else:
            entry["error"] = "config.json not found"
        entries.append(entry)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_count": len(entries),
        "entries": entries,
    }


def _format_text(bundle: dict) -> str:
    lines: list[str] = [
        f"FP6 config bundle — generated {bundle['generated_at_utc']}",
        f"Checkpoints: {bundle['checkpoint_count']}",
        "",
    ]
    for e in bundle["entries"]:
        lines.append("=" * 72)
        lines.append(f"label: {e['label']}")
        lines.append(f"path:  {e['path']}")
        if e.get("size_gb") is not None:
            lines.append(f"size:  {e['size_gb']} GB")
        if e.get("config_mtime_utc"):
            lines.append(f"config.json mtime (UTC): {e['config_mtime_utc']}")
        if e.get("error"):
            lines.append(f"ERROR: {e['error']}")
            lines.append("")
            continue
        summary = e.get("quantization_config_summary") or {}
        lines.append(
            f"quant: {summary.get('quant_method')}/{summary.get('quant_algo')} "
            f"W={summary.get('weight_format')} A={summary.get('activation_format')} "
            f"exclude={summary.get('exclude_modules_count')}"
        )
        exclude = summary.get("exclude_modules") or []
        if exclude:
            lines.append("exclude_modules:")
            for mod in exclude[:40]:
                lines.append(f"  - {mod}")
            if len(exclude) > 40:
                lines.append(f"  ... ({len(exclude) - 40} more)")
        lines.append("")
        lines.append("--- full config.json ---")
        lines.append(json.dumps(e.get("config", {}), indent=2, sort_keys=True))
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dirs",
        nargs="*",
        default=list(DEFAULT_DIRS),
        help="checkpoint directories (default: known 27B FP6 variants on main rig)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="output file (.json or .txt; use --format to override)",
    )
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default=None,
        help="json (default) or text; inferred from --out suffix if omitted",
    )
    args = parser.parse_args()

    out_path = pathlib.Path(args.out)
    fmt = args.format
    if fmt is None:
        fmt = "text" if out_path.suffix.lower() in (".txt", ".md") else "json"

    dirs = [pathlib.Path(p) for p in args.dirs]
    bundle = collect(dirs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        out_path.write_text(
            json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        out_path.write_text(_format_text(bundle) + "\n", encoding="utf-8")

    missing = sum(1 for e in bundle["entries"] if e.get("error"))
    print(f"wrote {out_path} ({fmt}, {len(dirs)} dirs, {missing} errors)")
    if missing:
        for e in bundle["entries"]:
            if e.get("error"):
                print(f"  missing: {e['path']} — {e['error']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
