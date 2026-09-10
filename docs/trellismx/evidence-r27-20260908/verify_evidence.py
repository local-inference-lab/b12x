"""Check pinned external KLD receipts; does not run or qualify the PR model."""

import argparse
import hashlib
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("evidence_dir", type=Path)
root = parser.parse_args().evidence_dir
manifest = json.loads((root / "SHA256.json").read_text())
for name, expected in manifest.items():
    path = root / name
    if not (path.parent == root and path.is_file()):
        raise ValueError(name)
    if not (hashlib.sha256(path.read_bytes()).hexdigest() == expected):
        raise ValueError(name)
current = json.loads((root / "comparison.json").read_text())
audit = json.loads((root / "audit.json").read_text())
historical = json.loads((root / "historical-fp8-audit.json").read_text())
if not (audit["status"] == historical["status"] == "passed"):
    raise ValueError('Evidence validation failed at check line 20')
if not (current["windows"] == historical["windows"] == 32):
    raise ValueError('Evidence validation failed at check line 21')
if not (current["true_decode_rows_per_window"] == 2046):
    raise ValueError('Evidence validation failed at check line 22')
if not (historical["true_decode_rows_per_window"] == 2046):
    raise ValueError('Evidence validation failed at check line 23')
if not (historical["conditions"]["kv_dtype"] == "fp8_ds_mla"):
    raise ValueError('Evidence validation failed at check line 24')
if not (historical["conditions"]["attention"] == "FLASHINFER_MLA_SPARSE_SM120"):
    raise ValueError('Evidence validation failed at check line 25')
if not (historical["dcp"] == 1 and not historical["mtp"]):
    raise ValueError('Evidence validation failed at check line 26')
if not (current["capture_image_local_id"] == (
    "sha256:a7fde8169ec24fff3d7f1a7a8a50375a90e6c9e8cf71254680a281f54be37ca6"
)):
    raise ValueError('Evidence validation failed at check line 27')
audited = {entry["arm"]: entry for entry in audit["arms"]}
if set(audited) != {"nvfp4", "fp8"} or len(audit["arms"]) != 2:
    raise ValueError("Audit must contain exactly both KV arms")
for entry in audited.values():
    for flag in ("all_receipt_hashes_and_masks_pass", "all_prefix_hit_counters_zero",
                 "raw_logits_retired_after_scoring"):
        if entry.get(flag) is not True:
            raise ValueError(f"Audit flag failed: {entry['arm']} {flag}")
    if (entry["windows"], entry["prediction_rows"], entry["true_decode_rows"]) != (32, 65504, 65472):
        raise ValueError("Audit row counts do not match CF32")
if audit["arms"] != current["receipt_audit"]:
    raise ValueError("Comparison and audit arms differ")
binding = historical["comparison_binding"]
if binding["file"] != "comparison.json" or binding["sha256"] != hashlib.sha256((root / "comparison.json").read_bytes()).hexdigest():
    raise ValueError("Historical comparison binding mismatch")
ids = None
for arm in ["nvfp4", "fp8"]:
    rows = current["arms"][arm]["per_window"]
    now = [r["window_id"] for r in rows]
    if not (len(now) == len(set(now)) == 32):
        raise ValueError('Evidence validation failed at check line 34')
    if ids is None:
        ids = now
    if not (now == ids):
        raise ValueError('Evidence validation failed at check line 37')
    for row in rows:
        if not (row["prediction_rows"] == 2047 and row["true_decode_rows"] == 2046):
            raise ValueError('Evidence validation failed at check line 39')
        if not (len(row["score_sha256"]) == len(row["raw_sha256"]) == 64):
            raise ValueError('Evidence validation failed at check line 40')
    mean = math.fsum(r["true_decode_mean_kld"] for r in rows) / 32
    if not (math.isclose(
        mean, current["arms"][arm]["mean_true_decode_kld"], abs_tol=1e-15, rel_tol=0
    )):
        raise ValueError('Evidence validation failed at check line 42')
old = historical["per_window"]
if not ([r["window_id"] for r in old] == ids):
    raise ValueError('Evidence validation failed at check line 46')
for old_row, row in zip(old, current["arms"]["fp8"]["per_window"], strict=True):
    if not (old_row["token_values_sha256"] == row["token_sha256"]):
        raise ValueError('Evidence validation failed at check line 48')
if not (math.isclose(
    math.fsum(r["true_decode_mean_kld"] for r in old) / 32,
    historical["mean_true_decode_kld"],
    abs_tol=1e-15,
    rel_tol=0,
)):
    raise ValueError('Evidence validation failed at check line 49')
delta = (
    current["arms"]["fp8"]["mean_true_decode_kld"]
    - current["arms"]["nvfp4"]["mean_true_decode_kld"]
)
if not (math.isclose(delta, current["fp8_minus_nvfp4"], abs_tol=1e-15, rel_tol=0)):
    raise ValueError('Evidence validation failed at check line 59')
print(
    "PASS: pinned external evidence, 64 current and 32 historical FP8 window receipts; no PR-head GPU qualification"
)
