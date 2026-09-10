#!/usr/bin/env python3
"""Render production-graph NVFP4 receipts without promoting failed timings."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def _qualified(receipt, case):
    return bool(
        receipt.get("version", 0) >= 4
        and case.get("qualified")
        and case.get("arm_identity_passed")
        and case.get("split_engaged")
        and case.get("correctness", {}).get("passed")
        and case.get("graph_check", {}).get("passed")
        and case.get("post_timing_correctness", {}).get("passed")
        and case.get("timing_allocation_stable")
        and case.get("timing_addresses_stable")
        and case.get("gpu_mode_check", {}).get("passed")
    )


def main() -> None:
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    r = json.loads(src.read_text())
    lines = []
    a = lines.append
    a("# NVFP4 split-materialized prefill — evidence")
    a("")
    a(f"- **Date (UTC):** {r['generated_utc']}")
    a(f"- **Commit:** `{r['commit']}` (branch `{r['branch']}`)")
    a(f"- **Actual worktree path:** `{r.get('worktree_path', 'not recorded')}`")
    worktree = "; ".join((r.get("worktree_status") or "clean").splitlines())
    a(f"- **Worktree status at launch:** `{worktree}`")
    a(f"- **Command working directory:** `{r.get('command_cwd', 'not recorded')}`")
    a(f"- **Invocation working directory:** `{r.get('invocation_cwd', 'not recorded')}`")
    a(f"- **GPU:** {r['gpu_name']}; device identity: `{json.dumps(r.get('gpu_device', {}))}`")
    a(f"- **Initial physical GPU/mode snapshot:** `{json.dumps(r.get('gpu_snapshot', {}))}`")
    a(f"- **Package versions:** `{json.dumps(r['package_versions'])}`")
    a(f"- **Initial process settings:** `{json.dumps(r.get('env', {}), sort_keys=True)}`")
    a(f"- **Explicit per-arm settings:** `{json.dumps(r.get('arm_settings', {}), sort_keys=True)}`")
    a(f"- **Fast math (both arms):** `{r.get('fast_math', 'not recorded')}`")
    a(f"- **Declared GPU-mode policy:** `{json.dumps(r.get('gpu_mode_policy', {}), sort_keys=True)}`")
    a(f"- **Samples per arm:** {r['iters']} iterations × {r['rounds']} rounds, "
      f"{r['warmup']} warmup replays per round; alternating lead arm, CUDA events.")
    a("- **Aggregation:** median of per-round medians; ratio = monolithic_us / split_us "
      "(>1 means split is faster). Raw per-replay samples and round order remain in JSON.")
    a(f"- **Command:** `{r['command']}`")
    a(f"- **argv:** `{json.dumps(r.get('argv', []))}`")
    a("")
    a("## Measured path and scope")
    a("")
    if r.get("version", 0) >= 2:
        a("Both arms use canonical `fused_moe.plan_weights/prepare_weights` and "
          "`plan_execution/prewarm/bind/run`, with shared synthetic NVFP4 weights, "
          "input, routing and scalar input scale. Each arm owns fixed-capacity "
          "scratch and output. One production CUDA graph is captured per arm with "
          "`B12X_NVFP4_DYNAMIC_MATERIALIZED=0` (monolithic) or `1` (split); the "
          "environment and its cache are restored afterward. Only graph replay is "
          "timed, with no Python policy/environment resolution or tensor allocation "
          "in the replay path. Tile and active-cluster values come from production, "
          "not a test launcher or a benchmark override.")
        a("")
        a("Engagement is verified on the actual dynamic launch resolved during "
          "capture: pass-through observers associate its compiled object with the "
          "backend supplied to the production compiler. The receipt records the "
          "backend's split flag, selected tile and actual active-cluster count. "
          "An unobserved cache identity, unengaged split, or misidentified monolithic "
          "arm cannot qualify a ratio. Observers are removed before replay/timing.")
    else:
        a("**Legacy receipt: not production-plan evidence.** This receipt predates "
          "the production graph benchmark. Its forced test-launcher geometry and "
          "separately constructed engagement probe cannot establish production "
          "dispatch or occupancy. No legacy timing is promoted to a qualified result here.")
    a("")
    a("These measurements do not establish equivalence to earlier forced, "
      "underoccupied test-launcher results, reproduce the PR's historical speedups, "
      "or justify reconciling those speedups by shape mix. Only freshly measured, "
      "qualified production-graph rows below support performance statements. This "
      "is synthetic kernel-path evidence, not a checkpoint/serving benchmark or "
      "a whole-workload average. Unengaged cases remain visible.")
    a("")
    a("## Qualification")
    a("")
    a("Before timing, each output must be finite and nonzero, cosine > 0.9999 "
      "against the independent torch `moe_reference_nvfp4` oracle, and global "
      "RMSE ≤ `max(8e-4, 3 · 2⁻⁸ · max|oracle|)`. The same cosine/RMSE gates "
      "apply split versus monolithic. The oracle uses direct-division NVFP4 "
      "quantization. `_bf16_output_bound` supplies the numerical bound; "
      "`max_abs` remains a diagnostic, not an acceptance gate. This does not "
      "claim that any max-absolute discrepancy is harmless.")
    a("")
    a("Both graph outputs are poisoned with NaN and rewritten twice, alternating "
      "the replay order, and rechecked against the oracle and each other. Live "
      "allocated bytes, cumulative allocation counts around replay, and bound "
      "tensor/scratch addresses must remain stable. Outputs, allocation counts "
      "and addresses are checked again after timing. Initial failures have no "
      "timing samples; post-timing failures retain raw samples as unqualified "
      "diagnostics only. No failed case gets headline latency or ratio.")
    a("")
    a("Timings are diagnostic-only, not formal release evidence. Both arms must "
      "remain P1 on the selected physical GPU with identical memory clocks and "
      "an SM-clock difference within the bound declared in the command and JSON, "
      "relative to the lower arm clock. Only throttle mask 0x0 is accepted by "
      "default. `--allow-software-power-cap` explicitly permits only 0x0/0x4 "
      "for interleaved Max-Q diagnostics; every other throttle reason is rejected. "
      "The observed delta, initial-to-active throttle-mask pairs, and inter-arm "
      "0x0/0x4 transitions are retained per case. The initial snapshot may be idle; "
      "P1 and clock-comparison requirements apply to the active arms. "
      "Receipts predating this complete check cannot qualify a timing ratio.")
    a("")
    a("## Qualified diagnostic results")
    a("")
    a("| E | K | n | top_k | M | split median (us) | mono median (us) | mono / split | status |")
    a("|---|---|---|---|---|---|---|---|---|")
    ratios = []
    for c in r["cases"]:
        s = c["shape"]
        qualified = _qualified(r, c)
        med = c.get("median_us") or {}
        if qualified:
            ratio = med["monolithic"] / med["split"]
            ratios.append(ratio)
            values = f"{med['split']:.2f} | {med['monolithic']:.2f} | {ratio:.3f}x"
        else:
            values = "— | — | —"
        status = "DIAGNOSTIC QUALIFIED" if qualified else c.get("status", "legacy/unqualified")
        a(f"| {s['E']} | {s['K']} | {s['n']} | {s['top_k']} | {s['M']} | {values} | {status} |")
    a("")
    if ratios:
        a(f"**Geomean mono/split ratio across {len(ratios)} qualified diagnostic shapes: "
          f"{statistics.geometric_mean(ratios):.3f}x.**")
    else:
        a("**No qualified timing result.**")
    a("")
    a("## Per-case identities and raw qualification diagnostics")
    for c in r["cases"]:
        s = c["shape"]
        a("")
        a(f"### E={s['E']}, K={s['K']}, n={s['n']}, top_k={s['top_k']}, M={s['M']}")
        a("")
        a(f"- Status: `{c.get('status', 'legacy/unqualified')}`; "
          f"split engaged: `{c.get('split_engaged')}`; qualified: `{_qualified(r, c)}`.")
        if c.get("error"):
            a(f"- Error: `{c['error']}`")
        for name, arm in c.get("arms", {}).items():
            a(f"- {name} identity/settings: `{json.dumps(arm, sort_keys=True)}`")
        a(f"- Initial correctness (raw metrics): `{json.dumps(c.get('correctness', {}), sort_keys=True)}`")
        graph = c.get("graph_check", {})
        a(f"- Graph poison/rewrite and stability: "
          f"`{json.dumps({key: value for key, value in graph.items() if key != 'addresses'}, sort_keys=True)}`")
        a(f"- Post-timing correctness: `{json.dumps(c.get('post_timing_correctness', {}), sort_keys=True)}`")
        a(f"- Post-timing allocation/address stability: "
          f"`{c.get('timing_allocation_stable')}` / `{c.get('timing_addresses_stable')}`.")
        samples = c.get("samples_us", {})
        counts = {name: sum(len(batch) for batch in rounds) for name, rounds in samples.items()}
        a(f"- Raw timing sample counts: `{json.dumps(counts)}`; "
          f"{'qualified' if _qualified(r, c) else 'not headline evidence'}.")
        a(f"- Per-arm active physical GPU/mode snapshots: "
          f"`{json.dumps(c.get('gpu_mode_active', {}), sort_keys=True)}`")
        a(f"- GPU-mode qualification: `{json.dumps(c.get('gpu_mode_check', {}), sort_keys=True)}`")
    a("")
    a("## Source artifact hashes (SHA-256)")
    a("")
    a(r.get("source_hash_scope", "Legacy source list is incomplete for imported input/oracle helpers."))
    if r.get("version", 0) >= 2:
        a("Hashes are collected again when saving, after lazy imports, so local input, "
          "oracle, preparation, planning, launch and compiler helpers are included.")
    a("")
    for path, digest in sorted(r.get("source_sha256", {}).items()):
        a(f"- `{path}`: {digest}")
    a("")
    a("## Raw data")
    a("")
    a(f"Receipt: `{src.name}`. Consult the JSON for raw samples and available "
      "correctness, identity and provenance fields. Version 2 records round "
      "order, fixed addresses and actual capture identities; version 3 adds the "
      "declared GPU-mode check; version 4 includes initial-to-active throttle "
      "transitions. Older receipts do not establish those facts. "
      "Failed or unengaged rows retain raw diagnostic samples, not speedup claims.")
    dst.write_text("\n".join(lines) + "\n")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
