#!/usr/bin/env bash
# Nsight Compute readiness check for FP6 kernel profiling (plan item E0).
#
# Non-destructive by default: it detects, verifies, and prints exact
# remediation. Pass --apply to write the modprobe drop-in that grants non-root
# GPU counter access (still requires sudo, and still requires a reboot).
#
# Why non-root access matters: profiling under `sudo` loses the venv, the
# CUDA/torch environment, and the SPARKINFER_* configuration that selects the
# kernel we are trying to measure, so a sudo-only setup tends to profile the
# wrong build. `benchmarks/analyze_ncu_source.py` documents the sudo form; this
# script exists so we can drop the sudo.
#
# Usage:
#   ./scripts/setup_ncu.sh            # detect + verify
#   ./scripts/setup_ncu.sh --apply    # also install the modprobe drop-in

set -uo pipefail

APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1

PYTHON="${PYTHON:-python3}"
CONF=/etc/modprobe.d/nvidia-profiler.conf
ok=1

say() { printf '%s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; ok=0; }

# ---------------------------------------------------------------- 1. locate
say "=== 1. Locating ncu ==="
NCU="$(command -v ncu 2>/dev/null || true)"
if [[ -z "$NCU" ]]; then
  # Nsight Compute ships inside the CUDA toolkit and as a standalone install.
  for cand in \
    /usr/local/cuda/bin/ncu \
    /usr/local/cuda-*/bin/ncu \
    /opt/nvidia/nsight-compute/*/ncu \
    /usr/local/NVIDIA-Nsight-Compute*/ncu; do
    [[ -x "$cand" ]] && NCU="$cand" && break
  done
fi

if [[ -z "$NCU" ]]; then
  fail "ncu not found."
  cat <<'EOS'

  Install one of:
    sudo apt-get update && sudo apt-get install -y nsight-compute
    # or, matching the installed CUDA toolkit:
    sudo apt-get install -y cuda-nsight-compute-12-8
    # or download the standalone installer from
    #   https://developer.nvidia.com/tools-overview/nsight-compute/get-started

  Then re-run this script. If it installed outside PATH, add it:
    export PATH=/opt/nvidia/nsight-compute/<version>:$PATH
EOS
  exit 1
fi
say "ncu: $NCU"

# --------------------------------------------------------------- 2. version
say ""
say "=== 2. Version / SM120 support ==="
VER_LINE="$("$NCU" --version 2>/dev/null | grep -i 'version' | head -n1)"
say "${VER_LINE:-unknown}"
# Blackwell consumer/workstation (SM120) needs Nsight Compute 2025.1 or newer;
# older builds attach but report "unsupported GPU" or emit empty metric sets.
VER_NUM="$(printf '%s' "$VER_LINE" | grep -oE '20[0-9]{2}\.[0-9]+' | head -n1)"
if [[ -n "$VER_NUM" ]]; then
  if awk "BEGIN{exit !($VER_NUM < 2025.1)}"; then
    fail "Nsight Compute $VER_NUM predates SM120 (Blackwell) support; need >= 2025.1."
    say "  Upgrade before trusting any counter values from the RTX PRO 6000 cards."
  else
    say "OK: $VER_NUM supports SM120."
  fi
else
  say "WARN: could not parse a version number; verify SM120 support manually."
fi

# ------------------------------------------------------- 3. counter access
say ""
say "=== 3. Non-root counter access ==="
# Not every driver build exposes this knob in /proc, so an absent line means
# "unknown", not "restricted". Step 4 is the authority either way; only a
# positively-read restriction is reported as a failure here.
PARAM="$(grep -s RestrictProfilingToAdminUsers /proc/driver/nvidia/params || true)"
if [[ -z "$PARAM" ]]; then
  RESTRICTED=unknown
  say "UNKNOWN: RestrictProfilingToAdminUsers is not exposed in /proc/driver/nvidia/params."
  say "  Deferring to the live test in step 4."
elif [[ "$PARAM" == *": 0"* ]]; then
  RESTRICTED=0
  say "$PARAM"
else
  RESTRICTED=1
  say "$PARAM"
fi

APPLIED=0
if [[ "$RESTRICTED" == "0" ]]; then
  say "OK: counters are available to non-root users."
else
  # Covers both a positively-read restriction and the unknown case: the drop-in
  # is idempotent and harmless on an already-permissive driver, so --apply must
  # not be skipped merely because /proc did not expose the knob.
  [[ "$RESTRICTED" == "1" ]] && fail "GPU performance counters are restricted to root."
  if [[ "$APPLY" == "1" ]]; then
    say "Applying $CONF ..."
    if printf 'options nvidia NVreg_RestrictProfilingToAdminUsers=0\n' \
        | sudo tee "$CONF" >/dev/null; then
      sudo update-initramfs -u || say "WARN: update-initramfs failed (non-Debian?)"
      say "Wrote $CONF."
      APPLIED=1
    else
      fail "could not write $CONF"
    fi
  else
    cat <<EOS

  Fix (then reboot):
    echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \\
      | sudo tee $CONF
    sudo update-initramfs -u
    sudo reboot

  Or re-run this script with --apply to do the first two steps.
  A module reload instead of a reboot only works if nothing holds the GPU,
  which is not the case while a vLLM server is running.
EOS
  fi
fi

# ------------------------------------------------------------- 4. live test
say ""
say "=== 4. Live profile smoke test ==="
VERIFIED=0
if ! "$PYTHON" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
  say "SKIP: no CUDA-capable torch in $PYTHON (activate the venv to run this step)."
  say "  This step is the ONLY authority on counter access when /proc does not"
  say "  expose the knob, so a skip cannot be reported as ready."
else
  TMP_OUT="$(mktemp)"
  # No set -e restore here: this script runs under `set -uo pipefail`, so
  # errexit was never on. Enabling it after the probe would abort before the
  # diagnostics and the final readiness verdict get printed.
  "$NCU" --target-processes all --metrics sm__cycles_elapsed.avg \
    --kernel-name-base function --launch-count 1 \
    "$PYTHON" -c '
import torch
a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
(a @ a).sum().item()
' >"$TMP_OUT" 2>&1
  rc=$?
  if grep -q 'ERR_NVGPUCTRPERM' "$TMP_OUT"; then
    fail "ERR_NVGPUCTRPERM - counters still admin-restricted (step 3 not applied or not rebooted)."
  elif [[ $rc -ne 0 ]]; then
    fail "ncu exited $rc. Output:"
    tail -n 25 "$TMP_OUT" >&2
  elif grep -q 'sm__cycles_elapsed' "$TMP_OUT"; then
    say "OK: collected sm__cycles_elapsed from a live bf16 matmul."
    VERIFIED=1
  else
    fail "ncu ran but produced no metric rows. Output:"
    tail -n 25 "$TMP_OUT" >&2
  fi
  say "(full output: $TMP_OUT)"
fi

say ""
# READY requires a positive live collection, not merely an absence of failures:
# when the venv is not active step 4 skips, and on this box /proc does not
# expose the restriction flag either, so nothing would have actually been
# proven.
if [[ "$ok" == "1" && "$VERIFIED" == "0" ]]; then
  say "=== UNVERIFIED - ncu is installed but counter access was never exercised. ==="
  say "Activate the venv (or set PYTHON=/path/to/venv/bin/python) and re-run."
  exit 3
fi
if [[ "$ok" == "1" ]]; then
  say "=== READY. Record this in the evidence header: ==="
  say "ncu_path: $NCU"
  say "ncu_version: ${VER_NUM:-unknown}"
  nvidia-smi --query-gpu=index,uuid,name,pstate --format=csv 2>/dev/null || true
  exit 0
fi
if [[ "$APPLIED" == "1" ]]; then
  say "=== REBOOT REQUIRED ==="
  say "$CONF is in place; the driver only reads it at module load, so the"
  say "smoke test above is EXPECTED to still fail. Run:  sudo reboot"
  say "Then re-run this script (no --apply) and expect all four steps to pass."
  exit 2
fi
say "=== NOT READY - resolve the FAIL lines above and re-run. ==="
exit 1
