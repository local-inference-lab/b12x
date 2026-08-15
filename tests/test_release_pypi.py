"""Hermetic contract tests for scripts/release_pypi.sh (issue #170).

The release helper delegates to scripts/release_pypi.py, a Python
launcher that owns all credential handling.  Tests run the launcher
directly with the real Python interpreter, while using recorder shims
for venv/pip/build/twine child processes.
"""

from __future__ import annotations

import os
import shutil
import shlex
import signal
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "release_pypi.sh"
LAUNCHER = REPO / "scripts" / "release_pypi.py"
REAL_PYTHON = shutil.which("python3") or "/usr/bin/python3"

SENTINEL = "SENTINEL-PYPI-TOKEN-170"

_REAL = {b: shutil.which(b) or f"/bin/{b}" for b in
         ("rm", "mkdir", "cp", "chmod", "cat", "sleep")}


def _recorder(
    root: Path,
    block: bool = False,
    fail_mod: str = "",
    fail_sub: str = "",
) -> str:
    recorder_log = shlex.quote(str(root / "invocations.log"))
    recorder_self = shlex.quote(str(root / "bin" / "python"))
    sentinel_file = shlex.quote(str(root / "sentinel.txt"))
    parts = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"RECORDER_LOG={recorder_log}",
        f"RECORDER_SELF={recorder_self}",
        f"SENTINEL_FILE={sentinel_file}",
        f"FAIL_MODULE={shlex.quote(fail_mod)}",
        f"FAIL_SUBCMD={shlex.quote(fail_sub)}",
        'sentinel=$(<"$SENTINEL_FILE")',
        'has_pw=no',
        'pw_match=no',
        '[[ -n "${TWINE_PASSWORD:-}" ]] && has_pw=yes',
        '[[ "${TWINE_PASSWORD:-}" == "$sentinel" ]] && pw_match=yes',
        "any_match=no",
        'while IFS= read -r var; do',
        '  if [[ "${!var}" == *"$sentinel"* ]]; then any_match=yes; break; fi',
        'done < <(compgen -e)',
        'decoy=na',
        'if [[ -n "${DIST_DECOY:-}" ]]; then',
        '  if [[ -e "$DIST_DECOY" ]]; then decoy=yes; else decoy=no; fi',
        'fi',
        'args_str=""',
        'for a in "$@"; do',
        "  if [[ -z \"$args_str\" ]]; then args_str=\"$a\"; else args_str=\"$args_str\"$'\\x1f'\"$a\"; fi",
        'done',
        "printf 'REC\\tmodule=%s\\tsubcmd=%s\\thas_pw=%s\\tpw_match=%s\\tany_match=%s\\tdecoy=%s\\targs=%s\\n' \\",
        '  "${2:-none}" "${3:-none}" "$has_pw" "$pw_match" "$any_match" "$decoy" "$args_str" \\',
        '  >> "$RECORDER_LOG"',
        'if [[ -n "${FAIL_MODULE:-}" && "${2:-}" == "$FAIL_MODULE" ]]; then',
        '  if [[ -z "${FAIL_SUBCMD:-}" || "${3:-}" == "$FAIL_SUBCMD" ]]; then',
        '    echo "INJECTED_FAILURE module=${2:-} subcmd=${3:-}" >&2',
        '    exit 1',
        '  fi',
        'fi',
    ]
    if block:
        parts.extend([
            'if [[ "${2:-}" == "twine" && "${3:-}" == "upload" ]]; then',
            f'  echo $$ > {shlex.quote(str(root / "block_pid.txt"))}',
            f'  while true; do {_REAL["sleep"]} 0.1; done',
            'fi',
        ])
    parts.extend([
        'if [[ "${1:-}" == "-m" && "${2:-}" == "venv" ]]; then',
        f'  {_REAL["mkdir"]} -p "$3/bin"',
        f'  {_REAL["cp"]} "$RECORDER_SELF" "$3/bin/python"',
        f'  {_REAL["chmod"]} +x "$3/bin/python"',
        'elif [[ "${1:-}" == "-m" && "${2:-}" == "build" ]]; then',
        f'  {_REAL["mkdir"]} -p dist',
        "  printf 'placeholder\\n' > dist/b12x-0.0.0.tar.gz",
        'fi',
        'exit 0',
    ])
    return "\n".join(parts) + "\n"


def _parse_log(text: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for line in text.splitlines():
        if not line.startswith("REC\t"):
            continue
        rec: dict[str, str] = {}
        for field in line[4:].split("\t"):
            if "=" in field:
                k, v = field.split("=", 1)
                rec[k] = v
        records.append(rec)
    return records


def _classify(records: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    pip_seen = 0
    for rec in records:
        mod, sub = rec.get("module", ""), rec.get("subcmd", "")
        if mod == "venv":
            out.setdefault("venv", rec)
        elif mod == "pip" and sub == "install":
            pip_seen += 1
            out[f"pip_{pip_seen}"] = rec
        elif mod == "build":
            out.setdefault("build", rec)
        elif mod == "twine" and sub == "check":
            out.setdefault("twine_check", rec)
        elif mod == "twine" and sub == "upload":
            out.setdefault("twine_upload", rec)
    return out


PRE_UPLOAD = ["venv", "pip_1", "pip_2", "build", "twine_check"]


@pytest.fixture
def release_root(tmp_path: Path) -> Path:
    root = tmp_path / "release-root"
    root.mkdir()
    scripts = root / "scripts"
    scripts.mkdir()
    shutil.copy2(SCRIPT, scripts / "release_pypi.sh")
    (scripts / "release_pypi.sh").chmod(0o755)
    shutil.copy2(LAUNCHER, scripts / "release_pypi.py")
    (scripts / "release_pypi.py").chmod(0o755)
    bin_dir = root / "bin"
    bin_dir.mkdir()
    rec = _recorder(root)
    for name in ("python", "python3"):
        (bin_dir / name).write_text(rec)
        (bin_dir / name).chmod(0o755)
    for name in ("mkdir", "cp", "chmod", "cat", "sleep"):
        (bin_dir / name).write_text(f"#!/bin/bash\nexec {_REAL[name]} \"$@\"\n")
        (bin_dir / name).chmod(0o755)
    return root


def _setup(root: Path, *, dotenv: str | None = None,
           dotenv_bytes: bytes | None = None,
           extra_env: dict[str, str] | None = None,
           precreate_venv: bool = True, dist_decoy: bool = False,
           sentinel: str = SENTINEL, fail_mod: str = "", fail_sub: str = "",
           ) -> tuple[dict[str, str], Path, Path]:
    if dotenv is not None:
        (root / ".env").write_text(dotenv)
    if dotenv_bytes is not None:
        (root / ".env").write_bytes(dotenv_bytes)
    recorder = root / "bin" / "python"
    if fail_mod or fail_sub:
        recorder.write_text(
            _recorder(root, fail_mod=fail_mod, fail_sub=fail_sub)
        )
        recorder.chmod(0o755)
    log = root / "invocations.log"
    sf = root / "sentinel.txt"
    sf.write_text(sentinel)
    decoy = str(root / "dist" / "decoy.txt") if dist_decoy else ""
    if dist_decoy:
        df = root / "dist" / "decoy.txt"
        df.parent.mkdir(parents=True, exist_ok=True)
        df.write_text("pre-existing\n")
    if precreate_venv:
        vb = root / ".venv-release" / "bin"
        vb.mkdir(parents=True)
        shutil.copy2(recorder, vb / "python")
        (vb / "python").chmod(0o755)
    env: dict[str, str] = {
        "PATH": f"{root / 'bin'}",
        "RECORDER_LOG": str(log),
        "RECORDER_SELF": str(recorder),
        "SENTINEL_FILE": str(sf),
        "DIST_DECOY": decoy,
        "HOME": str(root),
    }
    if fail_mod:
        env["FAIL_MODULE"] = fail_mod
    if fail_sub:
        env["FAIL_SUBCMD"] = fail_sub
    for k in ("TWINE_USERNAME", "TWINE_PASSWORD"):
        env.pop(k, None)
    if extra_env:
        env.update(extra_env)
    return env, log, root


def _run(root: Path, **kw) -> tuple[list[dict[str, str]], subprocess.CompletedProcess, Path]:
    env, log, _ = _setup(root, **kw)
    proc = subprocess.run(
        [REAL_PYTHON, str(root / "scripts" / "release_pypi.py")],
        cwd=root, env=env, capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        raise AssertionError(
            f"release failed (rc={proc.returncode}):\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    records = _parse_log(log.read_text()) if log.exists() else []
    return records, proc, root


def _run_raw(root: Path, **kw) -> subprocess.CompletedProcess:
    env, _, _ = _setup(root, **kw)
    return subprocess.run(
        [REAL_PYTHON, str(root / "scripts" / "release_pypi.py")],
        cwd=root, env=env, capture_output=True, text=True, timeout=30)


def _assert_no_recs(root: Path) -> None:
    log = root / "invocations.log"
    recs = _parse_log(log.read_text()) if log.exists() else []
    assert len(recs) == 0, f"expected zero records, got {len(recs)}"


# ---------------------------------------------------------------------------
# Core boundary
# ---------------------------------------------------------------------------
def test_credential_absent_from_preupload(release_root: Path) -> None:
    records, _, root = _run(release_root,
                           dotenv=f"TWINE_USERNAME=__token__\nTWINE_PASSWORD={SENTINEL}\n# c\nOTHER=v\n",
                           dist_decoy=True, precreate_venv=True)
    inv = _classify(records)
    assert set(inv) == {"pip_1", "pip_2", "build", "twine_check", "twine_upload"}
    for l in PRE_UPLOAD:
        if l in inv:
            r = inv[l]
            assert r["has_pw"] == "no", f"leaked into {l}"
            assert r["pw_match"] == "no", f"sentinel in {l}"
            assert r["any_match"] == "no", f"sentinel in env of {l}"
    u = inv["twine_upload"]
    assert u["pw_match"] == "yes" and u["any_match"] == "yes"
    for r in records:
        assert SENTINEL not in r.get("args", ""), "sentinel in argv"
    assert not (root / "dist" / "decoy.txt").exists()
    assert (root / "dist" / "b12x-0.0.0.tar.gz").exists()


def test_no_env_value_contains_sentinel(release_root: Path) -> None:
    records, _, _ = _run(release_root, dotenv=f"TWINE_PASSWORD={SENTINEL}\n", precreate_venv=True)
    inv = _classify(records)
    for l in PRE_UPLOAD:
        if l in inv:
            assert inv[l]["any_match"] == "no"
    assert inv["twine_upload"]["any_match"] == "yes"


# ---------------------------------------------------------------------------
# Entry boundary (Python launcher is immune to shell startup hooks)
# ---------------------------------------------------------------------------
def test_hostile_bash_env_safe(release_root: Path) -> None:
    be = release_root / "be.sh"
    be.write_text(f'echo "BE_RAN ${{TWINE_PASSWORD:-none}}" >> "{release_root / "be.log"}"\n')
    be.chmod(0o755)
    records, proc, _ = _run(release_root, extra_env={"TWINE_PASSWORD": SENTINEL, "BASH_ENV": str(be)},
                           precreate_venv=True)
    assert not (release_root / "be.log").exists(), "BASH_ENV ran"
    assert SENTINEL not in proc.stdout and SENTINEL not in proc.stderr
    inv = _classify(records)
    for l in PRE_UPLOAD:
        if l in inv:
            assert inv[l]["any_match"] == "no"
    assert inv["twine_upload"]["pw_match"] == "yes"


def test_hostile_ps4_safe(release_root: Path) -> None:
    records, proc, _ = _run(release_root, extra_env={"TWINE_PASSWORD": SENTINEL, "PS4": f"+{SENTINEL} "},
                           precreate_venv=True)
    assert SENTINEL not in proc.stdout and SENTINEL not in proc.stderr
    inv = _classify(records)
    for l in PRE_UPLOAD:
        if l in inv:
            assert inv[l]["any_match"] == "no"
    assert inv["twine_upload"]["pw_match"] == "yes"


def test_hostile_shellopts_privileged_safe(release_root: Path) -> None:
    be = release_root / "be.sh"
    be.write_text(f'echo "BE" >> "{release_root / "be.log"}"\n')
    be.chmod(0o755)
    records, proc, _ = _run(release_root, extra_env={"TWINE_PASSWORD": SENTINEL,
                           "SHELLOPTS": "xtrace:allexport:privileged", "BASH_ENV": str(be)},
                           precreate_venv=True)
    assert not (release_root / "be.log").exists(), "BASH_ENV ran"
    assert SENTINEL not in proc.stdout and SENTINEL not in proc.stderr
    inv = _classify(records)
    for l in PRE_UPLOAD:
        if l in inv:
            assert inv[l]["any_match"] == "no"
    assert inv["twine_upload"]["pw_match"] == "yes"


def test_hostile_exported_function_safe(release_root: Path) -> None:
    records, proc, _ = _run(release_root, extra_env={"TWINE_PASSWORD": SENTINEL,
                           "BASH_FUNC_rm%%": "() { echo PWNED_RM; }"}, precreate_venv=True)
    assert "PWNED_RM" not in proc.stdout
    inv = _classify(records)
    for l in PRE_UPLOAD:
        if l in inv:
            assert inv[l]["any_match"] == "no"
    assert inv["twine_upload"]["pw_match"] == "yes"


def test_hostile_unset_function_safe(release_root: Path) -> None:
    records, proc, _ = _run(release_root, extra_env={"TWINE_PASSWORD": SENTINEL,
                           "BASH_FUNC_unset%%": "() { echo PWNED_UNSET; }"}, precreate_venv=True)
    assert "PWNED_UNSET" not in proc.stdout
    inv = _classify(records)
    for l in PRE_UPLOAD:
        if l in inv:
            assert inv[l]["any_match"] == "no"
    assert inv["twine_upload"]["pw_match"] == "yes"


# ---------------------------------------------------------------------------
# Upload env allowlisted
# ---------------------------------------------------------------------------
def test_upload_env_allowlisted(release_root: Path) -> None:
    records, _, _ = _run(release_root, dotenv=f"TWINE_PASSWORD={SENTINEL}\n",
                        extra_env={"TWINE_REPOSITORY_URL": "https://evil.example.com",
                                   "TWINE_REPOSITORY": "evil", "PYTHONPATH": "/evil",
                                   "PYTHONSTARTUP": "/evil.py", "PIP_CONFIG_FILE": "/evil/pip.conf",
                                   "PIP_INDEX_URL": "https://evil.example.com/simple",
                                   "BASH_FUNC_evil%%": "() { echo PWNED; }"},
                        precreate_venv=True)
    inv = _classify(records)
    u = inv["twine_upload"]
    assert u["pw_match"] == "yes"
    assert "evil" not in u.get("args", "")


# ---------------------------------------------------------------------------
# Pre-exported internal names
# ---------------------------------------------------------------------------
def test_pre_exported_internal_names_no_leak(release_root: Path) -> None:
    extra = {"TWINE_PASSWORD": SENTINEL, "RELEASE_TWINE_PASSWORD": SENTINEL,
             "_env_line": SENTINEL, "_env_key": SENTINEL, "_env_raw": SENTINEL,
             "_env_val": SENTINEL, "_env_stripped": SENTINEL,
             "_release_upload_pid": SENTINEL, "_release_upload_pgid": SENTINEL}
    records, _, _ = _run(release_root, extra_env=extra, precreate_venv=True)
    inv = _classify(records)
    for l in PRE_UPLOAD:
        if l in inv:
            assert inv[l]["any_match"] == "no"
    assert inv["twine_upload"]["pw_match"] == "yes"


# ---------------------------------------------------------------------------
# Root discovery
# ---------------------------------------------------------------------------
def test_root_discovery_env_credential(release_root: Path) -> None:
    records, _, _ = _run(release_root, extra_env={"TWINE_PASSWORD": SENTINEL}, precreate_venv=True)
    inv = _classify(records)
    assert inv["twine_upload"]["pw_match"] == "yes"
    for l in PRE_UPLOAD:
        if l in inv:
            assert inv[l]["any_match"] == "no"


# ---------------------------------------------------------------------------
# .env not executed
# ---------------------------------------------------------------------------
def test_env_not_executed(release_root: Path, tmp_path: Path) -> None:
    ma, mb, mc = (tmp_path / f"pwned_{c}" for c in "abc")
    dotenv = (f"TWINE_USERNAME=__token__\nTWINE_PASSWORD={SENTINEL}\n"
              f'EVIL_A=$(touch "{ma}")\nEVIL_B=`touch "{mb}"`\n'
              f'EVIL_C=value; touch "{mc}"\n')
    records, _, _ = _run(release_root, dotenv=dotenv, precreate_venv=True)
    assert not ma.exists() and not mb.exists() and not mc.exists()
    inv = _classify(records)
    assert inv["twine_upload"]["pw_match"] == "yes"
    for l in PRE_UPLOAD:
        if l in inv:
            assert inv[l]["any_match"] == "no"


def test_credential_value_literal(release_root: Path, tmp_path: Path) -> None:
    marker = tmp_path / "lp"
    pw = f'$(touch "{marker}"){SENTINEL}'
    records, _, _ = _run(release_root, dotenv=f"TWINE_PASSWORD='{pw}'\n", sentinel=pw, precreate_venv=True)
    assert not marker.exists()
    inv = _classify(records)
    assert inv["twine_upload"]["pw_match"] == "yes"
    for l in PRE_UPLOAD:
        if l in inv:
            assert inv[l]["any_match"] == "no"


# ---------------------------------------------------------------------------
# Parser grammar: valid cases
# ---------------------------------------------------------------------------
def test_parser_dq_comment(release_root: Path) -> None:
    pw = "T-DQ"
    assert _classify(_run(release_root, dotenv=f'TWINE_PASSWORD="{pw}" # c\n', sentinel=pw, precreate_venv=True)[0])["twine_upload"]["pw_match"] == "yes"

def test_parser_sq_comment(release_root: Path) -> None:
    pw = "T-SQ"
    assert _classify(_run(release_root, dotenv=f"TWINE_PASSWORD='{pw}' # c\n", sentinel=pw, precreate_venv=True)[0])["twine_upload"]["pw_match"] == "yes"

def test_parser_uq_comment(release_root: Path) -> None:
    pw = "T-UQ"
    assert _classify(_run(release_root, dotenv=f"TWINE_PASSWORD={pw} # c\n", sentinel=pw, precreate_venv=True)[0])["twine_upload"]["pw_match"] == "yes"

def test_parser_hash_unquoted(release_root: Path) -> None:
    pw = "T#NC"
    assert _classify(_run(release_root, dotenv=f"TWINE_PASSWORD={pw}\n", sentinel=pw, precreate_venv=True)[0])["twine_upload"]["pw_match"] == "yes"

def test_parser_hash_quoted(release_root: Path) -> None:
    pw = "T#I"
    assert _classify(_run(release_root, dotenv=f'TWINE_PASSWORD="{pw}"\n', sentinel=pw, precreate_venv=True)[0])["twine_upload"]["pw_match"] == "yes"

def test_parser_crlf(release_root: Path) -> None:
    pw = "T-CRLF"
    assert _classify(_run(release_root, dotenv=f"TWINE_PASSWORD={pw}\r\nOTHER=v\r\n", sentinel=pw, precreate_venv=True)[0])["twine_upload"]["pw_match"] == "yes"

def test_parser_dups_last_wins(release_root: Path) -> None:
    pw = "T-2"
    assert _classify(_run(release_root, dotenv=f"TWINE_PASSWORD=first\nTWINE_PASSWORD={pw}\n", sentinel=pw, precreate_venv=True)[0])["twine_upload"]["pw_match"] == "yes"

def test_parser_eof_no_newline(release_root: Path) -> None:
    pw = "T-EOF"
    assert _classify(_run(release_root, dotenv=f"TWINE_PASSWORD={pw}", sentinel=pw, precreate_venv=True)[0])["twine_upload"]["pw_match"] == "yes"

def test_parser_export_prefix(release_root: Path) -> None:
    pw = "T-EX"
    assert _classify(_run(release_root, dotenv=f"export TWINE_PASSWORD={pw}\n", sentinel=pw, precreate_venv=True)[0])["twine_upload"]["pw_match"] == "yes"

def test_parser_backup_key_unknown(release_root: Path) -> None:
    pw = SENTINEL
    records, _, _ = _run(release_root, dotenv=f"TWINE_PASSWORD_BACKUP=other\nTWINE_PASSWORD={pw}\n", precreate_venv=True)
    assert _classify(records)["twine_upload"]["pw_match"] == "yes"


# ---------------------------------------------------------------------------
# Parser: empty value fails
# ---------------------------------------------------------------------------
def test_parser_empty_fails(release_root: Path) -> None:
    proc = _run_raw(release_root, dotenv="TWINE_PASSWORD=\n", precreate_venv=True)
    assert proc.returncode != 0
    assert "TWINE_PASSWORD must be set" in proc.stderr

def test_parser_empty_comment_fails(release_root: Path) -> None:
    proc = _run_raw(release_root, dotenv="TWINE_PASSWORD= # blank\n",
                    extra_env={"TWINE_PASSWORD": "PRODUCTION"}, precreate_venv=True)
    assert proc.returncode != 0
    assert "TWINE_PASSWORD must be set" in proc.stderr


# ---------------------------------------------------------------------------
# Parser: malformed credential aborts
# ---------------------------------------------------------------------------
def test_parser_missing_eq_aborts(release_root: Path) -> None:
    proc = _run_raw(release_root, dotenv="TWINE_PASSWORD intended\n", extra_env={"TWINE_PASSWORD": "PROD"})
    assert proc.returncode != 0 and "Malformed .env" in proc.stderr
    _assert_no_recs(release_root)

def test_parser_key_ws_aborts(release_root: Path) -> None:
    proc = _run_raw(release_root, dotenv="TWINE_PASSWORD = v\n", extra_env={"TWINE_PASSWORD": "PROD"})
    assert proc.returncode != 0 and "Malformed .env" in proc.stderr
    _assert_no_recs(release_root)

def test_parser_username_key_ws_aborts(release_root: Path) -> None:
    proc = _run_raw(release_root, dotenv="TWINE_USERNAME = u\nTWINE_PASSWORD=v\n")
    assert proc.returncode != 0 and "Malformed .env" in proc.stderr

def test_parser_username_missing_eq_aborts(release_root: Path) -> None:
    proc = _run_raw(release_root, dotenv="TWINE_USERNAME u\nTWINE_PASSWORD=v\n")
    assert proc.returncode != 0 and "Malformed .env" in proc.stderr

def test_parser_plus_eq_aborts(release_root: Path) -> None:
    proc = _run_raw(release_root, dotenv="TWINE_PASSWORD+=v\n", extra_env={"TWINE_PASSWORD": "PROD"})
    assert proc.returncode != 0 and "Malformed .env" in proc.stderr

def test_parser_array_aborts(release_root: Path) -> None:
    proc = _run_raw(release_root, dotenv="TWINE_PASSWORD[0]=v\n", extra_env={"TWINE_PASSWORD": "PROD"})
    assert proc.returncode != 0 and "Malformed .env" in proc.stderr

def test_parser_hash_after_key_aborts(release_root: Path) -> None:
    proc = _run_raw(release_root, dotenv="TWINE_PASSWORD#typo\n", extra_env={"TWINE_PASSWORD": "PROD"})
    assert proc.returncode != 0 and "Malformed .env" in proc.stderr

def test_parser_embedded_quote_aborts(release_root: Path) -> None:
    proc = _run_raw(release_root, dotenv='TWINE_PASSWORD=abc"\n', extra_env={"TWINE_PASSWORD": "PROD"})
    assert proc.returncode != 0 and "Malformed .env" in proc.stderr

def test_parser_embedded_squote_aborts(release_root: Path) -> None:
    proc = _run_raw(release_root, dotenv="TWINE_PASSWORD=abc'\n")
    assert proc.returncode != 0 and "Malformed .env" in proc.stderr

def test_parser_unterminated_quote_aborts(release_root: Path) -> None:
    proc = _run_raw(release_root, dotenv='TWINE_PASSWORD="unterm\n', extra_env={"TWINE_PASSWORD": "PROD"})
    assert proc.returncode != 0 and "Malformed .env" in proc.stderr

def test_parser_trailing_content_aborts(release_root: Path) -> None:
    proc = _run_raw(release_root, dotenv='TWINE_PASSWORD="b" extra\n')
    assert proc.returncode != 0 and "Malformed .env" in proc.stderr

def test_parser_malformed_after_valid_aborts(release_root: Path) -> None:
    proc = _run_raw(release_root, dotenv='TWINE_PASSWORD=V\nTWINE_PASSWORD="unterm\n')
    assert proc.returncode != 0 and "Malformed .env" in proc.stderr

def test_parser_malformed_export_aborts(release_root: Path) -> None:
    proc = _run_raw(release_root, dotenv="export TWINE_PASSWORD intended\n")
    assert proc.returncode != 0 and "Malformed .env" in proc.stderr

def test_parser_malformed_crlf_aborts(release_root: Path) -> None:
    proc = _run_raw(release_root, dotenv='TWINE_PASSWORD="unterm\r\n')
    assert proc.returncode != 0 and "Malformed .env" in proc.stderr

def test_parser_username_plus_eq_aborts(release_root: Path) -> None:
    proc = _run_raw(release_root, dotenv="TWINE_USERNAME+=u\nTWINE_PASSWORD=v\n")
    assert proc.returncode != 0 and "Malformed .env" in proc.stderr


# ---------------------------------------------------------------------------
# Parser: byte-level — BOM, NUL, control bytes, invalid UTF-8
# ---------------------------------------------------------------------------
def test_parser_bom_accepted(release_root: Path) -> None:
    pw = SENTINEL
    data = b"\xef\xbb\xbf" + f"TWINE_PASSWORD={pw}\n".encode()
    records, _, _ = _run(release_root, dotenv_bytes=data, sentinel=pw, precreate_venv=True)
    assert _classify(records)["twine_upload"]["pw_match"] == "yes"

def test_parser_bom_no_fallback(release_root: Path) -> None:
    data = b"\xef\xbb\xbf" + b"TWINE_PASSWORD=staging\n"
    records, _, _ = _run(release_root, dotenv_bytes=data, sentinel="staging",
                        extra_env={"TWINE_PASSWORD": "PRODUCTION"}, precreate_venv=True)
    inv = _classify(records)
    assert inv["twine_upload"]["pw_match"] == "yes"

def test_parser_nul_aborts(release_root: Path) -> None:
    proc = _run_raw(release_root, dotenv_bytes=b"TWINE_PASSWORD=valid\x00bad\n",
                   extra_env={"TWINE_PASSWORD": "PROD"})
    assert proc.returncode != 0 and "Malformed .env" in proc.stderr

def test_parser_control_byte_aborts(release_root: Path) -> None:
    proc = _run_raw(release_root, dotenv_bytes=b"TWINE_PASSWORD=valid\x01bad\n",
                   extra_env={"TWINE_PASSWORD": "PROD"})
    assert proc.returncode != 0 and "Malformed .env" in proc.stderr

def test_parser_invalid_utf8_aborts(release_root: Path) -> None:
    proc = _run_raw(release_root, dotenv_bytes=b"TWINE_PASSWORD=valid\xff\xfeprom\n",
                   extra_env={"TWINE_PASSWORD": "PROD"})
    assert proc.returncode != 0 and "Malformed .env" in proc.stderr


# ---------------------------------------------------------------------------
# Parser: literal $() and backticks
# ---------------------------------------------------------------------------
def test_parser_literal_dollar_paren_dquotes(release_root: Path, tmp_path: Path) -> None:
    marker = tmp_path / "qp"
    pw = f"$(touch {marker})"
    records, _, _ = _run(release_root, dotenv=f'TWINE_PASSWORD="{pw}"\n', sentinel=pw, precreate_venv=True)
    assert not marker.exists()
    assert _classify(records)["twine_upload"]["pw_match"] == "yes"

def test_parser_literal_backticks_squotes(release_root: Path, tmp_path: Path) -> None:
    marker = tmp_path / "bp"
    pw = f"`touch {marker}`"
    records, _, _ = _run(release_root, dotenv=f"TWINE_PASSWORD='{pw}'\n", sentinel=pw, precreate_venv=True)
    assert not marker.exists()
    assert _classify(records)["twine_upload"]["pw_match"] == "yes"


# ---------------------------------------------------------------------------
# Precedence and fallback
# ---------------------------------------------------------------------------
def test_dotenv_overrides_env(release_root: Path) -> None:
    records, _, _ = _run(release_root, dotenv=f"TWINE_PASSWORD={SENTINEL}\n",
                        extra_env={"TWINE_PASSWORD": "ENV"}, precreate_venv=True)
    assert _classify(records)["twine_upload"]["pw_match"] == "yes"

def test_env_supplies_no_dotenv(release_root: Path) -> None:
    records, _, _ = _run(release_root, extra_env={"TWINE_PASSWORD": SENTINEL}, precreate_venv=True)
    inv = _classify(records)
    for l in PRE_UPLOAD:
        if l in inv:
            assert inv[l]["any_match"] == "no"
    assert inv["twine_upload"]["pw_match"] == "yes"

def test_missing_credential_fails(release_root: Path) -> None:
    proc = _run_raw(release_root)
    assert proc.returncode != 0
    assert "TWINE_PASSWORD must be set" in proc.stderr


# ---------------------------------------------------------------------------
# Failure cleanup
# ---------------------------------------------------------------------------
def test_build_failure_cleanup(release_root: Path) -> None:
    proc = _run_raw(release_root, dotenv=f"TWINE_PASSWORD={SENTINEL}\n", fail_mod="build", precreate_venv=True)
    assert proc.returncode != 0
    assert SENTINEL not in proc.stdout and SENTINEL not in proc.stderr
    records = _parse_log((release_root / "invocations.log").read_text()) if (release_root / "invocations.log").exists() else []
    inv = _classify(records)
    for l in PRE_UPLOAD:
        if l in inv:
            assert inv[l]["any_match"] == "no"
    assert "twine_upload" not in inv

def test_check_failure_cleanup(release_root: Path) -> None:
    proc = _run_raw(release_root, dotenv=f"TWINE_PASSWORD={SENTINEL}\n", fail_mod="twine", fail_sub="check", precreate_venv=True)
    assert proc.returncode != 0
    assert SENTINEL not in proc.stdout and SENTINEL not in proc.stderr
    records = _parse_log((release_root / "invocations.log").read_text()) if (release_root / "invocations.log").exists() else []
    inv = _classify(records)
    assert "twine_upload" not in inv

def test_upload_failure_cleanup(release_root: Path) -> None:
    proc = _run_raw(release_root, dotenv=f"TWINE_PASSWORD={SENTINEL}\n", fail_mod="twine", fail_sub="upload", precreate_venv=True)
    assert proc.returncode != 0
    assert SENTINEL not in proc.stdout and SENTINEL not in proc.stderr
    records = _parse_log((release_root / "invocations.log").read_text()) if (release_root / "invocations.log").exists() else []
    inv = _classify(records)
    assert inv["twine_upload"]["pw_match"] == "yes"

def test_pip_failure_cleanup(release_root: Path) -> None:
    proc = _run_raw(release_root, dotenv=f"TWINE_PASSWORD={SENTINEL}\n", fail_mod="pip", precreate_venv=True)
    assert proc.returncode != 0
    assert SENTINEL not in proc.stdout and SENTINEL not in proc.stderr
    records = _parse_log((release_root / "invocations.log").read_text()) if (release_root / "invocations.log").exists() else []
    inv = _classify(records)
    assert "twine_upload" not in inv


# ---------------------------------------------------------------------------
# Signal forwarding
# ---------------------------------------------------------------------------
def test_signal_term_during_upload(release_root: Path) -> None:
    (release_root / ".env").write_text(f"TWINE_PASSWORD={SENTINEL}\n")
    block_rec = _recorder(release_root, block=True)
    recorder = release_root / "bin" / "python"
    recorder.write_text(block_rec)
    recorder.chmod(0o755)
    vb = release_root / ".venv-release" / "bin"
    vb.mkdir(parents=True, exist_ok=True)
    (vb / "python").write_text(block_rec)
    (vb / "python").chmod(0o755)
    env, log, _ = _setup(release_root, extra_env={}, precreate_venv=False)
    proc = subprocess.Popen(
        [REAL_PYTHON, str(release_root / "scripts" / "release_pypi.py")],
        cwd=release_root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        child_pid = None
        deadline = time.time() + 20
        bpf = release_root / "block_pid.txt"
        while time.time() < deadline:
            if bpf.exists():
                child_pid = bpf.read_text().strip()
                if child_pid:
                    break
            time.sleep(0.1)
        assert child_pid, "blocking upload child did not start"
        child_pid_int = int(child_pid)
        log_recs = _parse_log(log.read_text()) if log.exists() else []
        inv = _classify(log_recs)
        assert "twine_upload" in inv
        assert inv["twine_upload"]["pw_match"] == "yes"
        proc.send_signal(signal.SIGTERM)
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            pytest.fail("did not exit after TERM")
        assert proc.returncode == 143, f"expected 143, got {proc.returncode}"
        try:
            os.kill(child_pid_int, 0)
            time.sleep(0.5)
            try:
                os.kill(child_pid_int, 0)
                alive = True
            except OSError:
                alive = False
        except OSError:
            alive = False
        assert not alive, "upload child still alive"
        assert SENTINEL not in stdout and SENTINEL not in stderr
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()


# ---------------------------------------------------------------------------
# Executable bit
# ---------------------------------------------------------------------------
def test_repository_script_is_executable() -> None:
    assert SCRIPT.stat().st_mode & 0o100, "script is not executable"
