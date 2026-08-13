from __future__ import annotations

from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).parents[1]
PCIE_SOURCE = ROOT / "b12x" / "comm" / "pcie"


def test_pcie_collectives_are_python_only() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    package_data = config["tool"]["setuptools"].get("package-data", {})

    assert "b12x.comm.pcie" not in package_data
    assert not list(PCIE_SOURCE.glob("*.cu"))
    assert not list(PCIE_SOURCE.glob("*.cuh"))
    assert not list(PCIE_SOURCE.glob("*.cpp"))
    for source in PCIE_SOURCE.glob("*.py"):
        text = source.read_text()
        assert "torch.utils.cpp_extension" not in text
        assert "cpp_extension.load" not in text
        assert "load_inline(" not in text


# --- CVE-2026-4372: transformers dependency + runtime guard ------------------

# Versions that the old floor (>=4.51) admitted but that are affected by
# CVE-2026-4372 — they must be rejected by both the pyproject specifier
# and the runtime guard.
_AFFECTED_VERSIONS = ["4.51", "5.2.99", "5.3.0rc1", "5.3.0.dev0"]

# Versions that are patched (>=5.3.0 final) — they must be accepted.
_SAFE_VERSIONS = ["5.3.0", "5.3.0+local", "5.3.0.post0", "5.3.1"]


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def _transformers_spec() -> str:
    """Return the transformers specifier string from pyproject.toml, parsed
    with packaging.requirements.Requirement for exact-name matching.
    """
    from packaging.requirements import Requirement

    deps = _pyproject()["project"]["dependencies"]
    tf_reqs = [Requirement(d) for d in deps if Requirement(d).name == "transformers"]
    assert len(tf_reqs) == 1, f"expected exactly one transformers dep, got {tf_reqs}"
    return str(tf_reqs[0].specifier)


def _script_floor() -> str:
    """Return TRANSFORMERS_FLOOR from the script, to verify it is bound to
    the pyproject spec.
    """
    import ast

    source = (ROOT / "scripts" / "capture_hidden_states.py").read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TRANSFORMERS_FLOOR":
                    assert isinstance(node.value, ast.Constant)
                    return node.value.value
    raise AssertionError("TRANSFORMERS_FLOOR not found in capture_hidden_states.py")


@pytest.mark.parametrize("version", _AFFECTED_VERSIONS)
def test_pyproject_rejects_affected_versions(version: str) -> None:
    """Every affected version must be excluded by the pyproject specifier."""
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    specs = SpecifierSet(_transformers_spec())
    assert Version(version) not in specs, (
        f"affected version {version} must be excluded by transformers "
        f"specifier '{specs}'"
    )


@pytest.mark.parametrize("version", _SAFE_VERSIONS)
def test_pyproject_accepts_safe_versions(version: str) -> None:
    """Every patched version must be accepted by the pyproject specifier."""
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    specs = SpecifierSet(_transformers_spec())
    assert Version(version) in specs, (
        f"safe version {version} must be accepted by transformers "
        f"specifier '{specs}'"
    )


def test_pyproject_floor_is_5_3_0() -> None:
    """The pyproject specifier must include 5.3.0 exactly."""
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    specs = SpecifierSet(_transformers_spec())
    assert Version("5.3.0") in specs, (
        f"transformers specifier '{specs}' must include 5.3.0"
    )


def test_script_floor_bound_to_pyproject() -> None:
    """TRANSFORMERS_FLOOR in the script must be '5.3.0', matching the
    pyproject.toml floor.
    """
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    floor = _script_floor()
    assert floor == "5.3.0", f"script floor is {floor}, expected 5.3.0"

    specs = SpecifierSet(_transformers_spec())
    assert Version(floor) in specs, (
        f"script floor {floor} must satisfy pyproject specifier '{specs}'"
    )


@pytest.mark.parametrize("version", _AFFECTED_VERSIONS)
def test_guard_rejects_affected_versions(monkeypatch, version: str) -> None:
    """The runtime guard must raise SystemExit for any affected version."""
    pytest.importorskip("torch")
    import importlib.metadata

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import capture_hidden_states

    monkeypatch.setattr(
        importlib.metadata, "version", lambda _name: version
    )

    with pytest.raises(SystemExit) as exc_info:
        capture_hidden_states.assert_safe_transformers()

    msg = str(exc_info.value)
    assert "5.3.0" in msg
    assert "CVE-2026-4372" in msg


@pytest.mark.parametrize("version", ["5.3.1rc1", "5.4.0.dev0"])
def test_guard_rejects_nonfinal_versions(monkeypatch, version: str) -> None:
    """The runtime guard must fail closed for future non-final releases."""
    pytest.importorskip("torch")
    import importlib.metadata

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import capture_hidden_states

    monkeypatch.setattr(importlib.metadata, "version", lambda _name: version)

    with pytest.raises(SystemExit) as exc_info:
        capture_hidden_states.assert_safe_transformers()

    msg = str(exc_info.value)
    assert "final release" in msg
    assert "CVE-2026-4372" in msg

@pytest.mark.parametrize("version", _SAFE_VERSIONS)
def test_guard_accepts_safe_versions(monkeypatch, version: str) -> None:
    """The runtime guard must pass without error for patched versions."""
    pytest.importorskip("torch")
    import importlib.metadata

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import capture_hidden_states

    monkeypatch.setattr(
        importlib.metadata, "version", lambda _name: version
    )

    # Must not raise.
    capture_hidden_states.assert_safe_transformers()


def test_guard_refuses_missing_transformers(monkeypatch) -> None:
    """If transformers is not installed at all, the guard must fail closed."""
    pytest.importorskip("torch")
    import importlib.metadata

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import capture_hidden_states

    def _raise(name: str):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _raise)

    with pytest.raises(SystemExit) as exc_info:
        capture_hidden_states.assert_safe_transformers()

    assert "not installed" in str(exc_info.value)


def test_guard_fails_closed_when_packaging_missing(monkeypatch) -> None:
    """If packaging itself cannot be imported, the guard must still give an
    actionable SystemExit rather than an unhandled ImportError — this is the
    fail-closed guarantee for environments that bypass the dependency floor.
    """
    pytest.importorskip("torch")
    import importlib.metadata
    import sys

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import capture_hidden_states

    # Simulate transformers being installed but packaging unavailable.
    monkeypatch.setattr(
        importlib.metadata, "version", lambda _name: "5.2.0"
    )

    # Block the 'packaging' module from being imported.
    monkeypatch.setitem(sys.modules, "packaging", None)
    monkeypatch.setitem(sys.modules, "packaging.version", None)

    with pytest.raises(SystemExit) as exc_info:
        capture_hidden_states.assert_safe_transformers()

    msg = str(exc_info.value)
    assert "packaging" in msg.lower()
    assert "pip install packaging" in msg


def test_guard_fails_closed_in_subprocess_without_transformers() -> None:
    """Subprocess-level proof: with transformers reported as missing via a
    monkeypatched importlib.metadata.version (not sys.modules blocking,
    which importlib.metadata ignores), the guard must produce an actionable
    SystemExit citing transformers and CVE-2026-4372 — even when packaging
    is also unavailable. No raw packaging ImportError should leak.
    """
    import subprocess
    import sys

    code = (
        "import sys, types, importlib.metadata; "
        # Stub torch so the script's top-level 'import torch' succeeds.
        "sys.modules['torch'] = types.ModuleType('torch'); "
        # Monkeypatch importlib.metadata.version to raise PackageNotFoundError
        # for transformers — this is what actually happens when transformers
        # is not installed; sys.modules['transformers']=None would NOT prevent
        # importlib.metadata.version from finding real metadata.
        "importlib.metadata.version = lambda name: (_ for _ in ()).throw("
        "importlib.metadata.PackageNotFoundError(name)); "
        # Block packaging so the deepest fail-closed path is exercised.
        "sys.modules['packaging'] = None; "
        "sys.modules['packaging.version'] = None; "
        "sys.path.insert(0, %r); "
        "from capture_hidden_states import assert_safe_transformers; "
        "assert_safe_transformers()"
    ) % str(ROOT / "scripts")

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0, (
        "guard must not silently pass without transformers"
    )
    combined = result.stdout + result.stderr
    # Must mention transformers and CVE-2026-4372 (the missing-transformers
    # path fires before packaging is even touched).
    assert "transformers is not installed" in combined, (
        f"expected 'transformers is not installed' message; got: {combined!r}"
    )
    assert "CVE-2026-4372" in combined, (
        f"expected CVE-2026-4372 in message; got: {combined!r}"
    )
    # SystemExit should surface only the actionable message, never a raw
    # import traceback from the deliberately blocked packaging module.
    assert "Traceback" not in combined, (
        f"missing-transformers guard leaked a traceback: {combined!r}"
    )


def test_guard_runs_before_from_pretrained() -> None:
    """AST proof: assert_safe_transformers() must be called before any
    'from transformers import ...' in main(). Compare the max guard-call
    line against the min transformers-import line.
    """
    import ast

    source = (ROOT / "scripts" / "capture_hidden_states.py").read_text()
    tree = ast.parse(source)

    main_fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    guard_lines = [
        node.lineno
        for node in ast.walk(main_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "assert_safe_transformers"
    ]
    import_lines = [
        node.lineno
        for node in ast.walk(main_fn)
        if isinstance(node, ast.ImportFrom) and node.module == "transformers"
    ]

    assert guard_lines, "assert_safe_transformers must be called in main()"
    assert import_lines, "'from transformers import ...' must appear in main()"
    assert max(guard_lines) < min(import_lines), (
        f"guard call(s) at {guard_lines} must precede transformers import(s) "
        f"at {import_lines}"
    )


def test_build_parser_help_text() -> None:
    """Behavioral test of format_help(): must mention trust-remote-code and
    CVE-2026-4372, and must NOT contain the false 'never execute code'
    promise.
    """
    import subprocess
    import sys

    code = (
        "import sys, types; "
        "sys.modules['torch'] = types.ModuleType('torch'); "
        "sys.path.insert(0, %r); "
        "from capture_hidden_states import build_parser; "
        "print(build_parser().format_help())"
    ) % str(ROOT / "scripts")

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"build_parser() failed: {result.stderr!r}"
    )
    help_text = result.stdout
    # Normalize whitespace so argparse line-wrapping doesn't break assertions.
    help_flat = " ".join(help_text.split())

    assert "--trust-remote-code" in help_text
    assert "CVE-2026-4372" in help_text
    # The false promise that untrusted repos "never execute code" must be gone.
    assert "never execute code" not in help_flat, (
        "help text must not promise 'never execute code' — CVE-2026-4372 "
        "bypasses trust_remote_code=False"
    )
    # Must explicitly say the flag alone is insufficient.
    assert "does NOT make loading untrusted models safe" in help_flat, (
        "help text must warn that the flag alone is insufficient"
    )


def test_packaging_declared_as_dependency() -> None:
    """packaging>=24 must be a declared project dependency (used by the
    runtime version guard).
    """
    from packaging.requirements import Requirement

    deps = _pyproject()["project"]["dependencies"]
    pkg_reqs = [Requirement(d) for d in deps if Requirement(d).name == "packaging"]
    assert pkg_reqs, "packaging must be a declared dependency"
    assert len(pkg_reqs) == 1
    assert ">=24" in str(pkg_reqs[0].specifier), (
        f"packaging specifier must be >=24, got {pkg_reqs[0].specifier}"
    )
