"""Registry contract: b12x._OPS and the on-disk op directories stay in
lockstep, and every op honors the META/__all__ shape (invariant #3)."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
B12X_DIR = REPO / "b12x"


def _on_disk_ops() -> list[str]:
    ops = []
    for group_dir in sorted(B12X_DIR.iterdir()):
        if not group_dir.is_dir() or group_dir.name.startswith("_"):
            continue
        for op_dir in sorted(group_dir.iterdir()):
            if (
                not op_dir.is_dir()
                or op_dir.name.startswith("_")
                or not (op_dir / "api.py").is_file()
            ):
                continue
            ops.append(f"{group_dir.name}.{op_dir.name}")
    return ops


def _b12x():
    return importlib.import_module("b12x")


def test_registry_matches_disk():
    b12x = _b12x()
    overrides = set(b12x._OP_MODULE_OVERRIDES)
    assert sorted(set(b12x._OPS) - overrides) == _on_disk_ops(), (
        "b12x._OPS and public op directories under b12x/ must be "
        "in bijection apart from explicit private-module overrides"
    )
    for qualname, module_path in b12x._OP_MODULE_OVERRIDES.items():
        assert qualname in b12x._OPS
        module = importlib.import_module(f"b12x.{module_path}")
        assert module.META.qualname == qualname


def test_list_ops_and_find_op():
    b12x = _b12x()
    metas = b12x.list_ops()
    assert len(metas) == len(b12x._OPS)
    for meta in metas:
        assert b12x.find_op(meta.qualname) is meta
    with pytest.raises(KeyError):
        b12x.find_op("no_such.op")


def test_every_op_meta_contract():
    b12x = _b12x()
    for meta in b12x.list_ops():
        module = importlib.import_module(
            f"b12x.{b12x._op_module_path(meta.qualname)}"
        )
        assert isinstance(module.META, b12x.OpMeta)
        assert set(module.__all__) == set(meta.entry_points) | {"META"}, meta.qualname
        assert any(
            name == "is_supported"
            or (name.startswith("is_") and name.endswith("_supported"))
            for name in meta.entry_points
        ), meta.qualname
        assert meta.archs and set(meta.archs) <= {"sm120a", "sm121a"}, meta.qualname
        assert meta.provenance.commit, f"{meta.qualname} missing provenance commit"
        assert meta.test_path and (REPO / meta.test_path).is_file(), (
            f"{meta.qualname} META.test_path {meta.test_path!r} does not exist"
        )


def test_clear_all_caches_never_forces_imports():
    b12x = _b12x()
    b12x.clear_all_caches()  # must be a no-op / safe with nothing imported


def test_every_op_api_resolves():
    """Force-load every op's api and resolve every declared entry point.

    Catches facade/alias typos without a GPU; needs cutlass (kernel modules
    import it), so the CPU-only CI job skips this one.
    """
    pytest.importorskip("cutlass")
    b12x = _b12x()
    for meta in b12x.list_ops():
        module = importlib.import_module(
            f"b12x.{b12x._op_module_path(meta.qualname)}"
        )
        for name in meta.entry_points:
            assert getattr(module, name) is not None, f"{meta.qualname}.{name}"
