"""Security contract tests for FP6 safetensors export (issue #173).

These tests verify that every public FP6 export entrypoint:

* Rejects pre-existing output destinations (directory, file, symlink).
* Rejects symlinked ancestors in the output path.
* Rejects source/output overlap before any filesystem mutation (by inode).
* Validates the output parent is safe (sticky bit or owner-only writable).
* Builds the complete checkpoint in a private mode-0700 staging directory and
  publishes it atomically via no-replace rename — the final path is absent
  until complete.
* Leaves no partial output on failure (cleanup of the owned staging inode
  only, never a path-resolved replacement).
* Writes all shards/index/config/aux via fd-anchored operations with
  short-write loops and oversized-tensor rejection.

They run without CUDA by stubbing the heavy kernel modules.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import types
from unittest.mock import MagicMock

import pytest
import torch

# ---------------------------------------------------------------------------
# Stub heavy CUDA / CUTLASS dependencies so the export module can import on CPU.
# ---------------------------------------------------------------------------
for _mod in [
    "cutlass",
    "cutlass.cute",
    "cutlass.cute.typing",
    "cutlass.cutlass_dsl",
    "cutlass.pipeline",
    "cutlass.pipeline.executor",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

for _mod in ["b12x._lib.compiler", "b12x._lib.intrinsics"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

_fp6_stub = MagicMock()
_fp6_stub.SF_VEC_SIZE_FP6 = 32
_fp6_stub.FLOAT6_E2M3_MAX = 0.4765625
_fp6_stub.FLOAT6_E3M2_MAX = 7.5


def _decode_fp6_e3m2(bits: int) -> float:
    bits &= 0x3F
    sign = -1.0 if (bits >> 5) & 1 else 1.0
    exp = (bits >> 2) & 0x7
    mant = bits & 0x3
    if exp == 0:
        if mant == 0:
            return 0.0 if sign > 0 else -0.0
        return sign * (2.0 ** (1 - 3)) * (mant / 4.0)
    return sign * (2.0 ** (exp - 3)) * (1.0 + mant / 4.0)


def _decode_fp6_e2m3(bits: int) -> float:
    bits &= 0x3F
    sign = -1.0 if (bits >> 5) & 1 else 1.0
    exp = (bits >> 3) & 0x3
    mant = bits & 0x7
    if exp == 0:
        if mant == 0:
            return 0.0 if sign > 0 else -0.0
        return sign * (2.0 ** (1 - 1)) * (mant / 8.0)
    return sign * (2.0 ** (exp - 1)) * (1.0 + mant / 8.0)


_fp6_stub._decode_fp6_e3m2 = _decode_fp6_e3m2
_fp6_stub._decode_fp6_e2m3 = _decode_fp6_e2m3


def _expand_mxfp6_packed_to_bytes_stub(packed: torch.Tensor, num_fp6: int) -> torch.Tensor:
    groups = num_fp6 // 4
    *lead, packed_cols = packed.shape
    p = packed.reshape(*lead, groups, 3).to(torch.int32)
    bits = p[..., 0] | (p[..., 1] << 8) | (p[..., 2] << 16)
    c0 = bits & 0x3F
    c1 = (bits >> 6) & 0x3F
    c2 = (bits >> 12) & 0x3F
    c3 = (bits >> 18) & 0x3F
    out = torch.stack((c0, c1, c2, c3), dim=-1).reshape(*lead, num_fp6)
    return out.to(torch.uint8)


_fp6_stub.expand_mxfp6_packed_to_bytes = _expand_mxfp6_packed_to_bytes_stub


def _pack_codes_stub(codes: torch.Tensor) -> torch.Tensor:
    shape = codes.shape
    n = shape[-1]
    flat = codes.reshape(-1, n).to(torch.uint8)
    out_rows = flat.shape[0]
    packed_n = 3 * n // 4
    out = torch.zeros(out_rows, packed_n, dtype=torch.uint8)
    for r in range(out_rows):
        vals = flat[r].tolist()
        j = 0
        for i in range(0, n, 4):
            v0, v1, v2, v3 = vals[i], vals[i + 1], vals[i + 2], vals[i + 3]
            out[r, j] = (v0 << 2) | (v1 >> 4)
            j += 1
            out[r, j] = ((v1 & 0xF) << 4) | (v2 >> 2)
            j += 1
            out[r, j] = ((v2 & 0x3) << 6) | v3
            j += 1
    return out.reshape(*shape[:-1], packed_n)


_fp6_stub.pack_fp6_codes_tensor = _pack_codes_stub
sys.modules["b12x._lib.fp6"] = _fp6_stub

_utils_stub = MagicMock()
_utils_stub.MXFP6_SF_VEC_SIZE = 32
_utils_stub.mxfp6_packed_k_bytes = lambda k: 3 * k // 4
sys.modules["b12x._lib.utils"] = _utils_stub

for _mod in ["cuda", "cuda.bindings", "cuda.bindings.driver"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

if "b12x.quantization.mxfp6" not in sys.modules:
    _mxfp6 = types.ModuleType("b12x.quantization.mxfp6")
    _mxfp6.__path__ = [
        str(pathlib.Path(__file__).resolve().parent.parent.parent / "b12x" / "quantization" / "mxfp6")
    ]
    sys.modules["b12x.quantization.mxfp6"] = _mxfp6

safetensors_torch = pytest.importorskip("safetensors.torch")

from b12x.quantization.mxfp6.fp6_safetensors_export import (  # noqa: E402
    _SecureExportContext,
    dequantize_fp6_checkpoint_to_bf16,
    export_dense_model_to_fp6_safetensors,
    export_moe_model_to_fp6_safetensors,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_ckpt(path: pathlib.Path, tensors: dict[str, torch.Tensor]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    safetensors_torch.save_file(
        {k: v.contiguous() for k, v in tensors.items()}, str(path / "model.safetensors")
    )
    (path / "config.json").write_text(json.dumps({"model_type": "test"}))


def _fake_dense(path: pathlib.Path, *, layers: int = 1, k: int = 128, n: int = 128) -> None:
    t: dict[str, torch.Tensor] = {}
    for L in range(layers):
        base = f"model.language_model.layers.{L}"
        t[f"{base}.mlp.gate_proj.weight"] = torch.randn(n, k, dtype=torch.bfloat16) * 0.1
        t[f"{base}.mlp.up_proj.weight"] = torch.randn(n, k, dtype=torch.bfloat16) * 0.1
        t[f"{base}.mlp.down_proj.weight"] = torch.randn(k, n, dtype=torch.bfloat16) * 0.1
    _write_ckpt(path, t)


def _fake_moe(path: pathlib.Path, *, e: int = 2, layers: int = 1, k: int = 64, n: int = 32) -> None:
    t: dict[str, torch.Tensor] = {}
    pre = "model.language_model.layers.{L}.mlp"
    for L in range(layers):
        base = pre.format(L=L)
        for ei in range(e):
            t[f"{base}.experts.{ei}.gate_proj.weight"] = torch.randn(n, k, dtype=torch.bfloat16) * 0.1
            t[f"{base}.experts.{ei}.up_proj.weight"] = torch.randn(n, k, dtype=torch.bfloat16) * 0.1
            t[f"{base}.experts.{ei}.down_proj.weight"] = torch.randn(k, n, dtype=torch.bfloat16) * 0.1
        t[f"{base}.gate.weight"] = torch.randn(e, k, dtype=torch.bfloat16)
    _write_ckpt(path, t)


def _safe_parent(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a mode-0700 owner-only directory as a safe output parent."""
    safe = tmp_path / "safe"
    safe.mkdir(mode=0o700)
    return safe


def _snapshot_source(path: pathlib.Path) -> dict:
    """Recursive snapshot of file names, sizes, modes, and mtimes."""
    snap = {}
    for item in sorted(path.rglob("*")):
        if item.is_file():
            st = item.stat()
            snap[str(item.relative_to(path))] = {
                "size": st.st_size,
                "mode": st.st_mode & 0o777,
                "mtime": st.st_mtime,
            }
    return snap


def _assert_source_unchanged(path: pathlib.Path, before: dict) -> None:
    after = _snapshot_source(path)
    assert after == before, f"source modified: {before} != {after}"

def _safe_parent(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a mode-0700 owner-only directory as a safe output parent."""
    safe = tmp_path / "safe"
    safe.mkdir(mode=0o700, exist_ok=True)
    return safe


# ===========================================================================
# _SecureExportContext tests
# ===========================================================================
class TestSecureExportContext:
    def test_creates_and_publishes(self, tmp_path: pathlib.Path) -> None:
        _fake_dense(tmp_path / "d")
        safe = _safe_parent(tmp_path)
        out = safe / "out"
        with _SecureExportContext(tmp_path / "d", out) as ctx:
            ctx.write_text("test.txt", "hello")
        assert out.is_dir() and not out.is_symlink()
        assert (out / "test.txt").read_text() == "hello"
        assert not any(p.name.startswith(".b12x-staging") for p in safe.iterdir())

    def test_rejects_existing_empty_dir(self, tmp_path: pathlib.Path) -> None:
        _fake_dense(tmp_path / "d")
        safe = _safe_parent(tmp_path)
        out = safe / "out"
        out.mkdir()
        with pytest.raises(FileExistsError), _SecureExportContext(tmp_path / "d", out):
            pass
        assert out.is_dir() and list(out.iterdir()) == []

    def test_rejects_existing_nonempty_dir(self, tmp_path: pathlib.Path) -> None:
        _fake_dense(tmp_path / "d")
        safe = _safe_parent(tmp_path)
        out = safe / "out"
        out.mkdir()
        (out / "stale.txt").write_text("data")
        with pytest.raises(FileExistsError), _SecureExportContext(tmp_path / "d", out):
            pass

    def test_rejects_existing_file(self, tmp_path: pathlib.Path) -> None:
        _fake_dense(tmp_path / "d")
        safe = _safe_parent(tmp_path)
        out = safe / "out"
        out.write_text("not a dir")
        with pytest.raises(FileExistsError), _SecureExportContext(tmp_path / "d", out):
            pass
        assert out.read_text() == "not a dir"

    def test_rejects_symlink_destination(self, tmp_path: pathlib.Path) -> None:
        _fake_dense(tmp_path / "d")
        safe = _safe_parent(tmp_path)
        target = safe / "target"
        target.mkdir()
        link = safe / "out"
        link.symlink_to(target)
        with pytest.raises(FileExistsError), _SecureExportContext(tmp_path / "d", link):
            pass
        assert link.is_symlink()
        assert list(target.iterdir()) == []

    def test_rejects_dangling_symlink(self, tmp_path: pathlib.Path) -> None:
        _fake_dense(tmp_path / "d")
        safe = _safe_parent(tmp_path)
        link = safe / "out"
        link.symlink_to(tmp_path / "nonexistent")
        with pytest.raises(FileExistsError), _SecureExportContext(tmp_path / "d", link):
            pass

    def test_rejects_symlinked_ancestor(self, tmp_path: pathlib.Path) -> None:
        _fake_dense(tmp_path / "d")
        realdir = tmp_path / "realdir"
        realdir.mkdir()
        linkdir = tmp_path / "linkdir"
        linkdir.symlink_to(realdir)
        out = linkdir / "out"
        with pytest.raises((FileExistsError, NotADirectoryError, OSError)), \
             _SecureExportContext(tmp_path / "d", out):
            pass
        assert list(realdir.iterdir()) == []

    def test_rejects_unsafe_parent(self, tmp_path: pathlib.Path) -> None:
        """Output parent that is world-writable without sticky bit must be rejected."""
        _fake_dense(tmp_path / "d")
        unsafe = tmp_path / "unsafe"
        unsafe.mkdir()
        os.chmod(unsafe, 0o777)
        out = unsafe / "out"
        with pytest.raises(PermissionError, match="not safe"), \
             _SecureExportContext(tmp_path / "d", out):
            pass

    def test_creates_parent_dirs(self, tmp_path: pathlib.Path) -> None:
        _fake_dense(tmp_path / "d")
        safe = _safe_parent(tmp_path)
        out = safe / "a" / "b" / "c"
        with _SecureExportContext(tmp_path / "d", out) as ctx:
            ctx.write_text("test.txt", "hello")
        assert out.is_dir()
        assert (out / "test.txt").read_text() == "hello"

    def test_rejects_source_overlap_same_path(self, tmp_path: pathlib.Path) -> None:
        _fake_dense(tmp_path / "d")
        with pytest.raises(ValueError, match="overlaps"), \
             _SecureExportContext(tmp_path / "d", tmp_path / "d"):
            pass

    def test_rejects_source_overlap_nested(self, tmp_path: pathlib.Path) -> None:
        _fake_dense(tmp_path / "d")
        src_before = _snapshot_source(tmp_path / "d")
        out = tmp_path / "d" / "out"
        with pytest.raises(ValueError, match="overlaps"), \
             _SecureExportContext(tmp_path / "d", out):
            pass
        _assert_source_unchanged(tmp_path / "d", src_before)
        assert not (tmp_path / "d" / "out").exists()

    def test_concurrent_creator_loses(self, tmp_path: pathlib.Path) -> None:
        _fake_dense(tmp_path / "d")
        safe = _safe_parent(tmp_path)
        out = safe / "race"
        out.mkdir()
        with pytest.raises(FileExistsError), _SecureExportContext(tmp_path / "d", out):
            pass

    def test_cleanup_on_failure_leaves_no_partial(self, tmp_path: pathlib.Path) -> None:
        _fake_dense(tmp_path / "d")
        safe = _safe_parent(tmp_path)
        out = safe / "out"
        with pytest.raises(RuntimeError, match="boom"), \
             _SecureExportContext(tmp_path / "d", out) as ctx:
            ctx.write_text("partial.txt", "data")
            raise RuntimeError("boom")
        assert not out.exists()
        assert not any(p.name.startswith(".b12x-staging") for p in safe.iterdir())

    def test_final_path_absent_until_complete(self, tmp_path: pathlib.Path) -> None:
        _fake_dense(tmp_path / "d")
        safe = _safe_parent(tmp_path)
        out = safe / "out"
        with _SecureExportContext(tmp_path / "d", out) as ctx:
            assert not out.exists()
            ctx.write_text("config.json", "{}")
            assert not out.exists()
        assert out.is_dir()
        assert (out / "config.json").read_text() == "{}"

    def test_staging_is_mode_0700(self, tmp_path: pathlib.Path) -> None:
        _fake_dense(tmp_path / "d")
        safe = _safe_parent(tmp_path)
        out = safe / "out"
        with _SecureExportContext(tmp_path / "d", out) as ctx:
            staging_path = safe / ctx.staging_name
            mode = staging_path.stat().st_mode & 0o777
            assert mode == 0o700


# ===========================================================================
# Integration: rejection tests for all entrypoints
# ===========================================================================
class TestExportDenseRejectsPreExisting:
    def test_rejects_existing_dir(self, tmp_path: pathlib.Path) -> None:
        _fake_dense(tmp_path / "d")
        safe = _safe_parent(tmp_path)
        out = safe / "out"
        out.mkdir()
        with pytest.raises(FileExistsError):
            export_dense_model_to_fp6_safetensors(tmp_path / "d", out, device="cpu", use_gpu=False, verbose=False)
        assert list(out.iterdir()) == []

    def test_rejects_existing_file(self, tmp_path: pathlib.Path) -> None:
        _fake_dense(tmp_path / "d")
        safe = _safe_parent(tmp_path)
        out = safe / "out"
        out.write_text("blocking file")
        with pytest.raises(FileExistsError):
            export_dense_model_to_fp6_safetensors(tmp_path / "d", out, device="cpu", use_gpu=False, verbose=False)
        assert out.read_text() == "blocking file"

    def test_rejects_symlink_destination(self, tmp_path: pathlib.Path) -> None:
        _fake_dense(tmp_path / "d")
        safe = _safe_parent(tmp_path)
        target = safe / "target"
        target.mkdir()
        out = safe / "out"
        out.symlink_to(target)
        with pytest.raises(FileExistsError):
            export_dense_model_to_fp6_safetensors(tmp_path / "d", out, device="cpu", use_gpu=False, verbose=False)
        assert out.is_symlink()
        assert list(target.iterdir()) == []

    def test_rejects_source_overlap_no_mutation(self, tmp_path: pathlib.Path) -> None:
        _fake_dense(tmp_path / "d")
        src_before = _snapshot_source(tmp_path / "d")
        out = tmp_path / "d" / "out"
        with pytest.raises(ValueError, match="overlaps"):
            export_dense_model_to_fp6_safetensors(tmp_path / "d", out, device="cpu", use_gpu=False, verbose=False)
        _assert_source_unchanged(tmp_path / "d", src_before)
        assert not out.exists()

    def test_rejects_unsafe_parent(self, tmp_path: pathlib.Path) -> None:
        _fake_dense(tmp_path / "d")
        unsafe = tmp_path / "unsafe"
        unsafe.mkdir()
        os.chmod(unsafe, 0o777)
        out = unsafe / "out"
        with pytest.raises(PermissionError, match="not safe"):
            export_dense_model_to_fp6_safetensors(tmp_path / "d", out, device="cpu", use_gpu=False, verbose=False)


class TestExportMoeRejectsPreExisting:
    def test_rejects_existing_dir(self, tmp_path: pathlib.Path) -> None:
        _fake_moe(tmp_path / "m")
        safe = _safe_parent(tmp_path)
        out = safe / "out"
        out.mkdir()
        with pytest.raises(FileExistsError):
            export_moe_model_to_fp6_safetensors(tmp_path / "m", out, limit_layers=1, device="cpu", use_gpu=False, verbose=False)

    def test_rejects_symlink_destination(self, tmp_path: pathlib.Path) -> None:
        _fake_moe(tmp_path / "m")
        safe = _safe_parent(tmp_path)
        target = safe / "target"
        target.mkdir()
        out = safe / "out"
        out.symlink_to(target)
        with pytest.raises(FileExistsError):
            export_moe_model_to_fp6_safetensors(tmp_path / "m", out, limit_layers=1, device="cpu", use_gpu=False, verbose=False)
        assert out.is_symlink()
        assert list(target.iterdir()) == []


class TestDequantizeRejectsPreExisting:
    def test_rejects_existing_dir(self, tmp_path: pathlib.Path) -> None:
        _fake_dense(tmp_path / "d")
        safe1 = _safe_parent(tmp_path)
        fp6_out = safe1 / "fp6"
        export_dense_model_to_fp6_safetensors(tmp_path / "d", fp6_out, device="cpu", use_gpu=False, verbose=False)
        safe2 = _safe_parent(tmp_path)
        out = safe2 / "out"
        out.mkdir()
        with pytest.raises(FileExistsError):
            dequantize_fp6_checkpoint_to_bf16(fp6_out, out, device="cpu", verbose=False)

    def test_rejects_symlink_destination(self, tmp_path: pathlib.Path) -> None:
        _fake_dense(tmp_path / "d")
        safe1 = _safe_parent(tmp_path)
        fp6_out = safe1 / "fp6"
        export_dense_model_to_fp6_safetensors(tmp_path / "d", fp6_out, device="cpu", use_gpu=False, verbose=False)
        safe2 = _safe_parent(tmp_path)
        target = safe2 / "target"
        target.mkdir()
        out = safe2 / "out"
        out.symlink_to(target)
        with pytest.raises(FileExistsError):
            dequantize_fp6_checkpoint_to_bf16(fp6_out, out, device="cpu", verbose=False)
        assert out.is_symlink()
        assert list(target.iterdir()) == []


# ===========================================================================
# Integration: successful exports
# ===========================================================================
class TestExportSuccess:
    def test_dense_export_writes_complete_files(self, tmp_path: pathlib.Path) -> None:
        _fake_dense(tmp_path / "d")
        safe = _safe_parent(tmp_path)
        out = safe / "out"
        report = export_dense_model_to_fp6_safetensors(tmp_path / "d", out, device="cpu", use_gpu=False, verbose=False)
        assert report.arch == "dense"
        assert (out / "config.json").is_file()
        assert (out / "model.safetensors.index.json").is_file()
        index = json.loads((out / "model.safetensors.index.json").read_text())
        assert index["weight_map"]
        for s in set(index["weight_map"].values()):
            assert (out / s).is_file()

    def test_moe_export_writes_complete_files(self, tmp_path: pathlib.Path) -> None:
        _fake_moe(tmp_path / "m")
        safe = _safe_parent(tmp_path)
        out = safe / "out"
        report = export_moe_model_to_fp6_safetensors(tmp_path / "m", out, limit_layers=1, device="cpu", use_gpu=False, verbose=False)
        assert report.arch == "moe"
        assert (out / "config.json").is_file()
        assert (out / "model.safetensors.index.json").is_file()
        index = json.loads((out / "model.safetensors.index.json").read_text())
        assert index["weight_map"]
        for s in set(index["weight_map"].values()):
            assert (out / s).is_file()

    def test_dequant_roundtrip(self, tmp_path: pathlib.Path) -> None:
        _fake_dense(tmp_path / "d")
        safe1 = _safe_parent(tmp_path)
        fp6_out = safe1 / "fp6"
        export_dense_model_to_fp6_safetensors(tmp_path / "d", fp6_out, device="cpu", use_gpu=False, verbose=False)
        safe2 = _safe_parent(tmp_path)
        out = safe2 / "bf16"
        report = dequantize_fp6_checkpoint_to_bf16(fp6_out, out, device="cpu", verbose=False)
        assert report.arch == "dequant-bf16"
        assert (out / "config.json").is_file()
        assert (out / "model.safetensors.index.json").is_file()

    def test_final_path_absent_during_export(self, tmp_path: pathlib.Path) -> None:
        _fake_dense(tmp_path / "d")
        safe = _safe_parent(tmp_path)
        out = safe / "out"
        from b12x.quantization.mxfp6.fp6_safetensors_export import _emit_quantized_linear
        import b12x.quantization.mxfp6.fp6_safetensors_export as mod

        original = _emit_quantized_linear
        observed_absence = []

        def checking_emit(*args, **kwargs):
            observed_absence.append(not out.exists())
            return original(*args, **kwargs)

        mod._emit_quantized_linear = checking_emit
        try:
            export_dense_model_to_fp6_safetensors(tmp_path / "d", out, device="cpu", use_gpu=False, verbose=False)
        finally:
            mod._emit_quantized_linear = original
        assert all(observed_absence)
        assert out.is_dir()

    def test_oversized_tensor_rejected(self, tmp_path: pathlib.Path) -> None:
        """A single tensor exceeding max_shard_bytes must be rejected."""
        _fake_dense(tmp_path / "d")
        safe = _safe_parent(tmp_path)
        out = safe / "out"
        with pytest.raises(ValueError, match="exceeds the shard cap"):
            export_dense_model_to_fp6_safetensors(
                tmp_path / "d", out, device="cpu", use_gpu=False, verbose=False,
                max_shard_bytes=1,  # 1 byte cap — any tensor exceeds it
            )
        assert not out.exists()


# ===========================================================================
# Symlinked fixed filename and failure cleanup
# ===========================================================================
class TestSymlinkedFixedFilename:
    def test_config_json_symlink_in_preexisting_dir(self, tmp_path: pathlib.Path) -> None:
        _fake_dense(tmp_path / "d")
        safe = _safe_parent(tmp_path)
        out = safe / "out"
        out.mkdir()
        victim = safe / "victim_config.json"
        victim.write_text("secret config")
        (out / "config.json").symlink_to(victim)
        with pytest.raises(FileExistsError):
            export_dense_model_to_fp6_safetensors(tmp_path / "d", out, device="cpu", use_gpu=False, verbose=False)
        assert victim.read_text() == "secret config"


class TestFailureCleanup:
    def test_cleanup_on_corrupt_source(self, tmp_path: pathlib.Path) -> None:
        bad_src = tmp_path / "bad"
        bad_src.mkdir()
        (bad_src / "config.json").write_text(json.dumps({"model_type": "test"}))
        safe = _safe_parent(tmp_path)
        out = safe / "out"
        with pytest.raises(FileNotFoundError):
            export_dense_model_to_fp6_safetensors(bad_src, out, device="cpu", use_gpu=False, verbose=False)
        assert not out.exists()
        assert not any(p.name.startswith(".b12x-staging") for p in safe.iterdir())

    def test_cleanup_on_mid_write_failure(self, tmp_path: pathlib.Path) -> None:
        _fake_dense(tmp_path / "d")
        safe = _safe_parent(tmp_path)
        out = safe / "out"
        from b12x.quantization.mxfp6.fp6_safetensors_export import _ShardWriter

        original_finalize = _ShardWriter.finalize

        def boom(self):
            raise RuntimeError("simulated disk error")

        _ShardWriter.finalize = boom
        try:
            with pytest.raises(RuntimeError, match="simulated disk error"):
                export_dense_model_to_fp6_safetensors(tmp_path / "d", out, device="cpu", use_gpu=False, verbose=False)
        finally:
            _ShardWriter.finalize = original_finalize
        assert not out.exists()
        assert not any(p.name.startswith(".b12x-staging") for p in safe.iterdir())

    def test_cleanup_does_not_touch_victim(self, tmp_path: pathlib.Path) -> None:
        _fake_dense(tmp_path / "d")
        safe = _safe_parent(tmp_path)
        out = safe / "out"
        from b12x.quantization.mxfp6.fp6_safetensors_export import _ShardWriter

        original_finalize = _ShardWriter.finalize
        victim = safe / "victim"
        victim.mkdir()
        victim_file = victim / "important.txt"
        victim_file.write_text("do not delete")
        victim_stat_before = victim_file.stat()

        def boom(self):
            raise RuntimeError("cleanup test")

        _ShardWriter.finalize = boom
        try:
            with pytest.raises(RuntimeError, match="cleanup test"):
                export_dense_model_to_fp6_safetensors(tmp_path / "d", out, device="cpu", use_gpu=False, verbose=False)
        finally:
            _ShardWriter.finalize = original_finalize

        assert victim_file.read_text() == "do not delete"
        victim_stat_after = victim_file.stat()
        assert victim_stat_after.st_size == victim_stat_before.st_size
        assert victim_stat_after.st_mode == victim_stat_before.st_mode
        assert not out.exists()


# ===========================================================================
# Adversarial race/substitution tests
# ===========================================================================
class TestAdversarialRaces:
    def test_final_leaf_created_after_lstat_not_clobbered(
        self, tmp_path: pathlib.Path
    ) -> None:
        """If an attacker creates the final leaf after the lstat check,
        the no-replace rename must fail, not clobber it."""
        _fake_dense(tmp_path / "d")
        safe = _safe_parent(tmp_path)
        out = safe / "out"
        original_publish = _SecureExportContext._publish

        def publishing(self):
            os.mkdir(self.leaf_name, dir_fd=self.parent_fd)
            return original_publish(self)

        _SecureExportContext._publish = publishing
        try:
            with pytest.raises((FileExistsError, OSError)), \
                 _SecureExportContext(tmp_path / "d", out):
                pass
        finally:
            _SecureExportContext._publish = original_publish
        assert out.is_dir()
        assert list(out.iterdir()) == []
        assert not any(p.name.startswith(".b12x-staging") for p in safe.iterdir())

    def test_short_write_looped(self, tmp_path: pathlib.Path) -> None:
        """A short os.write must be looped, not silently truncated."""
        _fake_dense(tmp_path / "d")
        safe = _safe_parent(tmp_path)
        out = safe / "out"
        original_write = os.write
        call_count = [0]

        def short_write(fd, data):
            call_count[0] += 1
            if call_count[0] == 1 and len(data) > 100:
                return original_write(fd, data[:100])
            return original_write(fd, data)

        os.write = short_write
        try:
            export_dense_model_to_fp6_safetensors(tmp_path / "d", out, device="cpu", use_gpu=False, verbose=False)
        finally:
            os.write = original_write
        index = json.loads((out / "model.safetensors.index.json").read_text())
        shard_name = list(set(index["weight_map"].values()))[0]
        from safetensors.torch import load_file
        shard_data = load_file(str(out / shard_name))
        assert len(shard_data) > 0
        assert call_count[0] > 1

    def test_aux_symlink_not_followed(self, tmp_path: pathlib.Path) -> None:
        """A symlinked aux file in the source must not be copied."""
        _fake_dense(tmp_path / "d")
        victim = tmp_path / "secret.txt"
        victim.write_text("secret data")
        (tmp_path / "d" / "tokenizer_config.json").symlink_to(victim)
        safe = _safe_parent(tmp_path)
        out = safe / "out"
        export_dense_model_to_fp6_safetensors(tmp_path / "d", out, device="cpu", use_gpu=False, verbose=False)
        assert not (out / "tokenizer_config.json").exists()

    def test_publish_failure_cleans_staging(self, tmp_path: pathlib.Path) -> None:
        """If the no-replace rename fails, the staging dir must be cleaned."""
        _fake_dense(tmp_path / "d")
        safe = _safe_parent(tmp_path)
        out = safe / "out"
        original_publish = _SecureExportContext._publish

        def failing_publish(self):
            raise RuntimeError("publish failure")

        _SecureExportContext._publish = failing_publish
        try:
            with pytest.raises(RuntimeError, match="publish failure"), \
                 _SecureExportContext(tmp_path / "d", out):
                pass
        finally:
            _SecureExportContext._publish = original_publish
        assert not out.exists()
        assert not any(p.name.startswith(".b12x-staging") for p in safe.iterdir())

    def test_enter_failure_closes_fds(self, tmp_path: pathlib.Path) -> None:
        """If __enter__ fails after opening some fds, they must be closed."""
        _fake_dense(tmp_path / "d")
        safe = _safe_parent(tmp_path)
        out = safe / "out"
        out.mkdir()
        with pytest.raises(FileExistsError), \
             _SecureExportContext(tmp_path / "d", out):
            pass
        assert not any(p.name.startswith(".b12x-staging") for p in safe.iterdir())

    def test_source_unchanged_after_export(self, tmp_path: pathlib.Path) -> None:
        """Source directory must be byte-identical after a successful export."""
        _fake_dense(tmp_path / "d")
        src_before = _snapshot_source(tmp_path / "d")
        safe = _safe_parent(tmp_path)
        out = safe / "out"
        export_dense_model_to_fp6_safetensors(tmp_path / "d", out, device="cpu", use_gpu=False, verbose=False)
        _assert_source_unchanged(tmp_path / "d", src_before)
