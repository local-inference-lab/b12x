"""Compile-cache rooting: the highest-risk mechanical edit of the restructure.

Wrong fingerprint root ⇒ silent stale disk-cache hits (running old kernels),
so these assertions are load-bearing.
"""

from __future__ import annotations

import errno
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

compiler = importlib.import_module("b12x._lib.compiler")
kernel_resources = importlib.import_module(
    "validation.cutlass_migration.evidence.kernel_resources"
)
ptx_capture = importlib.import_module(
    "validation.cutlass_migration.acceptance.corpus.ptx_capture"
)


def test_package_root_is_the_b12x_package():
    root = compiler._PACKAGE_ROOT
    assert root.name == "b12x", root
    assert (root / "_lib" / "compiler.py").is_file()


def test_fingerprint_tracks_source_edits(tmp_path, monkeypatch):
    (tmp_path / "kernel.py").write_text("x = 1\n")
    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    monkeypatch.setattr(compiler, "_PACKAGE_ROOT", tmp_path)

    before = compiler._compute_b12x_package_fingerprint()

    (pycache / "kernel.cpython-312.pyc").write_bytes(b"ignored")
    assert compiler._compute_b12x_package_fingerprint() == before, (
        "__pycache__ must not affect the fingerprint"
    )

    (tmp_path / "kernel.py").write_text("x = 2\n")
    after = compiler._compute_b12x_package_fingerprint()
    assert after != before, "editing any source must change the fingerprint"


def test_cache_dir_resolution_order(monkeypatch):
    for name in ("B12X_COMPILE_CACHE_DIR", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)

    assert compiler._cute_compile_cache_dir() == (
        Path.home() / ".cache" / "b12x" / "compile"
    )

    monkeypatch.setenv("XDG_CACHE_HOME", "/xdg")
    assert compiler._cute_compile_cache_dir() == Path("/xdg/b12x/compile")

    monkeypatch.setenv("B12X_COMPILE_CACHE_DIR", "/explicit")
    assert compiler._cute_compile_cache_dir() == Path("/explicit")


def test_disk_cache_key_includes_device_uuid_and_forwards_ordinal(monkeypatch):
    compile_callable = object()
    seen = {"calls": 0}

    def _fake_device_uuid_key(ordinal):
        seen["calls"] += 1
        seen["ordinal"] = ordinal
        return ("device_uuid", "gpu-3")

    monkeypatch.setattr(compiler, "_current_device_ordinal", lambda: 3)
    monkeypatch.setattr(compiler, "_device_uuid_key", _fake_device_uuid_key)
    monkeypatch.setattr(
        compiler,
        "_static_compile_cache_context",
        lambda _callable: (
            "package",
            "toolchain",
            (),
            (),
        ),
    )

    payload = compiler._compile_disk_cache_payload(
        compile_callable,
        test_disk_cache_key_includes_device_uuid_and_forwards_ordinal,
        (),
        {},
    )
    repeated_payload = compiler._compile_disk_cache_payload(
        compile_callable,
        test_disk_cache_key_includes_device_uuid_and_forwards_ordinal,
        (),
        {},
    )

    assert payload[0] == "b12x_cute_compile_cache_v3"
    assert payload[4] == ("device_uuid", "gpu-3")
    assert repeated_payload == payload
    assert seen["ordinal"] == 3
    assert seen["calls"] == 1


def test_device_uuid_key_retries_after_unavailable(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(compiler, "_DEVICE_UUID_KEYS", {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert compiler._device_uuid_key() is None
    assert compiler._DEVICE_UUID_KEYS == {}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(uuid="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
    )
    expected = (
        "device_uuid",
        "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    )
    assert compiler._device_uuid_key() == expected
    assert {0: expected} == compiler._DEVICE_UUID_KEYS


def test_device_uuid_key_is_memoized_per_ordinal(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(compiler, "_DEVICE_UUID_KEYS", {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    boards = {0: "gpu-zero", 1: "gpu-one"}
    probes = {0: 0, 1: 0}
    current = {"index": 0}
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current["index"])

    def _properties(device):
        probes[device] += 1
        return SimpleNamespace(uuid=boards[device])

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        _properties,
    )

    assert compiler._device_uuid_key() == ("device_uuid", "gpu-zero")
    current["index"] = 1
    assert compiler._device_uuid_key() == ("device_uuid", "gpu-one")
    assert compiler._device_uuid_key(0) == ("device_uuid", "gpu-zero")
    assert compiler._DEVICE_UUID_KEYS == {
        0: ("device_uuid", "gpu-zero"),
        1: ("device_uuid", "gpu-one"),
    }
    assert probes == {0: 1, 1: 1}
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: (_ for _ in ()).throw(AssertionError("cached UUID was re-probed")),
    )
    assert compiler._device_uuid_key(0) == ("device_uuid", "gpu-zero")


def test_disk_payload_retries_uuid_probe_failure(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(compiler, "_DEVICE_UUID_KEYS", {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        compiler,
        "_static_compile_cache_context",
        lambda _callable: ("package", "toolchain", (), ()),
    )
    attempts = {"count": 0}

    def _properties(_device):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("transient CUDA probe failure")
        return SimpleNamespace(uuid="gpu-zero")

    monkeypatch.setattr(torch.cuda, "get_device_properties", _properties)
    compile_callable = object()

    first = compiler._compile_disk_cache_payload(
        compile_callable,
        test_disk_payload_retries_uuid_probe_failure,
        (),
        {},
    )
    second = compiler._compile_disk_cache_payload(
        compile_callable,
        test_disk_payload_retries_uuid_probe_failure,
        (),
        {},
    )
    third = compiler._compile_disk_cache_payload(
        compile_callable,
        test_disk_payload_retries_uuid_probe_failure,
        (),
        {},
    )

    assert first[4] is None
    assert second[4] == ("device_uuid", "gpu-zero")
    assert third == second
    assert attempts["count"] == 2


def test_explicit_cache_payload_includes_device_uuid(monkeypatch):
    compile_callable = object()
    device_uuid = ("device_uuid", "gpu-seven")
    compile_options = ("--opt-level=2",)
    compile_environment = (("CUTE_DSL_ARCH", "sm_120"),)
    monkeypatch.setattr(compiler, "_current_device_ordinal", lambda: 7)
    monkeypatch.setattr(
        compiler,
        "_device_uuid_key",
        lambda ordinal: device_uuid if ordinal == 7 else None,
    )
    monkeypatch.setattr(
        compiler,
        "_static_compile_cache_context",
        lambda _callable: (
            "a" * 64,
            (("python", "cpython", (3, 12, 0)),),
            compile_options,
            compile_environment,
        ),
    )
    compile_spec = compiler.KernelCompileSpec.from_facts(
        "test.uuid.cache",
        1,
        ("rows", 8),
    )
    kwargs = {"mode": "test"}
    kwargs_json, kwargs_hash = compiler._compile_kwargs_json_key(kwargs)

    payload = compiler._compile_disk_cache_payload(
        compile_callable,
        test_explicit_cache_payload_includes_device_uuid,
        (),
        kwargs,
        compile_spec,
    )

    assert len(payload) == 11
    assert payload[0] == "b12x_cute_compile_cache_v6_explicit_spec"
    assert payload[4] == device_uuid
    assert payload[5:11] == (
        compile_spec.hash_key,
        compile_spec.json_key,
        kwargs_hash,
        kwargs_json,
        compile_options,
        compile_environment,
    )


def test_uuid_unavailable_disables_disk_cache(monkeypatch):
    monkeypatch.setenv("B12X_COMPILE_DISK_CACHE", "1")
    payload = (
        "b12x_cute_compile_cache_v3",
        ("function", "test", "kernel"),
        "package",
        "toolchain",
        None,
        (),
        (),
        (),
        (),
    )

    assert not compiler._cute_compile_disk_cache_enabled_for_payload(payload)


def test_explicit_memory_cache_hit_skips_freeze_and_disk_payload(monkeypatch):
    cute = pytest.importorskip("cutlass.cute")
    compile_callable = object()
    compiled = object()
    compile_spec = compiler.KernelCompileSpec.from_facts(
        "test.uuid.hot_path",
        1,
        ("rows", 8),
    )
    monkeypatch.setattr(cute, "compile", compile_callable)
    monkeypatch.setattr(
        compiler,
        "_device_uuid_key",
        lambda _ordinal: (_ for _ in ()).throw(
            AssertionError("memory hit probed the device UUID")
        ),
    )
    monkeypatch.setattr(compiler, "_memory_cache_get", lambda key: compiled)
    monkeypatch.setattr(
        compiler,
        "_compile_disk_cache_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("memory hit rebuilt the disk payload")
        ),
    )
    runtime_control = importlib.import_module("b12x._lib.runtime_control")

    runtime_control.freeze_kernel_resolution("cached compile remains launchable")
    try:
        assert (
            compiler.compile(
                test_explicit_memory_cache_hit_skips_freeze_and_disk_payload,
                compile_spec=compile_spec,
            )
            is compiled
        )
    finally:
        runtime_control.unfreeze_kernel_resolution()


def test_frozen_memory_miss_rejects_before_disk_cache_load(monkeypatch):
    cute = pytest.importorskip("cutlass.cute")
    runtime_control = importlib.import_module("b12x._lib.runtime_control")
    compile_spec = compiler.KernelCompileSpec.from_facts(
        "test.freeze.disk_hit",
        1,
        ("rows", 8),
    )
    monkeypatch.setattr(cute, "compile", object())
    monkeypatch.setattr(compiler, "_memory_cache_get", lambda _key: None)
    monkeypatch.setattr(
        compiler,
        "_compile_disk_cache_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("frozen miss built a persistent-cache payload")
        ),
    )
    monkeypatch.setattr(
        compiler,
        "_load_cute_compile_from_disk",
        lambda _key: (_ for _ in ()).throw(
            AssertionError("frozen miss loaded a persistent module")
        ),
    )

    runtime_control.freeze_kernel_resolution("disk hits must not bypass freeze")
    try:
        with pytest.raises(runtime_control.KernelResolutionFrozenError):
            compiler.compile(
                test_frozen_memory_miss_rejects_before_disk_cache_load,
                compile_spec=compile_spec,
            )
    finally:
        runtime_control.unfreeze_kernel_resolution()


def test_v6_semantic_payload_matches_independent_validators(monkeypatch):
    device_uuid = ("device_uuid", "gpu-contract")
    monkeypatch.setattr(compiler, "_current_device_ordinal", lambda: 0)
    monkeypatch.setattr(compiler, "_device_uuid_key", lambda ordinal: device_uuid)
    monkeypatch.setattr(
        compiler,
        "_static_compile_cache_context",
        lambda _callable: (
            "a" * 64,
            (("python", "cpython", (3, 12, 0)),),
            ("--opt-level=2",),
            (("CUTE_DSL_ARCH", "sm_120"),),
        ),
    )
    compile_spec = compiler.KernelCompileSpec.from_facts(
        "test.uuid.manifest",
        1,
        ("rows", 8),
    )
    payload = compiler._compile_disk_cache_payload(
        object(),
        test_v6_semantic_payload_matches_independent_validators,
        (),
        {"mode": "test"},
        compile_spec,
    )
    serialized_payload = compiler._manifest_json_value(payload)
    expected = compiler._semantic_compile_manifest_payload(payload)

    assert (
        ptx_capture._semantic_payload_from_cache_payload(serialized_payload) == expected
    )
    assert (
        kernel_resources._semantic_payload_from_cache_payload(serialized_payload)
        == expected
    )



# ---------------------------------------------------------------------------
# Issue #168: native cache integrity / ownership / symlink hardening
# ---------------------------------------------------------------------------

import hashlib
import json
import multiprocessing
import os
import stat
import threading


def _open_cache_root_worker(path: str, queue) -> None:
    try:
        fd = compiler._open_secure_dir_fd(Path(path))
        os.close(fd)
        queue.put(None)
    except BaseException as exc:
        queue.put(f"{type(exc).__name__}: {exc}")


_SIMPLE_PAYLOAD = (
    "b12x_cute_compile_cache_v6",
    ("target",),
    "fp",
    "tc",
    None,
    None,
    None,
    None,
    "env",
)


def _simple_func():
    pass


def _payload_cache_key(payload):
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


_CK = _payload_cache_key(_SIMPLE_PAYLOAD)


def _install_fake_external_binary_module(monkeypatch, fake_cls):
    for mod_name in (
        "cutlass",
        "cutlass.base_dsl",
        "cutlass.base_dsl.export",
        "cutlass.base_dsl.export.external_binary_module",
    ):
        if mod_name not in sys.modules:
            monkeypatch.setitem(
                sys.modules, mod_name, types.ModuleType(mod_name)
            )
    sys.modules[
        "cutlass.base_dsl.export.external_binary_module"
    ].ExternalBinaryModule = fake_cls


def _make_full_manifest(cache_key, payload, func, object_bytes):
    return compiler._build_compile_manifest(
        cache_key, payload, func, object_bytes, compiled=None
    )


def _make_secure_cache(tmp_path, cache_key, payload=_SIMPLE_PAYLOAD, func=_simple_func):
    cache_root = tmp_path / "cache"
    shard_dir = cache_root / cache_key[:2]
    shard_dir.mkdir(parents=True, exist_ok=True)
    object_name = f"{cache_key}.o"
    manifest_name = f"{cache_key}.json"
    object_path = shard_dir / object_name
    manifest_path = shard_dir / manifest_name
    object_bytes = b"\x7fELF\x02\x01\x01\x00fake-native-object"
    object_path.write_bytes(object_bytes)
    manifest = _make_full_manifest(cache_key, payload, func, object_bytes)
    manifest_path.write_text(json.dumps(manifest) + "\n")
    os.chmod(object_path, 0o600)
    os.chmod(manifest_path, 0o600)
    os.chmod(shard_dir, 0o700)
    os.chmod(cache_root, 0o700)
    return {
        "root": cache_root,
        "shard": shard_dir,
        "object": object_path,
        "manifest": manifest_path,
        "bytes": object_bytes,
        "manifest_dict": manifest,
        "cache_key": cache_key,
    }


def _recompute_evidence_sha(cache_key, sha, launch_metadata):
    evidence = json.dumps(
        {"cache_key": cache_key, "object_sha256": sha, "launch_metadata": launch_metadata},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    )
    return hashlib.sha256(evidence.encode("utf-8")).hexdigest()


# -- _open_secure_dir_fd ---------------------------------------------------


def test_open_secure_dir_fd_creates_private_dir(tmp_path):
    d = tmp_path / "newcache"
    fd = compiler._open_secure_dir_fd(d)
    try:
        st = os.fstat(fd)
        assert st.st_mode & 0o777 == 0o700
    finally:
        os.close(fd)


def test_open_secure_dir_fd_rejects_symlink_root(tmp_path):
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(OSError) as ei:
        compiler._open_secure_dir_fd(link)
    assert ei.value.errno in (errno.ELOOP, errno.ENOTDIR)


def test_open_secure_dir_fd_rejects_world_writable(tmp_path):
    d = tmp_path / "ww"
    d.mkdir()
    os.chmod(d, 0o707)
    with pytest.raises(RuntimeError, match="group/other"):
        compiler._open_secure_dir_fd(d)


def test_open_secure_dir_fd_rejects_world_writable_ancestor(tmp_path):
    p = tmp_path / "wwp"
    p.mkdir()
    os.chmod(p, 0o777)
    c = p / "cache"
    c.mkdir(mode=0o700)
    with pytest.raises(RuntimeError, match="world-writable"):
        compiler._open_secure_dir_fd(c)
    os.chmod(p, 0o755)


def test_open_secure_dir_fd_rejects_root():
    with pytest.raises(RuntimeError, match="must not be /"):
        compiler._open_secure_dir_fd(Path("/"))


# -- _fstat_assert_regular_0600: nlink==1 ---------------------------------


def test_fstat_rejects_hardlink(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"x")
    os.chmod(p, 0o600)
    os.link(p, tmp_path / "link")
    fd = os.open(str(p), os.O_RDONLY)
    try:
        with pytest.raises(RuntimeError, match="hard links"):
            compiler._fstat_assert_regular_0600(fd, "f")
    finally:
        os.close(fd)


# -- _validate_launch_metadata ---------------------------------------------


def test_validate_launch_metadata_accepts_exact():
    compiler._validate_launch_metadata({
        "status": "exact",
        "source": "cutlass-final-llvm-launch-config-field-2",
        "launch_dynamic_smem_bytes": {"kernel0": [1024]},
    })


def test_validate_launch_metadata_accepts_unknown():
    compiler._validate_launch_metadata({
        "status": "unknown",
        "source": "cutlass-final-llvm-launch-config-field-2",
        "reason": "compiled-ir-unavailable",
        "launch_dynamic_smem_bytes": {},
    })


def test_validate_launch_metadata_rejects_bad_status():
    with pytest.raises(ValueError, match="status"):
        compiler._validate_launch_metadata({
            "status": "evil", "source": "s",
            "launch_dynamic_smem_bytes": {},
        })


def test_validate_launch_metadata_rejects_non_dict():
    with pytest.raises(ValueError, match="not a dict"):
        compiler._validate_launch_metadata("string")


def test_validate_launch_metadata_rejects_exact_empty_smem():
    with pytest.raises(ValueError, match="non-empty"):
        compiler._validate_launch_metadata({
            "status": "exact", "source": "s",
            "launch_dynamic_smem_bytes": {},
        })


def test_validate_launch_metadata_rejects_unknown_with_smem():
    with pytest.raises(ValueError, match="empty"):
        compiler._validate_launch_metadata({
            "status": "unknown", "source": "s", "reason": "r",
            "launch_dynamic_smem_bytes": {"k": [1]},
        })


def test_validate_launch_metadata_rejects_missing_reason():
    with pytest.raises(ValueError, match="reason"):
        compiler._validate_launch_metadata({
            "status": "unknown", "source": "s",
            "launch_dynamic_smem_bytes": {},
        })


def test_validate_launch_metadata_rejects_extra_key():
    with pytest.raises(ValueError, match="extra"):
        compiler._validate_launch_metadata({
            "status": "exact", "source": "s",
            "launch_dynamic_smem_bytes": {"k": [1]},
            "evil": True,
        })


def test_validate_launch_metadata_rejects_negative_smem():
    with pytest.raises(ValueError, match="nonneg"):
        compiler._validate_launch_metadata({
            "status": "exact", "source": "s",
            "launch_dynamic_smem_bytes": {"k": [-1]},
        })


def test_validate_launch_metadata_rejects_exact_reason():
    with pytest.raises(ValueError, match="reason"):
        compiler._validate_launch_metadata({
            "status": "exact",
            "source": "s",
            "reason": "not canonical for exact status",
            "launch_dynamic_smem_bytes": {"k": [0]},
        })


# -- _verify_manifest_identity: exact v3 with launch_metadata --------------


def test_verify_manifest_identity_accepts_valid():
    data = b"object"
    m = _make_full_manifest(_CK, _SIMPLE_PAYLOAD, _simple_func, data)
    sha = compiler._verify_manifest_identity(
        m, _CK, _SIMPLE_PAYLOAD, _simple_func, len(data)
    )
    assert sha == m["object_sha256"]


def test_verify_manifest_rejects_bad_launch_status():
    data = b"object"
    m = _make_full_manifest(_CK, _SIMPLE_PAYLOAD, _simple_func, data)
    m["launch_metadata"]["status"] = "evil"
    m["artifact_evidence_sha256"] = _recompute_evidence_sha(
        _CK, m["object_sha256"], m["launch_metadata"]
    )
    with pytest.raises(ValueError, match="status"):
        compiler._verify_manifest_identity(
            m, _CK, _SIMPLE_PAYLOAD, _simple_func, len(data)
        )


def test_verify_manifest_rejects_missing_launch_key():
    data = b"object"
    m = _make_full_manifest(_CK, _SIMPLE_PAYLOAD, _simple_func, data)
    del m["launch_metadata"]["source"]
    m["artifact_evidence_sha256"] = _recompute_evidence_sha(
        _CK, m["object_sha256"], m["launch_metadata"]
    )
    with pytest.raises(ValueError, match="launch_metadata"):
        compiler._verify_manifest_identity(
            m, _CK, _SIMPLE_PAYLOAD, _simple_func, len(data)
        )


def test_verify_manifest_rejects_extra_launch_key():
    data = b"object"
    m = _make_full_manifest(_CK, _SIMPLE_PAYLOAD, _simple_func, data)
    m["launch_metadata"]["evil"] = True
    m["artifact_evidence_sha256"] = _recompute_evidence_sha(
        _CK, m["object_sha256"], m["launch_metadata"]
    )
    with pytest.raises(ValueError, match="extra"):
        compiler._verify_manifest_identity(
            m, _CK, _SIMPLE_PAYLOAD, _simple_func, len(data)
        )


def test_verify_manifest_rejects_wrong_semantic_payload():
    m = _make_full_manifest(_CK, _SIMPLE_PAYLOAD, _simple_func, b"x")
    m["semantic_payload"] = {"tampered": True}
    with pytest.raises(ValueError, match="semantic_payload"):
        compiler._verify_manifest_identity(
            m, _CK, _SIMPLE_PAYLOAD, _simple_func, 1
        )


def test_verify_manifest_rejects_wrong_toolchain():
    m = _make_full_manifest(_CK, _SIMPLE_PAYLOAD, _simple_func, b"x")
    m["toolchain"] = "wrong"
    with pytest.raises(ValueError, match="toolchain"):
        compiler._verify_manifest_identity(
            m, _CK, _SIMPLE_PAYLOAD, _simple_func, 1
        )


def test_verify_manifest_rejects_missing_field():
    m = _make_full_manifest(_CK, _SIMPLE_PAYLOAD, _simple_func, b"x")
    del m["cache_payload"]
    with pytest.raises(ValueError, match="key set"):
        compiler._verify_manifest_identity(
            m, _CK, _SIMPLE_PAYLOAD, _simple_func, 1
        )


def test_verify_manifest_rejects_extra_field():
    m = _make_full_manifest(_CK, _SIMPLE_PAYLOAD, _simple_func, b"x")
    m["extra"] = True
    with pytest.raises(ValueError, match="key set"):
        compiler._verify_manifest_identity(
            m, _CK, _SIMPLE_PAYLOAD, _simple_func, 1
        )


def test_verify_manifest_rejects_non_dict():
    with pytest.raises(ValueError, match="not a JSON object"):
        compiler._verify_manifest_identity(
            "string", _CK, _SIMPLE_PAYLOAD, _simple_func, 1
        )


# -- _load_cute_compile_from_disk: no fallback, requires payload+func -------


def test_load_without_payload_returns_none(tmp_path, monkeypatch):
    entry = _make_secure_cache(tmp_path, _CK)
    monkeypatch.setenv("B12X_COMPILE_CACHE_DIR", str(entry["root"]))
    assert compiler._load_cute_compile_from_disk(_CK) is None


def test_load_rejects_missing_manifest(tmp_path, monkeypatch):
    entry = _make_secure_cache(tmp_path, _CK)
    entry["manifest"].unlink()
    monkeypatch.setenv("B12X_COMPILE_CACHE_DIR", str(entry["root"]))
    assert compiler._load_cute_compile_from_disk(
        _CK, _SIMPLE_PAYLOAD, _simple_func
    ) is None


def test_load_rejects_digest_mismatch(tmp_path, monkeypatch):
    entry = _make_secure_cache(tmp_path, _CK)
    entry["object"].unlink()
    entry["object"].write_bytes(b"evil")
    os.chmod(entry["object"], 0o600)
    monkeypatch.setenv("B12X_COMPILE_CACHE_DIR", str(entry["root"]))
    assert compiler._load_cute_compile_from_disk(
        _CK, _SIMPLE_PAYLOAD, _simple_func
    ) is None


def test_load_rejects_symlink_object(tmp_path, monkeypatch):
    entry = _make_secure_cache(tmp_path, _CK)
    evil = tmp_path / "evil.o"
    evil.write_bytes(b"attacker")
    entry["object"].unlink()
    entry["object"].symlink_to(evil)
    monkeypatch.setenv("B12X_COMPILE_CACHE_DIR", str(entry["root"]))
    assert compiler._load_cute_compile_from_disk(
        _CK, _SIMPLE_PAYLOAD, _simple_func
    ) is None


def test_load_rejects_hardlinked_object(tmp_path, monkeypatch):
    entry = _make_secure_cache(tmp_path, _CK)
    os.link(entry["object"], tmp_path / "hl.o")
    monkeypatch.setenv("B12X_COMPILE_CACHE_DIR", str(entry["root"]))
    assert compiler._load_cute_compile_from_disk(
        _CK, _SIMPLE_PAYLOAD, _simple_func
    ) is None


def test_load_rejects_hardlinked_manifest(tmp_path, monkeypatch):
    entry = _make_secure_cache(tmp_path, _CK)
    os.link(entry["manifest"], tmp_path / "hl.json")
    monkeypatch.setenv("B12X_COMPILE_CACHE_DIR", str(entry["root"]))
    assert compiler._load_cute_compile_from_disk(
        _CK, _SIMPLE_PAYLOAD, _simple_func
    ) is None


def test_load_rejects_world_writable_shard(tmp_path, monkeypatch):
    entry = _make_secure_cache(tmp_path, _CK)
    os.chmod(entry["shard"], 0o707)
    monkeypatch.setenv("B12X_COMPILE_CACHE_DIR", str(entry["root"]))
    assert compiler._load_cute_compile_from_disk(
        _CK, _SIMPLE_PAYLOAD, _simple_func
    ) is None


def test_load_rejects_wrong_semantic(tmp_path, monkeypatch):
    entry = _make_secure_cache(tmp_path, _CK)
    monkeypatch.setenv("B12X_COMPILE_CACHE_DIR", str(entry["root"]))
    assert compiler._load_cute_compile_from_disk(
        _CK, _SIMPLE_PAYLOAD + ("extra",), _simple_func
    ) is None


# -- Load: /dev/fd inode-bound, unlinked staging ---------------------------


def test_load_uses_dev_fd_path(tmp_path, monkeypatch):
    entry = _make_secure_cache(tmp_path, _CK)
    monkeypatch.setenv("B12X_COMPILE_CACHE_DIR", str(entry["root"]))
    paths: list[str] = []

    class FakeModule:
        def __init__(self, path):
            paths.append(path)
            with open(path, "rb") as f:
                assert f.read() == entry["bytes"]

        def __getattr__(self, name):
            return "loaded"

    _install_fake_external_binary_module(monkeypatch, FakeModule)
    result = compiler._load_cute_compile_from_disk(
        _CK, _SIMPLE_PAYLOAD, _simple_func
    )
    assert result == "loaded"
    assert paths[0].startswith("/dev/fd/")
    # Staging dir cleaned up.
    shard = entry["root"] / _CK[:2]
    assert not [d for d in shard.iterdir() if d.name.startswith("._stage_")]


def test_load_staging_unlinked_nlink_zero(tmp_path, monkeypatch):
    """The staged inode is unlinked; fstat must show nlink==0."""
    entry = _make_secure_cache(tmp_path, _CK)
    monkeypatch.setenv("B12X_COMPILE_CACHE_DIR", str(entry["root"]))
    nlinks: list[int] = []
    orig_fstat = os.fstat

    class FakeModule:
        def __init__(self, path):
            fd = int(path.split("/")[-1])
            nlinks.append(orig_fstat(fd).st_nlink)

        def __getattr__(self, name):
            return "loaded"

    _install_fake_external_binary_module(monkeypatch, FakeModule)
    result = compiler._load_cute_compile_from_disk(
        _CK, _SIMPLE_PAYLOAD, _simple_func
    )
    assert result == "loaded"
    assert nlinks == [0]


def test_load_swap_stage_dir_after_verify(tmp_path, monkeypatch):
    """After digest verification, swap the staging directory. The unlinked
    fd must still deliver the verified bytes — no name to swap."""
    entry = _make_secure_cache(tmp_path, _CK)
    monkeypatch.setenv("B12X_COMPILE_CACHE_DIR", str(entry["root"]))
    staged_bytes: list[bytes] = []

    class FakeModule:
        def __init__(self, path):
            with open(path, "rb") as f:
                staged_bytes.append(f.read())

        def __getattr__(self, name):
            return "loaded"

    _install_fake_external_binary_module(monkeypatch, FakeModule)
    result = compiler._load_cute_compile_from_disk(
        _CK, _SIMPLE_PAYLOAD, _simple_func
    )
    assert result == "loaded"
    assert staged_bytes == [entry["bytes"]]


# -- _store_cute_compile_to_disk: real publication with inode verify -------


class _FakeCompiled:
    def __init__(self, object_bytes: bytes):
        self._b = object_bytes

    def dump_to_object(self, prefix: str):
        return self._b


def test_store_publishes_with_inode_verify(tmp_path, monkeypatch):
    ck = _payload_cache_key(_SIMPLE_PAYLOAD)
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("B12X_COMPILE_CACHE_DIR", str(cache_root))
    obj_bytes = b"\x7fELFfake"
    compiler._store_cute_compile_to_disk(
        ck, _FakeCompiled(obj_bytes),
        cache_payload=_SIMPLE_PAYLOAD, func=_simple_func,
    )
    obj_path = cache_root / ck[:2] / f"{ck}.o"
    man_path = cache_root / ck[:2] / f"{ck}.json"
    assert obj_path.read_bytes() == obj_bytes
    assert (os.stat(obj_path).st_mode & 0o777) == 0o600
    assert (os.stat(man_path).st_mode & 0o777) == 0o600
    assert (os.stat(obj_path.parent).st_mode & 0o777) == 0o700


def test_store_then_load_roundtrip(tmp_path, monkeypatch):
    ck = _payload_cache_key(_SIMPLE_PAYLOAD)
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("B12X_COMPILE_CACHE_DIR", str(cache_root))
    obj_bytes = b"\x7fELFroundtrip"
    compiler._store_cute_compile_to_disk(
        ck, _FakeCompiled(obj_bytes),
        cache_payload=_SIMPLE_PAYLOAD, func=_simple_func,
    )
    staged: list[bytes] = []

    class FakeModule:
        def __init__(self, path):
            with open(path, "rb") as f:
                staged.append(f.read())

        def __getattr__(self, name):
            return "loaded"

    _install_fake_external_binary_module(monkeypatch, FakeModule)
    result = compiler._load_cute_compile_from_disk(
        ck, _SIMPLE_PAYLOAD, _simple_func
    )
    assert result == "loaded"
    assert staged == [obj_bytes]


def test_store_under_hostile_umask_still_0600(tmp_path, monkeypatch):
    ck = _payload_cache_key(_SIMPLE_PAYLOAD)
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("B12X_COMPILE_CACHE_DIR", str(cache_root))
    old = os.umask(0o000)
    try:
        compiler._store_cute_compile_to_disk(
            ck, _FakeCompiled(b"x"),
            cache_payload=_SIMPLE_PAYLOAD, func=_simple_func,
        )
    finally:
        os.umask(old)
    obj = cache_root / ck[:2] / f"{ck}.o"
    assert (os.stat(obj).st_mode & 0o777) == 0o600


def test_store_object_without_manifest_is_miss(tmp_path, monkeypatch):
    ck = _payload_cache_key(_SIMPLE_PAYLOAD)
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("B12X_COMPILE_CACHE_DIR", str(cache_root))
    compiler._store_cute_compile_to_disk(ck, _FakeCompiled(b"x"))
    man = cache_root / ck[:2] / f"{ck}.json"
    assert not man.exists()
    assert compiler._load_cute_compile_from_disk(ck) is None


def test_store_leaves_no_tmp(tmp_path, monkeypatch):
    ck = _payload_cache_key(_SIMPLE_PAYLOAD)
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("B12X_COMPILE_CACHE_DIR", str(cache_root))
    compiler._store_cute_compile_to_disk(ck, _FakeCompiled(b"x"))
    shard = cache_root / ck[:2]
    assert not list(shard.glob(".*.tmp"))


# -- _disk_cache_key_lock --------------------------------------------------


def test_lock_creates_secure(tmp_path, monkeypatch):
    entry = _make_secure_cache(tmp_path, _CK)
    monkeypatch.setenv("B12X_COMPILE_CACHE_DIR", str(entry["root"]))
    with compiler._disk_cache_key_lock(_CK) as fd:
        assert fd is not None
        lp = entry["shard"] / f"{_CK}.lock"
        assert (os.stat(lp).st_mode & 0o777) == 0o600


def test_lock_rejects_symlink(tmp_path, monkeypatch):
    entry = _make_secure_cache(tmp_path, _CK)
    lp = entry["shard"] / f"{_CK}.lock"
    rl = tmp_path / "r.lock"
    rl.write_text("")
    lp.symlink_to(rl)
    monkeypatch.setenv("B12X_COMPILE_CACHE_DIR", str(entry["root"]))
    with pytest.raises(OSError), compiler._disk_cache_key_lock(_CK):
        pass


def test_lock_rejects_hardlink(tmp_path, monkeypatch):
    entry = _make_secure_cache(tmp_path, _CK)
    lp = entry["shard"] / f"{_CK}.lock"
    lp.write_text("")
    os.chmod(lp, 0o600)
    os.link(lp, tmp_path / "hl.lock")
    monkeypatch.setenv("B12X_COMPILE_CACHE_DIR", str(entry["root"]))
    with pytest.raises(RuntimeError, match="hard links"), compiler._disk_cache_key_lock(_CK):
        pass


def test_lock_serializes_threads(tmp_path, monkeypatch):
    entry = _make_secure_cache(tmp_path, _CK)
    monkeypatch.setenv("B12X_COMPILE_CACHE_DIR", str(entry["root"]))
    held = threading.Event()
    done = threading.Event()
    order: list[str] = []

    def worker(name):
        with compiler._disk_cache_key_lock(_CK):
            order.append(f"{name}-enter")
            if name == "first":
                held.set()
                done.wait(timeout=5)
            order.append(f"{name}-exit")

    t1 = threading.Thread(target=worker, args=("first",))
    t2 = threading.Thread(target=worker, args=("second",))
    t1.start()
    held.wait(timeout=5)
    t2.start()
    import time
    time.sleep(0.1)
    done.set()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert order == ["first-enter", "first-exit", "second-enter", "second-exit"]


def test_cache_directory_creation_is_multiprocess_safe(tmp_path):
    root = tmp_path / "multiprocess-cache"
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    processes = [
        ctx.Process(target=_open_cache_root_worker, args=(str(root), queue))
        for _ in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0
    assert [queue.get(timeout=2) for _ in processes] == [None] * len(processes)


def test_cache_directory_creation_tolerates_mkdir_race(tmp_path, monkeypatch):
    """A peer creating the missing final component is a successful race."""
    root = tmp_path / "raced-cache"
    real_mkdir = compiler.os.mkdir
    injected = False

    def racing_mkdir(path, mode=0o777, *, dir_fd=None):
        nonlocal injected
        if not injected:
            injected = True
            real_mkdir(path, mode, dir_fd=dir_fd)
            raise FileExistsError(errno.EEXIST, "simulated peer mkdir", path)
        return real_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(compiler.os, "mkdir", racing_mkdir)
    fd = compiler._open_secure_dir_fd(root)
    try:
        st = os.fstat(fd)
        assert stat.S_ISDIR(st.st_mode)
        assert st.st_uid == os.geteuid()
        assert not st.st_mode & 0o077
    finally:
        os.close(fd)


def test_trellis_directory_creation_tolerates_mkdir_race(tmp_path, monkeypatch):
    """Concurrent JIT starters may observe FileExistsError after ENOENT."""
    small_m = importlib.import_module("b12x.gemm.trellis_linear._small_m")
    root = tmp_path / "raced-trellis"
    real_mkdir = small_m.os.mkdir
    injected = False

    def racing_mkdir(path, mode=0o777, *, dir_fd=None):
        nonlocal injected
        if not injected:
            injected = True
            real_mkdir(path, mode, dir_fd=dir_fd)
            raise FileExistsError(errno.EEXIST, "simulated peer mkdir", path)
        return real_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(small_m.os, "mkdir", racing_mkdir)
    assert small_m._validate_secure_build_dir(root) == str(root)


# -- Trellis build directory -----------------------------------------------


def test_trellis_build_dir_ignores_torch_extensions(tmp_path, monkeypatch):
    monkeypatch.delenv("B12X_TRELLIS_BUILD_DIR", raising=False)
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "evil"))
    monkeypatch.setenv("B12X_COMPILE_CACHE_DIR", str(tmp_path / "b12xcache"))
    small_m = importlib.import_module("b12x.gemm.trellis_linear._small_m")
    result = small_m._trellis_build_dir()
    assert "evil" not in result
    assert "trellis_build" in result


def test_trellis_build_dir_env_override(tmp_path, monkeypatch):
    d = tmp_path / "custom"
    monkeypatch.setenv("B12X_TRELLIS_BUILD_DIR", str(d))
    small_m = importlib.import_module("b12x.gemm.trellis_linear._small_m")
    result = small_m._trellis_build_dir()
    assert result == str(d.resolve())
    assert (os.stat(result).st_mode & 0o777) == 0o700


def test_trellis_build_dir_rejects_symlink(tmp_path, monkeypatch):
    real = tmp_path / "r"
    real.mkdir(mode=0o700)
    link = tmp_path / "l"
    link.symlink_to(real)
    monkeypatch.setenv("B12X_TRELLIS_BUILD_DIR", str(link))
    small_m = importlib.import_module("b12x.gemm.trellis_linear._small_m")
    with pytest.raises(OSError):
        small_m._trellis_build_dir()


def test_trellis_build_dir_rejects_world_writable(tmp_path, monkeypatch):
    d = tmp_path / "ww"
    d.mkdir()
    os.chmod(d, 0o707)
    monkeypatch.setenv("B12X_TRELLIS_BUILD_DIR", str(d))
    small_m = importlib.import_module("b12x.gemm.trellis_linear._small_m")
    with pytest.raises(RuntimeError, match="group/other"):
        small_m._trellis_build_dir()
    os.chmod(d, 0o755)


def test_trellis_build_dir_returns_normalized(tmp_path, monkeypatch):
    d = tmp_path / "real"
    d.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(d)
    monkeypatch.setenv("B12X_TRELLIS_BUILD_DIR", str(tmp_path / "link" / ".." / "real"))
    small_m = importlib.import_module("b12x.gemm.trellis_linear._small_m")
    result = small_m._trellis_build_dir()
    assert ".." not in result


def test_trellis_validate_secure_build_dir_accepts_private(tmp_path):
    small_m = importlib.import_module("b12x.gemm.trellis_linear._small_m")
    d = tmp_path / "good"
    result = small_m._validate_secure_build_dir(d)
    assert result == str(d.resolve())
    assert (os.stat(result).st_mode & 0o777) == 0o700


def test_trellis_extension_uses_build_only_and_fd_import(tmp_path, monkeypatch):
    """_extension must build without executing the .so, validate it, and
    import via /dev/fd — not via torch.load which executes before return."""
    monkeypatch.delenv("B12X_TRELLIS_BUILD_DIR", raising=False)
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "evil"))
    monkeypatch.setenv("B12X_COMPILE_CACHE_DIR", str(tmp_path / "b12xcache"))
    small_m = importlib.import_module("b12x.gemm.trellis_linear._small_m")
    small_m._extension.cache_clear()

    build_called = {"n": 0}
    extension_load = {"n": 0, "names": [], "fds": []}

    def fake_build(**kwargs):
        build_called["n"] += 1
        bd = kwargs.get("build_directory", "")
        name = kwargs.get("name", "ext")
        so_path = Path(bd) / f"{name}.so"
        so_path.write_bytes(b"\x7fELFfake_so")
        os.chmod(so_path, 0o755)  # Simulate normal linker output

    monkeypatch.setattr(
        "torch.utils.cpp_extension._write_ninja_file_and_build_library",
        fake_build,
    )
    monkeypatch.setattr(
        "torch.utils.cpp_extension._is_cuda_file",
        lambda s: True,
    )

    fake_module = SimpleNamespace(launch_k6_mcg=object())

    def fake_load_extension(name, fd):
        extension_load["n"] += 1
        extension_load["names"].append(name)
        extension_load["fds"].append(fd)
        assert not os.get_inheritable(fd)
        return fake_module

    monkeypatch.setattr(small_m, "_load_python_extension_from_fd", fake_load_extension)

    try:
        assert small_m._extension() is fake_module
        # Clearing the process-local memo simulates a fresh process. The
        # content-addressed artifact is loaded without invoking Ninja again.
        small_m._extension.cache_clear()
        assert small_m._extension() is fake_module
        assert build_called["n"] == 1
        assert extension_load["n"] == 2
        assert extension_load["names"] == [small_m._extension_name()] * 2
        assert len(extension_load["fds"]) == 2
        # The .so was fchmod'd to 0600 after build
        bd = Path(small_m._trellis_build_dir()) / small_m._extension_name()
        so_files = list(bd.glob("*.so"))
        assert len(so_files) == 1
        assert (os.stat(so_files[0]).st_mode & 0o777) == 0o600
        assert (bd / ".b12x-build.lock").is_file()
        assert (bd / ".b12x-build-complete").read_text() == f"{so_files[0].name}\n"
    finally:
        small_m._extension.cache_clear()


def test_trellis_extension_concurrent_starters_share_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("B12X_TRELLIS_BUILD_DIR", str(tmp_path / "trellis"))
    small_m = importlib.import_module("b12x.gemm.trellis_linear._small_m")
    small_m._extension.cache_clear()

    build_started = threading.Event()
    release_build = threading.Event()
    build_count = 0

    def fake_build(**kwargs):
        nonlocal build_count
        build_count += 1
        build_started.set()
        assert release_build.wait(timeout=5)
        build_dir = Path(kwargs["build_directory"])
        (build_dir / f"{kwargs['name']}.so").write_bytes(b"\x7fELFfake_so")

    monkeypatch.setattr(
        "torch.utils.cpp_extension._write_ninja_file_and_build_library",
        fake_build,
    )
    monkeypatch.setattr("torch.utils.cpp_extension._is_cuda_file", lambda _s: True)
    fake_module = SimpleNamespace(launch_k6_mcg=object())
    monkeypatch.setattr(
        small_m,
        "_load_python_extension_from_fd",
        lambda _name, _fd: fake_module,
    )

    results = []
    errors = []

    def load_extension():
        try:
            results.append(small_m._extension())
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=load_extension) for _ in range(2)]
    try:
        for thread in threads:
            thread.start()
        assert build_started.wait(timeout=5)
        release_build.set()
        for thread in threads:
            thread.join(timeout=10)
        assert not [thread for thread in threads if thread.is_alive()]
        assert errors == []
        assert results == [fake_module, fake_module]
        assert build_count == 1
    finally:
        release_build.set()
        small_m._extension.cache_clear()


def test_trellis_extension_rebuilds_interrupted_generation(tmp_path, monkeypatch):
    monkeypatch.setenv("B12X_TRELLIS_BUILD_DIR", str(tmp_path / "trellis"))
    small_m = importlib.import_module("b12x.gemm.trellis_linear._small_m")
    small_m._extension.cache_clear()

    build_count = 0

    def fake_build(**kwargs):
        nonlocal build_count
        build_count += 1
        so_path = Path(kwargs["build_directory"]) / f"{kwargs['name']}.so"
        so_path.write_bytes(b"\x7fELFpartial")
        if build_count == 1:
            raise RuntimeError("interrupted build")
        so_path.write_bytes(b"\x7fELFcomplete")

    monkeypatch.setattr(
        "torch.utils.cpp_extension._write_ninja_file_and_build_library",
        fake_build,
    )
    monkeypatch.setattr("torch.utils.cpp_extension._is_cuda_file", lambda _s: True)
    fake_module = SimpleNamespace(launch_k6_mcg=object())
    monkeypatch.setattr(
        small_m,
        "_load_python_extension_from_fd",
        lambda _name, _fd: fake_module,
    )

    try:
        with pytest.raises(RuntimeError, match="interrupted build"):
            small_m._extension()
        build_dir = Path(small_m._trellis_build_dir()) / small_m._extension_name()
        assert not (build_dir / ".b12x-build-complete").exists()

        small_m._extension.cache_clear()
        assert small_m._extension() is fake_module
        assert build_count == 2
        assert (build_dir / ".b12x-build-complete").is_file()
    finally:
        small_m._extension.cache_clear()


def test_trellis_extension_rejects_invalid_cached_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("B12X_TRELLIS_BUILD_DIR", str(tmp_path / "trellis"))
    small_m = importlib.import_module("b12x.gemm.trellis_linear._small_m")
    small_m._extension.cache_clear()

    ext_name = small_m._extension_name()
    build_dir = Path(
        small_m._validate_secure_build_dir(
            Path(small_m._trellis_build_dir()) / ext_name
        )
    )
    outside = tmp_path / "untrusted.so"
    outside.write_bytes(b"\x7fELFevil")
    (build_dir / f"{ext_name}.so").symlink_to(outside)

    monkeypatch.setattr(
        "torch.utils.cpp_extension._write_ninja_file_and_build_library",
        lambda **_kwargs: pytest.fail("invalid artifacts must not be rebuilt"),
    )
    try:
        with pytest.raises(OSError):
            small_m._extension()
    finally:
        small_m._extension.cache_clear()
