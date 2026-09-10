"""Compile-cache rooting: the highest-risk mechanical edit of the restructure.

Wrong fingerprint root ⇒ silent stale disk-cache hits (running old kernels),
so these assertions are load-bearing.
"""

from __future__ import annotations

import importlib
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


def test_fused_oneshot_capacity_variants_reuse_launchers_for_live_rows(
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    oneshot = importlib.import_module("b12x.comm.pcie._oneshot_cute")
    pcie = importlib.import_module("b12x.comm.pcie.pcie_oneshot")
    runtime_control = importlib.import_module("b12x._lib.runtime_control")

    monkeypatch.delenv("B12X_PCIE_ONESHOT_PUSH", raising=False)
    monkeypatch.setenv("B12X_PCIE_TP8_OWNER_REDUCE", "0")
    state = pcie._CuTeOneshotState(
        rank=0,
        world_size=8,
        signal_ptrs=tuple(range(8)),
        rank_data=torch.empty(256, dtype=torch.uint8),
        signal_table_address=0,
        next_table_offset=128,
        registered_tables={},
        eager_tables=(300, 400),
        eager_buffer_bytes=256 * 1024,
        transport_policy=(False, False, False, False, False),
        scatter_gather_storage=True,
    )
    plans = [
        pcie._CuTeOneshotBackend._fused_launch_plan(
            state,
            torch.empty((rows, 6144), dtype=torch.bfloat16),
        )
        for rows in (3, 4, 5, 6, 8, 12, 16)
    ]

    compile_specs = []

    def compile_stub(*_args, **kwargs):
        compile_specs.append(kwargs["compile_spec"])
        return lambda *_args: None

    monkeypatch.setattr(oneshot, "b12x_compile", compile_stub)
    monkeypatch.setattr(oneshot, "current_cuda_stream", lambda: 0)
    monkeypatch.setattr(oneshot, "make_ptr", lambda *_args, **_kwargs: object())
    oneshot.get_fused_oneshot_launcher.cache_clear()
    oneshot._PREPARED_FUSED_ONESHOT_LAUNCHERS.clear()

    def resolve(plan):
        variant = plan.variant
        return oneshot.get_fused_oneshot_launcher(
            "bfloat16",
            8,
            0,
            variant.mode,
            variant.single_cta,
            variant.register_normalize,
            False,
            0,
            variant.threads,
            variant.reg_packs,
            0,
        )

    try:
        warmed = [resolve(plan) for plan in plans]
        assert len(compile_specs) == 2

        runtime_control.freeze_kernel_resolution(
            "fused one-shot live rows must reuse capacity launchers"
        )
        try:
            reused = [resolve(plan) for plan in plans]
        finally:
            runtime_control.unfreeze_kernel_resolution()

        assert all(
            actual is expected for actual, expected in zip(reused, warmed, strict=True)
        )
        assert len(compile_specs) == 2
    finally:
        oneshot.get_fused_oneshot_launcher.cache_clear()
        oneshot._PREPARED_FUSED_ONESHOT_LAUNCHERS.clear()


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
