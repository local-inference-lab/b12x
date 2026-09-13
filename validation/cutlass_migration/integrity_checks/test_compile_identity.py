"""Offline checks for raw cache integrity and independent comparison identity."""

from copy import deepcopy
import hashlib

import pytest

from b12x._lib import compiler
from validation.cutlass_migration.acceptance.corpus import ptx_capture
from validation.cutlass_migration.core.comparison_identity import (
    comparison_semantic_key_from_manifest,
)


@pytest.fixture
def manifest_factory(monkeypatch):
    monkeypatch.setattr(compiler, "_current_device_ordinal", lambda: 0)
    monkeypatch.setattr(compiler, "_device_uuid_key", lambda ordinal: ("device_uuid", "test-gpu"))

    def build(dump_dir):
        environment = (("CUTE_DSL_ARCH", "sm_103a"), ("CUTE_DSL_DUMP_DIR", dump_dir),
                       ("CUTE_DSL_KEEP", "ptx"))
        monkeypatch.setattr(
            compiler, "_static_compile_cache_context",
            lambda func: ("a" * 64, (("python", "cpython", (3, 12, 0)),), (), environment),
        )
        spec = compiler.KernelCompileSpec.from_facts("test.identity", 1, ("capacity", 8))
        payload = compiler._compile_disk_cache_payload(object(), build, (), {}, spec)
        key = hashlib.sha256(repr(payload).encode()).hexdigest()
        object_bytes = b"manifest-bound test object"
        manifest = compiler._build_compile_manifest(key, payload, build, object_bytes)
        return manifest, object_bytes

    return build


@pytest.mark.parametrize("schema", ["b12x._lib.compile_manifest.v3", "b12x.cute.compile_manifest.v3"])
def test_manifest_validator_accepts_exact_current_and_historical_schemas(manifest_factory, schema):
    manifest, object_bytes = manifest_factory("/tmp/identity-a")
    manifest["schema"] = schema
    assert ptx_capture._validate_compile_manifest(
        manifest, cache_key=manifest["cache_key"], object_bytes=object_bytes
    ) == manifest


@pytest.mark.parametrize("field", ["semantic_key", "object_sha256", "package_fingerprint", "artifact_evidence_sha256"])
def test_manifest_validator_rejects_tampered_identity(manifest_factory, field):
    manifest, object_bytes = manifest_factory("/tmp/identity-a")
    manifest[field] = "0" * 64
    with pytest.raises(RuntimeError):
        ptx_capture._validate_compile_manifest(
            manifest, cache_key=manifest["cache_key"], object_bytes=object_bytes
        )


def test_operational_normalization_preserves_raw_manifest_identity(manifest_factory):
    first, object_bytes = manifest_factory("/tmp/identity-a")
    second, _ = manifest_factory("/tmp/identity-b")
    originals = deepcopy((first, second))
    assert first["cache_key"] != second["cache_key"]
    assert first["semantic_key"] != second["semantic_key"]
    assert comparison_semantic_key_from_manifest(first) == comparison_semantic_key_from_manifest(second)
    assert (first, second) == originals
    for manifest in (first, second):
        ptx_capture._validate_compile_manifest(
            manifest, cache_key=manifest["cache_key"], object_bytes=object_bytes
        )
    with pytest.raises(RuntimeError, match="object SHA-256"):
        ptx_capture._validate_compile_manifest(
            first, cache_key=first["cache_key"], object_bytes=object_bytes + b"tampered"
        )


def test_capture_uses_current_compiler_without_replacing_identity_function(monkeypatch, tmp_path):
    monkeypatch.setenv("CORPUS_RETAIN_FRONTEND_PTX", "1")
    monkeypatch.setenv("CUTE_DSL_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("CUTE_DSL_KEEP", "cubin")
    monkeypatch.setattr(ptx_capture, "_INSTALLED", False)
    monkeypatch.setattr(ptx_capture, "_INSTALLATION_EVIDENCE", {})
    monkeypatch.setattr(compiler, "compile_cache_info", lambda: {})
    # Register the hook target with monkeypatch so teardown restores it.
    monkeypatch.setattr(compiler, "_store_cute_compile_to_disk", compiler._store_cute_compile_to_disk)
    environment_key = compiler._compile_environment_key
    try:
        ptx_capture.install()
        assert ptx_capture._INSTALLED
        assert compiler._compile_environment_key is environment_key
        environment = dict(environment_key())
        assert environment["CUTE_DSL_DUMP_DIR"] == str(tmp_path)
        assert environment["CUTE_DSL_KEEP"] == "cubin,ptx"
        assert ptx_capture._INSTALLATION_EVIDENCE["hook_target"] == (
            "b12x._lib.compiler._store_cute_compile_to_disk"
        )
    finally:
        environment_key.cache_clear()
        compiler._static_compile_cache_context.cache_clear()
        compiler._DEVICE_COMPILE_CACHE_CONTEXTS.clear()
