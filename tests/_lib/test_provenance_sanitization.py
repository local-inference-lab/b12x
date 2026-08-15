"""Focused sentinel-secret tests for provenance sanitization (issue #177).

Every collected prefix is exercised with a sentinel secret variable to verify
that raw values never appear in generated artifacts or logs.  Known-safe
architecture/tuning values are confirmed to remain readable, and behavior is
verified to be consistent across all producers that share the helper.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.parse
from pathlib import Path

import pytest

from b12x._lib.provenance import (
    COLLECTED_PREFIXES,
    DIGEST_ALGORITHM,
    DIGEST_DOMAIN,
    REDACTED_MARKER,
    REDACTED_REASON,
    comparison_safe_cute_dsl_libs,
    is_secret_name,
    safe_env_string,
    safe_env_tuple,
    sanitize_environment_map,
    sanitize_environment_tuple,
    sanitize_value,
)


_SENTINEL = "SENTINEL/secret+raw=="

_SENTINEL_SECRETS: dict[str, str] = {
    f"{prefix}AUTH_TOKEN": _SENTINEL
    for prefix in COLLECTED_PREFIXES
}

_SECRET_NAME_VARIANTS = [
    "B12X_API_KEY", "CUDA_PASSWORD", "TORCH_SECRET", "NCCL_CREDENTIAL",
    "TRITON_COOKIE", "B12X_PASSWD", "CUTLASS_PWD", "PYTORCH_APIKEY",
    "NVCC_ACCESS_KEY", "PTXAS_PRIVATE_KEY", "B12X_CERTIFICATE",
    "B12X_PASS_WORD", "B12X_PASS-PHRASE", "B12X_BEARER",
    "B12X_SESSION_ID", "B12X_JWT", "B12X_CRED_ENTIAL",
    "B12X_CRED-ENTIAL", "B12X.password", "B12X-AUTH-TOKEN",
    "B12X_PASS/WORD", "B12X_PASS WORD",
]

_SAFE_VARS: dict[str, str] = {
    "CUDA_VISIBLE_DEVICES": "0,1",
    "CUDA_DEVICE_ORDER": "FAST_FIRST",
    "CUDA_LAUNCH_BLOCKING": "1",
    "CUDA_DEVICE_MAX_CONNECTIONS": "8",
    "CUDA_CACHE_MAXSIZE": "262144",
    "B12X_TIMING": "1",
    "B12X_PCIE_DMA_FP8": "i8_ring",
    "B12X_COMPILE_MEMORY_CACHE_SIZE": "1024",
    "NCCL_P2P_DISABLE": "1",
    "CUTE_DSL_ARCH": "sm_120",
    "OMP_NUM_THREADS": "4",
    "NVIDIA_TF32_OVERRIDE": "1",
}

_PATH_VARS: dict[str, str] = {
    "CUDA_CACHE_PATH": "/tmp/cache" + _SENTINEL,
    "TORCH_EXTENSIONS_DIR": "/tmp/ext" + _SENTINEL,
    "CUDA_HOME": "/usr/local/cuda",
    "CUDA_PATH": "/opt/cuda",
    "CC": "gcc-12",
    "CXX": "g++-12",
    "CUDACXX": "/usr/local/cuda/bin/nvcc",
    "NVCC_APPEND_FLAGS": "-DMY_TOKEN=" + _SENTINEL,
    "NVCC_PREPEND_FLAGS": "-O2",
    "B12X_COMPILE_CACHE_DIR": "/tmp/b12x-cache",
    "CUTE_DSL_CACHE_DIR": "/tmp/cute-cache",
    "CUTE_DSL_LIBS": "/usr/lib/libcute.so:" + "/tmp/lib" + _SENTINEL,
}

_UNKNOWN_VARS: dict[str, str] = {
    "B12X_SOME_UNKNOWN_KNOB": "unknown_value_42",
    "CUDA_OBSCURE_FLAG": "xyz",
}

_UNBOUNDED_VARS: dict[str, str] = {
    "OMP_NUM_THREADS": "9" * 1000,
    "CUTE_DSL_ARCH": "sm_" + "9" * 100,
    "CUDA_DEVICE_MAX_CONNECTIONS": "999999999999",
}


def _scan_for_sentinel(text: str, sentinel: str = _SENTINEL) -> list[str]:
    hits: list[str] = []
    if sentinel in text:
        hits.append("raw")
    b64 = base64.b64encode(sentinel.encode()).decode()
    if b64 in text:
        hits.append("base64")
    urlenc = urllib.parse.quote(sentinel)
    if urlenc in text:
        hits.append("url-encoded")
    urlenc_safe = urllib.parse.quote(sentinel, safe="")
    if urlenc_safe in text:
        hits.append("url-safe-encoded")
    substr = sentinel[:10]
    if len(substr) >= 5 and substr in text:
        hits.append("substring")
    return hits


def _assert_no_sentinel(text: str, sentinel: str = _SENTINEL) -> None:
    hits = _scan_for_sentinel(text, sentinel)
    assert not hits, f"sentinel found in output as {hits}"


class TestIsSecretName:
    @pytest.mark.parametrize("name", list(_SENTINEL_SECRETS.keys()))
    def test_sentinel_secrets_under_every_prefix_detected(self, name: str) -> None:
        assert is_secret_name(name)

    @pytest.mark.parametrize("name", _SECRET_NAME_VARIANTS)
    def test_secret_name_variants_with_separators_detected(self, name: str) -> None:
        assert is_secret_name(name), f"{name} should be detected as secret-like"

    @pytest.mark.parametrize("name", list(_SAFE_VARS.keys()))
    def test_safe_names_not_secret(self, name: str) -> None:
        assert not is_secret_name(name)


class TestSanitizeValue:
    @pytest.mark.parametrize("name", list(_SENTINEL_SECRETS.keys()))
    def test_sentinel_secret_redacted(self, name: str) -> None:
        result = sanitize_value(name, _SENTINEL)
        assert result["status"] == "redacted-set"
        assert result["reason"] == REDACTED_REASON
        assert "value" not in result
        _assert_no_sentinel(json.dumps(result))

    @pytest.mark.parametrize("name,value", list(_SAFE_VARS.items()))
    def test_safe_value_canonicalized(self, name: str, value: str) -> None:
        result = sanitize_value(name, value)
        assert result["status"] == "set-safe"
        assert isinstance(result["value"], str)

    @pytest.mark.parametrize("name,value", list(_PATH_VARS.items()))
    def test_path_value_digested(self, name: str, value: str) -> None:
        result = sanitize_value(name, value)
        assert result["status"] == "set-digest"
        assert "value" not in result
        _assert_no_sentinel(json.dumps(result))

    @pytest.mark.parametrize("name,value", list(_UNKNOWN_VARS.items()))
    def test_unknown_value_digested(self, name: str, value: str) -> None:
        result = sanitize_value(name, value)
        assert result["status"] == "set-digest"
        assert result["digest"]["algorithm"] == DIGEST_ALGORITHM
        assert result["digest"]["domain"] == DIGEST_DOMAIN

    @pytest.mark.parametrize("name,value", list(_UNBOUNDED_VARS.items()))
    def test_unbounded_values_digested(self, name: str, value: str) -> None:
        result = sanitize_value(name, value)
        assert result["status"] == "set-digest"

    def test_secret_constant_not_hash(self) -> None:
        r1 = sanitize_value("B12X_AUTH_TOKEN", "secret_one")
        r2 = sanitize_value("B12X_AUTH_TOKEN", "secret_two")
        assert r1 == r2

    def test_bool_canonicalized(self) -> None:
        assert sanitize_value("B12X_TIMING", "TRUE")["value"] == "1"
        assert sanitize_value("B12X_TIMING", "off")["value"] == "0"
        assert sanitize_value("B12X_TIMING", "yes")["value"] == "1"

    def test_int_canonicalized(self) -> None:
        assert sanitize_value("OMP_NUM_THREADS", "04")["value"] == "4"

    def test_device_list_canonicalized(self) -> None:
        assert sanitize_value("CUDA_VISIBLE_DEVICES", "0, 1")["value"] == "0,1"

    def test_invalid_bool_digested(self) -> None:
        assert sanitize_value("B12X_TIMING", "maybe")["status"] == "set-digest"

    def test_empty_device_list_digested(self) -> None:
        assert sanitize_value("CUDA_VISIBLE_DEVICES", "")["status"] == "set-digest"

    def test_set_empty_canonicalizable(self) -> None:
        result = sanitize_value("B12X_TIMING", "")
        assert result["status"] == "set-safe"
        assert result["value"] == "0"


class TestSafeEnvString:
    @pytest.mark.parametrize("name", list(_SENTINEL_SECRETS.keys()))
    def test_sentinel_secret_constant_marker(self, name: str) -> None:
        assert safe_env_string(name, _SENTINEL) == REDACTED_MARKER
        _assert_no_sentinel(safe_env_string(name, _SENTINEL))

    @pytest.mark.parametrize("name,value", list(_SAFE_VARS.items()))
    def test_safe_value_canonical(self, name: str, value: str) -> None:
        result = safe_env_string(name, value)
        assert isinstance(result, str)
        assert result != ""

    def test_path_value_versioned_digest(self) -> None:
        result = safe_env_string("CUDA_HOME", "/opt/cuda")
        assert result.startswith("v1:")
        assert len(result) == 3 + 64

    def test_secret_constant_same_for_different_values(self) -> None:
        assert safe_env_string("B12X_AUTH_TOKEN", "a") == safe_env_string("B12X_AUTH_TOKEN", "b")

    def test_tuple_remains_hashable(self) -> None:
        env = tuple(_SENTINEL_SECRETS.items()) + tuple(_SAFE_VARS.items())
        sanitized = safe_env_tuple(env)
        hash(sanitized)
        assert all(isinstance(v, str) for _, v in sanitized)


class TestSanitizeEnvironmentMap:
    def test_sentinel_secrets_never_appear(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name, value in _SENTINEL_SECRETS.items():
            monkeypatch.setenv(name, value)
        for name, value in _SAFE_VARS.items():
            monkeypatch.setenv(name, value)
        result = sanitize_environment_map()
        _assert_no_sentinel(json.dumps(result, sort_keys=True))
        for name in _SENTINEL_SECRETS:
            assert result[name]["status"] == "redacted-set"

    def test_safe_values_readable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name, value in _SAFE_VARS.items():
            monkeypatch.setenv(name, value)
        result = sanitize_environment_map(
            explicit_names=frozenset(_SAFE_VARS.keys()),
        )
        for name in _SAFE_VARS:
            assert result[name]["status"] == "set-safe"

    def test_path_values_digested(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name, value in _PATH_VARS.items():
            monkeypatch.setenv(name, value)
        result = sanitize_environment_map(
            explicit_names=frozenset(_PATH_VARS.keys()),
        )
        _assert_no_sentinel(json.dumps(result, sort_keys=True))
        for name in _PATH_VARS:
            assert result[name]["status"] == "set-digest"

    def test_only_prefix_matching_vars_collected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RANDOM_UNRELATED_VAR", "foo")
        result = sanitize_environment_map()
        assert "RANDOM_UNRELATED_VAR" not in result

    def test_all_entries_are_tagged_dicts(self) -> None:
        env = {"B12X_TIMING": "1", "CUDA_HOME": "/opt/cuda", "B12X_AUTH_TOKEN": "secret"}
        result = sanitize_environment_map(env)
        for name, entry in result.items():
            assert isinstance(entry, dict), f"{name} must be a tagged dict"
            assert "status" in entry


class TestSanitizeEnvironmentTuple:
    def test_sentinel_secrets_redacted_in_tuple(self) -> None:
        env = tuple(_SENTINEL_SECRETS.items())
        sanitized = sanitize_environment_tuple(env)
        _assert_no_sentinel(json.dumps(dict(sanitized), sort_keys=True))


class TestCrossProducerConsistency:
    def test_same_env_same_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        all_vars = {**_SENTINEL_SECRETS, **_SAFE_VARS, **_UNKNOWN_VARS, **_PATH_VARS}
        for name, value in all_vars.items():
            monkeypatch.setenv(name, value)
        result_a = sanitize_environment_map(prefixes=COLLECTED_PREFIXES)
        result_b = sanitize_environment_map(prefixes=COLLECTED_PREFIXES)
        assert result_a == result_b
        for name in _SENTINEL_SECRETS:
            assert result_a[name]["status"] == "redacted-set"


class TestCompilerEnvSanitization:
    def test_compile_environment_key_redacts_secrets(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("B12X_AUTH_TOKEN", _SENTINEL)
        monkeypatch.setenv("CUTE_SECRET_KEY", _SENTINEL)
        monkeypatch.setenv("CUTLASS_PASSWORD", _SENTINEL)
        monkeypatch.setenv("CUTE_DSL_ARCH", "sm_120")
        monkeypatch.setenv("NVCC_APPEND_FLAGS", "-DMY_TOKEN=" + _SENTINEL)
        monkeypatch.setenv("CUDA_HOME", "/opt/cuda")
        from b12x._lib.compiler import _compile_environment_key
        _compile_environment_key.cache_clear()
        env_key = _compile_environment_key()
        _compile_environment_key.cache_clear()
        env_dict = dict(env_key)
        _assert_no_sentinel(json.dumps(env_dict, sort_keys=True))
        assert env_dict["B12X_AUTH_TOKEN"] == REDACTED_MARKER
        assert env_dict["CUTE_SECRET_KEY"] == REDACTED_MARKER
        assert env_dict["CUTLASS_PASSWORD"] == REDACTED_MARKER
        assert env_dict["NVCC_APPEND_FLAGS"].startswith("v1:")
        assert env_dict["CUDA_HOME"].startswith("v1:")
        assert env_dict["CUTE_DSL_ARCH"] == "sm_120"

    def test_unset_distinct_from_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CUTE_DSL_ARCH", "sm_120")
        from b12x._lib.compiler import _compile_environment_key
        _compile_environment_key.cache_clear()
        env_key = _compile_environment_key()
        _compile_environment_key.cache_clear()
        env_dict = dict(env_key)
        assert "CC" not in env_dict

    def test_set_empty_distinct_from_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("B12X_DENSE_SPLITK_TURBO", "")
        monkeypatch.setenv("CUTE_DSL_ARCH", "sm_120")
        from b12x._lib.compiler import _compile_environment_key
        _compile_environment_key.cache_clear()
        env_key = _compile_environment_key()
        _compile_environment_key.cache_clear()
        env_dict = dict(env_key)
        assert "B12X_DENSE_SPLITK_TURBO" in env_dict
        assert env_dict["B12X_DENSE_SPLITK_TURBO"] == "0"
        assert "CC" not in env_dict

    def test_cute_dsl_libs_production_vs_comparison(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        package_runtime = (
            "/tmp/site-packages/nvidia_cutlass_dsl/cu13/lib/"
            "libcute_dsl_runtime.so"
        )
        custom_runtime = "/opt/b12x-custom/libserving_runtime.so"
        monkeypatch.setenv(
            "CUTE_DSL_LIBS",
            package_runtime + os.pathsep + custom_runtime,
        )
        from b12x._lib.compiler import (
            _compile_environment_key,
            _comparison_compile_environment,
        )
        _compile_environment_key.cache_clear()
        env_key = _compile_environment_key()
        _compile_environment_key.cache_clear()
        env_dict = dict(env_key)
        assert env_dict["CUTE_DSL_LIBS"].startswith("v1:")
        comp_env = _comparison_compile_environment(env_key)
        comp_dict = dict(comp_env)
        assert comp_dict["CUTE_DSL_LIBS"].startswith("v1:")
        assert env_dict["CUTE_DSL_LIBS"] != comp_dict["CUTE_DSL_LIBS"]

    def test_production_identity_preserves_order(self) -> None:
        lib_a = "/opt/lib_a.so"
        lib_b = "/opt/lib_b.so"
        digest_ab = safe_env_string("CUTE_DSL_LIBS", lib_a + os.pathsep + lib_b)
        digest_ba = safe_env_string("CUTE_DSL_LIBS", lib_b + os.pathsep + lib_a)
        assert digest_ab != digest_ba

    def test_compile_environment_key_hashable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("B12X_AUTH_TOKEN", _SENTINEL)
        monkeypatch.setenv("CUTE_DSL_ARCH", "sm_120")
        from b12x._lib.compiler import _compile_environment_key
        _compile_environment_key.cache_clear()
        env_key = _compile_environment_key()
        _compile_environment_key.cache_clear()
        hash(env_key)

    def test_compiler_manifest_scan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("B12X_AUTH_TOKEN", _SENTINEL)
        monkeypatch.setenv("NVCC_APPEND_FLAGS", "-DTOKEN=" + _SENTINEL)
        monkeypatch.setenv("CUTE_DSL_ARCH", "sm_120")
        monkeypatch.setenv("CUDA_HOME", "/opt/cuda")
        from b12x._lib.compiler import (
            _compile_environment_key,
            _comparison_compile_environment,
        )
        _compile_environment_key.cache_clear()
        env_key = _compile_environment_key()
        _compile_environment_key.cache_clear()
        manifest = {
            "compile_environment": [[n, v] for n, v in env_key],
            "comparison_compile_environment": _comparison_compile_environment(env_key),
            "cache_payload": ["v6", ("target",), "fp", ("toolchain",),
                              None, "spec_hash", "spec_json",
                              "kw_hash", "kw_json", ("opts",),
                              [[n, v] for n, v in env_key]],
            "cache_payload_repr": repr(env_key),
        }
        _assert_no_sentinel(json.dumps(manifest, sort_keys=True))

    def test_compiler_log_value_includes_empty(self) -> None:
        from b12x._lib.compiler import _environment_log_value
        env = (("B12X_TIMING", "0"), ("CUTE_DSL_ARCH", "sm_120"))
        result = _environment_log_value(env)
        assert "B12X_TIMING" in result
        assert result["B12X_TIMING"] == "0"


class TestValidatorAcceptance:
    def _build_v2_payload(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
        from validation.cutlass_migration.diagnostics.graph_replay_abba import (
            _RUNTIME_ENVIRONMENT_PREFIXES,
            _RUNTIME_ENVIRONMENT_EXPLICIT_CONTROLS,
            _json_sha256,
        )
        monkeypatch.setenv("B12X_AUTH_TOKEN", _SENTINEL)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
        monkeypatch.setenv("B12X_TIMING", "1")
        monkeypatch.setenv("CUDA_CACHE_PATH", "/tmp/" + _SENTINEL)
        set_vars = sanitize_environment_map(prefixes=_RUNTIME_ENVIRONMENT_PREFIXES)
        explicit: dict[str, object] = {}
        for name in _RUNTIME_ENVIRONMENT_EXPLICIT_CONTROLS:
            if name not in os.environ:
                explicit[name] = {"status": "unset"}
            else:
                explicit[name] = sanitize_value(name, os.environ[name])
        payload: dict[str, object] = {
            "schema": "b12x-runtime-environment-v2",
            "complete_set_variable_prefixes": list(_RUNTIME_ENVIRONMENT_PREFIXES),
            "set_variables": set_vars,
            "explicit_controls": explicit,
            "nvidia_enumeration": {
                "policy": "explicit-only",
                "included": [
                    "NVIDIA_VISIBLE_DEVICES",
                    "NVIDIA_DRIVER_CAPABILITIES",
                    "NVIDIA_TF32_OVERRIDE",
                ],
                "reason": "avoid collecting unrelated NVIDIA_ variables that may contain secrets",
            },
        }
        payload["sha256"] = _json_sha256(payload)
        return payload

    def test_validator_accepts_v2(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from validation.cutlass_migration.diagnostics.graph_replay_abba import (
            _validate_runtime_environment,
        )
        payload = self._build_v2_payload(monkeypatch)
        _validate_runtime_environment(tmp_path / "fake.jsonl", {"runtime_environment": payload})

    def test_validator_rejects_bare_string(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from validation.cutlass_migration.diagnostics.graph_replay_abba import (
            _validate_runtime_environment, _json_sha256,
        )
        payload = self._build_v2_payload(monkeypatch)
        payload["set_variables"]["B12X_TIMING"] = "1"
        del payload["sha256"]
        payload["sha256"] = _json_sha256(payload)
        with pytest.raises(ValueError, match="set-variable map is invalid"):
            _validate_runtime_environment(tmp_path / "fake.jsonl", {"runtime_environment": payload})

    def test_validator_rejects_plaintext_in_redacted(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from validation.cutlass_migration.diagnostics.graph_replay_abba import (
            _validate_runtime_environment, _json_sha256,
        )
        payload = self._build_v2_payload(monkeypatch)
        payload["set_variables"]["B12X_AUTH_TOKEN"] = {
            "status": "redacted-set", "reason": "test", "value": _SENTINEL,
        }
        del payload["sha256"]
        payload["sha256"] = _json_sha256(payload)
        with pytest.raises(ValueError, match="set-variable map is invalid"):
            _validate_runtime_environment(tmp_path / "fake.jsonl", {"runtime_environment": payload})

    def test_validator_rejects_malformed_digest(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from validation.cutlass_migration.diagnostics.graph_replay_abba import (
            _validate_runtime_environment, _json_sha256,
        )
        payload = self._build_v2_payload(monkeypatch)
        payload["set_variables"]["B12X_BAD"] = {
            "status": "set-digest",
            "digest": {"algorithm": "sha256", "domain": "wrong", "value": "a" * 64},
        }
        del payload["sha256"]
        payload["sha256"] = _json_sha256(payload)
        with pytest.raises(ValueError, match="set-variable map is invalid"):
            _validate_runtime_environment(tmp_path / "fake.jsonl", {"runtime_environment": payload})

    def test_validator_rejects_extra_nvidia_keys(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from validation.cutlass_migration.diagnostics.graph_replay_abba import (
            _validate_runtime_environment, _json_sha256,
        )
        payload = self._build_v2_payload(monkeypatch)
        payload["nvidia_enumeration"]["extra_plaintext"] = _SENTINEL
        del payload["sha256"]
        payload["sha256"] = _json_sha256(payload)
        with pytest.raises(ValueError, match="NVIDIA environment enumeration policy"):
            _validate_runtime_environment(tmp_path / "fake.jsonl", {"runtime_environment": payload})

    def test_validator_accepts_v1_and_sanitizes(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from validation.cutlass_migration.diagnostics.graph_replay_abba import (
            _validate_runtime_environment, _json_sha256,
        )
        payload: dict[str, object] = {
            "schema": "b12x-runtime-environment-v1",
            "complete_set_variable_prefixes": list(COLLECTED_PREFIXES),
            "set_variables": {"B12X_AUTH_TOKEN": _SENTINEL, "B12X_TIMING": "1"},
            "explicit_controls": {
                name: {"status": "missing"}
                for name in (
                    "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER",
                    "CUDA_MODULE_LOADING", "CUDA_LAUNCH_BLOCKING",
                    "CUDA_DEVICE_MAX_CONNECTIONS", "CUDA_CACHE_DISABLE",
                    "CUDA_CACHE_PATH", "CUDA_CACHE_MAXSIZE",
                    "CUDA_FORCE_PTX_JIT", "CUDA_DISABLE_PTX_JIT",
                    "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
                    "NVIDIA_TF32_OVERRIDE",
                )
            },
            "nvidia_enumeration": {
                "policy": "explicit-only",
                "included": [
                    "NVIDIA_VISIBLE_DEVICES",
                    "NVIDIA_DRIVER_CAPABILITIES",
                    "NVIDIA_TF32_OVERRIDE",
                ],
                "reason": "legacy v1",
            },
        }
        payload["sha256"] = _json_sha256(payload)
        provenance = {"runtime_environment": payload}
        _validate_runtime_environment(tmp_path / "fake.jsonl", provenance)
        env = provenance["runtime_environment"]
        assert env["schema"] == "b12x-runtime-environment-v2"
        _assert_no_sentinel(json.dumps(env))


class TestComparisonSafeCuteDslLibs:
    def test_removes_package_owned_runtime(self) -> None:
        package_runtime = (
            "/tmp/site-packages/nvidia_cutlass_dsl/cu13/lib/"
            "libcute_dsl_runtime.so"
        )
        custom = "/opt/b12x-custom/libserving_runtime.so"
        result = comparison_safe_cute_dsl_libs(
            package_runtime + os.pathsep + custom
        )
        assert result == custom

    def test_preserves_order(self) -> None:
        a = "/opt/lib_a.so"
        b = "/opt/lib_b.so"
        result = comparison_safe_cute_dsl_libs(a + os.pathsep + b)
        assert result == a + os.pathsep + b

    def test_empty_components_dropped(self) -> None:
        assert comparison_safe_cute_dsl_libs("/opt/lib.so::") == "/opt/lib.so"


class TestGpuFieldSanitization:
    def test_cuda_visible_devices_gpu_field_sanitized(self) -> None:
        result = sanitize_value("CUDA_VISIBLE_DEVICES", _SENTINEL)
        assert result["status"] == "set-digest"
        _assert_no_sentinel(json.dumps(result))


class TestComparisonIdentityReturn:
    def test_normalize_comparison_compile_environment_returns_list(self) -> None:
        from validation.cutlass_migration.core.comparison_identity import (
            normalize_comparison_compile_environment,
        )
        result = normalize_comparison_compile_environment([
            ["CC", "v1:abc123"],
            ["CUTE_DSL_LIBS", "v1:def456"],
        ])
        assert isinstance(result, list)
        assert len(result) == 2

    def test_comparison_identity_different_package_runtimes_agree(self) -> None:
        from b12x._lib.provenance import safe_env_string, comparison_safe_cute_dsl_libs
        pkg_a = "/tmp/site-packages/nvidia_cutlass_dsl/cu12/lib/libcute_dsl_runtime.so"
        pkg_b = "/tmp/site-packages/nvidia_cutlass_dsl/cu13/lib/libcute_dsl_runtime.so"
        custom = "/opt/custom.so"
        prod_a = safe_env_string("CUTE_DSL_LIBS", pkg_a + os.pathsep + custom)
        prod_b = safe_env_string("CUTE_DSL_LIBS", pkg_b + os.pathsep + custom)
        assert prod_a != prod_b
        comp_a = safe_env_string("CUTE_DSL_LIBS", comparison_safe_cute_dsl_libs(pkg_a + os.pathsep + custom))
        comp_b = safe_env_string("CUTE_DSL_LIBS", comparison_safe_cute_dsl_libs(pkg_b + os.pathsep + custom))
        assert comp_a == comp_b
