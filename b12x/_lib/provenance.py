"""Strict provenance sanitization for environment-variable captures.

All benchmark, validation, and compiler provenance producers route through
these helpers so that secret-like environment variables are never persisted
in generated artifacts or logs.

Policy (one strict safe-variable policy shared by every producer):

1.  **Secret-like names** — any variable whose name (after normalizing ALL
    non-alphanumeric separators) contains ``token``, ``key``, ``password``,
    ``secret``, ``credential``, ``auth``, ``cookie``, ``passwd``, ``pwd``,
    ``apikey``, ``passphrase``, ``bearer``, ``session``, ``jwt``, ``cred``
    is *always* redacted with a constant tagged marker — never hashed.

2.  **Per-name canonicalizers** — known-safe variable names have an
    associated canonicalizer that validates and canonicalizes the value
    (ASCII-only, bounded ranges, exact cardinality, canonical booleans/enums).
    Only canonicalized values are recorded.  Non-canonical or out-of-range
    values are digested.

3.  **Unknown broadly-prefixed variables** — variables matching a collected
    prefix but absent from the canonicalizer set record a versioned
    domain-separated SHA-256 digest, never raw values.

4.  **Tagged representations** — every entry carries an explicit status:
    ``unset``, ``set-safe``, ``set-digest``, or ``redacted-set``.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Mapping

__all__ = [
    "COLLECTED_PREFIXES",
    "DIGEST_ALGORITHM",
    "DIGEST_DOMAIN",
    "REDACTED_MARKER",
    "REDACTED_REASON",
    "is_secret_name",
    "sanitize_value",
    "sanitize_environment_map",
    "sanitize_environment_tuple",
    "safe_env_string",
    "safe_env_tuple",
    "comparison_safe_cute_dsl_libs",
]

# ── Domain separation ──────────────────────────────────────────────────
DIGEST_ALGORITHM = "sha256"
DIGEST_DOMAIN = "b12x.provenance.env.v1"
REDACTED_MARKER = "<redacted:secret-like-name>"
REDACTED_REASON = "secret-like-name"

# ── Secret-like name fragments ────────────────────────────────────────
# Matched as substrings after NFKC normalization + casefolding + stripping
# non-ASCII-alphanumerics.  An exception list prevents false positives on
# known-safe controls like B12X_MHC_PREFILL_BF16_MIN_TOKENS.
_SECRET_NAME_FRAGMENTS: tuple[str, ...] = (
    "token", "key", "password", "secret", "credential", "auth", "cookie",
    "passwd", "pwd", "apikey", "privatekey", "certificate",
    "passphrase", "bearer", "session", "jwt", "cred",
)

# Normalized substrings that exempt a name from secret classification even
# if it contains a secret-like fragment (e.g. MIN_TOKENS is a tuning integer).
_SAFE_SUBSTRINGS: tuple[str, ...] = (
    "mintokens", "maxtokens",
)

import unicodedata


def _normalize_name(name: str) -> str:
    """NFKC-normalize, casefold, then keep only ASCII alphanumerics."""
    nfkc = unicodedata.normalize("NFKC", name)
    casefolded = nfkc.casefold()
    return re.sub(r"[^a-z0-9]", "", casefolded)


def is_secret_name(name: str) -> bool:
    """Return True if *name* contains a secret-like fragment.

    Uses NFKC normalization + casefolding and substring matching with an
    exception list for known-safe controls (MIN_TOKENS, MAX_TOKENS).
    """
    normalized = _normalize_name(name)
    # Check exceptions first.
    if any(exc in normalized for exc in _SAFE_SUBSTRINGS):
        return False
    return any(frag in normalized for frag in _SECRET_NAME_FRAGMENTS)


# ── Digest ─────────────────────────────────────────────────────────────

def _digest(name: str, value: str) -> str:
    h = hashlib.new(DIGEST_ALGORITHM)
    h.update(DIGEST_DOMAIN.encode("ascii"))
    h.update(b"\x00")
    h.update(name.encode("utf-8"))
    h.update(b"\x00")
    h.update(value.encode("utf-8"))
    return h.hexdigest()


# ── Per-name canonicalizers ────────────────────────────────────────────
# Each canonicalizer returns a canonical string if the value is valid,
# or None if it should be digested.  Canonicalizers enforce ASCII-only,
# bounded ranges, exact cardinality, and canonical forms.

_ASCII_INT_RE = re.compile(r"^[0-9]+$")
_ASCII_FLOAT_RE = re.compile(r"^[0-9]+(\.[0-9]+)?$")


def _canon_bool(value: str) -> str | None:
    low = value.lower()
    if low in ("1", "true", "yes", "on"):
        return "1"
    if low in ("0", "false", "no", "off", ""):
        return "0"
    return None


def _canon_int_in_range(lo: int, hi: int, max_digits: int = 10) -> Callable[[str], str | None]:
    def canon(value: str) -> str | None:
        if not _ASCII_INT_RE.fullmatch(value):
            return None
        if len(value) > max_digits:
            return None
        v = int(value)
        if lo <= v <= hi:
            return str(v)
        return None
    return canon


def _canon_float_or_int(lo: float, hi: float, max_digits: int = 12) -> Callable[[str], str | None]:
    def canon(value: str) -> str | None:
        if not _ASCII_FLOAT_RE.fullmatch(value):
            return None
        if len(value) > max_digits:
            return None
        v = float(value)
        if lo <= v <= hi:
            return str(v)
        return None
    return canon


def _canon_in_set(valid: frozenset[str]) -> Callable[[str], str | None]:
    def canon(value: str) -> str | None:
        return value if value in valid else None
    return canon


def _canon_csv_in_set(valid: frozenset[str], min_items: int = 1, max_items: int = 16) -> Callable[[str], str | None]:
    def canon(value: str) -> str | None:
        if not value or len(value) > 1024:
            return None
        # Split and reject empty parts (no filtering).
        parts = value.split(",")
        if any(not p.strip() for p in parts):
            return None
        parts = [p.strip() for p in parts]
        if len(parts) < min_items or len(parts) > max_items:
            return None
        if not all(p in valid for p in parts):
            return None
        return ",".join(parts)
    return canon


def _canon_device_list(value: str) -> str | None:
    if not value or len(value) > 1024:
        return None
    # Split and reject empty parts.
    parts = value.split(",")
    if any(not p.strip() for p in parts):
        return None
    parts = [p.strip() for p in parts]
    if len(parts) > 64:
        return None
    if not all(_ASCII_INT_RE.fullmatch(p) and len(p) <= 4 and int(p) < 256 for p in parts):
        return None
    return ",".join(str(int(p)) for p in parts)


def _canon_arch(value: str) -> str | None:
    m = re.fullmatch(r"sm_([0-9]{1,3})a?", value)
    if not m:
        return None
    arch_num = int(m.group(1))
    if not (50 <= arch_num <= 999):
        return None
    return value


_LOG_LEVEL_VALUES = frozenset({
    "0", "1", "stack", "trace", "traceback", "full",
})

_DMA_FP8_VALUES = frozenset({"0", "1", "i8_ring", "i8_a2a", "bf16"})
_NCCL_P2P_LEVEL_VALUES = frozenset({"0", "1", "2", "LOC", "NL", "NVL", "SYS"})
_NVIDIA_CAPS = frozenset({
    "compute", "utility", "video", "graphics", "compat32", "display", "all",
})
_CUDA_MODULE_LOADING = frozenset({"LAZY", "EAGER", "AUTO", "0", "1"})
_CUDA_ORDER = frozenset({"FAST_FIRST", "PCI_BUS_ID"})
_BOOL_OR_KEYWORD = frozenset({"0", "1", "false", "true"})


_SAFE_CANONICALIZERS: dict[str, Callable[[str], str | None]] = {
    "CUDA_VISIBLE_DEVICES": _canon_device_list,
    "CUDA_DEVICE_ORDER": _canon_in_set(_CUDA_ORDER),
    "CUDA_MODULE_LOADING": _canon_in_set(_CUDA_MODULE_LOADING),
    "CUDA_LAUNCH_BLOCKING": _canon_bool,
    "CUDA_DEVICE_MAX_CONNECTIONS": _canon_int_in_range(1, 256),
    "CUDA_CACHE_DISABLE": _canon_bool,
    "CUDA_CACHE_MAXSIZE": _canon_int_in_range(0, 2**31 - 1),
    "CUDA_FORCE_PTX_JIT": _canon_bool,
    "CUDA_DISABLE_PTX_JIT": _canon_bool,
    "NVIDIA_VISIBLE_DEVICES": _canon_device_list,
    "NVIDIA_DRIVER_CAPABILITIES": _canon_csv_in_set(_NVIDIA_CAPS),
    "NVIDIA_TF32_OVERRIDE": _canon_bool,
    "NCCL_P2P_DISABLE": _canon_bool,
    "NCCL_P2P_LEVEL": _canon_in_set(_NCCL_P2P_LEVEL_VALUES),
    "B12X_TIMING": _canon_bool,
    "B12X_COMPILE_DISK_CACHE": _canon_bool,
    "B12X_COMPILE_MEMORY_CACHE": _canon_bool,
    "B12X_COMPILE_MEMORY_CACHE_SIZE": _canon_int_in_range(1, 2**31 - 1),
    "B12X_COMPILE_SPEC_MEMO": _canon_bool,
    "B12X_LOG_CUTE_COMPILES": _canon_in_set(_LOG_LEVEL_VALUES),
    "B12X_LOG_CUTE_COMPILES_AFTER_ENGINE_START": _canon_in_set(_LOG_LEVEL_VALUES),
    "B12X_LOG_CUTE_COMPILE_ARGS": _canon_bool,
    "B12X_LOG_CUTE_COMPILE_STACK": _canon_in_set(_LOG_LEVEL_VALUES),
    "B12X_LOG_CUTE_COMPILE_STACK_DEPTH": _canon_int_in_range(1, 256),
    "B12X_PRINT_COMPILE_PROGRESS": _canon_bool,
    "B12X_TIMING_THRESHOLD_MS": _canon_float_or_int(0.0, 1e9),
    "B12X_DENSE_SPLITK_TURBO": _canon_bool,
    "B12X_DENSE_ATOM_24": _canon_bool,
    "B12X_PACKED_B_EXPAND_AHEAD": _canon_bool,
    "B12X_DENSE_FUSED_QUANT": _canon_bool,
    "B12X_PCIE_DMA_FP8": _canon_in_set(_DMA_FP8_VALUES),
    "B12X_PCIE_DMA_PIECES": _canon_int_in_range(0, 64),
    "B12X_PCIE_DMA_A2A_CHUNKS": _canon_int_in_range(0, 64),
    "B12X_PCIE_DCP_THREADS": _canon_int_in_range(0, 4096),
    "B12X_PCIE_DCP_BLOCK_LIMIT": _canon_int_in_range(0, 4096),
    "B12X_PCIE_HIERARCHICAL_NANOSLEEP_CYCLES": _canon_int_in_range(0, 1024),
    "B12X_PCIE_HIERARCHICAL_THREADS": _canon_int_in_range(32, 1024),
    "B12X_PCIE_HIERARCHICAL_BF16X2": _canon_bool,
    "B12X_PCIE_HIERARCHICAL_BF16X2_MAX_ELEMENTS": _canon_int_in_range(0, 2**30),
    "B12X_PCIE_HIERARCHICAL_DOUBLE_BUFFER": _canon_bool,
    "B12X_PCIE_HIERARCHICAL_DEFERRED_CONSUMPTION": _canon_bool,
    "B12X_MLA_SM120_PREFILL_MG": _canon_in_set(_BOOL_OR_KEYWORD),
    "B12X_MLA_SM120_PREFILL_PACK_HILO_ROWS": _canon_in_set(_BOOL_OR_KEYWORD),
    "B12X_NSA_VALIDATE_PAGE_IDS": _canon_bool,
    "B12X_MSA_DECODE_PAGEMAX": _canon_in_set(frozenset({"0", "1"})),
    "B12X_VALIDATE_PAGED_INDEXER_CUDA_VALUES": _canon_bool,
    "B12X_INDEXER_DIRECT_K": _canon_bool,
    "CUTE_DSL_ARCH": _canon_arch,
    "OMP_NUM_THREADS": _canon_int_in_range(1, 2**31 - 1),
    "B12X_MHC_PREFILL_BF16_MIN_TOKENS": _canon_int_in_range(0, 2**31 - 1),
    "B12X_MHC_PREFILL_TF32_MIN_TOKENS": _canon_int_in_range(0, 2**31 - 1),
}

# ── Broad prefixes ─────────────────────────────────────────────────────
COLLECTED_PREFIXES: tuple[str, ...] = (
    "B12X_", "CUTE_", "CUTLASS_", "CUDA_", "TORCH_", "PYTORCH_",
    "TRITON_", "NVCC_", "PTXAS_", "NCCL_",
)


# ── Canonicalization ──────────────────────────────────────────────────

def _canonicalize(name: str, value: str) -> str | None:
    """Return canonical value if safe, or None to digest."""
    canon = _SAFE_CANONICALIZERS.get(name)
    if canon is None:
        return None
    return canon(value)


# ── Sanitization (tagged object form) ─────────────────────────────────

def sanitize_value(name: str, value: str) -> dict[str, object]:
    """Return a tagged dict for one environment variable value."""
    if is_secret_name(name):
        return {"status": "redacted-set", "reason": REDACTED_REASON}
    canonical = _canonicalize(name, value)
    if canonical is not None:
        return {"status": "set-safe", "value": canonical}
    return {
        "status": "set-digest",
        "digest": {
            "algorithm": DIGEST_ALGORITHM,
            "domain": DIGEST_DOMAIN,
            "value": _digest(name, value),
        },
    }


def safe_env_string(name: str, value: str) -> str:
    """Return a plain-string sanitized value for hashable cache keys.

    * Secret-like names → constant ``REDACTED_MARKER``
    * Safe canonicalized values → the canonical string
    * Unknown/failing values → ``"v1:" + hex_digest``
    """
    if is_secret_name(name):
        return REDACTED_MARKER
    canonical = _canonicalize(name, value)
    if canonical is not None:
        return canonical
    return f"v1:{_digest(name, value)}"


def sanitize_environment_map(
    env: Mapping[str, str] | None = None,
    *,
    prefixes: tuple[str, ...] = COLLECTED_PREFIXES,
    explicit_names: frozenset[str] | None = None,
) -> dict[str, dict[str, object]]:
    """Return a sanitized ``{name: tagged_dict}`` map."""
    source = os.environ if env is None else env
    result: dict[str, dict[str, object]] = {}
    for name in sorted(source):
        if name.startswith(prefixes) or (explicit_names and name in explicit_names):
            result[name] = sanitize_value(name, source[name])
    return result


def sanitize_environment_tuple(
    env: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, dict[str, object]], ...]:
    """Return a sanitized ``((name, tagged_dict), ...)`` tuple."""
    return tuple((name, sanitize_value(name, value)) for name, value in env)


def safe_env_tuple(
    env: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    """Return a hashable sanitized tuple for compiler cache keys."""
    return tuple((name, safe_env_string(name, value)) for name, value in env)


# ── CUTE_DSL_LIBS comparison-safe normalization ───────────────────────

def _is_package_owned_cute_dsl_runtime(component: str) -> bool:
    from pathlib import Path as _Path
    candidate = _Path(component)
    return (
        candidate.name == "libcute_dsl_runtime.so"
        and "nvidia_cutlass_dsl" in candidate.parts
    )


def comparison_safe_cute_dsl_libs(value: str) -> str:
    """Return comparison-safe form: package runtime removed, custom kept ordered."""
    retained = [
        c for c in value.split(os.pathsep)
        if c and not _is_package_owned_cute_dsl_runtime(c)
    ]
    return os.pathsep.join(retained)
