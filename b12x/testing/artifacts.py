"""Verify source-built companion artifacts without trusting version suffixes."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import zipfile


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def wheel_manifest(wheel, source):
    """Bind every packaged Python/native file and matching source Python file."""
    wheel, source = Path(wheel).resolve(), Path(source).resolve()
    with zipfile.ZipFile(wheel) as archive:
        files = {
            n: hashlib.sha256(archive.read(n)).hexdigest()
            for n in archive.namelist()
            if n.startswith("vllm/") and (n.endswith(".py") or ".so" in n)
        }
    if not files or not any(n.endswith(".so") for n in files):
        raise ValueError("companion wheel lacks Python or native artifacts")
    verified = {}
    for name, digest in files.items():
        path = source / name
        if name.endswith(".py") and path.is_file() and name != "vllm/_version.py":
            if sha256(path) != digest:
                raise ValueError(f"wheel differs from companion source: {name}")
            verified[name] = digest
    for name in (
        "vllm/v1/engine/async_llm.py",
        "vllm/v1/engine/core.py",
        "vllm/v1/worker/gpu_worker.py",
        "vllm/model_executor/layers/fused_moe/b12x_cache.py",
    ):
        if name not in verified:
            raise ValueError(f"companion source coverage is incomplete: {name}")
    return dict(
        schema="b12x-expert-cache-artifact/v1",
        wheel=str(wheel),
        wheel_sha256=sha256(wheel),
        files=files,
        source_files=verified,
    )


def verify_installed(manifest, *, loaded_only=False):
    """Reject shadowed modules and mismatched installed or loaded binaries."""
    import vllm

    if manifest.get("schema") != "b12x-expert-cache-artifact/v1":
        raise ValueError("unsupported companion artifact manifest")
    root = Path(vllm.__file__).resolve().parent.parent
    files = manifest["files"]
    if not loaded_only:
        for name, digest in files.items():
            if sha256(root / name) != digest:
                raise ValueError(f"installed companion differs from wheel: {name}")
    modules = {}
    for name, module in tuple(sys.modules.items()):
        if name == "vllm" or name.startswith("vllm."):
            path = getattr(module, "__file__", None)
            if path is None:
                continue
            path = Path(path).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"shadowed companion module: {name}")
            relative = str(path.relative_to(root))
            digest = sha256(path)
            if relative not in files or files[relative] != digest:
                raise ValueError(f"loaded companion module differs from wheel: {name}")
            modules[relative] = digest
    native = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or "/vllm/" not in fields[5] or ".so" not in fields[5]:
            continue
        path = Path(fields[5]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"shadowed loaded companion library: {path}")
        relative = str(path.relative_to(root))
        if relative not in files or sha256(path) != files[relative]:
            raise ValueError(f"loaded native library differs from wheel: {path}")
        native[relative] = files[relative]
    return dict(
        root=str(root),
        modules=modules,
        native=native,
        wheel_sha256=manifest["wheel_sha256"],
        packages={
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
    )


def verify_from_file(path, **kwargs):
    return verify_installed(json.loads(Path(path).read_text()), **kwargs)
