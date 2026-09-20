"""Acceptance rejects incomplete source and shutdown evidence."""

import hashlib
import zipfile

import pytest

from b12x.testing.artifacts import wheel_manifest
from scripts.qualify_expert_cache import HOST, GPU


def test_wheel_binding_rejects_shadowed_source(tmp_path):
    names = (
        "vllm/v1/engine/async_llm.py",
        "vllm/v1/engine/core.py",
        "vllm/v1/worker/gpu_worker.py",
        "vllm/model_executor/layers/fused_moe/b12x_cache.py",
    )
    wheel = tmp_path / "engine.whl"
    source = tmp_path / "source"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in names:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"# source\n")
            archive.writestr(name, path.read_bytes())
        archive.writestr("vllm/_C.so", b"native")
    result = wheel_manifest(wheel, source)
    assert len(result["source_files"]) == len(names)
    assert result["files"]["vllm/_C.so"] == hashlib.sha256(b"native").hexdigest()
    (source / names[0]).write_text("# changed\n")
    with pytest.raises(ValueError, match="differs"):
        wheel_manifest(wheel, source)


def test_acceptance_keeps_registry_and_real_gpu_paths():
    assert "tests/test_registry.py" in HOST
    assert "tests/moe/test_prepared_expert_cache.py" in GPU
    assert "tests/moe/test_routing_profile_gpu.py" in GPU


def test_process_manager_force_kill_cannot_pass_serving_acceptance(tmp_path):
    from scripts.qualify_expert_cache import require_complete

    log = tmp_path / "engine.log"
    log.write_text(
        "[shutdown] Process manager: force killing remaining process EngineCore"
    )
    with pytest.raises(ValueError, match="unclean shutdown"):
        require_complete(tmp_path / "receipt.jsonl", log)
