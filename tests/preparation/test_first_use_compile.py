"""An in-process compile never inherits the artifacts that compile planning left behind."""
from types import SimpleNamespace

from b12x._lib import compile_plan, program_cache
from b12x._lib.compile_plan import DeferredCuTeKernel, DeferredTritonKernel, ProgramKey


def test_evict_planning_artifacts_drops_unresolved_deferred_programs_only(monkeypatch):
    """Planning eviction drops only unresolved deferred compiler artifacts."""
    planned = DeferredCuTeKernel(ProgramKey("cute", "a" * 64, "k"), memory_key=("m",))
    resolved = DeferredCuTeKernel(ProgramKey("cute", "b" * 64, "k"), memory_key=("n",))
    resolved._resolved = object()
    carrier = SimpleNamespace()
    object.__setattr__(carrier, "__b12x_programs__", (planned.__b12x_programs__[0],))
    object.__setattr__(carrier, "__b12x_dependencies__", (planned,))
    memo = {"planned": planned, "carrier": carrier, "resolved": resolved, "compiled": object()}
    mirror = {"planned": 1, "carrier": 2, "resolved": 3}
    monkeypatch.setattr(program_cache, "_MAPPING_CACHES", [(memo, (mirror,), None)])
    triton_planned = DeferredTritonKernel(ProgramKey("triton", "c" * 64, "t"), source=SimpleNamespace(name="t"))
    triton_resolved = DeferredTritonKernel(ProgramKey("triton", "d" * 64, "t"), source=SimpleNamespace(name="t"))
    triton_resolved._resolved = object()
    kernel_cache = {"k1": triton_planned, "k2": triton_resolved}
    key_cache = {"sig1": "k1", "sig2": "k2"}
    jit = type("Jit", (), {"device_caches": {0: (kernel_cache, key_cache, None, None)}})()
    monkeypatch.setattr(compile_plan, "_NATIVE_JITS", {jit})

    removed = compile_plan.evict_planning_artifacts([planned.__b12x_programs__[0]])

    assert removed == 2
    assert set(memo) == {"resolved", "compiled"}
    assert set(mirror) == {"resolved"}
    assert set(kernel_cache) == {"k1", "k2"} and set(key_cache) == {"sig1", "sig2"}

    assert compile_plan.evict_planning_artifacts() == 1
    assert set(kernel_cache) == {"k2"} and set(key_cache) == {"sig2"}


def test_program_keys_skip_torch_scalars_inside_launch_records():
    """Program discovery ignores Torch metadata alongside compiled kernels."""
    import torch
    from typing import NamedTuple

    class Launch(NamedTuple):
        compiled: object
        dtype: torch.dtype
        table: torch.Tensor

    kernel = DeferredTritonKernel(ProgramKey("triton", "e" * 64, "t"), source=SimpleNamespace(name="t"))
    assert compile_plan.program_keys(Launch(kernel, torch.int32, torch.zeros(1))) == kernel.__b12x_programs__


def test_program_keys_accept_plain_functions_without_triton(monkeypatch):
    """Host closures without retained programs do not import optional Triton."""
    import builtins

    original_import = builtins.__import__

    def reject_triton(name, *args, **kwargs):
        """Fail if program-key discovery imports the optional Triton package."""
        if name == "triton.compiler.compiler":
            raise AssertionError("program_keys imported Triton for a plain function")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_triton)
    assert compile_plan.program_keys(lambda: None) == ()


def test_first_use_evicts_deferred_launchers_from_decorated_kernel_memos(monkeypatch):
    """Deferred launchers are evicted while resolved launcher memo entries remain."""
    planned = DeferredCuTeKernel(ProgramKey("cute", "f" * 64, "planned"), memory_key=("p",))
    resolved = DeferredCuTeKernel(ProgramKey("cute", "1" * 64, "resolved"), memory_key=("r",))
    resolved._resolved = object()
    calls = []

    @program_cache.program_cache
    def factory(name):
        calls.append(name)
        kernel = planned if name == "planned" else resolved
        return compile_plan.attach_programs(lambda: None, kernel)

    monkeypatch.setattr(program_cache, "_CACHES", {factory})
    monkeypatch.setattr(program_cache, "_MAPPING_CACHES", [])
    monkeypatch.setattr(compile_plan, "_NATIVE_JITS", set())
    deferred_launcher = factory("planned")
    resident_launcher = factory("resolved")

    assert compile_plan.evict_planning_artifacts(planned.__b12x_programs__) == 1
    assert factory("resolved") is resident_launcher
    assert factory("planned") is not deferred_launcher
    assert calls == ["planned", "resolved", "planned"]


def test_mhc_program_bundle_reuse_obeys_deferred_and_resident_reclamation(monkeypatch):
    """MHC program bundles preserve resolved entries across planning eviction."""
    from b12x.norm.mhc import _preparation as mhc
    from b12x.preparation import FrozenMapping

    key = ProgramKey("cute", "2" * 64, "mhc")
    built = []

    def build(*_args):
        program = DeferredCuTeKernel(key, memory_key=("mhc",))
        built.append(program)
        return {"partial": program}

    mhc._compile_mhc.cache_clear()
    monkeypatch.setattr(mhc._compile_mhc, "_function", build)
    monkeypatch.setattr(mhc, "_codegen_snapshot", lambda: FrozenMapping({"constant": 1}))
    monkeypatch.setattr(program_cache, "_CACHES", {mhc._compile_mhc})
    monkeypatch.setattr(program_cache, "_MAPPING_CACHES", [])
    monkeypatch.setattr(compile_plan, "_NATIVE_JITS", set())
    payload = {"codegen": {"constant": 1}}
    try:
        planned = mhc.compile_mhc(payload, {}, {}, 0)
        assert mhc.compile_mhc(dict(payload), {}, {}, 0) is planned
        assert compile_plan.program_keys(planned) == (key,)
        assert compile_plan.evict_planning_artifacts((key,)) == 1
        resident = mhc.compile_mhc(payload, {}, {}, 0)
        assert resident is not planned
        resident["partial"]._resolved = object()
        assert compile_plan.evict_planning_artifacts((key,)) == 0
        assert mhc.compile_mhc(payload, {}, {}, 0) is resident
        assert mhc._compile_mhc.evict_unretained(frozenset({key})) == 0
        assert mhc._compile_mhc.evict_unretained(frozenset()) == 1
        assert len(built) == 2
    finally:
        mhc._compile_mhc.cache_clear()


def test_mhc_program_bundle_hit_still_validates_codegen_snapshot(monkeypatch):
    import pytest
    from b12x.norm.mhc import _preparation as mhc
    from b12x.preparation import FrozenMapping

    mhc._compile_mhc.cache_clear()
    monkeypatch.setattr(mhc._compile_mhc, "_function", lambda *_args: {})
    monkeypatch.setattr(mhc, "_codegen_snapshot", lambda: FrozenMapping({"constant": 1}))
    try:
        payload = {"codegen": {"constant": 1}}
        mhc.compile_mhc(payload, {}, {}, 0)
        monkeypatch.setattr(mhc, "_codegen_snapshot", lambda: FrozenMapping({"constant": 2}))
        with pytest.raises(ValueError, match="code-generation snapshot"):
            mhc.compile_mhc(payload, {}, {}, 0)
    finally:
        mhc._compile_mhc.cache_clear()
