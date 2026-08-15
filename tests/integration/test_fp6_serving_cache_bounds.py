"""Focused integration tests for bounded FP6 serving output/scratch caches (issue #164).

Tests the real production classes from :mod:`b12x.integration.vllm.fp6_serving`:
:class:`CaptureOutputCache`, :class:`ServingScratchArena`, and
:class:`B12XFP6MoEMethod` — with a patched ``fused_moe`` module that provides
a proper ``Caps`` dataclass and matching ``scratch_specs``/``shapes_and_dtypes``.

Also tests the plugin-side catalog derivation function
:func:`capture_sizes_from_config`.

Covers: catalog-bounded retention across M=1..512, stable addresses for
catalogued sizes, one bounded reusable uncatalogued eager slot, capture
rejection before allocation, multi-owner isolation/churn, shared arena across
layers, dtype/device validation, pointer stability/zeroing, and teardown.
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest

try:
    import torch

    _has_torch = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _has_torch = False

requires_torch = pytest.mark.skipif(not _has_torch, reason="torch not installed")

_DTYPE = torch.float32
_DEV = torch.device("cpu")


@pytest.fixture(autouse=True)
def _patch_capture_check(monkeypatch):
    """Patch is_current_stream_capturing to return False (eager mode) for CPU."""
    if _has_torch:
        monkeypatch.setattr(
            torch.cuda, "is_current_stream_capturing", lambda: False
        )


# ---------------------------------------------------------------------------
# Fake fused_moe module with proper Caps and matching scratch specs
# ---------------------------------------------------------------------------


@dataclass
class _FakeCaps:
    max_tokens: int
    num_topk: int
    device: Any
    weight_plan: Any
    core_token_counts: tuple
    route_num_experts: int
    quant_mode: str
    apply_router_weight_on_input: bool


class _FakeScratchSpec:
    def __init__(self, device: Any):
        self.device = device


class _FakePlan:
    """Fake plan with scratch_specs count matching shapes_and_dtypes."""

    def __init__(self, m: int, topk: int, device: Any):
        self._m = m
        self._topk = topk
        self._device = device

    def scratch_specs(self):
        return [_FakeScratchSpec(self._device), _FakeScratchSpec(self._device)]

    def shapes_and_dtypes(self):
        return [
            ((self._m, 64), torch.float32),
            ((self._topk, self._m), torch.int32),
        ]


class _FakeFusedMoe:
    Caps = _FakeCaps

    @staticmethod
    def plan(caps: _FakeCaps) -> _FakePlan:
        return _FakePlan(caps.max_tokens, caps.num_topk, caps.device)

    @staticmethod
    def bind(plan: Any, **kw: Any) -> dict:
        return kw

    @staticmethod
    def run(binding: dict) -> Any:
        return binding["output"]

    @staticmethod
    def plan_weights(**kw: Any) -> Any:
        # Use a class that supports __weakref__ so WeakValueDictionary works.
        class _WeakPlan:
            def __init__(self, **kw):
                self._kw = kw
            def __getattr__(self, name):
                return self._kw[name]
        return _WeakPlan(**kw)

    @staticmethod
    def prepare_weights(**kw: Any) -> Any:
        return kw.get("plan")


@pytest.fixture
def mock_fused_moe(monkeypatch):
    """Inject a fake fused_moe so production classes work without CUDA kernels.

    Saves and restores both sys.modules entries AND the package attribute
    on the b12x.moe package (if imported) to avoid order-poisoning.
    """
    saved_mod = sys.modules.get("b12x.moe.fused_moe")
    saved_pkg = sys.modules.get("b12x.moe")
    saved_attr = None
    if saved_pkg is not None and hasattr(saved_pkg, "fused_moe"):
        saved_attr = saved_pkg.fused_moe
    fake = _FakeFusedMoe()
    mod = types.ModuleType("b12x.moe.fused_moe")
    mod.Caps = fake.Caps
    mod.plan = fake.plan
    mod.bind = fake.bind
    mod.run = fake.run
    mod.plan_weights = fake.plan_weights
    mod.prepare_weights = fake.prepare_weights
    monkeypatch.setitem(sys.modules, "b12x.moe.fused_moe", mod)
    # Also patch the package attribute if b12x.moe is already imported
    if saved_pkg is not None:
        saved_pkg.fused_moe = mod
    yield mod
    # Restore original sys.modules state
    if saved_mod is not None:
        sys.modules["b12x.moe.fused_moe"] = saved_mod
    else:
        sys.modules.pop("b12x.moe.fused_moe", None)
    if saved_pkg is not None:
        sys.modules["b12x.moe"] = saved_pkg
        if saved_attr is not None:
            saved_pkg.fused_moe = saved_attr
        elif hasattr(saved_pkg, "fused_moe"):
            del saved_pkg.fused_moe


@pytest.fixture
def clean_plan_cache():
    """Clear the process-wide weight-plan cache before and after each test."""
    from b12x.integration.vllm import fp6_serving

    fp6_serving._WEIGHT_PLAN_CACHE.clear()
    yield
    fp6_serving._WEIGHT_PLAN_CACHE.clear()


def _make_arena(catalog, **kw):
    from b12x.integration.vllm.fp6_serving import ServingScratchArena

    plan = types.SimpleNamespace()
    return ServingScratchArena(
        catalog, plan, kw.get("topk", 2), _DEV,
        source_format=kw.get("source_format", "mxfp6_e2m3"),
        activation=kw.get("activation", "silu"),
        num_experts=kw.get("num_experts", 8),
        hidden_size=kw.get("hidden_size", 512),
        intermediate_size=kw.get("intermediate_size", 256),
        apply_router_weight_on_input=kw.get("apply_router_weight_on_input", False),
        eager_max_tokens=kw.get("eager_max_tokens", 512),
    )


# ---------------------------------------------------------------------------
# CaptureOutputCache (production class from fp6_serving)
# ---------------------------------------------------------------------------


@requires_torch
class TestCaptureOutputCache:
    """Contract: output cache is bounded by the declared catalog."""

    def test_catalogued_m_retains_and_reuses(self):
        from b12x.integration.vllm.fp6_serving import CaptureOutputCache

        cache = CaptureOutputCache(
            catalog=(1, 2, 4, 8, 16, 32),
            hidden_size=128, dtype=_DTYPE, device=_DEV,
        )
        buf1 = cache.get(8, _DTYPE, _DEV)
        assert buf1 is not None
        assert buf1.shape == (8, 128)
        buf1.fill_(1.0)
        buf2 = cache.get(8, _DTYPE, _DEV)
        assert buf2 is buf1
        assert buf2.sum() == 0.0, "reused buffer must be zeroed"

    def test_uncatalogued_eager_reuses_capacity_view(self):
        from b12x.integration.vllm.fp6_serving import CaptureOutputCache

        cache = CaptureOutputCache(
            catalog=(1, 2, 4, 8), hidden_size=128, dtype=_DTYPE, device=_DEV,
            eager_max_tokens=64,
        )
        first = cache.get(7, _DTYPE, _DEV)
        assert first is not None and first.shape == (7, 128)
        assert cache.eager_capacity == 8
        ptr = first.data_ptr()
        first.fill_(1.0)
        second = cache.get(6, _DTYPE, _DEV)
        assert second is not None and second.data_ptr() == ptr
        assert second.shape == (6, 128) and second.sum() == 0.0

    def test_retained_count_equals_catalog_size_across_m1_to_512(self):
        from b12x.integration.vllm.fp6_serving import CaptureOutputCache

        catalog = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
        cache = CaptureOutputCache(
            catalog=catalog, hidden_size=128, dtype=_DTYPE, device=_DEV,
            eager_max_tokens=512,
        )
        for m in range(1, 513):
            cache.get(m, _DTYPE, _DEV)
        assert len(cache) == len(catalog)

    def test_clear_empties_all(self):
        from b12x.integration.vllm.fp6_serving import CaptureOutputCache

        cache = CaptureOutputCache(
            catalog=(1, 2, 4, 8), hidden_size=128, dtype=_DTYPE, device=_DEV,
        )
        for m in (1, 2, 4, 8):
            cache.get(m, _DTYPE, _DEV)
        assert len(cache) == 4
        cache.clear()
        assert len(cache) == 0

    def test_stable_address_across_diversity(self):
        from b12x.integration.vllm.fp6_serving import CaptureOutputCache

        catalog = (1, 8, 32)
        cache = CaptureOutputCache(
            catalog=catalog, hidden_size=64, dtype=_DTYPE, device=_DEV,
            eager_max_tokens=512,
        )
        first = {m: cache.get(m, _DTYPE, _DEV) for m in catalog}
        for _ in range(10):
            for m in range(1, 513):
                cache.get(m, _DTYPE, _DEV)
        for m in catalog:
            assert cache.get(m, _DTYPE, _DEV) is first[m], f"M={m} address changed"

    def test_uncatalogued_capture_rejected(self, monkeypatch):
        from b12x.integration.vllm.fp6_serving import CaptureOutputCache

        cache = CaptureOutputCache(
            catalog=(1, 2, 4, 8), hidden_size=128, dtype=_DTYPE, device=_DEV,
        )
        prepared = cache.get(8, _DTYPE, _DEV)
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        with pytest.raises(RuntimeError, match="uncatalogued M=7"):
            cache.get(7, _DTYPE, _DEV)
        buf = cache.get(8, _DTYPE, _DEV)
        assert buf is prepared and buf.shape == (8, 128)

    def test_catalogued_capture_miss_rejected_before_allocation(self, monkeypatch):
        from b12x.integration.vllm.fp6_serving import CaptureOutputCache

        cache = CaptureOutputCache(
            catalog=(1, 2, 4, 8), hidden_size=128, dtype=_DTYPE, device=_DEV,
        )
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        with pytest.raises(RuntimeError, match="was not prepared"):
            cache.get(8, _DTYPE, _DEV)
        assert len(cache) == 0

    def test_catalogued_hit_does_not_query_capture_state(self, monkeypatch):
        from b12x.integration.vllm.fp6_serving import CaptureOutputCache

        cache = CaptureOutputCache(
            catalog=(1, 2, 4, 8), hidden_size=128, dtype=_DTYPE, device=_DEV,
        )
        prepared = cache.get(8, _DTYPE, _DEV)
        monkeypatch.setattr(
            torch.cuda,
            "is_current_stream_capturing",
            lambda: pytest.fail("prepared output queried capture state"),
        )
        assert cache.get(8, _DTYPE, _DEV) is prepared

    def test_uncatalogued_eager_capacity_bound(self):
        from b12x.integration.vllm.fp6_serving import CaptureOutputCache

        cache = CaptureOutputCache(
            catalog=(1, 2, 4, 8), hidden_size=128, dtype=_DTYPE, device=_DEV,
            eager_max_tokens=32,
        )
        with pytest.raises(RuntimeError, match="exceeds configured capacity 32"):
            cache.get(33, _DTYPE, _DEV)

    def test_dtype_mismatch_rejected(self):
        from b12x.integration.vllm.fp6_serving import CaptureOutputCache

        cache = CaptureOutputCache(
            catalog=(1, 2, 4, 8), hidden_size=128, dtype=_DTYPE, device=_DEV,
        )
        with pytest.raises(RuntimeError, match="dtype/device mismatch"):
            cache.get(8, torch.bfloat16, _DEV)

    def test_device_mismatch_rejected(self):
        from b12x.integration.vllm.fp6_serving import CaptureOutputCache

        cache = CaptureOutputCache(
            catalog=(1, 2, 4, 8), hidden_size=128, dtype=_DTYPE, device=_DEV,
        )
        with pytest.raises(RuntimeError, match="dtype/device mismatch"):
            cache.get(8, _DTYPE, torch.device("meta"))


# ---------------------------------------------------------------------------
# ServingScratchArena (production class from fp6_serving)
# ---------------------------------------------------------------------------


@requires_torch
class TestServingScratchArena:
    """Contract: scratch arena is model-owned and bounded by catalog."""

    def test_catalogued_m_retains_scratch(self, mock_fused_moe):
        arena = _make_arena((1, 2, 4, 8))
        p1, s1 = arena.get_or_build(4)
        p2, s2 = arena.get_or_build(4)
        assert p1 is p2 and s1 is s2

    def test_uncatalogued_eager_reuses_one_capacity_slot(self, mock_fused_moe):
        arena = _make_arena((1, 2, 4, 8))
        p1, s1 = arena.get_or_build(7)
        assert arena.eager_capacity == 8
        p2, s2 = arena.get_or_build(6)
        assert p1 is p2 and s1 is s2
        assert len(arena) == 0

    def test_retained_count_bounded_across_m1_to_512(self, mock_fused_moe):
        catalog = (1, 2, 4, 8, 16, 32)
        arena = _make_arena(catalog)
        for m in range(1, 513):
            arena.get_or_build(m)
        assert len(arena) == len(catalog)

    def test_catalogued_scratch_stable_across_diversity(self, mock_fused_moe):
        catalog = (1, 8, 32)
        arena = _make_arena(catalog)
        refs = {m: arena.get_or_build(m) for m in catalog}
        for m in range(1, 513):
            if m not in set(catalog):
                arena.get_or_build(m)
        for m in catalog:
            p, s = arena.get_or_build(m)
            assert p is refs[m][0] and s is refs[m][1]

    def test_empty_catalog_retains_nothing(self, mock_fused_moe):
        arena = _make_arena(())
        for m in (1, 2, 4, 8, 16, 32, 64, 128):
            arena.get_or_build(m)
        assert len(arena) == 0

    def test_uncatalogued_capture_rejected_before_allocation(
        self, mock_fused_moe, monkeypatch
    ):
        arena = _make_arena((1, 2, 4, 8))
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        with pytest.raises(RuntimeError, match="uncatalogued M=7"):
            arena.get_or_build(7)
        assert len(arena) == 0

    def test_catalogued_capture_miss_rejected_before_allocation(
        self, mock_fused_moe, monkeypatch
    ):
        arena = _make_arena((1, 2, 4, 8))
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        with pytest.raises(RuntimeError, match="was not prepared"):
            arena.get_or_build(8)
        assert len(arena) == 0 and arena.eager_capacity == 0

    def test_catalogued_hit_does_not_query_capture_state(
        self, mock_fused_moe, monkeypatch
    ):
        arena = _make_arena((1, 2, 4, 8))
        prepared = arena.get_or_build(8)
        monkeypatch.setattr(
            torch.cuda,
            "is_current_stream_capturing",
            lambda: pytest.fail("prepared scratch queried capture state"),
        )
        assert arena.get_or_build(8) == prepared

    def test_uncatalogued_eager_capacity_bound(self, mock_fused_moe):
        arena = _make_arena((1, 2, 4, 8), eager_max_tokens=32)
        with pytest.raises(RuntimeError, match="exceeds configured capacity 32"):
            arena.get_or_build(33)

    def test_clear_empties_all(self, mock_fused_moe):
        arena = _make_arena((1, 2, 4, 8))
        for m in (1, 2, 4, 8):
            arena.get_or_build(m)
        assert len(arena) == 4
        arena.clear()
        assert len(arena) == 0

    def test_different_geometries_dont_alias(self, mock_fused_moe):
        arena_a = _make_arena((1, 2, 4, 8), hidden_size=512, intermediate_size=256)
        arena_b = _make_arena((1, 2, 4, 8), hidden_size=1024, intermediate_size=512)
        _, s_a = arena_a.get_or_build(4)
        _, s_b = arena_b.get_or_build(4)
        assert s_a is not s_b

    def test_validate_geometry_mismatch_raises(self, mock_fused_moe):
        """Arena must reject a layer with different geometry."""
        arena = _make_arena((1, 2, 4, 8), hidden_size=512, intermediate_size=256)
        with pytest.raises(RuntimeError, match="geometry mismatch"):
            arena.validate_geometry(
                weight_plan=arena.weight_plan, topk=2, device=_DEV,
                source_format="mxfp6_e2m3", activation="silu",
                num_experts=8, hidden_size=1024, intermediate_size=512,
                apply_router_weight_on_input=False,
            )

    def test_validate_geometry_match_succeeds(self, mock_fused_moe):
        """Arena must accept a layer with matching geometry."""
        arena = _make_arena((1, 2, 4, 8), hidden_size=512, intermediate_size=256)
        arena.validate_geometry(
            weight_plan=arena.weight_plan, topk=2, device=_DEV,
            source_format="mxfp6_e2m3", activation="silu",
            num_experts=8, hidden_size=512, intermediate_size=256,
            apply_router_weight_on_input=False,
        )  # should not raise


# ---------------------------------------------------------------------------
# Multi-owner isolation
# ---------------------------------------------------------------------------


@requires_torch
class TestMultiOwnerIsolation:
    """Contract: separate model owners do not share or accumulate scratch."""

    def test_two_owners_isolated(self, mock_fused_moe):
        arena_a = _make_arena((1, 2))
        arena_b = _make_arena((4, 8))
        arena_a.get_or_build(1)
        arena_a.get_or_build(2)
        arena_b.get_or_build(4)
        arena_b.get_or_build(8)
        assert len(arena_a) == 2 and len(arena_b) == 2
        arena_a.get_or_build(4)
        assert len(arena_a) == 2
        arena_b.get_or_build(1)
        assert len(arena_b) == 2

    def test_churn_does_not_accumulate(self, mock_fused_moe):
        for catalog in [(1, 2), (4, 8), (16, 32), (64, 128)]:
            arena = _make_arena(catalog)
            for m in range(1, 513):
                arena.get_or_build(m)
            assert len(arena) == len(catalog)
            arena.clear()
            assert len(arena) == 0

    def test_clear_one_does_not_affect_other(self, mock_fused_moe):
        arena_a = _make_arena((1, 2))
        arena_b = _make_arena((4, 8))
        arena_a.get_or_build(1)
        arena_b.get_or_build(4)
        arena_a.clear()
        assert len(arena_a) == 0 and len(arena_b) == 1


# ---------------------------------------------------------------------------
# Shared arena across layers: one arena serves multiple B12XFP6MoEMethod instances
# ---------------------------------------------------------------------------


@requires_torch
class TestSharedArenaAcrossLayers:
    """Contract: one arena shared across layers — scratch is catalog-sized, not
    catalog × layers."""

    def test_shared_arena_scratch_count_independent_of_layer_count(
        self, mock_fused_moe
    ):
        from b12x.integration.vllm.fp6_serving import B12XFP6MoEMethod

        catalog = (1, 2, 4, 8)
        arena = _make_arena(catalog)
        plan = arena.weight_plan
        methods = [
            B12XFP6MoEMethod(
                experts_prepared=None, weight_plan=plan, arena=arena,
            )
            for _ in range(40)
        ]
        for method in methods:
            for m in catalog:
                x = torch.zeros(m, 64, dtype=torch.bfloat16)
                ids = torch.zeros(m, 2, dtype=torch.int32)
                w = torch.full((m, 2), 0.5, dtype=torch.float32)
                method.apply(x, w, ids)
        assert len(arena) == len(catalog), (
            f"shared arena retained {len(arena)}, expected {len(catalog)}"
        )

    def test_shared_arena_stable_addresses_across_layers(self, mock_fused_moe):
        from b12x.integration.vllm.fp6_serving import B12XFP6MoEMethod

        catalog = (1, 2, 4, 8)
        arena = _make_arena(catalog)
        plan = arena.weight_plan
        m1 = B12XFP6MoEMethod(experts_prepared=None, weight_plan=plan, arena=arena)
        m2 = B12XFP6MoEMethod(experts_prepared=None, weight_plan=plan, arena=arena)
        x = torch.zeros(4, 64, dtype=torch.bfloat16)
        ids = torch.zeros(4, 2, dtype=torch.int32)
        w = torch.full((4, 2), 0.5, dtype=torch.float32)
        m1.apply(x, w, ids)
        p1, s1 = arena.get_or_build(4)
        m2.apply(x, w, ids)
        p2, s2 = arena.get_or_build(4)
        assert p1 is p2 and s1 is s2


# ---------------------------------------------------------------------------
# B12XFP6MoEMethod: public apply() route with capture rejection
# ---------------------------------------------------------------------------


@requires_torch
class TestB12XFP6MoEMethodCaptureRejection:
    """Contract: public apply() rejects uncatalogued capture before allocation."""

    def test_apply_uncatalogued_capture_raises_with_caller_output(
        self, mock_fused_moe, monkeypatch
    ):
        from b12x.integration.vllm.fp6_serving import B12XFP6MoEMethod

        arena = _make_arena((1, 2, 4, 8))
        method = B12XFP6MoEMethod(
            experts_prepared=None, weight_plan=arena.weight_plan, arena=arena,
        )
        x = torch.zeros(7, 64, dtype=torch.bfloat16)
        ids = torch.zeros(7, 2, dtype=torch.int32)
        w = torch.full((7, 2), 0.5, dtype=torch.float32)
        output = torch.zeros(7, 64, dtype=torch.bfloat16)
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        with pytest.raises(RuntimeError, match="uncatalogued M=7"):
            method.apply(x, w, ids, output=output)

    def test_apply_uncatalogued_eager_reuses_bounded_slot(self, mock_fused_moe):
        from b12x.integration.vllm.fp6_serving import B12XFP6MoEMethod

        arena = _make_arena((1, 2, 4, 8))
        method = B12XFP6MoEMethod(
            experts_prepared=None, weight_plan=arena.weight_plan, arena=arena,
        )
        x = torch.zeros(7, 64, dtype=torch.bfloat16)
        ids = torch.zeros(7, 2, dtype=torch.int32)
        w = torch.full((7, 2), 0.5, dtype=torch.float32)
        result = method.apply(x, w, ids)
        assert result is not None and result.shape == (7, 64)
        assert len(arena) == 0 and arena.eager_capacity == 8
        _, first_scratch = arena.get_or_build(7)
        method.apply(x, w, ids)
        _, second_scratch = arena.get_or_build(7)
        assert first_scratch is second_scratch

    def test_apply_no_arena_capture_raises(self, mock_fused_moe, monkeypatch):
        from b12x.integration.vllm.fp6_serving import B12XFP6MoEMethod

        method = B12XFP6MoEMethod(
            experts_prepared=None, weight_plan=types.SimpleNamespace(),
        )
        x = torch.zeros(4, 64, dtype=torch.bfloat16)
        ids = torch.zeros(4, 2, dtype=torch.int32)
        w = torch.full((4, 2), 0.5, dtype=torch.float32)
        output = torch.zeros(4, 64, dtype=torch.bfloat16)
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        with pytest.raises(RuntimeError, match="no scratch arena"):
            method.apply(x, w, ids, output=output)

    def test_apply_catalogued_eager_reuses_scratch(self, mock_fused_moe):
        from b12x.integration.vllm.fp6_serving import B12XFP6MoEMethod

        arena = _make_arena((1, 2, 4, 8))
        method = B12XFP6MoEMethod(
            experts_prepared=None, weight_plan=arena.weight_plan, arena=arena,
        )
        x = torch.zeros(4, 64, dtype=torch.bfloat16)
        ids = torch.zeros(4, 2, dtype=torch.int32)
        w = torch.full((4, 2), 0.5, dtype=torch.float32)
        method.apply(x, w, ids)
        assert len(arena) == 1
        method.apply(x, w, ids)
        assert len(arena) == 1


# ---------------------------------------------------------------------------
# Pointer stability and zeroing
# ---------------------------------------------------------------------------


@requires_torch
class TestPointerStabilityAndZeroing:
    """Contract: catalogued output buffers are stable and zeroed on reuse."""

    def test_output_zeroed_on_reuse(self):
        from b12x.integration.vllm.fp6_serving import CaptureOutputCache

        cache = CaptureOutputCache(
            catalog=(8,), hidden_size=128, dtype=_DTYPE, device=_DEV,
        )
        buf = cache.get(8, _DTYPE, _DEV)
        buf.fill_(3.14)
        buf2 = cache.get(8, _DTYPE, _DEV)
        assert buf2 is buf and buf2.sum() == 0.0

    def test_scratch_stable_address_across_calls(self, mock_fused_moe):
        arena = _make_arena((4, 8))
        p1, s1 = arena.get_or_build(4)
        p2, s2 = arena.get_or_build(8)
        p3, s3 = arena.get_or_build(4)
        p4, s4 = arena.get_or_build(8)
        assert p1 is p3 and s1 is s3 and p2 is p4 and s2 is s4


# ---------------------------------------------------------------------------
# Teardown ordering
# ---------------------------------------------------------------------------


@requires_torch
class TestTeardownOrdering:
    """Contract: clear drops references; cleared arena has no retained tensors."""

    def test_clear_then_empty(self, mock_fused_moe):
        arena = _make_arena((1, 2, 4, 8))
        for m in (1, 2, 4, 8):
            arena.get_or_build(m)
        assert len(arena) == 4
        arena.clear()
        assert len(arena) == 0 and arena.eager_capacity == 0
        arena.get_or_build(3)
        assert len(arena) == 0

    def test_output_cache_clear_then_empty(self):
        from b12x.integration.vllm.fp6_serving import CaptureOutputCache

        cache = CaptureOutputCache(
            catalog=(1, 2, 4, 8), hidden_size=128, dtype=_DTYPE, device=_DEV,
        )
        for m in (1, 2, 4, 8):
            cache.get(m, _DTYPE, _DEV)
        assert len(cache) == 4
        cache.clear()
        assert len(cache) == 0 and cache.eager_capacity == 0
        buf = cache.get(8, _DTYPE, _DEV)
        assert buf is not None and len(cache) == 1


@requires_torch
class TestDenseGraphVisibleReload:
    """Dense reload keeps every graph-visible tensor address stable."""

    @staticmethod
    def _weight(value: float, *, gscale_shape: tuple[int, ...] = (1,)):
        return types.SimpleNamespace(
            scale_storage=torch.full((8,), value, dtype=torch.uint8),
            global_scale=torch.full(gscale_shape, value, dtype=torch.float32),
            fmt="e2m3",
            act_fmt="e4m3",
            out_features=128,
            in_features=256,
        )

    def test_initial_load_installs_gscale(self):
        from b12x.integration.vllm.plugin import _store_graph_visible_dense_state

        layer = types.SimpleNamespace()
        weight = self._weight(1)
        gemm = torch.full((4, 8), 1, dtype=torch.uint8)
        _store_graph_visible_dense_state(layer, weight, gemm)
        assert layer.b12x_fp6_gscale is weight.global_scale
        assert layer.b12x_fp6_gemm_weight is gemm

    def test_reload_copies_gscale_without_changing_pointer(self):
        from b12x.integration.vllm.plugin import _store_graph_visible_dense_state

        layer = types.SimpleNamespace()
        first = self._weight(1)
        _store_graph_visible_dense_state(
            layer, first, torch.full((4, 8), 1, dtype=torch.uint8)
        )
        pointers = {
            name: getattr(layer, name).data_ptr()
            for name in (
                "b12x_fp6_gemm_weight",
                "b12x_fp6_scales",
                "b12x_fp6_gscale",
            )
        }
        second = self._weight(2)
        _store_graph_visible_dense_state(
            layer, second, torch.full((4, 8), 2, dtype=torch.uint8)
        )
        for name, pointer in pointers.items():
            assert getattr(layer, name).data_ptr() == pointer
        assert torch.equal(layer.b12x_fp6_gscale, second.global_scale)

    def test_reload_rejects_gscale_shape_change_before_copying(self):
        from b12x.integration.vllm.plugin import _store_graph_visible_dense_state

        layer = types.SimpleNamespace()
        first = self._weight(1)
        first_gemm = torch.full((4, 8), 1, dtype=torch.uint8)
        _store_graph_visible_dense_state(layer, first, first_gemm)
        second = self._weight(2, gscale_shape=(2,))
        with pytest.raises(RuntimeError, match="cannot preserve b12x_fp6_gscale"):
            _store_graph_visible_dense_state(
                layer, second, torch.full((4, 8), 2, dtype=torch.uint8)
            )
        assert torch.equal(layer.b12x_fp6_gemm_weight, first_gemm)

    def test_reset_serving_caches_clears_plan_cache(
        self, mock_fused_moe, clean_plan_cache
    ):
        from b12x.integration.vllm import fp6_serving
        from b12x.integration.vllm.fp6_serving import reset_serving_caches

        class _WeakObj:
            pass
        obj = _WeakObj()  # hold strong reference
        fp6_serving._WEIGHT_PLAN_CACHE[("test",)] = obj
        assert len(fp6_serving._WEIGHT_PLAN_CACHE) > 0
        reset_serving_caches()
        assert len(fp6_serving._WEIGHT_PLAN_CACHE) == 0
        del obj  # drop strong ref — weak entry already cleared by reset


# ---------------------------------------------------------------------------
# Catalog derivation: capture_sizes_from_config (fp6_serving)
# ---------------------------------------------------------------------------


@requires_torch
class TestCaptureCatalogDerivation:
    """Contract: catalog derived from authoritative vLLM config, not env."""

    def test_object_config_form(self):
        from b12x.integration.vllm.fp6_serving import capture_sizes_from_config

        cfg = types.SimpleNamespace(cudagraph_capture_sizes=[1, 2, 4, 8, 16])
        assert capture_sizes_from_config(cfg) == (1, 2, 4, 8, 16)

    def test_dict_config_form(self):
        from b12x.integration.vllm.fp6_serving import capture_sizes_from_config

        cfg = {"cudagraph_capture_sizes": [1, 2, 4, 8]}
        assert capture_sizes_from_config(cfg) == (1, 2, 4, 8)

    def test_none_means_disabled(self):
        from b12x.integration.vllm.fp6_serving import capture_sizes_from_config

        cfg = types.SimpleNamespace(cudagraph_capture_sizes=None)
        assert capture_sizes_from_config(cfg) == ()

    def test_retains_sizes_above_512(self):
        from b12x.integration.vllm.fp6_serving import capture_sizes_from_config

        cfg = types.SimpleNamespace(cudagraph_capture_sizes=[1, 512, 1024, 2048])
        result = capture_sizes_from_config(cfg)
        assert 1024 in result and 2048 in result and len(result) == 4

    def test_drops_zero_and_negative(self):
        from b12x.integration.vllm.fp6_serving import capture_sizes_from_config

        cfg = types.SimpleNamespace(cudagraph_capture_sizes=[0, -1, 1, 4])
        assert capture_sizes_from_config(cfg) == (1, 4)

    def test_unsupported_type_raises(self):
        from b12x.integration.vllm.fp6_serving import capture_sizes_from_config

        cfg = types.SimpleNamespace(cudagraph_capture_sizes="not-a-list")
        with pytest.raises(RuntimeError, match="unsupported type"):
            capture_sizes_from_config(cfg)


@requires_torch
class TestResolveCaptureCatalogEnvFallback:
    """Contract: _resolve_capture_catalog env fallback and fail-closed."""

    def test_env_fallback_when_vllm_unavailable(self, monkeypatch):
        """B12X_FP6_CAPTURE_SIZES provides catalog when vLLM config is absent."""
        try:
            from b12x.integration.vllm.plugin import _resolve_capture_catalog
        except ImportError:
            pytest.skip("plugin import requires cutlass/vLLM")
        monkeypatch.setenv("B12X_FP6_CAPTURE_SIZES", "1,2,4,8,16")
        result = _resolve_capture_catalog()
        assert result == (1, 2, 4, 8, 16)

    def test_fails_closed_when_no_source(self, monkeypatch):
        """Without vLLM config or env, raises RuntimeError."""
        try:
            from b12x.integration.vllm.plugin import _resolve_capture_catalog
        except ImportError:
            pytest.skip("plugin import requires cutlass/vLLM")
        monkeypatch.delenv("B12X_FP6_CAPTURE_SIZES", raising=False)
        with pytest.raises(RuntimeError, match="cannot resolve capture-size catalog"):
            _resolve_capture_catalog()

    def test_warm_env_does_not_affect_catalog(self, monkeypatch):
        """B12X_MOE_WARM_MS must not appear in capture catalog."""
        try:
            from b12x.integration.vllm.plugin import _resolve_capture_catalog
        except ImportError:
            pytest.skip("plugin import requires cutlass/vLLM")
        monkeypatch.setenv("B12X_MOE_WARM_MS", "1,2,4,8")
        monkeypatch.setenv("B12X_FP6_CAPTURE_SIZES", "16,32")
        result = _resolve_capture_catalog()
        assert result == (16, 32)
        assert 1 not in result and 8 not in result


@requires_torch
class TestWeakPlanCache:
    """Contract: plan cache uses weak refs — plans GC'd when no arena holds them."""

    def test_plan_garbage_collected_when_arena_dropped(self, mock_fused_moe, clean_plan_cache):
        import gc
        from b12x.integration.vllm import fp6_serving
        from b12x.integration.vllm.fp6_serving import get_fp6_moe_weight_plan

        plan = get_fp6_moe_weight_plan(
            source_format="mxfp6_e2m3", activation="silu",
            num_experts=8, hidden_size=512, intermediate_size=256,
        )
        assert len(fp6_serving._WEIGHT_PLAN_CACHE) > 0
        # Drop our strong reference
        del plan
        gc.collect()
        # Weak cache should now be empty (no strong refs from arena)
        assert len(fp6_serving._WEIGHT_PLAN_CACHE) == 0, (
            "plan cache must be weak — plans GC'd when no arena references them"
        )
