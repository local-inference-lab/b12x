"""Lockstep tests for the NVFP4 split-materialized dispatch policy.

Covers the structural predicate, environment gate, and consumer-visible
clamping, allocation-free caller-arena reuse, and fallback behavior through production dispatch.
"""

from __future__ import annotations

import os
import pytest
import torch

from b12x.moe import fused_moe
from b12x.moe.fused_moe import _impl
from b12x.moe._shared.kernels.reference import compare_to_reference, moe_reference_nvfp4
from b12x.moe.fused_moe._impl import (
    _nvfp4_dynamic_dense_candidate,
    _nvfp4_dynamic_materialized_enabled,
    _nvfp4_materialized_env_refresh,
    _DYNAMIC_NVFP4_MATERIALIZED_ENV,
    _DYNAMIC_WORK_SOURCE_ENV,
)
from tests._reference.helpers import (
    make_tp_moe_fp4_binding,
    prepare_tp_moe_fp4_experts,
    require_b12x,
)
from tests.moe.test_nvfp4_phase_kernels import _bf16_output_bound, _build_domain


def _dense_args(**overrides):
    """Canonical dense-prefill arguments that should match the predicate.
    Only pass fields that the structural predicate accepts (no share_input)."""
    args = dict(
        quant_mode="nvfp4",
        activation="silu",
        routed_rows=4096,
        num_experts=16,
        k=4096,
        n=2048,
        deterministic_output=False,
    )
    args.update(overrides)
    return args


def _enabled_args(**overrides):
    """Arguments for the enabled gate (structural + share_input_across_experts)."""
    args = dict(
        **_dense_args(),
        share_input_across_experts=True,
    )
    args.update(overrides)
    return args


class TestNvfp4SplitPredicate:
    @pytest.fixture(autouse=True)
    def _env_backup(self):
        saved = os.environ.get(_DYNAMIC_NVFP4_MATERIALIZED_ENV)
        yield
        if saved is None:
            os.environ.pop(_DYNAMIC_NVFP4_MATERIALIZED_ENV, None)
        else:
            os.environ[_DYNAMIC_NVFP4_MATERIALIZED_ENV] = saved
        _nvfp4_materialized_env_refresh()

    def test_accepted_dense(self):
        """Reference dense prefill satisfies both candidate and enabled.

        ``_nvfp4_dynamic_materialized_enabled`` auto-enables matching
        structural and ``share_input_across_experts`` arguments when
        ``B12X_NVFP4_DYNAMIC_MATERIALIZED`` is unset, so establish that
        default explicitly instead of relying on the inherited process env.
        """
        os.environ.pop(_DYNAMIC_NVFP4_MATERIALIZED_ENV, None)
        _nvfp4_materialized_env_refresh()
        assert _nvfp4_dynamic_dense_candidate(**_dense_args()) is True
        # Default auto-on: predicate + share_input → enabled when env is unset.
        assert _nvfp4_dynamic_materialized_enabled(**_enabled_args()) is True
        # Explicit toggle-off works.
        os.environ[_DYNAMIC_NVFP4_MATERIALIZED_ENV] = "0"
        _nvfp4_materialized_env_refresh()
        assert _nvfp4_dynamic_materialized_enabled(**_enabled_args()) is False
        # Explicit toggle-on works.
        os.environ[_DYNAMIC_NVFP4_MATERIALIZED_ENV] = "1"
        _nvfp4_materialized_env_refresh()
        assert _nvfp4_dynamic_materialized_enabled(**_enabled_args()) is True

    def test_rejects_non_nvfp4(self):
        # w4a8_mx is a valid quant_mode that is not nvfp4 → predicate False.
        assert _nvfp4_dynamic_dense_candidate(**_dense_args(quant_mode="w4a8_mx")) is False
        # w6a8_mx is also a valid non-nvfp4 mode.
        assert _nvfp4_dynamic_dense_candidate(**_dense_args(quant_mode="w6a8_mx")) is False

    def test_rejects_bad_activation(self):
        assert _nvfp4_dynamic_dense_candidate(**_dense_args(activation="relu2")) is False

    def test_rejects_k_not_divisible_by_128(self):
        assert _nvfp4_dynamic_dense_candidate(**_dense_args(k=2047)) is False
        assert _nvfp4_dynamic_dense_candidate(**_dense_args(k=2048)) is True

    def test_rejects_n_not_divisible_by_128(self):
        assert _nvfp4_dynamic_dense_candidate(**_dense_args(n=700)) is False
        assert _nvfp4_dynamic_dense_candidate(**_dense_args(n=512)) is True

    def test_rejects_deterministic_output(self):
        assert _nvfp4_dynamic_dense_candidate(**_dense_args(deterministic_output=True)) is False

    def test_rejects_tile_64_and_16(self):
        """E=288 domains that drive the tile planner to 64 and 16 are both
        rejected: the split phase kernels are validated only for the M128
        source tile (the 64-row multi-tile path is known-wrong)."""
        big_e = dict(_dense_args(), num_experts=288)
        # routed_rows >= 96*E -> tile 128 (accepted)
        assert _nvfp4_dynamic_dense_candidate(**{**big_e, "routed_rows": 128 * 288}) is True
        # routed_rows in [48*E, 96*E) -> tile 64 (now rejected, not validated)
        assert _nvfp4_dynamic_dense_candidate(**{**big_e, "routed_rows": 80 * 288}) is False
        # routed_rows < 48*E -> tile 16 (rejected)
        assert _nvfp4_dynamic_dense_candidate(**{**big_e, "routed_rows": 10 * 288}) is False

    def test_rejects_ready_queue_work_source(self):
        """Streaming (ready_queue) work source is not supported by the split."""
        saved = os.environ.get(_DYNAMIC_WORK_SOURCE_ENV)
        os.environ[_DYNAMIC_WORK_SOURCE_ENV] = "ready_queue"
        try:
            assert _nvfp4_dynamic_dense_candidate(**_dense_args()) is False
        finally:
            if saved is None:
                os.environ.pop(_DYNAMIC_WORK_SOURCE_ENV, None)
            else:
                os.environ[_DYNAMIC_WORK_SOURCE_ENV] = saved

    def test_rejects_non_shared_input_even_with_env(self):
        """Without share_input_across_experts the enabled gate rejects even with env."""
        os.environ[_DYNAMIC_NVFP4_MATERIALIZED_ENV] = "1"
        _nvfp4_materialized_env_refresh()
        # Predicate is satisfied (share_input is not a predicate check).
        assert _nvfp4_dynamic_dense_candidate(**_dense_args()) is True
        # But the enabled gate demands share_input.
        args_on = _dense_args(share_input_across_experts=False)
        assert _nvfp4_dynamic_materialized_enabled(**args_on) is False

    def test_env_auto_on_for_matching(self):
        """Without any env set, predicate+share_input → auto-on (matching shapes)."""
        os.environ.pop(_DYNAMIC_NVFP4_MATERIALIZED_ENV, None)
        _nvfp4_materialized_env_refresh()
        assert _nvfp4_dynamic_materialized_enabled(**_enabled_args()) is True

    def test_env_off_explicitly(self):
        os.environ[_DYNAMIC_NVFP4_MATERIALIZED_ENV] = "0"
        _nvfp4_materialized_env_refresh()
        assert _nvfp4_dynamic_materialized_enabled(**_enabled_args()) is False

    def test_non_matching_defaults_false(self):
        """Non-matching shapes (no share_input) default False even without env."""
        os.environ.pop(_DYNAMIC_NVFP4_MATERIALIZED_ENV, None)
        _nvfp4_materialized_env_refresh()
        assert _nvfp4_dynamic_materialized_enabled(**_enabled_args(share_input_across_experts=False)) is False


def _prepare_dispatch_experts(domain):
    ones = torch.ones(domain["E"], device="cuda")
    return prepare_tp_moe_fp4_experts(
        a=domain["x"],
        # A scalar scale admits the shared-input split path.
        a1_gscale=torch.ones(1, device="cuda"),
        a2_gscale=ones,
        w1_fp4=domain["w13_packed"].clone(),
        w1_blockscale=domain["w13_sfb"].reshape(domain["E"], -1).clone(),
        w1_alphas=ones,
        w2_fp4=domain["w2_packed"].clone(),
        w2_blockscale=domain["w2_sfb"].reshape(domain["E"], -1).clone(),
        w2_alphas=ones,
    )


def _dispatch_oracle(domain, *, swiglu_limit=None):
    ones = torch.ones(domain["E"], device="cuda")
    return moe_reference_nvfp4(
        domain["x"],
        domain["w13_packed"],
        domain["w13_sfb"].reshape(domain["E"], -1),
        ones,
        domain["w2_packed"],
        domain["w2_sfb"].reshape(domain["E"], -1),
        ones,
        ones,
        ones,
        domain["topk_ids"],
        domain["topk_weights"],
        domain["E"],
        domain["K"],
        domain["n"],
        swiglu_limit=swiglu_limit,
    )


def _assert_dispatch_matches(actual, reference):
    assert torch.isfinite(actual).all()
    metrics = compare_to_reference(actual, reference)
    bound = _bf16_output_bound(reference)
    assert metrics.cos > 0.9999, metrics
    assert metrics.max_abs <= bound, (metrics, bound)
    assert metrics.rmse <= bound, (metrics, bound)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestNvfp4SplitDispatch:
    @pytest.fixture(autouse=True)
    def _isolated_dispatch(self):
        require_b12x()
        try:
            with pytest.MonkeyPatch.context() as patch:
                patch.delenv(_DYNAMIC_NVFP4_MATERIALIZED_ENV, raising=False)
                patch.setenv("B12X_ENABLE_DYNAMIC_DOWN_SCALE", "0")
                patch.setenv("B12X_DYNAMIC_SWAP_AB", "0")
                patch.setenv(_DYNAMIC_WORK_SOURCE_ENV, "materialized_queue")
                patch.setenv("B12X_DYNAMIC_DETERMINISTIC_OUTPUT", "0")
                patch.delenv("B12X_DYNAMIC_TILE_MN", raising=False)
                patch.delenv("B12X_MICRO_DYNAMIC_CUTOVER_PAIRS", raising=False)
                patch.setattr(_impl, "_DYNAMIC_TILE_MN_OVERRIDE", None)
                _impl.clear_tp_moe_caches()
                _nvfp4_materialized_env_refresh()
                yield patch
        finally:
            # Restore the environment before refreshing its module caches,
            # including when binding/compilation or a GPU assertion fails.
            _impl.clear_tp_moe_caches()
            _nvfp4_materialized_env_refresh()

    @pytest.fixture
    def domain(self):
        return _build_domain(E=8, K=256, n=128, m=512, top_k=2, seed=42)

    def test_swiglu_limit_changes_split_output(self, domain):
        experts = _prepare_dispatch_experts(domain)
        clamped_oracle = _dispatch_oracle(domain, swiglu_limit=2.0)
        outputs = []
        for limit in (None, 2.0):
            binding = make_tp_moe_fp4_binding(
                a=domain["x"],
                experts=experts,
                topk_ids=domain["topk_ids"],
                topk_weights=domain["topk_weights"],
                input_scales_static=True,
                fast_math=False,
                swiglu_limit=limit,
            )
            outputs.append(fused_moe.run(binding=binding).clone())
        torch.cuda.synchronize()
        unclamped, clamped = outputs
        _assert_dispatch_matches(unclamped, domain["oracle"])
        _assert_dispatch_matches(clamped, clamped_oracle)
        # A clamp silently ignored by dispatch must not pass on inputs whose
        # gate/up values happen to stay within the limit.
        effect = compare_to_reference(clamped, unclamped)
        assert effect.rmse > _bf16_output_bound(domain["oracle"]), effect

    def test_public_rebind_reuses_planned_scratch_under_frozen_resolution(self, domain):
        from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution

        experts = _prepare_dispatch_experts(domain)
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        plan = fused_moe.plan(
            fused_moe.Caps(
                max_tokens=domain["m"],
                num_topk=domain["top_k"],
                device=domain["x"].device,
                weight_plan=experts.plan,
                quant_mode="nvfp4",
                core_token_counts=(256, domain["m"]),
                route_num_experts=0,
            )
        )
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
        # The caller, as in vLLM, allocates exactly the authoritative SiLU
        # plan's specifications. b12x binds views, not an owning workspace.
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
            for spec in plan.scratch_specs()
        )
        output = torch.empty_like(domain["x"])

        def bind(rows):
            return fused_moe.bind(
                plan,
                scratch=scratch,
                a=domain["x"][:rows],
                experts=experts,
                topk_ids=domain["topk_ids"][:rows],
                topk_weights=domain["topk_weights"][:rows],
                output=output[:rows],
                input_scales_static=True,
                fast_math=False,
            )

        for rows in (512, 256):
            fused_moe.run(binding=bind(rows))
        torch.cuda.synchronize()
        storage = (*scratch, output)
        addresses = tuple(tensor.data_ptr() for tensor in storage)

        freeze_kernel_resolution("NVFP4 public rebind and graph replay use warmed plans")
        try:
            for rows in (512, 256, 512):
                allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                binding = bind(rows)
                actual = fused_moe.run(binding=binding)
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
                assert tuple(tensor.data_ptr() for tensor in storage) == addresses
                assert actual.data_ptr() == output.data_ptr()
                _assert_dispatch_matches(actual, domain["oracle"][:rows])

                graph = torch.cuda.CUDAGraph()
                capture_stream = torch.cuda.Stream()
                capture_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(capture_stream), torch.cuda.graph(graph):
                    # PyTorch allocates graph state on context entry. Measure
                    # b12x's captured call, not the caller's graph construction.
                    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                    fused_moe.run(binding=binding)
                    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
                torch.cuda.current_stream().wait_stream(capture_stream)
                torch.cuda.synchronize()

                for _ in range(3):
                    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                    output.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
                    assert tuple(tensor.data_ptr() for tensor in storage) == addresses
                    _assert_dispatch_matches(output[:rows], domain["oracle"][:rows])
        finally:
            unfreeze_kernel_resolution()

    @pytest.mark.parametrize(
        "policy_env",
        ["B12X_ENABLE_DYNAMIC_DOWN_SCALE", "B12X_DYNAMIC_SWAP_AB"],
        ids=["dynamic-down-scale", "swap-ab"],
    )
    def test_automatic_dispatch_preserves_monolithic_policy(
        self, domain, _isolated_dispatch, policy_env
    ):
        patch = _isolated_dispatch
        patch.setenv(policy_env, "1")
        # One M128 tile per expert makes dynamic down-scaling independent of
        # concurrent route-packing order while preserving split eligibility.
        topk_ids = torch.arange(
            domain["m"] * domain["top_k"], dtype=torch.int32, device="cuda"
        ).reshape(domain["m"], domain["top_k"]).remainder_(domain["E"])
        experts = _prepare_dispatch_experts(domain)
        outputs = []
        for materialized in ("0", None):
            if materialized is None:
                patch.delenv(_DYNAMIC_NVFP4_MATERIALIZED_ENV, raising=False)
            else:
                patch.setenv(_DYNAMIC_NVFP4_MATERIALIZED_ENV, materialized)
            _impl.clear_tp_moe_caches()
            _nvfp4_materialized_env_refresh()
            binding = make_tp_moe_fp4_binding(
                a=domain["x"],
                experts=experts,
                topk_ids=topk_ids,
                topk_weights=domain["topk_weights"],
                input_scales_static=True,
                fast_math=False,
            )
            outputs.append(fused_moe.run(binding=binding).clone())
        torch.cuda.synchronize()
        monolithic, automatic = outputs
        # Dynamic down-scaling changes intermediate quantization; compare to
        # the real monolithic policy, not a static-scale oracle.
        _assert_dispatch_matches(automatic, monolithic)