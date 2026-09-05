"""Plan-owned mHC post/pre schedule selection and split-kernel qualification."""

from __future__ import annotations

import pytest
import torch

from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x._lib.compiler import compile_cache_info
from b12x.norm import mhc
from b12x.norm.mhc import _kernels as mhc_kernels
from b12x.norm.mhc._policy import MHC_POLICY, MhcConfig, MhcQuery
from b12x.policy import MHC, PolicyContext, PolicyMode

from ..conftest import require_b12x as require_sm120
from .test_mhc import _make_inputs, _mhc_pre_reference

_HIDDEN = 4096
_ALLOCATOR_COUNTERS = (
    "allocation.all.allocated",
    "allocation.all.freed",
    "segment.all.allocated",
    "segment.all.freed",
    "num_alloc_retries",
    "num_ooms",
)


def _native_config(
    *,
    post_pre_backend: str = "decode",
    decode_source_splits: int = 0,
    decode_tile_n: int = 0,
    decode_bf16x2: bool = False,
    decode_partials_per_cta: int = 4,
    decode_finalize_threads: int = 0,
    decode_finalize_ctas: int = 1,
    prefill_block_m: int = 0,
    prefill_tile_n: int = 0,
) -> MhcConfig:
    return MhcConfig(
        backend="native",
        native_post_pre_backend=post_pre_backend,
        decode_source_splits=decode_source_splits,
        decode_tile_n=decode_tile_n,
        decode_bf16x2=decode_bf16x2,
        decode_partials_per_cta=decode_partials_per_cta,
        decode_finalize_threads=decode_finalize_threads,
        decode_finalize_ctas=decode_finalize_ctas,
        prefill_block_m=prefill_block_m,
        prefill_tile_n=prefill_tile_n,
        projection_tile_m=16,
        projection_tile_n=8,
        projection_tile_k=256,
        projection_num_stages=1,
        projection_num_m_warps=1,
        projection_num_n_warps=1,
        projection_k_splits=1,
    )


def _allocator_counters(device: torch.device) -> dict[str, int]:
    stats = torch.cuda.memory_stats(device)
    return {key: int(stats[key]) for key in _ALLOCATOR_COUNTERS}


def _binding(
    *,
    tokens: int,
    device: torch.device,
    config: MhcConfig,
) -> mhc.Binding:
    policy = PolicyContext.for_device(
        device,
        mode=PolicyMode.HEURISTIC_ONLY,
    ).with_override(MHC, config)
    plan = mhc.plan(
        mhc.Caps(
            device=device,
            max_tokens=tokens,
            hidden_size=_HIDDEN,
            split_k=64,
        ),
        policy=policy,
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=device)
        for shape, dtype in plan.shapes_and_dtypes()
    )
    return mhc.bind(
        plan,
        scratch=scratch,
        tokens=tokens,
        y=torch.empty((tokens, _HIDDEN), dtype=torch.bfloat16, device=device),
        post=torch.empty((tokens, 4), dtype=torch.float32, device=device),
        comb=torch.empty((tokens, 4, 4), dtype=torch.float32, device=device),
        out=torch.empty((tokens, 4, _HIDDEN), dtype=torch.bfloat16, device=device),
    )


def test_native_schedule_is_part_of_the_validated_policy_config() -> None:
    query = MhcQuery(
        dtype="bfloat16",
        max_tokens=128,
        hidden_size=_HIDDEN,
        split_k=64,
    )
    for config in (
        _native_config(),
        _native_config(decode_source_splits=4, decode_tile_n=6),
        _native_config(decode_source_splits=8, decode_tile_n=6),
        _native_config(
            post_pre_backend="prefill_block_m",
            prefill_block_m=2,
            prefill_tile_n=24,
        ),
    ):
        MHC_POLICY.validate_config(query, config, None)


@pytest.mark.parametrize(
    "config",
    (
        _native_config(decode_source_splits=4, decode_tile_n=0),
        _native_config(decode_source_splits=3, decode_tile_n=6),
        _native_config(decode_bf16x2=True),
        _native_config(
            decode_source_splits=4,
            decode_tile_n=6,
            decode_partials_per_cta=9,
        ),
        _native_config(decode_finalize_threads=0, decode_finalize_ctas=8),
        _native_config(
            post_pre_backend="prefill_block_m",
            decode_source_splits=4,
            decode_tile_n=6,
            prefill_block_m=2,
            prefill_tile_n=24,
        ),
    ),
)
def test_invalid_native_schedules_fail_closed(config: MhcConfig) -> None:
    with pytest.raises(ValueError):
        MHC_POLICY.validate_config(
            MhcQuery(
                dtype="bfloat16",
                max_tokens=128,
                hidden_size=_HIDDEN,
                split_k=64,
            ),
            config,
            None,
        )


def test_planned_decode_launch_does_not_query_the_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int, bool, int]] = []

    def launch(*args: object) -> None:
        calls.append(
            (int(args[-4]), int(args[-3]), bool(args[-2]), int(args[-1]))
        )

    def unexpected_selection(**_kwargs: object) -> tuple[int, int]:
        raise AssertionError("planned decode must not select a runtime schedule")

    monkeypatch.setattr(
        mhc_kernels,
        "_selected_post_pre_decode_split_n",
        unexpected_selection,
    )
    monkeypatch.setattr(
        mhc_kernels,
        "_selected_post_pre_partials_per_cta",
        unexpected_selection,
    )
    monkeypatch.setattr(
        torch.ops.b12x,
        "mhc_post_pre_partial_launch",
        launch,
    )
    config = _native_config(decode_source_splits=4, decode_tile_n=6)
    x = torch.empty((1, _HIDDEN), dtype=torch.bfloat16)
    residual = torch.empty((1, 4, _HIDDEN), dtype=torch.bfloat16)
    mhc_kernels.run_mhc_post_pre_partial(
        x=x,
        residual=residual,
        prev_post=torch.empty((1, 4), dtype=torch.float32),
        prev_comb=torch.empty((1, 4, 4), dtype=torch.float32),
        fn=torch.empty((24, 4 * _HIDDEN), dtype=torch.float32),
        partials=torch.empty((1, 64, 25), dtype=torch.float32),
        out=torch.empty_like(residual),
        compute_gram=True,
        config=config,
    )

    assert calls == [(4, 6, False, 4)]


def test_planned_finalize_launch_does_not_query_the_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []

    def launch(*args: object) -> None:
        calls.append((int(args[-2]), int(args[-1])))

    def unexpected_selection(**_kwargs: object) -> int:
        raise AssertionError("planned finalize must not select a runtime schedule")

    monkeypatch.setattr(
        mhc_kernels,
        "_selected_mhc_decode_finalize_threads",
        unexpected_selection,
    )
    monkeypatch.setattr(torch.ops.b12x, "mhc_finalize_gram_launch", launch)
    mhc_kernels.run_mhc_finalize_gram(
        residual=torch.empty((1, 4, _HIDDEN), dtype=torch.bfloat16),
        partials=torch.empty((1, 64, 25), dtype=torch.float32),
        scale=torch.empty((3,), dtype=torch.float32),
        bias=torch.empty((24,), dtype=torch.float32),
        y=torch.empty((1, _HIDDEN), dtype=torch.bfloat16),
        post=torch.empty((1, 4), dtype=torch.float32),
        comb=torch.empty((1, 4, 4), dtype=torch.float32),
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
        norm_weight=torch.empty((_HIDDEN,), dtype=torch.bfloat16),
        norm_eps=1e-6,
        decode_finalize_threads=128,
        decode_finalize_ctas=8,
    )

    assert calls == [(128, 8)]


def test_planned_decode_reuses_one_compiled_kernel_for_live_rows() -> None:
    device = require_sm120()
    if tuple(torch.cuda.get_device_capability(device)) != (12, 0):
        pytest.skip("SM120 split decode qualification")
    config = _native_config(decode_source_splits=4, decode_tile_n=6)
    policy = PolicyContext.for_device(
        device,
        mode=PolicyMode.HEURISTIC_ONLY,
    ).with_override(MHC, config)
    plan = mhc.plan(
        mhc.Caps(
            device=device,
            max_tokens=128,
            hidden_size=_HIDDEN,
            split_k=64,
        ),
        policy=policy,
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=device)
        for shape, dtype in plan.shapes_and_dtypes()
    )
    generator = torch.Generator(device=device).manual_seed(8_481)
    x = torch.randn(
        (128, _HIDDEN),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    residual = torch.randn(
        (128, 4, _HIDDEN),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    prev_post = torch.randn(
        (128, 4), generator=generator, dtype=torch.float32, device=device
    )
    prev_comb = torch.randn(
        (128, 4, 4), generator=generator, dtype=torch.float32, device=device
    )
    fn = torch.randn(
        (24, 4 * _HIDDEN),
        generator=generator,
        dtype=torch.float32,
        device=device,
    )

    def launch(tokens: int) -> torch.Tensor:
        out = torch.empty(
            (tokens, 4, _HIDDEN), dtype=torch.bfloat16, device=device
        )
        binding = mhc.bind(
            plan,
            scratch=scratch,
            tokens=tokens,
            out=out,
        )
        mhc_kernels.run_mhc_post_pre_partial(
            x=x[:tokens],
            residual=residual[:tokens],
            prev_post=prev_post[:tokens],
            prev_comb=prev_comb[:tokens],
            fn=fn,
            partials=binding.partials,
            out=out,
            compute_gram=True,
            config=plan.config,
        )
        return out

    assert torch.count_nonzero(launch(32)).item() > 0
    torch.cuda.synchronize(device)
    warm_compile_misses = int(compile_cache_info()["compile_misses"])
    freeze_kernel_resolution("mHC live rows must reuse the planned decode kernel")
    try:
        assert torch.count_nonzero(launch(128)).item() > 0
        torch.cuda.synchronize(device)
    finally:
        unfreeze_kernel_resolution()

    assert int(compile_cache_info()["compile_misses"]) == warm_compile_misses


def test_planned_pipeline_reuses_compiled_kernels_for_live_rows() -> None:
    device = require_sm120()
    if tuple(torch.cuda.get_device_capability(device)) != (12, 0):
        pytest.skip("SM120 planned mHC qualification")
    config = _native_config(
        decode_source_splits=4,
        decode_tile_n=6,
        decode_bf16x2=True,
        decode_finalize_threads=128,
        decode_finalize_ctas=8,
    )
    policy = PolicyContext.for_device(
        device,
        mode=PolicyMode.HEURISTIC_ONLY,
    ).with_override(MHC, config)
    plan = mhc.plan(
        mhc.Caps(
            device=device,
            max_tokens=16,
            hidden_size=_HIDDEN,
            split_k=64,
        ),
        policy=policy,
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=device)
        for shape, dtype in plan.shapes_and_dtypes()
    )
    residual, x, fn, scale, bias = _make_inputs(
        tokens=16,
        hidden_size=_HIDDEN,
        seed=9_913,
        device=device,
    )
    _, prev_post, prev_comb = _mhc_pre_reference(
        residual,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )
    prev_post = prev_post.contiguous()
    prev_comb = prev_comb.contiguous()
    norm_weight = torch.ones((_HIDDEN,), dtype=torch.bfloat16, device=device)
    y = torch.empty((16, _HIDDEN), dtype=torch.bfloat16, device=device)
    post = torch.empty((16, 4), dtype=torch.float32, device=device)
    comb = torch.empty((16, 4, 4), dtype=torch.float32, device=device)
    out = torch.empty((16, 4, _HIDDEN), dtype=torch.bfloat16, device=device)

    def launch(tokens: int) -> None:
        binding = mhc.bind(
            plan,
            scratch=scratch,
            tokens=tokens,
            y=y[:tokens],
            post=post[:tokens],
            comb=comb[:tokens],
            out=out[:tokens],
        )
        mhc.run_post_pre(
            x[:tokens],
            residual[:tokens],
            prev_post[:tokens],
            prev_comb[:tokens],
            fn,
            scale,
            bias,
            rms_eps=1e-6,
            hc_eps=1e-6,
            sinkhorn_iters=20,
            norm_weight=norm_weight,
            norm_eps=1e-6,
            binding=binding,
        )

    launch(1)
    torch.cuda.synchronize(device)
    warm_compile_misses = int(compile_cache_info()["compile_misses"])
    freeze_kernel_resolution("mHC live rows must reuse the planned pipeline")
    try:
        launch(16)
        torch.cuda.synchronize(device)
    finally:
        unfreeze_kernel_resolution()

    assert int(compile_cache_info()["compile_misses"]) == warm_compile_misses
    assert torch.count_nonzero(out).item() > 0


@pytest.mark.parametrize("tokens", [32, 64, 128])
def test_split_decode_matches_unsplit_under_graph_replay(tokens: int) -> None:
    """Split reduction changes only documented floating-point association."""
    device = require_sm120()
    if tuple(torch.cuda.get_device_capability(device)) != (12, 0):
        pytest.skip("SM120 split decode qualification")
    residual, x, fn, scale, bias = _make_inputs(
        tokens=tokens,
        hidden_size=_HIDDEN,
        seed=4_242 + tokens,
        device=device,
    )
    _, prev_post, prev_comb = _mhc_pre_reference(
        residual,
        fn,
        scale,
        bias,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )
    prev_post = prev_post.contiguous()
    prev_comb = prev_comb.contiguous()

    def run(config: MhcConfig) -> tuple[torch.Tensor, ...]:
        binding = _binding(tokens=tokens, device=device, config=config)

        def launch() -> None:
            mhc.run_post_pre(
                x,
                residual,
                prev_post,
                prev_comb,
                fn,
                scale,
                bias,
                rms_eps=1e-6,
                hc_eps=1e-6,
                sinkhorn_iters=20,
                binding=binding,
            )

        launch()
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        outputs = (
            binding.y,
            binding.post_buffer,
            binding.comb_buffer,
            binding.out,
        )
        assert all(output is not None for output in outputs)
        concrete_outputs = tuple(output for output in outputs if output is not None)
        pointers = tuple(output.data_ptr() for output in concrete_outputs)
        for output in concrete_outputs:
            output.fill_(float("nan"))
        counters = _allocator_counters(device)
        graph.replay()
        torch.cuda.synchronize(device)
        assert _allocator_counters(device) == counters
        assert tuple(output.data_ptr() for output in concrete_outputs) == pointers
        return tuple(output.clone() for output in concrete_outputs)

    y_split, post_split, comb_split, out_split = run(
        _native_config(decode_source_splits=4, decode_tile_n=6)
    )
    y_unsplit, post_unsplit, comb_unsplit, out_unsplit = run(_native_config())

    assert torch.equal(out_split, out_unsplit)
    torch.testing.assert_close(
        y_split.float(), y_unsplit.float(), rtol=2**-7, atol=2**-14
    )
    torch.testing.assert_close(post_split, post_unsplit, rtol=2**-20, atol=2**-22)
    torch.testing.assert_close(comb_split, comb_unsplit, rtol=2**-20, atol=2**-22)
    assert torch.isfinite(y_split.float()).all()
