from __future__ import annotations

import pytest
import torch

from b12x.gemm import bf16_vocab_projection as projection
from b12x.gemm.bf16_vocab_projection._tuning import TUNING
from b12x.preparation import DeviceIdentity, PreparationSession, PreparedCall

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def test_unknown_device_default_uses_selected_torch_backend() -> None:
    query = projection.Bf16VocabProjectionQuery(
        dtype="bfloat16", max_tokens=1, in_features=2_560, out_features=248_320,
    )
    device = DeviceIdentity(
        vendor="nvidia", compute_capability=(9, 0), sm_count=120,
        product_name="Synthetic GPU",
    )

    config = TUNING.configure(query, device=device).default

    assert config.backend == "torch"


@pytest.mark.parametrize("capacity", [1, 8, 17])
def test_sm103_policy_selects_cute_for_planned_capacity(capacity):
    from b12x.policy import PolicyContext, PolicySource

    device = DeviceIdentity(
        vendor="nvidia",
        product_name="Synthetic B300",
        compute_capability=(10, 3),
        sm_count=148,
    )
    query = Bf16VocabProjectionQuery(
        dtype="bfloat16", max_tokens=capacity, in_features=4096, out_features=154880
    )
    result = PolicyContext.for_identity(device).resolve(
        BF16_VOCAB_PROJECTION_POLICY, query
    )
    assert result.source is PolicySource.HEURISTIC
    assert result.config.backend == "cute"
    BF16_VOCAB_PROJECTION_POLICY.validate_config(query, result.config, device)
    with pytest.raises(ValueError, match="SM103"):
        BF16_VOCAB_PROJECTION_POLICY.validate_config(
            query,
            Bf16VocabProjectionConfig(
                backend="triton", algorithm="row", block_k=4096, num_warps=8
            ),
            device,
        )


def _cute_plan(n, k, capacity):
    from b12x.gemm import bf16_vocab_projection as projection
    from b12x.gemm.bf16_vocab_projection._policy import CUTE_CONFIG
    from b12x.policy import PolicyContext, PolicyMode

    device = torch.device("cuda", torch.cuda.current_device())
    context = PolicyContext.for_device(device, mode=PolicyMode.HEURISTIC_ONLY)
    return projection.plan(
        projection.Caps(
            device=device, max_tokens=capacity, in_features=k, out_features=n
        ),
        policy=context.with_override(
            BF16_VOCAB_PROJECTION, Bf16VocabProjectionConfig(**CUTE_CONFIG)
        ),
    )


def test_sm103_generator_qualifies_only_the_cute_candidate():
    from types import SimpleNamespace

    from b12x.policy.generation.providers.gemm import (
        _Bf16VocabProjectionSession,
        _bf16_vocab_projection_cases,
    )

    context = SimpleNamespace(device=SimpleNamespace(compute_capability=(10, 3)))
    session = _Bf16VocabProjectionSession(context)
    candidates = session.candidates(_bf16_vocab_projection_cases()[0])
    assert len(candidates) == 1
    assert candidates[0].config["backend"] == "cute"


@cuda_required
@pytest.mark.parametrize("n,k", [(97, 259), (4096, 5120), (154880, 4096)])
def test_cute_vocabulary_reuses_callable_and_storage_across_live_rows(n, k):
    from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
    from b12x.gemm import bf16_vocab_projection as projection
    from b12x.gemm.bf16_vocab_projection import _cute

    torch.manual_seed(1137)
    planned = _cute_plan(n, k, 17)
    source = torch.randn(17, k, device="cuda", dtype=torch.bfloat16) * 0.25
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.125
    projection.run(projection.bind(planned, source=source[:1], weight=weight))
    before = _cute.compile_kernel.cache_info().misses
    freeze_kernel_resolution("vocabulary capacity plan covers every live row count")
    try:
        for rows in (1, 4, 8, 9, 17):
            binding = projection.bind(planned, source=source[:rows], weight=weight)
            actual = projection.run(binding)
            expected = torch.nn.functional.linear(
                source[:rows].float(), weight.float()
            ).bfloat16()
            torch.testing.assert_close(
                actual.float(), expected.float(), rtol=1e-2, atol=2e-3
            )
            assert torch.equal(actual.argmax(-1), expected.argmax(-1))
            assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
            assert actual.data_ptr() == planned.output.data_ptr()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                projection.run(binding)
            source[:rows].neg_()
            actual.fill_(float("nan"))
            allocated = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            torch.testing.assert_close(
                actual.float(), -expected.float(), rtol=1e-2, atol=2e-3
            )
            source[:rows].neg_()
        assert _cute.compile_kernel.cache_info().misses == before
        assert planned.kernel is _cute.compile_kernel(
            n, k, planned.caps.device.index, planned.architecture
        )
    finally:
        unfreeze_kernel_resolution()


@cuda_required
def test_cute_vocabulary_caller_output_and_inductor():
    from b12x.gemm import bf16_vocab_projection as projection

    planned = _cute_plan(97, 128, 9)
    source = torch.randn(9, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(97, 128, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(9, 97, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="overlap"):
        projection.bind(
            planned,
            source=source,
            weight=weight,
            out=weight.flatten()[: 9 * 97].view(9, 97),
        )
    with pytest.raises(ValueError, match="rows"):
        projection.bind(planned, source=source[:0], weight=weight)

    def run(x, out):
        return projection.run(
            projection.bind(planned, source=x, weight=weight, out=out)
        )

    run(source, output)
    compiled = torch.compile(run, fullgraph=True, dynamic=True)
    for rows in (1, 4, 9):
        actual = compiled(source[:rows], output[:rows])
        expected = torch.nn.functional.linear(
            source[:rows].float(), weight.float()
        ).bfloat16()
        assert actual.data_ptr() == output.data_ptr()
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=1e-2, atol=2e-3
        )


@cuda_required
@pytest.mark.parametrize("backend", ["cute", "triton_row", "triton_loop"])
def test_vocabulary_weight_rows_past_int32_element_offset(backend):
    from b12x.gemm import bf16_vocab_projection as projection
    from b12x.policy import PolicyContext, PolicyMode

    k = 4096
    n = 2**31 // k + 9
    planned = _cute_plan(n, k, 1)
    if backend.startswith("triton"):
        if torch.cuda.get_device_capability() == (10, 3):
            pytest.skip("SM103 uses CuTe vocabulary projection")
        context = PolicyContext.for_device("cuda", mode=PolicyMode.HEURISTIC_ONLY)
        planned = projection.plan(
            planned.caps,
            policy=context.with_override(
                BF16_VOCAB_PROJECTION,
                Bf16VocabProjectionConfig(
                    backend="triton",
                    algorithm=backend.removeprefix("triton_"),
                    block_k=k if backend == "triton_row" else 256,
                    num_warps=8,
                ),
            ),
        )
    weight = torch.empty(n, k, device="cuda", dtype=torch.bfloat16)
    weight[-9:].fill_(1)
    source = torch.ones(1, k, device="cuda", dtype=torch.bfloat16)
    actual = projection.run(projection.bind(planned, source=source, weight=weight))
    torch.testing.assert_close(
        actual[:, -9:], torch.full_like(actual[:, -9:], k), rtol=0, atol=0
    )


@cuda_required
def test_prepared_projection_matches_reference_and_replays_graph() -> None:
    torch.manual_seed(4)
    device = torch.device("cuda")
    source = torch.randn(1, 256, device=device, dtype=torch.bfloat16)
    weight = torch.randn(4_096, 256, device=device, dtype=torch.bfloat16)
    declaration = projection.plan(
        projection.Caps(
            device=device, max_tokens=1, in_features=256, out_features=4_096,
        ),
        override=projection.Bf16VocabProjectionConfig(
            backend="triton", algorithm="row", block_k=256, num_warps=8,
        ),
    )

    def prepare_call(state):
        return PreparedCall(run=lambda: state.run(source, weight))

    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare((declaration.request(
            name="vocab", prepare_call=prepare_call,
        ),))
        binding = projection.bind(
            declaration, source=source, weight=weight,
        )
        expected = torch.nn.functional.linear(source, weight)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            actual = projection.run(binding)
        actual.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(device)

    torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-2, atol=1e-2)
