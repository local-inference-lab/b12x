from __future__ import annotations

import math
from contextlib import ExitStack
from contextvars import ContextVar

import pytest
import torch
import torch.nn.functional as F

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.sequence import mtp_feedback as mtp
from b12x.sequence.mtp_feedback import _cute_norm
from b12x.preparation import PreparationSession, PreparedCall, require_prepared

from ..conftest import require_sm103_or_sm12x as require_sm120
from ..conftest import require_sm103_or_sm12x


def _make_concat_case(device, hidden=256, capacity=33, position_dtype=torch.int64):
    planned = mtp.plan(
        mtp.Caps(
            device=device,
            max_tokens=capacity,
            hidden_size=hidden,
            streams=1,
            contract="rms_concat",
        )
    )
    (spec,) = planned.scratch_specs()
    tensors = {
        "scratch": torch.empty(spec.shape, dtype=spec.dtype, device=device),
        "token_embedding": _randn((capacity, hidden), device=device),
        "multi_state": _randn((capacity, hidden), device=device),
        "token_norm_weight": 1 + _randn((hidden,), device=device),
        "state_norm_weight": 1 + _randn((hidden,), device=device),
        "combined_fc_weight": _randn(
            (hidden, 2 * hidden), device=device, scale=(2 * hidden) ** -0.5
        ),
        "positions": torch.arange(capacity, dtype=position_dtype, device=device),
        "output": torch.full(
            (capacity, hidden), 7.0, dtype=torch.bfloat16, device=device
        ),
    }
    return planned, tensors


def _concat_reference(binding, eps=1e-6):
    return mtp.reference.rms_concat(
        binding.token_embedding,
        binding.multi_state,
        binding.positions,
        binding.token_norm_weight,
        binding.state_norm_weight,
        binding.combined_fc_weight,
        eps=eps,
    )


def _make_fp8_case(device, hidden=256, streams=4, capacity=17):
    planned = mtp.plan(
        mtp.Caps(
            device=device,
            max_tokens=capacity,
            hidden_size=hidden,
            streams=streams,
            contract="rms_streams_fp8",
        )
    )
    (spec,) = planned.scratch_specs()
    tensors = {
        "scratch": torch.empty(spec.shape, dtype=spec.dtype, device=device),
        "token_embedding": _randn((capacity, hidden), device=device),
        "multi_state": _randn((capacity, streams, hidden), device=device),
        "token_norm_weight": 1 + _randn((hidden,), device=device),
        "state_norm_weight": 1 + _randn((hidden,), device=device),
        "embedding_fc_weight": _randn((hidden, hidden), device=device, scale=2).to(
            torch.float8_e4m3fn
        ),
        "hidden_fc_weight": _randn((hidden, hidden), device=device, scale=2).to(
            torch.float8_e4m3fn
        ),
        "embedding_fc_scale": torch.rand(hidden // 128, hidden // 128, device=device)
        * hidden**-0.5,
        "hidden_fc_scale": torch.rand(hidden // 128, hidden // 128, device=device)
        * hidden**-0.5,
        "positions": torch.arange(capacity, dtype=torch.int64, device=device),
        "output": torch.full(
            (capacity, streams, hidden), 7.0, dtype=torch.bfloat16, device=device
        ),
    }
    return planned, tensors


def _fp8_reference(binding):
    return mtp.reference.rms_streams_fp8(
        binding.token_embedding,
        binding.multi_state,
        binding.positions,
        binding.token_norm_weight,
        binding.state_norm_weight,
        binding.embedding_fc_weight,
        binding.hidden_fc_weight,
        binding.embedding_fc_scale,
        binding.hidden_fc_scale,
    )


def test_rms_streams_fp8_quantizer_rounding_boundaries_and_dynamic_rows():
    import cutlass as c
    from b12x._lib.compiler import run_compiled
    from b12x._lib.utils import current_cuda_stream
    from b12x.sequence.mtp_feedback import _fp8

    device = require_sm103_or_sm12x()
    major, minor = torch.cuda.get_device_capability(device)
    quant, _ = _fp8.compile_aux(128, 1, device.index, f"sm_{major}{minor}a")
    rows = 65537
    source = torch.zeros(rows, 128, device=device, dtype=torch.bfloat16)
    # 2.40625/448 is exactly representable. Multiplication by a rounded
    # reciprocal changes the E4M3 tie at 0.408203125 from 80 to 72.
    source[::3, 0] = 2.40625
    source[::3, 1] = 0.408203125
    source[::3, 2] = -0.408203125
    source[1::3].fill_(1e-20)
    output = torch.empty_like(source, dtype=torch.float8_e4m3fn)
    scales = torch.empty(rows, 1, device=device)
    args = (
        _fp8._pointer(source),
        _fp8._pointer(output, c.Uint32),
        _fp8._pointer(scales, c.Float32),
    )
    with kernel_resolution_guard("FP8 quantizer live rows use retained callable"):
        for live in (1, 17, rows):
            run_compiled(quant, (*args, c.Int32(live), current_cuda_stream()))
            expected_scale = (
                source[:live]
                .float()
                .abs()
                .amax(-1, keepdim=True)
                .clamp_min(1e-10)
                .double()
                / 448
            ).float()
            expected = (
                (source[:live].float() / expected_scale)
                .clamp(-448, 448)
                .to(torch.float8_e4m3fn)
            )
            assert torch.equal(scales[:live], expected_scale)
            assert torch.equal(
                output[:live].view(torch.uint8), expected.view(torch.uint8)
            )
            assert output[0, 1].item() == 80


def test_rms_streams_fp8_rejects_invalid_bindings_and_accepts_int32_positions():
    device = require_sm103_or_sm12x()
    planned, tensors = _make_fp8_case(device)
    for changes, message in (
        ({"output": tensors["multi_state"]}, "overlap"),
        ({"hidden_fc_scale": None}, "hidden_fc_scale"),
        (
            {"hidden_fc_weight": tensors["hidden_fc_weight"].bfloat16()},
            "hidden_fc_weight",
        ),
        ({"positions": tensors["positions"].float()}, "positions"),
    ):
        with pytest.raises(ValueError, match=message):
            mtp.bind(planned, **(tensors | changes), tokens=1)
    with pytest.raises(ValueError, match="capacity"):
        mtp.bind(planned, **tensors, tokens=18)
    binding = mtp.bind(planned, **(tensors | {"positions": tensors["positions"].int()}))
    mtp.run(binding)
    torch.testing.assert_close(
        binding.output, _fp8_reference(binding), rtol=0.02, atol=0.04
    )


def test_rms_streams_fp8_matches_optional_vllm_native_quantization():
    from b12x._lib.scratch_layout import materialize_scratch_view

    device = require_sm103_or_sm12x()
    pytest.importorskip("vllm._custom_ops")
    planned, tensors = _make_fp8_case(device, hidden=5120, streams=4)
    binding = mtp.bind(planned, **tensors)
    mtp.run(binding)
    for path, source in (
        ("embedding", binding.token_normalized),
        ("state", binding.state_normalized),
    ):
        source = source.reshape(-1, 5120)
        quant = torch.empty_like(source, dtype=torch.float8_e4m3fn)
        scales = torch.empty(source.shape[0], 40, device=device)
        torch.ops._C.per_token_group_fp8_quant(
            source,
            quant,
            scales,
            128,
            1e-10,
            -448.0,
            448.0,
            False,
            False,
            False,
        )
        for name, expected in ((f"{path}_quant", quant), (f"{path}_scale", scales)):
            offset, shape, dtype = require_prepared(planned, "sequence.mtp_feedback").layout._backend_plan.layout[name]
            actual = materialize_scratch_view(
                binding.scratch,
                offset_bytes=offset,
                shape=shape,
                dtype=dtype,
            )[0]
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("hidden,streams", [(128, 1), (256, 3), (5120, 4)])
def test_rms_streams_fp8_projection_quantization_and_graph(hidden, streams):
    from b12x._lib.scratch_layout import materialize_scratch_view
    from b12x.sequence.mtp_feedback import _fp8

    device = require_sm103_or_sm12x()
    planned, tensors = _make_fp8_case(device, hidden, streams)
    tensors["token_embedding"][0].fill_(float("nan"))
    tensors["multi_state"].mul_(
        torch.arange(1, streams + 1, device=device).view(1, streams, 1)
    )
    binding = mtp.bind(planned, **tensors, tokens=1)
    mtp.run(binding)
    cached = _fp8.compile_projection.cache_info()
    with kernel_resolution_guard("FP8 feedback retains norm, quantization, projection and add kernels"):
        for live in (1, 4, 17, 0):
            tensors["output"].fill_(7)
            binding = mtp.bind(planned, **tensors, tokens=live)
            mtp.run(binding)
            expected = _fp8_reference(binding)
            torch.testing.assert_close(binding.output, expected, rtol=0.02, atol=0.04)
            if live:
                assert (
                    torch.isfinite(binding.output).all()
                    and binding.output.count_nonzero()
                )
                assert (
                    F.cosine_similarity(
                        binding.output.float().flatten(),
                        expected.float().flatten(),
                        dim=0,
                    )
                    > 0.9999
                )
                assert torch.equal(binding.output.argmax(-1), expected.argmax(-1))
            for path, source in (
                ("embedding", binding.token_normalized),
                ("state", binding.state_normalized),
            ):
                groups = source.float().reshape(
                    live * (streams if path == "state" else 1), hidden // 128, 128
                )
                scale = (groups.abs().amax(-1).clamp_min(1e-10).double() / 448).float()
                quant = (
                    (groups / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
                )
                for name, reference in (
                    (f"{path}_quant", quant.reshape(-1, hidden)),
                    (f"{path}_scale", scale),
                ):
                    offset, shape, dtype = require_prepared(planned, "sequence.mtp_feedback").layout._backend_plan.layout[name]
                    actual = materialize_scratch_view(
                        binding.scratch, offset_bytes=offset, shape=shape, dtype=dtype
                    )[0][: reference.shape[0]]
                    if dtype == torch.float8_e4m3fn:
                        assert torch.equal(
                            actual.view(torch.uint8), reference.view(torch.uint8)
                        )
                    else:
                        torch.testing.assert_close(actual, reference, rtol=1e-6, atol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                mtp.run(binding)
            tensors["hidden_fc_scale"].mul_(0.75)
            tensors["embedding_fc_scale"].mul_(1.125)
            tensors["multi_state"].add_(0.15)
            expected = _fp8_reference(binding)
            tensors["scratch"].fill_(0xFF)
            binding.output.fill_(float("nan"))
            allocated = torch.cuda.memory_allocated(device)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated(device) == allocated
            torch.testing.assert_close(binding.output, expected, rtol=0.02, atol=0.04)
            assert torch.equal(
                tensors["output"][live:], torch.full_like(tensors["output"][live:], 7)
            )
    assert _fp8.compile_projection.cache_info() == cached


@pytest.mark.parametrize("hidden", [256, 4096])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
def test_rms_concat_graph_reuses_capacity_and_reads_mutated_inputs(
    hidden, position_dtype
):
    device = require_sm103_or_sm12x()
    planned, tensors = _make_concat_case(device, hidden, position_dtype=position_dtype)
    tensors["token_embedding"][0].fill_(float("nan"))
    # Zero-position embeddings must be discarded even when poisoned.
    binding = mtp.bind(planned, **tensors, tokens=4)
    mtp.run(binding)
    torch.cuda.synchronize()
    from b12x.sequence.mtp_feedback import _concat

    cache_before = _concat.compile_norm.cache_info()
    projection = require_prepared(planned, "sequence.mtp_feedback").layout._backend_plan.projection
    with kernel_resolution_guard("RMS-concat live counts reuse planned callables"):
        for live in (1, 4, 17, 33, 0):
            tensors["output"].fill_(7)
            binding = mtp.bind(planned, **tensors, tokens=live)
            mtp.run(binding)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                mtp.run(binding)
            tensors["multi_state"].add_(0.1)
            tensors["state_norm_weight"].neg_()
            tensors["combined_fc_weight"].mul_(-0.75)
            if live > 1:
                tensors["positions"][1] = 0
                tensors["token_embedding"][1].fill_(float("nan"))
            expected = _concat_reference(binding)
            tensors["output"][:live].fill_(float("nan"))
            tensors["scratch"].fill_(0xFF)
            before = torch.cuda.memory_allocated(device)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated(device) == before
            assert binding.output.data_ptr() == tensors["output"][:live].data_ptr()
            assert torch.equal(
                tensors["output"][live:], torch.full_like(tensors["output"][live:], 7)
            )
            torch.testing.assert_close(binding.output, expected, rtol=0.02, atol=0.04)
            if live:
                assert torch.isfinite(binding.output).all()
                assert binding.output.count_nonzero() > 0
                assert (
                    F.cosine_similarity(
                        binding.output.float().flatten(),
                        expected.float().flatten(),
                        dim=0,
                    )
                    > 0.9999
                )
                assert torch.equal(binding.output.argmax(-1), expected.argmax(-1))
                assert not binding.token_normalized[0].count_nonzero()
                # Learned weights are ordinary RMS weights, not Gemma (1+w).
                assert binding.state_normalized.count_nonzero() > 0
            del graph
    assert _concat.compile_norm.cache_info() == cache_before
    assert require_prepared(planned, "sequence.mtp_feedback").layout._backend_plan.projection is projection


def test_rms_concat_rejects_aliases_missing_weights_and_capacity_overflow():
    device = require_sm103_or_sm12x()
    planned, tensors = _make_concat_case(device)
    with pytest.raises(ValueError, match="overlap"):
        mtp.bind(planned, **(tensors | {"output": tensors["multi_state"]}), tokens=1)
    with pytest.raises(TypeError, match="combined_fc_weight"):
        mtp.bind(planned, **(tensors | {"combined_fc_weight": None}), tokens=1)
    with pytest.raises(ValueError, match="capacity"):
        mtp.bind(planned, **tensors, tokens=34)
    with pytest.raises(ValueError, match="positions"):
        mtp.bind(planned, **(tensors | {"positions": tensors["positions"].float()}))
    live = {
        name: value[:3]
        if name in {"token_embedding", "multi_state", "positions", "output"}
        else value
        for name, value in tensors.items()
    }
    binding = mtp.bind(planned, **live)
    assert binding.tokens == 3
    mtp.run(binding)
    torch.testing.assert_close(
        binding.output, _concat_reference(binding), rtol=0.02, atol=0.04
    )


_case_resources = ContextVar("mtp_case_resources")


@pytest.fixture(autouse=True)
def _prepared_case_lifetime():
    with ExitStack() as resources:
        token = _case_resources.set(resources)
        try:
            yield
        finally:
            _case_resources.reset(token)


def _randn(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    scale: float = 0.25,
) -> torch.Tensor:
    return (
        torch.randn(shape, dtype=torch.float32, device=device)
        .mul_(scale)
        .to(torch.bfloat16)
        .contiguous()
    )


def _make_case(
    *,
    device: torch.device,
    max_tokens: int = 16,
    tokens: int | None = None,
    streams: int = 4,
    hidden_size: int = 2560,
) -> tuple[mtp.Binding, dict[str, torch.Tensor]]:
    caps = mtp.Caps(
        device=device,
        max_tokens=max_tokens,
        streams=streams,
        hidden_size=hidden_size,
    )
    tensors = {
        "token_embedding": _randn((max_tokens, hidden_size), device=device, scale=0.4),
        "multi_state": _randn(
            (max_tokens, streams, hidden_size), device=device, scale=0.4
        ),
        "token_norm_weight": _randn((hidden_size,), device=device, scale=0.05),
        "state_norm_weight": _randn(
            (streams * hidden_size,), device=device, scale=0.05
        ),
        "embedding_fc_weight": _randn(
            (hidden_size, hidden_size),
            device=device,
            scale=hidden_size**-0.5,
        ),
        "hidden_fc_weight": _randn(
            (hidden_size, hidden_size),
            device=device,
            scale=hidden_size**-0.5,
        ),
        "output": torch.full(
            (max_tokens, streams, hidden_size),
            7.0,
            dtype=torch.bfloat16,
            device=device,
        ),
    }
    declaration = mtp.plan(caps, invocation=mtp.invocation_from_tensors(**tensors))
    original_output = tensors["output"].clone()

    def prepare_call(state):
        (spec,) = state.layout.scratch_specs()
        tensors["scratch"] = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        # Prime the capacity path even when this test's requested live view is empty.
        full = state.bind(**tensors)
        return PreparedCall(
            run=lambda: state.run(full), restore=lambda: tensors["output"].copy_(original_output),
        )

    resources = _case_resources.get()
    session = resources.enter_context(PreparationSession(device=device, autotune=False, compile_workers=2))
    request = declaration.request(name="feedback", prepare_call=prepare_call)
    resources.enter_context(session.prepare((request,)))
    binding = mtp.bind(declaration, **tensors, tokens=tokens)
    return binding, tensors


def _reference(binding: mtp.Binding) -> torch.Tensor:
    return mtp.reference.feedback(
        binding.token_embedding,
        binding.multi_state,
        binding.token_norm_weight,
        binding.state_norm_weight,
        binding.embedding_fc_weight,
        binding.hidden_fc_weight,
    )


def _parameterize_weights(tensors: dict[str, torch.Tensor]) -> None:
    for name in (
        "token_norm_weight",
        "state_norm_weight",
        "embedding_fc_weight",
        "hidden_fc_weight",
    ):
        tensors[name] = torch.nn.Parameter(tensors[name], requires_grad=False)


def test_reference_matches_explicit_transformers_cast_points() -> None:
    torch.manual_seed(19)
    device = torch.device("cpu")
    tokens, streams, hidden = 2, 3, 32
    token_embedding = _randn((tokens, hidden), device=device, scale=0.7)
    multi_state = _randn((tokens, streams, hidden), device=device, scale=0.7)
    token_norm_weight = _randn((hidden,), device=device, scale=0.1)
    state_norm_weight = _randn((streams * hidden,), device=device, scale=0.1)
    embedding_fc_weight = _randn((hidden, hidden), device=device, scale=hidden**-0.5)
    hidden_fc_weight = _randn((hidden, hidden), device=device, scale=hidden**-0.5)

    actual = mtp.reference.feedback(
        token_embedding,
        multi_state,
        token_norm_weight,
        state_norm_weight,
        embedding_fc_weight,
        hidden_fc_weight,
    )
    token_normalized = mtp.reference.gemma_rmsnorm(token_embedding, token_norm_weight)
    state_normalized = mtp.reference.gemma_rmsnorm(
        multi_state.flatten(-2), state_norm_weight
    ).view(tokens, streams, hidden)
    token_path = F.linear(token_normalized, embedding_fc_weight).to(torch.bfloat16)
    state_path = F.linear(state_normalized, hidden_fc_weight).to(torch.bfloat16)
    expected = (state_path + token_path.unsqueeze(1)).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    unrounded = (
        F.linear(state_normalized.float(), hidden_fc_weight.float())
        + F.linear(token_normalized.float(), embedding_fc_weight.float()).unsqueeze(1)
    ).to(torch.bfloat16)
    assert torch.count_nonzero(actual != unrounded).item() > 0


def test_state_norm_uses_one_flattened_stream_group() -> None:
    hidden = 16
    state = (
        torch.stack(
            (
                torch.full((hidden,), 0.25),
                torch.full((hidden,), 4.0),
            )
        )
        .to(torch.bfloat16)[None]
        .contiguous()
    )
    weight = torch.zeros((2 * hidden,), dtype=torch.bfloat16)
    flattened = mtp.reference.gemma_rmsnorm(state.flatten(-2), weight).view_as(state)
    per_stream = torch.stack(
        [
            mtp.reference.gemma_rmsnorm(
                state[:, stream], weight[stream * hidden : (stream + 1) * hidden]
            )
            for stream in range(2)
        ],
        dim=1,
    )

    assert not torch.equal(flattened, per_stream)
    assert flattened[0, 0, 0].abs() < flattened[0, 1, 0].abs()
    torch.testing.assert_close(per_stream[0, 0], per_stream[0, 1], rtol=0, atol=0)




def test_bind_rejects_bad_shapes_dtypes_and_mutable_aliases() -> None:
    device = require_sm120()
    binding, tensors = _make_case(device=device, max_tokens=2)
    planned = binding.plan
    bad = dict(tensors)
    bad["token_norm_weight"] = torch.empty((2559,), dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match="token_norm_weight must have shape"):
        mtp.bind(planned, **bad)

    bad = dict(tensors)
    bad["embedding_fc_weight"] = torch.empty(
        (2560, 2560), dtype=torch.float32, device=device
    )
    with pytest.raises(TypeError, match="embedding_fc_weight must have dtype"):
        mtp.bind(planned, **bad)

    bad = dict(tensors)
    bad["output"] = tensors["multi_state"]
    with pytest.raises(ValueError, match="output.*multi_state"):
        mtp.bind(planned, **bad)

    bad = dict(tensors)
    bad["output"] = (
        tensors["scratch"][: 2 * 4 * 2560 * torch.bfloat16.itemsize]
        .view(torch.bfloat16)
        .view(2, 4, 2560)
    )
    with pytest.raises(ValueError, match="scratch and output"):
        mtp.bind(planned, **bad)

    bad = dict(tensors)
    bad["token_embedding"] = (
        tensors["scratch"][: 2 * 2560 * torch.bfloat16.itemsize]
        .view(torch.bfloat16)
        .view(2, 2560)
    )
    with pytest.raises(ValueError, match="scratch.*token_embedding"):
        mtp.bind(planned, **bad)

    bad = dict(tensors)
    bad["scratch"] = tensors["scratch"][:-1]
    with pytest.raises(ValueError, match="scratch"):
        mtp.bind(planned, **bad)


def test_zero_tokens_is_a_noop_and_live_count_is_capacity_checked() -> None:
    device = require_sm120()
    binding, tensors = _make_case(device=device, max_tokens=3, tokens=0)
    output_before = tensors["output"].clone()

    actual = mtp.run(binding)

    assert actual.shape == (0, 4, 2560)
    torch.testing.assert_close(tensors["output"], output_before, rtol=0, atol=0)
    for tokens in (-1, 4):
        with pytest.raises(ValueError, match="tokens="):
            mtp.bind(binding.plan, **tensors, tokens=tokens)


@pytest.mark.parametrize(("tokens", "max_tokens"), [(1, 1), (3, 3), (17, 17)])
def test_target_s4_h2560_geometry_matches_reference(
    tokens: int, max_tokens: int
) -> None:
    device = require_sm120()
    binding, tensors = _make_case(
        device=device,
        max_tokens=max_tokens,
        tokens=tokens,
        streams=4,
        hidden_size=2560,
    )
    _parameterize_weights(tensors)
    binding = mtp.bind(binding.plan, **tensors, tokens=tokens)
    expected = _reference(binding)
    actual = mtp.run(binding)
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=4e-2)


def test_non_tile_aligned_geometry_preserves_inputs_and_output_tail() -> None:
    device = require_sm120()
    binding, tensors = _make_case(
        device=device,
        max_tokens=19,
        tokens=17,
    )
    read_only_before = {
        name: tensor.clone()
        for name, tensor in tensors.items()
        if name not in {"scratch", "output"}
    }
    output_tail_before = tensors["output"][binding.tokens :].clone()
    expected = _reference(binding)

    actual = mtp.run(binding)
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=3e-2)
    torch.testing.assert_close(
        tensors["output"][binding.tokens :], output_tail_before, rtol=0, atol=0
    )
    for name, before in read_only_before.items():
        torch.testing.assert_close(tensors[name], before, rtol=0, atol=0)


def test_cuda_graph_replay_uses_bound_scratch_and_output() -> None:
    device = require_sm120()
    binding, _ = _make_case(
        device=device,
        max_tokens=16,
        tokens=2,
    )
    mtp.run(binding)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = mtp.run(binding)

    assert captured.data_ptr() == binding.output.data_ptr()
    for _ in range(3):
        binding.token_embedding.copy_(
            torch.randn_like(binding.token_embedding).mul_(0.3)
        )
        binding.multi_state.copy_(torch.randn_like(binding.multi_state).mul_(0.3))
        expected = _reference(binding)
        allocated_before = torch.cuda.memory_allocated(device)
        graph.replay()
        torch.cuda.synchronize(device)
        allocated_after = torch.cuda.memory_allocated(device)

        assert allocated_after == allocated_before
        torch.testing.assert_close(captured, expected, rtol=2e-2, atol=4e-2)


def test_capacity_specialization_is_reused_for_distinct_live_counts_when_frozen() -> None:
    device = require_sm120()
    one_token, tensors = _make_case(device=device, max_tokens=17, tokens=1)
    with kernel_resolution_guard("MTP live rows reuse prepared capacity kernels"):
        for tokens in (1, 17, 0, 3):
            binding = mtp.bind(one_token.plan, **tensors, tokens=tokens)
            expected = _reference(binding)
            actual = mtp.run(binding)
            torch.cuda.synchronize(device)
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=4e-2)


def test_target_geometry_cuda_graph_replay_uses_bound_storage() -> None:
    device = require_sm120()
    binding, tensors = _make_case(
        device=device,
        max_tokens=16,
        tokens=3,
        streams=4,
        hidden_size=2560,
    )
    _parameterize_weights(tensors)
    binding = mtp.bind(binding.plan, **tensors, tokens=3)
    mtp.run(binding)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = mtp.run(binding)
    output_address = captured.data_ptr()
    scratch_address = binding.scratch.data_ptr()

    binding.token_embedding.copy_(torch.randn_like(binding.token_embedding).mul_(0.3))
    binding.multi_state.copy_(torch.randn_like(binding.multi_state).mul_(0.3))
    expected = _reference(binding)
    allocated_before_replay = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize(device)
    allocated_after_replay = torch.cuda.memory_allocated(device)

    assert captured.data_ptr() == output_address == binding.output.data_ptr()
    assert binding.scratch.data_ptr() == scratch_address
    assert allocated_after_replay == allocated_before_replay
    torch.testing.assert_close(captured, expected, rtol=2e-2, atol=4e-2)




@pytest.mark.filterwarnings("ignore:The CUDA Graph is empty.*:UserWarning")
def test_standalone_cute_norm_rejects_cold_cuda_graph_capture() -> None:
    device = require_sm120()
    hidden_size = 2560
    source = _randn((1, hidden_size), device=device, scale=0.4)
    weight = _randn((hidden_size,), device=device, scale=0.05)
    output = torch.empty_like(source)
    _cute_norm.clear_caches()

    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="compiled and warm-run"):
        with torch.cuda.graph(graph):
            _cute_norm.token_norm(
                source,
                weight,
                output,
                eps=1.0e-6,
                hidden_size=hidden_size,
            )


def test_standalone_cute_norm_reuses_binaries_across_live_token_counts_when_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = require_sm120()
    streams, hidden_size = 4, 2560
    compile_targets: list[str] = []
    original_compile = _cute_norm.compile_cute

    def traced_compile(entry: object, *args: object, **kwargs: object) -> object:
        compile_targets.append(type(entry).__name__)
        return original_compile(entry, *args, **kwargs)

    def launch(tokens: int) -> None:
        token_source = _randn((tokens, hidden_size), device=device, scale=0.4)
        state_source = _randn((tokens, streams, hidden_size), device=device, scale=0.4)
        token_weight = _randn((hidden_size,), device=device, scale=0.05)
        state_weight = _randn((streams * hidden_size,), device=device, scale=0.05)
        token_output = torch.empty_like(token_source)
        state_output = torch.empty_like(state_source)

        _cute_norm.token_norm(
            token_source,
            token_weight,
            token_output,
            eps=1.0e-6,
            hidden_size=hidden_size,
        )
        _cute_norm.state_norm(
            state_source,
            state_weight,
            state_output,
            eps=1.0e-6,
            streams=streams,
            hidden_size=hidden_size,
        )
        torch.cuda.synchronize(device)

        expected_token = mtp.reference.gemma_rmsnorm(token_source, token_weight)
        expected_state = mtp.reference.gemma_rmsnorm(
            state_source.flatten(-2), state_weight
        ).view_as(state_source)
        torch.testing.assert_close(token_output, expected_token, rtol=2e-2, atol=4e-2)
        torch.testing.assert_close(state_output, expected_state, rtol=2e-2, atol=4e-2)

    _cute_norm.clear_caches()
    monkeypatch.setattr(_cute_norm, "compile_cute", traced_compile)
    try:
        launch(1)
        compiled_after_first_launch = tuple(compile_targets)
        assert compiled_after_first_launch.count("_TokenNorm") == 1
        assert compiled_after_first_launch.count("_StateNorm") == 1

        with kernel_resolution_guard('MTP normalization live-token cache reuse test'):
            launch(17)
            assert tuple(compile_targets) == compiled_after_first_launch
    finally:
        _cute_norm.clear_caches()


def test_standalone_cute_norm_uses_source_device_when_non_current() -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("two visible CUDA GPUs are required")
    original_device = torch.cuda.current_device()
    target_index = next(
        index for index in range(torch.cuda.device_count()) if index != original_device
    )
    target = torch.device("cuda", target_index)
    tokens, streams, hidden_size = 2, 4, 2560
    token_source = _randn((tokens, hidden_size), device=target, scale=0.4)
    state_source = _randn((tokens, streams, hidden_size), device=target, scale=0.4)
    token_weight = _randn((hidden_size,), device=target, scale=0.05)
    state_weight = _randn((streams * hidden_size,), device=target, scale=0.05)
    token_output = torch.empty_like(token_source)
    state_output = torch.empty_like(state_source)
    _cute_norm.clear_caches()

    _cute_norm.token_norm(
        token_source,
        token_weight,
        token_output,
        eps=1.0e-6,
        hidden_size=hidden_size,
    )
    _cute_norm.state_norm(
        state_source,
        state_weight,
        state_output,
        eps=1.0e-6,
        streams=streams,
        hidden_size=hidden_size,
    )
    torch.cuda.synchronize(target)

    assert torch.cuda.current_device() == original_device
    expected_token = mtp.reference.gemma_rmsnorm(token_source, token_weight)
    expected_state = mtp.reference.gemma_rmsnorm(
        state_source.flatten(-2), state_weight
    ).view_as(state_source)
    torch.testing.assert_close(token_output, expected_token, rtol=2e-2, atol=4e-2)
    torch.testing.assert_close(state_output, expected_state, rtol=2e-2, atol=4e-2)


def test_standalone_cute_norm_correctness_and_graph_stability() -> None:
    device = require_sm120()
    tokens, streams, hidden_size = 4, 4, 2560
    token_source = _randn((tokens, hidden_size), device=device, scale=0.4)
    state_source = _randn((tokens, streams, hidden_size), device=device, scale=0.4)
    token_weight = _randn((hidden_size,), device=device, scale=0.05)
    state_weight = _randn((streams * hidden_size,), device=device, scale=0.05)
    token_output = torch.empty_like(token_source)
    state_output = torch.empty_like(state_source)

    def launch() -> None:
        _cute_norm.token_norm(
            token_source,
            token_weight,
            token_output,
            eps=1.0e-6,
            hidden_size=hidden_size,
        )
        _cute_norm.state_norm(
            state_source,
            state_weight,
            state_output,
            eps=1.0e-6,
            streams=streams,
            hidden_size=hidden_size,
        )

    launch()
    torch.cuda.synchronize(device)
    expected_token = mtp.reference.gemma_rmsnorm(token_source, token_weight)
    expected_state = mtp.reference.gemma_rmsnorm(
        state_source.flatten(-2), state_weight
    ).view_as(state_source)
    torch.testing.assert_close(token_output, expected_token, rtol=2e-2, atol=4e-2)
    torch.testing.assert_close(state_output, expected_state, rtol=2e-2, atol=4e-2)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    token_address = token_output.data_ptr()
    state_address = state_output.data_ptr()

    token_source.copy_(torch.randn_like(token_source).mul_(0.3))
    state_source.copy_(torch.randn_like(state_source).mul_(0.3))
    token_output.fill_(float("nan"))
    state_output.fill_(float("nan"))
    expected_token = mtp.reference.gemma_rmsnorm(token_source, token_weight)
    expected_state = mtp.reference.gemma_rmsnorm(
        state_source.flatten(-2), state_weight
    ).view_as(state_source)
    allocated_before = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize(device)
    allocated_after = torch.cuda.memory_allocated(device)

    assert token_output.data_ptr() == token_address
    assert state_output.data_ptr() == state_address
    assert allocated_after == allocated_before
    torch.testing.assert_close(token_output, expected_token, rtol=2e-2, atol=4e-2)
    torch.testing.assert_close(state_output, expected_state, rtol=2e-2, atol=4e-2)


def test_torch_compile_fullgraph_keeps_feedback_op_opaque() -> None:
    device = require_sm120()
    binding, _ = _make_case(
        device=device,
        max_tokens=16,
        tokens=2,
    )

    def launch() -> torch.Tensor:
        return mtp.run(binding)

    launch()
    compiled = torch.compile(launch, fullgraph=True)
    binding.token_embedding.copy_(torch.randn_like(binding.token_embedding).mul_(0.3))
    binding.multi_state.copy_(torch.randn_like(binding.multi_state).mul_(0.3))
    expected = _reference(binding)
    actual = compiled()
    torch.cuda.synchronize(device)

    assert actual.data_ptr() == binding.output.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=4e-2)


def test_target_torch_compile_accepts_parameter_weights() -> None:
    device = require_sm120()
    binding, tensors = _make_case(
        device=device,
        max_tokens=16,
        tokens=1,
        streams=4,
        hidden_size=2560,
    )
    _parameterize_weights(tensors)
    binding = mtp.bind(binding.plan, **tensors, tokens=1)

    def launch() -> torch.Tensor:
        return mtp.run(binding)

    launch()
    compiled = torch.compile(launch, fullgraph=True)
    binding.token_embedding.copy_(torch.randn_like(binding.token_embedding).mul_(0.3))
    binding.multi_state.copy_(torch.randn_like(binding.multi_state).mul_(0.3))
    expected = _reference(binding)
    actual = compiled()
    torch.cuda.synchronize(device)

    assert actual.data_ptr() == binding.output.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=4e-2)


def test_caps_and_run_validate_contract() -> None:
    device = require_sm120()
    with pytest.raises(ValueError, match="Qwen3.8 CuTe contract"):
        mtp.Caps(device=device, max_tokens=1, hidden_size=63)
    with pytest.raises(ValueError, match="Qwen3.8 CuTe contract"):
        mtp.Caps(device=device, max_tokens=1, streams=3)
    with pytest.raises(TypeError, match="torch.bfloat16"):
        mtp.Caps(device=device, max_tokens=1, dtype=torch.float16)
    binding, _ = _make_case(device=device, max_tokens=16, tokens=1)
    for eps in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="eps must be finite and positive"):
            mtp.run(binding, eps=eps)


@pytest.mark.parametrize("contract", ["rms_concat", "rms_streams_fp8"])
def test_prepared_concat_and_fp8_feedback_fullgraph(contract):
    device = require_sm103_or_sm12x()
    if contract == "rms_concat":
        planned, tensors = _make_concat_case(device, 256)
        reference = _concat_reference
    else:
        planned, tensors = _make_fp8_case(device, 256, 4)
        reference = _fp8_reference
    binding = mtp.bind(planned, **tensors, tokens=3)
    mtp.run(binding)
    with kernel_resolution_guard("prepared MTP fullgraph retains compiled programs"):
        compiled = torch.compile(lambda: mtp.run(binding), fullgraph=True)
        compiled()
        tensors["multi_state"].mul_(0.75)
        expected = reference(binding)
        binding.output.fill_(float("nan"))
        torch.testing.assert_close(compiled(), expected, rtol=0.02, atol=0.04)
