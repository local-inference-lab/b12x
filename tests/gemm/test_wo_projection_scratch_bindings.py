from __future__ import annotations

import pytest
import torch

import b12x.gemm._shared.wo_mxfp8 as wo_impl
from b12x.gemm._shared.wo_mxfp8 import WOProjectionBinding, WOProjectionInvRopeBinding, WOProjectionScratchCaps, empty_mxfp8_rows_for_dense_gemm, plan_wo_projection_scratch
from b12x.gemm._shared.wo_mxfp8 import WOProjectionMXFP8Weights


def _weights(
    *,
    groups: int = 2,
    group_width: int = 128,
    rank: int = 64,
    hidden: int = 256,
) -> WOProjectionMXFP8Weights:
    return WOProjectionMXFP8Weights(
        wo_a=empty_mxfp8_rows_for_dense_gemm(
            rank,
            group_width,
            num_groups=groups,
            device="cpu",
        ),
        wo_b=empty_mxfp8_rows_for_dense_gemm(
            hidden,
            rank * groups,
            num_groups=1,
            device="cpu",
        ),
        groups=groups,
        group_width=group_width,
        rank=rank,
        hidden=hidden,
    )


def _plan():
    return plan_wo_projection_scratch(
        WOProjectionScratchCaps(
            device="cpu",
            max_tokens=4,
            groups=2,
            group_width=128,
            rank=64,
            hidden=256,
        )
    )


def test_wo_projection_scratch_plan_exposes_one_component_scratch_spec() -> None:
    plan = _plan()

    specs = plan.scratch_specs()
    assert len(specs) == 1
    assert specs[0].name == "wo_projection.scratch"
    assert specs[0].dtype == torch.uint8
    assert specs[0].shape == plan.shapes_and_dtypes()[0][0]
    assert specs[0].nbytes == specs[0].shape[0]
    assert specs[0].nbytes == plan.layout.nbytes


def test_wo_projection_scratch_plan_binds_live_shape(monkeypatch) -> None:
    monkeypatch.setattr(wo_impl, "_check_gpu_tensor", lambda *args, **kwargs: None)
    plan = _plan()
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    source = torch.empty((3, 2, 128), dtype=torch.bfloat16)
    weights = _weights()

    binding = plan.bind(scratch=scratch, source_tgd=source, weights=weights)

    assert isinstance(binding, WOProjectionBinding)
    assert binding.source_tgd is source
    assert binding.weights is weights
    assert not hasattr(binding, "workspace")
    assert binding.x_q.values.shape == (3, 128, 2)
    assert binding.tmp.shape == (3, 64, 2)
    assert binding.output.shape == (3, 256, 1)


def test_wo_projection_plan_binding_maps_scratch_views(monkeypatch) -> None:
    monkeypatch.setattr(wo_impl, "_check_gpu_tensor", lambda *args, **kwargs: None)
    plan = _plan()
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    source = torch.empty((3, 2, 128), dtype=torch.bfloat16)
    weights = _weights()

    binding = plan.bind(scratch=scratch, source_tgd=source, weights=weights)

    assert isinstance(binding, WOProjectionBinding)
    assert not hasattr(binding, "workspace")
    assert binding.x_q.values.data_ptr() == scratch.data_ptr()
    assert binding.tmp.shape == (3, 64, 2)
    assert binding.tmp_q.values.shape == (3, 128)
    assert binding.output.shape == (3, 256, 1)
    assert binding.source_tgd is source
    assert binding.weights is weights


def test_wo_projection_binding_supplies_runtime_tensors(monkeypatch) -> None:
    monkeypatch.setattr(wo_impl, "_check_gpu_tensor", lambda *args, **kwargs: None)
    plan = _plan()
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    source = torch.empty((3, 2, 128), dtype=torch.bfloat16)
    weights = _weights()
    binding = plan.bind(scratch=scratch, source_tgd=source, weights=weights)
    calls = {}

    def fake_quantize_a(source_tgd, *, out):
        calls["source_tgd"] = source_tgd
        calls["x_q_out"] = out
        return out

    def fake_wo_a(x_q, wo_a, *, out, expected_m=None, **kwargs):
        calls["x_q"] = x_q
        calls["wo_a"] = wo_a
        calls["tmp_out"] = out
        return out

    def fake_wo_b_fused(tmp, wo_b, *, out, expected_m=None, **kwargs):
        # Small-M bindings route WO-B through the fused-quant GEMM; the bound
        # tmp_q scratch is intentionally unused there.
        calls["tmp"] = tmp
        calls["wo_b"] = wo_b
        calls["output_out"] = out
        out.zero_()
        return out

    monkeypatch.setattr(wo_impl, "quantize_wo_a_input_mxfp8", fake_quantize_a)
    monkeypatch.setattr(wo_impl, "wo_a_dense_gemm_mxfp8", fake_wo_a)
    monkeypatch.setattr(
        wo_impl, "wo_b_dense_gemm_fused_quant_mxfp8", fake_wo_b_fused
    )

    out = wo_impl.wo_projection_mxfp8(binding=binding)

    assert calls["source_tgd"] is source
    assert calls["x_q_out"] is binding.x_q
    assert calls["tmp_out"] is binding.tmp
    assert calls["tmp"] is binding.tmp
    assert calls["output_out"] is binding.output
    assert out.shape == (3, 256)


def test_wo_projection_inv_rope_binding_supplies_runtime_tensors(monkeypatch) -> None:
    monkeypatch.setattr(wo_impl, "_check_gpu_tensor", lambda *args, **kwargs: None)
    plan = _plan()
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    o = torch.empty((3, 2, 128), dtype=torch.bfloat16)
    positions = torch.empty((3,), dtype=torch.int64)
    cos_sin_cache = torch.empty((16, 32), dtype=torch.bfloat16)
    weights = _weights()

    binding = plan.bind_inv_rope(
        scratch=scratch,
        o=o,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        weights=weights,
        heads_per_group=1,
        nope_dim=96,
        rope_dim=32,
        return_3d=True,
    )
    calls = {}

    def fake_quantize(
        o_arg,
        positions_arg,
        cos_sin_cache_arg,
        out_values,
        out_scale_rows,
        out_scale_mma,
        tokens,
        groups,
        heads_per_group,
        group_width,
        head_dim,
        nope_dim,
        rope_dim,
        clear_output=None,
    ):
        calls["o"] = o_arg
        calls["positions"] = positions_arg
        calls["cos_sin_cache"] = cos_sin_cache_arg
        calls["x_q_values"] = out_values
        calls["x_q_scale_rows"] = out_scale_rows
        calls["x_q_scale_mma"] = out_scale_mma
        calls["groups"] = groups
        calls["group_width"] = group_width
        calls["heads_per_group"] = heads_per_group
        calls["head_dim"] = head_dim
        calls["nope_dim"] = nope_dim
        calls["rope_dim"] = rope_dim
        calls["clear_output"] = clear_output

    def fake_wo_a(x_q, wo_a, *, out, expected_m=None, stream=None, **kwargs):
        calls["x_q"] = x_q
        calls["wo_a"] = wo_a
        calls["tmp_out"] = out
        calls["expected_m"] = expected_m
        calls["wo_a_stream"] = stream
        return out

    def fake_wo_b(tmp, wo_b, *, out, expected_m=None, stream=None, **kwargs):
        calls["tmp"] = tmp
        calls["wo_b"] = wo_b
        calls["output_out"] = out
        calls["wo_b_stream"] = stream
        return out

    monkeypatch.setattr(wo_impl, "_run_wo_a_quant_kernel", fake_quantize)
    monkeypatch.setattr(wo_impl, "wo_a_dense_gemm_mxfp8", fake_wo_a)
    monkeypatch.setattr(wo_impl, "wo_b_dense_gemm_fused_quant_mxfp8", fake_wo_b)

    out = wo_impl.wo_projection_inv_rope_mxfp8(binding=binding, stream=123)

    assert isinstance(binding, WOProjectionInvRopeBinding)
    assert not hasattr(binding, "workspace")
    assert calls["o"] is o
    assert calls["positions"] is positions
    assert calls["cos_sin_cache"] is cos_sin_cache
    assert calls["x_q_values"] is binding.x_q.values
    assert calls["x_q_scale_rows"] is binding.x_q.scale_rows
    assert calls["x_q_scale_mma"] is binding.x_q.scale_mma
    assert calls["x_q"] is binding.x_q
    assert calls["wo_a"] is weights.wo_a
    assert calls["tmp_out"] is binding.tmp
    assert calls["tmp"] is binding.tmp
    assert calls["wo_b"] is weights.wo_b
    assert calls["output_out"] is binding.output
    assert calls["groups"] == 2
    assert calls["group_width"] == 128
    assert calls["heads_per_group"] == 1
    assert calls["head_dim"] == 128
    assert calls["nope_dim"] == 96
    assert calls["rope_dim"] == 32
    assert calls["expected_m"] == 3
    assert calls["clear_output"] is binding.output
    assert calls["wo_a_stream"] == 123
    assert calls["wo_b_stream"] == 123
    assert out.data_ptr() == binding.output.data_ptr()
    assert out.shape == (3, 256, 1)


def test_wo_projection_binding_owns_runtime_tensors(monkeypatch) -> None:
    monkeypatch.setattr(wo_impl, "_check_gpu_tensor", lambda *args, **kwargs: None)
    plan = _plan()
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    source = torch.empty((3, 2, 128), dtype=torch.bfloat16)
    weights = _weights()
    binding = plan.bind(scratch=scratch, source_tgd=source, weights=weights)

    with pytest.raises(ValueError, match="binding owns source_tgd"):
        wo_impl.wo_projection_mxfp8(
            source,
            weights,
            binding=binding,
        )


@pytest.mark.parametrize(
    "tokens,sm_count,dsv4,quantized,preclear",
    [
        (8, 48, True, False, True),
        (9, 48, True, True, False),
        (15, 64, True, True, False),
        (16, 48, False, True, False),
        (17, 48, True, False, False),
        (9, 48, False, False, False),
        (15, 65, True, False, False),
        (16, 188, True, False, False),
    ],
)
def test_inv_rope_route_preserves_spark_boundaries(
    tokens, sm_count, dsv4, quantized, preclear
):
    """Keep the Spark-only exact-B16 and DSV4 B9–15 routes distinct."""
    from types import SimpleNamespace

    weights = SimpleNamespace(
        groups=4, group_width=512, rank=1024, hidden=4096 if dsv4 else 2048
    )
    route = wo_impl._select_wo_inv_rope_route(
        tokens=tokens,
        sm_count=sm_count,
        weights=weights,
        heads_per_group=1,
        nope_dim=448,
        rope_dim=64,
    )
    assert route.quantized_intermediate is quantized
    assert route.atomic_output_precleared is preclear
