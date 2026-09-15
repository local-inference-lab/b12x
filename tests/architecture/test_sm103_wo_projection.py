"""WO backend selection, fixed scratch, and compiler cache contracts."""

from dataclasses import replace

import pytest
import torch

from b12x.gemm import wo_projection as wo
from b12x.gemm.wo_projection import _quant_cute as quant
from b12x.gemm.wo_projection._tuning import TUNING
from b12x.gemm.wo_projection._tuning import WoProjectionConfig, WoProjectionQuery
from b12x._lib.runtime_control import kernel_resolution_guard
from tests.architecture.test_sm103_blockscaled import B300
from types import SimpleNamespace
from b12x.preparation import FrozenMapping


def query(capacity=129):
    return WoProjectionQuery(dtype="bfloat16", max_tokens=capacity, groups=2,
                             group_width=128, rank=64, hidden=256)


def test_preparation_selects_native_backend_and_rejects_incompatible_overrides():
    result = TUNING.configure(query(), device=B300)
    assert result.default == WoProjectionConfig(backend="mxfp8_tcgen05")
    assert len(list(TUNING.iterate(result))) == 1
    legacy = replace(B300, product_name="unknown", compute_capability=(12, 0))
    assert TUNING.configure(query(), device=legacy).default.backend == "mxfp8"
    with pytest.raises(ValueError):
        TUNING.configure(query(), device=B300, override=WoProjectionConfig(backend="mxfp8"))
    for invalid in (replace(query(), hidden=255), replace(query(), max_tokens=2**31),
                    replace(query(), group_width=127), replace(query(), dtype="float16")):
        with pytest.raises(ValueError):
            TUNING.configure(invalid, device=B300)



def _weights():
    from b12x.gemm._shared.wo_mxfp8 import (
        WOProjectionMXFP8Weights, empty_mxfp8_rows_for_dense_gemm,
    )
    return WOProjectionMXFP8Weights(
        empty_mxfp8_rows_for_dense_gemm(64, 128, num_groups=2, device="cpu"),
        empty_mxfp8_rows_for_dense_gemm(256, 128, num_groups=1, device="cpu"),
        2, 128, 64, 256,
    )


def native_plan(monkeypatch, inverse=False, stages=None):
    from b12x.gemm._shared import wo_mxfp8 as impl
    from b12x.gemm.wo_projection._preparation import _PreparedWO
    from tests.architecture._prepared import install_host_state
    monkeypatch.setattr(impl, "_check_gpu_tensor", lambda *a: None)
    caps = wo.Caps(device="cpu", max_tokens=129, groups=2, group_width=128, rank=64, hidden=256)
    invocation = {"operation": "inv_rope" if inverse else "plain", "dynamic_tokens": True}
    if inverse:
        invocation.update(heads_per_group=1, nope_dim=96, rope_dim=32)
    plan = wo.plan(caps, invocation=FrozenMapping(invocation))
    config = TUNING.configure(plan.query, device=B300).default
    scratch_state = impl._materialize_wo_projection_scratch(caps, config=config)
    def quantize(source, out, *args, **kwargs):
        if stages is not None:
            stages.append(("quant", source, out.values))
        for tensor in (out.values, out.scale_rows, out.scale_mma):
            tensor.view(torch.uint8).zero_()
    def inverse_quantize(source, positions, cache, out, **kwargs):
        quantize(source, out)
    def dense(lhs, rhs, *, out, **kwargs):
        if stages is not None:
            stages.append(("gemm", lhs[0], out))
        out.fill_(7)
    quantizers = SimpleNamespace(quantize_a=quantize, quantize_b=quantize,
                                 quantize_a_inv_rope=inverse_quantize)
    state = _PreparedWO(scratch_state, plan.query, SimpleNamespace(run=dense),
                        SimpleNamespace(run=dense), None, quantizers)
    install_host_state(plan, state, config, scratch=scratch_state.scratch_specs())
    return plan



def test_native_binding_preserves_capacity_and_offsets_without_writes(monkeypatch):
    plan = native_plan(monkeypatch)
    assert plan.prepared.selection.config.backend == "mxfp8_tcgen05"
    spec, = plan.scratch_specs()
    scratch = torch.full(spec.shape, 0xA5, dtype=spec.dtype)
    original = scratch.clone()
    source = torch.empty(129, 2, 128, dtype=torch.bfloat16)
    weights = _weights()
    addresses = None
    for m in (1, 4, 8, 16, 127, 128, 129):
        binding = plan.prepared.state.bind( scratch=scratch, source_tgd=source[:m], weights=weights)
        assert binding.expected_m == 129 and binding.backend == "mxfp8_tcgen05"
        ptrs = tuple(t.data_ptr() for t in (binding.x_q.values, binding.x_q.scale_mma,
                                            binding.tmp, binding.tmp_q.values, binding.output))
        if addresses is None:
            addresses = ptrs
        assert addresses == ptrs
        assert binding.output.shape == (m, 256, 1)
    assert torch.equal(scratch, original)
    with pytest.raises(ValueError, match="capacity"):
        plan.prepared.state.bind( scratch=scratch, source_tgd=torch.empty(130, 2, 128, dtype=source.dtype), weights=weights)
    aliased = scratch[:129 * 2 * 128 * 2].view(torch.bfloat16).view(129, 2, 128)
    with pytest.raises(ValueError, match="overlap"):
        plan.prepared.state.bind( scratch=scratch, source_tgd=aliased, weights=weights)


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("external_output", [False, True])
def test_ready_state_runs_bound_quantizers_and_dense_stages(monkeypatch, inverse, external_output):
    # Host stubs verify state wiring and storage only; GPU tests cover the math.
    stages = []
    plan = native_plan(monkeypatch, inverse=inverse, stages=stages)
    spec, = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype)
    source = torch.empty(3, 2, 128, dtype=torch.bfloat16)
    output = torch.empty(3, 256, dtype=torch.bfloat16) if external_output else None
    if inverse:
        binding = plan.prepared.state.bind_inv_rope( scratch=scratch, o=source, positions=torch.arange(3),
                                  cos_sin_cache=torch.ones(8, 32, dtype=torch.bfloat16),
                                  weights=_weights(), heads_per_group=1, nope_dim=96, rope_dim=32,
                                  out=output)
        out = plan.prepared.state.run_inv_rope(binding)
    else:
        binding = plan.prepared.state.bind( scratch=scratch, source_tgd=source, weights=_weights(), out=output)
        out = plan.prepared.state.run(binding)
    assert out.data_ptr() == binding.output.data_ptr() and torch.all(out == 7)
    assert [stage[0] for stage in stages] == ["quant", "gemm", "quant", "gemm"]
    assert stages[0][1] is source and stages[1][2] is binding.tmp
    assert stages[3][2] is binding.output
    if external_output:
        assert out.data_ptr() == output.data_ptr()



def test_quantizer_cache_tracks_device_and_architecture_without_live_rows(monkeypatch):
    quant._get_compiled_wo_quant.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(quant, "current_cuda_stream", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(multi_processor_count=148))
    specs = []
    def compile(*args, compile_spec, **kwargs):
        specs.append(compile_spec)
        from b12x._lib.compile_plan import attach_programs
        return attach_programs(lambda *args: None)
    monkeypatch.setattr(quant, "b12x_compile", compile)
    def get(device, architecture):
        return quant._get_compiled_wo_quant("grouped", 256, 128, torch.bfloat16,
                                          False, 0, 0, 0, torch.int64, torch.bfloat16,
                                          device, architecture)
    first = get(0, "sm_103a")
    assert get(0, "sm_103a") is first
    assert get(1, "sm_103a") is not first and get(0, "sm_120a") is not first
    assert len(specs) == 3 and specs[0] != specs[1] and specs[0] != specs[2]
    try:
        with kernel_resolution_guard("WO quantizer device warmup"):
            assert get(0, "sm_103a") is first
            with pytest.raises(RuntimeError, match="frozen"):
                get(2, "sm_103a")
    finally:
        quant._get_compiled_wo_quant.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="prewarmed"):
        get(0, "sm_103a")
