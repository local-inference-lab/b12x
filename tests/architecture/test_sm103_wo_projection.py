"""WO backend selection, fixed scratch, and compiler cache contracts."""

from dataclasses import replace

import pytest
import torch

from b12x.gemm import wo_projection as wo
from b12x.gemm.wo_projection import _quant_cute as quant
from b12x.gemm.wo_projection._policy import WO_PROJECTION_POLICY as POLICY
from b12x.gemm.wo_projection._policy import WoProjectionConfig, WoProjectionQuery
from b12x._lib.runtime_control import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x.policy import PolicyContext, PolicySource
from tests.architecture.test_sm103_blockscaled import B300
from tests.gemm.test_wo_projection_scratch_bindings import _weights


def query(capacity=129):
    return WoProjectionQuery(dtype="bfloat16", max_tokens=capacity, groups=2,
                             group_width=128, rank=64, hidden=256)


def test_policy_selects_native_backend_and_rejects_incompatible_overrides():
    context = PolicyContext.for_identity(B300)
    result = context.resolve(POLICY, query())
    assert result.config == WoProjectionConfig(backend="mxfp8_tcgen05")
    assert result.source is PolicySource.HEURISTIC and context.profile_id is None
    legacy = replace(B300, product_name="unknown", compute_capability=(12, 0))
    assert PolicyContext.for_identity(legacy).resolve(POLICY, query()).config.backend == "mxfp8"
    with pytest.raises(ValueError, match="backend"):
        context.resolve(POLICY, query(), override=WoProjectionConfig(backend="mxfp8"))
    for invalid in (replace(query(), hidden=255), replace(query(), max_tokens=2**31),
                    replace(query(), group_width=127), replace(query(), dtype="float16")):
        with pytest.raises(ValueError):
            context.resolve(POLICY, invalid)


def native_plan(monkeypatch):
    from b12x.gemm._shared import wo_mxfp8 as impl
    context = PolicyContext.for_identity(B300)
    monkeypatch.setattr(context, "require_device", lambda device: None)
    monkeypatch.setattr(impl, "get_auto_policy", lambda device: context)
    monkeypatch.setattr(impl, "_check_gpu_tensor", lambda *a: None)
    plan = wo.plan(wo.Caps(device="cpu", max_tokens=129, groups=2, group_width=128, rank=64, hidden=256))
    monkeypatch.setattr(context, "resolve", lambda *a, **kw: pytest.fail("binding resolved policy"))
    return plan


def test_native_binding_preserves_capacity_and_offsets_without_writes(monkeypatch):
    plan = native_plan(monkeypatch)
    assert plan.backend == "mxfp8_tcgen05"
    assert plan.policy_resolution.source is PolicySource.HEURISTIC
    spec, = plan.scratch_specs()
    scratch = torch.full(spec.shape, 0xA5, dtype=spec.dtype)
    original = scratch.clone()
    source = torch.empty(129, 2, 128, dtype=torch.bfloat16)
    weights = _weights()
    addresses = None
    for m in (1, 4, 8, 16, 127, 128, 129):
        binding = wo.bind(plan, scratch=scratch, source_tgd=source[:m], weights=weights)
        assert binding.expected_m == 129 and binding.backend == plan.backend
        ptrs = tuple(t.data_ptr() for t in (binding.x_q.values, binding.x_q.scale_mma,
                                            binding.tmp, binding.tmp_q.values, binding.output))
        if addresses is None:
            addresses = ptrs
        assert addresses == ptrs
        assert binding.output.shape == (m, 256, 1)
    assert torch.equal(scratch, original)
    with pytest.raises(ValueError, match="capacity"):
        wo.bind(plan, scratch=scratch, source_tgd=torch.empty(130, 2, 128, dtype=source.dtype), weights=weights)
    aliased = scratch[:129 * 2 * 128 * 2].view(torch.bfloat16).view(129, 2, 128)
    with pytest.raises(ValueError, match="overlap"):
        wo.bind(plan, scratch=scratch, source_tgd=aliased, weights=weights)


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("external_output", [False, True])
def test_public_native_run_uses_bound_tensors(monkeypatch, inverse, external_output):
    from b12x.gemm.wo_projection import _execution
    plan = native_plan(monkeypatch)
    spec, = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype)
    source = torch.empty(3, 3, 128, dtype=torch.bfloat16)[:, :2]
    weights = _weights()
    positions = torch.arange(3)
    cache = torch.ones(8, 32, dtype=torch.bfloat16)
    output = torch.empty(3, 256, dtype=torch.bfloat16) if external_output else None
    if inverse:
        binding = wo.bind_inv_rope(plan, scratch=scratch, o=source, positions=positions,
                                  cos_sin_cache=cache, weights=weights, heads_per_group=1,
                                  nope_dim=96, rope_dim=32, out=output)
    else:
        binding = wo.bind(plan, scratch=scratch, source_tgd=source, weights=weights, out=output)
    calls = []
    def execute(*args):
        calls.append(args)
        args[14].fill_(7)
    monkeypatch.setattr(_execution, "_execute", execute)
    out = (wo.run_inv_rope if inverse else wo.run)(binding=binding, stream=123)
    assert out.data_ptr() == binding.output.data_ptr() and torch.all(out == 7)
    args, = calls
    assert args[0] is source and args[7] is binding.x_q.values and args[10] is binding.tmp
    assert args[-1] == 123 and args[14] is binding.output
    assert args[1] is (positions if inverse else None)
    assert args[2] is (cache if inverse else None)
    if external_output:
        assert out.data_ptr() == output.data_ptr()
        scratch.zero_()
        assert torch.all(out == 7)


def test_quantizer_cache_tracks_device_and_architecture_without_live_rows(monkeypatch):
    quant._get_compiled_wo_quant.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(quant, "current_cuda_stream", lambda: 0)
    specs = []
    def compile(*args, compile_spec, **kwargs):
        specs.append(compile_spec)
        return lambda *args: None
    monkeypatch.setattr(quant, "b12x_compile", compile)
    def get(device, architecture):
        return quant._get_compiled_wo_quant("grouped", 256, 128, torch.bfloat16,
                                          False, 0, 0, 0, torch.int64, torch.bfloat16,
                                          device, architecture)
    first = get(0, "sm_103a")
    assert get(0, "sm_103a") is first
    assert get(1, "sm_103a") is not first and get(0, "sm_120a") is not first
    assert len(specs) == 3 and specs[0] != specs[1] and specs[0] != specs[2]
    freeze_kernel_resolution("WO quantizer device warmup")
    try:
        assert get(0, "sm_103a") is first
        with pytest.raises(RuntimeError, match="frozen"):
            get(2, "sm_103a")
    finally:
        unfreeze_kernel_resolution()
        quant._get_compiled_wo_quant.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="prewarmed"):
        get(0, "sm_103a")


def test_generator_and_metadata_cover_native_wo():
    from b12x.policy.generation.providers.gemm import WoProjectionGenerator
    from scripts.qualify_sm103 import SUITES
    assert "sm103a" in wo.META.archs
    assert WoProjectionGenerator().config_schema_version == POLICY.config_schema_version == 2
    assert "tests/gemm/test_sm103_wo_projection.py" in SUITES["wo_projection"]


@pytest.mark.parametrize("native", [False, True])
def test_prewarm_owns_capacity_schedule_and_rejects_capture_or_freeze(monkeypatch, native):
    from b12x.gemm.wo_projection import api
    plan = native_plan(monkeypatch)
    if not native:
        plan = replace(plan, backend="mxfp8")
    calls = []
    monkeypatch.setattr(api, "run_inv_rope", lambda *, binding: calls.append(binding))
    kwargs = dict(weights=_weights(), cos_sin_cache=torch.ones(8, 32),
                  heads_per_group=1, nope_dim=96, rope_dim=32)
    wo.prewarm_inv_rope(plan, **kwargs)
    assert [binding.o.shape[0] for binding in calls] == ([1] if native else [*range(1, 17), 129])
    assert all(binding.expected_m == 129 for binding in calls)
    assert len({binding.x_q.values.data_ptr() for binding in calls}) == 1
    assert all(binding.output.untyped_storage().data_ptr() != binding.tmp.untyped_storage().data_ptr()
               for binding in calls)
    freeze_kernel_resolution("prewarm must precede freeze")
    try:
        with pytest.raises(RuntimeError, match="frozen"):
            wo.prewarm_inv_rope(plan, **kwargs)
    finally:
        unfreeze_kernel_resolution()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="eager"):
        wo.prewarm_inv_rope(plan, **kwargs)


def test_native_custom_op_accepts_disjoint_views_of_one_arena(monkeypatch):
    from b12x.gemm.blockscaled import _sm103 as gemm
    plan = native_plan(monkeypatch)
    spec, = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype)
    source = torch.empty(3, 2, 128, dtype=torch.bfloat16)
    binding = wo.bind(plan, scratch=scratch, source_tgd=source, weights=_weights())
    stages = []
    def quantize(source, values, rows, mma, **kwargs):
        stages.append(("quant", source, values))
        for tensor in (values, rows, mma):
            tensor.view(torch.uint8).zero_()
    def execute(lhs, rhs, out, **kwargs):
        stages.append(("gemm", lhs[0], out))
        out.fill_(7)
    monkeypatch.setattr(quant, "quantize_wo_grouped_rows_cute", quantize)
    monkeypatch.setattr(quant, "quantize_wo_group_major_rows_cute", quantize)
    monkeypatch.setattr(gemm, "execute", execute)
    out = wo.run(binding=binding)
    assert torch.all(out == 7) and out.data_ptr() == binding.output.data_ptr()
    assert [stage[0] for stage in stages] == ["quant", "gemm", "quant", "gemm"]
    assert stages[1][2] is binding.tmp and stages[3][2] is binding.output


def test_embedded_wo_resolution_overrides_and_invalid_matching_data():
    from b12x.policy import EMBEDDED_REGISTRY, FrozenMapping, ProfileRegistry
    original = EMBEDDED_REGISTRY.get("nvidia.gb10.48sm")
    device = original.targets[0]
    context = PolicyContext.for_identity(device)
    assert context.resolve(POLICY, query()).source is PolicySource.PREPLANNED
    chosen = WoProjectionConfig(backend="mxfp8")
    override = context.resolve(POLICY, query(), override=chosen)
    assert override.source is PolicySource.OVERRIDE and override.config is chosen
    component = original.component(POLICY.component_id)
    invalid = replace(component, planner=replace(component.planner, config=FrozenMapping({"backend": "mxfp8_tcgen05"})))
    profile = replace(original, components=tuple(invalid if c.component_id == POLICY.component_id else c for c in original.components))
    registry = ProfileRegistry()
    registry.register(profile)
    registry.freeze()
    with pytest.raises(ValueError, match="invalid preplanned") as error:
        PolicyContext.for_identity(device, registry=registry).resolve(POLICY, query())
    assert "backend" in str(error.value.__cause__)
