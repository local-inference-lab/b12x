"""Planned block-FP8 policy, compiler identity, and storage contracts."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from b12x._lib.quant import mxfp8_rows as quant
from b12x._lib.runtime_control import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x.gemm import block_fp8_linear as linear
from b12x.gemm.block_fp8_linear._policy import BLOCK_FP8_LINEAR_POLICY as POLICY
from b12x.gemm.block_fp8_linear._policy import BlockFp8LinearConfig, BlockFp8LinearQuery
from b12x.policy import PolicyContext, PolicySource
from tests.architecture.test_sm103_blockscaled import B300


@pytest.mark.parametrize("capacity", [1, 8, 129])
def test_sm103_policy_is_unmeasured_and_preserves_capacity(capacity):
    query = BlockFp8LinearQuery(max_tokens=capacity, in_features=160,
                               out_features=136, output_dtype="bfloat16", weight_block_size=32)
    context = PolicyContext.for_identity(B300)
    result = context.resolve(POLICY, query)
    assert context.profile_id is None
    assert result.source is PolicySource.HEURISTIC
    assert result.config == BlockFp8LinearConfig(backend="mxfp8_tcgen05", tile_m=128, tile_n=128)
    assert POLICY.encode_query(query)["max_tokens"] == capacity
    legacy = replace(B300, product_name="unknown", compute_capability=(12, 0))
    assert PolicyContext.for_identity(legacy).resolve(POLICY, query).config.backend == "mxfp8"
    with pytest.raises(ValueError, match="backend"):
        context.resolve(POLICY, query, override=replace(result.config, backend="mxfp8"))
    with pytest.raises(ValueError, match="128x128"):
        context.resolve(POLICY, query, override=replace(result.config, tile_m=64))
    with pytest.raises(ValueError, match="divisible"):
        context.resolve(POLICY, replace(query, out_features=132))


def test_public_plan_retains_native_config_and_bind_performs_no_resolution(monkeypatch):
    from b12x.gemm._shared import block_fp8 as impl
    from tests.gemm.test_block_fp8_linear_scratch_bindings import _packed_weight
    context = PolicyContext.for_identity(B300)
    monkeypatch.setattr(context, "require_device", lambda device: None)
    monkeypatch.setattr(impl, "get_auto_policy", lambda device: context)
    monkeypatch.setattr(impl, "_check_mxfp8_rows_storage", lambda *a, **kw: None)
    plan = linear.plan(linear.Caps(device="cpu", max_tokens=129, in_features=128, out_features=256))
    assert plan.backend == "mxfp8_tcgen05"
    assert plan.mma_tiler_mn == (128, 128)
    assert plan.policy_resolution.source is PolicySource.HEURISTIC
    monkeypatch.setattr(context, "resolve", lambda *a, **kw: pytest.fail("bind resolved policy"))
    spec, = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype)
    source = torch.empty(129, 128, dtype=torch.bfloat16)
    output = torch.empty(129, 256, 1, dtype=source.dtype)
    for m in (1, 8, 129):
        binding = linear.bind(plan, scratch=scratch, source=source[:m],
                              packed_weight=_packed_weight(), output=output[:m])
        assert binding.backend == plan.backend and binding.expected_m == 129


def test_generator_owns_native_candidate_and_schema():
    from b12x.policy.generation.providers.gemm import _BlockFp8Session, _block_fp8_cases, BlockFp8LinearGenerator
    case = _block_fp8_cases()[0]
    candidates = _BlockFp8Session(SimpleNamespace(device=B300)).candidates(case)
    assert len(candidates) == 1
    config = BlockFp8LinearConfig.from_profile(candidates[0].config)
    POLICY.validate_config(BlockFp8LinearQuery(**case.query), config, B300)
    legacy = replace(B300, compute_capability=(12, 0))
    assert len(_BlockFp8Session(SimpleNamespace(device=legacy)).candidates(case)) == 8
    provider = BlockFp8LinearGenerator()
    assert provider.config_schema_version == POLICY.config_schema_version == 3
    assert provider._candidate_contract_version == 2


def test_quantizer_cache_includes_device_and_architecture(monkeypatch):
    quant._get_compiled_mxfp8_rows_quant.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(quant, "current_cuda_stream", lambda: 0)
    specs = []
    def compile(*a, compile_spec, **kw):
        specs.append(compile_spec)
        return lambda *args: None
    monkeypatch.setattr(quant, "b12x_compile", compile)
    def get(device, arch):
        return quant._get_compiled_mxfp8_rows_quant(128, torch.bfloat16, 8, 128, "linear", 0.0, device, arch)
    first = get(0, "sm_103a")
    assert get(0, "sm_103a") is first
    assert get(1, "sm_103a") is not first
    assert get(0, "sm_120a") is not first
    assert len(specs) == 3 and specs[0] != specs[1] and specs[0] != specs[2]
    freeze_kernel_resolution("quantizer requires prewarming on each device")
    try:
        assert get(0, "sm_103a") is first
        with pytest.raises(RuntimeError, match="frozen"):
            get(2, "sm_103a")
    finally:
        unfreeze_kernel_resolution()
        quant._get_compiled_mxfp8_rows_quant.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="prewarmed"):
        get(0, "sm_103a")


def test_sm103_quantizer_selection_ignores_live_counts(monkeypatch):
    from contextlib import nullcontext
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(major=10, minor=3))
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(quant, "_validate_storage", lambda *a: None)
    keys = []
    def resolve(*key):
        keys.append(key)
        return lambda *a: None
    monkeypatch.setattr(quant, "_get_compiled_mxfp8_rows_quant", resolve)
    for count in (1, 4, 8, 9, 129):
        source = SimpleNamespace(shape=(count, 128), ndim=2, dtype=torch.bfloat16,
                                 device=torch.device("cuda:0"), is_contiguous=lambda: True)
        quant.quantize_mxfp8_rows_cute(source, None, None, None)
    assert all(key == keys[0] for key in keys)
    assert keys[0] == (128, torch.bfloat16, 8, 128, "linear", 0.0, 0, "sm_103a")


def test_quantizer_rejects_layout_capacity_and_alias_errors():
    from b12x.gemm._shared.wo_mxfp8 import empty_mxfp8_rows_for_dense_gemm
    source = torch.empty(129, 128, dtype=torch.bfloat16)
    storage = empty_mxfp8_rows_for_dense_gemm(129, 128, device="cpu")
    args = [source, storage.values, storage.scale_rows, storage.scale_mma, 128]
    quant._validate_storage(*args)
    for index, replacement in (
        (1, storage.values[:1]),
        (1, storage.values.T),
        (1, source.view(torch.uint8)[:, :128].contiguous().to(torch.int32)),
        (1, source.view(torch.uint8).view(-1)[:129 * 128].view(129, 128)),
        (2, storage.values.view(torch.uint8).view(-1)[:129 * 4].view(1, 129, 4)),
        (3, storage.scale_mma.contiguous()),
    ):
        bad = args.copy()
        bad[index] = replacement
        with pytest.raises(ValueError):
            quant._validate_storage(*bad)
