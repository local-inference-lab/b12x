"""FP32 split-K planning, scratch ownership, and quantized decode correctness."""

from dataclasses import replace

import pytest
import torch

from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x.gemm import block_fp8_linear as bfl
from b12x.gemm.block_fp8_linear._policy import (
    BLOCK_FP8_LINEAR_POLICY,
    BlockFp8LinearConfig,
    BlockFp8LinearQuery,
    _validate,
)
from b12x.policy import BLOCK_FP8_LINEAR, get_auto_policy
from tests._reference.helpers import require_b12x
from tests.gemm.test_gemm_block_fp8_linear import (
    _assert_v41_accumulation_matches_reference,
    _make_block_fp8_weight,
)


@pytest.mark.parametrize("slices", (2, 4))
@pytest.mark.parametrize("replacement", [
    {"max_tokens": 1}, {"max_tokens": 9}, {"weight_block_size": 128},
    {"output_dtype": "float16"}, {"in_features": 1280},
])
def test_splitk_policy_rejects_unsupported_queries(replacement, slices):
    query = BlockFp8LinearQuery(max_tokens=8, in_features=5120,
                               out_features=1792, output_dtype="bfloat16",
                               weight_block_size=32)
    config = BlockFp8LinearConfig(backend=f"mxfp8_split{slices}_fp32", tile_m=16, tile_n=64)
    _validate(query, config, None)
    with pytest.raises(ValueError, match="FP32 split"):
        _validate(replace(query, **replacement), config, None)


def test_four_slices_require_divisible_k_tiles():
    query = BlockFp8LinearQuery(max_tokens=8, in_features=1536,
                               out_features=1792, output_dtype="bfloat16",
                               weight_block_size=32)
    config = BlockFp8LinearConfig(backend="mxfp8_split2_fp32", tile_m=16, tile_n=64)
    _validate(query, config, None)
    with pytest.raises(ValueError, match="K divisible by 1024"):
        _validate(query, replace(config, backend="mxfp8_split4_fp32"), None)


def test_split2_generator_preserves_schema_and_eligibility():
    from b12x.policy.generation.providers.gemm import (
        BlockFp8LinearGenerator, _BlockFp8Session, _block_fp8_cases,
    )

    generator = BlockFp8LinearGenerator()
    assert generator.config_schema_version == BLOCK_FP8_LINEAR_POLICY.config_schema_version == 4
    session = _BlockFp8Session(None)
    for case in _block_fp8_cases():
        query = BlockFp8LinearQuery(**dict(case.query))
        for candidate in session.candidates(case):
            config = BlockFp8LinearConfig.from_profile(candidate.config)
            _validate(query, config, None)


@pytest.mark.parametrize("slices", (2, 4))
@pytest.mark.parametrize("capacity", (2, 4, 8))
@pytest.mark.parametrize("n", (1152, 1792))
def test_splitk_public_plan_frozen_live_rows_and_fp64_oracle(capacity, n, slices, monkeypatch):
    """Live rows share a fixed partial buffer and do not enable BF16 atomics."""
    import b12x._lib.dense_gemm as dense_module

    require_b12x()
    monkeypatch.setattr(dense_module, "_B12X_DENSE_SPLITK_TURBO", True)
    torch.manual_seed(414112 + n + capacity)
    k = 5120
    device = torch.device("cuda", torch.cuda.current_device())
    source = torch.randn((capacity, k), dtype=torch.bfloat16, device=device)
    weight, scales = _make_block_fp8_weight(n, k, block_size=32)
    packed = bfl.pack_weight(weight, scales, block_size=(32, 32))
    caps = bfl.Caps(device=device, max_tokens=capacity, in_features=k,
                    out_features=n, block_size=(32, 32))
    config = BlockFp8LinearConfig(backend=f"mxfp8_split{slices}_fp32", tile_m=16, tile_n=64)
    policy = get_auto_policy(device).with_override(BLOCK_FP8_LINEAR, config)
    plan = bfl.plan(caps, policy=policy)
    spec, = plan.scratch_specs()
    scratch = torch.empty(spec.shape, device=device, dtype=spec.dtype)
    output = torch.empty((capacity, n, 1), dtype=source.dtype, device=device)
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor, props.multi_processor_count) == (12, 0, 188):
        assert bfl.plan(caps).policy_resolution.config == replace(config, backend="mxfp8_split4_fp32")

    def bind(rows):
        return bfl.bind(plan, scratch=scratch, source=source[:rows],
                        packed_weight=packed, output=output[:rows])

    binding = bind(capacity)
    assert binding.split_k_partials.dtype == torch.float32
    assert binding.split_k_partials.shape == (slices, capacity, n)
    assert binding.split_k_partials.data_ptr() == scratch.data_ptr() + plan.split_k_offset_bytes
    bfl.run(binding=binding)
    pointers = (source.data_ptr(), scratch.data_ptr(), output.data_ptr())
    freeze_kernel_resolution("block32 FP32 partials at fixed decode capacity")
    try:
        for rows in sorted({1, max(1, capacity - 1), capacity}):
            bound = bind(rows)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                bfl.run(binding=bound)
            for _ in range(3):
                source.normal_()
                source[:, :32].mul_(1e-5)
                scratch.fill_(255)
                output.fill_(float("nan"))
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                assert before["allocation.all.allocated"] == after["allocation.all.allocated"]
                assert pointers == (source.data_ptr(), scratch.data_ptr(), output.data_ptr())
                assert torch.isfinite(bound.split_k_partials).all()
                _assert_v41_accumulation_matches_reference(
                    source[:rows], weight, scales, output[:rows, :, 0],
                )
    finally:
        unfreeze_kernel_resolution()
