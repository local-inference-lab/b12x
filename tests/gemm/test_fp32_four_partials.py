"""Research-only four-partial FP32 projection reduction with dynamic rows."""

import pytest
import torch

from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x._lib.dense_gemm import dense_gemm
from b12x.gemm import block_fp8_linear as bfl
from b12x.gemm._shared.block_fp8 import _run_block_fp8_quant_kernel
from b12x.policy import BLOCK_FP8_LINEAR, get_auto_policy
from tests._reference.helpers import require_b12x
from tests.gemm.test_gemm_block_fp8_linear import (
    _assert_v41_accumulation_matches_reference,
    _make_block_fp8_weight,
)


@pytest.mark.parametrize("n", (1152, 1792))
@pytest.mark.parametrize("capacity", (2, 4, 8))
def test_fp32_four_partials_frozen_dynamic_rows(n, capacity, monkeypatch):
    """Every live slice is written; source mutation does not allocate on replay."""
    import b12x._lib.dense_gemm as dense_module

    require_b12x()
    monkeypatch.setattr(dense_module, "_B12X_DENSE_SPLITK_TURBO", True)
    torch.manual_seed(42313 + n + capacity)
    device = torch.device("cuda", torch.cuda.current_device())
    k = 5120
    source = torch.randn((capacity, k), dtype=torch.bfloat16, device=device)
    weight, scales = _make_block_fp8_weight(n, k, block_size=32)
    packed = bfl.pack_weight(weight, scales, block_size=(32, 32))
    policy = get_auto_policy(device).with_override(
        BLOCK_FP8_LINEAR,
        bfl.BlockFp8LinearConfig(backend="mxfp8", tile_m=16, tile_n=64),
    )
    plan = bfl.plan(bfl.Caps(device=device, max_tokens=capacity, in_features=k,
                            out_features=n, block_size=(32, 32)), policy=policy)
    spec, = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
    partials = torch.empty((4, capacity, n), dtype=torch.float32, device=device)
    output = torch.empty((capacity, n, 1), dtype=torch.bfloat16, device=device)

    def make_call(rows):
        binding = bfl.bind(plan, scratch=scratch, source=source[:rows],
                           packed_weight=packed, output=output[:rows])
        quant = binding.x_q

        def run():
            _run_block_fp8_quant_kernel(
                source[:rows], quant.values, quant.scale_rows, quant.scale_mma,
                rows, k, expected_m=capacity, min_amax=1e-4,
            )
            dense_gemm(
                (quant.values.reshape(rows, k, 1), quant.scale_mma),
                (packed.weight.values.reshape(n, k, 1), packed.weight.scale_mma),
                ab_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu",
                c_dtype="bfloat16", sf_vec_size=32, expected_m=capacity,
                mma_tiler_mn=(16, 64), out=output[:rows],
                _split_k_slices_override=4, _split_k_atomic_bf16_override=False,
                _split_k_workspace=partials,
            )

        return run

    make_call(capacity)()
    pointers = tuple(t.data_ptr() for t in (source, scratch, partials, output))
    freeze_kernel_resolution("four FP32 partials at fixed projection capacity")
    try:
        for rows in sorted({1, max(1, capacity - 1), capacity}):
            run = make_call(rows)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            for _ in range(3):
                source.normal_()
                source[:, :32].mul_(1e-5)
                scratch.fill_(255)
                partials.fill_(float("nan"))
                output.fill_(float("nan"))
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
                assert pointers == tuple(t.data_ptr() for t in (source, scratch, partials, output))
                assert torch.isfinite(partials.view(-1)[:4 * rows * n]).all()
                assert torch.isnan(partials.view(-1)[4 * rows * n:]).all()
                _assert_v41_accumulation_matches_reference(
                    source[:rows], weight, scales, output[:rows, :, 0],
                )
    finally:
        unfreeze_kernel_resolution()
