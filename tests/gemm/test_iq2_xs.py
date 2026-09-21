"""IQ2_XS dense decoding, capacity reuse, and CUDA graph contracts."""

from dataclasses import replace

import pytest
import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.gemm import blockscaled
from b12x.testing.iq2_xs_reference import dequantize_blocks
from ._blockscaled import prepared
from ..conftest import require_b12x


def blocks(n, k):
    generator = torch.Generator().manual_seed(472)
    raw = torch.randint(0, 256, (n, k // 256, 74), dtype=torch.uint8, generator=generator)
    base = torch.randn((n, k // 256, 1), generator=generator).half() / 128
    base[::7] = 0
    raw[..., :2] = base.contiguous().view(torch.uint8)
    return raw


def assert_gemm_close(actual, expected):
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual) > 0
    error = (actual.float() - expected.float()).square().sum().sqrt()
    norm = expected.float().square().sum().sqrt()
    assert (error / norm).item() < 0.004
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.015, atol=0.03125)


@pytest.mark.parametrize("config", [(1, 4, 256, 1), (2, 4, 256, 1), (4, 4, 256, 1), (8, 4, 256, 1),
                                   (16, 64, 64, 1), (16, 128, 128, 4), (16, 64, 128, 8), (16, 128, 64, 2),
                                   (32, 64, 64, 1), (32, 128, 128, 4), (64, 64, 128, 8), (64, 128, 64, 2),
                                   (16, 64, 256, 1), (16, 128, 256, 8),
                                   (32, 128, 256, 4), (64, 64, 256, 8), (64, 128, 256, 1),
                                   (8, 64, 64, 1), (8, 128, 128, 1), (8, 128, 256, 1)])
@pytest.mark.parametrize("n", [136, 256])
def test_iq2_xs_dynamic_graph_replay(config, n):
    device = require_b12x()
    k, capacity = 768, 65
    raw = blocks(n, k)
    weight = blockscaled.pack_weight(raw.to(device), recipe="iq2_xs")
    reference = dequantize_blocks(raw).bfloat16().to(device)
    source = torch.randn((capacity, k), dtype=torch.bfloat16, device=device)
    out = torch.empty((capacity, n), dtype=torch.bfloat16, device=device)
    bm, bn, bk, split = config
    override = blockscaled.BlockscaledConfig(mode="a16", tile_m=bm, tile_n=bn, tile_k=bk, split_k=split)
    with prepared(source, weight, out=out, override=override) as plan:
        from b12x.preparation.types import require_prepared
        state = require_prepared(plan, "gemm.blockscaled_precision", source.device)
        callable_before = state.programs["gemm"]
        workspace_ptr = None if state.workspace is None else state.workspace.data_ptr()
        with kernel_resolution_guard("IQ2_XS dense replay"):
            for m in (1, 3, 8, 17, capacity):
                x, y = source[:m], out[:m]
                expected = (x.float() @ reference.float().T).bfloat16()
                blockscaled.mm(x, weight, out=y, plan=plan)
                assert_gemm_close(y, expected)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    blockscaled.mm(x, weight, out=y, plan=plan)
                x.neg_()
                y.fill_(float("nan"))
                allocated = torch.cuda.memory_allocated(device)
                graph.replay()
                torch.cuda.synchronize(device)
                assert torch.cuda.memory_allocated(device) == allocated
                assert_gemm_close(y, -expected)
                assert state.programs["gemm"] is callable_before
                assert workspace_ptr == (None if state.workspace is None else state.workspace.data_ptr())
        with pytest.raises(ValueError, match="capacity"):
            blockscaled.mm(torch.empty((capacity + 1, k), device=device, dtype=torch.bfloat16), weight,
                           out=torch.empty((capacity + 1, n), device=device, dtype=torch.bfloat16), plan=plan)


@pytest.mark.parametrize("tile_m,tile_n,tile_k", [(1, 4, 256), (2, 4, 256), (4, 4, 256), (8, 4, 256),
                                                (16, 64, 64), (32, 64, 64), (64, 64, 64),
                                                (16, 64, 256), (32, 64, 256), (64, 64, 256),
                                                (8, 128, 256)])
def test_iq2_xs_every_descriptor_and_scale_matches_independent_oracle(tile_m, tile_n, tile_k):
    device = require_b12x()
    n, k = 2048, 256
    raw = blocks(n, k)
    raw[..., :2] = torch.full((n, 1, 1), 0.03125, dtype=torch.float16).view(torch.uint8)
    raw[..., 2:66] = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.uint8).view(n, 1, 64)
    raw[..., 66:] = torch.arange(n * 8, dtype=torch.int32).to(torch.uint8).view(n, 1, 8)
    weight = blockscaled.pack_weight(raw.to(device), recipe="iq2_xs")
    source = torch.eye(k, device=device, dtype=torch.bfloat16)
    expected = dequantize_blocks(raw).bfloat16().T.contiguous().to(device)
    with prepared(source, weight, override=blockscaled.BlockscaledConfig(mode="a16", tile_m=tile_m,
                  tile_n=tile_n, tile_k=tile_k, split_k=1)) as plan:
        actual = blockscaled.mm(source, weight, plan=plan)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert weight.values.numel() + weight.metadata.numel() == raw.numel()


def test_iq2_xs_policy_rejects_activation_quantization():
    from b12x.gemm.blockscaled._tuning import _validate_query
    q = blockscaled.BlockscaledQuery(recipe="iq2_xs", num_tokens=8, in_features=256,
                                     padded_in_features=256, out_features=128)
    _validate_query(q, None)
    for invalid in (replace(q, activation_mode="quantized"), replace(q, activation_scale_available=True),
                    replace(q, in_features=128), replace(q, padded_in_features=512)):
        with pytest.raises(ValueError, match="IQ2_XS"):
            _validate_query(invalid, None)


def test_iq2_xs_compiled_api_with_caller_workspace():
    device = require_b12x()
    raw = blocks(128, 256)
    weight = blockscaled.pack_weight(raw.to(device), recipe="iq2_xs")
    reference = dequantize_blocks(raw).bfloat16().to(device).float()
    source = torch.randn((8, 256), device=device, dtype=torch.bfloat16)
    out = torch.empty((8, 128), device=device, dtype=torch.bfloat16)
    scratch = torch.empty(8 * 8 * 128 * 4, device=device, dtype=torch.uint8)
    with prepared(source, weight, out=out, workspace=scratch) as plan:
        def project(x, y):
            return blockscaled.mm(x, weight, out=y, workspace=scratch, plan=plan)

        compiled = torch.compile(project, fullgraph=True, dynamic=True)
        for m in (1, 3, 8):
            source.normal_()
            out.fill_(float("nan"))
            scratch.fill_(255)
            actual = compiled(source[:m], out[:m])
            assert actual.data_ptr() == out.data_ptr()
            assert_gemm_close(actual, (source[:m].float() @ reference.T).bfloat16())
    torch._dynamo.reset()


def test_iq2_xs_rejects_malformed_payload():
    device = require_b12x()
    raw = blocks(8, 256).to(device)
    with pytest.raises(ValueError, match="embedded"):
        blockscaled.pack_weight(raw, torch.ones(1, device=device), recipe="iq2_xs")
    raw[..., :2] = torch.tensor([float("inf")], dtype=torch.float16, device=device).view(torch.uint8)
    with pytest.raises(ValueError, match="finite"):
        blockscaled.pack_weight(raw, recipe="iq2_xs")
