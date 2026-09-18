"""Portable Blackwell numerical gates for the residency route and scale ABI."""
from functools import lru_cache

import pytest
import torch


@lru_cache(None)
def programs(id_dtype, gate_first=False):
    from tests.conftest import require_sm103_or_sm12x
    require_sm103_or_sm12x()
    import cutlass
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda
    from b12x._lib.architecture import architecture_for
    from b12x.moe._shared.kernels.sm103.launch import pointer
    from b12x.moe._shared.kernels.sm103.residency import PartitionRoutes, OrderedFinalize, QuantizeMxRoutes
    target = architecture_for(torch.cuda.get_device_capability()).compilation_target
    it = cutlass.Int32 if id_dtype == torch.int32 else cutlass.Int64
    def compile(kernel, dtypes, scalars):
        return cute.compile(kernel, *[pointer(t) for t in dtypes],
                            *[cutlass.Int32(1) for _ in range(scalars)], cuda.CUstream(0),
                            options=f"--gpu-arch={target}")
    partition = compile(PartitionRoutes(4, 32), (it, cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Int32), 1)
    final = compile(OrderedFinalize(256, 4), (cutlass.BFloat16, it, cutlass.Float32, cutlass.BFloat16), 2)
    quant = compile(QuantizeMxRoutes(256, 4), (cutlass.BFloat16, it, cutlass.Float8E4M3FN, cutlass.Uint8), 2)
    act = compile(QuantizeMxRoutes(256, 4, activation=True, gate_first=gate_first, limit=10.), (cutlass.Float32, it, cutlass.Float8E4M3FN, cutlass.Uint8), 2)
    return partition, final, quant, act


def invoke(program, tensors, scalars):
    import cutlass
    import cuda.bindings.driver as cuda
    from b12x.moe._shared.kernels.sm103.launch import pointer
    types = {torch.int32: cutlass.Int32, torch.int64: cutlass.Int64,
             torch.float32: cutlass.Float32, torch.bfloat16: cutlass.BFloat16,
             torch.uint8: cutlass.Uint8, torch.float8_e4m3fn: cutlass.Float8E4M3FN}
    args = tuple(pointer(types[x.dtype], x) for x in tensors) + tuple(cutlass.Int32(x) for x in scalars)
    return program(*args, cuda.CUstream(torch.cuda.current_stream().cuda_stream))


def ordered_reference(rows, ids, weights, experts=4):
    # A BF16 significand times FP32 plus FP32 is exactly representable in
    # FP64 for these finite probes; rounding after each slot emulates fmaf.
    rows, ids, weights = rows.cpu(), ids.cpu(), weights.cpu()
    result = torch.zeros(ids.shape[0], rows.shape[-1], dtype=torch.float32)
    for token in range(ids.shape[0]):
        for rank in range(ids.shape[1]):
            if 0 <= int(ids[token, rank]) < experts:
                result[token] = (result[token].double() + weights[token, rank].double()*rows[token*ids.shape[1]+rank].double()).float()
    return result.to(torch.bfloat16)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_compaction_finalize_graph_and_live_capacity(dtype, monkeypatch):
    partition, final, _, _ = programs(dtype)
    mapping = torch.tensor([[0, 1], [1, 1], [0, 0], [1, 0]], device="cuda", dtype=torch.int32)
    local = torch.empty(2, 32, device="cuda", dtype=torch.int32)
    indices = torch.empty_like(local)
    counts = torch.empty(8, device="cuda", dtype=torch.int32)
    torch.manual_seed(741)
    pointers = (local.data_ptr(), indices.data_ptr(), counts.data_ptr())
    import cutlass.cute as cute
    def forbidden(*args, **kwargs):
        pytest.fail("compiled callables must be retained across live counts and graph replay")
    monkeypatch.setattr(cute, "compile", forbidden)
    for m, top_k in ((1, 1), (1, 3), (8, 3), (2, 2)):
        ids = torch.arange(m*top_k, device="cuda", dtype=dtype).reshape(m, top_k) % 6 - 1
        rows = torch.randn(m*top_k, 256, device="cuda", dtype=torch.bfloat16) * 32
        weights = torch.randn(m, top_k, device="cuda")
        out = torch.empty(m, 256, device="cuda", dtype=torch.bfloat16)
        def run():
            invoke(partition, (ids, mapping, local, indices, counts), (m*top_k,))
            invoke(final, (rows, ids, weights, out), (m, top_k))
        run()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                run()
            for mode in ("mixed", "hot", "cold", "invalid"):
                if mode == "hot": ids.fill_(0)
                elif mode == "cold": ids.fill_(3)
                elif mode == "invalid": ids.fill_(2**40 if dtype == torch.int64 else -1)
                rows.mul_(0.75)
                out.fill_(float("nan"))
                local.fill_(-99)
                indices.fill_(-99)
                expected = ordered_reference(rows, ids, weights)
                before = torch.cuda.memory_stats()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for key in ("allocated_bytes.all.current", "allocation.all.allocated", "allocation.all.freed"):
                    assert after[key] == before[key]
                torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
                flat = ids.cpu().flatten().tolist()
                for tier in (0, 1):
                    selected = [r for r, e in enumerate(flat) if 0 <= e < 4 and int(mapping[e, 0]) == tier]
                    assert int(counts[tier*4]) == len(selected)
                    assert indices[tier, :len(selected)].cpu().tolist() == selected
                    assert local[tier, :len(selected)].cpu().tolist() == [int(mapping[flat[r], 1]) for r in selected]
                assert pointers == (local.data_ptr(), indices.data_ptr(), counts.data_ptr())
        finally:
            graph.reset()


def test_adversarial_fma_and_tier_rounding():
    _, final, _, _ = programs(torch.int32)
    # Slot zero cancels the rounded part of slot one's product. A separate
    # multiply/add loses the residual; fmaf retains it.
    rows = torch.zeros(3, 256, dtype=torch.bfloat16, device="cuda")
    rows[0].fill_(-1.)
    rows[1].fill_(1.0078125)
    weights = torch.tensor([[1.0078126192092896, 1.0000001192092896, 0.]], device="cuda")
    ids = torch.tensor([[0, 1, -1]], device="cuda", dtype=torch.int32)
    out = torch.empty(1, 256, device="cuda", dtype=torch.bfloat16)
    invoke(final, (rows, ids, weights, out), (1, 3))
    expected = ordered_reference(rows, ids, weights)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    assert torch.count_nonzero(expected) == 256
    unfused = (rows[0].float().cpu()*weights[0, 0].cpu() + rows[1].float().cpu()*weights[0, 1].cpu()).to(torch.bfloat16)
    assert not torch.equal(expected[0], unfused)
    # Rounding two tiers independently erases cancellation across a BF16 ulp.
    rows[0].fill_(256.)
    rows[1].fill_(-256.)
    rows[2].fill_(1.)
    ids.copy_(torch.tensor([[0, 1, 0]], device="cuda", dtype=torch.int32))
    weights.fill_(1.)
    invoke(final, (rows, ids, weights, out), (1, 3))
    separately = (rows[0].float()+rows[2].float()).to(torch.bfloat16) + rows[1]
    assert torch.equal(out, torch.ones_like(out))
    assert not torch.equal(out[0], separately)
    rows[0].fill_(2**24)
    rows[1].fill_(1.)
    rows[2].fill_(-(2**24))
    invoke(final, (rows, ids, weights, out), (1, 3))
    assert torch.equal(out, torch.zeros_like(out))
    reordered = (rows[0].float()+rows[2].float())+rows[1].float()
    assert not torch.equal(out[0], reordered.to(torch.bfloat16))
    rows[0].fill_(256.)
    rows[1].fill_(-256.)
    weights[0, 2] = float("nan")
    ids[0, 2] = -1
    invoke(final, (rows, ids, weights, out), (1, 3))
    assert torch.equal(out, torch.zeros_like(out))


@pytest.mark.parametrize("activation", [False, True])
@pytest.mark.parametrize("gate_first", [False, True])
def test_mxfp8_quantization_boundary(activation, gate_first):
    _, _, quant, act = programs(torch.int64, gate_first)
    torch.manual_seed(955)
    ids = torch.tensor([0, 3, -1, 4, 2**40, 1], dtype=torch.int64, device="cuda")
    x = torch.randn((6, 512) if activation else (3, 256), device="cuda", dtype=torch.float32 if activation else torch.bfloat16)
    x[0].zero_()
    q = torch.empty(6, 256, dtype=torch.float8_e4m3fn, device="cuda")
    sf = torch.empty(6, 128*8, dtype=torch.uint8, device="cuda")
    invoke(act if activation else quant, (x, ids, q, sf), (6, 2))
    if activation:
        up, gate = x.float().chunk(2, dim=-1)
        if gate_first:
            gate, up = up, gate
        values = torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
    else:
        values = x.float().repeat_interleave(2, dim=0)
    values[(ids < 0) | (ids >= 4)] = 0
    groups = values.reshape(6, 8, 32)
    amax = groups.abs().amax(-1)
    from b12x._lib.intrinsics import pow2_ceil_ue8m0_torch
    scales, codes = pow2_ceil_ue8m0_torch(amax / 448.)
    scales = torch.where(amax == 0, 1., scales)
    codes = torch.where(amax == 0, 127, codes)
    expected = (groups / scales[..., None]).to(torch.float8_e4m3fn).reshape(6, 256)
    torch.testing.assert_close(q.float(), expected.float(), atol=0, rtol=0)
    offsets = torch.tensor([(g//4)*512+g%4 for g in range(8)], device="cuda")
    torch.testing.assert_close(sf[:, offsets], codes.to(torch.uint8), atol=0, rtol=0)


def test_exact_size_mapped_host_owner_is_readable_by_native_finalizer():
    """Exercise the allocator on SM12x without claiming coherent Grace support."""
    _, final, _, _ = programs(torch.int32)
    from b12x.sequence._shared.disk_table import MappedHostAllocation
    device = torch.device("cuda", torch.cuda.current_device())
    owner = MappedHostAllocation((256,), torch.bfloat16, device)
    try:
        assert owner.nbytes == 512
        ids = torch.zeros((1, 1), device=device, dtype=torch.int32)
        weights = torch.ones((1, 1), device=device)
        out = torch.empty((1, 256), device=device, dtype=torch.bfloat16)
        owner.host_view.copy_(torch.arange(256, dtype=torch.bfloat16))
        invoke(final, (owner.device_view, ids, weights, out), (1, 1))
        torch.cuda.synchronize()
        torch.testing.assert_close(out.cpu().flatten(), owner.host_view, rtol=0, atol=0)
    finally:
        owner.close()
