"""Physical SM103 gates for the public hierarchical expert preparation path."""
from dataclasses import replace

import pytest
import torch

from b12x.moe import fused_moe as moe
from b12x.preparation import PreparationSession, PreparedCall
from b12x._lib.runtime_control import kernel_resolution_guard
from tests.moe.test_expert_residency import declaration, placement
from tests.moe.test_residency_kernels import ordered_reference


def require_device(grace=False):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("physical SM103 required for tcgen05 residency qualification")
    device = torch.device("cuda", torch.cuda.current_device())
    if grace:
        from b12x._lib.platform import probe_platform
        if not probe_platform(device).grace_coherent:
            pytest.skip("verified Grace coherency required for mapped expert operands")
    return device


def randomize(weights):
    generator = torch.Generator().manual_seed(413)
    for w in (weights.w13, weights.w2):
        w.random_(0, 256, generator=generator)
    for s in (weights.w13_block_scales, weights.w2_block_scales):
        s.random_(121, 125, generator=generator)


def reference(a, weights, ids, route_weights):
    """Independent FP32 projections and per-slot FMA after BF16 FC2."""
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
    def unpack(w, sf):
        codes = torch.stack((w & 15, w >> 4), dim=-1).reshape(w.shape[0], -1).long()
        return lut[codes] * torch.exp2(sf.float()-127).repeat_interleave(32, -1)
    def quant(value):
        shape = value.shape
        blocks = value.float().reshape(-1, 32)
        maximum = blocks.abs().amax(-1, keepdim=True)
        exponent = torch.ceil(torch.log2(torch.where(maximum > 0, maximum/448., 1.))).clamp(-126, 127)
        scale = torch.exp2(exponent)
        return ((blocks/scale).to(torch.float8_e4m3fn).float()*scale).reshape(shape)
    a, ids = a.float().cpu(), ids.cpu()
    rows = torch.zeros(ids.numel(), a.shape[1], dtype=torch.bfloat16)
    for token in range(a.shape[0]):
        for rank in range(ids.shape[1]):
            expert = int(ids[token, rank])
            if not 0 <= expert < weights.w13.shape[0]:
                continue
            fc1 = quant(a[token]) @ unpack(weights.w13[expert], weights.w13_block_scales[expert]).T
            up, gate = fc1.chunk(2)
            intermediate = torch.nn.functional.silu(gate)*up
            rows[token*ids.shape[1]+rank] = quant(intermediate) @ unpack(weights.w2[expert], weights.w2_block_scales[expert]).T
    return ordered_reference(rows, ids, route_weights)


def prepare(session, plan, a, ids, weights, name):
    def factory(state):
        binding = state.bind(a=a, topk_ids=ids, topk_weights=weights)
        return PreparedCall(run=binding.run, output=binding.output, owners=(state, binding))
    return session.prepare((plan.request(name=name, prepare_call=factory),))


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("hot", [(0, 1, 2, 3), (0, 2), ()])
def test_public_preparation_split_parity_and_graph_replay(id_dtype, hot, tmp_path):
    device = require_device(grace=len(hot) < 4)
    profile = placement(hot=hot, cold=tuple(e for e in range(4) if e not in hot))
    plan, source = declaration(profile=profile)
    control, source_control = declaration(profile=placement(hot=(0, 1, 2, 3), cold=()))
    randomize(source)
    randomize(source_control)
    a = torch.randn(8, 256, dtype=torch.bfloat16, device=device) * .1
    ids = torch.tensor([[0, 1, 3]]*8, dtype=id_dtype, device=device)
    weights = torch.tensor([[.3, -.2, .7]]*8, device=device)
    with PreparationSession(device=device, autotune=False, cache_dir=tmp_path, compile_workers=0) as session:
        prepare(session, plan, a, ids, weights, "tiered")
        prepare(session, control, a, ids, weights, "hbm_control")
        session.freeze()
        state = plan.prepared.state
        ptrs = (state.slab.data_ptr(), state.mapping.data_ptr(), *(t.slab.data_ptr() for t in state.tiers if t))
        with kernel_resolution_guard("hierarchical replay"):
            for m, top_k in ((1, 1), (2, 2), (4, 3), (8, 3)):
                x = a[:m]
                live_ids = ids[:m, :top_k].contiguous()
                live_weights = weights[:m, :top_k].contiguous()
                binding = moe.bind(plan, a=x, topk_ids=live_ids, topk_weights=live_weights)
                baseline = moe.bind(control, a=x, topk_ids=live_ids, topk_weights=live_weights)
                actual = moe.run(binding=binding)
                expected = moe.run(binding=baseline)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                oracle = reference(x, source, live_ids, live_weights)
                torch.testing.assert_close(actual.cpu(), oracle, atol=.001, rtol=.03)
                assert torch.count_nonzero(actual) > 0 and torch.isfinite(actual).all()
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph):
                        moe.run(binding=binding)
                    for mode in ("hot", "cold", "mixed", "invalid"):
                        x.mul_(.75)
                        if mode == "hot": live_ids.fill_(0)
                        elif mode == "cold": live_ids.fill_(3)
                        elif mode == "mixed": live_ids.copy_(torch.arange(m*top_k, device=device).reshape(m, top_k) % 4)
                        else: live_ids.fill_(2**40 if id_dtype == torch.int64 else -1)
                        expected = moe.run(binding=baseline).clone()
                        actual.fill_(float("nan"))
                        before = torch.cuda.memory_stats()
                        graph.replay()
                        torch.cuda.synchronize()
                        after = torch.cuda.memory_stats()
                        for key in ("allocated_bytes.all.current", "allocation.all.allocated", "allocation.all.freed"):
                            assert after[key] == before[key]
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        assert ptrs == (state.slab.data_ptr(), state.mapping.data_ptr(), *(t.slab.data_ptr() for t in state.tiers if t))
                finally:
                    graph.reset()
