import pytest
import torch


@pytest.mark.parametrize('n,k,fused', [
    pytest.param(2048, 128, True, id='fused-short'),
    pytest.param(2048, 640, True, id='fused-wrap'),
    pytest.param(5120, 5120, True, id='fused-deep'),
    pytest.param(12160, 1152, True, id='fused-multitile'),
    pytest.param(5120, 5120, False, id='unfused-deep'),
])
def test_named_fp6_ring_graph_and_work_tiles(n, k, fused, monkeypatch):
    import os
    from tests._reference.helpers import require_b12x
    require_b12x()
    if os.environ.get('B12X_NAMED_EVIDENCE'):
        prop = torch.cuda.get_device_properties(torch.cuda.current_device())
        assert prop.name == 'NVIDIA RTX PRO 6000 Blackwell Server Edition'
        assert (prop.major, prop.minor, prop.multi_processor_count) == (12, 0, 188)
        from cutlass.cute import _dsl
        assert os.environ.get('CUTE_DSL_ARCH') == 'sm_120a'
        assert _dsl.CuTeDSL._get_dsl().envar.arch == 'sm_120a'
    from b12x._lib import dense_gemm as dg
    assert dg._select_default_mma_tiler_mn(1, n, 188, is_mxfp8=False, is_mxfp6=True, k=k) == (16, 64)
    from b12x.quantization.mxfp6 import fp6_dense_weights as fw
    from b12x._lib.compile_plan import observe_programs
    monkeypatch.setattr(dg, '_DENSE_FUSED_QUANT', fused)
    monkeypatch.setattr(fw, '_DENSE_FUSED_QUANT', fused)
    if n == 12160:
        import cutlass
        assert dg._dense_gemm_target_occupancy(
            n=n, k=k, l=1, ab_dtype=cutlass.Float8E4M3FN, c_dtype=cutlass.BFloat16,
            tile_k=128, mma_tiler_mn=(16, 64), cluster_shape_mn=(1, 1), sm_count=188,
            load_path='tma', swap_ab=False, b_tile_major=False, is_mxfp6=True,
        ) == 2
        monkeypatch.setattr(dg, '_dense_gemm_target_occupancy', lambda **kwargs: 1)
    seen = []
    setup = dg.DenseGemmKernel._setup_attributes
    def observe(kernel):
        setup(kernel)
        seen.append(dict(named=bool(kernel.fp6_named_publication), stages=int(kernel.ab_stage),
                         single_work_tile=bool(kernel.single_work_tile_per_cta),
                         threads=int(kernel.threads_per_cta)))
    monkeypatch.setattr(dg.DenseGemmKernel, '_setup_attributes', observe)
    torch.manual_seed(n + k)
    a = torch.randn((1, k), device='cuda', dtype=torch.bfloat16) * .2
    w = torch.randn((n, k), device='cuda', dtype=torch.bfloat16) * .2
    b = fw.quantize_dense_weight_to_fp6(w)
    b.expanded_weight()
    out = torch.empty((1, n), device='cuda', dtype=torch.bfloat16)
    def run():
        return fw.dense_fp6_linear(a, b, out=out)
    def check():
        expected = a.float() @ w.float().T
        assert torch.isfinite(out).all()
        if torch.count_nonzero(a) == 0:
            torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)
        else:
            assert torch.count_nonzero(out) > 0
            assert torch.nn.functional.cosine_similarity(out.float().flatten(), expected.flatten(), dim=0) > .99
    out.fill_(float('nan'))
    with observe_programs() as programs:
        run()
    torch.cuda.synchronize()
    assert any(s['named'] and s['stages'] == 4 and s['threads'] == 96 for s in seen)
    if n == 12160:
        assert any(s['named'] and not s['single_work_tile'] for s in seen)
    check()
    initial = out.clone()
    out.fill_(float('nan'))
    run()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, initial, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    original = a.clone()
    for multiplier in (2., 0., .5):
        a.copy_(original * multiplier)
        out.fill_(float('nan'))
        run()
        torch.cuda.synchronize()
        check()
        eager = out.clone()
        out.fill_(float('nan'))
        graph.replay()
        torch.cuda.synchronize()
        check()
        torch.testing.assert_close(out, eager, rtol=0, atol=0)
    graph.reset()
    import json
    import os
    from pathlib import Path
    if evidence := os.environ.get('B12X_NAMED_EVIDENCE'):
        Path(evidence).write_text(json.dumps(dict(
            shape=dict(m=1, n=n, k=k), fused=fused, forced_occupancy=n == 12160,
            scopes=seen, programs=[dict(dialect=p.dialect, key=p.key, name=p.name)
                                  for p in sorted(programs, key=lambda p: (p.dialect, p.key))],
            oracle='bf16-cosine>0.99; finite/nonzero; zero-exact; same-input-eager-replay-exact',
        ), indent=2) + '\n')
