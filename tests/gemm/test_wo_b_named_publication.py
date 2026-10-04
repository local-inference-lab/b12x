import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def scope_expression():
    from b12x._lib import dense_gemm
    tree = ast.parse(Path(dense_gemm.__file__).read_text())
    return next(node.value for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Attribute) and target.attr == 'wo_b_named_publication'
                        for target in node.targets))


def scope_state(m):
    import cutlass
    expr = scope_expression()
    state = {node.attr: False for node in ast.walk(expr)
             if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
             and node.value.id == 'self'}
    state.update(wo_b_geometry=(4096, 4096, 1), fused_quant_a=True,
                 fused_quant_a_inner_span=1024, mxfp6_fmt_a=None, mxfp6_fmt_b=None,
                 a_dtype=cutlass.Float8E4M3FN, c_dtype=cutlass.BFloat16,
                 sf_dtype=cutlass.Float8E8M0FNU, sf_vec_size=32,
                 single_work_tile_per_cta=True, load_path='tma', threads_per_cta=96,
                 num_mma_warps=2, ab_stage=4, cluster_shape_mnk=(1, 1, 1),
                 tile_shape_mnk=(16, 64 if m == 1 else 128, 128),
                 fused_quant_a_wide=m == 1, split_k_slices=2 if m == 1 else 1,
                 split_k_atomic_bf16=True, direct_one_m_tile_scheduler=m == 1,
                 use_m1_non_tma_c=m == 1)
    return state


def eligible(state):
    import cutlass
    return eval(compile(ast.Expression(scope_expression()), '<wo-b-scope>', 'eval'),
                {'self': SimpleNamespace(**state), 'cutlass': cutlass})


@pytest.mark.parametrize('m', [1, 4])
def test_wo_b_scope_accepts_exact_config(m):
    assert eligible(scope_state(m))
    assert 2 * scope_state(m)['ab_stage'] + 3 <= 16


@pytest.mark.parametrize('field,value', [
    ('wo_b_geometry', (4096, 2048, 1)), ('wo_b_geometry', (2048, 4096, 1)),
    ('wo_b_geometry', (4096, 4096, 2)), ('fused_quant_a', False),
    ('fused_quant_a_inner_span', 512), ('mxfp6_fmt_a', 'e4m3'), ('mxfp6_fmt_b', 'e2m3'),
    ('plain_fp8', True), ('sf_vec_size', 16),
    ('single_work_tile_per_cta', False), ('load_path', 'cpasync'),
    ('threads_per_cta', 288), ('num_mma_warps', 8), ('ab_stage', 7), ('ab_stage', 5),
    ('cluster_shape_mnk', (2, 1, 1)), ('mgroup_labels', True), ('mgroup_masked', True),
    ('b_packed', True), ('swap_ab', True), ('b_tile_major', True), ('manual_bk64_sf', True),
    ('mgroup_compact_sfb', True), ('mgroup_deferred_wait', True), ('mgroup_sfb_stage_reuse', True),
    ('direct_sfa_prefix', True), ('direct_sfb_representative', True), ('fused_quant_a_inv_rope', True),
    ('fused_quant_a_row_stride', 4096), ('fused_quant_a_l_stride', 4096),
    ('quantize_c', True), ('row_scale', True), ('split_k_slices', 4),
])
@pytest.mark.parametrize('m', [1, 4])
def test_wo_b_scope_excludes_other_configs(m, field, value):
    state = scope_state(m)
    state[field] = value
    assert not eligible(state)


def test_wo_b_scope_excludes_nonatomic_split_and_wrong_roles():
    for field in ('split_k_atomic_bf16', 'fused_quant_a_wide', 'direct_one_m_tile_scheduler',
                  'use_m1_non_tma_c'):
        state = scope_state(1)
        state[field] = False
        assert not eligible(state)
    for field in ('fused_quant_a_wide', 'direct_one_m_tile_scheduler', 'use_m1_non_tma_c'):
        state = scope_state(4)
        state[field] = True
        assert not eligible(state)


@pytest.mark.parametrize('m', [1, 4], ids=['t1-split2', 't4-single'])
def test_wo_b_named_quantized_oracle_and_graph(m, monkeypatch):
    from tests._reference.helpers import require_b12x
    require_b12x()
    from b12x._lib import dense_gemm as dg
    from b12x._lib.compile_plan import observe_programs
    from b12x.gemm._shared import wo_mxfp8 as wo
    seen = []
    setup = dg.DenseGemmKernel._setup_attributes
    def observe(kernel):
        setup(kernel)
        seen.append(dict(named=bool(kernel.wo_b_named_publication),
                         fp6=bool(kernel.fp6_named_publication), stages=int(kernel.ab_stage),
                         threads=int(kernel.threads_per_cta), split=int(kernel.split_k_slices),
                         atomic=bool(kernel.split_k_atomic_bf16),
                         packed_a=bool(kernel.wo_b_packed_a_stores),
                         wide_packed_a=bool(kernel.wo_b_wide_packed_a_stores)))
    monkeypatch.setattr(dg.DenseGemmKernel, '_setup_attributes', observe)
    torch.manual_seed(654 + m)
    a = wo.empty_dense_gemm_mnl_view(m, 1024, 4, device='cuda', dtype=torch.bfloat16)
    original = torch.zeros_like(a)
    for group in (0, 2):
        original[:, :32, group] = torch.randint(-2, 3, (m, 32), device='cuda').to(a.dtype) * .125
    a.copy_(original)
    weights = torch.randint(-2, 3, (4096, 4096), device='cuda').to(a.dtype) * .125
    b = wo.quantize_mxfp8_rows_torch(weights)
    rhs = wo.dequantize_mxfp8_rows_torch(b.values, b.scale_rows).double()
    out = torch.empty((m, 4096, 1), device='cuda', dtype=a.dtype)
    def run():
        return wo.wo_b_dense_gemm_fused_quant_mxfp8(a, b, out=out, expected_m=m)
    def oracle():
        aq = wo.quantize_wo_b_input_mxfp8(a)
        lhs = wo.dequantize_mxfp8_rows_torch(aq.values, aq.scale_rows).double()
        span = 1024 if m == 8 else 2048
        partials = [(lhs[:, i:i + span] @ rhs[:, i:i + span].T).to(out.dtype)
                    for i in range(0, 4096, span)]
        expected = torch.stack([p.float() for p in partials]).sum(0).to(out.dtype)
        if torch.count_nonzero(a):
            assert sum(bool(torch.count_nonzero(partial)) for partial in partials) == 2
        return expected.unsqueeze(-1)
    def check(expected):
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        if torch.count_nonzero(a):
            assert torch.count_nonzero(out) > 0
    with observe_programs() as programs:
        out.fill_(float('nan'))
        run()
    torch.cuda.synchronize()
    assert any(s['named'] == (m != 8) and not s['fp6'] and s['stages'] == 4
               and s['threads'] == 96 and s['split'] == (4 if m == 8 else 2 if m == 1 else 1)
               and s['atomic'] and s['packed_a'] == (2 <= m <= 7)
               and s['wide_packed_a'] == (m == 1) for s in seen)
    check(oracle())
    first = out.clone()
    if m == 1:
        out.fill_(float('nan'))
        wo.wo_b_dense_gemm_fused_quant_mxfp8(
            a, b, out=out, expected_m=m, _atomic_output_precleared=True)
        torch.cuda.synchronize()
        assert not torch.isfinite(out).any()
        run()
        torch.cuda.synchronize()
        check(first)
    for _ in range(3):
        out.fill_(float('nan'))
        run()
        torch.cuda.synchronize()
        check(first)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    try:
        for multiplier in (2., 0., -.5, 1.):
            a.copy_(original * multiplier)
            expected = oracle()
            for _ in range(2):
                out.fill_(float('nan'))
                graph.replay()
                torch.cuda.synchronize()
                check(expected)
            replay = out.clone()
            out.fill_(float('nan'))
            run()
            torch.cuda.synchronize()
            check(replay)
    finally:
        graph.reset()
    if evidence := os.environ.get('B12X_WO_B_NAMED_EVIDENCE'):
        Path(evidence).write_text(json.dumps(dict(m=m, n=4096, k=4096, scopes=seen,
            programs=[dict(dialect=p.dialect, key=p.key, name=p.name)
                      for p in sorted(programs, key=lambda p: (p.dialect, p.key))]), indent=2) + '\n')
