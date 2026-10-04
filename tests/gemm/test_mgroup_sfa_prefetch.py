import ast
from pathlib import Path

import pytest
import torch


def test_prefetch_static_identity_and_scope():
    from b12x.gemm.mgroup_fp8_gemm._joint import _JointLaunch
    base = _JointLaunch(4096, 6144, 64, 16384, 188)
    candidate = _JointLaunch(4096, 6144, 64, 16384, 188, True)
    assert base.compile_key()[0] == 'mgroup-joint-v1'
    assert candidate.compile_key()[0] == 'mgroup-joint-sfa-prefetch-v1'
    assert candidate.compile_key()[1:] == base.compile_key()[1:]
    assert candidate.full.mgroup_sfa_prefetch and not candidate.narrow.mgroup_sfa_prefetch
    assert not base.full.mgroup_sfa_prefetch
    from b12x._lib.dense_gemm import DenseGemmKernel
    with pytest.raises(ValueError):
        DenseGemmKernel(32, (128, 128), (1, 1), mgroup_sfa_prefetch=True)


def test_prefetch_offsets_and_last_stage_guards():
    from b12x._lib import dense_gemm
    tree = ast.parse(Path(dense_gemm.__file__).read_text())
    assignments = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'pf_offset' for t in n.targets)]
    assert len(assignments) == 2
    for expression in assignments:
        code = compile(ast.Expression(expression), 'pf_offset', 'eval')
        for tile, ktiles in [(0, 2), (7, 10), (1023, 8194)]:
            for k in [0, 1, ktiles-1]:
                actual = eval(code, dict(Int64=int, tile_coord_mnl=[tile], k_tile_cnt=ktiles, k_tile_start=k, pf_k=k))
                assert actual == (tile*(ktiles//2)+k//2)*512
        assert eval(code, dict(Int64=int, tile_coord_mnl=[1023], k_tile_cnt=8194, k_tile_start=8193, pf_k=8193)) > 2**31
    source = ast.unparse(tree)
    assert 'mainloop_producer_state.count < k_tile_iter_cnt' in source
    assert 'Int64(pf_k % 2)' in source and 'Int64(k_tile_start % 2)' in source


@pytest.mark.parametrize('k', [128, 384, 640])
def test_prefetch_nonuniform_graph_live(k):
    from tests._reference.helpers import require_b12x
    require_b12x()
    from b12x.gemm.mgroup_fp8_gemm._joint import compile_joint
    from b12x.gemm.mgroup_fp8_gemm._contiguous_packing import workspace_sizes, pack_contiguous
    rows, n, groups = 4096, 256, 3
    a = torch.ones((rows, k), device='cuda', dtype=torch.float8_e4m3fn)
    b = torch.ones((groups, n, k), device='cuda', dtype=torch.float8_e4m3fn)
    sfa = (2. ** ((torch.arange(rows, device='cuda')[:, None] + torch.arange(k//32, device='cuda')[None, :]) % 5 - 2)).float()
    sfb = torch.ones((groups, n, k//128), device='cuda')
    wa, wb = [torch.empty(v, device='cuda', dtype=torch.uint8) for v in workspace_sizes(rows,n,k,groups,compact=True)]
    labels = (torch.arange(rows, device='cuda',dtype=torch.int32)//128)%groups
    selector = torch.zeros(1,device='cuda',dtype=torch.int32)
    alpha = torch.ones(1,device='cuda');out=torch.empty((rows,n),device='cuda',dtype=torch.bfloat16)
    sm=torch.cuda.get_device_properties(0).multi_processor_count
    gemm=compile_joint(n,k,groups,rows,sm,sfa_prefetch=True)
    def invoke():
        pack_contiguous(sfa,sfb,wa,wb,labels,selector,rows=rows,capacity=rows,n=n,k=k,groups=groups,compact=True,use_selector=False)
        gemm(a,labels,b,wa,wb,out,alpha,selector)
    invoke();torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):invoke()
    try:
        for empty in (False,True,False):
            labels.copy_(torch.full_like(labels,-1) if empty else (torch.arange(rows,device='cuda',dtype=torch.int32)//128)%groups)
            sfa.mul_(2)
            out.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
            if not empty:
                expected=sfa.sum(1)*32
                torch.testing.assert_close(out,expected[:,None].expand(-1,n).to(out.dtype),rtol=0,atol=0)
    finally:graph.reset()


def test_prefetch_eager_memcheck():
    from tests._reference.helpers import require_b12x
    require_b12x()
    from b12x.gemm.mgroup_fp8_gemm._joint import compile_joint
    from b12x.gemm.mgroup_fp8_gemm._contiguous_packing import workspace_sizes, pack_contiguous
    rows,n,k,g=512,128,384,2
    a=torch.ones((rows,k),device='cuda',dtype=torch.float8_e4m3fn)
    b=torch.ones((g,n,k),device='cuda',dtype=torch.float8_e4m3fn)
    sa=torch.ones((rows,k//32),device='cuda');sa[:,1::2]=2
    sb=torch.ones((g,n,k//128),device='cuda')
    wa,wb=[torch.empty(v,device='cuda',dtype=torch.uint8) for v in workspace_sizes(rows,n,k,g,compact=True)]
    labels=torch.tensor([0,-1,1,-1],device='cuda',dtype=torch.int32).repeat_interleave(128)
    selector=torch.zeros(1,device='cuda',dtype=torch.int32);alpha=torch.ones(1,device='cuda')
    out=torch.empty((rows,n),device='cuda',dtype=torch.bfloat16)
    gemm=compile_joint(n,k,g,rows,torch.cuda.get_device_properties(0).multi_processor_count,sfa_prefetch=True)
    pack_contiguous(sa,sb,wa,wb,labels,selector,rows=rows,capacity=rows,n=n,k=k,groups=g,compact=True)
    gemm(a,labels,b,wa,wb,out,alpha,selector);torch.cuda.synchronize()
    torch.testing.assert_close(out[labels>=0],torch.full_like(out[labels>=0],576),rtol=0,atol=0)


def test_prefetch_sfa_actual_load_past_2gib():
    from tests._reference.helpers import require_b12x
    require_b12x()
    from b12x.gemm.mgroup_fp8_gemm._joint import compile_joint
    rows,n,k=131072,128,524416
    free,_=torch.cuda.mem_get_info()
    required=rows*k+(rows//128)*(k//128)*512+n*k
    if free <= required+4*1024**3:
        pytest.skip('requires about 70GiB free')
    a=torch.empty((rows,k),device='cuda',dtype=torch.float8_e4m3fn);a[-128:].fill_(1)
    b=torch.ones((1,n,k),device='cuda',dtype=torch.float8_e4m3fn)
    sa=torch.empty((rows//128)*(k//128)*512,device='cuda',dtype=torch.uint8);sa[-(k//128)*512:].fill_(127)
    sb=torch.full((((k+511)//512)*512,),127,device='cuda',dtype=torch.uint8)
    labels=torch.full((rows,),-1,device='cuda',dtype=torch.int32);labels[-128:]=0
    out=torch.empty((rows,n),device='cuda',dtype=torch.bfloat16)
    alpha=torch.ones(1,device='cuda');selector=torch.zeros(1,device='cuda',dtype=torch.int32)
    gemm=compile_joint(n,k,1,rows,torch.cuda.get_device_properties(0).multi_processor_count,sfa_prefetch=True)
    assert ((rows//128-1)*(k//128)+k//128-1)*512+510>2**31
    gemm(a,labels,b,sa,sb,out,alpha,selector);torch.cuda.synchronize()
    torch.testing.assert_close(out[-128:],torch.full_like(out[-128:],k),rtol=0,atol=0)
