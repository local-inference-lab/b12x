import pytest
import torch

from b12x.gemm import mgroup_fp8_gemm as mgg
from b12x._lib.runtime_control import kernel_resolution_guard
from tests._reference.helpers import require_b12x
from tests.gemm.test_mgroup_fp8_gemm import _masked_case, _masked_reference, _zero_masked_tail, _prepared_mgroup


@pytest.mark.parametrize('groups,k,tile_k', [(1,128,128),(3,640,128),(3,640,64)])
def test_compact_masked_eager_edges(groups,k,tile_k):
    require_b12x()
    from b12x.gemm.mgroup_fp8_gemm._packing import pack_grouped_scales_fast
    lhs,rhs,d,counts = _masked_case(groups,35,136,k,[17]+[0]*(groups-1),seed=181)
    source=rhs[1] if groups>1 else rhs[1][0]
    packed=pack_grouped_scales_fast(source,rows=136,k=k,num_groups=groups,gran=128,compact128=True)
    expected=torch.full_like(packed,127)
    r=torch.arange(136,device='cuda')[:,None]
    c=torch.arange(k//128,device='cuda')[None,:]
    atoms=(k+511)//512
    for g in range(groups):
        offsets=((g*2+r//128)*atoms+c//4)*512+(r%32)*16+(r//32%4)*4+c%4
        expected[offsets]=((rhs[1][g].view(torch.int32)>>23)&255).to(torch.uint8)
    torch.testing.assert_close(packed,expected,rtol=0,atol=0)
    q=mgg.MGroupFP8GemmQuery(mode='masked',num_groups=groups,n=136,k=k,m_capacity=35,a_sf_gran=128)
    cfg=mgg.MGroupFP8GemmConfig(backend='cutedsl',tile_m=128,tile_n=128,tile_k=tile_k,implementation='masked_compact')
    with _prepared_mgroup(q,lhs,rhs,d,counts,override=cfg) as plan:
        mgg.masked_mm(lhs,rhs,d,counts,plan=plan)
        torch.cuda.synchronize()
        torch.testing.assert_close(_zero_masked_tail(d,counts).float(),_masked_reference(lhs,rhs,counts).float(),rtol=.02,atol=.02)
        counts.zero_()
        mgg.masked_mm(lhs,rhs,d,counts,plan=plan)
        torch.cuda.synchronize()


def test_compact_masked_stream_graph_live():
    require_b12x()
    lhs,rhs,d,counts=_masked_case(1,35,136,640,[17],seed=191)
    q=mgg.MGroupFP8GemmQuery(mode='masked',num_groups=1,n=136,k=640,m_capacity=35,a_sf_gran=128,expected_m=17)
    cfg=mgg.MGroupFP8GemmConfig(backend='cutedsl',tile_m=32,tile_n=64,tile_k=128,implementation='masked_compact')
    with _prepared_mgroup(q,lhs,rhs,d,counts,override=cfg) as plan:
        stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):mgg.masked_mm(lhs,rhs,d,counts,plan=plan,stream=stream)
        for length in (35,0,1,17):
            with torch.cuda.stream(stream):
                counts.fill_(length);rhs[1].mul_(2);lhs[1].mul_(.5)
            with kernel_resolution_guard():mgg.masked_mm(lhs,rhs,d,counts,plan=plan,stream=stream)
            stream.synchronize()
            with torch.cuda.stream(stream),kernel_resolution_guard():graph.replay()
            stream.synchronize()
            torch.testing.assert_close(_zero_masked_tail(d,counts).float(),_masked_reference(lhs,rhs,counts).float(),rtol=.02,atol=.02)
        graph.reset()
