import ast
from pathlib import Path

import pytest
import torch

from b12x.gemm.mgroup_fp8_gemm._contiguous_packing import normalize_g1, workspace_sizes


@pytest.mark.parametrize('rows,n,k,g', [(0,128,128,1),(1,136,640,1),(129,257,2304,2)])
def test_workspace_sizes(rows,n,k,g):
    a,b=workspace_sizes(rows,n,k,g,compact=True)
    assert a==((rows+127)//128)*(k//128)*512
    assert b==g*((n+127)//128)*((k+511)//512)*512
    assert workspace_sizes(rows,n,k,g,compact=False)[1]==g*((n+127)//128)*(k//128)*512


def test_g1_view_is_exact():
    x=torch.ones(1,35,5)
    y=normalize_g1(x,35,5,1)
    assert y.shape==(35,5) and y.data_ptr()==x.data_ptr()
    assert normalize_g1(x,34,5,1) is x
    assert normalize_g1(x,35,5,2) is x


def test_live_rows_not_specialized():
    from b12x.gemm.mgroup_fp8_gemm import _contiguous_packing as packing
    tree=ast.parse(Path(packing.__file__).read_text())
    for name in ('_pack_contiguous','_zero_flat'):
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        arg=next(a for a in fn.args.args if a.arg=='rows')
        assert arg.annotation is None
        decorator=ast.unparse(fn.decorator_list[0])
        assert "do_not_specialize=['rows']" in decorator
        assert "do_not_specialize_on_alignment=['rows']" in decorator


def test_label_contract_cpu_boundary_model():
    for labels,expected in (([0]*129,True),([0]*64+[1]*64,False),([-1]*128+[1],True),([0]*64+[-1]*64+[0],True),([0]*129+[1],False),([2]*128,False),([-2],False)):
        valid=all(-1<=g<2 and (g<0 or i%128==0 or g==labels[i-1]) for i,g in enumerate(labels))
        assert valid==expected


def test_gpu_combined_bytes_live_rows_and_stream():
    from tests._reference.helpers import require_b12x
    require_b12x()
    from b12x.gemm.mgroup_fp8_gemm._contiguous_packing import pack_contiguous, zero_padding
    from b12x.gemm.mgroup_fp8_gemm._packing import pack_grouped_scales
    capacity,n,k,g=257,136,640,1
    device=torch.device('cuda')
    stream=torch.cuda.Stream()
    a=torch.ones((capacity,k//32),device=device)
    b=torch.ones((g,n,k//128),device=device)
    labels=torch.zeros(capacity,device=device,dtype=torch.int32)
    selector=torch.empty(1,device=device,dtype=torch.int32)
    d=torch.ones((capacity,n),device=device,dtype=torch.bfloat16)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for compact in (False,True):
            oa,ob=[torch.empty(s,device=device,dtype=torch.uint8) for s in workspace_sizes(capacity,n,k,g,compact=compact)]
            for rows in (1,129,257,0,1):
                a.fill_(2.);b.fill_(.5)
                pack_contiguous(a[:rows],b,oa,ob,labels[:rows],selector,rows=rows,capacity=capacity,n=n,k=k,groups=g,compact=compact)
                if rows:
                    ref=pack_grouped_scales(a[:rows],rows=rows,k=k,num_groups=1,gran=32)
                    raw=ref.permute(5,2,4,0,1,3).contiguous().view(torch.uint8).flatten()
                    torch.testing.assert_close(oa[:raw.numel()],raw,rtol=0,atol=0)
                if not compact:
                    ref=pack_grouped_scales(b[0],rows=n,k=k,num_groups=1,gran=128)
                    raw=ref.permute(5,2,4,0,1,3).contiguous().view(torch.uint8).flatten()
                    torch.testing.assert_close(ob,raw,rtol=0,atol=0)
                else:
                    columns=k//128;atoms=(columns+3)//4
                    padded=torch.full((g,((n+127)//128)*128,atoms*4),127,device=device,dtype=torch.uint8)
                    padded[:,:n,:columns]=((b.view(torch.int32)>>23)&255).to(torch.uint8)
                    raw=padded.reshape(g,(n+127)//128,4,32,atoms,4).permute(0,1,4,3,2,5).contiguous().flatten()
                    torch.testing.assert_close(ob,raw,rtol=0,atol=0)
            labels.fill_(-1)
            zero_padding(d,labels,capacity=capacity)
    stream.synchronize()
    assert torch.count_nonzero(d)==0


def test_gpu_public_nondefault_stream_and_plan_workspace():
    from tests._reference.helpers import require_b12x
    require_b12x()
    from tests.gemm.test_mgroup_fp8_gemm import _contiguous_case, _contiguous_reference, _prepared_mgroup
    from b12x.gemm import mgroup_fp8_gemm as mgg
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.preparation._measurement import no_compilation
    lhs,rhs,d,labels,intervals=_contiguous_case(1,136,640,[257],128,seed=73)
    query=mgg.MGroupFP8GemmQuery(mode='contiguous',num_groups=1,n=136,k=640,m_capacity=257,a_sf_gran=32)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with _prepared_mgroup(query,lhs,rhs,d,labels) as plan:
        state=plan._prepared.state
        pointers=tuple(t.data_ptr() for t in (state.sfa_workspace,state.sfb_workspace,state.selector))
        stream.wait_stream(torch.cuda.current_stream())
        with kernel_resolution_guard('fixed plan unseen live row views'),no_compilation():
            for rows in (257,1,129,257):
                live_lhs=(lhs[0][:rows],lhs[1][:rows])
                mgg.contiguous_mm(live_lhs,rhs,d[:rows],labels[:rows],plan=plan,stream=stream.cuda_stream)
                stream.synchronize()
                ref=_contiguous_reference(live_lhs,rhs,[(0,rows)])
                torch.testing.assert_close(d[:rows].float(),ref.float(),rtol=.02,atol=.02)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            mgg.contiguous_mm(lhs,rhs,d,labels,plan=plan,stream=stream.cuda_stream)
        for scale in (2.,.5):
            with torch.cuda.stream(stream):
                lhs[1].mul_(scale);rhs[1].mul_(scale)
                graph.replay()
            stream.synchronize()
            torch.testing.assert_close(d.float(),_contiguous_reference(lhs,rhs,intervals).float(),rtol=.02,atol=.02)
        graph.reset()
        assert pointers==tuple(t.data_ptr() for t in (state.sfa_workspace,state.sfb_workspace,state.selector))


@pytest.mark.parametrize('bad', ['range','negative','unaligned'])
def test_gpu_bad_labels_trap_in_isolated_process(bad):
    from tests._reference.helpers import require_b12x
    require_b12x()
    import subprocess
    import sys
    code='''
import torch
from b12x.gemm.mgroup_fp8_gemm._contiguous_packing import pack_contiguous,workspace_sizes
rows,n,k,g=129,136,128,2
a=torch.ones((rows,k//32),device='cuda');b=torch.ones((g,n,k//128),device='cuda')
labels=torch.zeros(rows,device='cuda',dtype=torch.int32)
oa,ob=[torch.empty(s,device='cuda',dtype=torch.uint8) for s in workspace_sizes(rows,n,k,g,compact=False)]
selector=torch.empty(1,device='cuda',dtype=torch.int32)
pack_contiguous(a,b,oa,ob,labels,selector,rows=rows,capacity=rows,n=n,k=k,groups=g,compact=False)
torch.cuda.synchronize()
graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    pack_contiguous(a,b,oa,ob,labels,selector,rows=rows,capacity=rows,n=n,k=k,groups=g,compact=False)
labels[64]=BAD
try:
    graph.replay();torch.cuda.synchronize()
except RuntimeError as error:
    print('EXPECTED_DEVICE_ERROR',str(error),flush=True)
    __import__('os')._exit(0)
__import__('os')._exit(7)
'''.replace('BAD',str({'range':2,'negative':-2,'unaligned':1}[bad]))
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=120)
    assert result.returncode==0 and 'EXPECTED_DEVICE_ERROR' in result.stdout,result.stdout+result.stderr


def test_active_group_packing_live_graph_and_repeated_groups():
    from tests._reference.helpers import require_b12x
    require_b12x()
    from b12x.gemm.mgroup_fp8_gemm._contiguous_packing import pack_contiguous
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.preparation._measurement import no_compilation

    capacity, n, k, groups = 513, 136, 640, 4
    a = torch.ones((capacity, k // 32), device='cuda')
    b = torch.ones((groups, n, k // 128), device='cuda')
    labels = torch.zeros(capacity, device='cuda', dtype=torch.int32)
    oa, ob = [torch.empty(s, device='cuda', dtype=torch.uint8)
              for s in workspace_sizes(capacity, n, k, groups, compact=True)]
    ra, rb = [torch.empty_like(t) for t in (oa, ob)]
    selector = torch.empty(groups + 1, device='cuda', dtype=torch.int32)
    reference_selector = torch.empty(1, device='cuda', dtype=torch.int32)
    def run(rows, reference=False):
        return pack_contiguous(a[:rows], b, ra if reference else oa, rb if reference else ob,
            labels[:rows], reference_selector if reference else selector, rows=rows,
            capacity=capacity, n=n, k=k, groups=groups, compact=True, use_selector=True, ctas=188)
    run(capacity)
    run(capacity, True)
    pointers = [t.data_ptr() for t in (oa, ob, selector)]
    with kernel_resolution_guard('active group live rows'), no_compilation():
        for rows in (capacity, 1, 129, 0, capacity):
            labels.fill_(-1)
            labels[:min(rows, 128)] = 2
            labels[256:min(rows, 384)] = 2
            labels[512:rows] = 1
            ob.fill_(17)
            run(rows)
            run(rows, True)
            torch.cuda.synchronize()
            active = {int(x) for x in labels[:rows].cpu().tolist() if x >= 0}
            assert selector[1:].tolist() == [int(g in active) for g in range(groups)]
            actual, expected = ob.reshape(groups, -1), rb.reshape(groups, -1)
            for g in range(groups):
                if g in active:
                    torch.testing.assert_close(actual[g], expected[g], rtol=0, atol=0)
                else:
                    assert torch.all(actual[g] == 17)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(capacity)
        for group in (3, -1, 0, 2):
            labels.fill_(group)
            b.fill_(2. if group % 2 else .5)
            ob.fill_(29)
            before = torch.cuda.memory_stats()
            graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            assert after['allocation.all.allocated'] == before['allocation.all.allocated']
            run(capacity, True)
            torch.cuda.synchronize()
            assert selector[1:].tolist() == [int(g == group) for g in range(groups)]
            if group >= 0:
                torch.testing.assert_close(ob.reshape(groups, -1)[group], rb.reshape(groups, -1)[group], rtol=0, atol=0)
            else:
                assert torch.all(ob == 29)
        graph.reset()
    assert pointers == [t.data_ptr() for t in (oa, ob, selector)]
