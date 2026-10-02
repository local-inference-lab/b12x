import ast
from pathlib import Path

import pytest


def test_role_local_static_identity_and_scope():
    from b12x.gemm.mgroup_fp8_gemm._joint import _JointLaunch, _Body
    base = _JointLaunch(4096, 6144, 64, 16384, 188)
    candidate = _JointLaunch(4096, 6144, 64, 16384, 188, role_local_scheduler=True)
    assert base.compile_key()[0] == 'mgroup-joint-v1'
    assert candidate.compile_key()[0] == 'mgroup-joint-role-local-v1'
    assert candidate.compile_key()[1:] == base.compile_key()[1:]
    assert not base.full.mgroup_role_local_scheduler
    assert candidate.full.mgroup_role_local_scheduler
    assert not candidate.narrow.mgroup_role_local_scheduler
    assert not candidate.full.mgroup_sfa_prefetch
    assert not candidate.narrow.mgroup_sfa_prefetch
    assert candidate.full.mma_register_requirement == base.full.mma_register_requirement
    assert candidate.full.load_register_requirement == base.full.load_register_requirement
    with pytest.raises(ValueError, match='separate experiments'):
        _JointLaunch(4096, 6144, 64, 16384, 188, True, True)
    with pytest.raises(ValueError, match='joint full'):
        _Body((128, 64, 128), 16384, role_local_scheduler=True)
    from b12x._lib.dense_gemm import DenseGemmKernel
    with pytest.raises(ValueError, match='joint full'):
        DenseGemmKernel(32, (128, 128), (1, 1), mgroup_role_local_scheduler=True)


def test_role_local_initializers_identical():
    from b12x._lib import dense_gemm
    tree = ast.parse(Path(dense_gemm.__file__).read_text())
    branches = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                and ast.unparse(node.test) == 'cutlass.const_expr(self.mgroup_role_local_scheduler)']
    assert len(branches) == 3
    placeholder = next(node for node in branches if node.orelse)
    roles = [node for node in branches if not node.orelse]
    assert ast.dump(ast.Module(body=roles[0].body, type_ignores=[])) == ast.dump(ast.Module(body=roles[1].body, type_ignores=[]))
    assert [ast.unparse(node) for node in placeholder.body[:3]] == [
        'mgroup_nt = Int32(0)', 'mgroup_ord = Int32(0)', 'mgroup_stride = Int32(0)']
    assert [ast.unparse(node) for node in roles[0].body] == [
        'mgroup_nt = Int32(tile_sched_params.problem_shape_ntile_mnl[1])',
        'mgroup_ord = Int32(cute.arch.block_idx()[2])',
        'mgroup_stride = Int32(cute.arch.grid_dim()[2])',
        'work_tile = self._mgroup_joint_tile(mgroup_ord, sPrefix, mgroup_nt, mgroup_hi)']


@pytest.mark.parametrize('rows,n,k', [(129, 128, 128), (17*128+37, 384, 384), (129*128+37, 512, 640)])
def test_role_local_persistent_graph_live(rows, n, k):
    import torch
    from tests._reference.helpers import require_b12x
    require_b12x()
    from b12x.gemm.mgroup_fp8_gemm._joint import compile_joint
    from b12x.gemm.mgroup_fp8_gemm._contiguous_packing import workspace_sizes, pack_contiguous, zero_padding
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.preparation._measurement import no_compilation
    groups = 3
    capacity = ((rows + 127) // 128 + 2) * 128
    a = torch.ones((rows, k), device='cuda', dtype=torch.float8_e4m3fn)
    b = torch.stack([torch.full((n, k), g+1, device='cuda', dtype=torch.float32) for g in range(groups)]).to(torch.float8_e4m3fn)
    sfa = (2. ** ((torch.arange(rows, device='cuda')[:, None] + torch.arange(k//32, device='cuda')[None, :]) % 5 - 2)).float()
    sfb = torch.ones((groups, n, k//128), device='cuda')
    wa, wb = [torch.empty(v, device='cuda', dtype=torch.uint8) for v in workspace_sizes(capacity, n, k, groups, compact=True)]
    row = torch.arange(rows, device='cuda', dtype=torch.int32)
    tile = row // 128
    labels = tile % groups
    selector = torch.zeros(1, device='cuda', dtype=torch.int32)
    alpha = torch.ones(1, device='cuda')
    out = torch.empty((rows, n), device='cuda', dtype=torch.bfloat16)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    gemm = compile_joint(n, k, groups, capacity, sms, role_local_scheduler=True)
    def invoke():
        pack_contiguous(sfa, sfb, wa, wb, labels, selector, rows=rows, capacity=capacity,
                        n=n, k=k, groups=groups, compact=True, use_selector=False)
        gemm(a, labels, b, wa, wb, out, alpha, selector)
        zero_padding(out, labels, capacity=capacity)
    invoke()
    torch.cuda.synchronize()
    addresses = [v.data_ptr() for v in (a, b, sfa, sfb, wa, wb, labels, selector, out)]
    with kernel_resolution_guard('role-local scheduler live replay'), no_compilation():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            invoke()
        try:
            for pattern in ('all', 'empty', 'sparse', 'all'):
                changed = (tile + (pattern == 'sparse')) % groups
                if pattern == 'empty':
                    changed = torch.full_like(labels, -1)
                elif pattern == 'sparse':
                    changed = torch.where(tile % 3 == 1, -1, changed)
                labels.copy_(changed)
                sfa.mul_(2)
                out.fill_(float('nan'))
                graph.replay()
                torch.cuda.synchronize()
                expected = torch.where(labels >= 0, sfa.sum(1)*32*(labels+1), 0)
                torch.testing.assert_close(out, expected[:, None].expand(-1, n).to(out.dtype), rtol=0, atol=0)
                assert [v.data_ptr() for v in (a, b, sfa, sfb, wa, wb, labels, selector, out)] == addresses
        finally:
            graph.reset()
