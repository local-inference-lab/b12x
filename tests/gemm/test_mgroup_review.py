import pytest
import torch

from b12x.gemm import mgroup_fp8_gemm as mgg
from b12x._lib.runtime_control import kernel_resolution_guard
from tests._reference.helpers import require_b12x
from tests.gemm.test_mgroup_fp8_gemm import (
    _masked_case, _masked_reference, _zero_masked_tail, _prepared_mgroup,
    _contiguous_case,
)


def test_contiguous_plans_are_private():
    query = mgg.MGroupFP8GemmQuery(mode='contiguous', num_groups=2, n=128, k=256, m_capacity=256, a_sf_gran=32)
    plans = [mgg.plan(query), mgg.plan(query)]
    assert all(not plan.shared for plan in plans)


@pytest.mark.parametrize('groups', [1, 8])
def test_masked_bk64_empty_tail_and_explicit_stream(groups):
    require_b12x()
    lhs, rhs, d, counts = _masked_case(groups, 257, 136, 640, [17] + [0] * (groups - 1), seed=133)
    query = mgg.MGroupFP8GemmQuery(mode='masked', num_groups=groups, n=136, k=640, m_capacity=257, a_sf_gran=128)
    config = mgg.MGroupFP8GemmConfig(backend='cutedsl', tile_m=128, tile_n=128, tile_k=64)
    with _prepared_mgroup(query, lhs, rhs, d, counts, override=config) as plan:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            mgg.masked_mm(lhs, rhs, d, counts, plan=plan, stream=stream)
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            mgg.masked_mm(lhs, rhs, d, counts, plan=plan, stream=stream)
        for length in [0, 35, 0, 17]:
            with torch.cuda.stream(stream):
                counts.zero_()
                counts[0] = length
                lhs[1].mul_(2)
                rhs[1].mul_(.5)
            with kernel_resolution_guard():
                mgg.masked_mm(lhs, rhs, d, counts, plan=plan, stream=stream)
            stream.synchronize()
            torch.testing.assert_close(_zero_masked_tail(d, counts).float(), _masked_reference(lhs, rhs, counts).float(), rtol=.02, atol=.02)
            with torch.cuda.stream(stream), kernel_resolution_guard():
                graph.replay()
            stream.synchronize()
            torch.testing.assert_close(_zero_masked_tail(d, counts).float(), _masked_reference(lhs, rhs, counts).float(), rtol=.02, atol=.02)
        graph.reset()


def test_single_labels_empty_and_private_concurrent_graphs():
    require_b12x()
    from b12x.preparation import PreparationSession, PreparedCall
    query = mgg.MGroupFP8GemmQuery(mode='contiguous', num_groups=2, n=136, k=640, m_capacity=256, a_sf_gran=32)
    config = mgg.MGroupFP8GemmConfig(backend='cutedsl', tile_m=128, tile_n=128, tile_k=128)
    plans = [mgg.plan(query, override=config) for _ in range(2)]
    operands = [_contiguous_case(2, 136, 640, [128, 128], 128, seed=151+i)[:4] for i in range(2)]
    def request(plan, args, i):
        return plan.request(name=f'private-{i}', prepare_call=lambda state: PreparedCall(run=lambda: state.run_contiguous(*args)))
    with PreparationSession(device=torch.device('cuda'), autotune=False, compile_workers=2) as session:
        session.prepare(tuple(request(p, args, i) for i, (p, args) in enumerate(zip(plans, operands))))
        session.freeze()
        assert plans[0]._prepared.state is not plans[1]._prepared.state
        for name in ('sfa_workspace', 'sfb_workspace', 'selector'):
            assert getattr(plans[0]._prepared.state, name).data_ptr() != getattr(plans[1]._prepared.state, name).data_ptr()
        streams = [torch.cuda.Stream() for _ in plans]
        graphs = [torch.cuda.CUDAGraph() for _ in plans]
        for stream, graph, plan, args in zip(streams, graphs, plans, operands):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.graph(graph, stream=stream):
                mgg.contiguous_mm(*args, plan=plan, stream=stream)
        for iteration in range(4):
            for i, (stream, graph, args) in enumerate(zip(streams, graphs, operands)):
                lhs, rhs, d, labels = args
                with torch.cuda.stream(stream), kernel_resolution_guard():
                    labels.fill_(-1 if iteration == 0 else i)
                    lhs[1].mul_(2 if i else .5)
                    rhs[1].mul_(.5 if i else 2)
                    graph.replay()
            for stream in streams:
                stream.synchronize()
            for lhs, rhs, d, labels in operands:
                a = lhs[0].float() * lhs[1].repeat_interleave(32, dim=1)
                b = rhs[0].float() * rhs[1].repeat_interleave(128, dim=2)
                ref = torch.zeros_like(d)
                for g in range(2):
                    mask = labels == g
                    ref[mask] = (a[mask] @ b[g].T).to(torch.bfloat16)
                torch.testing.assert_close(d.float(), ref.float(), rtol=.02, atol=.02)
        for graph in graphs:
            graph.reset()


def test_joint_sfa_actual_load_past_2gib():
    require_b12x()
    from b12x.gemm.mgroup_fp8_gemm._joint import compile_joint
    rows, n, k = 131072, 128, 524416
    free, _ = torch.cuda.mem_get_info()
    required = rows * k + (rows // 128) * (k // 128) * 512 + n * k
    if free < required + 4 * 1024**3:
        pytest.skip('joint large-SFA load requires about 70 GiB free')
    a = torch.empty((rows, k), device='cuda', dtype=torch.float8_e4m3fn)
    b = torch.empty((1, n, k), device='cuda', dtype=torch.float8_e4m3fn)
    a[-128:].fill_(1)
    b.fill_(1)
    sfa = torch.empty((rows // 128) * (k // 128) * 512, device='cuda', dtype=torch.uint8)
    sfa[-(k // 128) * 512:].fill_(127)
    sfb = torch.full((((k + 511) // 512) * 512,), 127, device='cuda', dtype=torch.uint8)
    labels = torch.full((rows,), -1, device='cuda', dtype=torch.int32)
    labels[-128:] = 0
    out = torch.empty((rows, n), device='cuda', dtype=torch.bfloat16)
    alpha = torch.ones(1, device='cuda', dtype=torch.float32)
    selector = torch.zeros(1, device='cuda', dtype=torch.int32)
    gemm = compile_joint(n, k, 1, rows, torch.cuda.get_device_properties(0).multi_processor_count)
    offset = ((rows // 128 - 1) * (k // 128) + k // 128 - 1) * 512 + 510
    assert offset > 2**31
    gemm(a, labels, b, sfa, sfb, out, alpha, selector, None)
    torch.cuda.synchronize()
    torch.testing.assert_close(out[-128:], torch.full_like(out[-128:], k), rtol=0, atol=0)


def test_masked_rejects_same_shape_transposed_b():
    from tests.gemm.test_mgroup_fp8_gemm import _state, _query, _masked_operands
    query = _query(n=256, k=256)
    lhs, rhs, d, counts = _masked_operands(n=256, k=256)
    transposed = rhs[0].transpose(1, 2)
    assert transposed.shape == rhs[0].shape and not transposed.is_contiguous()
    with pytest.raises(ValueError, match='B must be contiguous'):
        _state(query).run_masked(lhs, (transposed, rhs[1]), d, counts)


def test_masked_rejects_same_shape_transposed_b_gpu():
    require_b12x()
    lhs, rhs, d, counts = _masked_case(2, 35, 256, 256, [17, 0], seed=221)
    query = mgg.MGroupFP8GemmQuery(mode='masked', num_groups=2, n=256, k=256, m_capacity=35, a_sf_gran=128)
    with _prepared_mgroup(query, lhs, rhs, d, counts) as plan:
        with pytest.raises(ValueError, match='B must be contiguous'):
            mgg.masked_mm(lhs, (rhs[0].transpose(1, 2), rhs[1]), d, counts, plan=plan)


@pytest.mark.parametrize('compact_sfb', [False, True])
def test_masked_group_sfa_actual_load_past_2gib(compact_sfb):
    require_b12x()
    from b12x._lib import dense_gemm as dense
    from b12x.gemm.mgroup_fp8_gemm._preparation import _mgroup_policy
    groups, rows, n, k = 1025, 1, 8, 2097280
    atoms = (k + 511) // 512 if compact_sfb else k // 128
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    required = groups * (rows + n) * k + groups * (k // 128 + atoms) * 512
    if free < required + 2 * 1024**3:
        pytest.skip('masked large-SF group test requires about 36 GiB free')
    a = torch.empty((groups, rows, k), device='cuda', dtype=torch.float8_e4m3fn)
    b = torch.empty((groups, n, k), device='cuda', dtype=torch.float8_e4m3fn)
    a[-1].fill_(1)
    b[-1].fill_(1)
    sfa = torch.empty(groups * (k // 128) * 512, device='cuda', dtype=torch.uint8)
    start = (groups - 1) * (k // 128) * 512
    assert start > 2**31
    sfa[start:start + k // 128 * 512].fill_(127)
    sfb = torch.empty(groups * atoms * 512, device='cuda', dtype=torch.uint8)
    assert (groups - 1) * atoms * 512 > 2**31
    sfb[-atoms * 512:].fill_(127)
    counts = torch.zeros(groups, device='cuda', dtype=torch.int32)
    counts[-1] = 1
    out = torch.empty((groups, rows, n), device='cuda', dtype=torch.bfloat16)
    alpha = torch.ones(1, device='cuda')
    gemm = dense._get_compiled_dense_gemm_masked_mgroup(n, k, groups, _mgroup_policy(), (128,128), 64, torch.cuda.get_device_properties(0).multi_processor_count, compact_sfb=compact_sfb)
    gemm(a, counts, b, sfa, sfb, out, alpha, None)
    torch.cuda.synchronize()
    torch.testing.assert_close(out[-1,:128], torch.full_like(out[-1,:128], k), rtol=0, atol=0)
