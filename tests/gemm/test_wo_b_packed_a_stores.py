import ast
from pathlib import Path
from types import SimpleNamespace

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Uint8, Uint32
from cutlass.utils import LayoutEnum, SmemAllocator
import pytest
import torch

from b12x._lib.compiler import compile as compile_kernel
from b12x._lib.dense_gemm import DenseGemmKernel
from b12x._lib.intrinsics import shared_ptr_to_u32, st_shared_u32
from b12x._lib.utils import current_cuda_stream, make_ptr
from tests.gemm.test_wo_b_named_publication import eligible, scope_state


def packed_expression():
    from b12x._lib import dense_gemm
    tree = ast.parse(Path(dense_gemm.__file__).read_text())
    return next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Attribute) and t.attr == 'wo_b_packed_a_stores'
                        for t in n.targets))


def packed_eligible(state, k_major=True):
    state = dict(state, wo_b_named_publication=eligible(state),
                 a_layout=SimpleNamespace(is_k_major_a=lambda: k_major))
    return eval(compile(ast.Expression(packed_expression()), '<packed-scope>', 'eval'),
                {'self': SimpleNamespace(**state)})


def test_packed_scope():
    assert packed_eligible(scope_state(4))
    assert not packed_eligible(scope_state(1))
    assert not packed_eligible(scope_state(4), False)
    for field, value in [('mgroup_labels', True), ('mgroup_masked', True),
                         ('mxfp6_fmt_a', 'e4m3'), ('mxfp6_fmt_b', 'e2m3'),
                         ('fused_quant_a_wide', True), ('split_k_slices', 2),
                         ('tile_shape_mnk', (16, 64, 128)), ('ab_stage', 3),
                         ('fused_quant_a_inner_span', 512), ('b_packed', True)]:
        state = scope_state(4)
        state[field] = value
        assert not packed_eligible(state), field
    assert not any(isinstance(n, ast.Attribute) and n.attr in ('m', 'expected_m', 'live_m')
                   for n in ast.walk(packed_expression()))


def test_packed_swizzle_addresses():
    occupied = set()
    for stage in range(4):
        addresses = set()
        for lane in range(32):
            row, group = divmod(lane, 4)
            for word in range(8):
                x = stage * 2048 + row * 128 + group * 32 + word * 4
                physical = x ^ ((x >> 3) & 0x70)
                assert physical % 4 == 0
                for byte in range(4):
                    logical = x + byte
                    assert logical ^ ((logical >> 3) & 0x70) == physical + byte
                    assert physical + byte not in addresses
                    addresses.add(physical + byte)
        assert len(addresses) == 1024
        assert not occupied.intersection(addresses)
        occupied.update(addresses)
    assert len(occupied) == 4096


class PackedStageProbe:
    def __init__(self, wide=False):
        self.wide = wide

    @cute.jit
    def __call__(self, payload: cute.Pointer, scales: cute.Pointer,
                 old: cute.Pointer, new: cute.Pointer, sf: cute.Pointer,
                 rows: Int32, stream):
        self.kernel(payload, scales, old, new, sf, rows).launch(
            grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, payload: cute.Pointer, scales: cute.Pointer,
               old: cute.Pointer, new: cute.Pointer, sf: cute.Pointer, rows: Int32):
        lane = Int32(cute.arch.thread_idx()[0])
        row = lane // Int32(4)
        group = lane % Int32(4)
        alloc = SmemAllocator()
        raw_old = alloc.allocate_tensor(Uint8, cute.make_layout(8192), 1024)
        raw_new = alloc.allocate_tensor(Uint8, cute.make_layout(8192), 1024)
        raw_sf = alloc.allocate_tensor(Uint8, cute.make_layout(128), 16)
        layout = DenseGemmKernel._make_smem_layouts(
            (16, 128, 128), (16, 128), cutlass.Float8E4M3FN, LayoutEnum.ROW_MAJOR,
            cutlass.Float8E4M3FN, LayoutEnum.COL_MAJOR, 4,
            cutlass.BFloat16, LayoutEnum.ROW_MAJOR, 1, 32, None, block_fp8=True)[0]
        assert cute.cosize(layout.outer) == 8192
        old_view = cute.make_tensor(
            cute.recast_ptr(raw_old.iterator, swizzle_=layout.inner), layout.outer)
        base = shared_ptr_to_u32(raw_new.iterator)
        for tile in range(32):
            stage = Int32(tile % 4)
            for i in range(256):
                raw_old[lane + i * 32] = Uint8(165)
                raw_new[lane + i * 32] = Uint8(165)
            raw_sf[stage * 32 + lane] = Uint8(165)
            cute.arch.sync_threads()
            if cutlass.const_expr(self.wide):
                if lane // Int32(4) < Int32(4):
                    for word in cutlass.range_constexpr(2):
                        value = payload[(tile * 32 + lane) * 8 + word]
                        k = (lane // Int32(4) % Int32(4)) * Int32(32) + lane % Int32(4) * Int32(8) + Int32(word * 4)
                        for byte in cutlass.range_constexpr(4):
                            old_view[(Int32(0), k + byte, stage)] = Uint8(value >> Uint32(8 * byte))
                        x = stage * Int32(2048) + k
                        physical = x ^ ((x >> Int32(3)) & Int32(0x70))
                        st_shared_u32(base + physical, value)
                    if lane % Int32(4) == Int32(0):
                        raw_sf[stage * 32 + lane] = scales[tile * 32 + lane]
            elif row < rows:
                for word in cutlass.range_constexpr(8):
                    value = payload[(tile * 32 + lane) * 8 + word]
                    k = group * Int32(32) + Int32(word * 4)
                    for byte in cutlass.range_constexpr(4):
                        old_view[(row, k + byte, stage)] = Uint8(value >> Uint32(8 * byte))
                    x = stage * Int32(2048) + row * Int32(128) + k
                    physical = x ^ ((x >> Int32(3)) & Int32(0x70))
                    st_shared_u32(base + physical, value)
                raw_sf[stage * 32 + lane] = scales[tile * 32 + lane]
            cute.arch.fence_proxy('async.shared', space='cta')
            cute.arch.sync_threads()
            for i in range(256):
                index = lane + i * 32
                old[tile * 8192 + index] = raw_old[index]
                new[tile * 8192 + index] = raw_new[index]
            sf[tile * 32 + lane] = raw_sf[stage * 32 + lane]
            cute.arch.sync_threads()


@pytest.mark.parametrize('m', range(1, 9))
def test_packed_payload_and_scale_bytes(m):
    check_payload_bytes(m, False)


def test_wide_packed_payload_and_scale_bytes():
    check_payload_bytes(1, True)


def check_payload_bytes(m, wide):
    from tests._reference.helpers import require_b12x
    require_b12x()
    values = torch.arange(32 * 32 * 32, device='cuda', dtype=torch.int64)
    values = ((values * 37 + values // 11) % 256).to(torch.uint8).reshape(32, 32, 32)
    values[::5].zero_()
    payload = values.contiguous().view(torch.uint32).flatten()
    scales = ((torch.arange(1024, device='cuda') * 13) % 254).to(torch.uint8).reshape(32, 32)
    scales[::5].fill_(127)
    old = torch.empty(32 * 8192, device='cuda', dtype=torch.uint8)
    new = torch.empty_like(old)
    sf = torch.empty(1024, device='cuda', dtype=torch.uint8)
    ptrs = [make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
            for dtype, t in [(Uint32, payload), (Uint8, scales), (Uint8, old),
                             (Uint8, new), (Uint8, sf)]]
    fn = compile_kernel(PackedStageProbe(wide), *ptrs, Int32(m), current_cuda_stream())
    fn(*ptrs, Int32(m), current_cuda_stream())
    torch.testing.assert_close(new, old, rtol=0, atol=0)
    expected_sf = torch.full_like(scales, 165)
    lanes = range(0, 16, 4) if wide else range(min(m * 4, 32))
    for lane in lanes:
        expected_sf[:, lane] = scales[:, lane]
    torch.testing.assert_close(sf.reshape(32, 32), expected_sf, rtol=0, atol=0)
    expected = torch.full((32, 8192), 165, dtype=torch.uint8)
    source = values.cpu()
    for tile in range(32):
        for lane in range(16 if wide else min(m * 4, 32)):
            for byte in range(8 if wide else 32):
                local = (lane // 4) * 32 + (lane % 4) * 8 if wide else (lane // 4) * 128 + (lane % 4) * 32
                x = (tile % 4) * 2048 + local + byte
                expected[tile, x ^ ((x >> 3) & 0x70)] = source[tile, lane, byte]
    torch.testing.assert_close(new.cpu().reshape(32, 8192), expected, rtol=0, atol=0)


@pytest.mark.parametrize('m', range(2, 9))
def test_packed_production_small_m_oracle(m, monkeypatch):
    from tests.gemm.test_wo_b_named_publication import test_wo_b_named_quantized_oracle_and_graph
    test_wo_b_named_quantized_oracle_and_graph(m, monkeypatch)


def test_wide_scope_and_addresses():
    from b12x._lib import dense_gemm
    tree = ast.parse(Path(dense_gemm.__file__).read_text())
    expr = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Attribute) and t.attr == 'wo_b_wide_packed_a_stores'
                        for t in n.targets))
    def accepts(state, k_major=True):
        state = dict(state, wo_b_named_publication=eligible(state),
                     a_layout=SimpleNamespace(is_k_major_a=lambda: k_major))
        return eval(compile(ast.Expression(expr), '<wide-scope>', 'eval'),
                    {'self': SimpleNamespace(**state)})
    assert accepts(scope_state(1))
    assert not accepts(scope_state(4))
    assert not accepts(scope_state(1), False)
    for field, value in [('split_k_atomic_bf16', False), ('mgroup_labels', True),
                         ('mgroup_masked', True), ('mxfp6_fmt_a', 'e4m3'),
                         ('split_k_slices', 1), ('fused_quant_a_wide', False)]:
        state = scope_state(1)
        state[field] = value
        assert not accepts(state)
    occupied = set()
    for stage in range(4):
        for lane in range(32):
            raw, lane4 = divmod(lane, 4)
            if raw >= 4:
                continue
            for word in range(2):
                x = stage * 2048 + (raw % 4) * 32 + lane4 * 8 + word * 4
                physical = x ^ ((x >> 3) & 0x70)
                assert physical % 4 == 0
                for byte in range(4):
                    assert ((x + byte) ^ (((x + byte) >> 3) & 0x70)) == physical + byte
                    assert physical + byte not in occupied
                    occupied.add(physical + byte)
    assert len(occupied) == 4 * 128
