"""Portable qualification of mixed-rate record selection and operand staging."""

from unittest.mock import patch

import cuda.bindings.driver as cuda
import cutlass as c
import cutlass.cute as cute
import cutlass.utils as utils
import pytest
import torch

from b12x._lib.architecture import architecture_for
from b12x.moe._shared.kernels.sm103.launch import pointer
from b12x.moe._shared.kernels.sm103.trellis_mixed_gemm import RoutedMixedTrellisGemm
from b12x.moe._shared.kernels.w4a16.mixed_trellis import build_projection_tiered_maps
from tests._reference.trellis_decode import native_weight


class InspectMixedOperands:
    def __init__(self, projection):
        self.projection = projection

    @cute.jit
    def __call__(
        self,
        a,
        packed: cute.Pointer,
        lut: cute.Pointer,
        ids: cute.Pointer,
        descriptors: cute.Pointer,
        out: cute.Pointer,
        projection: c.Int32,
        descriptor_stride: c.Int64,
        packed_words: c.Int64,
        offsets: tuple[c.Int64, c.Int64, c.Int64],
        counts: tuple[c.Int32, c.Int32, c.Int32],
        live_routes: c.Int32,
        stage: c.Int32,
        n_base: c.Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            a,
            packed,
            lut,
            ids,
            descriptors,
            out,
            projection,
            descriptor_stride,
            packed_words,
            offsets,
            counts,
            stage,
            n_base,
        ).launch(grid=(live_routes, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        a,
        packed,
        lut,
        ids,
        descriptors,
        out,
        projection: c.Int32,
        descriptor_stride: c.Int64,
        packed_words: c.Int64,
        offsets,
        counts,
        stage: c.Int32,
        n_base: c.Int64,
    ):
        thread, _, _ = cute.arch.thread_idx()
        route, _, _ = cute.arch.block_idx()
        route = c.Int64(route)
        layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3), 0, cute.make_layout((128, 64), stride=(64, 1))
        )
        allocator = utils.SmemAllocator()
        sa = allocator.allocate_tensor(
            c.Float16, layout.outer, 128, swizzle=layout.inner
        )
        sb = allocator.allocate_tensor(
            c.Float16, layout.outer, 128, swizzle=layout.inner
        )
        source = self.projection.input_tensors(a, c.Int64(self.projection.k))
        weights = cute.make_tensor(packed, cute.make_layout(packed_words))
        routes = cute.make_tensor(ids, cute.make_layout(self.projection.capacity))
        table = cute.make_tensor(descriptors, cute.make_layout(3 * descriptor_stride))
        selection = self.projection.select_record(
            table,
            c.Int64(routes[route]),
            projection,
            descriptor_stride,
            packed_words,
            offsets,
            counts,
        )
        for item in c.range(thread, 8192, 128):
            sa[item // 64, item % 64] = c.Float16(float("nan"))
            sb[item // 64, item % 64] = c.Float16(float("nan"))
        cute.arch.barrier()
        self.projection.stage_operands(
            source,
            weights,
            lut,
            selection,
            route,
            c.Int64(self.projection.k),
            stage,
            n_base,
            sa,
            sb,
        )
        cute.arch.barrier()
        output = cute.make_tensor(
            out, cute.make_layout(self.projection.capacity * 16384)
        )
        for item in c.range(thread, 8192, 128):
            output[route * 16384 + c.Int64(item)] = sa[item // 64, item % 64]
            output[route * 16384 + 8192 + c.Int64(item)] = sb[item // 64, item % 64]


def _require_gpu():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 3),
        (12, 0),
        (12, 1),
    ):
        pytest.skip("physical Blackwell GPU required")


@pytest.mark.parametrize("dual_input", [False, True])
@pytest.mark.parametrize("local_bits", [8, 24])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_mixed_records_projection_rows_tails_and_graph(
    local_bits, id_dtype, dual_input
):
    _require_gpu()
    torch.manual_seed(992)
    experts, routes, n, k = 5, 9, 144, 80
    tiers = ([0, 1, 2, 0, 1], [2, 0, 1, 1, 2], [1, 2, 0, 2, 0])
    _, descriptors = build_projection_tiered_maps(
        *tiers,
        tier_slots=(experts,) * 3,
        device=torch.device("cuda"),
        local_index_bits=local_bits,
    )
    offsets = [[0] * 3 for _ in range(3)]
    counts = [[0] * 3 for _ in range(3)]
    decoded = {}
    chunks, cursor = [], 0
    for tier, bits in enumerate((3, 4, 5)):
        for projection in range(3):
            members = [i for i, t in enumerate(tiers[projection]) if t == tier]
            records = torch.randint(
                -32768,
                32768,
                (len(members), k // 16, n // 16, 16 * bits),
                dtype=torch.int16,
            )
            for expert, weight in zip(
                members, native_weight(records, bits, "mcg"), strict=True
            ):
                decoded[projection, expert] = weight.cuda()
            offsets[projection][tier] = cursor
            counts[projection][tier] = len(members)
            chunk = records.view(torch.int32).flatten()
            chunks.append(chunk)
            cursor += chunk.numel()
    packed = torch.cat(chunks).cuda()
    source = torch.randn(routes, k, dtype=torch.float16, device="cuda")
    ids = (torch.arange(routes, device="cuda", dtype=id_dtype) % experts).contiguous()
    ids[5], ids[6] = -1, experts
    if id_dtype == torch.int64:
        ids[7] = 2**32 + 1
    source[5:7] = float("nan")
    lut = torch.zeros(16, dtype=torch.uint8, device="cuda")
    output = torch.empty(routes, 2, 128, 64, device="cuda", dtype=torch.float16)
    args = [
        pointer(dtype, tensor)
        for dtype, tensor in (
            (c.Float16, source),
            (c.Uint32, packed),
            (c.Uint8, lut),
            (c.Int32 if id_dtype == torch.int32 else c.Int64, ids),
            (c.Int32, descriptors),
            (c.Float16, output),
        )
    ]
    descriptor_stride = descriptors.numel() // 3
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    def scalars(projection, live, stage=1, n_base=128):
        return [
            c.Int32(projection),
            c.Int64(descriptor_stride),
            c.Int64(packed.numel()),
            tuple(c.Int64(v) for v in offsets[projection]),
            tuple(c.Int32(v) for v in counts[projection]),
            c.Int32(live),
            c.Int32(stage),
            c.Int64(n_base),
        ]

    alternate = -source
    if dual_input:
        args[0] = (args[0], pointer(c.Float16, alternate), c.Int64(64))
    kernel = InspectMixedOperands(
        RoutedMixedTrellisGemm(
            n,
            k,
            experts,
            routes,
            descriptor_local_bits=local_bits,
            dual_input=dual_input,
        )
    )
    fn = cute.compile(
        kernel,
        *args,
        *scalars(0, 1),
        stream,
        options=f"--gpu-arch={architecture_for(torch.cuda.get_device_capability()).compilation_target}",
    )

    def expected(projection, live, stage=1, n_base=128):
        result = torch.zeros(live, 2, 128, 64, dtype=torch.float16, device="cuda")
        for row, expert in enumerate(ids[:live].cpu().tolist()):
            if 0 <= expert < experts:
                k_len, n_len = min(64, k - stage * 64), min(128, n - n_base)
                result[row, 0, 0, :k_len] = source[row, stage * 64 : stage * 64 + k_len]
                if dual_input:
                    result[row, 0, 1, :k_len] = alternate[
                        row, stage * 64 : stage * 64 + k_len
                    ]
                result[row, 1, :n_len, :k_len] = decoded[projection, expert][
                    n_base : n_base + n_len, stage * 64 : stage * 64 + k_len
                ]
        return result

    with patch.object(
        cute, "compile", side_effect=AssertionError("kernel resolution is frozen")
    ):
        for projection in range(3):
            for live in (routes, 1, 4, 3):
                output.fill_(float("nan"))
                fn(*args, *scalars(projection, live), stream)
                torch.testing.assert_close(
                    output[:live], expected(projection, live), atol=0, rtol=0
                )
                assert torch.isnan(output[live:]).all()
        fn(*args, *scalars(2, routes, 0, 0), stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn(
                *args,
                *scalars(2, routes, 0, 0),
                cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            )
        source[:5].neg_()
        alternate[:5].mul_(0.5)
        ids[:5] = ids[:5].flip(0)
        changed = expected(2, routes, 0, 0)
        addresses = tuple(
            t.data_ptr() for t in (source, ids, descriptors, packed, output)
        )
        allocated = torch.cuda.memory_stats()["allocated_bytes.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocated_bytes.all.allocated"] == allocated
        assert (
            tuple(t.data_ptr() for t in (source, ids, descriptors, packed, output))
            == addresses
        )
        torch.testing.assert_close(output, changed, atol=0, rtol=0)
        # Descriptor-local bounds must reject out-of-payload records, even
        # when their encoded tier is valid for this projection.
        descriptors[:experts].fill_((2 << local_bits) | ((1 << local_bits) - 1))
        fn(*args, *scalars(0, routes), stream)
        assert torch.count_nonzero(output) == 0
        for malformed in (-1, -(2**31), 3 << local_bits):
            descriptors[:experts].fill_(malformed)
            fn(*args, *scalars(0, routes), stream)
            assert torch.count_nonzero(output) == 0
        descriptors[:experts].zero_()
        for field, value in (
            (0, c.Int32(-1)),
            (0, c.Int32(3)),
            (1, c.Int64(experts - 1)),
            (2, c.Int64(0)),
            (3, (c.Int64(-1),) * 3),
            (3, (c.Int64(packed.numel()),) * 3),
            (4, (c.Int32(-1),) * 3),
            (4, (c.Int32(experts + 1),) * 3),
        ):
            invalid = scalars(0, routes)
            invalid[field] = value
            fn(*args, *invalid, stream)
            assert torch.count_nonzero(output) == 0


def test_mixed_record_offsets_cross_int32_word_boundary():
    _require_gpu()
    # E=384 with a wide matrix places local record 205 beyond 2**31 words.
    # Only its first operand tiles are initialized; all preceding records are idle.
    n = k = 8192
    experts, local, bits = 384, 205, 5
    words = (n // 16) * (k // 16) * 8 * bits
    start, total = local * words, (local + 1) * words
    assert start > 2**31
    free, _ = torch.cuda.mem_get_info()
    if free < total * 4 + 1024**3:
        pytest.skip("large-offset test requires about 10 GiB free GPU memory")
    packed = torch.empty(total, dtype=torch.int32, device="cuda")
    _, descriptors = build_projection_tiered_maps(
        [2] * experts,
        [2] * experts,
        [2] * experts,
        tier_slots=(experts,) * 3,
        device=torch.device("cuda"),
        local_index_bits=24,
    )
    source = torch.randn(1, k, dtype=torch.float16, device="cuda")
    ids = torch.tensor([local], dtype=torch.int64, device="cuda")
    lut = torch.zeros(16, dtype=torch.uint8, device="cuda")
    output = torch.full(
        (1, 2, 128, 64), float("nan"), device="cuda", dtype=torch.float16
    )
    expected = torch.zeros_like(output)
    expected[0, 0, 0] = source[0, :64]
    native = torch.randint(-32768, 32768, (1, 4, 8, 16 * bits), dtype=torch.int16)
    decoded = native_weight(native, bits, "mcg")[0].cuda()
    expected[0, 1] = decoded
    for ktile in range(4):
        for ntile in range(8):
            offset = start + (ktile * (n // 16) + ntile) * 8 * bits
            packed[offset : offset + 8 * bits].copy_(
                native[0, ktile, ntile].view(torch.int32)
            )
    args = [
        pointer(t, value)
        for t, value in (
            (c.Float16, source),
            (c.Uint32, packed),
            (c.Uint8, lut),
            (c.Int64, ids),
            (c.Int32, descriptors),
            (c.Float16, output),
        )
    ]
    scalars = [
        c.Int32(0),
        c.Int64(3 * experts),
        c.Int64(total),
        (c.Int64(0),) * 3,
        (c.Int32(0), c.Int32(0), c.Int32(local + 1)),
        c.Int32(1),
        c.Int32(0),
        c.Int64(0),
    ]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn = cute.compile(
        InspectMixedOperands(
            RoutedMixedTrellisGemm(n, k, experts, 1, descriptor_local_bits=24)
        ),
        *args,
        *scalars,
        stream,
        options=f"--gpu-arch={architecture_for(torch.cuda.get_device_capability()).compilation_target}",
    )
    fn(*args, *scalars, stream)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    # A record must fit in full even if the requested operand tiles do fit.
    scalars[2] = c.Int64(total - 1)
    fn(*args, *scalars, stream)
    assert torch.count_nonzero(output) == 0
    del packed
    torch.cuda.empty_cache()
