"""Portable checks of production grouped atom selection and shared operands."""

from dataclasses import replace
from unittest.mock import patch

import cuda.bindings.driver as cuda
import cutlass as c
import cutlass.cute as cute
import cutlass.utils as utils
import pytest
import torch

from b12x._lib.architecture import architecture_for
from b12x.moe import fused_moe
from b12x.moe._shared.kernels.sm103.launch import pointer
from b12x.moe._shared.kernels.sm103.trellis_atoms_gemm import RoutedAtomTrellisGemm
from tests._reference.trellis_atoms import atom_fixture, exl3_atom_fixture
from tests.moe.test_sm103_trellis_mixed_staging import _require_gpu


class InspectAtomOperands:
    def __init__(self, projection):
        self.projection = projection

    @cute.jit
    def __call__(
        self,
        a,
        packed,
        lut,
        ids,
        offsets,
        rates,
        out,
        projection: c.Int32,
        row_stride: c.Int64,
        packed_words: c.Int64,
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
            offsets,
            rates,
            out,
            projection,
            row_stride,
            packed_words,
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
        offsets,
        rates,
        out,
        projection: c.Int32,
        row_stride: c.Int64,
        packed_words: c.Int64,
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
        selection = self.projection.selection(
            ids, offsets, rates, route, projection, row_stride, packed_words
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


def compile_probe(payload, source, ids, output, *, fc1, dual_input=False):
    from tests._reference.trellis_decode import codebook_tensor

    lut = codebook_tensor(payload.trellis.codebook, "cuda")
    args = [
        pointer(dtype, tensor)
        for dtype, tensor in (
            (c.Float16, source),
            (c.Uint32, payload.w13),
            (c.Uint8, lut),
            (c.Int64, ids),
            (c.Int64, payload.offsets),
            (c.Uint8, payload.rates),
            (c.Float16, output),
        )
    ]
    alternate = -source
    if dual_input:
        args[0] = (args[0], pointer(c.Float16, alternate), c.Int64(128))
    n, k = (
        (payload.intermediate_size, payload.hidden_size)
        if fc1
        else (payload.hidden_size, payload.intermediate_size)
    )
    kernel = InspectAtomOperands(
        RoutedAtomTrellisGemm(
            n,
            k,
            payload.num_experts,
            len(ids),
            group_size=payload.group_size,
            fc1=fc1,
            codebook=payload.trellis.codebook,
            dual_input=dual_input,
            paired_records=payload.paired_records,
        )
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    def scalars(projection, live, stage=0, n_base=0):
        return [
            c.Int32(projection),
            c.Int64(payload.row_stride_words),
            c.Int64(payload.w13.numel()),
            c.Int32(live),
            c.Int32(stage),
            c.Int64(n_base),
        ]

    fn = cute.compile(
        kernel,
        *args,
        *scalars(0 if fc1 else 2, 1),
        stream,
        options=f"--gpu-arch={architecture_for(torch.cuda.get_device_capability()).compilation_target}",
    )
    return fn, args, scalars, stream, (lut, alternate)


@pytest.mark.parametrize("codebook", ["mcg", "lut_e4m3", "lut_fp16"])
@pytest.mark.parametrize(
    "group_size,fc1,dual_input",
    [
        (32, True, False),
        (64, True, True),
        (256, False, False),
        (32, False, False),
        (None, True, False),
    ],
)
def test_grouped_atom_planes_boundaries_graph_and_mutation(
    codebook, group_size, fc1, dual_input
):
    _run_atom_staging(codebook, group_size, fc1, dual_input)


@pytest.mark.parametrize("codebook", ["mcg", "lut_e4m3"])
@pytest.mark.parametrize("fc1,dual_input", [(True, False), (True, True), (False, False)])
def test_exl3_pair_planes_boundaries_graph_and_mutation(tmp_path, codebook, fc1, dual_input):
    _run_atom_staging(codebook, 256, fc1, dual_input, exl3_path=tmp_path)


def _run_atom_staging(codebook, group_size, fc1, dual_input, *, exl3_path=None):
    _require_gpu()
    if exl3_path is None:
        public, bundle, logical, matrices = atom_fixture(
            codebook=codebook, group_size=group_size, device="cuda"
        )
        prepared = fused_moe.prepare_weights(plan=public, weights=bundle)
        payload = prepared._impl.representation_for("w4a16")
        assert payload.w13.data_ptr() == bundle.codes.data_ptr()
        torch.testing.assert_close(payload.rates.cpu(), logical, atol=0, rtol=0)
    else:
        public, layer, matrices = exl3_atom_fixture(exl3_path, codebook=codebook)
        # Only layout selection is counterfactual; the real GPU executes the
        # production operand staging below using its own compilation target.
        with patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)):
            prepared = fused_moe.prepare_weights(
                plan=public, exl3_layer=layer, exl3_device="cuda", params_dtype=torch.bfloat16,
            )
        payload = prepared.representation_for("w4a16")
        assert payload.paired_records
    n, k = (
        (payload.intermediate_size, payload.hidden_size)
        if fc1 else (payload.hidden_size, payload.intermediate_size)
    )
    source = torch.randn(8, k, device="cuda", dtype=torch.float16)
    ids = torch.tensor([0, 1, 2, 0, 1, -1, 3, 2**32 + 1], device="cuda")
    source[5:] = float("nan")
    output = torch.empty(8, 2, 128, 64, device="cuda", dtype=torch.float16)
    fn, args, scalars, stream, owners = compile_probe(
        payload, source, ids, output, fc1=fc1, dual_input=dual_input
    )

    def expected(projection, live, stage, n_base):
        result = torch.zeros(live, 2, 128, 64, dtype=torch.float16, device="cuda")
        for row, expert in enumerate(ids[:live].cpu().tolist()):
            if 0 <= expert < 3:
                nk, nn = min(64, k - stage * 64), min(128, n - n_base)
                result[row, 0, 0, :nk] = source[row, stage * 64 : stage * 64 + nk]
                if dual_input:
                    result[row, 0, 1, :nk] = owners[1][
                        row, stage * 64 : stage * 64 + nk
                    ]
                result[row, 1, :nn, :nk] = matrices[projection, expert][
                    n_base : n_base + nn, stage * 64 : stage * 64 + nk
                ].cuda()
        return result

    with patch.object(cute, "compile", side_effect=AssertionError("resolution frozen")):
        for projection in (0, 1) if fc1 else (2,):
            for live, stage, n_base in (
                (8, 0, 0),
                (1, 1, 128),
                (4, k // 64 - 1, n - 16),
                (3, 2, 0),
                (8, min(4, k // 64 - 1), min(256, n - 16)),
            ):
                output.fill_(float("nan"))
                fn(*args, *scalars(projection, live, stage, n_base), stream)
                torch.testing.assert_close(
                    output[:live],
                    expected(projection, live, stage, n_base),
                    atol=0,
                    rtol=0,
                )
                assert torch.isnan(output[live:]).all()
        phase = 0 if fc1 else 2
        fn(*args, *scalars(phase, 8), stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn(
                *args,
                *scalars(phase, 8),
                cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            )
        source[:5].mul_(0.5)
        owners[1][:5].neg_()
        ids[:5] = ids[:5].flip(0)
        # Move each expert's rate and offset together, preserving valid payloads.
        payload.rates.copy_(payload.rates.flip(1))
        payload.offsets.copy_(payload.offsets.flip(1))
        matrices = {(p, e): matrices[p, 2 - e] for p in range(3) for e in range(3)}
        reference = expected(phase, 8, 0, 0)
        tensors = (
            source,
            ids,
            output,
            payload.w13,
            payload.rates,
            payload.offsets,
            *owners,
        )
        addresses = tuple(t.data_ptr() for t in tensors)
        allocated = torch.cuda.memory_stats()["allocated_bytes.all.allocated"]
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocated_bytes.all.allocated"] == allocated
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
        assert tuple(t.data_ptr() for t in tensors) == addresses
        torch.testing.assert_close(output, reference, atol=0, rtol=0)

        saved_rates, saved_offsets = payload.rates.clone(), payload.offsets.clone()
        invalid_rates = (0x13, 0x37, 0xFF)
        if payload.paired_records:
            invalid_rates += (0x23, 0x32, 0x24, 0x43, 0x55, 0x66)
        if codebook == "lut_fp16":
            invalid_rates += (0x45, 0x54, 0x75, 0x57)
        for code in invalid_rates:
            payload.rates.fill_(code)
            fn(*args, *scalars(phase, 8), stream)
            assert torch.count_nonzero(output[:, 1]) == 0
        payload.rates.copy_(saved_rates)
        for offset in (-1, payload.row_stride_words, 2**40):
            payload.offsets.fill_(offset)
            fn(*args, *scalars(phase, 8), stream)
            assert torch.count_nonzero(output[:, 1]) == 0
        payload.offsets.copy_(saved_offsets)
        for field, value in ((0, -1), (0, 3), (1, -1), (1, 0), (2, 0)):
            invalid = scalars(phase, 8)
            invalid[field] = c.Int32(value) if field == 0 else c.Int64(value)
            fn(*args, *invalid, stream)
            assert torch.count_nonzero(output[:, 1]) == 0
            if field == 0:
                assert torch.count_nonzero(output) == 0


@pytest.mark.parametrize("exl3", [False, True])
def test_atom_row_offsets_cross_int32_word_boundary(tmp_path, exl3):
    _require_gpu()
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    stride = 2**29 + 4096
    total = 8 * stride
    if free < total * 4 + 2**30:
        pytest.skip("17 GiB of free GPU memory required for high atom row offsets")
    if exl3:
        public, layer, matrices = exl3_atom_fixture(tmp_path, experts=1, width=256)
        with patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)):
            prepared = fused_moe.prepare_weights(
                plan=public, exl3_layer=layer, exl3_device="cuda", params_dtype=torch.bfloat16,
            )
        payload = prepared.representation_for("w4a16")
    else:
        public, bundle, _, matrices = atom_fixture(experts=1, device="cuda")
        prepared = fused_moe.prepare_weights(plan=public, weights=bundle)
        payload = prepared._impl.representation_for("w4a16")
    pool = torch.empty(total, dtype=torch.int32, device="cuda")
    rows = payload.w13.view(8, -1)
    for row in range(0 if exl3 else 4, 8):
        pool[row * stride : row * stride + rows.shape[1]].copy_(rows[row])
    payload = replace(payload, w13=pool, row_stride_words=stride)
    assert 4 * stride > 2**31
    source = torch.randn(1, 512, device="cuda", dtype=torch.float16)
    ids = torch.zeros(1, dtype=torch.int64, device="cuda")
    output = torch.empty(1, 2, 128, 64, device="cuda", dtype=torch.float16)
    fn, args, scalars, stream, owners = compile_probe(
        payload, source, ids, output, fc1=True
    )
    fn(*args, *scalars(0, 1, 0, 128), stream)
    torch.testing.assert_close(
        output[0, 1], matrices[0, 0][128:256, :64].cuda(), atol=0, rtol=0
    )
