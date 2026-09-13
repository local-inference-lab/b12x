"""Portable qualification of planned DeepSeek compressed ordinary-MMA attention."""

import pytest
import torch

from b12x.attention import compressed_sparse_mla as mla
from b12x.attention.compressed_sparse_mla._policy import SparseMlaConfig
from b12x.attention._shared.mla.compressed_reference import (
    compressed_sparse_mla_reference,
    pack_compressed_sparse_mla_kv_cache_reference,
    pack_deepseek_v41_cache_reference,
)


def require_blackwell():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 3),
        (12, 0),
        (12, 1),
    ):
        pytest.skip("physical Blackwell GPU required")
    return torch.device("cuda", torch.cuda.current_device())


def pack(values, page_size, recipe, kind):
    if recipe == "deepseek_v41":
        return pack_deepseek_v41_cache_reference(
            values, page_size=page_size, cache_kind=kind
        )
    return pack_compressed_sparse_mla_kv_cache_reference(
        values[:, :448].contiguous(), values[:, 448:].contiguous(), page_size=page_size
    )


@pytest.mark.parametrize(
    "recipe,precision",
    [("deepseek_v4", "fp8"), ("deepseek_v41", "bf16"), ("deepseek_v41", "fp8")],
)
@pytest.mark.parametrize("mode", ["decode", "extend"])
@pytest.mark.parametrize("heads", [1, 8, 12, 16, 20, 32])
@pytest.mark.parametrize("with_sink", [False, True])
def test_compressed_warp_numerics(
    recipe, precision, mode, heads, with_sink, source="both"
):
    device = require_blackwell()
    torch.manual_seed(934)
    rows, width, page_size = 3, 128, 64
    scale = 512**-0.5
    q = torch.randn(rows, heads, 512, device=device, dtype=torch.bfloat16) * 0.2
    q[:, 0, :448] = 0
    q[:, 0, 448:] *= 8
    magnitudes = torch.linspace(0.05, 0.8, 32, device=device).repeat_interleave(16)
    swa = pack(
        (torch.randn(width, 512, device=device) * magnitudes).bfloat16(),
        page_size,
        recipe,
        "swa",
    )
    indexed = pack(
        (torch.randn(width, 512, device=device) * magnitudes.flip(0) * 2).bfloat16(),
        page_size,
        recipe,
        "indexed",
    )
    indices = torch.arange(width, device=device, dtype=torch.int32).repeat(rows, 1)
    indices[:, 90:] = -1
    lengths = torch.tensor([0, 65, width], device=device, dtype=torch.int32)
    swa_ids = indices if source != "indexed" else indices[:, :0]
    indexed_ids = indices if source != "swa" else None
    if source == "swa":
        indexed = None
    sink = torch.linspace(-1, 5, heads, device=device) if with_sink else None
    config = SparseMlaConfig(
        max_chunks_per_row=4, v41_compute_mode=precision, backend="warp"
    )
    plan = mla.plan(
        mla.Caps(
            device=device,
            num_q_heads=heads,
            max_q_rows=19,
            max_width=(2 if source == "both" else 1) * width,
            swa_width=width if source != "indexed" else 0,
            indexed_width=width if source != "swa" else 0,
            cache_format=recipe,
            mode=mode,
            swa_page_size=page_size,
            indexed_page_size=page_size,
        ),
        config=config,
    )
    (spec,) = plan.scratch_specs()
    storage = torch.full(spec.shape, 0x7F, device=device, dtype=spec.dtype)
    binding = mla.bind(
        plan,
        scratch=storage,
        q=q,
        swa_indices=swa_ids,
        swa_lengths=lengths,
        indexed_indices=indexed_ids,
        indexed_lengths=lengths if indexed is not None else None,
    )
    expected, expected_lse = compressed_sparse_mla_reference(
        q,
        swa,
        swa_ids,
        lengths,
        extra_k_cache=indexed,
        extra_indices=indexed_ids,
        extra_topk_lengths=lengths if indexed is not None else None,
        swa_page_size=page_size,
        extra_page_size=page_size,
        sm_scale=scale,
        return_lse=True,
        cache_format=recipe,
        attn_sink=sink,
    )
    if recipe == "deepseek_v41" and precision == "fp8":
        from tests._reference.v41_fp8 import canonical_fp8_rows, split64_fp8_attention

        swa_values, swa_scales = canonical_fp8_rows(swa, "swa")
        indexed_values, indexed_scales = (
            canonical_fp8_rows(indexed, "indexed")
            if indexed is not None
            else (swa_values, swa_scales)
        )
        gather = indices.long().clamp_min(0)
        valid = (indices >= 0) & (
            torch.arange(width, device=device)[None] < lengths[:, None]
        )
        expected, expected_lse = split64_fp8_attention(
            q,
            torch.cat((swa_values[gather], indexed_values[gather]), dim=1),
            torch.cat((swa_scales[gather], indexed_scales[gather]), dim=1),
            torch.cat(
                (valid & (source != "indexed"), valid & (source != "swa")), dim=1
            ),
            scale,
            attn_sink=sink,
            qk_fp8=mode == "decode",
            round_split_outputs=mode == "decode",
        )
    actual, lse = mla.run(
        binding=binding,
        swa_k_cache=swa[:0] if source == "indexed" else swa,
        indexed_k_cache=indexed,
        swa_page_size=page_size,
        indexed_page_size=page_size,
        sm_scale=scale,
        return_lse=True,
        lse_scale="natural",
        out=torch.empty_like(q),
        attn_sink=sink,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all() and torch.count_nonzero(actual[1:])
    torch.testing.assert_close(actual[0], torch.zeros_like(actual[0]), atol=0, rtol=0)
    if sink is None:
        assert torch.isneginf(lse[0]).all()
    else:
        torch.testing.assert_close(lse[0], sink)
    cosine = torch.nn.functional.cosine_similarity(
        actual[1:].float().flatten(), expected[1:].float().flatten(), dim=0
    )
    error = torch.linalg.vector_norm(
        actual[1:].float() - expected[1:].float()
    ) / torch.linalg.vector_norm(expected[1:].float())
    print(
        recipe,
        precision,
        mode,
        "cosine",
        float(cosine),
        "relative_l2",
        float(error),
        flush=True,
    )
    assert cosine > 0.999 and error < 0.045
    tolerance = 0.008 if recipe == "deepseek_v41" and precision == "fp8" else 0.035
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(lse, expected_lse, atol=0.005, rtol=0.005)


@pytest.mark.parametrize(
    "recipe,precision",
    [("deepseek_v4", "fp8"), ("deepseek_v41", "bf16"), ("deepseek_v41", "fp8")],
)
@pytest.mark.parametrize("mode", ["decode", "extend"])
def test_compressed_warp_serving(recipe, precision, mode, monkeypatch):
    from b12x._lib.runtime_control import (
        freeze_kernel_resolution,
        unfreeze_kernel_resolution,
    )
    from b12x.attention.compressed_sparse_mla import _warp
    from b12x.policy import PolicyContext

    device = require_blackwell()
    torch.manual_seed(719)
    rows, heads, capacity = 19, 12, 65
    swa_page, index_page = 64, 32
    q = torch.randn(rows, heads, 512, device=device, dtype=torch.bfloat16) * 0.2
    swa_values = torch.randn(128, 512, device=device).bfloat16()
    small_swa = pack(swa_values, swa_page, recipe, "swa")
    small_index = pack(
        torch.randn(128, 512, device=device).bfloat16(), index_page, recipe, "indexed"
    )

    def large_pool(packed):
        stride = packed.shape[1] + 256
        pid = (2**31 // stride) + 3
        backing = torch.empty(
            (pid + packed.shape[0], stride), device=device, dtype=torch.uint8
        )
        view = backing[:, : packed.shape[1]]
        view[0].fill_(0xFF)
        view[pid:].copy_(packed)
        assert pid * view.stride(0) > 2**31
        return view, pid

    swa, swa_pid = large_pool(small_swa)
    indexed, index_pid = large_pool(small_index)
    writer_slots = (
        torch.arange(128, device=device, dtype=torch.int64) + swa_pid * swa_page
    )

    def write_swa():
        if recipe == "deepseek_v41":
            mla.write_cache(
                swa_values, swa, writer_slots, page_size=swa_page, cache_kind="swa"
            )

    write_swa()
    torch.testing.assert_close(swa[swa_pid:], small_swa, atol=0, rtol=0)
    local_swa = torch.randint(
        0, 128, (rows, capacity), device=device, dtype=torch.int32
    )
    swa_indices = local_swa + swa_pid * swa_page
    logical = torch.randint(0, 128, (rows, capacity), device=device, dtype=torch.int32)
    # Recycled, missing, and out-of-range pages all occur in one selection.
    table = torch.tensor([3, -1, 0, 2], device=device, dtype=torch.int32).repeat(
        rows, 1
    )
    table = torch.where(table >= 0, table + index_pid, table)
    logical[:, -1] = 999
    swa_indices[:, -1] = 2147483647
    logical[4].fill_(-1)
    swa_indices[4].fill_(-1)
    lengths = torch.full((rows,), capacity, device=device, dtype=torch.int32)
    lengths[0], lengths[1], lengths[2], lengths[3] = -1, 1, 13, 999
    index_lengths = lengths.clone()
    sink = torch.linspace(-1, 4, heads, device=device)
    plan = mla.plan(
        mla.Caps(
            device=device,
            num_q_heads=heads,
            max_q_rows=rows,
            max_width=2 * capacity,
            swa_width=capacity,
            indexed_width=capacity,
            swa_page_size=swa_page,
            indexed_page_size=index_page,
            cache_format=recipe,
            mode=mode,
            use_cuda_graph=True,
        ),
        config=SparseMlaConfig(
            max_chunks_per_row=4, v41_compute_mode=precision, backend="warp"
        ),
    )
    (spec,) = plan.scratch_specs()
    storage = torch.full(spec.shape, 0x7F, device=device, dtype=spec.dtype)
    output = torch.empty_like(q)

    def bind(live, width, *, extra=True, mapping=table, start=0):
        active = slice(start, start + live)
        return mla.bind(
            plan,
            scratch=storage,
            q=q[active],
            swa_indices=swa_indices[active, :width].contiguous(),
            swa_lengths=lengths[active],
            indexed_indices=logical[active, :width].contiguous() if extra else None,
            indexed_lengths=index_lengths[active] if extra else None,
            indexed_page_table=mapping[active] if extra else None,
        )

    def run(binding, *, extra=True, with_sink=True, natural=True, default_output=False):
        return mla.run(
            binding=binding,
            swa_k_cache=swa,
            swa_page_size=swa_page,
            indexed_k_cache=indexed if extra else None,
            indexed_page_size=index_page if extra else None,
            attn_sink=sink if with_sink else None,
            sm_scale=512**-0.5,
            return_lse=True,
            lse_scale="natural" if natural else "base2",
            out=None if default_output else output[: binding.q.shape[0]],
        )

    def check(result, binding, *, extra=True, with_sink=True, natural=True):
        actual, lse = result
        live, width = binding.swa_indices.shape
        if live == 0:
            assert actual.shape == (0, heads, 512) and lse.shape == (0, heads)
            return
        local = binding.swa_indices.long() - swa_pid * swa_page
        local = torch.where((local >= 0) & (local < 128), local, -1).int()
        active_lengths = binding.swa_lengths.clamp(0, width)
        physical = None
        active_index_lengths = None
        if extra:
            logical_indices = binding.indexed_indices.long()
            pages = logical_indices // index_page
            mapped = binding.indexed_page_table.long().gather(1, pages.clamp(0, 3))
            physical = (mapped - index_pid) * index_page + logical_indices % index_page
            valid = (
                (logical_indices >= 0)
                & (pages < 4)
                & (physical >= 0)
                & (physical < 128)
            )
            physical = torch.where(valid, physical, -1).int()
            active_index_lengths = binding.indexed_lengths.clamp(0, width)
        expected, expected_lse = compressed_sparse_mla_reference(
            binding.q,
            small_swa,
            local,
            active_lengths,
            extra_k_cache=small_index if extra else None,
            extra_indices=physical,
            extra_topk_lengths=active_index_lengths,
            swa_page_size=swa_page,
            extra_page_size=index_page if extra else None,
            sm_scale=512**-0.5,
            return_lse=True,
            cache_format=recipe,
            attn_sink=sink if with_sink else None,
        )
        if recipe == "deepseek_v41" and precision == "fp8":
            from tests._reference.v41_fp8 import (
                canonical_fp8_rows,
                split64_fp8_attention,
            )

            data, scales, masks = [], [], []
            for packed, kind, ids, count in (
                (small_swa, "swa", local, active_lengths),
                (small_index, "indexed", physical, active_index_lengths),
            ):
                values, factors = canonical_fp8_rows(packed, kind)
                padded_ids = torch.full(
                    (live, 128), -1, dtype=torch.long, device=device
                )
                if ids is not None:
                    padded_ids[:, :width] = ids
                valid = padded_ids >= 0
                if count is not None:
                    valid &= torch.arange(128, device=device)[None] < count[:, None]
                data.append(values[padded_ids.clamp_min(0)])
                scales.append(factors[padded_ids.clamp_min(0)])
                masks.append(valid)
            expected, expected_lse = split64_fp8_attention(
                binding.q,
                torch.cat(data, dim=1),
                torch.cat(scales, dim=1),
                torch.cat(masks, dim=1),
                512**-0.5,
                sink if with_sink else None,
                qk_fp8=mode == "decode",
                round_split_outputs=mode == "decode",
            )
        if not natural:
            expected_lse = expected_lse * 1.4426950408889634
        torch.testing.assert_close(
            actual,
            expected,
            atol=0.008 if recipe == "deepseek_v41" and precision == "fp8" else 0.05,
            rtol=0.025,
        )
        torch.testing.assert_close(lse, expected_lse, atol=0.01, rtol=0.005)
        assert torch.isfinite(actual).all()
        if width and live > 1:
            assert torch.count_nonzero(actual[1:])

    # Warm without an indexed source or sink; both remain legal under frozen resolution.
    initial = bind(rows, capacity, extra=False)
    initial_result = run(initial, extra=False, with_sink=False)
    check(initial_result, initial, extra=False, with_sink=False)
    callables = dict(_warp._CACHE[plan.backend_key])

    def no_resolution(*args, **kwargs):
        raise AssertionError("serving must reuse the precompiled plan")

    monkeypatch.setattr(PolicyContext, "resolve", no_resolution)
    monkeypatch.setattr(_warp, "b12x_compile", no_resolution)
    freeze_kernel_resolution("compressed MLA runtime counts, sink, and page tables")
    try:
        for live, width in ((rows, capacity), (3, 13), (1, 1), (0, 0), (5, 0)):
            binding = bind(live, width)
            for natural in (False, True):
                check(run(binding, natural=natural), binding, natural=natural)
            check(run(binding, default_output=True), binding)
        binding = bind(3, capacity, start=1)
        assert binding.swa_lengths.data_ptr() % 16 == 4
        check(run(binding), binding)
        binding = bind(rows, capacity, mapping=table[:1].expand(rows, -1))
        check(run(binding), binding)
        compiled = torch.compile(lambda: run(binding), backend="eager", fullgraph=True)
        check(compiled(), binding)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            write_swa()
            result = run(binding)
        pointers = tuple(t.data_ptr() for t in (q, swa, indexed, storage, output))
        q.mul_(1.25)
        lengths[1:].fill_(3)
        index_lengths[1:].fill_(7)
        table[0].copy_(table[0].flip(0))
        swa_values.copy_(torch.randn_like(swa_values))
        small_swa.copy_(pack(swa_values, swa_page, recipe, "swa"))
        if recipe == "deepseek_v41":
            swa[swa_pid:].fill_(0xFF)
        else:
            swa[swa_pid:].copy_(small_swa)
        output.fill_(float("nan"))
        allocation = torch.cuda.memory_allocated(device)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated(device) == allocation
        check(result, binding)
        assert pointers == tuple(
            t.data_ptr() for t in (q, swa, indexed, storage, output)
        )
        assert callables == _warp._CACHE[plan.backend_key]
    finally:
        unfreeze_kernel_resolution()


@pytest.mark.parametrize(
    "recipe,precision",
    [("deepseek_v4", "fp8"), ("deepseek_v41", "bf16"), ("deepseek_v41", "fp8")],
)
@pytest.mark.parametrize("mode", ["decode", "extend"])
@pytest.mark.parametrize("source", ["swa", "indexed"])
def test_compressed_warp_single_source(recipe, precision, mode, source):
    test_compressed_warp_numerics(recipe, precision, mode, 12, True, source=source)
