"""V4.1 native indexer oracles (no reference implementation in serving paths)."""

from __future__ import annotations

import pytest
import torch

from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x.attention import dsa_indexer as api

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for native MXFP4 indexer"
)


@pytest.fixture(autouse=True)
def _sm120():
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("native b12x indexer requires SM12x")


def _oracle_quant(x):
    # kernel.py:128-204: floor per-group amax, ceil UE8M0, E2M1 RN-even,
    # BF16 dequantization before the published einsum.
    groups = x.float().reshape(*x.shape[:-1], 4, 32)
    amax = groups.abs().amax(-1).clamp_min(6 * 2.0**-126)
    bits = (amax / 6).contiguous().view(torch.int32)
    sf = ((bits >> 23) + ((bits & 0x7FFFFF) != 0)).to(torch.uint8)
    scale = (sf.int() << 23).view(torch.float32)
    scaled = groups / scale[..., None]
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=x.device)
    order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=x.device)
    # Even code wins exact halfway cases, independently of LUT ordering.
    pick = (scaled.abs()[..., None] - lut[order]).abs().argmin(-1)
    code = order[pick].to(torch.uint8) | (torch.signbit(scaled).to(torch.uint8) << 3)
    dequant = (
        lut[(code & 7).long()] * torch.where((code & 8) != 0, -1, 1) * scale[..., None]
    ).to(torch.bfloat16)
    code = code.reshape(*x.shape[:-1], 128)
    return code[..., ::2] | (code[..., 1::2] << 4), sf, dequant.reshape_as(x)


def _oracle_scores(q, k, weights):
    _, _, q = _oracle_quant(q)
    _, _, k = _oracle_quant(k)
    dot = torch.einsum("rhd,kd->rhk", q, k)
    return (dot.relu() * weights[..., None]).sum(1)


def _allocate(
    q,
    keys,
    lengths,
    *,
    max_candidates=0,
    source=False,
    high_pages=False,
    max_rows=None,
    page_order=None,
    page_size=64,
    mode="decode",
):
    device = q.device
    rows, heads, _ = q.shape
    width = keys.shape[0]
    pages = (width + page_size - 1) // page_size
    if page_order is None:
        page_order = torch.randperm(pages, device=device).int()
    page_bytes = api.index_mxfp4_page_bytes(page_size)
    base = 2**31 // page_bytes + 17 if high_pages else 7
    physical = page_order + base
    pool = torch.empty((base + pages, page_bytes), device=device, dtype=torch.uint8)
    slots = (
        physical[torch.arange(width, device=device) // page_size].long() * page_size
        + torch.arange(width, device=device) % page_size
    )
    api.quantize_write_index_k_mxfp4(
        keys, index_k_cache=pool, slot_mapping=slots, page_size=page_size
    )
    packed = torch.empty((rows, heads, 64), device=device, dtype=torch.uint8)
    sf = torch.empty((rows, heads, 4), device=device, dtype=torch.uint8)
    api.quantize_q_mxfp4(q, q_mxfp4=packed, q_scales=sf)
    plan = api.plan(
        api.Caps(
            device=device,
            num_q_heads=heads,
            max_q_rows=max_rows or rows,
            max_page_table_width=pages,
            topk=512,
            cache_format="mxfp4",
            max_candidates=max_candidates,
            candidate_topk_blocks=2048 if source else 0,
            page_size=page_size,
            mode=mode,
        )
    )
    (spec,) = plan.scratch_specs()
    args = dict(
        scratch=torch.empty(spec.shape, device=device, dtype=spec.dtype),
        q_mxfp4=packed,
        q_scales=sf,
        index_k_cache=pool,
        page_table=physical[None],
        cache_lengths=lengths,
        active_width=torch.tensor([width], dtype=torch.int32, device=device),
        output_indices=torch.empty((rows, 512), dtype=torch.int32, device=device),
        output_scores=torch.empty((rows, 512), dtype=torch.float32, device=device),
    )
    if source:
        args.update(
            candidate_output=torch.empty(
                (rows, 16384), dtype=torch.int32, device=device
            ),
            candidate_output_lengths=torch.empty(
                (rows,), dtype=torch.int32, device=device
            ),
        )
    return plan, args


def _assert_topk(scores, output, output_scores, logical_positions=None):
    for row in range(scores.shape[0]):
        valid = torch.isfinite(scores[row])
        count = min(512, int(valid.sum()))
        selected = output[row, :count].long()
        assert bool((output[row, count:] == -1).all())
        assert bool((selected[1:] > selected[:-1]).all())
        assert bool(torch.isneginf(output_scores[row, count:]).all())
        expected = scores[row, valid].float().topk(count).values.sort().values
        actual = output_scores[row, :count].sort().values
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if logical_positions is None:
            torch.testing.assert_close(
                output_scores[row, :count],
                scores[row, selected].float(),
                rtol=0,
                atol=0,
            )
        else:
            assert bool(
                torch.isin(selected, logical_positions[row, valid].long()).all()
            )


@pytest.mark.parametrize("page_size", [64, 128, 256])
@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_per_group_quantization_and_permuted_high_page_writer(page_size, mode):
    torch.manual_seed(128)
    device = torch.device("cuda")
    q = torch.randn((2, 4, 128), device=device, dtype=torch.bfloat16)
    q = (
        (
            q.view(2, 4, 4, 32)
            * torch.tensor([2.0**-120, 0.03125, 8, 256], device=device)[
                None, None, :, None
            ]
        )
        .to(torch.bfloat16)
        .view_as(q)
    )
    q[:, 0, :32] = 0
    # Exact E2M1 half-way boundaries, including signs.
    q[0, 1, :8] = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6], device=device)
    width = 2 * page_size + 2
    keys = torch.randn((width, 128), device=device, dtype=torch.bfloat16)
    lengths = torch.tensor([0, width - 1], dtype=torch.int32, device=device)
    plan, args = _allocate(q, keys, lengths, high_pages=True, page_size=page_size, mode=mode)
    packed, scales, _ = _oracle_quant(q)
    torch.testing.assert_close(args["q_mxfp4"], packed)
    torch.testing.assert_close(args["q_scales"], scales)
    kp, ks, _ = _oracle_quant(keys)
    positions = torch.arange(width, device=device)
    pages = args["page_table"][0, positions // page_size].long()
    data_offsets = (positions % page_size)[:, None] * 64 + torch.arange(
        64, device=device
    )
    scale_offsets = (
        page_size * 64
        + (positions % page_size)[:, None] * 4
        + torch.arange(4, device=device)
    )
    pool = args["index_k_cache"]
    torch.testing.assert_close(pool[pages[:, None], data_offsets], kp)
    torch.testing.assert_close(pool[pages[:, None], scale_offsets], ks)
    weights = torch.randn((2, 4), dtype=torch.bfloat16, device=device) / 64
    binding = api.bind(plan, query_weights=weights, **args)
    scores = api.score(binding)
    expected = _oracle_scores(q, keys, weights)
    expected = expected.masked_fill(positions[None] >= lengths[:, None], -torch.inf)
    torch.testing.assert_close(scores[:, :width], expected, rtol=0, atol=0)
    assert bool(torch.isneginf(scores[:, width:]).all())
    api.select(binding)
    _assert_topk(expected, args["output_indices"], args["output_scores"])


@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_bf16_stages_and_tp_reduce_before_selection(mode):
    torch.manual_seed(555)
    device = torch.device("cuda")
    q = torch.randn((2, 4, 128), dtype=torch.bfloat16, device=device)
    keys = torch.randn((768, 128), dtype=torch.bfloat16, device=device)
    weights = torch.randn((2, 4), dtype=torch.bfloat16, device=device) / 64
    lengths = torch.tensor([768, 517], dtype=torch.int32, device=device)
    plan, args = _allocate(q, keys, lengths, mode=mode)
    binding = api.bind(plan, query_weights=weights, **args)
    scores = api.score(binding)
    expected = _oracle_scores(q, keys, weights)
    pos = torch.arange(768, device=device)
    expected.masked_fill_(pos[None] >= lengths[:, None], -torch.inf)
    torch.testing.assert_close(scores, expected, rtol=0, atol=0)
    # A second rank contributes a different BF16 head-sum; the staged contract
    # is observable because selection must consume these mutated scores.
    other = torch.randn_like(expected)
    scores.add_(other)
    expected.add_(other)
    api.select(binding)
    _assert_topk(expected, args["output_indices"], args["output_scores"])


@pytest.mark.parametrize("source", [False, True])
@pytest.mark.parametrize("high_pages", [False, True])
@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_compact_score_extent_preserves_selection_and_frozen_replay(source, high_pages, mode):
    """Invisible capacity columns must not be required by score or selection."""
    torch.manual_seed(4141)
    device = torch.device("cuda")
    q = torch.randn((3, 8, 128), dtype=torch.bfloat16, device=device)
    keys = torch.randn((2048, 128), dtype=torch.bfloat16, device=device)
    weights = torch.randn((3, 8), dtype=torch.bfloat16, device=device) / 64
    lengths = torch.tensor([641, 517, 12], dtype=torch.int32, device=device)
    plan, args = _allocate(q, keys, lengths, source=source, high_pages=high_pages, mode=mode)
    full = api.bind(plan, query_weights=weights, **args)
    expected_scores = api.score(full).clone()
    api.select(full)
    expected_indices = args["output_indices"].clone()
    expected_values = args["output_scores"].clone()
    expected_candidates = args["candidate_output"].clone() if source else None
    expected_lengths = args["candidate_output_lengths"].clone() if source else None
    freeze_kernel_resolution("MXFP4 compact score extent replay")
    try:
        for width in (641, 768, 1024):
            binding = api.bind(plan, query_weights=weights, score_width=width, **args)
            scores = api.score(binding)
            assert scores.shape == (3, width) and scores.is_contiguous()
            torch.testing.assert_close(scores, expected_scores[:, :width], rtol=0, atol=0)
            api.select(binding)
            torch.testing.assert_close(args["output_indices"], expected_indices, rtol=0, atol=0)
            torch.testing.assert_close(args["output_scores"], expected_values, rtol=0, atol=0)
            if source:
                torch.testing.assert_close(args["candidate_output"], expected_candidates, rtol=0, atol=0)
                torch.testing.assert_close(args["candidate_output_lengths"], expected_lengths, rtol=0, atol=0)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                api.score(binding)
                api.select(binding)
            torch.cuda.current_stream().wait_stream(stream)
            args["scratch"].fill_(0xA5)
            graph.replay()
            torch.testing.assert_close(args["output_indices"], expected_indices, rtol=0, atol=0)
    finally:
        unfreeze_kernel_resolution()


@pytest.mark.parametrize("width,exception", [(0, ValueError), (-1, ValueError),
                                           (1025, ValueError), (False, TypeError),
                                           (512.0, TypeError)])
def test_compact_score_extent_rejects_invalid_reservations(width, exception):
    q = torch.zeros((1, 8, 128), dtype=torch.bfloat16, device="cuda")
    keys = torch.zeros((1024, 128), dtype=torch.bfloat16, device="cuda")
    lengths = torch.tensor([1024], dtype=torch.int32, device="cuda")
    plan, args = _allocate(q, keys, lengths)
    weights = torch.ones((1, 8), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(exception, match="score_width"):
        api.bind(plan, query_weights=weights, score_width=width, **args)


@pytest.mark.parametrize("heads", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("high_pages", [False, True])
def test_tensorcore_prefill_preserves_bf16_scores_and_selection(heads, high_pages):
    torch.manual_seed(41016 + heads)
    q = torch.randn((3, heads, 128), device="cuda", dtype=torch.bfloat16)
    keys = torch.randn((768, 128), device="cuda", dtype=torch.bfloat16)
    weights = torch.randn((3, heads), device="cuda", dtype=torch.bfloat16) / 64
    lengths = torch.tensor([768, 531, 0], device="cuda", dtype=torch.int32)
    plan, args = _allocate(q, keys, lengths, high_pages=high_pages, max_rows=256)
    scalar = api.bind(plan, query_weights=weights, **args)
    expected = api.score(scalar).clone()
    tensorcore = api.plan(api.Caps(
        device=q.device, num_q_heads=heads, max_q_rows=256,
        max_page_table_width=12, topk=512, cache_format="mxfp4",
        mode="prefill",
    ))
    binding = api.bind(tensorcore, query_weights=weights, **args)
    actual = api.score(binding)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    api.select(binding)
    _assert_topk(expected, args["output_indices"], args["output_scores"])


@pytest.mark.parametrize("exponents", [(-40, -4, 6, 40), (-6, -2, 3, 7)])
@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_varied_scales_preserve_staged_rounding_against_fp64(exponents, mode):
    """Use FP64 dots so the oracle does not lose opposed exponent products.

    A BF16 einsum differs from the FP64 dot rounded to BF16 at 130 output
    positions in the opposed-exponent fixture. Both scoring kernels must
    preserve the explicit BF16 stage boundaries and ordered FP32 head sum.
    """
    torch.manual_seed(41081)
    scales = torch.exp2(torch.tensor(exponents, device="cuda", dtype=torch.float32))
    q = (torch.randn((9, 8, 4, 32), device="cuda") * scales[None, None, :, None]).bfloat16().view(9, 8, 128)
    keys = (torch.randn((641, 4, 32), device="cuda") / scales[None, :, None]).bfloat16().view(641, 128)
    weights = torch.randn((9, 8), device="cuda").bfloat16() / 64
    lengths = torch.tensor([0, 1, 63, 64, 65, 127, 512, 640, 641], device="cuda", dtype=torch.int32)
    plan, args = _allocate(q, keys, lengths, mode=mode)
    _, _, dq = _oracle_quant(q)
    _, _, dk = _oracle_quant(keys)
    dot = torch.einsum("rhd,kd->rhk", dq.double(), dk.double()).bfloat16()
    products = (dot.relu() * weights[:, :, None]).float()
    total = torch.zeros((9, 641), device="cuda")
    for head in range(8):
        total += products[:, head]
    expected = total.bfloat16()
    expected.masked_fill_(torch.arange(641, device="cuda")[None] >= lengths[:, None], -torch.inf)
    binding = api.bind(plan, query_weights=weights, **args)
    actual = api.score(binding)
    torch.testing.assert_close(actual[:, :641], expected, rtol=0, atol=0)
    assert bool(torch.isneginf(actual[:, 641:]).all())
    api.select(binding)
    _assert_topk(expected, args["output_indices"], args["output_scores"])


@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_source_blockmax_newest_and_bounded_candidate_reindex(mode):
    torch.manual_seed(2048)
    device = torch.device("cuda")
    width = 16448  # More than 2048 blocks; the newest visible block is partial.
    q = torch.ones((3, 4, 128), dtype=torch.bfloat16, device=device)
    keys = torch.ones((width, 128), dtype=torch.bfloat16, device=device)
    lengths = torch.tensor([0, 9, 16441], dtype=torch.int32, device=device)
    plan, args = _allocate(q, keys, lengths, source=True, mode=mode)
    binding = api.bind(
        plan,
        query_weights=torch.ones((3, 4), dtype=torch.bfloat16, device=device),
        **args,
    )
    scores = api.score(binding)
    # Directly supply the globally reduced published index_score: every block
    # has a distinct max, with the newest block deliberately worst-ranked.
    position = torch.arange(width, device=device)
    logits = (-(position // 8)).float().expand(3, -1).clone()
    # BF16 has too few distinct negative magnitudes at this width; assign a
    # representable score gap at the selected/unselected block boundary.
    logits[:, : 2047 * 8] = 4
    logits[:, 2047 * 8 :] = -4
    logits.masked_fill_(position[None] >= lengths[:, None], -torch.inf)
    scores.copy_(logits)
    api.select(binding)
    candidates = args["candidate_output"]
    candidate_lengths = args["candidate_output_lengths"]
    assert candidate_lengths.tolist() == [0, 9, 16377]
    assert bool((candidates[0] == -1).all())
    torch.testing.assert_close(
        candidates[1, :9], torch.arange(9, dtype=torch.int32, device=device)
    )
    torch.testing.assert_close(
        candidates[2, : 2047 * 8],
        torch.arange(2047 * 8, dtype=torch.int32, device=device),
    )
    assert candidates[2, 2047 * 8].item() == 16440
    assert bool((candidates[2, 2047 * 8 + 1 :] == -1).all())
    # Reindex uses its own Q/weights and cannot read the excluded positions.
    rq = torch.randn_like(q)
    weights = torch.randn((3, 4), dtype=torch.bfloat16, device=device) / 64
    rplan, rargs = _allocate(rq, keys, lengths, max_candidates=16384, mode=mode)
    rargs.update(candidate_indices=candidates, candidate_lengths=candidate_lengths)
    reindex = api.bind(rplan, query_weights=weights, **rargs)
    rescored = api.score(reindex)
    expected_full = _oracle_scores(rq, keys, weights)
    expected = expected_full.gather(1, candidates.clamp_min(0).long())
    expected.masked_fill_(candidates < 0, -torch.inf)
    torch.testing.assert_close(rescored, expected, rtol=0, atol=0)
    api.select(reindex)
    _assert_topk(expected, rargs["output_indices"], rargs["output_scores"], candidates)


@pytest.mark.parametrize("page_size", [64, 128, 256])
@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_fixed_graph_buffers_multiple_live_rows_and_visibility(page_size, mode):
    device = torch.device("cuda")
    q = torch.ones((4, 4, 128), dtype=torch.bfloat16, device=device)
    keys = torch.ones((256, 128), dtype=torch.bfloat16, device=device)
    lengths = torch.tensor([0, 1, 63, 256], dtype=torch.int32, device=device)
    plan, args = _allocate(q, keys, lengths, max_candidates=128, page_size=page_size, mode=mode)
    candidates = (
        torch.arange(128, device=device, dtype=torch.int32).expand(4, -1).contiguous()
    )
    candidate_lengths = torch.tensor([0, 1, 63, 128], device=device, dtype=torch.int32)
    weights = torch.ones((4, 4), device=device, dtype=torch.bfloat16)
    args.update(candidate_indices=candidates, candidate_lengths=candidate_lengths)
    binding = api.bind(plan, query_weights=weights, **args)
    api.run(binding)
    expected = args["output_indices"].clone()
    addresses = [view.data_ptr() for view in binding.runtime.scratch.values()]
    freeze_kernel_resolution("MXFP4 static-capacity replay")
    try:
        for rows in (1, 3, 4):
            api.quantize_q_mxfp4(
                q[:rows],
                q_mxfp4=args["q_mxfp4"][:rows],
                q_scales=args["q_scales"][:rows],
            )
            live = dict(args)
            for name in (
                "q_mxfp4",
                "q_scales",
                "cache_lengths",
                "output_indices",
                "output_scores",
                "candidate_indices",
                "candidate_lengths",
            ):
                live[name] = args[name][:rows]
            rebound = api.bind(plan, query_weights=weights[:rows], **live)
            api.run(rebound)
            torch.testing.assert_close(live["output_indices"], expected[:rows])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            api.run(binding)
        allocated = torch.cuda.memory_allocated()
        for visible in (0, 5, 101):
            lengths.fill_(visible)
            candidate_lengths.fill_(visible)
            args["output_indices"].fill_(-99)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            assert [
                view.data_ptr() for view in binding.runtime.scratch.values()
            ] == addresses
            expected.fill_(-1)
            expected[:, :visible] = torch.arange(
                visible, dtype=torch.int32, device=device
            )
            torch.testing.assert_close(args["output_indices"], expected)
    finally:
        unfreeze_kernel_resolution()


@pytest.mark.parametrize("source", [False, True])
def test_bounded_score_width_reuses_capacity_and_preserves_selection(source):
    """A live column bound must shrink real score work, not its allocation."""
    torch.manual_seed(416)
    device = torch.device("cuda")
    q = torch.randn((3, 4, 128), dtype=torch.bfloat16, device=device)
    keys = torch.randn((2304, 128), dtype=torch.bfloat16, device=device)
    lengths = torch.tensor([0, 17, 2304], dtype=torch.int32, device=device)
    plan, args = _allocate(
        q, keys, lengths, source=source, high_pages=True, page_size=256
    )
    weights = torch.ones((3, 4), dtype=torch.bfloat16, device=device) / 64
    api.run(api.bind(plan, query_weights=weights, **args))
    expected_scores = _oracle_scores(q, keys, weights)
    freeze_kernel_resolution("bounded MXFP4 score width")
    try:
        for width in (129, 513, 2051):
            lengths.copy_(
                torch.tensor([0, width // 2, width], dtype=torch.int32, device=device)
            )
            binding = api.bind(plan, query_weights=weights, score_width=width, **args)
            scores = api.score(binding)
            assert scores.shape == (3, width)
            assert scores.is_contiguous()
            positions = torch.arange(width, device=device)
            expected = expected_scores[:, :width].masked_fill(
                positions[None] >= lengths[:, None], -torch.inf
            )
            torch.testing.assert_close(scores, expected, atol=0, rtol=0)
            api.select(binding)
            _assert_topk(expected, args["output_indices"], args["output_scores"])
            if source:
                for row, count in enumerate((0, width // 2, width)):
                    assert args["candidate_output_lengths"][row].item() == count
                    torch.testing.assert_close(
                        args["candidate_output"][row, :count],
                        torch.arange(count, device=device, dtype=torch.int32),
                    )
                    assert args["candidate_output"][row, count:].eq(-1).all()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            api.run(binding)
        lengths.copy_(torch.tensor([0, 3, 1023], dtype=torch.int32, device=device))
        args["scratch"].fill_(0xA5)
        graph.replay()
        positions = torch.arange(width, device=device)
        expected = expected_scores[:, :width].masked_fill(
            positions[None] >= lengths[:, None], -torch.inf
        )
        _assert_topk(expected, args["output_indices"], args["output_scores"])
    finally:
        unfreeze_kernel_resolution()
