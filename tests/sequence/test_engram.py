"""Engram contracts: DEAD, speculative chunks, row TP and 64-bit addressing."""

import pytest
import torch

from b12x.sequence import engram
from b12x.sequence.engram.reference import hash_reference
from b12x._lib.runtime_control import (
    freeze_kernel_resolution,
    kernel_resolution_frozen,
    unfreeze_kernel_resolution,
)
from ..conftest import require_b12x


def _plan(device, *, layer=1, tokens=7, rank=0, tp=2, base=101):
    geometry = engram.build_geometry(base_table_size=base, compressed_vocab_size=32)
    return engram.plan(
        engram.Caps(
            device=device,
            max_tokens=tokens,
            max_seqs=3,
            max_requests=4,
            vocab_size=32,
            layer_id=layer,
            tp_rank=rank,
            tp_size=tp,
        ),
        token_map=list(range(32)),
        geometry=geometry,
    )


def _binding(plan):
    c, spec = plan.caps, plan.scratch_specs()[0]

    def tensor(data, dtype):
        return torch.tensor(data, dtype=dtype, device=c.device)

    return engram.bind(
        plan,
        scratch=torch.empty(spec.shape, dtype=spec.dtype, device=c.device),
        token_ids=torch.zeros(c.max_tokens, dtype=torch.int64, device=c.device),
        token_mask=torch.ones(c.max_tokens, dtype=torch.bool, device=c.device),
        query_start_loc=tensor([0, 0, 0, 0], torch.int32),
        request_slots=tensor([2, 0, 3], torch.int32),
        committed_history=tensor(
            [[-1, -1, -1], [4, 5, 6], [7, -1, 9], [10, 11, 12]], torch.int64
        ),
        num_seqs=tensor([3], torch.int32),
        num_tokens=tensor([c.max_tokens], torch.int32),
        hash_ids=torch.empty((c.max_tokens, 24), dtype=torch.int64, device=c.device),
    )


def test_checkpoint_geometry_and_normalization():
    geometry = engram.build_geometry()
    assert geometry.num_embeddings == (384006168, 384016682)
    assert len(set(sum(geometry.primes, ()))) == 48
    # Independent explicit PCG64 equation catches dense-ordinal seeds.
    import numpy as np

    bound = (((1 << 63) - 1) // 99092) // 2
    for layer, values in zip((1, 14), geometry.multipliers, strict=True):
        expected = (
            np.random.Generator(np.random.PCG64(10007 * layer)).integers(
                0, bound, size=4, dtype=np.int64
            )
            * 2
            + 1
        )
        assert values == tuple(expected)

    class Backend:
        text = [
            " The",
            "THE",
            "thé",
            "ＴＨＥ",
            " ",
            "\t\n",
            "",
            "�",
            "�",
            "É",
            "e\u0301",
        ]

        def decode(self, ids, skip_special_tokens):
            return self.text[ids[0]]

        def id_to_token(self, token):
            return f"raw-byte-{token}"

    class Tokenizer:
        backend_tokenizer = Backend()

        def __len__(self):
            return len(Backend.text)

    compressed, count = engram.build_compressed_token_map(Tokenizer())
    assert compressed == [0, 0, 0, 0, 1, 1, 2, 3, 4, 5, 5]
    assert count == 6


@pytest.mark.parametrize("layer", [1, 14])
def test_packed_hash_dead_chunks_reorder_graph_and_no_commit(layer):
    device = require_b12x()
    p = _plan(device, layer=layer)
    b = _binding(p)
    history = b.committed_history.clone()
    cases = [
        (
            [1, 13, 14, 15, 16, 17, 18],
            [True, True, False, True, True, True, True],
            [0, 3, 3, 7],
            [2, 0, 3],
        ),
        ([19, 1, 20, 21, 22], [True] * 5, [0, 1, 4, 5], [3, 2, 0]),
        ([23, 24, 25], [True, False, True], [0, 1, 2, 3], [0, 3, 2]),
    ]

    def prepare(case):
        tokens, mask, starts, slots = case
        b.token_ids[: len(tokens)].copy_(torch.tensor(tokens, device=device))
        b.token_mask[: len(tokens)].copy_(torch.tensor(mask, device=device))
        b.query_start_loc.copy_(torch.tensor(starts, device=device))
        b.request_slots.copy_(torch.tensor(slots, device=device))
        b.num_tokens.fill_(len(tokens))

    prepare(cases[0])
    engram.run(b)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        engram.run(b)
    was_frozen = kernel_resolution_frozen()
    freeze_kernel_resolution("Engram capacity reuse")
    try:
        for case in cases:
            prepare(case)
            engram.run(b)
            graph.replay()
            tokens, mask, starts, slots = case
            expected = hash_reference(
                tokens,
                mask,
                starts,
                slots,
                history.cpu().tolist(),
                list(range(32)),
                p.geometry,
                layer,
            )
            torch.testing.assert_close(
                b.hash_ids[: len(tokens)].cpu(), expected, rtol=0, atol=0
            )
            assert torch.all(b.hash_ids[len(tokens) :] == -1)
            assert int(b.error_code) == 0
            torch.testing.assert_close(b.committed_history, history, rtol=0, atol=0)
    finally:
        if not was_frozen:
            unfreeze_kernel_resolution()
    # Accept only the first odd chunk, then decode reordered requests. No
    # implicit commit happened during either speculative call or graph replay.
    b.committed_history[2].copy_(torch.tensor([1, 13, -1], device=device))
    prepare(cases[2])
    engram.run(b)
    expected = hash_reference(
        *cases[2],
        b.committed_history.cpu().tolist(),
        list(range(32)),
        p.geometry,
        layer,
    )
    torch.testing.assert_close(b.hash_ids[:3].cpu(), expected, rtol=0, atol=0)


def test_fp8_e8m0_row_shards_sum_exactly_and_graph_missing_rows():
    device = require_b12x()
    partials, expected = [], None
    for rank in range(3):
        p = _plan(device, tokens=2, rank=rank, tp=3)
        ids = torch.tensor(
            [
                [0, p.shard_rows - 1, p.shard_rows, p.table_rows - 1, p.table_rows, -1]
                * 4
            ]
            * 2,
            dtype=torch.int64,
            device=device,
        )
        # Exact representable quantized payload; row-dependent values catch
        # incorrect global/per-head sharding and scale addressing.
        row = torch.arange(p.shard_rows, device=device) + p.shard_start
        w = (
            ((row[:, None] % 7 + 1).expand(-1, 256))
            .to(torch.float8_e4m3fn)
            .contiguous()
        )
        scales = (
            torch.arange(123, 131, dtype=torch.uint8, device=device)
            .expand(p.shard_rows, -1)
            .contiguous()
        )
        out = torch.empty((2, 6144), dtype=torch.bfloat16, device=device)
        count = torch.tensor([2], dtype=torch.int32, device=device)
        b = engram.bind_lookup(
            p, weight=w, scales=scales, hash_ids=ids, num_tokens=count, out=out
        )
        engram.run_lookup(b)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            engram.run_lookup(b)
        g.replay()
        value = (ids % 7 + 1).float()[..., None] * torch.pow(
            2.0, torch.arange(-4, 4, device=device).repeat_interleave(32)
        )[None, None, :]
        owned = (ids >= p.shard_start) & (ids < p.shard_end) & (ids < p.table_rows)
        local = (
            torch.where(owned[..., None], value, 0).to(torch.bfloat16).reshape(2, -1)
        )
        torch.testing.assert_close(out, local, rtol=0, atol=0)
        partials.append(out.clone())
        expected = (
            torch.where(((ids >= 0) & (ids < p.table_rows))[..., None], value, 0)
            .to(torch.bfloat16)
            .reshape(2, -1)
        )
        count.fill_(1)
        g.replay()
        assert torch.count_nonzero(out[1]) == 0
    torch.testing.assert_close(sum(partials), expected, rtol=0, atol=0)


def test_high_global_ids_and_local_stride_beyond_int32():
    device = require_b12x()
    # 2.4GB mostly untouched FP8 allocation, not a full BF16 materialization.
    p = _plan(device, tokens=1, rank=255, tp=256, base=100_000_000)
    local_row = 2**31 // 256 + 9
    assert local_row < p.shard_rows
    global_row = p.shard_start + local_row
    assert global_row > 2**31
    w = torch.empty(p.weight_shape, dtype=torch.float8_e4m3fn, device=device)
    s = torch.empty(p.scale_shape, dtype=torch.uint8, device=device)
    payload = (
        torch.arange(256, device=device).remainder(9).sub(4).to(torch.float8_e4m3fn)
    )
    w[local_row].copy_(payload)
    s[local_row].copy_(torch.arange(123, 131, device=device, dtype=torch.uint8))
    ids = torch.tensor(
        [[global_row, -1, p.shard_start - 1] * 8], dtype=torch.int64, device=device
    )
    out = torch.empty((1, 6144), dtype=torch.bfloat16, device=device)
    b = engram.bind_lookup(
        p,
        weight=w,
        scales=s,
        hash_ids=ids,
        num_tokens=torch.tensor([1], dtype=torch.int32, device=device),
        out=out,
    )
    engram.run_lookup(b)
    expected = (
        payload.float()
        * torch.pow(2.0, torch.arange(-4, 4, device=device).repeat_interleave(32))
    ).to(torch.bfloat16)
    torch.testing.assert_close(
        out.view(24, 256)[::3], expected.expand(8, -1), rtol=0, atol=0
    )
    assert torch.count_nonzero(out.view(24, 256)[1::3]) == 0
    assert torch.count_nonzero(out.view(24, 256)[2::3]) == 0


def test_e8m0_extreme_bytes_and_missing_nan_row():
    device = require_b12x()
    p = _plan(device, tokens=1, rank=1, tp=2)
    w = torch.empty(p.weight_shape, dtype=torch.float8_e4m3fn, device=device)
    scales = torch.empty(p.scale_shape, dtype=torch.uint8, device=device)
    w[0].fill_(float("nan"))
    scales[0].fill_(255)
    w[1].fill_(2)
    scale_row = torch.tensor(
        [0, 1, 126, 127, 128, 253, 255, 127], dtype=torch.uint8, device=device
    )
    scales[1].copy_(scale_row)
    ids = torch.tensor(
        [[p.shard_start + 1, -1, p.shard_start - 1] * 8],
        dtype=torch.int64,
        device=device,
    )
    out = torch.empty((1, 6144), dtype=torch.bfloat16, device=device)
    b = engram.bind_lookup(
        p,
        weight=w,
        scales=scales.view(torch.float8_e8m0fnu),
        hash_ids=ids,
        num_tokens=torch.tensor([1], dtype=torch.int32, device=device),
        out=out,
    )
    engram.run_lookup(b)
    expected = (
        2 * scale_row.view(torch.float8_e8m0fnu).float().repeat_interleave(32)
    ).to(torch.bfloat16)
    torch.testing.assert_close(
        out.view(24, 256)[::3], expected.expand(8, -1), rtol=0, atol=0, equal_nan=True
    )
    assert torch.count_nonzero(out.view(24, 256)[1::3]) == 0
    assert torch.count_nonzero(out.view(24, 256)[2::3]) == 0


def _disk_lookup_pair(plan, tmp_path, *, shard_rows=None, extreme_scales=False):
    rows = torch.arange(plan.table_rows)
    weight = ((rows[:, None] % 7 + 1) * torch.tensor([1, -1]).repeat(128)[None, :]).to(
        torch.float8_e4m3fn
    )
    scale_row = (
        [0, 1, 126, 127, 128, 253, 254, 255]
        if extreme_scales
        else list(range(123, 131))
    )
    scales = (
        torch.tensor(scale_row, dtype=torch.uint8)
        .expand(plan.table_rows, -1)
        .contiguous()
    )
    table = engram.DiskTable(plan, shard_rows=shard_rows, queue_depth=4)
    source_rows = plan.table_rows if shard_rows is None else shard_rows
    for scale, payload, offset in ((False, weight, 4093), (True, scales, 19)):
        for index, start in enumerate(range(0, plan.table_rows, source_rows)):
            path = tmp_path / f"{scale}-{index}.bin"
            path.write_bytes(
                bytes(offset)
                + payload[start : start + source_rows]
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
            table.add_shard(index, str(path), offset, scale=scale)
    device = plan.caps.device
    ids = torch.full((plan.caps.max_tokens, 24), -1, dtype=torch.int64, device=device)
    count = torch.tensor([plan.caps.max_tokens], dtype=torch.int32, device=device)
    disk = engram.bind_lookup(
        plan,
        disk_table=table,
        hash_ids=ids,
        num_tokens=count,
        out=torch.empty(
            (plan.caps.max_tokens, 6144), dtype=torch.bfloat16, device=device
        ),
    )
    local_weight = torch.zeros(plan.weight_shape, dtype=torch.float8_e4m3fn)
    local_scales = torch.zeros(plan.scale_shape, dtype=torch.uint8)
    end = min(plan.shard_end, plan.table_rows)
    length = end - plan.shard_start
    local_weight[:length].copy_(weight[plan.shard_start : end])
    local_scales[:length].copy_(scales[plan.shard_start : end])
    resident = engram.bind_lookup(
        plan,
        weight=local_weight.to(device),
        scales=local_scales.to(device),
        hash_ids=ids,
        num_tokens=count,
        out=torch.empty_like(disk.out),
    )
    return resident, disk


@torch.inference_mode()
@pytest.mark.parametrize("rank", [0, 3])
@pytest.mark.parametrize("shard_rows", [None, 31])
def test_disk_separate_planes_duplicates_tp_edges_and_raw_e8m0(
    rank, shard_rows, tmp_path
):
    device = require_b12x()
    p = _plan(device, tokens=3, rank=rank, tp=4, base=2)
    resident, disk = _disk_lookup_pair(
        p, tmp_path, shard_rows=shard_rows, extreme_scales=True
    )
    edge = (p.shard_start // 31 + 1) * 31
    row_ids = [
        p.shard_start,
        p.shard_start,
        edge - 1,
        edge,
        p.shard_start - 1,
        p.shard_end - 1,
        p.shard_end,
        -1,
        p.table_rows - 1,
        p.table_rows,
        -2,
        edge + 1,
    ] * 2
    disk.hash_ids.copy_(torch.tensor([row_ids] * 3, dtype=torch.int64, device=device))
    disk.num_tokens.fill_(2)
    expected = engram.run_lookup(resident).clone()
    torch.testing.assert_close(
        engram.run_lookup(disk), expected, rtol=0, atol=0, equal_nan=True
    )
    assert torch.count_nonzero(disk.out[2]) == 0
    missing = (
        (disk.hash_ids < p.shard_start)
        | (disk.hash_ids >= p.shard_end)
        | (disk.hash_ids >= p.table_rows)
    )
    assert torch.count_nonzero(disk.out.view(3, 24, 256)[missing]) == 0
    for count, prepared in ((3, 1), (0, 3), (-1, 3), (4, 3), (3, 0)):
        disk.num_tokens.fill_(count)
        expected = engram.run_lookup(resident, token_count=prepared).clone()
        torch.testing.assert_close(
            engram.run_lookup(disk, token_count=prepared),
            expected,
            rtol=0,
            atol=0,
            equal_nan=True,
        )
        if prepared < 3:
            assert torch.count_nonzero(disk.out[prepared:]) == 0
    with pytest.raises(ValueError):
        engram.run_lookup(disk, token_count=4)


@torch.inference_mode()
def test_disk_fresh_graph_consumer_stream_reuse_and_eager_boundary(
    tmp_path, monkeypatch
):
    device = require_b12x()
    p = _plan(device, tokens=3, tp=4, base=2)
    resident, disk = _disk_lookup_pair(p, tmp_path)
    disk.hash_ids.fill_(p.shard_start)
    engram.run_lookup(disk)
    consumed = torch.empty_like(disk.out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.mul(disk.out, 2, out=consumed)
    for row, live in ((p.shard_start + 1, 2), (p.shard_start + 5, 1)):
        disk.hash_ids.fill_(row)
        disk.num_tokens.fill_(live)
        expected = engram.run_lookup(resident).clone()
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            engram.run_lookup(disk, token_count=2)
        torch.cuda.current_stream(device).wait_stream(stream)
        graph.replay()
        torch.testing.assert_close(consumed, expected * 2, rtol=0, atol=0)

    def forbidden_read(*args):
        raise AssertionError("consumer graph must not perform disk preparation")

    with monkeypatch.context() as patch:
        patch.setattr(disk.disk_table._cache, "read_rows", forbidden_read)
        graph.replay()
        torch.testing.assert_close(consumed, expected * 2, rtol=0, atol=0)
        patch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        with pytest.raises(RuntimeError):
            engram.run_lookup(disk)
    with monkeypatch.context() as patch:
        patch.setattr(torch.compiler, "is_compiling", lambda: True)
        with pytest.raises(RuntimeError):
            engram.run_lookup(disk)
    with pytest.raises(RuntimeError):
        disk.disk_table.add_shard(0, str(tmp_path / "False-0.bin"), 4093)


@torch.inference_mode()
def test_disk_registration_requires_both_planes_and_exact_owner(tmp_path):
    device = require_b12x()
    p = _plan(device, tokens=1, tp=4, base=2)
    table = engram.DiskTable(p)
    weight_path, scale_path = tmp_path / "weight.bin", tmp_path / "scale.bin"
    weight_path.write_bytes(bytes(p.table_rows * 256))
    scale_path.write_bytes(bytes([127]) * (p.table_rows * 8))
    table.add_shard(0, str(weight_path), 0)
    args = dict(
        hash_ids=torch.zeros((1, 24), dtype=torch.int64, device=device),
        num_tokens=torch.ones((1,), dtype=torch.int32, device=device),
        out=torch.empty((1, 6144), dtype=torch.bfloat16, device=device),
    )
    with pytest.raises(ValueError):
        engram.bind_lookup(p, disk_table=table, **args)
    table.add_shard(0, str(scale_path), 0, scale=True)
    other = _plan(device, tokens=1, tp=4, base=2)
    with pytest.raises(ValueError):
        engram.bind_lookup(other, disk_table=table, **args)
    with pytest.raises(ValueError):
        engram.bind_lookup(
            p,
            disk_table=table,
            weight=torch.empty(
                p.weight_shape, dtype=torch.float8_e4m3fn, device=device
            ),
            **args,
        )
    with pytest.raises(ValueError):
        engram.bind_lookup(p, **args)
    binding = engram.bind_lookup(p, disk_table=table, **args)
    del table
    assert torch.count_nonzero(engram.run_lookup(binding)) == 0


@torch.inference_mode()
def test_disk_sparse_high_global_rows_without_whole_table_staging(tmp_path):
    device = require_b12x()
    p = _plan(device, tokens=1, rank=3, tp=4, base=100_000_000)
    selected = [p.shard_start, p.table_rows - 1]
    assert selected[-1] > 2**31
    table = engram.DiskTable(p, queue_depth=4)
    weight = torch.arange(256).remainder(9).sub(4).to(torch.float8_e4m3fn)
    scale = torch.arange(123, 131, dtype=torch.uint8)
    for is_scale, payload, width, offset in (
        (False, weight, 256, 4093),
        (True, scale, 8, 19),
    ):
        path = tmp_path / f"{is_scale}.bin"
        # Sparse sources have global checkpoint extents but only two real rows.
        # A table-sized allocation or host staging is not needed by this test.
        with path.open("wb") as stream:
            stream.truncate(offset + p.table_rows * width)
            for row in selected:
                stream.seek(offset + row * width)
                stream.write(payload.view(torch.uint8).numpy().tobytes())
        table.add_shard(0, str(path), offset, scale=is_scale)
    ids = torch.tensor(
        [[selected[1], selected[0], -1] * 8], dtype=torch.int64, device=device
    )
    binding = engram.bind_lookup(
        p,
        disk_table=table,
        hash_ids=ids,
        num_tokens=torch.ones((1,), dtype=torch.int32, device=device),
        out=torch.empty((1, 6144), dtype=torch.bfloat16, device=device),
    )
    engram.run_lookup(binding)
    expected = (
        weight.float() * scale.view(torch.float8_e8m0fnu).float().repeat_interleave(32)
    ).to(torch.bfloat16)
    result = binding.out.view(24, 256).cpu()
    torch.testing.assert_close(result[::3], expected.expand(8, -1), rtol=0, atol=0)
    torch.testing.assert_close(result[1::3], expected.expand(8, -1), rtol=0, atol=0)
    assert torch.count_nonzero(result[2::3]) == 0
    stats = table.stats()
    assert stats["cache_bytes"] == 24 * (256 + 8)
    assert stats["ids_host_bytes"] == 24 * 8
    assert stats["owned_staging_bytes"] < 1 << 20
