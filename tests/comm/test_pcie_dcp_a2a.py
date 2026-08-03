from __future__ import annotations

import pytest
import torch

from sparkinfer.comm.pcie.pcie_dcp_a2a import (
    PCIeDCPA2A,
    PCIeDCPA2APool,
    _SINGLE_CHANNEL_ID,
    _staging_layout,
    lse_reduce_scatter_reference,
)


class _FakeExt:
    def __init__(self) -> None:
        self.init_calls = []
        self.run_calls = []
        self.dispose_calls = []

    def init_dcp_a2a(
        self,
        signal_ptrs,
        staging0_ptrs,
        staging1_ptrs,
        output_capacity_elems,
        lse_offset,
        lse_capacity,
        rank,
    ):
        self.init_calls.append(
            (
                tuple(signal_ptrs),
                tuple(staging0_ptrs),
                tuple(staging1_ptrs),
                output_capacity_elems,
                lse_offset,
                lse_capacity,
                rank,
            )
        )
        return 1234

    def lse_reduce_scatter(
        self,
        pointer,
        partial_output,
        partial_lse,
        out,
        natural_log,
        threads,
        block_limit,
    ):
        self.run_calls.append(
            (pointer, natural_log, threads, block_limit, tuple(partial_output.shape))
        )
        heads_per_rank = partial_output.shape[1] // 2
        out.copy_(partial_output[:, :heads_per_rank])

    def all_gather_heads(
        self,
        pointer,
        local_input,
        out,
        threads,
        block_limit,
    ):
        self.run_calls.append(
            (
                pointer,
                "all_gather_heads",
                threads,
                block_limit,
                tuple(local_input.shape),
            )
        )
        out.copy_(torch.cat((local_input, local_input), dim=1))

    def dispose(self, pointer):
        self.dispose_calls.append(pointer)


def _make_runtime(ext: _FakeExt | None = None) -> PCIeDCPA2A:
    return PCIeDCPA2A(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        signal_ptrs=(100, 200),
        staging0_ptrs=(300, 400),
        staging1_ptrs=(500, 600),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        output_capacity_elems=4 * 32 * 64,
        lse_offset=4 * 32 * 64 * 2,
        lse_capacity=4 * 32,
        ext_module=ext or _FakeExt(),
    )


def test_staging_layout_has_aligned_disjoint_slots():
    layout = _staging_layout(
        signal_bytes=12345,
        world_size=2,
        max_batch_size=4,
        total_heads=32,
        head_dim=512,
    )

    assert layout.staging0_offset % 256 == 0
    assert layout.staging1_offset == layout.staging0_offset + layout.slot_bytes
    assert layout.lse_offset % 256 == 0
    assert layout.slab_bytes == layout.staging1_offset + layout.slot_bytes
    assert layout.output_capacity_elems >= 4 * 32 * 512
    assert layout.lse_capacity >= 4 * 32

    wider_query_layout = _staging_layout(
        signal_bytes=12345,
        world_size=2,
        max_batch_size=4,
        total_heads=32,
        head_dim=512,
        query_head_dim=576,
    )
    assert wider_query_layout.output_capacity_elems >= 4 * 32 * 576
    assert wider_query_layout.lse_offset > layout.lse_offset


def test_reference_selects_destination_heads_and_combines_lse_weights():
    outputs = torch.tensor(
        [
            [[[[1.0], [2.0], [10.0], [20.0]]]],
            [[[[3.0], [4.0], [30.0], [40.0]]]],
        ],
        dtype=torch.float32,
    ).reshape(2, 1, 4, 1)
    lses = torch.tensor(
        [
            [[0.0, 0.0, 0.0, -torch.inf]],
            [[0.0, -torch.inf, 0.0, -torch.inf]],
        ]
    )

    rank0 = lse_reduce_scatter_reference(outputs, lses, 0)
    rank1 = lse_reduce_scatter_reference(outputs, lses, 1)

    torch.testing.assert_close(rank0, torch.tensor([[[2.0], [2.0]]]))
    torch.testing.assert_close(rank1, torch.tensor([[[20.0], [0.0]]]))


def test_reference_ignores_nan_output_from_empty_shard():
    outputs = torch.tensor(
        [
            [[[[torch.nan], [2.0]]]],
            [[[[4.0], [6.0]]]],
        ],
        dtype=torch.float32,
    ).reshape(2, 1, 2, 1)
    lses = torch.tensor([[[-torch.inf, 0.0]], [[0.0, 0.0]]])

    actual = lse_reduce_scatter_reference(outputs, lses, 0)

    torch.testing.assert_close(actual, torch.tensor([[[4.0]]]))


def test_runtime_validates_and_dispatches_to_extension():
    ext = _FakeExt()
    runtime = _make_runtime(ext)
    partial_output = torch.arange(2 * 32 * 64, dtype=torch.bfloat16).reshape(2, 32, 64)
    partial_lse = torch.zeros(2, 32, dtype=torch.float32)

    out = runtime.lse_reduce_scatter(
        partial_output,
        partial_lse,
        is_lse_base_on_e=False,
        threads=256,
        block_limit=32,
    )

    assert out.shape == (2, 16, 64)
    assert torch.equal(out, partial_output[:, :16])
    assert ext.run_calls == [(1234, False, 256, 32, (2, 32, 64))]

    local_input = partial_output[:, :16].contiguous()
    gathered = runtime.all_gather_heads(
        local_input,
        threads=64,
        block_limit=16,
    )
    assert gathered.shape == partial_output.shape
    assert torch.equal(gathered, torch.cat((local_input, local_input), dim=1))
    assert ext.run_calls[-1] == (
        1234,
        "all_gather_heads",
        64,
        16,
        (2, 16, 64),
    )

    fp8_input = torch.arange(16 * 64, dtype=torch.float32).reshape(1, 16, 64)
    fp8_input = fp8_input.to(torch.float8_e4m3fn)
    fp8_gathered = runtime.all_gather_heads(fp8_input)
    assert fp8_gathered.dtype == torch.float8_e4m3fn
    expected_fp8 = torch.cat((fp8_input, fp8_input), dim=1)
    assert torch.equal(fp8_gathered.view(torch.uint8), expected_fp8.view(torch.uint8))
    runtime.close()
    assert ext.dispose_calls == [1234]


def test_runtime_accepts_head_major_input_and_output():
    ext = _FakeExt()
    runtime = _make_runtime(ext)
    input_storage = torch.arange(32 * 4 * 64, dtype=torch.bfloat16).reshape(32, 4, 64)
    partial_output = input_storage.transpose(0, 1)[:2]
    partial_lse = torch.zeros(2, 32, dtype=torch.float32)
    output_storage = torch.empty(16, 2, 64, dtype=torch.bfloat16)
    out = output_storage.transpose(0, 1)

    actual = runtime.lse_reduce_scatter(partial_output, partial_lse, out=out)

    assert actual is out
    assert actual.stride() == (64, 2 * 64, 1)
    torch.testing.assert_close(actual, partial_output[:, :16])


def test_runtime_rejects_shape_dtype_and_capacity_mismatches():
    runtime = _make_runtime()
    good_output = torch.zeros(1, 32, 64, dtype=torch.bfloat16)
    good_lse = torch.zeros(1, 32, dtype=torch.float32)

    with pytest.raises(ValueError, match="float32"):
        runtime.lse_reduce_scatter(good_output, good_lse.bfloat16())
    with pytest.raises(ValueError, match="configured heads/head_dim"):
        runtime.lse_reduce_scatter(good_output[:, :, :32], good_lse)
    with pytest.raises(ValueError, match="exceeds configured capacity"):
        runtime.lse_reduce_scatter(
            torch.zeros(5, 32, 64, dtype=torch.bfloat16),
            torch.zeros(5, 32, dtype=torch.float32),
        )
    unsupported_view = torch.zeros(1, 33, 64, dtype=torch.bfloat16)[:, :32]
    with pytest.raises(ValueError, match="packed token-major or head-major"):
        runtime.lse_reduce_scatter(unsupported_view, good_lse)

    with pytest.raises(ValueError, match="configured local heads/head_dim"):
        runtime.all_gather_heads(good_output[:, :8])
    with pytest.raises(ValueError, match="exceeds configured capacity"):
        runtime.all_gather_heads(torch.zeros(5, 16, 64, dtype=torch.bfloat16))


def test_constructor_rejects_invalid_query_and_staging_capacity():
    common = dict(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        signal_ptrs=(100, 200),
        staging0_ptrs=(300, 400),
        staging1_ptrs=(500, 600),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        lse_offset=4 * 32 * 64 * 2,
        lse_capacity=4 * 32,
        ext_module=_FakeExt(),
    )
    with pytest.raises(ValueError, match="query_head_dim"):
        PCIeDCPA2A(
            **common,
            query_head_dim=0,
            output_capacity_elems=4 * 32 * 64,
        )
    with pytest.raises(ValueError, match="output capacity"):
        PCIeDCPA2A(
            **common,
            query_head_dim=32,
            output_capacity_elems=4 * 32 * 32,
        )


def test_cuda_direct_runtime_requires_collective_factory():
    with pytest.raises(ValueError, match="exchange_group is required"):
        PCIeDCPA2A(
            rank=0,
            world_size=2,
            device=torch.device("cuda:0"),
            signal_ptrs=(100, 200),
            staging0_ptrs=(300, 400),
            staging1_ptrs=(500, 600),
            max_batch_size=4,
            total_heads=32,
            head_dim=64,
            output_capacity_elems=4 * 32 * 64,
            lse_offset=4 * 32 * 64 * 2,
            lse_capacity=4 * 32,
            ext_module=_FakeExt(),
        )


def test_pool_uses_distinct_channels_for_target_and_draft_captures(monkeypatch):
    created = []
    current_stream = [7]
    capturing = [False]

    def make_channel(stream_key):
        runtime = _make_runtime()
        created.append((stream_key, runtime))
        return runtime

    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=make_channel,
    )
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_dcp_a2a._current_stream_key",
        lambda device, stream=None: (
            current_stream[0] if stream is None else int(stream)
        ),
    )
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_dcp_a2a._is_current_stream_capturing",
        lambda device: capturing[0],
    )

    with pool.capture(7) as target_channel:
        capturing[0] = True
        current_stream[0] = 70
        assert pool.for_stream() is target_channel
        capturing[0] = False

    with pool.capture(8) as draft_channel:
        capturing[0] = True
        current_stream[0] = 80
        assert pool.for_stream() is draft_channel
        capturing[0] = False

    assert target_channel is not draft_channel
    assert pool._channels == {}
    assert target_channel in pool._all_channels
    assert draft_channel in pool._all_channels
    assert [entry[0] for entry in created] == [7, 8]

    capturing[0] = True
    current_stream[0] = 70
    with pytest.raises(RuntimeError, match="no channel during CUDA graph capture"):
        pool.for_stream()


def test_pool_collectively_prepares_logical_channels_in_canonical_order(
    monkeypatch,
):
    created = []
    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=lambda stream_key: _make_runtime(),
    )
    pool._channel_factory = None
    pool.exchange_group = object()
    monkeypatch.setattr(
        pool,
        "_new_channel",
        lambda stream_key: created.append(stream_key) or object(),
    )
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_dcp_a2a._broadcast_gather_object",
        lambda local_state, group: [local_state, local_state],
    )
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_oneshot._broadcast_gather_object",
        lambda local_status, group: [local_status, local_status],
    )

    pool.prepare_channels(("target", "draft"))
    pool.prepare_channels(("draft", "target"))

    assert list(pool._logical_channels) == ["draft", "target"]
    assert created == [None, None]


def test_pool_rejects_logical_channel_set_mismatch_before_allocation(monkeypatch):
    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=lambda stream_key: _make_runtime(),
    )
    pool._channel_factory = None
    pool.exchange_group = object()
    monkeypatch.setattr(
        pool,
        "_new_channel",
        lambda stream_key: pytest.fail("channel allocation must not start"),
    )

    def gather(local_state, group):
        if not local_state or isinstance(local_state[0], str):
            return [local_state, ()]
        _requested, existing = local_state
        return [(("other",), existing), local_state]

    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_dcp_a2a._broadcast_gather_object", gather
    )
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_oneshot._broadcast_gather_object", gather
    )

    with pytest.raises(RuntimeError, match="differs across ranks"):
        pool.prepare_channels(("target",))


def test_pool_invalid_logical_id_is_rejected_collectively(monkeypatch):
    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=lambda stream_key: _make_runtime(),
    )
    pool._channel_factory = None
    pool.exchange_group = object()
    monkeypatch.setattr(
        pool,
        "_new_channel",
        lambda stream_key: pytest.fail("channel allocation must not start"),
    )
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_oneshot._broadcast_gather_object",
        lambda local_status, group: [local_status, ()],
    )

    with pytest.raises(RuntimeError, match="must not be empty"):
        pool.prepare_channels(("",))


def test_pool_eager_channel_requires_id_and_rejects_duplicate_stream_owner(
    monkeypatch,
):
    eager = _make_runtime()
    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=lambda stream_key: eager,
    )
    pool._channel_factory = None
    pool.exchange_group = object()
    pool._logical_channels["eager:dcp"] = eager
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_dcp_a2a._current_stream_key",
        lambda device, stream=None: int(stream),
    )

    with pytest.raises(RuntimeError, match="explicit semantic channel_id"):
        pool.for_stream(7)
    assert pool.for_stream(7, channel_id="eager:dcp") is eager
    with pytest.raises(RuntimeError, match="stream-affine"):
        pool.for_stream(8, channel_id="eager:dcp")


def test_distributed_single_channel_pool_uses_prepared_default() -> None:
    eager = _make_runtime()
    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        single_channel=True,
        channel_factory=lambda stream_key: eager,
    )
    pool._channel_factory = None
    pool.exchange_group = object()
    pool._logical_channels[_SINGLE_CHANNEL_ID] = eager

    assert pool.for_stream() is eager
    assert pool._channels == {0: eager}


def test_pool_capture_requires_stable_semantic_id(monkeypatch):
    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=lambda stream_key: _make_runtime(),
    )
    pool._channel_factory = None
    pool.exchange_group = object()

    monkeypatch.setattr(
        pool,
        "_new_channel",
        lambda stream_key: pytest.fail("channel allocation must not start"),
    )
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_oneshot._broadcast_gather_object",
        lambda local_status, group: [local_status, ()],
    )

    with (
        pytest.raises(RuntimeError, match="stable semantic channel_id"),
        pool.capture(7),
    ):
        pass


def test_pool_capture_allows_opposite_order_from_agreed_catalog(monkeypatch):
    target = _make_runtime()
    draft = _make_runtime()
    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=lambda stream_key: _make_runtime(),
    )
    pool._channel_factory = None
    pool.exchange_group = object()
    pool._logical_channels.update({"graph:draft": draft, "graph:target": target})
    monkeypatch.setattr(
        pool,
        "_new_channel",
        lambda stream_key: pytest.fail("channel allocation must not start"),
    )
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_dcp_a2a._current_stream_key",
        lambda device, stream=None: int(stream),
    )

    def gather(local_state, group):
        if local_state == ():
            return [(), ()]
        requested, catalog = local_state
        assert requested == "graph:target"
        return [local_state, ("graph:draft", catalog)]

    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_oneshot._broadcast_gather_object", gather
    )

    with pool.capture(7, channel_id="graph:target") as channel:
        assert channel is target

    assert pool._captured_channel_ids == {"graph:target"}


def test_pool_capture_rejects_divergent_catalog_before_allocation(monkeypatch):
    target = _make_runtime()
    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=lambda stream_key: _make_runtime(),
    )
    pool._channel_factory = None
    pool.exchange_group = object()
    pool._logical_channels["graph:target"] = target
    monkeypatch.setattr(
        pool,
        "_new_channel",
        lambda stream_key: pytest.fail("channel allocation must not start"),
    )

    def gather(local_state, group):
        if local_state == ():
            return [(), ()]
        return [local_state, ("graph:target", ())]

    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_oneshot._broadcast_gather_object", gather
    )

    with (
        pytest.raises(RuntimeError, match="prepared channel catalog differs"),
        pool.capture(7, channel_id="graph:target"),
    ):
        pass


def test_pool_capture_rejects_differing_unprepared_ids(monkeypatch):
    target = _make_runtime()
    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=lambda stream_key: _make_runtime(),
    )
    pool._channel_factory = None
    pool.exchange_group = object()
    pool._logical_channels["graph:target"] = target
    monkeypatch.setattr(
        pool,
        "_new_channel",
        lambda stream_key: pytest.fail("channel allocation must not start"),
    )

    def gather(local_state, group):
        if local_state == ():
            return [(), ()]
        _, catalog = local_state
        return [local_state, ("graph:unknown", catalog)]

    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_oneshot._broadcast_gather_object", gather
    )

    with (
        pytest.raises(RuntimeError, match="unprepared logical channels"),
        pool.capture(7, channel_id="graph:target"),
    ):
        pass


def test_pool_capture_preserves_same_id_convenience_allocation(monkeypatch):
    created = []
    channel = _make_runtime()
    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=lambda stream_key: _make_runtime(),
    )
    pool._channel_factory = None
    pool.exchange_group = object()
    monkeypatch.setattr(
        pool,
        "_new_channel",
        lambda stream_key: created.append(stream_key) or channel,
    )
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_dcp_a2a._current_stream_key",
        lambda device, stream=None: int(stream),
    )
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_oneshot._broadcast_gather_object",
        lambda local_state, group: [local_state, local_state],
    )
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_dcp_a2a._broadcast_gather_object",
        lambda local_state, group: [local_state, local_state],
    )

    with pool.capture(7, channel_id="graph:target") as captured:
        assert captured is channel

    assert created == [None]
    assert pool._logical_channels == {"graph:target": channel}


def test_pool_capture_routes_eager_warmup_to_graph_channel(monkeypatch):
    eager = _make_runtime()
    target = _make_runtime()
    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=lambda stream_key: _make_runtime(),
    )
    pool._channel_factory = None
    pool.exchange_group = object()
    pool._logical_channels.update({"eager:dcp": eager, "graph:target": target})
    monkeypatch.setattr(
        pool,
        "_new_channel",
        lambda stream_key: pytest.fail("channel allocation must not start"),
    )
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_dcp_a2a._current_stream_key",
        lambda device, stream=None: 7 if stream is None else int(stream),
    )
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_oneshot._broadcast_gather_object",
        lambda local_state, group: [local_state, local_state],
    )

    with pool.capture(7, channel_id="graph:target") as captured:
        assert captured is target
        assert pool.for_stream(7, channel_id="eager:dcp") is target
        with pytest.raises(RuntimeError, match="stream-affine"):
            pool.for_stream(8, channel_id="eager:dcp")

    assert pool.for_stream(7, channel_id="eager:dcp") is eager


def test_pool_isolates_reused_capture_stream_keys(monkeypatch):
    created = []
    current_stream = [7]
    capturing = [False]

    def make_channel(stream_key):
        runtime = _make_runtime()
        created.append((stream_key, runtime))
        return runtime

    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=make_channel,
    )
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_dcp_a2a._current_stream_key",
        lambda device, stream=None: (
            current_stream[0] if stream is None else int(stream)
        ),
    )
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_dcp_a2a._is_current_stream_capturing",
        lambda device: capturing[0],
    )

    with pool.capture(7) as target_channel:
        capturing[0] = True
        current_stream[0] = 70
        assert pool.for_stream() is target_channel
        capturing[0] = False

    with pool.capture(7) as draft_channel:
        capturing[0] = True
        current_stream[0] = 70
        assert pool.for_stream() is draft_channel
        capturing[0] = False

    assert target_channel is not draft_channel
    assert pool._channels == {}
    assert target_channel in pool._all_channels
    assert draft_channel in pool._all_channels
    assert [entry[0] for entry in created] == [7, 7]

    pool.close()
    assert target_channel._ext.dispose_calls == [1234]
    assert draft_channel._ext.dispose_calls == [1234]


def test_pool_restores_eager_mapping_after_capture(monkeypatch):
    current_stream = [7]
    capturing = [False]
    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=lambda stream_key: _make_runtime(),
    )
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_dcp_a2a._current_stream_key",
        lambda device, stream=None: (
            current_stream[0] if stream is None else int(stream)
        ),
    )
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_dcp_a2a._is_current_stream_capturing",
        lambda device: capturing[0],
    )

    eager_channel = pool.for_stream()
    with pool.capture(7) as graph_channel:
        capturing[0] = True
        current_stream[0] = 70
        assert pool.for_stream() is graph_channel
        capturing[0] = False

    assert graph_channel is not eager_channel
    assert pool._channels == {7: eager_channel}
    assert graph_channel in pool._all_channels


def test_pool_restores_nested_capture_mappings(monkeypatch):
    current_stream = [7]
    capturing = [False]
    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=lambda stream_key: _make_runtime(),
    )
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_dcp_a2a._current_stream_key",
        lambda device, stream=None: (
            current_stream[0] if stream is None else int(stream)
        ),
    )
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_dcp_a2a._is_current_stream_capturing",
        lambda device: capturing[0],
    )

    eager_channel = pool.for_stream()
    with pool.capture(7) as outer_channel:
        assert pool._channels == {7: outer_channel}

        current_stream[0] = 8
        with pool.capture() as inner_channel:
            capturing[0] = True
            current_stream[0] = 80
            assert pool.for_stream() is inner_channel
            assert pool._channels == {
                7: outer_channel,
                8: inner_channel,
                80: inner_channel,
            }
            capturing[0] = False

        assert pool._channels == {7: outer_channel}
        capturing[0] = True
        current_stream[0] = 70
        assert pool.for_stream() is outer_channel
        capturing[0] = False

    assert eager_channel is not outer_channel
    assert outer_channel is not inner_channel
    assert pool._channels == {7: eager_channel}
    assert pool._capture_channel_stack == []


def test_pool_rolls_back_throwaway_capture_channels(monkeypatch):
    created = []

    def make_channel(stream_key):
        runtime = _make_runtime()
        created.append((stream_key, runtime))
        return runtime

    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=make_channel,
    )
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_dcp_a2a._current_stream_key",
        lambda device, stream=None: 3 if stream is None else int(stream),
    )

    eager_channel = pool.for_stream()
    checkpoint = pool.checkpoint_channels()
    with pool.capture(7) as profile_channel:
        pass

    pool.rollback_channels(checkpoint)

    assert pool._all_channels == [eager_channel]
    assert pool._channels == {3: eager_channel}
    assert profile_channel._ext.dispose_calls == [1234]
    assert eager_channel._ext.dispose_calls == []


def test_pool_coordinates_ipc_teardown_across_ranks(monkeypatch):
    events = []

    class FakeChannel:
        def _close_ipc_imports(self):
            events.append("close-imports")

        def _free_ipc_exports(self):
            events.append("free-exports")

    group = object()
    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        exchange_group=group,
        channel_factory=lambda stream_key: _make_runtime(),
    )
    retained = FakeChannel()
    transient = FakeChannel()
    pool._all_channels = [retained]
    pool._channels = {3: retained}
    checkpoint = pool.checkpoint_channels()
    pool._all_channels.append(transient)
    pool._channels[7] = transient
    monkeypatch.setattr(
        "sparkinfer.comm.pcie.pcie_dcp_a2a.dist.barrier",
        lambda *, group: events.append("barrier"),
    )

    pool.rollback_channels(checkpoint)

    assert events == [
        "barrier",
        "close-imports",
        "barrier",
        "free-exports",
        "barrier",
    ]
    assert pool._all_channels == [retained]
    assert pool._channels == {3: retained}


def test_pool_rejects_channel_rollback_during_capture():
    pool = PCIeDCPA2APool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        max_batch_size=4,
        total_heads=32,
        head_dim=64,
        channel_factory=lambda stream_key: _make_runtime(),
    )
    checkpoint = pool.checkpoint_channels()

    with pool.capture(7), pytest.raises(RuntimeError, match="during capture"):
        pool.rollback_channels(checkpoint)
