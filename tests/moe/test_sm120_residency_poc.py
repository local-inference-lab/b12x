"""Bounded SM120 native NVFP4 mapped-host cache qualification."""

import pytest
import torch

from benchmarks.moe.sm120_residency_poc import Experiment, exercise, load_layer


def checkpoint_fields(experts=4, hidden=128, intermediate=128):
    generator = torch.Generator().manual_seed(78)
    fields = {}
    for e in range(experts):
        for proj, n, k in (
            ("up_proj", intermediate, hidden),
            ("gate_proj", intermediate, hidden),
            ("down_proj", hidden, intermediate),
        ):
            prefix = f"layer.experts.{e}.{proj}"
            fields[prefix + ".weight"] = torch.randint(
                0, 256, (n, k // 2), dtype=torch.uint8, generator=generator
            )
            fields[prefix + ".weight_scale"] = (
                torch.rand(n, k // 16, generator=generator) * 0.1 + 0.125
            ).to(torch.float8_e4m3fn)
            fields[prefix + ".weight_scale_2"] = torch.tensor(0.025 * (e + 1))
    return fields


def write_layer(path, fields):
    from safetensors.torch import save_file

    save_file(fields, path / "model-00001-of-00001.safetensors")


def test_unindexed_checkpoint_bytes_and_scale_contract(tmp_path):
    fields = checkpoint_fields()
    write_layer(tmp_path, fields)
    source, digest = load_layer(tmp_path, "layer.experts", 4)
    assert digest == load_layer(tmp_path, "layer.experts", 4)[1]
    assert source["w13"].shape == (4, 256, 64)
    torch.testing.assert_close(
        source["w13"][0, :128], fields["layer.experts.0.up_proj.weight"], atol=0, rtol=0
    )
    torch.testing.assert_close(
        source["w13"][0, 128:],
        fields["layer.experts.0.gate_proj.weight"],
        atol=0,
        rtol=0,
    )
    assert source["g13"].view(torch.float32).flatten().tolist() == [
        fields[f"layer.experts.{e}.up_proj.weight_scale_2"].item() for e in range(4)
    ]
    fields["layer.experts.0.gate_proj.weight_scale_2"] *= 2
    write_layer(tmp_path, fields)
    with pytest.raises(ValueError, match="never requantizes"):
        load_layer(tmp_path, "layer.experts", 4)


def test_checkpoint_missing_or_wrong_scales_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="missing checkpoint"):
        load_layer(tmp_path, "layer.experts", 4)
    fields = checkpoint_fields()
    fields["layer.experts.0.down_proj.weight_scale"] = torch.ones(128, 8)
    write_layer(tmp_path, fields)
    with pytest.raises(ValueError, match="E4M3 K16"):
        load_layer(tmp_path, "layer.experts", 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires physical SM120")
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("write_combined", [False, True])
def test_native_policy_exchange_same_graph(dtype, write_combined, tmp_path):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    write_layer(tmp_path, checkpoint_fields())
    source, _ = load_layer(tmp_path, "layer.experts", 4)
    experiment = Experiment(
        source,
        hot=2,
        capacity=4,
        topk=4,
        id_dtype=dtype,
        cache_dir=tmp_path / "cache",
        backing_write_combined=write_combined,
        journal_write_combined=write_combined,
    )
    try:
        from tests._reference.w4a16_reference import (
            moe_reference_w4a16,
            compare_to_reference,
        )
        from b12x.moe import fused_moe as moe

        experiment.ids[0].copy_(torch.arange(4))
        experiment.refresh(1)
        actual = moe.run(binding=experiment.bind(1)[2])
        raw = [source[name].cuda() for name in ("w13", "s13", "g13", "w2", "s2", "g2")]
        raw[2], raw[5] = (
            raw[2].view(torch.float32).flatten(),
            raw[5].view(torch.float32).flatten(),
        )
        reference = moe_reference_w4a16(
            experiment.a[:1],
            *raw,
            experiment.ids[:1],
            experiment.weights[:1],
            4,
            128,
            128,
        )
        assert compare_to_reference(actual, reference).cos > 0.9999
        records = exercise(experiment, live_counts=(1, 4, 2))
        policy = [row for row in records if row["kind"] == "policy"]
        assert len(records) == 30 and len(policy) == 18
        assert all(
            row["bitwise_equal"] and row["replay_allocator_events"] == 0
            for row in policy
        )
        assert all(
            row["ordered_sum_exact"]
            for row in records
            if row["kind"] == "route-boundary"
        )
        assert any(row["decision"]["pairs"] for row in policy)
    finally:
        experiment.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires physical SM120")
def test_ordered_sum_rounding_and_invalid_routes():
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    from benchmarks.moe.sm120_residency_support import compile_support, invoke

    _, reduce = compile_support(4, 128, 4, torch.int64)
    # BF16 route contributions: separately casting tier sums loses the unit.
    values = torch.tensor(
        [256.0, 1.0, -256.0, 0.0], dtype=torch.bfloat16, device="cuda"
    )
    hot = values[:, None].expand(4, 128).contiguous()
    cold = hot.clone()
    ids = torch.tensor([[0, 1, 2, 2**40]], dtype=torch.int64, device="cuda")
    mapping = torch.tensor(
        [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.int32, device="cuda"
    )
    out = torch.empty(1, 128, dtype=torch.bfloat16, device="cuda")
    invoke(reduce, (hot, cold, ids, mapping, out), (1,))
    torch.testing.assert_close(out, torch.ones_like(out), atol=0, rtol=0)
    separate = values[:2].float().sum().bfloat16() + values[2]
    assert separate.item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires physical SM120")
@pytest.mark.parametrize("failure_call", [2, 14, 26])
def test_cacheable_host_exchange_failure_preserves_graph(
    failure_call, tmp_path, monkeypatch
):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    from b12x.moe.residency import ResidencyUpdateError
    from benchmarks.moe.sm120_residency_spectrum import Graphs

    write_layer(tmp_path, checkpoint_fields())
    source, _ = load_layer(tmp_path, "layer.experts", 4)
    e = Experiment(
        source,
        hot=2,
        capacity=4,
        topk=4,
        backing_write_combined=False,
        journal_write_combined=False,
        cache_dir=tmp_path / "cache",
    )
    graphs = None
    try:
        graphs = Graphs(e, 1)
        graphs.inputs(torch.arange(4).reshape(1, 4))
        graphs.validate(allocator=True)
        expected, pointers = e.updates.snapshot(), e.pointers()
        copy = e.updates.transfer.copy
        calls = 0

        def fail_after_write(destination, source):
            nonlocal calls
            calls += 1
            copy(destination, source)
            if calls == failure_call:
                raise OSError("injected failure after a submitted copy")

        monkeypatch.setattr(e.updates.transfer, "copy", fail_after_write)
        with pytest.raises(ResidencyUpdateError) as caught:
            e.updates.exchange(((0, 2),), expected=expected, quiescent=True)
        assert caught.value.resumable
        assert expected == e.updates.snapshot()
        graphs.validate(allocator=True)
        assert pointers == e.pointers()
        e.updates.exchange(((0, 2),), expected=expected, quiescent=True)
        graphs.validate(allocator=True)
    finally:
        if graphs:
            graphs.close()
        e.close()
