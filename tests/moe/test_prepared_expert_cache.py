"""Deferred CPU loading and prepared canonical-cache serving invariants."""

from dataclasses import replace

import pytest
import torch

from b12x.moe import fused_moe as moe
from b12x.moe.fused_moe._cache_updates import CanonicalSlotUpdates
from b12x.moe.fused_moe.cache_source import ExpertWeightSource
from b12x.moe.residency import ResidencyUpdateError
from tests.moe.test_residency_updates import HostTransfer


def source(h=128, i=128, e=4):
    torch.manual_seed(472)
    plan = moe.plan_weights(
        source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w31"),
        activation=moe.ActivationSpec(
            mode="a16", nonlinearity="silu", io_dtype=torch.bfloat16
        ),
        geometry=moe.MoEGeometry(num_experts=e, hidden_size=h, intermediate_size=i),
        constraints=moe.WeightPlanConstraints(required_packing="source_native"),
    )
    return ExpertWeightSource(
        plan=plan,
        weights=moe.PackedWeights(
            w13=torch.randint(0, 256, (e, 2 * i, h // 2), dtype=torch.uint8),
            w2=torch.randint(0, 256, (e, h, i // 2), dtype=torch.uint8),
            w13_block_scales=(torch.rand(e, 2 * i, h // 16) * 0.25 + 0.125).to(
                torch.float8_e4m3fn
            ),
            w2_block_scales=(torch.rand(e, h, i // 16) * 0.25 + 0.125).to(
                torch.float8_e4m3fn
            ),
            w13_global_scales=torch.rand(e) * 0.125 + 0.0625,
            w2_global_scales=torch.rand(e) * 0.125 + 0.0625,
            checkpoint_fingerprint="a" * 64,
            layer_name="layer",
        ),
    )


def declaration(s, capacity=4, top_k=4, hot_count=2):
    e = s.plan.geometry.num_experts
    placement = moe.ExpertResidencyPlan(
        total_experts=e,
        hbm_expert_ids=tuple(range(hot_count)),
        grace_expert_ids=tuple(range(hot_count, e)),
        layer="layer",
        model_fingerprint="a" * 64,
        workload="test",
        provenance="deterministic random source",
    )
    return moe.plan_execution(
        experts=s,
        capacity=moe.ExecutionCapacity(max_tokens=capacity, top_k=top_k),
        placement=placement,
        memory_budget=moe.ExpertMemoryBudget(hbm_bytes=2 << 30, grace_bytes=2 << 30),
        updates=moe.ResidencyUpdateCapacity(max_pairs=min(2, hot_count, e - hot_count))
        if hot_count < e
        else None,
    )


def test_declaration_retains_cpu_source_without_device_work(monkeypatch):
    s = source()

    def forbidden(*args, **kwargs):
        raise AssertionError("declaration initialized CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    plan = declaration(s)
    assert plan.prepared is None and plan.query.experts == 4
    assert s.weights.w13.device.type == "cpu"
    with pytest.raises(ValueError, match="CPU"):
        replace(s, weights=replace(s.weights, w13=s.weights.w13.to("meta")))
    with pytest.raises(ValueError, match="CPU"):
        replace(
            s, weights=replace(s.weights, input_scale=torch.empty(1, device="meta"))
        )
    with pytest.raises(ValueError, match="CPU"):
        replace(s, owners=(torch.empty(1, device="meta"),))
    with pytest.raises(ValueError, match="checkpoint"):
        declaration(
            replace(s, weights=replace(s.weights, checkpoint_fingerprint="b" * 64))
        )


def updates(transfer=None):
    canonical = {
        "w": torch.arange(64, dtype=torch.uint8).reshape(4, 16),
        "s": torch.arange(16, dtype=torch.uint8).reshape(4, 4),
    }
    mapping = torch.tensor(((0, 0), (0, 1), (1, 2), (1, 3)), dtype=torch.int32)
    return CanonicalSlotUpdates(
        resident={n: v[:2].clone() for n, v in canonical.items()},
        canonical=canonical,
        mapping=mapping,
        expert_map=mapping.tolist(),
        before=torch.empty_like(mapping),
        after=torch.empty_like(mapping),
        transfer=transfer or HostTransfer(),
        max_pairs=2,
    )


@pytest.mark.parametrize("failure", range(1, 7))
def test_batch_failure_restores_every_overwritten_victim(failure):
    state = updates(HostTransfer(fail=failure))
    before = state.snapshot()
    with pytest.raises(ResidencyUpdateError) as error:
        state.apply(((2, 0), (3, 1)), expected=before, quiescent=True)
    assert error.value.resumable and before == state.snapshot()
    assert tuple(map(tuple, state.mapping.tolist())) == before.expert_map
    for name, value in state.resident.items():
        torch.testing.assert_close(value, state.canonical[name][:2], atol=0, rtol=0)
    after = state.apply(((2, 0), (3, 1)), expected=before, quiescent=True)
    assert after.generation == 1
    for name, value in state.resident.items():
        torch.testing.assert_close(value, state.canonical[name][2:], atol=0, rtol=0)
    with pytest.raises(ValueError, match="stale"):
        state.apply(((0, 2),), expected=before, quiescent=True)


def test_failed_recovery_poison_is_not_resumable():
    state = updates(HostTransfer(fail=3, persistent=True))
    with pytest.raises(ResidencyUpdateError) as error:
        state.apply(((2, 0), (3, 1)), expected=state.snapshot(), quiescent=True)
    assert not error.value.resumable and not state.snapshot().healthy


@pytest.mark.skipif(not torch.cuda.is_available(), reason="physical SM120 required")
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("geometry", [(128, 128, 4), (2048, 768, 64)])
def test_prepared_native_cache_graph_matches_resident_reference(
    tmp_path, dtype, geometry
):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("physical SM120 required")
    h, i, capacity = geometry
    _graph_parity(tmp_path, dtype, source(h, i), capacity, 4, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="physical SM120 required")
def test_qwen4_main_geometry_same_graph_with_synthetic_weights(tmp_path):
    """Exercise the NVIDIA main-expert dimensions/top-k, not checkpoint values."""
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("physical SM120 required")
    _graph_parity(tmp_path, torch.int32, source(2560, 640, 16), 4, 10, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="physical SM120 required")
@pytest.mark.parametrize("hot_count", [1, 3, 4])
def test_capacity_extremes_match_all_resident_whole_k(tmp_path, hot_count):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("physical SM120 required")
    _graph_parity(tmp_path, torch.int32, source(), 4, 4, hot_count)


def _graph_parity(tmp_path, dtype, s, capacity, top_k, hot_count):
    from b12x.preparation import PreparationSession, PreparedCall
    from b12x._lib.runtime_control import kernel_resolution_guard
    from benchmarks.moe.sm120_residency_poc import make_tier

    e, h = s.plan.geometry.num_experts, s.plan.geometry.hidden_size
    plan = declaration(s, capacity, top_k, hot_count)
    rows = [s.row(expert) for expert in range(e)]
    fields = {n: torch.stack([row[n] for row in rows]) for n in rows[0]}
    control_tier = make_tier(fields, range(e), torch.device("cuda", 0), mapped=False)
    f = control_tier.fields
    experts = moe.prepare_weights(
        plan=s.plan,
        weights=moe.PackedWeights(
            w13=f["w13"],
            w2=f["w2"],
            w13_block_scales=f["s13"],
            w2_block_scales=f["s2"],
            w13_global_scales=f["g13"].view(torch.float32).reshape(e),
            w2_global_scales=f["g2"].view(torch.float32).reshape(e),
        ),
    )
    control = moe.plan_execution(
        experts=experts,
        capacity=moe.ExecutionCapacity(
            max_tokens=capacity, top_k=top_k, route_num_experts=e
        ),
        routing=moe.RoutingSpec(deterministic_output=True),
        override=moe.MoeDecodeConfig(
            backend="w4a16",
            route_planner="internal",
            max_active_clusters=None,
            w4a16_route_mode="packed",
        ),
    )
    a = torch.randn((capacity, h), device="cuda", dtype=torch.bfloat16) * 0.125
    ids = (
        torch.arange(capacity * top_k, device="cuda", dtype=dtype).reshape(
            capacity, top_k
        )
        % e
    )
    weights = torch.rand((capacity, top_k), device="cuda")
    output = torch.empty_like(a)
    identity = torch.arange(e, device="cuda", dtype=torch.int32)
    scratch = None

    def control_call(state):
        nonlocal scratch
        scratch = tuple(
            torch.empty(x.shape, dtype=x.dtype, device=x.device)
            for x in state.scratch.scratch_specs()
        )
        binding = state.bind(
            scratch=scratch,
            a=a,
            topk_ids=ids,
            topk_weights=weights,
            output=output,
            input_scales_static=True,
            route_expert_map=identity,
        )
        return PreparedCall(run=binding.run, output=output, owners=(binding, scratch))

    def call(state):
        binding = state.bind(a=a, topk_ids=ids, topk_weights=weights)
        return PreparedCall(
            run=binding.run, output=binding.output, owners=(binding,), close=state.close
        )

    with PreparationSession(
        autotune=False, compile_workers=0, cache_dir=tmp_path
    ) as session:
        session.prepare(
            (
                plan.request(name="cache", prepare_call=call),
                control.request(name="reference", prepare_call=control_call),
            )
        )
        state = plan.prepared.state
        pointers = state.pointers()
        graphs = []
        for m in dict.fromkeys((1, 4, 2, capacity)):
            binding = moe.bind(
                plan, a=a[:m], topk_ids=ids[:m], topk_weights=weights[:m]
            )
            ref = moe.bind(
                control,
                scratch=scratch,
                a=a[:m],
                topk_ids=ids[:m],
                topk_weights=weights[:m],
                output=output[:m],
                input_scales_static=True,
                route_expert_map=identity,
            )
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                moe.run(binding=binding)
            graphs.append((graph, binding, ref))
        session.freeze()
        pair_count = min(2, hot_count, e - hot_count)
        forward = ((hot_count, 0), (e - 1, 1))[:pair_count]
        backward = tuple((b, a) for a, b in forward)
        for pairs in (forward, backward, forward[:1]):
            ids.random_(0, e)
            ids[0, 0] = min(hot_count, e - 1)
            ids[0, 1] = min(hot_count, e - 1)
            ids[0, 2] = e - 1
            ids[0, 3] = 2**40 if dtype == torch.int64 else -1
            a.mul_(0.9375)
            original_outputs = []
            for graph, binding, _ in graphs:
                graph.replay()
                original_outputs.append(binding.output.clone())
            torch.cuda.synchronize()
            if pairs:
                state.updates.apply(
                    pairs, expected=state.updates.snapshot(), quiescent=True
                )
            for (graph, binding, ref), original_output in zip(
                graphs, original_outputs, strict=True
            ):
                before = torch.cuda.memory_stats()
                with kernel_resolution_guard("prepared cache replay"):
                    graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                assert (
                    before["allocation.all.allocated"]
                    == after["allocation.all.allocated"]
                )
                torch.testing.assert_close(
                    binding.output, original_output, atol=0, rtol=0
                )
                # Reference route pack accepts -1 but not oversized int64 IDs.
                original = ids.clone()
                ids.masked_fill_(ids >= e, -1)
                moe.run(binding=ref)
                ids.copy_(original)
                torch.testing.assert_close(binding.output, ref.output, atol=0, rtol=0)
                assert torch.isfinite(binding.output).all() and torch.count_nonzero(
                    binding.output
                )
            assert state.pointers() == pointers
        for graph, _, _ in graphs:
            graph.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="physical SM120 required")
def test_checkpoint_layer_same_graph_canonical_fills(tmp_path):
    """Use all source experts and production route density when a checkpoint exists."""
    import hashlib
    import json
    import os
    from pathlib import Path
    from safetensors import safe_open

    location = os.environ.get("B12X_TEST_NVFP4_CHECKPOINT")
    if not location or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("set B12X_TEST_NVFP4_CHECKPOINT on physical SM120")
    checkpoint = Path(location)
    config = json.loads((checkpoint / "config.json").read_text())
    e, h, i, top_k = (
        config[n]
        for n in (
            "num_experts",
            "hidden_size",
            "moe_intermediate_size",
            "num_experts_per_tok",
        )
    )
    prefix = "model.layers.12.mlp.experts"
    values = {}
    for shard in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(prefix + "."):
                    values[key] = handle.get_tensor(key)
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())

    def field(projections, suffix):
        return torch.stack(
            [
                torch.cat(
                    [
                        values[f"{prefix}.{expert}.{p}.{suffix}"].reshape(-1)
                        if suffix == "weight_scale_2"
                        else values[f"{prefix}.{expert}.{p}.{suffix}"]
                        for p in projections
                    ]
                )
                for expert in range(e)
            ]
        )

    globals13 = field(("gate_proj", "up_proj"), "weight_scale_2")
    torch.testing.assert_close(globals13[:, 0], globals13[:, 1], atol=0, rtol=0)
    weight_plan = moe.plan_weights(
        source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w31"),
        activation=moe.ActivationSpec(
            mode="a16", nonlinearity="silu", io_dtype=torch.bfloat16
        ),
        geometry=moe.MoEGeometry(num_experts=e, hidden_size=h, intermediate_size=i),
        constraints=moe.WeightPlanConstraints(required_packing="source_native"),
    )
    s = ExpertWeightSource(
        plan=weight_plan,
        weights=moe.PackedWeights(
            w13=field(("gate_proj", "up_proj"), "weight"),
            w2=field(("down_proj",), "weight"),
            w13_block_scales=field(("gate_proj", "up_proj"), "weight_scale"),
            w2_block_scales=field(("down_proj",), "weight_scale"),
            w13_global_scales=globals13[:, 0].contiguous(),
            w2_global_scales=field(("down_proj",), "weight_scale_2").reshape(e),
            checkpoint_fingerprint="a" * 64,
            layer_name="layer",
        ),
    )
    print("checkpoint layer SHA256:", digest.hexdigest())
    _graph_parity(tmp_path, torch.int64, s, 64, top_k, e // 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="physical SM120 required")
def test_single_token_duplicate_routes_match_independent_top1(tmp_path):
    """An expert-packed block may contain several routes even at one token."""
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("physical SM120 required")
    from b12x.preparation import PreparationSession, PreparedCall
    from b12x._lib.runtime_control import kernel_resolution_guard

    s = source(2048, 768)
    plans = (declaration(s, 1, 4, 2), declaration(s, 1, 1, 4))
    a = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16) * 0.125
    ids = torch.tensor([[0, 0, 2, 2]], device="cuda", dtype=torch.int64)
    weights = torch.tensor([[0.125, 0.25, 0.375, 0.25]], device="cuda")
    row_ids = [ids[:, rank : rank + 1].clone() for rank in range(4)]
    outputs = [torch.empty_like(a) for _ in range(4)]
    calls = []

    def prepare(state, single):
        bindings = (
            tuple(
                state.bind(
                    a=a,
                    topk_ids=row_ids[rank],
                    topk_weights=weights[:, rank : rank + 1],
                    output=outputs[rank],
                )
                for rank in range(4)
            )
            if single
            else (state.bind(a=a, topk_ids=ids, topk_weights=weights),)
        )

        def run():
            for binding in bindings:
                binding.run()
            return bindings[0].output

        calls.append((run, bindings))
        return PreparedCall(
            run=run, output=bindings[0].output, owners=bindings, close=state.close
        )

    with PreparationSession(
        autotune=False, compile_workers=0, cache_dir=tmp_path
    ) as session:
        session.prepare(
            tuple(
                p.request(
                    name=str(index),
                    prepare_call=lambda state, single=bool(index): prepare(
                        state, single
                    ),
                )
                for index, p in enumerate(plans)
            )
        )
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            for run, _ in calls:
                run()
        session.freeze()
        state = plans[0].prepared.state
        pointers = state.pointers()
        for routes in ((0, 0, 2, 2), (2, 2, 2, 2), (1, 3, 1, 3)):
            ids.copy_(torch.tensor([routes], device="cuda"))
            for rank, row in enumerate(row_ids):
                row.copy_(ids[:, rank : rank + 1])
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            with kernel_resolution_guard("duplicate expert graph"):
                graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            expected = torch.zeros_like(a, dtype=torch.float32)
            for output in outputs:
                expected.add_(output.float())
            torch.testing.assert_close(
                calls[0][1][0].output, expected.bfloat16(), atol=0, rtol=0
            )
            assert state.pointers() == pointers
        graph.reset()
