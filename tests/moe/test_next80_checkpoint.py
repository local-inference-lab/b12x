"""Real Qwen3-Next layer gates; these do not qualify full-model serving."""

import json
import os
from pathlib import Path

import pytest
import torch

from b12x.moe import fused_moe as moe
from b12x.moe.fused_moe.cache_source import ExpertWeightSource, checkpoint_fingerprint
from b12x.preparation import PreparationSession, PreparedCall


def test_parameter_inventory_distinguishes_cuda_host_views():
    if not torch.cuda.is_available():
        pytest.skip("physical CUDA device required")
    from b12x.sequence._shared.disk_table import MappedHostAllocation
    from b12x.testing.lifecycle import parameter_storage

    owner = MappedHostAllocation((16,), torch.float32, torch.device("cuda", 0))
    try:
        model = torch.nn.Module()
        model.mapped = torch.nn.Parameter(owner.device_view, requires_grad=False)
        model.resident = torch.nn.Parameter(torch.empty(16, device="cuda"))
        rows = {r["name"]: r for r in parameter_storage(model)}
        assert rows["mapped"]["mapped_host"]
        assert not rows["resident"]["mapped_host"]
        assert rows["mapped"]["bytes"] == rows["resident"]["bytes"] == 64
        del model
    finally:
        owner.close()


def shared_module(checkpoint, layer, values):
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
    from vllm.model_executor.models.qwen3_next import Qwen3NextMLP
    from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper

    quant = ModelOptNvFp4Config.from_config(
        json.loads((checkpoint / "hf_quant_config.json").read_text())
    )
    config = VllmConfig(
        model_config=ModelConfig(model=str(checkpoint), dtype="bfloat16")
    )
    prefix = f"model.layers.{layer}.mlp"
    with set_current_vllm_config(config), torch.device("cuda"):
        gate = ReplicatedLinear(
            2048,
            1,
            bias=False,
            params_dtype=torch.bfloat16,
            prefix=prefix + ".shared_expert_gate",
        )
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            shared = Qwen3NextMLP(
                2048,
                512,
                "silu",
                quant_config=quant,
                expert_gate=gate,
                disable_tp=True,
                prefix=prefix + ".shared_expert",
            )
        finally:
            torch.set_default_dtype(original_dtype)
        weights = [
            (n.removeprefix("shared_expert."), v)
            for n, v in values.items()
            if n.startswith("shared_expert.")
        ]
        weights.append(("expert_gate.weight", values["shared_expert_gate.weight"]))
        holder = torch.nn.Module()
        holder.shared = shared
        loaded = AutoWeightsLoader(holder).load_weights(
            [("shared." + n, v) for n, v in weights],
            mapper=WeightsMapper(
                orig_to_new_stacked={
                    ".gate_proj.": (".gate_up_proj.", 0),
                    ".up_proj.": (".gate_up_proj.", 1),
                }
            ),
        )
        for module in shared.modules():
            method = getattr(module, "quant_method", None)
            if method is not None:
                method.process_weights_after_loading(module)
    return shared, config, loaded


@pytest.fixture(scope="module")
def single_gpu_group(tmp_path_factory):
    if (
        not os.environ.get("B12X_TEST_NEXT80_CHECKPOINT")
        or not torch.cuda.is_available()
    ):
        pytest.skip("complete pinned Next80 checkpoint and physical SM120 required")
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
        destroy_model_parallel,
        destroy_distributed_environment,
    )

    rendezvous = tmp_path_factory.mktemp("distributed") / "store"
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        try:
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=f"file://{rendezvous}",
            )
            initialize_model_parallel(
                tensor_model_parallel_size=1, pipeline_model_parallel_size=1
            )
            yield
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


def load_layer(checkpoint, layer):
    from safetensors import safe_open

    config = json.loads((checkpoint / "config.json").read_text())
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    prefix = f"model.layers.{layer}.mlp"
    names = [n for n in index if n.startswith(prefix + ".")]
    values = {}
    for shard in sorted({index[n] for n in names}):
        with safe_open(checkpoint / shard, framework="pt", device="cpu") as handle:
            for name in names:
                if index[name] == shard:
                    values[name.removeprefix(prefix + ".")] = handle.get_tensor(name)
    e, h, i = (
        config[n] for n in ("num_experts", "hidden_size", "moe_intermediate_size")
    )

    def field(projections, suffix):
        return torch.stack(
            [
                torch.cat(
                    [
                        values[f"experts.{expert}.{p}.{suffix}"].reshape(-1)
                        if suffix == "weight_scale_2"
                        else values[f"experts.{expert}.{p}.{suffix}"]
                        for p in projections
                    ]
                )
                for expert in range(e)
            ]
        )

    globals13 = field(("gate_proj", "up_proj"), "weight_scale_2")
    torch.testing.assert_close(globals13[:, 0], globals13[:, 1], atol=0, rtol=0)
    plan = moe.plan_weights(
        source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w31"),
        activation=moe.ActivationSpec(
            mode="a16", nonlinearity="silu", io_dtype=torch.bfloat16
        ),
        geometry=moe.MoEGeometry(num_experts=e, hidden_size=h, intermediate_size=i),
        constraints=moe.WeightPlanConstraints(required_packing="source_native"),
    )
    source = ExpertWeightSource(
        plan=plan,
        weights=moe.PackedWeights(
            w13=field(("gate_proj", "up_proj"), "weight"),
            w2=field(("down_proj",), "weight"),
            w13_block_scales=field(("gate_proj", "up_proj"), "weight_scale"),
            w2_block_scales=field(("down_proj",), "weight_scale"),
            w13_global_scales=globals13[:, 0].contiguous(),
            w2_global_scales=field(("down_proj",), "weight_scale_2").reshape(e),
            checkpoint_fingerprint=checkpoint_fingerprint(checkpoint),
            layer_name=prefix + ".experts",
        ),
    )
    source.validate_values()
    return (
        config,
        source,
        {n: v for n, v in values.items() if not n.startswith("experts.")},
    )


def routed_oracle(source, x, ids, weights):
    """Independent FP32 matvec with the declared BF16 boundaries and ordered sum."""
    w = source.weights
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=x.device,
    )

    def dequant(packed, scales):
        packed = packed.to(x.device)
        raw = torch.stack(
            (lut[(packed & 15).long()], lut[(packed >> 4).long()]), -1
        ).flatten(-2)
        return (
            (raw * scales.to(x.device).float().repeat_interleave(16, -1))
            .bfloat16()
            .float()
        )

    result = torch.zeros_like(x)
    for row in range(x.shape[0]):
        for rank in range(ids.shape[1]):
            expert = int(ids[row, rank])
            gate, up = (
                dequant(w.w13[expert], w.w13_block_scales[expert])
                @ x[row].float()
                * w.w13_global_scales[expert].item()
            ).chunk(2)
            intermediate = (torch.nn.functional.silu(gate) * up).bfloat16().float()
            down = (
                dequant(w.w2[expert], w.w2_block_scales[expert])
                @ intermediate
                * w.w2_global_scales[expert].item()
            ).bfloat16()
            contribution = (down.float() * weights[row, rank]).bfloat16()
            result[row] = (result[row].float() + contribution.float()).bfloat16()
    return result


@pytest.mark.parametrize("layer", [0, 24, 47])
def test_real_next80_routes_graph_and_independent_arithmetic(
    tmp_path, layer, single_gpu_group
):
    location = os.environ.get("B12X_TEST_NEXT80_CHECKPOINT")
    if not location or not torch.cuda.is_available():
        pytest.skip("complete pinned Next80 checkpoint and physical SM120 required")
    assert torch.cuda.get_device_capability() == (12, 0)
    config, source, ordinary = load_layer(Path(location), layer)
    assert config["model_type"] == "qwen3_next"
    torch.manual_seed(131 + layer)
    bounded = os.environ.get("B12X_TEST_NEXT80_ROUTE_SUBSET") == "1"
    rows = 1 if bounded else 4
    x = (torch.randn(rows, config["hidden_size"], device="cuda") * 0.125).bfloat16()
    logits = torch.nn.functional.linear(x, ordinary["gate.weight"].cuda())
    scores = logits.float().softmax(-1)
    weights, ids = torch.topk(scores, config["num_experts_per_tok"], dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    if bounded:
        # Sanitizer control retains real selected rows and weights, with an
        # explicit compact-to-checkpoint ID table. Full geometry is tested above
        # this optional mode; this is not a 512-expert sanitizer claim.
        from dataclasses import replace
        import faulthandler

        faulthandler.dump_traceback_later(90, repeat=True)
        selected = sorted(set(ids.cpu().flatten().tolist()))
        selected += [e for e in range(512) if e not in selected][: 16 - len(selected)]
        index = torch.tensor(selected)
        fields = {
            name: value[index].contiguous()
            for name, value in vars(source.weights).items()
            if isinstance(value, torch.Tensor)
        }
        source = replace(
            source,
            plan=replace(
                source.plan, geometry=replace(source.plan.geometry, num_experts=16)
            ),
            weights=replace(source.weights, **fields),
        )
        ids.copy_(
            torch.tensor(
                [[selected.index(int(e)) for e in row] for row in ids.cpu()],
                device="cuda",
            )
        )
        print("bounded sanitizer compact-to-checkpoint IDs:", selected, flush=True)
    experts = source.plan.geometry.num_experts
    # Preserve actual router selections/weights in one case; adversarial cases
    # separately exercise duplicate and reversed logical routes.
    routes = [(ids.clone(), weights.clone()), (ids.flip(-1), weights.flip(-1))]
    repeated = ids.clone()
    repeated[:, 1] = repeated[:, 0]
    routes.append((repeated, weights.clone()))
    shared, engine_config, loaded = shared_module(Path(location), layer, ordinary)
    assert "shared.expert_gate.weight" in loaded
    from vllm.config import set_current_vllm_config

    plans = [
        moe.plan_execution(
            experts=source,
            capacity=moe.ExecutionCapacity(max_tokens=rows, top_k=10),
            placement=moe.ExpertResidencyPlan(
                total_experts=experts,
                hbm_expert_ids=tuple(range(n)),
                grace_expert_ids=tuple(range(n, experts)),
                layer=source.weights.layer_name,
                model_fingerprint=source.weights.checkpoint_fingerprint,
                workload="real checkpoint layer qualification",
                provenance="pinned native checkpoint",
            ),
            memory_budget=moe.ExpertMemoryBudget(
                hbm_bytes=2 << 30, grace_bytes=2 << 30
            ),
            updates=moe.ResidencyUpdateCapacity(max_pairs=1) if n < experts else None,
        )
        for n in (experts, experts // 2, 1)
    ]

    def prepare(state):
        binding = state.bind(a=x, topk_ids=ids, topk_weights=weights)
        return PreparedCall(
            run=binding.run, output=binding.output, owners=(binding,), close=state.close
        )

    with (
        set_current_vllm_config(engine_config),
        PreparationSession(
            autotune=False, compile_workers=0, cache_dir=tmp_path
        ) as session,
    ):
        session.prepare(
            tuple(
                p.request(name=f"placement-{n}", prepare_call=prepare)
                for n, p in enumerate(plans)
            )
        )
        graphs = []
        for plan in plans:
            binding = plan.prepared.state.bind(a=x, topk_ids=ids, topk_weights=weights)
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                binding.run()
            graphs.append((graph, binding, plan.prepared.state.pointers()))
        from types import SimpleNamespace
        from vllm.model_executor.layers.fused_moe.b12x_cache import (
            ModelOptNvFp4CacheMoE,
        )
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
        from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
            SharedExperts,
        )

        method = object.__new__(ModelOptNvFp4CacheMoE)
        method.moe_kernel = None
        method.prefix = source.weights.layer_name
        method.provider = SimpleNamespace(
            model=SimpleNamespace(
                plans={method.prefix: plans[1]}, observe=lambda *_: None
            )
        )
        wrapper = SharedExperts(
            shared,
            SimpleNamespace(
                moe_parallel_config=SimpleNamespace(
                    enable_eplb=False, use_fi_nvl_two_sided_kernels=False
                )
            ),
            False,
            lambda: False,
        )
        runner = object.__new__(MoERunner)
        torch.nn.Module.__init__(runner)
        runner._shared_experts = wrapper
        runner.router = SimpleNamespace(select_experts=lambda **_: (weights, ids))
        runner.routed_experts = SimpleNamespace(
            quant_method=method,
            forward_modular=lambda **kwargs: method.apply(None, **kwargs),
        )
        calls = []
        hook = shared.register_forward_hook(lambda *_: calls.append(1))

        def composed():
            overlap = wrapper.maybe_forward_async(x)
            shared_result, routed = runner._apply_quant_method(
                x, x, x, shared_experts_overlapping=overlap
            )
            return shared_result + routed

        for _ in range(3):
            combined = composed()
        assert len(calls) == 3
        # Decompose the actual ModelOpt shared MLP at its established BF16
        # activation/output boundaries and apply the checkpoint sigmoid gate.
        gate_up = shared.gate_up_proj(x)[0]
        gate, up = gate_up.float().chunk(2, dim=-1)
        intermediate = (torch.nn.functional.silu(gate) * up).bfloat16()
        shared_expected = shared.down_proj(intermediate)[0] * torch.sigmoid(
            torch.nn.functional.linear(x, ordinary["shared_expert_gate.weight"].cuda())
        )
        torch.testing.assert_close(shared(x), shared_expected, atol=0, rtol=0)
        graph_shared = torch.cuda.CUDAGraph()
        before_calls = len(calls)
        torch.cuda.synchronize()
        with session.capture(), torch.cuda.graph(graph_shared):
            combined = composed()
        assert len(calls) == before_calls + 1
        hook.remove()
        session.freeze()
        for case, (route_ids, route_weights) in enumerate(routes):
            ids.copy_(route_ids)
            weights.copy_(route_weights)
            oracle = routed_oracle(source, x, ids, weights)
            results = []
            for plan, (graph, binding, pointers) in zip(plans, graphs, strict=True):
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize()
                assert before == torch.cuda.memory_stats()["allocation.all.allocated"]
                assert pointers == plan.prepared.state.pointers()
                results.append(binding.output.clone())
                torch.testing.assert_close(binding.output, oracle, atol=1e-3, rtol=0.03)
                assert torch.isfinite(binding.output).all() and torch.count_nonzero(
                    binding.output
                )
            for output in results[1:]:
                torch.testing.assert_close(output, results[0], atol=0, rtol=0)
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph_shared.replay()
            torch.cuda.synchronize()
            assert before == torch.cuda.memory_stats()["allocation.all.allocated"]
            torch.testing.assert_close(
                combined, shared_expected + results[1], atol=0, rtol=0
            )
            if case == 0:
                for plan in plans[1:]:
                    state = plan.prepared.state
                    state.updates.apply(
                        ((experts - 1, 0),),
                        expected=state.updates.snapshot(),
                        quiescent=True,
                    )
                for graph, _binding, _ in graphs:
                    graph.replay()
                torch.cuda.synchronize()
                for (_, binding, _), expected in zip(graphs, results, strict=True):
                    torch.testing.assert_close(binding.output, expected, atol=0, rtol=0)
                graph_shared.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    combined, shared_expected + results[1], atol=0, rtol=0
                )
        graph_shared.reset()
        for graph, _, _ in graphs:
            graph.reset()
    assert all(p.prepared is None for p in plans)
    if bounded:
        faulthandler.cancel_dump_traceback_later()
