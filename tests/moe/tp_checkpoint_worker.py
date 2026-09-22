"""Physical TP layer diagnostic using the maintained checkpoint loader."""

import argparse
import importlib.util
import json
import os
from dataclasses import asdict
from pathlib import Path
import torch
import torch.distributed as dist
from safetensors import safe_open
from vllm.config import (
    VllmConfig,
    ModelConfig,
    KernelConfig,
    ParallelConfig,
    set_current_vllm_config,
)
from vllm.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
    destroy_model_parallel,
    destroy_distributed_environment,
    tensor_model_parallel_all_reduce,
    get_tp_group,
)
from vllm.distributed.parallel_state import graph_capture
from vllm.forward_context import set_forward_context
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextSparseMoeBlock,
    Qwen3NextModel,
)
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.v1.worker.workspace import (
    init_workspace_manager,
    reset_workspace_manager,
    collect_cuda_graph_capture_resources,
)
from b12x.moe import fused_moe as moe
from b12x.preparation import PreparationSession, PreparedCall

p = argparse.ArgumentParser()
p.add_argument("--layer", type=int, default=0)
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--resident-counts", type=int, nargs="+", default=[512, 256, 1])
args = p.parse_args()
if any(n < 1 or n > 512 for n in args.resident_counts):
    p.error("resident counts must be between 1 and the checkpoint's 512 experts")
rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
checkpoint = args.checkpoint
root = args.output
root.mkdir(parents=True, exist_ok=True)
settings = dict(
    mode="profile",
    activation="w4a16",
    profile_path=str(root / "unused-profile.json"),
    workload="tp-real-layer",
    expert_device_bytes=2 << 30,
    host_bytes=128 << 30,
    kv_reserved_bytes=2 << 30,
    graph_reserved_bytes=512 << 20,
    device_safety_bytes=1 << 30,
    host_safety_bytes=8 << 30,
)
config = VllmConfig(
    model_config=ModelConfig(
        model=str(checkpoint),
        dtype="bfloat16",
        quantization="modelopt_fp4",
        max_model_len=2048,
    ),
    parallel_config=ParallelConfig(
        tensor_parallel_size=world, disable_custom_all_reduce=True
    ),
    kernel_config=KernelConfig(moe_backend="b12x"),
    additional_config={"b12x_expert_cache": settings},
)
spec = importlib.util.spec_from_file_location(
    "oracle", Path(__file__).with_name("test_next80_checkpoint.py")
)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)
progress = oracle.checkpoint_progress
with set_current_vllm_config(config), torch.inference_mode():
    from vllm.distributed.parallel_state import set_custom_all_reduce

    set_custom_all_reduce(False)
    progress(f"layer-{args.layer}:model-parallel-init-enter")
    init_distributed_environment(
        world_size=world, rank=rank, local_rank=rank, distributed_init_method="env://"
    )
    initialize_model_parallel(
        tensor_model_parallel_size=world, pipeline_model_parallel_size=1
    )
    init_workspace_manager(torch.device("cuda", rank))
    progress(f"layer-{args.layer}:model-parallel-init-return")
    try:
        torch.set_default_dtype(torch.bfloat16)
        prefix = f"model.layers.{args.layer}."
        with torch.device("cuda"):
            holder = torch.nn.Module()
            holder.mlp = Qwen3NextSparseMoeBlock(config, prefix=prefix + "mlp")
        index = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        names = [n for n in index if n.startswith(prefix + "mlp.")]

        def weights():
            for shard in sorted({index[n] for n in names}):
                with safe_open(checkpoint / shard, framework="pt", device="cpu") as f:
                    for name in names:
                        if index[name] == shard:
                            yield name.removeprefix(prefix), f.get_tensor(name)

        loaded = AutoWeightsLoader(holder).load_weights(
            weights(), mapper=Qwen3NextModel.hf_to_vllm_mapper
        )
        for module in holder.modules():
            method = getattr(module, "quant_method", None)
            if method is not None:
                method.process_weights_after_loading(module)
        routed = holder.mlp.experts.routed_experts
        method = routed.quant_method
        model = method.provider.model
        source = model.sources[method.prefix]
        progress(f"layer-{args.layer}:loaded")
        assert source.plan.geometry.intermediate_size == 512 // world
        # Full matrices below belong only to the independent layer oracle, not the runtime source.
        _, full, _ = oracle.load_layer(checkpoint, args.layer)
        local = source.weights
        global_w = full.weights
        i = 512 // world
        s = rank * i
        oracle.assert_checkpoint_close(
            local.w13,
            torch.cat(
                [global_w.w13[:, s : s + i], global_w.w13[:, 512 + s : 512 + s + i]], 1
            ),
            atol=0,
            rtol=0,
        )
        oracle.assert_checkpoint_close(
            local.w2, global_w.w2[:, :, s // 2 : (s + i) // 2], atol=0, rtol=0
        )
        torch.manual_seed(131 + args.layer)
        x = (torch.randn(4, 2048, device="cuda") * 0.125).bfloat16()
        logits = holder.mlp.gate(x)[0]
        weights, ids = holder.mlp.experts.router.select_experts(
            hidden_states=x,
            router_logits=logits,
            topk_indices_dtype=method.topk_indices_dtype,
        )
        ids = ids.clone()
        weights = weights.clone()
        cpu_ids = ids.cpu()
        gathered = [torch.empty_like(cpu_ids) for _ in range(world)]
        progress(f"layer-{args.layer}:route-all-gather-enter")
        # Route agreement is a host assertion, not a model collective. Use the
        # engine's CPU control group instead of creating another NCCL context.
        dist.all_gather(gathered, cpu_ids, group=get_tp_group().cpu_group)
        assert all(torch.equal(cpu_ids, v) for v in gathered)
        progress(f"layer-{args.layer}:route-all-gather-return")
        reports = []
        placement_reference = None
        for resident in args.resident_counts:
            plan = moe.plan_execution(
                experts=source,
                capacity=moe.ExecutionCapacity(max_tokens=4, top_k=10),
                placement=moe.ExpertResidencyPlan(
                    total_experts=512,
                    hbm_expert_ids=tuple(range(resident)),
                    grace_expert_ids=tuple(range(resident, 512)),
                    layer=method.prefix,
                    model_fingerprint=model.checkpoint_id,
                    workload="tp-real-layer",
                    provenance="checkpoint loader",
                ),
                memory_budget=moe.ExpertMemoryBudget(
                    hbm_bytes=2 << 30, grace_bytes=2 << 30
                ),
                updates=moe.ResidencyUpdateCapacity(max_pairs=1)
                if resident < 512
                else None,
            )

            def prepare(state):
                b = state.bind(a=x, topk_ids=ids, topk_weights=weights)
                return PreparedCall(
                    run=b.run, output=b.output, owners=(b,), close=state.close
                )

            with PreparationSession(
                autotune=False,
                compile_workers=0,
                cache_dir=root / f"tp-layer-cache-{rank}",
            ) as session:
                session.prepare((plan.request(name="layer", prepare_call=prepare),))
                progress(f"layer-{args.layer}:resident-{resident}:prepared")
                state = plan.prepared.state
                model.plans[method.prefix] = plan
                binding = state.bind(a=x, topk_ids=ids, topk_weights=weights)
                graph = torch.cuda.CUDAGraph()
                with session.capture(), torch.cuda.graph(graph):
                    binding.run()
                pointers = state.pointers()
                original_ids, original_weights = ids.clone(), weights.clone()
                arithmetic = []
                for case in ("ordinary", "reordered", "duplicate"):
                    ids.copy_(
                        original_ids if case != "reordered" else original_ids.flip(-1)
                    )
                    weights.copy_(
                        original_weights
                        if case != "reordered"
                        else original_weights.flip(-1)
                    )
                    if case == "duplicate":
                        ids[:, 1].copy_(ids[:, 0])
                    before = torch.cuda.memory_stats()["allocation.all.allocated"]
                    graph.replay()
                    torch.cuda.synchronize()
                    assert (
                        before == torch.cuda.memory_stats()["allocation.all.allocated"]
                    )
                    assert pointers == state.pointers()
                    assert torch.isfinite(
                        binding.output.cpu()
                    ).all() and torch.count_nonzero(binding.output.cpu())
                    expected_local = oracle.routed_oracle(source, x, ids, weights)
                    oracle.assert_checkpoint_close(
                        binding.output, expected_local, atol=0.001, rtol=0.03
                    )
                    reduced = tensor_model_parallel_all_reduce(binding.output.clone())
                    expected = oracle.routed_oracle(full, x, ids, weights)
                    oracle.assert_checkpoint_close(
                        reduced, expected, atol=0.003, rtol=0.06
                    )
                    arithmetic.append(
                        dict(
                            routes=case,
                            local_max_abs=float(
                                (
                                    binding.output.cpu().float()
                                    - expected_local.cpu().float()
                                )
                                .abs()
                                .max()
                            ),
                            reduced_max_abs=float(
                                (reduced.cpu().float() - expected.cpu().float())
                                .abs()
                                .max()
                            ),
                            reduced_relative_l2=float(
                                (reduced.cpu().float() - expected.cpu().float()).norm()
                                / expected.cpu().float().norm()
                            ),
                            reduced_cosine=float(
                                torch.nn.functional.cosine_similarity(
                                    reduced.cpu().float().flatten(),
                                    expected.cpu().float().flatten(),
                                    dim=0,
                                )
                            ),
                        )
                    )
                    if case == "ordinary":
                        if placement_reference is None:
                            placement_reference = reduced.clone()
                        else:
                            oracle.assert_checkpoint_close(
                                reduced, placement_reference, atol=0, rtol=0
                            )
                    progress(f"layer-{args.layer}:resident-{resident}:case-{case}")
                ids.copy_(original_ids)
                weights.copy_(original_weights)
                # Exercise the real model wrapper and its shared-expert/final-reduction ownership.
                with set_forward_context(None, config, num_tokens=4):
                    for _ in range(3):
                        combined = holder.mlp(x)
                    shared = holder.mlp.shared_expert(x)
                    graph.replay()
                    torch.cuda.synchronize()
                    combined_expected = tensor_model_parallel_all_reduce(
                        shared + binding.output
                    )
                    oracle.assert_checkpoint_close(
                        combined, combined_expected, atol=0, rtol=0
                    )
                    whole_graph = torch.cuda.CUDAGraph()
                    with (
                        graph_capture(torch.device("cuda", rank)) as gc,
                        session.capture(),
                        collect_cuda_graph_capture_resources() as owners,
                        torch.cuda.graph(whole_graph, stream=gc.stream),
                    ):
                        combined_graph = holder.mlp(x)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                whole_graph.replay()
                torch.cuda.synchronize()
                assert before == torch.cuda.memory_stats()["allocation.all.allocated"]
                oracle.assert_checkpoint_close(combined_graph, combined, atol=0, rtol=0)
                if resident < 512:
                    slots = state.updates.snapshot()
                    pairs = ((resident, 0),)
                    state.updates.stage(pairs, expected=slots, quiescent=True)
                    get_tp_group().barrier()
                    state.updates.publish_staged()
                    get_tp_group().barrier()
                    state.updates.finish_staged()
                    get_tp_group().barrier()
                    before = torch.cuda.memory_stats()["allocation.all.allocated"]
                    whole_graph.replay()
                    torch.cuda.synchronize()
                    assert (
                        before == torch.cuda.memory_stats()["allocation.all.allocated"]
                    )
                    assert pointers == state.pointers()
                    oracle.assert_checkpoint_close(
                        combined_graph, combined, atol=0, rtol=0
                    )
                    assert state.updates.snapshot().generation == 1
                reports.append(
                    dict(
                        resident=resident,
                        source_bytes=source.source_bytes,
                        storage=asdict(source.storage),
                        loaded_parameters=len(loaded),
                        exact_placement=True,
                        exact_shared_composition=True,
                        no_replay_allocations=True,
                        arithmetic=arithmetic,
                    )
                )
                whole_graph.reset()
                graph.reset()
                del owners, binding
                model.plans.clear()
                progress(f"layer-{args.layer}:resident-{resident}:released")
        (root / f"tp-layer-{args.layer}-rank{rank}.json").write_text(
            json.dumps(reports, indent=2, default=str)
        )
        model.close()
        del holder, method, model, source, full
    finally:
        reset_workspace_manager()
        destroy_model_parallel()
        destroy_distributed_environment()
        progress(f"layer-{args.layer}:released")
