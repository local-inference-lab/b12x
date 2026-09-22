"""Frozen real Qwen3-Next routed W4A4 repeatability diagnostic.

The input fixture contains CPU tensors ``x``, logical ``ids`` and actual route
``weights`` from a retained model execution. This tool does not change backend
arithmetic. The optional reference independently dequantizes weights and runs
GEMMs/reduction, while retaining the native activation quantizer.
"""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
import torch
from safetensors import safe_open
from vllm.config import VllmConfig, ModelConfig, KernelConfig, set_current_vllm_config
from vllm.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
    destroy_model_parallel,
    destroy_distributed_environment,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextSparseMoeBlock,
    Qwen3NextModel,
)
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.v1.worker.workspace import (
    init_workspace_manager,
    current_workspace_manager,
    reset_workspace_manager,
    collect_cuda_graph_capture_resources,
)

p = argparse.ArgumentParser()
p.add_argument("--backend", default="flashinfer_cutlass")
p.add_argument("--autotune", action="store_true")
p.add_argument("--tactics")
p.add_argument("--uva", action="store_true")
p.add_argument("--reference", action="store_true")
p.add_argument("--model", type=Path, required=True)
p.add_argument("--input-fixture", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--build-manifest", type=Path, required=True)
p.add_argument("--checkpoint-identity", type=Path, required=True)
args = p.parse_args()
args.name = "repeatability"
root = args.output
root.mkdir(parents=True, exist_ok=False)
checkpoint = args.model
from b12x.testing.artifacts import verify_from_file, sha256
from scripts._sm103_source import source_identity
from b12x.integration.vllm.checkpoint_identity import checkpoint_identity

build = verify_from_file(args.build_manifest)
identity = checkpoint_identity(checkpoint, receipt=args.checkpoint_identity)
config = VllmConfig(
    model_config=ModelConfig(
        model=str(checkpoint), dtype="bfloat16", quantization="modelopt_fp4"
    ),
    kernel_config=KernelConfig(moe_backend=args.backend),
)


def tensor_hash(value):
    return hashlib.sha256(
        value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


with (
    tempfile.TemporaryDirectory() as tmp,
    set_current_vllm_config(config),
    torch.inference_mode(),
):
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"file://{tmp}/store",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1
    )
    init_workspace_manager(torch.device("cuda", 0))
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("cuda"):
            holder = torch.nn.Module()
            holder.mlp = Qwen3NextSparseMoeBlock(config, prefix="model.layers.0.mlp")
        if args.uva:
            from vllm.model_executor.offloader import UVAOffloader

            offloader = UVAOffloader(2 << 30, {"experts"})
            offloader.wrap_modules(iter([holder]))
        index = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        prefix = "model.layers.0."
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
        raw = {
            n: v.detach().cpu().clone()
            for n, v in holder.mlp.experts.routed_experts.named_parameters()
        }
        from vllm.model_executor.model_loader.utils import device_loading_context

        for module in holder.modules():
            method = getattr(module, "quant_method", None)
            if method is not None:
                with device_loading_context(module, torch.device("cuda", 0)):
                    method.process_weights_after_loading(module)
        routed = holder.mlp.experts.routed_experts
        method = routed.quant_method
        captured = torch.load(args.input_fixture, weights_only=True, map_location="cpu")
        x, ids, weights = (captured[n].cuda() for n in ("x", "ids", "weights"))
        assert x.dtype == torch.bfloat16 and x.ndim == 2
        assert ids.shape == weights.shape == (x.shape[0], 10)
        assert ids.min() >= 0 and ids.max() < 512 and torch.isfinite(weights).all()

        def run(method=method, routed=routed):
            return method.apply(routed, x, weights, ids, None, None)

        hashes = {
            n: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=tensor_hash(v))
            for n, v in routed.named_parameters()
        }
        if args.tactics:
            from flashinfer.autotuner import AutoTuner

            AutoTuner.get().load_configs(args.tactics)
        if args.autotune:
            from flashinfer.autotuner import autotune, AutoTuner

            with autotune():
                run()
            AutoTuner.get().save_configs(str(root / (args.name + "-tactics.json")))
        warm = []
        for _ in range(8):
            warm.append(run().clone())
        torch.cuda.synchronize()
        outputs = []
        for _ in range(20):
            outputs.append(run().clone())
        graph = torch.cuda.CUDAGraph()
        with collect_cuda_graph_capture_resources() as owners, torch.cuda.graph(graph):
            out = run()
        for _ in range(20):
            graph.replay()
            outputs.append(out.clone())
        for _ in range(10):
            junk = torch.empty(17 << 20, device="cuda", dtype=torch.uint8)
            junk.fill_(0xA5)
            del junk
            graph.replay()
            outputs.append(out.clone())
        torch.cuda.synchronize()
        report = dict(
            backend=args.backend,
            selected=str(method.nvfp4_backend),
            experts_class=type(method.moe_kernel.fused_experts).__name__,
            weights=hashes,
            input_hash=tensor_hash(x),
            ids_hash=tensor_hash(ids),
            route_weights_hash=tensor_hash(weights),
            finite=all(bool(torch.isfinite(v).all()) for v in outputs),
            hashes=[tensor_hash(v) for v in outputs],
            max_difference=max(
                (v.float() - outputs[0].float()).abs().max().item() for v in outputs
            ),
            workspace=str(vars(current_workspace_manager()).keys()),
            loaded_parameters=len(loaded),
        )
        if args.reference:
            from benchmarks.moe.nvfp4_arithmetic_reference import reference

            ref = reference(raw, x, ids, weights, method.nvfp4_backend)
            delta = outputs[0].float() - ref.float()
            report["reference"] = dict(
                max_abs=delta.abs().max().item(),
                relative_l2=(delta.norm() / ref.float().norm()).item(),
                cosine=torch.nn.functional.cosine_similarity(
                    outputs[0].float().flatten(), ref.float().flatten(), dim=0
                ).item(),
                max_output=ref.abs().max().item(),
            )
        report["post_weights"] = {
            n: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=tensor_hash(v))
            for n, v in routed.named_parameters()
        }
        assert report["post_weights"] == hashes, (
            "postprocessed weights changed during execution"
        )
        if args.reference:
            report["reference_all"] = [
                dict(
                    max_abs=(v.float() - ref.float()).abs().max().item(),
                    relative_l2=(
                        (v.float() - ref.float()).norm() / ref.float().norm()
                    ).item(),
                )
                for v in outputs
            ]
            assert all(
                r["max_abs"] <= 0.004 and r["relative_l2"] <= 0.01
                for r in report["reference_all"]
            )
        report["artifacts"] = verify_from_file(args.build_manifest, loaded_only=True)
        report["source"] = source_identity(Path(__file__).resolve().parents[2])
        report["checkpoint"] = identity
        report["input_fixture_sha256"] = sha256(args.input_fixture)
        report["arguments"] = {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        }
        report["uva"] = args.uva
        report["mapped_parameters"] = [
            n
            for n, v in routed.named_parameters()
            if getattr(v, "_vllm_is_uva_offloaded", False)
        ]
        torch.save(
            dict(
                x=x.cpu(), ids=ids.cpu(), weights=weights.cpu(), output=outputs[0].cpu()
            ),
            root / (args.name + ".pt"),
        )
        (root / (args.name + ".json")).write_text(json.dumps(report, indent=2))
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k not in ("weights", "post_weights", "artifacts")
                }
            )
        )
        graph.reset()
        del owners, outputs, warm, holder, routed, method, run
    finally:
        reset_workspace_manager()
        destroy_model_parallel()
        destroy_distributed_environment()
