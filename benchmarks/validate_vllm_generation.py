"""Validate configured vLLM generation, kernel selection and graph execution.

Run one engine configuration per process. Receipts retain exact token IDs and
checkpoint hashes for comparison with a separately executed reference arm.
This is a correctness smoke test, not a model accuracy or performance benchmark.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

from scripts._sm103_source import source_identity


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checkpoint_identity(model: str) -> dict:
    root = Path(model).resolve()
    if not root.is_dir():
        raise ValueError("Use a local checkpoint directory for reproducible validation")
    files = sorted(
        p
        for p in root.iterdir()
        if p.suffix in {".safetensors", ".json", ".jinja"} and p.is_file()
    )
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError(f"No safetensors checkpoint in {root}")
    return {
        "path": str(root),
        "files": {
            p.name: {"bytes": p.stat().st_size, "sha256": file_hash(p)} for p in files
        },
    }


def package_identity(name: str) -> dict:
    module = __import__(name)
    package = Path(module.__file__).resolve().parent
    files = sorted(package.rglob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(package)).encode() + b"\0")
        digest.update(bytes.fromhex(file_hash(path)))
    return {
        **source_identity(package.parent),
        "package_python_sha256": digest.hexdigest(),
        "native_files": {
            str(p.relative_to(package)): file_hash(p)
            for p in sorted(package.rglob("*.so"))
        },
    }


def gpu_snapshot() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--format=csv",
            "--query-gpu=index,uuid,name,compute_cap,compute_mode,pstate,memory.used,clocks.sm,clocks.mem,clocks_throttle_reasons.active",
        ],
        text=True,
    )


def cuda_compiler_identity() -> dict:
    root = os.environ.get("CUDA_HOME")
    if not root:
        return {"cuda_home": None, "nvcc": None}
    compiler = Path(root) / "bin" / "nvcc"
    return {
        "cuda_home": root,
        "nvcc": str(compiler),
        "nvcc_sha256": file_hash(compiler),
        "nvcc_version": subprocess.check_output(
            [str(compiler), "--version"], text=True
        ),
        "runtime_header_sha256": file_hash(
            Path(root) / "include" / "cuda_runtime_api.h"
        ),
    }


def install_replay_counter(manager):
    """Count successful replays while preserving the graph manager's result."""
    from collections import Counter

    if not hasattr(manager, "_generation_validation_replays"):
        counts = Counter()
        original = manager.run_fullgraph

        def counted(desc):
            result = original(desc)
            counts[str(desc)] += 1
            return result

        manager.run_fullgraph = counted
        manager._generation_validation_replays = counts


def inspect_worker(worker):
    """Retain backend provenance and count calls to real captured graph replay."""
    import torch

    runner = worker.model_runner
    speculator = getattr(runner, "speculator", None)
    result = {
        "runner": type(runner).__name__,
        "speculator": type(speculator).__name__,
        "models": {},
        "graphs": {},
        "nvfp4_comparisons": getattr(worker, "_nvfp4_comparisons", []),
    }
    for label, model in (
        ("target", runner.get_model()),
        ("draft", getattr(speculator, "model", None)),
    ):
        if model is None:
            continue
        routes = {}
        for name, module in model.named_modules():
            method = getattr(module, "quant_method", None)
            kernel = getattr(method, "kernel", None)
            gdn = getattr(module, "gdn_decode_kernel", None)
            if method is not None or gdn is not None:
                routes[name] = {
                    "method": type(method).__name__,
                    "kernel": type(kernel).__name__,
                    "gdn_decode": gdn,
                }
        result["models"][label] = {"class": type(model).__name__, "routes": routes}

    for label, owner in (("target", runner), ("draft", speculator)):
        for attr in ("cudagraph_manager", "query_cudagraph_manager"):
            manager = getattr(owner, attr, None)
            if manager is None:
                continue
            install_replay_counter(manager)
            result["graphs"][f"{label}.{attr}"] = {
                "mode": str(manager.cudagraph_mode),
                "captured": sorted(str(key) for key in manager.graphs),
                "replays": dict(manager._generation_validation_replays),
            }
    result["device"] = str(
        torch.cuda.get_device_properties(torch.cuda.current_device())
    )
    return result


class GenerationValidationWorkerExtension:
    """Expose inspection through vLLM's named worker RPC interface."""

    def inspect_generation_validation(self):
        return inspect_worker(self)

    def compare_generation_nvfp4_layers(self, names):
        """Compare selected eager calls using the same loaded weights and inputs."""
        import torch

        from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
            FlashInferCutlassNvFp4LinearKernel,
        )

        modules = dict(self.model_runner.get_model().named_modules())
        comparisons = self._nvfp4_comparisons = []
        for name in names:
            layer = modules[name]
            kernel = layer.quant_method.kernel
            if type(kernel).__name__ != "B12xNvFp4LinearKernel":
                raise ValueError(f"{name} does not use B12xNvFp4LinearKernel")
            # Both providers consume the same swizzled, K-aligned weight layout.
            # Padded checkpoint geometry needs a separate layout comparison.
            if layer.weight.shape[1] * 2 % 128 or layer.weight.shape[0] % 128:
                raise ValueError(f"{name} requires aligned NVFP4 diagnostic geometry")
            reference = FlashInferCutlassNvFp4LinearKernel(kernel.config)

            def install(name, kernel, reference):
                original = kernel.apply_weights
                seen = set()

                def compared(layer, x, bias=None):
                    actual = original(layer, x, bias)
                    rows = x.numel() // x.shape[-1]
                    if rows not in seen and len(seen) < 5:
                        seen.add(rows)
                        retained = actual.float().clone()
                        expected = reference.apply_weights(layer, x, bias).float()
                        if not bool(
                            torch.isfinite(retained).all()
                            and torch.isfinite(expected).all()
                        ):
                            raise AssertionError(f"{name} produced nonfinite values")
                        delta = retained - expected
                        denominator = expected.square().mean().sqrt()
                        if float(denominator) == 0:
                            raise AssertionError(f"{name} reference output is zero")
                        comparisons.append(
                            {
                                "layer": name,
                                "rows": rows,
                                "input_shape": list(x.shape),
                                "weight_shape": list(layer.weight.shape),
                                "finite": True,
                                "reference_rms": float(denominator),
                                "max_absolute_error": float(delta.abs().max()),
                                "relative_rms_error": float(
                                    delta.square().mean().sqrt()
                                    / denominator.clamp_min(1e-30)
                                ),
                                "cosine": float(
                                    torch.nn.functional.cosine_similarity(
                                        retained.flatten(), expected.flatten(), dim=0
                                    )
                                ),
                                "unequal_elements": int((retained != expected).sum()),
                                "elements": retained.numel(),
                            }
                        )
                    return actual

                kernel.apply_weights = compared

            install(name, kernel, reference)
        return list(names)


def compare_runs(reference: dict, actual: dict) -> list[dict]:
    """Reject incompatible receipts before comparing token-by-token outputs."""
    if reference["status"] != "passed":
        raise ValueError("Reference receipt did not pass its own validation")
    for key in (
        "target_checkpoint",
        "prompts",
        "sampling",
        "counts",
        "chat_template_kwargs",
    ):
        left, right = reference[key], actual[key]
        if key == "target_checkpoint":
            left, right = left["files"], right["files"]
        if left != right:
            raise ValueError(f"Reference and candidate differ in {key}")
    mismatches = []
    for run_id, (expected, observed) in enumerate(
        zip(reference["runs"], actual["runs"], strict=True)
    ):
        for request_id, (left, right) in enumerate(
            zip(expected, observed, strict=True)
        ):
            for field in ("prompt_token_ids", "token_ids", "finish_reason"):
                if left[field] != right[field]:
                    mismatches.append(
                        {
                            "run": run_id,
                            "request": request_id,
                            "field": field,
                            "expected": left[field],
                            "actual": right[field],
                        }
                    )
    return mismatches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine-config",
        type=Path,
        required=True,
        help="JSON object of LLM constructor arguments",
    )
    parser.add_argument(
        "--prompts", type=Path, required=True, help="JSON array of user prompt strings"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 4, 1])
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--require-kernel",
        action="append",
        default=[],
        help="Required kernel class in target routes; repeatable",
    )
    parser.add_argument("--require-speculator")
    parser.add_argument("--require-full-graphs", action="store_true")
    parser.add_argument("--require-prefix-cache", action="store_true")
    parser.add_argument("--require-repeat-equality", action="store_true")
    parser.add_argument(
        "--compare-nvfp4-layer",
        action="append",
        default=[],
        help="Eager diagnostic: compare this target layer with FlashInfer CUTLASS",
    )
    args = parser.parse_args()
    config = json.loads(args.engine_config.read_text())
    if args.compare_nvfp4_layer and not config.get("enforce_eager", False):
        parser.error("NVFP4 layer comparisons require enforce_eager=true")
    extension = (
        "benchmarks.validate_vllm_generation.GenerationValidationWorkerExtension"
    )
    if config.get("worker_extension_cls") not in (None, "", extension):
        parser.error("generation validation requires its own worker extension")
    config["worker_extension_cls"] = extension
    prompts = json.loads(args.prompts.read_text())
    if (
        not isinstance(prompts, list)
        or not prompts
        or not all(isinstance(p, str) and p for p in prompts)
    ):
        parser.error("prompts must be a nonempty array of nonempty strings")
    if not args.counts or min(args.counts) < 1 or max(args.counts) > len(prompts):
        parser.error("counts must be positive and no greater than the prompt count")
    if args.max_tokens < 1:
        parser.error("max-tokens must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write("{}\n")
    report = {
        "status": "starting",
        "command": [sys.executable, *sys.argv],
        "harness_sha256": file_hash(Path(__file__)),
        "engine_config": config,
        "prompts": prompts,
        "counts": args.counts,
        "sampling": {
            "temperature": 0,
            "seed": 41,
            "max_tokens": args.max_tokens,
            "logprobs": 5,
        },
        "chat_template_kwargs": {"enable_thinking": False},
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "CUDA_HOME",
                "CUTE_DSL_ARCH",
                "VLLM_USE_V2_MODEL_RUNNER",
                "VLLM_GDN_DECODE_KERNEL",
                "VLLM_PLUGINS",
                "HF_HUB_OFFLINE",
            )
        },
        "runs": [],
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    llm = None
    try:
        save()
        report["target_checkpoint"] = checkpoint_identity(config["model"])
        spec = config.get("speculative_config")
        if spec and spec.get("model"):
            report["draft_checkpoint"] = checkpoint_identity(spec["model"])
        report["sources"] = {name: package_identity(name) for name in ("b12x", "vllm")}
        report["versions"] = {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "nvidia-cutlass-dsl",
                "cuda-python",
                "triton",
                "flashinfer-python",
                "transformers",
            )
        }
        report["gpu_before"] = gpu_snapshot()
        report["cuda_compiler"] = cuda_compiler_identity()
        save()
        from vllm import LLM, SamplingParams

        llm = LLM(**config)
        if args.compare_nvfp4_layer:
            llm.collective_rpc(
                "compare_generation_nvfp4_layers", args=(args.compare_nvfp4_layer,)
            )
        report["workers_before"] = llm.collective_rpc("inspect_generation_validation")
        for worker in report["workers_before"]:
            kernels = {
                r["kernel"] for r in worker["models"]["target"]["routes"].values()
            }
            if missing := set(args.require_kernel) - kernels:
                raise AssertionError(
                    f"Required target kernels not selected: {sorted(missing)}"
                )
            if (
                args.require_speculator
                and worker["speculator"] != args.require_speculator
            ):
                raise AssertionError(
                    f"Required speculator not selected: {worker['speculator']}"
                )
        report["status"] = "loaded"
        save()
        for count in args.counts:
            responses = llm.chat(
                [[{"role": "user", "content": p}] for p in prompts[:count]],
                SamplingParams(**report["sampling"]),
                chat_template_kwargs=report["chat_template_kwargs"],
                use_tqdm=False,
            )
            outputs = []
            for response in responses:
                result = response.outputs[0]
                if (
                    not result.token_ids
                    or not result.text
                    or result.cumulative_logprob is None
                    or not math.isfinite(result.cumulative_logprob)
                ):
                    raise AssertionError(
                        "Generation returned empty or nonfinite output"
                    )
                outputs.append(
                    {
                        "prompt_token_ids": response.prompt_token_ids,
                        "token_ids": list(result.token_ids),
                        "text": result.text,
                        "finish_reason": result.finish_reason,
                        "cumulative_logprob": result.cumulative_logprob,
                        "num_cached_tokens": response.num_cached_tokens,
                    }
                )
            if len(outputs) != count:
                raise AssertionError("Generation did not return every request")
            report["runs"].append(outputs)
            save()
        report["workers_after"] = llm.collective_rpc("inspect_generation_validation")
        if args.compare_nvfp4_layer:
            for worker in report["workers_after"]:
                observed = {v["layer"] for v in worker["nvfp4_comparisons"]}
                if observed != set(args.compare_nvfp4_layer):
                    raise AssertionError("Not every requested NVFP4 layer executed")
        report["metrics"] = [
            dataclasses.asdict(m) for m in llm.get_metrics() if "spec_decode" in m.name
        ]
        report["gpu_after"] = gpu_snapshot()
        report["sources_after"] = {
            name: package_identity(name) for name in ("b12x", "vllm")
        }
        save()
        for name, before in report["sources"].items():
            after = report["sources_after"][name]
            for field in ("package_python_sha256", "native_files"):
                if before[field] != after[field]:
                    raise AssertionError(f"{name} {field} changed during generation")
        if report["harness_sha256"] != file_hash(Path(__file__)):
            raise AssertionError("Validation harness changed during generation")
        if args.require_prefix_cache and not any(
            (output["num_cached_tokens"] or 0) > 0
            for outputs in report["runs"]
            for output in outputs
        ):
            raise AssertionError("No request reused cached prompt tokens")
        if args.require_repeat_equality:
            preceding = {}
            repeats = 0
            for count, outputs in zip(args.counts, report["runs"], strict=True):
                tokens = [output["token_ids"] for output in outputs]
                if count in preceding:
                    repeats += 1
                    if tokens != preceding[count]:
                        raise AssertionError(
                            "Repeated request batch changed output tokens"
                        )
                preceding[count] = tokens
            if repeats == 0:
                raise AssertionError("No request batch was repeated")
        if spec:
            counters = {m["name"]: m.get("value", 0) for m in report["metrics"]}
            if counters.get("vllm:spec_decode_num_draft_tokens", 0) <= 0:
                raise AssertionError("Speculative decoding proposed no tokens")
        if args.require_full_graphs:
            for worker in report["workers_after"]:
                target_graphs = [
                    g
                    for name, g in worker["graphs"].items()
                    if name.startswith("target.")
                ]
                if not any(sum(g["replays"].values()) > 0 for g in target_graphs):
                    raise AssertionError(
                        "Target executed no recorded full graph replays"
                    )
        if args.reference:
            report["reference_sha256"] = file_hash(args.reference)
            report["mismatches"] = compare_runs(
                json.loads(args.reference.read_text()), report
            )
            if report["mismatches"]:
                raise AssertionError("Generation differs from the reference receipt")
        report["status"] = "passed"
    except BaseException:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        save()
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
