"""Freeze workloads and retain reference/native outputs, timings, and diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path


REFERENCE_REVISION = "fb2764a5cf321eaa5070ca8f9e892818f477c16d"


def _load(path):
    return json.loads(Path(path).read_text())


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _reference_modules(root):
    sys.path[:0] = [str(root / "inference"), str(root / "encoding")]
    return (
        importlib.import_module("model"),
        importlib.import_module("generate"),
        importlib.import_module("encoding"),
        importlib.import_module("image_processor"),
    )


def _media_records(records, root):
    result = []
    for record in records:
        record = dict(record)
        url = record.get("url")
        if url and not str(url).startswith(("http:", "https:", "data:")):
            record["url"] = str((root / "inference" / url).resolve())
        result.append(record)
    return result


def freeze(args):
    import torch
    from transformers import AutoTokenizer

    root = Path(args.reference).resolve()
    if root.name != REFERENCE_REVISION:
        raise ValueError("freeze requires the pinned reference snapshot")
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    torch.set_default_dtype(torch.bfloat16)
    model, _, encoding, images = _reference_modules(root)
    config = model.ModelArgs(**_load(root / "inference/config.json"))
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    published = encoding.load_cases(str(root / "inference/examples/example_harmony.json"))
    long_text = "\n".join(f"Entry {i:04d}: the check value is blue." for i in range(2048))
    long_text += "\nWhat is the check value in the last entry? Reply with one word."
    cases = [
        {"id": "arithmetic", "messages": [{"role": "user", "content": "Compute 37 * 19. Reply with the integer only."}]},
        {"id": "python", "messages": [{"role": "user", "content": "Write a Python function that returns the maximum of a non-empty list. Return only code."}]},
        {"id": "chinese", **published[1]},
        {"id": "midturn_system", **published[3]},
        {"id": "tool_request", **published[2]},
        {"id": "two_images", **published[0]},
        {"id": "hierarchical_context", "messages": [{"role": "user", "content": long_text}]},
    ]
    frozen = []
    for case in cases:
        prompt, records = encoding.encode_case(case, "chat")
        records = _media_records(records, root)
        tokens, types, image_inputs = images.prepare_vl_inputs(prompt, records, tokenizer, config)
        if not tokens or len(tokens) + args.max_new_tokens > 32768:
            raise ValueError(f"case {case['id']} does not fit the qualification context: {len(tokens)}")
        if case["id"] == "hierarchical_context" and len(tokens) <= 16384:
            raise ValueError("hierarchical case must exceed the candidate-pool capacity")
        media = []
        for record, image in zip(records, image_inputs or (), strict=True):
            path = Path(record["url"])
            media.append({"path": str(path), "sha256": _sha(path),
                          "start": image.start, "vit_grid": [image.n_vit_h, image.n_vit_w]})
        frozen.append({"id": case["id"], "case": case, "encoded_prompt": prompt,
                       "prompt_token_ids": tokens, "token_types": types, "images": media})
    batches = [{"id": case["id"], "cases": [case["id"]]} for case in frozen]
    batches.append({"id": "mixed_text_batch", "cases": ["arithmetic", "python", "chinese"]})
    suite = {
        "schema_version": 1, "reference_revision": REFERENCE_REVISION,
        "reference_root": str(root), "reference_sources": {
            name: _sha(root / name) for name in (
                "inference/model.py", "inference/kernel.py", "inference/generate.py",
                "inference/vision.py", "inference/engram.py", "inference/image_processor.py",
                "encoding/encoding.py", "inference/config.json", "tokenizer.json", "tokenizer_config.json",
            )
        },
        "max_new_tokens": args.max_new_tokens, "seed": 33377335,
        "eos_token_id": tokenizer.eos_token_id,
        "temperature": 0.0, "thinking_mode": "chat", "cases": frozen, "batches": batches,
    }
    _save(args.output, suite)
    print(json.dumps({"suite": str(Path(args.output).resolve()), "sha256": _sha(args.output),
                      "prompt_lengths": {case["id"]: len(case["prompt_token_ids"]) for case in frozen}}))


def _checked_suite(path):
    suite = _load(path)
    root = Path(suite["reference_root"])
    if suite["reference_revision"] != REFERENCE_REVISION:
        raise ValueError("suite reference revision differs")
    for name, digest in suite["reference_sources"].items():
        if _sha(root / name) != digest:
            raise ValueError(f"reference source changed: {name}")
    for case in suite["cases"]:
        for image in case["images"]:
            if _sha(image["path"]) != image["sha256"]:
                raise ValueError(f"reference image changed: {image['path']}")
    return suite, root


def run_reference(args):
    import torch
    import torch.distributed as dist
    from safetensors.torch import load_model
    from transformers import AutoTokenizer

    suite, root = _checked_suite(args.suite)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    torch.set_num_threads(8)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(suite["seed"])
    reference, generate, _, images = _reference_modules(root)
    raw_config = _load(root / "inference/config.json")
    raw_config.update(max_batch_size=max(len(batch["cases"]) for batch in suite["batches"]),
                      max_seq_len=max(len(case["prompt_token_ids"]) for case in suite["cases"]) + suite["max_new_tokens"],
                      temperature=0.0)
    model_args = reference.ModelArgs(**raw_config)
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    with torch.device("cuda"):
        model = reference.Transformer(model_args, tokenizer)
    load_model(model, str(Path(args.checkpoint) / f"model{rank}-mp{world}.safetensors"))
    torch.set_default_device("cuda")
    cases = {case["id"]: case for case in suite["cases"]}
    result = {"schema_version": 1, "implementation": "published_reference",
              "suite_path": str(Path(args.suite).resolve()),
              "suite_sha256": _sha(args.suite), "checkpoint": str(Path(args.checkpoint).resolve()),
              "tensor_parallel_size": world, "completed": False, "batches": []}
    try:
        for batch in suite["batches"]:
            selected = [cases[name] for name in batch["cases"]]
            prepared = []
            for case in selected:
                records = [{"url": image["path"]} for image in case["images"]]
                tokens, types, media = images.prepare_vl_inputs(case["encoded_prompt"], records, tokenizer, model_args)
                if tokens != case["prompt_token_ids"] or types != case["token_types"]:
                    raise AssertionError(f"reference preprocessing changed for {case['id']}")
                prepared.append((tokens, types, media))
            have_images = any(item[2] for item in prepared)
            generated = generate.generate(
                model, [item[0] for item in prepared], suite["max_new_tokens"], tokenizer.eos_token_id,
                [item[1] for item in prepared] if have_images else None,
                [item[2] for item in prepared] if have_images else None,
            )
            record = {"id": batch["id"], "outputs": [
                {"case": case["id"], "prompt_token_ids": case["prompt_token_ids"],
                 "token_ids": output, "eos_terminated": len(output) < suite["max_new_tokens"],
                 "text": tokenizer.decode(output)}
                for case, output in zip(selected, generated, strict=True)
            ]}
            result["batches"].append(record)
            if rank == 0:
                _save(args.output, result)
                print(json.dumps({"completed_reference_batch": batch["id"],
                                  "generated_counts": [len(item) for item in generated]}), flush=True)
        result["completed"] = True
        if rank == 0:
            _save(args.output, result)
    finally:
        if world > 1:
            dist.destroy_process_group()


def run_vllm(args):
    from PIL import Image
    from vllm import LLM, SamplingParams

    suite, root = _checked_suite(args.suite)
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    engine = _load(args.engine)
    engine["model"] = str(root)
    llm = LLM(**engine)
    sampling = SamplingParams(temperature=0.0, max_tokens=suite["max_new_tokens"], seed=suite["seed"])
    cases = {case["id"]: case for case in suite["cases"]}
    result = {"schema_version": 1, "implementation": "vllm_b12x",
              "suite_path": str(Path(args.suite).resolve()),
              "suite_sha256": _sha(args.suite), "engine": engine,
              "repeats": args.repeats, "completed": False, "batches": []}
    for repeat in range(args.repeats):
        for batch in suite["batches"]:
            selected = [cases[name] for name in batch["cases"]]
            prompts = []
            for case in selected:
                if case["images"]:
                    media = [Image.open(image["path"]).convert("RGB") for image in case["images"]]
                    prompts.append({"prompt": case["encoded_prompt"], "multi_modal_data": {"image": media}})
                else:
                    prompts.append({"prompt_token_ids": case["prompt_token_ids"]})
            started = time.perf_counter()
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            elapsed = time.perf_counter() - started
            records = []
            for case, output in zip(selected, outputs, strict=True):
                if list(output.prompt_token_ids) != case["prompt_token_ids"]:
                    raise AssertionError(f"vLLM prompt tokens differ for {case['id']}")
                token_ids = list(output.outputs[0].token_ids)
                eos_terminated = suite["eos_token_id"] in token_ids
                if eos_terminated:
                    eos_position = token_ids.index(suite["eos_token_id"])
                    if eos_position != len(token_ids) - 1:
                        raise AssertionError("vLLM returned tokens after EOS")
                    token_ids = token_ids[:eos_position]
                records.append({"case": case["id"], "prompt_token_ids": list(output.prompt_token_ids),
                                "token_ids": token_ids, "eos_terminated": eos_terminated,
                                "text": output.outputs[0].text,
                                "finish_reason": output.outputs[0].finish_reason,
                                "stop_reason": output.outputs[0].stop_reason,
                                "metrics": None if output.metrics is None else asdict(output.metrics)})
            result["batches"].append({"id": batch["id"], "repeat": repeat,
                                      "elapsed_seconds": elapsed, "outputs": records})
            _save(args.output, result)
            print(json.dumps({"completed_vllm_batch": batch["id"], "repeat": repeat}), flush=True)
    if engine.get("speculative_config") is not None:
        result["speculative_metrics"] = [
            asdict(metric) for metric in llm.get_metrics()
            if "spec_decode" in metric.name
        ]
    result["completed"] = True
    _save(args.output, result)


def compare(args):
    reference, actual = _load(args.reference_output), _load(args.vllm_output)
    if not reference.get("completed") or not actual.get("completed"):
        _save(args.output, {"passed": False, "reason": "incomplete generation run"})
        raise SystemExit(1)
    if reference["suite_sha256"] != actual["suite_sha256"]:
        raise AssertionError("results used different frozen suites")
    suite, _ = _checked_suite(reference["suite_path"])
    if _sha(reference["suite_path"]) != reference["suite_sha256"]:
        raise AssertionError("frozen suite changed after generation")
    wanted = {
        (batch["id"], case)
        for batch in suite["batches"] for case in batch["cases"]
    }
    reference_rows = [
        ((batch["id"], output["case"]), output)
        for batch in reference["batches"] for output in batch["outputs"]
    ]
    expected = dict(reference_rows)
    repeats = actual.get("repeats")
    if (
        not wanted or set(expected) != wanted
        or len(reference_rows) != len(wanted)
        or type(repeats) is not int or repeats <= 0
    ):
        raise AssertionError("generation results do not cover the frozen suite")
    wanted_repeats = {(repeat, *key) for repeat in range(repeats) for key in wanted}
    seen = set()
    failures = []
    for batch in actual["batches"]:
        for output in batch["outputs"]:
            key = batch["id"], output["case"]
            repeated_key = batch.get("repeat"), *key
            correct = expected.get(key)
            mismatch = (
                repeated_key in seen or repeated_key not in wanted_repeats
                or correct is None
                or output["prompt_token_ids"] != correct["prompt_token_ids"]
                or output["token_ids"] != correct["token_ids"]
                or output["eos_terminated"] != correct["eos_terminated"]
            )
            seen.add(repeated_key)
            if mismatch:
                failures.append({
                    "batch": batch["id"], "case": output["case"],
                    "repeat": batch.get("repeat"),
                    "expected": None if correct is None else correct["token_ids"],
                    "actual": output["token_ids"],
                })
    missing = sorted(wanted_repeats - seen)
    if failures or missing:
        _save(args.output, {"passed": False, "failures": failures, "missing": missing})
        raise SystemExit(1)
    _save(args.output, {
        "passed": True, "cases_per_repeat": len(wanted), "repeats": repeats,
        "batches_checked": len(actual["batches"]),
        "suite_sha256": actual["suite_sha256"],
        "termination": "EOS excluded from IDs as in reference generate(); EOS termination compared separately",
    })
    print("Exact greedy token-ID parity passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("freeze")
    command.add_argument("--reference", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--max-new-tokens", type=int, default=32)
    command.set_defaults(run=freeze)
    command = commands.add_parser("reference")
    command.add_argument("--suite", required=True)
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(run=run_reference)
    command = commands.add_parser("vllm")
    command.add_argument("--suite", required=True)
    command.add_argument("--engine", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--repeats", type=int, default=2)
    command.set_defaults(run=run_vllm)
    command = commands.add_parser("compare")
    command.add_argument("--reference-output", required=True)
    command.add_argument("--vllm-output", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(run=compare)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
