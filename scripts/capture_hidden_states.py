#!/usr/bin/env python
"""Capture real layer-input hidden states for the activation ablations.

Runs the BF16 model with HF transformers on real text, hooks the chosen
layer's MLP pre-forward input — i.e. exactly the tensor the FP6 kernel
quantizes at FC1 — and saves it as a ``(tokens, K)`` bf16 **safetensors**
file (key ``hidden_states``; pickle ``.pt`` outputs are rejected).

Feed the result to scripts/ablate_moe_act_quant.py or
scripts/ablate_hadamard_fp6.py via ``--x-from`` so the ablation sees real
outlier-channel structure instead of gaussian noise.

Example (Qwen3.6 ships custom modeling code, hence --trust-remote-code):
    python scripts/capture_hidden_states.py \
        --model /media/fmodels/Qwen/Qwen3.6-35B-A3B --layer 20 \
        --trust-remote-code \
        --out /tmp/l20_hidden.safetensors
    python scripts/ablate_moe_act_quant.py \
        --model /media/fmodels/Qwen/Qwen3.6-35B-A3B --layer 20 \
        --x-from /tmp/l20_hidden.safetensors
"""
from __future__ import annotations

import argparse
import pathlib

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="BF16 model directory")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument(
        "--out", required=True, help="output *.safetensors path (pickle rejected)"
    )
    parser.add_argument(
        "--text-file",
        default=None,
        help="UTF-8 text to tokenize; default: --dataset (wikitext, same "
        "source as the KLD script)",
    )
    parser.add_argument(
        "--dataset", default="wikitext", help="HF dataset (KLD script default)"
    )
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--tokens", type=int, default=4096, help="tokens to capture")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="allow the checkpoint's custom modeling code to run (required "
        "for e.g. Qwen3.6; off by default so untrusted repos never execute "
        "code without an explicit opt-in)",
    )
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    if args.text_file:
        text = pathlib.Path(args.text_file).read_text(encoding="utf-8")
    else:
        # Same source + assembly as score_mode_kld.py: non-empty rows of the
        # dataset's "text" column joined with blank lines (test split first).
        from datasets import load_dataset

        dataset = None
        for split in ("test", "train", "validation"):
            try:
                dataset = load_dataset(args.dataset, args.dataset_config, split=split)
                break
            except Exception:
                continue
        if dataset is None:
            raise SystemExit(f"could not load dataset {args.dataset}")
        text = "\n\n".join(
            ex["text"] for ex in dataset if ex.get("text", "").strip()
        )
        print(f"[capture] dataset {args.dataset}/{args.dataset_config} split={split}")
    text = text[: args.tokens * 8]  # ~8 chars/token upper bound, keep tokenize cheap
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    if ids.numel() < args.tokens:
        print(
            f"[capture] text has only {ids.numel()} tokens "
            f"(< {args.tokens}); capturing what's there"
        )
    ids = ids[: args.tokens]

    try:
        import accelerate  # noqa: F401

        extra = {"device_map": "auto"}
        move_after = False
    except ImportError:
        # device_map requires accelerate; load on CPU and move instead.
        extra = {}
        move_after = torch.cuda.is_available()
    print(f"[capture] loading model (bf16, {extra or 'cpu->cuda'}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
        **extra,
    )
    if move_after:
        model = model.to("cuda")
    model.eval()

    # The MoE block is the module named "...layers.{L}.mlp" that owns experts.
    suffix = f"layers.{args.layer}.mlp"
    target = None
    for name, mod in model.named_modules():
        if name.endswith(suffix) and hasattr(mod, "experts"):
            target = (name, mod)
            break
    if target is None:
        for name, mod in model.named_modules():
            if name.endswith(suffix):
                target = (name, mod)
                break
    if target is None:
        raise SystemExit(f"no module found ending with '{suffix}'")
    print(f"[capture] hooking {target[0]}")

    captured: list[torch.Tensor] = []

    def hook(_mod, hook_args):
        h = hook_args[0]
        captured.append(h.detach().reshape(-1, h.shape[-1]).to("cpu", torch.bfloat16))

    handle = target[1].register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            for start in range(0, ids.numel(), args.seq_len):
                chunk = ids[start : start + args.seq_len]
                if chunk.numel() == 0:
                    break
                model(chunk.unsqueeze(0).to(model.device))
                print(
                    f"[capture] {min(start + args.seq_len, ids.numel())}"
                    f"/{ids.numel()} tokens"
                )
    finally:
        handle.remove()

    from sparkinfer.quantization.mxfp6.calib_io import save_hidden_states

    x = torch.cat(captured, dim=0)
    out = save_hidden_states(
        x,
        args.out,
        metadata={
            "model": str(args.model),
            "layer": str(args.layer),
            "tokens": str(x.shape[0]),
        },
    )
    print(f"[capture] saved {tuple(x.shape)} bf16 hidden states -> {out}")


if __name__ == "__main__":
    main()
