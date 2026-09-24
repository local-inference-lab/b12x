"""Qualify safetensors IQ2_XS shared-expert weights through the dense API.

Run from the repository root with --model-path DIR --evidence FILE.jsonl.
This checks numerical and graph contracts; it does not measure performance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.gemm import blockscaled
from b12x.preparation import PreparationSession, PreparedCall, require_prepared
from b12x.testing.iq2_xs_reference import dequantize_blocks


def check(actual, reference):
    actual = actual.float()
    if not torch.isfinite(actual).all() or not torch.count_nonzero(actual):
        raise AssertionError("dense output must be finite and nonzero")
    delta = actual - reference
    relative = float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference))
    cosine = float(torch.nn.functional.cosine_similarity(actual.flatten(), reference.flatten(), dim=0))
    if relative > 0.004 or cosine < 0.99998:
        raise AssertionError(f"relative_l2={relative}, cosine={cosine}")
    return dict(relative_l2=relative, cosine=cosine, max_abs=float(delta.abs().max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 3, 8, 17, 128, 512])
    args = parser.parse_args()
    if not args.rows or min(args.rows) < 1:
        parser.error("rows must be positive")
    torch.cuda.set_device(args.device)
    torch.set_num_threads(4)
    device = torch.device("cuda", args.device)
    if torch.cuda.get_device_capability(device) not in ((12, 0), (12, 1)):
        raise RuntimeError("IQ2_XS dense qualification requires SM120/SM121")
    model = args.model_path.resolve()
    index_path = model / "model.safetensors.index.json"
    quant_path = model / "hf_quant_config.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    recipes = json.loads(quant_path.read_text())["quantization"]["quantized_layers"]
    names = sorted(name for name in weight_map if ".shared_experts." in name
                   and name.endswith(".weight")
                   and recipes.get(name.removesuffix(".weight"), {}).get("quant_algo") == "IQ2_XS")
    if not names:
        raise ValueError("checkpoint contains no IQ2_XS shared-expert weights")
    root = Path(__file__).resolve().parents[2]
    paths = [*sorted((root / "b12x/gemm/blockscaled").glob("*.py")),
             root / "b12x/_lib/dense_gemm.py", root / "b12x/_lib/intrinsics.py",
             root / "b12x/_lib/quant/iq2_xs.py", root / "b12x/testing/iq2_xs_reference.py",
             Path(__file__).resolve()]
    props = torch.cuda.get_device_properties(device)
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    with args.evidence.open("x") as evidence:
        def record(value):
            evidence.write(json.dumps(value) + "\n")
            evidence.flush()

        record(dict(kind="manifest", command=sys.argv, model=str(model), weight_count=len(names),
                    gpu=props.name, uuid=str(props.uuid), capability=torch.cuda.get_device_capability(device),
                    torch=torch.__version__, cutlass=importlib.metadata.version("nvidia-cutlass-dsl"),
                    source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                    index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
                    quant_sha256=hashlib.sha256(quant_path.read_bytes()).hexdigest(),
                    performance_measured=False))
        for position, name in enumerate(names):
            recipe = recipes[name.removesuffix(".weight")]
            if recipe.get("group_size") != 256 or recipe.get("block_payload_bytes") != 74 or recipe.get("packing") != "ggml":
                raise ValueError(f"unsupported IQ2_XS payload metadata: {name}")
            with safe_open(model / weight_map[name], framework="pt", device="cpu") as handle:
                raw = handle.get_tensor(name)
            payload_hash = hashlib.sha256(raw.numpy().tobytes()).hexdigest()
            weight = blockscaled.pack_weight(raw.to(device), recipe="iq2_xs")
            reference_weight = dequantize_blocks(raw).bfloat16().to(device).float()
            torch.manual_seed(position)
            source = torch.randn((max(args.rows), weight.in_features), device=device, dtype=torch.bfloat16)
            output = torch.empty((max(args.rows), weight.out_features), device=device, dtype=torch.bfloat16)
            query = blockscaled.query_from_call(source, weight, out=output)
            plan = blockscaled.plan(query)

            def prepare_call(state):
                return PreparedCall(run=lambda: state.run(source, weight.values, weight.metadata, None, out=output))

            with PreparationSession(device=device, autotune=False, compile_workers=1) as session:
                session.prepare((plan.request(name=name, prepare_call=prepare_call),))
                session.freeze()
                state = require_prepared(plan, "gemm.blockscaled_precision", device)
                native = state.programs["gemm"]
                addresses = (weight.values.data_ptr(), weight.metadata.data_ptr(), source.data_ptr(), output.data_ptr(),
                             None if state.workspace is None else state.workspace.data_ptr())
                with kernel_resolution_guard("IQ2_XS checkpoint qualification"):
                    for m in args.rows:
                        x, out = source[:m], output[:m]
                        reference = x.float() @ reference_weight.T
                        blockscaled.mm(x, weight, out=out, plan=plan)
                        eager = check(out, reference)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            blockscaled.mm(x, weight, out=out, plan=plan)
                        x.neg_()
                        out.fill_(float("nan"))
                        if state.workspace is not None:
                            state.workspace.fill_(255)
                        allocated = torch.cuda.memory_allocated(device)
                        graph.replay()
                        torch.cuda.synchronize(device)
                        if torch.cuda.memory_allocated(device) != allocated:
                            raise AssertionError("CUDA graph replay changed live allocation")
                        replay = check(out, -reference)
                        assert state.programs["gemm"] is native
                        assert addresses == (weight.values.data_ptr(), weight.metadata.data_ptr(), source.data_ptr(), output.data_ptr(),
                                             None if state.workspace is None else state.workspace.data_ptr())
                        record(dict(kind="case", weight=name, shard=weight_map[name], payload_sha256=payload_hash,
                                    n=weight.out_features, k=weight.in_features, rows=m,
                                    config=state.config.to_dict(), eager=eager, changed_input_replay=replay,
                                    stable_addresses=True, stable_callable=True, replay_allocation_delta=0))
                        del graph
            print(f"{position + 1}/{len(names)} {name}: passed {len(args.rows)} live row counts", flush=True)
        record(dict(kind="result", status="qualified", weights=len(names), cases=len(names) * len(args.rows),
                    contract="BF16 dense GEMM correctness and changed-input CUDA graph replay"))


if __name__ == "__main__":
    main()
