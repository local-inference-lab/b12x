"""Check inline MXFP4-CSF W4A8 scales on real checkpoint layers.

Reads complete expert layers and TP ranks through the vLLM MXFP4-CSF reader
(the serving loader), prepares the same weights three ways and compares them:

- native: the checkpoint's decoded E8M0 scales as uncompressed W4A8 weights;
- expand: CSF scales expanded into the shared scratch per call
  (B12X_W4A8_CSF_INLINE=0);
- inline: compact W4A8 kernels reading the inline storage (the default).

For every layer and rank it verifies that the inline storage decodes to the
native compact scale bytes and that MoE outputs of all three arms are equal
bit for bit, at each token count, from CUDA graph replay with the shared
expansion scratch poisoned. Requires one SM120 GPU and the vLLM integration.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def prepare_arms(weights, *, e, h, n, layout):
    from b12x._lib.quant.x4t_scales import decode_x4t_scales, make_x4t_scale_batch
    from b12x.moe import fused_moe as moe

    device = weights.w13.device
    plan = moe.plan_weights(
        source=moe.PackedSource(format="fp4_e8m0_k32", w13_layout=layout),
        activation=moe.ActivationSpec(mode="a8", nonlinearity="silu", io_dtype=torch.bfloat16),
        geometry=moe.MoEGeometry(num_experts=e, hidden_size=h, intermediate_size=n),
    )
    ids = torch.arange(e, dtype=torch.int32, device=device)
    planes, grids = [], []
    for source, rows, columns in (
        (weights.w13_scales, 2 * n, h // 32),
        (weights.w2_scales, h, n // 32),
    ):
        batch = make_x4t_scale_batch(
            source.fixed, source.exceptions, rows=rows, columns=columns, device=device
        )
        grid = torch.empty((e, rows, columns), dtype=torch.uint8, device=device)
        decode_x4t_scales(batch, ids, grid)
        planes.append(batch)
        grids.append(grid)
    one = torch.ones(e, device=device)
    arms = {
        "native": moe.prepare_weights(
            plan=plan,
            weights=moe.PackedWeights(
                w13=weights.w13.clone(), w2=weights.w2.clone(),
                w13_block_scales=grids[0].clone(), w2_block_scales=grids[1].clone(),
                w13_global_scales=one, w2_global_scales=one,
            ),
        )
    }
    scratch = (torch.empty_like(grids[0]), torch.empty_like(grids[1]))
    for name, inline in (("expand", "0"), ("inline", "1")):
        os.environ["B12X_W4A8_CSF_INLINE"] = inline
        arms[name] = moe.prepare_weights(
            plan=plan,
            weights=moe.Mxfp4CsfWeights(
                w13=weights.w13.clone(), w2=weights.w2.clone(),
                w13_scales=planes[0], w2_scales=planes[1],
                w13_scale_scratch=scratch[0], w2_scale_scratch=scratch[1],
            ),
        )
    os.environ.pop("B12X_W4A8_CSF_INLINE")
    return arms, scratch


def run_arms(arms, scratch, *, tokens, topk, h, seed):
    from b12x.moe import fused_moe as moe
    from b12x.preparation import PreparationSession, PreparedCall

    device = scratch[0].device
    e = arms["native"].num_experts
    generator = torch.Generator(device=device).manual_seed(seed)
    x = (torch.randn(tokens, h, device=device, generator=generator) * 0.5).to(torch.bfloat16)
    ids = torch.stack(
        [torch.randperm(e, device=device, generator=generator)[:topk] for _ in range(tokens)]
    ).to(torch.int32)
    weights = torch.softmax(
        torch.randn(tokens, topk, device=device, generator=generator), -1
    )
    plans = {
        name: moe.plan_execution(
            experts=experts,
            capacity=moe.ExecutionCapacity(max_tokens=tokens, top_k=topk),
            invocation={"fast_math": True},
            routing=moe.RoutingSpec(deterministic_output=True),
        )
        for name, experts in arms.items()
    }

    def prepare(state):
        buffers = tuple(
            torch.empty(s.shape, dtype=s.dtype, device=device)
            for s in state.scratch.scratch_specs()
        )
        output = torch.empty_like(x)
        binding = state.bind(a=x, topk_ids=ids, topk_weights=weights, output=output,
                             scratch=buffers, input_scales_static=True)
        return PreparedCall(run=lambda: state.run(binding), output=output,
                            owners=(buffers, binding))

    outputs, graphs, owners = {}, {}, []
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        session.prepare(tuple(
            p.request(name=f"csf-inline-check-{name}", prepare_call=prepare)
            for name, p in plans.items()
        ))
        for name, p in plans.items():
            buffers = tuple(
                torch.empty(s.shape, dtype=s.dtype, device=device) for s in p.scratch_specs()
            )
            outputs[name] = torch.empty_like(x)
            binding = moe.bind(p, a=x, topk_ids=ids, topk_weights=weights,
                               output=outputs[name], scratch=buffers, input_scales_static=True)
            owners.append((buffers, binding))
        session.freeze()
        for (name, _), (_, binding) in zip(plans.items(), owners, strict=True):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                moe.run(binding=binding)
            graphs[name] = graph
        # The inline arm never reads the scratch; the expansion arm rewrites
        # its routed experts before every call.
        for buffer in scratch:
            buffer.view(torch.uint8).fill_(0xFF)
        for name, graph in graphs.items():
            outputs[name].fill_(float("nan"))
            graph.replay()
        torch.cuda.synchronize()
        for graph in graphs.values():
            graph.reset()
    native = outputs["native"].view(torch.int16)
    return {
        "backend": owners[-1][1].execution_plan.decode_config.backend,
        "finite": bool(torch.isfinite(outputs["native"]).all()),
        "expand_equal": bool(torch.equal(native, outputs["expand"].view(torch.int16))),
        "inline_equal": bool(torch.equal(native, outputs["inline"].view(torch.int16))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[0])
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=384)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--intermediate-size", type=int, default=2304)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 64, 128])
    parser.add_argument("--w13-layout", choices=("w13", "w31"), default="w31")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from b12x._lib.quant.mxfp4_csf_inline import decode_mxfp4_csf_inline
    from vllm.model_executor.model_loader.mxfp4_csf_loader import read_mxfp4_csf_layer

    e, h = args.num_experts, args.hidden_size
    n = args.intermediate_size // args.tp
    device = torch.device("cuda")
    report = {"checkpoint": str(args.checkpoint), "tp": args.tp, "results": []}
    for layer in args.layers:
        for rank in args.ranks:
            reader_scratch = (
                torch.empty((e, h // 32, 2 * n), dtype=torch.uint8, device=device),
                torch.empty((e, n // 32, h), dtype=torch.uint8, device=device),
            )
            weights = read_mxfp4_csf_layer(
                args.checkpoint, layer, num_experts=e, hidden_size=h,
                intermediate_size=args.intermediate_size, tp_rank=rank, tp_size=args.tp,
                device=device, w13_scale_scratch=reader_scratch[0],
                w2_scale_scratch=reader_scratch[1],
            )
            arms, scratch = prepare_arms(weights, e=e, h=h, n=n, layout=args.w13_layout)
            del weights, reader_scratch
            planes = arms["inline"]._impl.mxfp4_csf_inline
            native = arms["native"]._impl.representation.value
            entry = {
                "layer": layer,
                "rank": rank,
                "planes_equal": [
                    bool(torch.equal(
                        decode_mxfp4_csf_inline(plane),
                        scales.view(torch.uint8).view(e, -1),
                    ))
                    for plane, scales in zip(planes, (native.w13_sfb, native.w2_sfb), strict=True)
                ],
                "raw_tiles": [plane.heavy_tiles for plane in planes],
                "inline_bytes": [plane.storage.numel() for plane in planes],
                "native_bytes": [int(native.w13_sfb.numel() * 4), int(native.w2_sfb.numel() * 4)],
                "tokens": {},
            }
            for tokens in args.tokens:
                entry["tokens"][tokens] = run_arms(
                    arms, scratch, tokens=tokens, topk=args.topk, h=h, seed=layer * 131 + tokens
                )
            print(json.dumps(entry), flush=True)
            report["results"].append(entry)
            del arms, scratch
            torch.cuda.empty_cache()
    ok = all(
        all(entry["planes_equal"])
        and all(
            t["finite"] and t["expand_equal"] and t["inline_equal"]
            for t in entry["tokens"].values()
        )
        for entry in report["results"]
    )
    report["ok"] = ok
    if args.output:
        args.output.write_text(json.dumps(report, indent=1))
    print("OK" if ok else "MISMATCH", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
