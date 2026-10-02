"""Compare CSF W4A8 experts with original DS4.1 tensors at TP4 rank zero."""

import json
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open

from b12x.moe import fused_moe as moe
from dataclasses import replace
from b12x.preparation import PreparationSession, PreparedCall

hub = Path("/root/.cache/huggingface/hub")
original = (
    hub
    / "models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277"
)
compressed = (
    hub
    / "models--local-inference-lab--DeepSeek-V4.1-Flash-MXFP4-CSF/snapshots/872da235166458bd6ffa9ee3f3c5c4771b63159c"
)
e, h, n, layer = 384, 5120, 576, 3
index = json.loads((original / "model.safetensors.index.json").read_text())[
    "weight_map"
]
weights = [
    torch.empty(e, 2 * n, h // 2, dtype=torch.uint8),
    torch.empty(e, h, n // 2, dtype=torch.uint8),
]
scales = [
    torch.empty(e, 2 * n, h // 32, dtype=torch.uint8),
    torch.empty(e, h, n // 32, dtype=torch.uint8),
]
with ExitStack() as stack:
    handles = {}
    for expert in range(e):
        for projection, offset in (("w1", 0), ("w3", n), ("w2", 0)):
            for suffix, arrays, divisor in (
                ("weight", weights, 2),
                ("scale", scales, 32),
            ):
                name = f"layers.{layer}.ffn.experts.{expert}.{projection}.{suffix}"
                filename = index[name]
                if filename not in handles:
                    handles[filename] = stack.enter_context(
                        safe_open(original / filename, framework="pt", device="cpu")
                    )
                value = handles[filename].get_slice(name)
                if projection == "w2":
                    arrays[1][expert].copy_(value[:, : n // divisor].view(torch.uint8))
                else:
                    arrays[0][expert, offset : offset + n].copy_(
                        value[:n, :].view(torch.uint8)
                    )
weights, scales = ([t.cuda() for t in items] for items in (weights, scales))
buffers = tuple(torch.empty_like(s) for s in scales)
one = torch.ones(e, device="cuda")
native = moe.PackedWeights(
    w13=weights[0],
    w2=weights[1],
    w13_block_scales=scales[0],
    w2_block_scales=scales[1],
    w13_global_scales=one,
    w2_global_scales=one,
)
weight_plan = moe.plan_weights(
    source=moe.PackedSource(format="fp4_e8m0_k32", w13_layout="w31"),
    activation=moe.ActivationSpec(
        mode="a8", nonlinearity="silu", io_dtype=torch.bfloat16, swiglu_limit=10.0
    ),
    geometry=moe.MoEGeometry(num_experts=e, hidden_size=h, intermediate_size=n),
)
compressed = replace(
    native,
    w13=native.w13.clone(),
    w2=native.w2.clone(),
    w13_block_scales=native.w13_block_scales.clone(),
    w2_block_scales=native.w2_block_scales.clone(),
)
compressed_plan = moe.plan_weights(
    source=weight_plan.source,
    activation=weight_plan.activation,
    geometry=weight_plan.geometry,
    constraints=moe.WeightPlanConstraints(scale_compression="csf"),
)
experts = [
    moe.prepare_weights(plan=weight_plan, weights=native),
    moe.prepare_weights(
        plan=compressed_plan, weights=compressed, scale_scratch=buffers
    ),
]
equal = {}
for attr in ("w1_fp4", "w2_fp4", "w1_blockscale", "w2_blockscale"):
    equal[attr] = torch.equal(
        getattr(experts[0]._impl, attr), getattr(experts[1]._impl, attr)
    )
assert all(equal.values()), equal
print(json.dumps(dict(prepared_tensors=equal)), flush=True)
cases = []
for tokens in (1, 17, 33, 128):
    torch.manual_seed(4132 + tokens)
    source = torch.randn(tokens, h, device="cuda", dtype=torch.bfloat16) * 0.1
    ids = torch.rand(tokens, e, device="cuda").topk(8, dim=1).indices.to(torch.int32)
    probabilities = torch.rand(tokens, 8, device="cuda").softmax(dim=1)
    plans = [
        moe.plan_execution(
            experts=o,
            capacity=moe.ExecutionCapacity(max_tokens=tokens, top_k=8),
            invocation={"fast_math": True},
            routing=moe.RoutingSpec(deterministic_output=True),
        )
        for o in experts
    ]

    def prepare(state):
        scratch = tuple(
            torch.empty(s.shape, dtype=s.dtype, device="cuda")
            for s in state.scratch.scratch_specs()
        )
        output = torch.empty_like(source)
        binding = state.bind(
            a=source,
            topk_ids=ids,
            topk_weights=probabilities,
            output=output,
            scratch=scratch,
            input_scales_static=True,
        )
        return PreparedCall(
            run=lambda: state.run(binding), output=output, owners=(scratch, binding)
        )

    with PreparationSession(
        device=source.device, autotune=False, compile_workers=0
    ) as session:
        session.prepare(
            tuple(
                p.request(name=f"ds41-real-{tokens}-{i}", prepare_call=prepare)
                for i, p in enumerate(plans)
            )
        )
        bindings, owners, outputs, graphs = [], [], [], []
        for p in plans:
            scratch = tuple(
                torch.empty(s.shape, dtype=s.dtype, device="cuda")
                for s in p.scratch_specs()
            )
            output = torch.empty_like(source)
            bindings.append(
                moe.bind(
                    p,
                    a=source,
                    topk_ids=ids,
                    topk_weights=probabilities,
                    output=output,
                    scratch=scratch,
                    input_scales_static=True,
                )
            )
            owners.append(scratch)
            outputs.append(output)
        session.freeze()
        for b in bindings:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                moe.run(binding=b)
            graphs.append(graph)
        for _ in range(3):
            ids.add_(1).remainder_(e)
            source.neg_()
            for buf in buffers:
                buf.fill_(0xFE)
            for out in outputs:
                out.fill_(float("nan"))
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            for graph in graphs:
                graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            assert torch.isfinite(outputs[0]).all() and torch.count_nonzero(outputs[0])
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        for graph in graphs:
            graph.reset()
        row = dict(
            tokens=tokens,
            exact=True,
            implementations=[b.implementation for b in bindings],
        )
        cases.append(row)
        print(json.dumps(row), flush=True)
Path("/results/ds41-real-online-mxfp8-parity.json").write_text(
    json.dumps(
        dict(
            original=str(original),
            scale_compression="csf",
            layer=layer,
            tp=4,
            rank=0,
            prepared=equal,
            cases=cases,
        ),
        indent=2,
    )
    + "\n"
)
