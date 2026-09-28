"""Compare checkpoint unquantized serving projections with prepared CuTe GEMV.

The serving arm uses cuBLAS for ordinary projections and the prepared vocabulary
API for logits. Q/K/V checkpoint tensors are concatenated in serving order.
Router weights and accumulation remain FP32; source values originate in BF16.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

import torch
from safetensors import safe_open

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.gemm import bf16_gemv, bf16_vocab_projection
from b12x.preparation import PreparationSession, PreparedCall, require_prepared
from benchmarks.benchmark_dense_gemm import bench_events
from benchmarks.checkpoint_dense import check_output, oracle
from benchmarks.common import make_l2_flush_fn, nvidia_smi_gpu_mode_snapshot


def checkpoint_cases(model):
    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    roles = {}
    for name in sorted(index):
        suffix = name.split(".mixer.")[-1]
        if (
            suffix
            not in (
                "fc1_latent_proj.weight",
                "fc2_latent_proj.weight",
                "gate.weight",
                "q_proj.weight",
                "o_proj.weight",
            )
            and name != "lm_head.weight"
        ):
            continue
        role = re.sub(r"layers\.\d+", "layers.*", name)
        roles.setdefault(role, []).append(name)
    cases = []
    for role, names in sorted(roles.items()):
        name = names[0]
        payloads = [name]
        if name.endswith("q_proj.weight"):
            payloads = [
                name.replace("q_proj.weight", p + "_proj.weight")
                for p in ("q", "k", "v")
            ]
            role = role.replace("q_proj.weight", "qkv_proj.weight")
        shapes = []
        for payload in payloads:
            with safe_open(model / index[payload], framework="pt", device="cpu") as f:
                shapes.append(f.get_slice(payload).get_shape())
        if any(len(s) != 2 or s[1] != shapes[0][1] for s in shapes):
            raise ValueError(f"incompatible projection payloads: {payloads}")
        cases.append(
            dict(
                role=role,
                payloads=payloads,
                equivalent_weights=names,
                n=sum(s[0] for s in shapes),
                k=shapes[0][1],
                router=name.endswith("gate.weight"),
                vocab=name == "lm_head.weight",
            )
        )
    if len(cases) != 6:
        raise ValueError(
            f"expected six unquantized Puzzle 3 projection roles, found {len(cases)}"
        )
    return index, cases


def qualify(output, expected, router):
    result = check_output(output, expected)
    if router:
        torch.testing.assert_close(output, expected, rtol=2e-4, atol=3e-5)
        for count in (4, 8, 10, 12, 14, 16, 18):
            if not torch.equal(
                output.topk(count).indices, expected.topk(count).indices
            ):
                raise AssertionError(f"router top-{count} changed")
        result["topk_equal"] = True
    return result


def run(args):
    if args.cache != "cold" or args.launches != 1 or args.warmup < 20:
        raise ValueError(
            "checkpoint projections require cold L2, one launch and at least 20 warmups"
        )
    if args.repeats * args.replays < 100:
        raise ValueError(
            "checkpoint projections require at least 100 paired samples per arm"
        )
    if args.layouts != ["contiguous"] or args.output_dtypes != ["bf16"]:
        raise ValueError(
            "checkpoint projections retain serving layouts and numerical types"
        )
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model = args.model_path.resolve()
    index, cases = checkpoint_cases(model)
    flush = make_l2_flush_fn(enabled=True, bytes_hint=args.l2_flush_bytes)
    root = Path(__file__).resolve().parents[1]
    paths = [
        *sorted((root / "b12x/gemm/bf16_gemv").glob("*.py")),
        *sorted((root / "b12x/gemm/bf16_vocab_projection").glob("*.py")),
        Path(__file__).resolve(),
        root / "benchmarks/benchmark_bf16_projection.py",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as evidence:

        def record(value):
            evidence.write(json.dumps(value) + "\n")
            evidence.flush()

        record(
            dict(
                kind="manifest",
                command=[sys.executable, *sys.argv],
                model=str(model),
                cases=cases,
                counts=args.rows,
                torch=torch.__version__,
                source_sha256={
                    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in paths
                },
                device=nvidia_smi_gpu_mode_snapshot(),
                metric="cold-L2 graph replay microseconds; candidate/serving below one is better",
            )
        )
        for case in cases:
            payloads, hashes = [], {}
            for name in case["payloads"]:
                with safe_open(model / index[name], framework="pt", device="cpu") as f:
                    raw = f.get_tensor(name)
                hashes[name] = hashlib.sha256(
                    raw.view(torch.uint8).numpy().tobytes()
                ).hexdigest()
                payloads.append(raw)
            weight = torch.cat(payloads).to(
                device="cuda", dtype=torch.float32 if case["router"] else torch.bfloat16
            )
            del payloads, raw
            for m in args.rows:
                torch.manual_seed(args.seed + m)
                source = (
                    torch.randn((m, case["k"]), device="cuda", dtype=torch.bfloat16)
                    * 0.25
                )
                expected = oracle(source, weight, 1.0)
                source_values = source.clone()
                query = bf16_gemv.query_from_call(
                    source, weight, output_dtype=weight.dtype
                )
                native = bf16_gemv.plan(query)

                def prepare(state):
                    return PreparedCall(
                        run=lambda: state.run(source, weight),
                        produce=lambda: source.copy_(source_values),
                        owners=(weight,),
                    )

                requests = [
                    native.request(
                        name="native", prepare_call=prepare, benchmark_call=prepare
                    )
                ]
                vocab = None
                if case["vocab"]:
                    vocab = bf16_vocab_projection.plan(
                        bf16_vocab_projection.Caps(
                            device=source.device,
                            max_tokens=m,
                            in_features=case["k"],
                            out_features=case["n"],
                        )
                    )
                    requests.append(
                        vocab.request(
                            name="serving_vocab",
                            prepare_call=prepare,
                            benchmark_call=prepare,
                        )
                    )
                with PreparationSession(
                    device=source.device, autotune=True, compile_workers=1
                ) as session:
                    session.prepare(tuple(requests))
                    session.freeze()
                    config = asdict(native.selection.config)
                    native_state = require_prepared(
                        native, "gemm.bf16_gemv", source.device
                    )
                    callable_before = native_state.launcher
                    arms = dict(
                        native=lambda: bf16_gemv.mm(source, weight, plan=native)
                    )
                    if vocab is not None:
                        binding = bf16_vocab_projection.bind(
                            vocab, source=source, weight=weight
                        )
                        arms["serving"] = lambda: bf16_vocab_projection.run(binding)
                        serving_config = asdict(vocab.selection.config)
                    else:
                        arms["serving"] = lambda: torch.nn.functional.linear(
                            source.float() if case["router"] else source, weight
                        )
                        serving_config = {
                            "backend": "torch",
                            "input_cast": "float32" if case["router"] else None,
                        }
                    graphs, outputs, checks = {}, {}, {}
                    with kernel_resolution_guard("checkpoint projection graph replay"):
                        for name, launch in arms.items():
                            checks[name] = qualify(launch(), expected, case["router"])
                            for _ in range(args.warmup):
                                launch()
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                output = launch()
                            graphs[name], outputs[name] = graph, output
                        source.neg_()
                        addresses = {
                            name: value.data_ptr() for name, value in outputs.items()
                        }
                        for value in outputs.values():
                            value.fill_(float("nan"))
                        allocated = torch.cuda.memory_allocated()
                        for name, graph in graphs.items():
                            graph.replay()
                            torch.cuda.synchronize()
                            if torch.cuda.memory_allocated() != allocated:
                                raise AssertionError("graph replay changed allocation")
                            qualify(outputs[name], -expected, case["router"])
                        samples = {name: [] for name in arms}
                        before = nvidia_smi_gpu_mode_snapshot()
                        for repeat in range(args.repeats):
                            order = (
                                ("serving", "native")
                                if repeat % 2 == 0
                                else ("native", "serving")
                            )
                            for name in order:
                                samples[name].extend(
                                    t * 1000
                                    for t in bench_events(
                                        graphs[name].replay,
                                        warmup=args.warmup,
                                        iters=args.replays,
                                        l2_flush=flush,
                                    )
                                )
                        after = nvidia_smi_gpu_mode_snapshot()
                        for name, output in outputs.items():
                            qualify(output, -expected, case["router"])
                            assert output.data_ptr() == addresses[name]
                        assert native_state.launcher is callable_before
                    medians = {
                        name: statistics.median(values)
                        for name, values in samples.items()
                    }
                    ratio = medians["native"] / medians["serving"]
                    record(
                        dict(
                            kind="case",
                            **case,
                            rows=m,
                            payload_sha256=hashes,
                            native_config=config,
                            serving_config=serving_config,
                            query=asdict(query),
                            correctness=checks,
                            changed_input_replay=True,
                            replay_allocation_delta=0,
                            stable_addresses=True,
                            stable_callable=True,
                            samples_us=samples,
                            median_us=medians,
                            native_over_serving=ratio,
                            device_before=before,
                            device_after=after,
                        )
                    )
                    print(
                        f"{case['role']} M={m}: serving={medians['serving']:.3f} us "
                        f"native={medians['native']:.3f} us ratio={ratio:.4f} config={config}",
                        flush=True,
                    )
                    del graphs, outputs, graph, output
        record(
            dict(kind="result", cases=len(cases) * len(args.rows), correctness="passed")
        )
