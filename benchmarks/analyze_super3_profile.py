"""Attribute Super3 serving traces using CUDA correlations and compile manifests.

Dense CuTe symbols omit the codec. Match their symbol, shared memory, split-K,
and N grid to immutable compile facts instead of guessing from kernel names.
This analyzer targets the validated Super3 checkpoint and b12x gemm.dense v4.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics


def dense_inventory(cache):
    rows = []
    for path in sorted(cache.glob("*/*.json")):
        try:
            manifest = json.loads(path.read_text())
        except PermissionError:
            continue
        if manifest.get("target") != "b12x._lib.dense_gemm._DenseGemmLaunch":
            continue
        spec = json.loads(manifest["compile_spec_json"])
        if spec["kernel"] != "gemm.dense" or spec["version"] != 4:
            continue
        facts = [field[2] for field in spec["facts"][3]]
        if len(facts) < 35 or facts[32] not in ("nvfp4", "iq2_xxs", "q8_0"):
            continue
        policy = dict(facts[16][3])
        for symbol, smem in (
            manifest.get("launch_metadata", {})
            .get("launch_dynamic_smem_bytes", {})
            .items()
        ):
            rows.append(
                dict(
                    path=str(path),
                    manifest=manifest,
                    symbol=symbol,
                    smem=smem,
                    n=facts[0],
                    k=facts[1],
                    codec=facts[32],
                    tile=facts[14],
                    split=policy["split_k_slices"],
                )
            )
    return rows


def dense_component(event, inventory, matched):
    args = event["args"]
    candidates = [
        row
        for row in inventory
        if row["symbol"] == event["name"]
        and args["shared memory"] in row["smem"]
        and args["grid"][1] == row["split"]
        and args["grid"][2] == (row["n"] + row["tile"][1] - 1) // row["tile"][1]
    ]
    identities = {(r["codec"], r["n"], r["k"]) for r in candidates}
    if len(identities) != 1:
        raise RuntimeError(f"Ambiguous dense attribution: {event}, {identities}")
    codec, n, k = identities.pop()
    for row in candidates:
        matched[row["path"]] = row["manifest"]
    if codec == "nvfp4":
        assert (n, k) in ((18560, 4096), (4096, 8192))
        return "NVFP4 Mamba projections"
    if codec == "iq2_xxs":
        assert (n, k) in ((6144, 4096), (4096, 6144))
        return "IQ2_XXS shared experts"
    assert (n, k) in ((131072, 4096), (1024, 4096), (4096, 1024))
    return "Q8_0 LM head" if n == 131072 else "Q8_0 latent projections"


def classify(event, inventory, matched, last_dense, vision_end):
    name = event["name"]
    args = event["args"]
    if vision_end is not None and event["ts"] < vision_end:
        return "Vision encoder/projector and input preparation"
    if "DenseGemmKernel" in name:
        component = dense_component(event, inventory, matched)
        last_dense[args["stream"]] = component
        return component
    if "WeightOnlySplitKReduce" in name:
        return last_dense.pop(args["stream"])
    if "sequenceembedding" in name:
        return "Q8_0 embedding"
    if "W4A16FusedMoeKernel" in name or "W4A16GemmKernel" in name:
        return "IQ2_XXS routed experts"
    if (
        "W4A16TopKSum" in name
        or "pack_topk_routes" in name
        or "single_group_topk" in name
        or "_w4a16_route_" in name
    ):
        return "MoE routing and route reduction"
    if any(
        s in name
        for s in (
            "_selective_scan",
            "_causal_conv",
            "_chunk_",
            "_state_passing",
            "_bmm_chunk",
        )
    ):
        return "Mamba convolution and scan"
    if any(
        s in name
        for s in (
            "b12xattention",
            "flash_fwd",
            "reshape_and_cache",
            "update_regular_decode_graph_metadata",
        )
    ):
        return "Attention and KV cache"
    if "internal::gemvx" in name:
        return (
            "BF16 latent/attention projections"
            if "__nv_bfloat16" in name
            else "FP32 router projections"
        )
    if "sgemm" in name or "gemmSN_TN_kernel<float," in name:
        return "FP32 router projections"
    if "cublasLt::splitKreduce_kernel" in name:
        return (
            "BF16 latent/attention projections"
            if "__nv_bfloat16" in name
            else "FP32 router projections"
        )
    if any(s in name for s in ("nvjet_", "gemm_bf16", "gemm_relu_bf16")):
        return "BF16 latent/attention projections"
    if any(
        s in name
        for s in ("norm", "activation_kernel", "fused_mul", "fused__to_copy_add")
    ):
        return "Norms, activations and elementwise fusion"
    return "Other GPU work"


def union_us(intervals):
    end = float("-inf")
    total = 0.0
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def summarize(path, inventory, matched):
    raw = path.read_bytes()
    document = json.loads(gzip.decompress(raw))
    events = document["traceEvents"]
    steps = sorted(
        [
            e
            for e in events
            if e.get("cat") == "user_annotation"
            and e["name"].startswith("execute_context_")
        ],
        key=lambda e: e["ts"],
    )
    starts = [s["ts"] for s in steps]
    runtime = {
        e["args"]["correlation"]: e
        for e in events
        if e.get("cat") in ("cuda_runtime", "cuda_driver")
        and "correlation" in e.get("args", {})
    }
    kernels = sorted(
        [e for e in events if e.get("cat") == "kernel"], key=lambda e: e["ts"]
    )
    vision_end = (
        next(e["ts"] for e in kernels if "sequenceembedding" in e["name"])
        if path.name.startswith("vision")
        else None
    )
    last_dense = {}
    per_step = defaultdict(list)
    for event in kernels:
        launch = runtime[event["args"]["correlation"]]
        step = bisect_right(starts, launch["ts"]) - 1
        assert step >= 0, event
        component = classify(event, inventory, matched, last_dense, vision_end)
        per_step[step].append((event, component))
    groups = defaultdict(list)
    decode_counts = Counter()
    step_rows = []
    for index, step in enumerate(steps):
        match = re.fullmatch(
            r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)", step["name"]
        )
        assert match, step["name"]
        contexts, context_tokens, requests, generation_tokens = map(int, match.groups())
        if contexts and requests:
            phase = "mixed_prefill_decode"
        elif contexts:
            phase = "prefill"
        elif requests:
            phase = f"decode_b{requests}"
        else:
            phase = "empty_scheduler_step"
        steady = True
        if requests and not contexts:
            decode_counts[phase] += 1
            steady = decode_counts[phase] > 8
        row = dict(
            index=index,
            annotation=step["name"],
            phase=phase,
            context_tokens=context_tokens,
            generation_tokens=generation_tokens,
            steady=steady,
            kernels=len(per_step[index]),
            device_us=sum(e["dur"] for e, _ in per_step[index]),
        )
        step_rows.append(row)
        if per_step[index] and (steady or phase == "prefill"):
            groups[phase].append(index)
    summaries = {}
    for phase, indices in groups.items():
        components = defaultdict(lambda: dict(calls=0, device_us=0.0))
        by_kernel = defaultdict(lambda: dict(calls=0, device_us=0.0))
        selected = [pair for index in indices for pair in per_step[index]]
        for event, component in selected:
            for entry in (components[component], by_kernel[(component, event["name"])]):
                entry["calls"] += 1
                entry["device_us"] += event["dur"]
        total = sum(row["device_us"] for row in components.values())
        summaries[phase] = dict(
            steps=len(indices),
            device_us=total,
            context_tokens=sum(step_rows[i]["context_tokens"] for i in indices),
            generation_tokens=sum(step_rows[i]["generation_tokens"] for i in indices),
            graph_kernel_instances=sum(
                bool(e["args"].get("graph id")) for e, _ in selected
            ),
            gpu_busy_union_us=union_us(
                (e["ts"], e["ts"] + e["dur"]) for e, _ in selected
            ),
            components=[
                dict(
                    component=name,
                    **row,
                    percent=100 * row["device_us"] / total,
                    ms_per_step=row["device_us"] / 1000 / len(indices),
                )
                for name, row in sorted(
                    components.items(), key=lambda pair: -pair[1]["device_us"]
                )
            ],
            kernels=[
                dict(component=key[0], name=key[1], **row)
                for key, row in sorted(
                    by_kernel.items(), key=lambda pair: -pair[1]["device_us"]
                )
            ],
        )
    annotated_path = path.parent.parent / "traces_annotated" / path.name
    annotated_path.parent.mkdir(exist_ok=True)
    for pairs in per_step.values():
        for event, component in pairs:
            event["args"]["original kernel name"] = event["name"]
            event["args"]["component"] = component
            event["name"] = f"[{component}] {event['name']}"
    annotated_path.write_bytes(gzip.compress(json.dumps(document).encode(), mtime=0))
    return dict(
        path=str(path),
        annotated_path=str(annotated_path),
        sha256=hashlib.sha256(raw).hexdigest(),
        kernel_instances=len(kernels),
        cuda_graph_launches=sum("GraphLaunch" in e["name"] for e in runtime.values()),
        attribution="CUDA correlation -> CPU launch -> scheduler iteration; "
        "dense codec from compile facts plus launch signature",
        decode_excluded_initial_steps=8,
        steps=step_rows,
        phases=summaries,
    )


def plot(out, summaries):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [
        summaries["decode_b1"]["phases"]["decode_b1"],
        summaries["prefill_2048"]["phases"]["prefill"],
    ]
    names = [row["component"] for row in selected[0]["components"][:8]]
    names.append("Remaining components")
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6), sharey=True, constrained_layout=True
    )
    for ax, data, title, divisor in zip(
        axes,
        selected,
        ["Decode: batch 1, per token", "Prefill: 2,048 tokens, whole request"],
        [selected[0]["steps"], 1],
        strict=True,
    ):
        values = {
            row["component"]: row["device_us"] / 1000 / divisor
            for row in data["components"]
        }
        values["Remaining components"] = sum(
            v for k, v in values.items() if k not in names
        )
        bars = ax.barh(
            names,
            [values[n] for n in names],
            color=list(plt.get_cmap("tab10").colors[: len(names)]),
        )
        total = data["device_us"] / 1000 / divisor
        ax.bar_label(
            bars,
            labels=[
                f"{values[n]:.2f} ms ({100 * values[n] / total:.1f}%)" for n in names
            ],
            padding=4,
            fontsize=9,
        )
        ax.set_xlim(0, max(values[n] for n in names) * 1.5)
        ax.set_title(title)
        ax.set_xlabel("Summed GPU kernel duration (ms)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Super3.5 IQ2_XXS-Packed · GB10 · CUDA graphs\n"
        "Components overlap across streams; these sums are not wall latency.",
        fontsize=12,
    )
    fig.savefig(out / "components.png", dpi=160)
    fig.savefig(out / "components.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--compile-cache", type=Path, default=Path.home() / ".cache/b12x/compile"
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    inventory = dense_inventory(args.compile_cache)
    matched = {}
    summaries = {
        p.name.split("-0.")[0]: summarize(p, inventory, matched)
        for p in sorted((out / "traces").glob("*-0.pt.trace.json.gz"))
    }
    result = dict(
        timing_kind="Profiled GPU kernel duration sums, not wall latency",
        denominator="Component duration / all GPU kernel duration in the phase",
        analyzer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        traces=summaries,
    )
    (out / "breakdown.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "matched-compile-manifests.json").write_text(
        json.dumps(matched, indent=2) + "\n"
    )
    measurements = json.loads((out / "measurements.json").read_text())
    lines = [
        "# Super3.5 component profile",
        "",
        result["timing_kind"] + ".",
        "Dense attribution uses the cached compile facts and launch geometry. "
        "Split-K reductions are charged to their owning projection. "
        "Decode excludes the first eight decode iterations in each batch size.",
        "Components overlap across streams. Percentages use summed kernel durations; "
        "they are not percentages of wall latency. CUDA copies and host work are "
        "visible in the full traces but excluded from the component kernel sums.",
        "",
        "## Unprofiled HTTP measurements",
        "",
        "| Case | Samples | Median TTFT (ms) | Median decode (ms/token/request) |",
        "|---|---:|---:|---:|",
    ]
    for case in summaries:
        requests = [
            r
            for row in measurements["records"]
            if row["case"] == case and not row["profiled"]
            for r in row["requests"]
        ]
        ttft = statistics.median(r["ttft_s"] for r in requests) * 1000
        decode = [
            r["decode_s_per_token"] * 1000
            for r in requests
            if r["decode_s_per_token"] is not None
        ]
        lines.append(
            f"| {case} | {len(requests)} | {ttft:.2f} | "
            + (f"{statistics.median(decode):.2f}" if decode else "—")
            + " |"
        )
    for case, trace in summaries.items():
        for phase, data in trace["phases"].items():
            lines += [
                "",
                f"## {case}: {phase}",
                "",
                f"{data['steps']} iterations; {data['context_tokens']} prefill tokens; "
                f"{data['generation_tokens']} decode tokens; "
                f"{data['device_us'] / 1000:.3f} ms summed GPU time.",
                f"Kernel-active GPU time after removing stream overlap: "
                f"{data['gpu_busy_union_us'] / 1000:.3f} ms.",
                "",
                "| Component | GPU ms total | GPU ms/iteration | GPU share |",
                "|---|---:|---:|---:|",
            ]
            for row in data["components"]:
                lines.append(
                    f"| {row['component']} | {row['device_us'] / 1000:.3f} | "
                    f"{row['ms_per_step']:.3f} | {row['percent']:.2f}% |"
                )
    (out / "breakdown.md").write_text("\n".join(lines) + "\n")
    if args.plot:
        plot(out, summaries)
    for case, trace in summaries.items():
        for phase, data in trace["phases"].items():
            print(
                case,
                phase,
                data["steps"],
                [
                    (r["component"], round(r["ms_per_step"], 3), round(r["percent"], 1))
                    for r in data["components"][:10]
                ],
            )


if __name__ == "__main__":
    main()
