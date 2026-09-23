"""Benchmark fixed-payload P24/P33 trellis pairs at Kimi-K3 TP12 geometry.

This is a graph-first benchmark of the public dense decoder primitive used to
feed the routed implementation.  Each local K3 expert projection is either
``[3584, 256]`` (gate/up, N-axis pair) or ``[256, 3584]`` (down, K-axis pair).
P24, P33, and the existing uniform-K3 control therefore each carry exactly
344,064 payload bytes per projection.

Example:

    python benchmarks/benchmark_trellis_pair_tp12.py \
        --rows 1,2,4,8 --replays 200 --cold-replays 50 \
        --output out/trellis-pair-tp12.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
from typing import Callable

import torch

try:
    from benchmarks.common import make_l2_flush_fn, nvidia_smi_gpu_mode_snapshot
except ModuleNotFoundError:  # Direct ``python benchmarks/...py`` execution.
    from common import make_l2_flush_fn, nvidia_smi_gpu_mode_snapshot
from b12x.gemm import trellis_linear


_HIDDEN = 3584
_LOCAL_INTERMEDIATE = 256
_PAYLOAD_BYTES = _HIDDEN * _LOCAL_INTERMEDIATE * 3 // 8
_PAIR_BITS = {"P24": (2, 4), "P33": (3, 3)}
_CODEBOOKS = ("mcg", "lut_e4m3")
_RESULT_KIND = "b12x_trellis_pair_tp12_benchmark"
_RESULT_SCHEMA_VERSION = 1


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ("git", *args),
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _source_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__).resolve(),
        root / "b12x/_lib/intrinsics.py",
        root / "b12x/gemm/trellis_linear/api.py",
        root / "b12x/moe/_shared/kernels/w4a16/kernel.py",
        root / "b12x/moe/_shared/kernels/w4a16/prepare.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _write_new_result(path: Path, payload: str) -> None:
    """Publish one complete benchmark result without clobbering evidence."""

    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        descriptor = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parse_rows(value: str) -> list[int]:
    rows = [int(item) for item in value.split(",") if item.strip()]
    if not rows or any(item <= 0 for item in rows):
        raise argparse.ArgumentTypeError("rows must be a non-empty positive list")
    return rows


def _payload(
    pair_kind: str,
    rate_axis: str,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    low_bits, high_bits = _PAIR_BITS[pair_kind]
    orthogonal_tiles = _HIDDEN // 16
    if rate_axis == "n":
        low_shape = (orthogonal_tiles, 8, 16 * low_bits)
        high_shape = (orthogonal_tiles, 8, 16 * high_bits)
    else:
        low_shape = (8, orthogonal_tiles, 16 * low_bits)
        high_shape = (8, orthogonal_tiles, 16 * high_bits)
    low = torch.randint(
        -32768,
        32767,
        low_shape,
        dtype=torch.int16,
        device=device,
        generator=generator,
    )
    high = torch.randint(
        -32768,
        32767,
        high_shape,
        dtype=torch.int16,
        device=device,
        generator=generator,
    )
    payload = torch.cat((low.reshape(-1), high.reshape(-1))).contiguous()
    if payload.numel() * payload.element_size() != _PAYLOAD_BYTES:
        raise AssertionError("pair construction violated the fixed TP12 byte count")
    return payload


def _capture_case(
    codec: str,
    rate_axis: str,
    rows: int,
    *,
    codebook: str,
    device: torch.device,
    seed: int,
    warmup: int,
) -> tuple[torch.cuda.CUDAGraph, list[object], dict[str, object]]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    if rate_axis == "n":
        in_features, out_features = _HIDDEN, _LOCAL_INTERMEDIATE
    else:
        in_features, out_features = _LOCAL_INTERMEDIATE, _HIDDEN
    suh = torch.ones(in_features, dtype=torch.float16, device=device)
    svh = torch.ones(out_features, dtype=torch.float16, device=device)
    codebook_kwargs = (
        {
            "mcg": torch.tensor(
                0xCBAC1FED, dtype=torch.uint32, device=device
            )
        }
        if codebook == "mcg"
        else {"codebook": codebook}
    )
    if codec == "K3":
        payload = torch.randint(
            -32768,
            32767,
            (in_features // 16, out_features // 16, 16 * 3),
            dtype=torch.int16,
            device=device,
            generator=generator,
        )
        weight = trellis_linear.prepare_weight(
            payload,
            suh,
            svh,
            **codebook_kwargs,
            params_dtype=torch.float16,
        )
    else:
        payload = _payload(
            codec,
            rate_axis,
            device=device,
            generator=generator,
        )
        weight = trellis_linear.prepare_pair_weight(
            payload,
            suh,
            svh,
            pair_kind=codec,
            rate_axis=rate_axis,
            **codebook_kwargs,
            params_dtype=torch.float16,
        )
    if payload.numel() * payload.element_size() != _PAYLOAD_BYTES:
        raise AssertionError("benchmark controls must have identical payload bytes")
    x = torch.randn(
        (rows, in_features),
        dtype=torch.float16,
        device=device,
        generator=generator,
    )
    output = torch.empty((rows, out_features), dtype=torch.float16, device=device)
    gemm_output = torch.empty_like(output)
    rotated_f16 = torch.empty_like(x)
    # The selected TP12 specializations do not split K, but retaining an
    # explicit scratch allocation keeps graph capture independent of host-side
    # lazy allocation.
    c_tmp = torch.empty((1 << 20,), dtype=torch.float32, device=device)

    def launch() -> None:
        trellis_linear.run(
            x,
            weight,
            output=output,
            gemm_output=gemm_output,
            rotated_f16=rotated_f16,
            c_tmp=c_tmp,
        )

    for _ in range(warmup):
        launch()
    torch.cuda.synchronize(device)
    eager = output.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    graph.replay()
    torch.cuda.synchronize(device)
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError(
            f"{codec}/axis={rate_axis}/rows={rows} produced non-finite output"
        )
    if not torch.equal(output, eager):
        raise RuntimeError(
            f"{codec}/axis={rate_axis}/rows={rows} graph replay differs from eager"
        )
    # Keep every graph-owned tensor alive for the full benchmark.
    keepalive: list[object] = [
        payload,
        suh,
        svh,
        weight,
        x,
        output,
        gemm_output,
        rotated_f16,
        c_tmp,
        eager,
    ]
    correctness = {
        "passed": True,
        "finite": True,
        "eager_graph_bit_exact": True,
        "output_dtype": str(output.dtype),
        "output_shape": list(output.shape),
    }
    return graph, keepalive, correctness


def _interleaved_times_us(
    graphs: dict[str, torch.cuda.CUDAGraph],
    *,
    replays: int,
    flush_l2: Callable[[], None] | None,
) -> dict[str, list[float]]:
    times: dict[str, list[float]] = {name: [] for name in graphs}
    names = list(graphs)
    events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
    for replay in range(replays):
        order = names if replay % 2 == 0 else list(reversed(names))
        for name in order:
            if flush_l2 is not None:
                flush_l2()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            events.append((name, start, end))
    torch.cuda.synchronize()
    for name, start, end in events:
        times[name].append(start.elapsed_time(end) * 1000.0)
    return times


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(ordered),
        "p10_us": ordered[max(0, int(0.10 * (len(ordered) - 1)))],
        "p90_us": ordered[min(len(ordered) - 1, int(0.90 * (len(ordered) - 1)))],
        "min_us": ordered[0],
        "max_us": ordered[-1],
    }


def _paired_ratio_bootstrap(
    samples: list[float],
    baseline: list[float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    if len(samples) != len(baseline) or not samples:
        raise ValueError("paired timing samples must have the same nonzero length")
    sample_tensor = torch.tensor(samples, dtype=torch.float64)
    baseline_tensor = torch.tensor(baseline, dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(
        len(samples),
        (replicates, len(samples)),
        dtype=torch.int64,
        generator=generator,
    )
    sample_medians = torch.quantile(sample_tensor[indices], 0.5, dim=1)
    baseline_medians = torch.quantile(baseline_tensor[indices], 0.5, dim=1)
    ratios = sample_medians / baseline_medians
    bounds = torch.quantile(
        ratios, torch.tensor([0.025, 0.975], dtype=torch.float64)
    )
    return {
        "point_estimate": statistics.median(samples)
        / statistics.median(baseline),
        "bootstrap_ci95_low": float(bounds[0]),
        "bootstrap_ci95_high": float(bounds[1]),
        "bootstrap_replicates": replicates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codebook",
        choices=_CODEBOOKS,
        default="mcg",
        help="procedural reconstruction used by K3, P24, and P33",
    )
    parser.add_argument("--rows", type=_parse_rows, default=_parse_rows("1,2,4,8"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--cold-replays", type=int, default=50)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major != 12 or minor not in (0, 1):
        raise SystemExit(f"SM120/SM121 is required, got SM{major}{minor}")
    if (
        args.warmup < 1
        or args.replays < 1
        or args.cold_replays < 0
        or args.bootstrap_replicates < 1
    ):
        raise SystemExit(
            "warmup/replays/bootstrap-replicates must be positive and "
            "cold-replays non-negative"
        )

    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    result: dict[str, object] = {
        "kind": _RESULT_KIND,
        "schema_version": _RESULT_SCHEMA_VERSION,
        "provenance": {
            "command": shlex.join([sys.executable, *sys.argv]),
            "commit": _git_value("rev-parse", "HEAD"),
            "branch": _git_value("branch", "--show-current"),
            "git_dirty": bool(_git_value("status", "--porcelain")),
            "source_sha256": _source_sha256(),
            "worktree": str(Path(__file__).resolve().parents[1]),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "contract": {
            "tp": 12,
            "hidden_size": _HIDDEN,
            "local_intermediate_size": _LOCAL_INTERMEDIATE,
            "payload_bytes_per_projection": _PAYLOAD_BYTES,
            "codebook": args.codebook,
            "rows": args.rows,
            "outer_hadamard_included": True,
            "cuda_graph_replay": True,
            "cuda_graph_replay_allocation_stable_required": True,
            "paired_timing_bootstrap_replicates": args.bootstrap_replicates,
            "correctness": "finite eager output and bit-exact eager/graph replay",
        },
        "device": {
            "name": props.name,
            "uuid": str(getattr(props, "uuid", "")),
            "capability": [major, minor],
            "torch": torch.__version__,
            "mode": nvidia_smi_gpu_mode_snapshot(),
        },
        "cases": [],
    }
    cold_flush = make_l2_flush_fn(args.cold_replays > 0)
    keepalive: list[object] = []
    cases = result["cases"]
    assert isinstance(cases, list)

    for rows in args.rows:
        for rate_axis in ("n", "k"):
            graphs: dict[str, torch.cuda.CUDAGraph] = {}
            correctness: dict[str, dict[str, object]] = {}
            for offset, codec in enumerate(("P24", "P33", "K3")):
                graph, owned, closure = _capture_case(
                    codec,
                    rate_axis,
                    rows,
                    codebook=args.codebook,
                    device=device,
                    seed=args.seed + rows * 100 + offset,
                    warmup=args.warmup,
                )
                graphs[codec] = graph
                correctness[codec] = closure
                keepalive.extend(owned)
            allocated_before_timing = int(torch.cuda.memory_allocated(device))
            reserved_before_timing = int(torch.cuda.memory_reserved(device))
            hot = _interleaved_times_us(
                graphs,
                replays=args.replays,
                flush_l2=None,
            )
            cold = (
                _interleaved_times_us(
                    graphs,
                    replays=args.cold_replays,
                    flush_l2=cold_flush,
                )
                if args.cold_replays
                else None
            )
            allocated_after_timing = int(torch.cuda.memory_allocated(device))
            reserved_after_timing = int(torch.cuda.memory_reserved(device))
            if allocated_before_timing != allocated_after_timing:
                raise RuntimeError(
                    f"rows={rows}/axis={rate_axis} CUDA graph replay changed "
                    f"Torch allocation: {allocated_before_timing} -> "
                    f"{allocated_after_timing} bytes"
                )
            hot_summary = {name: _summary(values) for name, values in hot.items()}
            cold_summary = (
                {name: _summary(values) for name, values in cold.items()}
                if cold is not None
                else None
            )
            case = {
                "rows": rows,
                "rate_axis": rate_axis,
                "shape": (
                    [_HIDDEN, _LOCAL_INTERMEDIATE]
                    if rate_axis == "n"
                    else [_LOCAL_INTERMEDIATE, _HIDDEN]
                ),
                "hot": hot_summary,
                "raw_hot_us": hot,
                "hot_p24_over_p33": (
                    hot_summary["P24"]["median_us"]
                    / hot_summary["P33"]["median_us"]
                ),
                "hot_p33_over_k3": (
                    hot_summary["P33"]["median_us"]
                    / hot_summary["K3"]["median_us"]
                ),
                "hot_p24_over_p33_bootstrap": _paired_ratio_bootstrap(
                    hot["P24"],
                    hot["P33"],
                    replicates=args.bootstrap_replicates,
                    seed=args.seed + rows * 1000 + (0 if rate_axis == "n" else 1),
                ),
                "cold": cold_summary,
                "raw_cold_us": cold,
                "cold_p24_over_p33": (
                    None
                    if cold_summary is None
                    else cold_summary["P24"]["median_us"]
                    / cold_summary["P33"]["median_us"]
                ),
                "cold_p33_over_k3": (
                    None
                    if cold_summary is None
                    else cold_summary["P33"]["median_us"]
                    / cold_summary["K3"]["median_us"]
                ),
                "correctness": correctness,
                "graph_replay_allocation": {
                    "allocated_before_bytes": allocated_before_timing,
                    "allocated_after_bytes": allocated_after_timing,
                    "allocated_stable": (
                        allocated_before_timing == allocated_after_timing
                    ),
                    "reserved_before_bytes": reserved_before_timing,
                    "reserved_after_bytes": reserved_after_timing,
                    "reserved_stable": (
                        reserved_before_timing == reserved_after_timing
                    ),
                },
            }
            cases.append(case)
            print(json.dumps(case, sort_keys=True))

    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        _write_new_result(args.output, encoded)
    else:
        print(encoded)


if __name__ == "__main__":
    main()
