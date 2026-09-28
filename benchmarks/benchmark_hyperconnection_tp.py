"""Compare replicated Qwen HC with sharded layouts over real RoCEnante.

Run one process per Spark with RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT set.
Uses checkpoint BF16 weights and synthetic normalized activations. All staging,
GEMMs, activations, and required collectives are inside the measured graph;
normalization and residual combine, unchanged across arms, are outside it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open

from b12x.comm import roce
from b12x.comm.roce import _preparation
from b12x.norm.hyperconnection import _cute
from b12x.preparation import PreparationSession, PreparedCall

HIDDEN = 2560
STREAMS = 4
WIDTH = HIDDEN * STREAMS
LOWRANK = 320
ARMS = (
    "replicated",
    "bottleneck_reduce_fp32",
    "two_gathers",
    "small_reduce_gather",
    "two_gathers_fp32_down",
)
BOTTLENECK_SHARDS = (
    "bottleneck_reduce_fp32",
    "two_gathers",
    "two_gathers_fp32_down",
)


def gpu_identity():
    props = torch.cuda.get_device_properties(0)
    return {
        "hostname": os.uname().nodename,
        "name": props.name,
        "uuid": str(props.uuid),
        "capability": [props.major, props.minor],
        "sms": props.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def load_weights(model, count, rank, world):
    index = json.loads((model / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    local_rank = LOWRANK // world
    local_hidden = HIDDEN // world

    def tensor(name):
        with safe_open(model / weight_map[name], framework="pt", device="cpu") as f:
            return f.get_tensor(name).clone()

    def upload(source):
        return source.contiguous().to(device="cuda")

    result = []
    for i in range(count):
        kind = "attn" if i % 2 == 0 else "mlp"
        prefix = f"model.language_model.layers.{i // 2}.{kind}_hyper_connection"
        down = tensor(prefix + ".input_mix_weight_down.weight")
        injection = tensor(prefix + ".block_inject_weight.weight")
        up = tensor(prefix + ".input_mix_weight_up.weight")
        merged = torch.zeros(336, WIDTH, dtype=torch.bfloat16)
        merged[:LOWRANK] = down
        merged[LOWRANK : LOWRANK + STREAMS] = injection
        shard_rows = ((local_rank + STREAMS + 15) // 16) * 16
        local_down = torch.zeros(shard_rows, WIDTH, dtype=torch.bfloat16)
        local_down[:local_rank] = down[rank * local_rank : (rank + 1) * local_rank]
        local_down[local_rank : local_rank + STREAMS] = injection
        local_up_k = up[:, rank * local_rank : (rank + 1) * local_rank].contiguous()
        local_up_n = (
            up.view(STREAMS, HIDDEN, LOWRANK)[
                :, rank * local_hidden : (rank + 1) * local_hidden
            ]
            .reshape(STREAMS * local_hidden, LOWRANK)
            .contiguous()
        )
        result.append(
            {
                "prefix": prefix,
                "down": upload(merged),
                "up": upload(up),
                "down_shard": upload(local_down),
                "down_k_shard": upload(
                    merged[:, rank * (WIDTH // world) : (rank + 1) * (WIDTH // world)]
                ),
                "up_k_shard": upload(local_up_k),
                "up_n_shard": upload(local_up_n),
                "norm": tensor(prefix + ".hc_norm.weight").cuda(),
            }
        )
    return result


class Mixer:
    def __init__(self, weights, rows, rank, world, runtime, plan):
        self.weights = weights
        self.rows = rows
        self.rank = rank
        self.world = world
        self.runtime = runtime
        self.plan = plan
        self.local_rank = LOWRANK // world
        self.local_hidden = HIDDEN // world
        raw = torch.randn(rows, STREAMS, HIDDEN, device="cuda", dtype=torch.bfloat16)
        norm = raw.float() * torch.rsqrt(
            raw.float().square().mean(-1, keepdim=True) + 1e-6
        )
        self.x = (norm.flatten(1) * (1 + weights["norm"].float())).to(torch.bfloat16)
        projected = F.linear(self.x.float(), weights["down"].float()).to(torch.bfloat16)
        bottleneck = F.silu(projected[:, :LOWRANK].float() / STREAMS).to(torch.bfloat16)
        gates = F.linear(bottleneck.float(), weights["up"].float()).to(torch.bfloat16)
        self.oracle_out = (
            (gates.float().sigmoid().to(torch.bfloat16) * self.x)
            .float()
            .view(rows, STREAMS, HIDDEN)
            .mean(1)
            .to(torch.bfloat16)
        )
        self.oracle_injection = (
            2 * (projected[:, LOWRANK : LOWRANK + STREAMS].float() / STREAMS).sigmoid()
        )
        self.buffers = {}
        for name in ARMS:
            local = name in BOTTLENECK_SHARDS
            feature_shard = name in (
                "two_gathers",
                "small_reduce_gather",
                "two_gathers_fp32_down",
            )
            down_width = weights["down_shard"].shape[0] if local else 336
            self.buffers[name] = {
                "down": torch.empty(
                    rows, down_width, device="cuda", dtype=torch.bfloat16
                ),
                "bottleneck": torch.empty(
                    rows,
                    self.local_rank if local else LOWRANK,
                    device="cuda",
                    dtype=torch.bfloat16,
                ),
                "full_bottleneck": torch.empty(
                    rows, LOWRANK, device="cuda", dtype=torch.bfloat16
                ),
                "gates": torch.empty(
                    rows,
                    STREAMS * self.local_hidden if feature_shard else WIDTH,
                    device="cuda",
                    dtype=torch.bfloat16,
                ),
                "partial": torch.empty(rows, WIDTH, device="cuda", dtype=torch.float32),
                "reduced": torch.empty(rows, WIDTH, device="cuda", dtype=torch.float32),
                "down_input": torch.empty(
                    rows, WIDTH // world, device="cuda", dtype=torch.bfloat16
                ),
                "down_partial": torch.empty(
                    rows, 336, device="cuda", dtype=torch.float32
                ),
                "local_down_fp32": torch.empty(
                    rows, down_width, device="cuda", dtype=torch.float32
                ),
                "down_reduced": torch.empty(
                    rows, 336, device="cuda", dtype=torch.float32
                ),
                "local_x": torch.empty(
                    rows,
                    STREAMS * self.local_hidden,
                    device="cuda",
                    dtype=torch.bfloat16,
                ),
                "local_out": torch.empty(
                    rows, self.local_hidden, device="cuda", dtype=torch.bfloat16
                ),
                "out": torch.empty(rows, HIDDEN, device="cuda", dtype=torch.bfloat16),
            }

    def run(self, name):
        w, b = self.weights, self.buffers[name]
        local = name in BOTTLENECK_SHARDS
        width = self.local_rank if local else LOWRANK
        if name == "small_reduce_gather":
            shard_width = WIDTH // self.world
            b["down_input"].copy_(
                self.x[:, self.rank * shard_width : (self.rank + 1) * shard_width]
            )
            torch.mm(
                b["down_input"],
                w["down_k_shard"].T,
                out=b["down_partial"],
                out_dtype=torch.float32,
            )
            self.runtime.all_reduce(
                b["down_partial"], out=b["down_reduced"], plan=self.plan
            )
            b["down"].copy_(b["down_reduced"])
        elif name == "two_gathers_fp32_down":
            torch.mm(
                self.x,
                w["down_shard"].T,
                out=b["local_down_fp32"],
                out_dtype=torch.float32,
            )
            b["down"].copy_(b["local_down_fp32"])
        else:
            torch.mm(self.x, w["down_shard" if local else "down"].T, out=b["down"])
        _cute.scaled_silu(b["down"][:, :width], b["bottleneck"], streams=STREAMS)
        if name == "replicated":
            torch.mm(b["bottleneck"], w["up"].T, out=b["gates"])
        elif name == "bottleneck_reduce_fp32":
            torch.mm(
                b["bottleneck"],
                w["up_k_shard"].T,
                out=b["partial"],
                out_dtype=torch.float32,
            )
            self.runtime.all_reduce(b["partial"], out=b["reduced"], plan=self.plan)
            b["gates"].copy_(b["reduced"])
        else:
            if name in ("two_gathers", "two_gathers_fp32_down"):
                self.runtime.all_gather(
                    b["bottleneck"], dim=-1, out=b["full_bottleneck"], plan=self.plan
                )
                bottleneck = b["full_bottleneck"]
            else:
                bottleneck = b["bottleneck"]
            torch.mm(bottleneck, w["up_n_shard"].T, out=b["gates"])
            b["local_x"].view(self.rows, STREAMS, self.local_hidden).copy_(
                self.x.view(self.rows, STREAMS, HIDDEN)[
                    :,
                    :,
                    self.rank * self.local_hidden : (self.rank + 1) * self.local_hidden,
                ]
            )
            _cute.gate_mean(
                b["local_x"],
                b["gates"],
                b["local_out"],
                streams=STREAMS,
                hidden_size=self.local_hidden,
            )
            self.runtime.all_gather(
                b["local_out"], dim=-1, out=b["out"], plan=self.plan
            )
            return
        _cute.gate_mean(
            self.x, b["gates"], b["out"], streams=STREAMS, hidden_size=HIDDEN
        )

    def check(self, name):
        ref = self.buffers["replicated"]
        got = self.buffers[name]
        expected = ref["out"].float()
        actual = got["out"].float()
        rms_relative = (
            ((actual - expected).square().mean() / expected.square().mean())
            .sqrt()
            .item()
        )
        cosine = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
        assert torch.isfinite(actual).all() and actual.count_nonzero()
        assert rms_relative < 0.006, (name, rms_relative)
        assert cosine > 0.99998, (name, cosine)
        injection = (
            got["down"][:, self.local_rank : self.local_rank + STREAMS]
            if name in BOTTLENECK_SHARDS
            else got["down"][:, LOWRANK : LOWRANK + STREAMS]
        )
        injection_factor = 2 * (injection.float() / STREAMS).sigmoid()
        baseline_injection = (
            2
            * (ref["down"][:, LOWRANK : LOWRANK + STREAMS].float() / STREAMS).sigmoid()
        )
        torch.testing.assert_close(
            injection_factor,
            self.oracle_injection,
            rtol=0,
            atol=0.005,
            msg=lambda msg: f"{self.weights['prefix']} {name}: {msg}",
        )
        oracle = self.oracle_out.float()
        oracle_relative_rms = (
            ((actual - oracle).square().mean() / oracle.square().mean()).sqrt().item()
        )
        assert oracle_relative_rms < 0.006, (name, oracle_relative_rms)
        if name in ("small_reduce_gather", "two_gathers_fp32_down"):
            assert oracle_relative_rms < 0.001, (name, oracle_relative_rms)
        baseline_oracle_rms = (
            ((expected - oracle).square().mean() / oracle.square().mean()).sqrt().item()
        )
        assert oracle_relative_rms <= 1.2 * baseline_oracle_rms + 0.001, (
            name,
            oracle_relative_rms,
            baseline_oracle_rms,
        )
        peers = [torch.empty_like(got["out"]) for _ in range(self.world)]
        dist.all_gather(peers, got["out"])
        assert all(torch.equal(peer, got["out"]) for peer in peers)
        return {
            "relative_rms": rms_relative,
            "cosine": cosine,
            "max_abs": (actual - expected).abs().max().item(),
            "exact_fraction": (actual == expected).float().mean().item(),
            "oracle_relative_rms": oracle_relative_rms,
            "injection_factor_max_abs_vs_baseline": (
                injection_factor - baseline_injection
            )
            .abs()
            .max()
            .item(),
            "injection_factor_max_abs_vs_oracle": (
                injection_factor - self.oracle_injection
            )
            .abs()
            .max()
            .item(),
        }


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    stream.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    return graph, stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mixers", type=int, default=32)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 4, 6, 8])
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert LOWRANK % world == HIDDEN % world == 0
    torch.manual_seed(20260921)
    runtime = roce.AllReduce(
        exchange_group=dist.group.WORLD,
        device="cuda:0",
        max_size=1 << 20,
        max_gather_bytes=1 << 20,
    )
    query = roce.query_from_runtime(
        runtime,
        surface="AllReduce.all_reduce",
        call={"dtypes": ("bfloat16", "float32")},
        topology="roce_rdma",
        peer_hosts=tuple(f"rank-{i}" for i in range(world)),
    )
    plan = roce.plan(query, runtime=runtime)
    seeds = [
        torch.zeros(16 // dtype.itemsize, dtype=dtype, device="cuda")
        for dtype in (torch.bfloat16, torch.float32)
    ]

    def prepare(state):
        calls = [_preparation.prepared_call(state, inp=seed) for seed in seeds]
        calls.append(_preparation.prepared_gather_call(state, inp=seeds[0]))
        return PreparedCall(run=lambda: [call.run() for call in calls])

    session = PreparationSession(
        device=torch.device("cuda:0"), autotune=False, compile_workers=2
    )
    session.prepare((plan.request(name="hc-tp-roce", prepare_call=prepare),))
    identities = [None] * world
    dist.all_gather_object(identities, gpu_identity())
    receipt = {
        "command": sys.argv,
        "identities": identities,
        "world": world,
        "weights_allocation": "pytorch",
        "mixers": args.mixers,
        "measurement": "interleaved cold-L2 CUDA graph, median of per-sample rank maxima, per mixer",
        "source_revision": os.environ.get("B12X_BENCH_GIT_REV"),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "geometry": {"hidden": HIDDEN, "streams": STREAMS, "lowrank": LOWRANK},
        "matmul_precision": {
            "float32": torch.get_float32_matmul_precision(),
            "allow_bf16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUTE_DSL_ARCH",
                "NCCL_IB_HCA",
                "NCCL_IB_GID_INDEX",
                "NCCL_IB_TC",
                "NCCL_NET",
                "B12X_ROCE_TRAFFIC_CLASS",
                "LD_PRELOAD",
                "B12X_COMPILE_CACHE_DIR",
            )
        },
        "validation": {
            "input": "seeded BF16 normal, per-stream RMSNorm with checkpoint affine",
            "reference": "FP32 GEMMs with model BF16 rounding points",
            "max_relative_rms_vs_replicated": 0.006,
            "min_cosine_vs_replicated": 0.99998,
            "max_relative_rms_vs_reference": 0.006,
            "max_relative_rms_vs_reference_fp32_down": 0.001,
            "max_injection_factor_abs_error_vs_reference": 0.005,
            "cross_rank": "bit-identical outputs required",
        },
        "cases": [],
    }
    weights = load_weights(args.model, args.mixers, rank, world)
    if rank == 0:
        print("Weights loaded; distributed correctness before timing", flush=True)
    flush = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
    barrier = torch.zeros(16, dtype=torch.float32, device="cuda")
    names = list(ARMS)
    for rows in args.rows:
        torch.manual_seed(20260921 + rows)
        mixers = [Mixer(w, rows, rank, world, runtime, plan) for w in weights]

        def run(name, mixers=mixers):
            for mixer in mixers:
                mixer.run(name)

        correct = {}
        for name in names:
            if rank == 0:
                print(f"rows={rows}: checking {name}", flush=True)
            run(name)
            torch.cuda.synchronize()
            correct[name] = [mixer.check(name) for mixer in mixers]
        if rank == 0:
            print(f"rows={rows}: correctness passed; capturing", flush=True)
        graphs = {name: capture(lambda name=name: run(name)) for name in names}
        for _ in range(6):
            for name in names:
                graphs[name][0].replay()
        torch.cuda.synchronize()
        samples = {name: [] for name in names}
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for sample in range(args.samples):
            order = names if sample % 2 == 0 else names[::-1]
            for name in order:
                flush.zero_()
                dist.all_reduce(barrier, op=dist.ReduceOp.MAX)
                start.record()
                graphs[name][0].replay()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) * 1000 / args.mixers)
        for name in names:
            graphs[name][0].replay()
            torch.cuda.synchronize()
            for mixer in mixers:
                mixer.check(name)
        if args.profile is not None and rows == 4:
            dist.barrier()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profile:
                for name in names:
                    flush.zero_()
                    dist.all_reduce(barrier, op=dist.ReduceOp.MAX)
                    with torch.profiler.record_function(name):
                        graphs[name][0].replay()
                        torch.cuda.synchronize()
            args.profile.mkdir(parents=True, exist_ok=True)
            profile.export_chrome_trace(
                str(args.profile / f"hc-rank{rank}.trace.json.gz")
            )
        per_rank = [None] * world
        dist.all_gather_object(per_rank, samples)
        maxima = {
            name: [max(v) for v in zip(*(r[name] for r in per_rank), strict=True)]
            for name in names
        }
        medians = {name: statistics.median(v) for name, v in maxima.items()}
        case = {
            "rows": rows,
            "median_us": medians,
            "raw_us_by_rank": per_rank,
            "rank_max_samples_us": maxima,
            "correctness_rank0": correct,
            "baseline_over_candidate": {
                name: medians["replicated"] / medians[name] for name in names
            },
        }
        receipt["cases"].append(case)
        if rank == 0:
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in case.items()
                        if k in ("rows", "median_us", "baseline_over_candidate")
                    }
                ),
                flush=True,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(receipt, indent=2) + "\n")
        del graphs, mixers
    torch.cuda.synchronize()
    dist.barrier()
    runtime.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    with torch.inference_mode():
        main()
