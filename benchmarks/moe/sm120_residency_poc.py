"""One-layer SM120 NVFP4 cache experiment; not a serving backend.

Compose two prepared native W4A16 operations with fixed device/mapped-host
slabs, the shared cache policy and the existing journaled slot transaction.
The per-tier final sums are discarded: combine weighted BF16 route outputs
once in original top-k order. This deliberately retains redundant launches.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import time

import torch

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.moe import fused_moe as moe
from b12x.moe.residency import (
    ExpertPlacement,
    ResidencyCacheConfig,
    ResidencyCacheController,
    ResidencyExchangeSpec,
    RoutingObservationSpec,
)
from b12x.moe.fused_moe._residency_storage import TierStorage, align, _swizzle_scale
from b12x.moe.fused_moe._residency_updates import _SlotUpdates, _CudaTransfer
from b12x.preparation import PreparationSession, PreparedCall
from b12x.sequence._shared.disk_table import MappedHostAllocation
from benchmarks.moe_preparation import prepared_call, request_for_capacity, scratch_for


def load_layer(checkpoint: Path, prefix: str, count: int):
    """Read separate ModelOpt gate/up/down rows, including unindexed exports.

    This test adapter accepts logical E4M3 K16 scales and equal gate/up global
    scales. It only permutes bytes; unsupported exports fail before preparation.
    """
    from safetensors import safe_open

    if count < 2:
        raise ValueError("the cache experiment needs at least two experts")
    names = {
        f"{prefix}.{e}.{proj}.{field}"
        for e in range(count)
        for proj in ("up_proj", "gate_proj", "down_proj")
        for field in ("weight", "weight_scale", "weight_scale_2")
    }
    values = {}
    for shard in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in names.intersection(handle.keys()):
                if name in values:
                    raise ValueError(f"duplicate checkpoint field: {name}")
                values[name] = handle.get_tensor(name)
    if names - values.keys():
        raise ValueError(f"missing checkpoint field: {min(names - values.keys())}")
    rows = {name: [] for name in ("w13", "w2", "s13", "s2", "g13", "g2")}
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name]
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    for e in range(count):

        def get(proj, field):
            return values[f"{prefix}.{e}.{proj}.{field}"]

        up, gate, down = (
            get(p, "weight") for p in ("up_proj", "gate_proj", "down_proj")
        )
        if (
            up.dtype != torch.uint8
            or gate.dtype != torch.uint8
            or down.dtype != torch.uint8
        ):
            raise ValueError("expected packed NVFP4 uint8 expert weights")
        if (
            up.ndim != 2
            or gate.shape != up.shape
            or down.shape != (up.shape[1] * 2, up.shape[0] // 2)
        ):
            raise ValueError("incompatible gated expert geometry")
        gs = [get(p, "weight_scale_2") for p in ("up_proj", "gate_proj", "down_proj")]
        if any(
            s.dtype != torch.float32
            or s.numel() != 1
            or not torch.isfinite(s).all()
            or (s <= 0).any()
            for s in gs
        ):
            raise ValueError("expected positive finite scalar FP32 global scales")
        if not torch.equal(gs[0], gs[1]):
            raise ValueError(
                "gate/up global scales differ; this prototype never requantizes"
            )
        scales = []
        for p, weight in zip(
            ("up_proj", "gate_proj", "down_proj"), (up, gate, down), strict=True
        ):
            s = get(p, "weight_scale")
            if s.dtype != torch.float8_e4m3fn or s.shape != (
                weight.shape[0],
                weight.shape[1] // 8,
            ):
                raise ValueError("expected logical E4M3 K16 block scales")
            scales.append(s.view(torch.uint8))
        rows["w13"].append(torch.cat((up, gate)))
        rows["w2"].append(down)
        rows["s13"].append(_swizzle_scale(torch.cat(scales[:2])))
        rows["s2"].append(_swizzle_scale(scales[2]))
        rows["g13"].append(gs[0].reshape(1).view(torch.uint8))
        rows["g2"].append(gs[2].reshape(1).view(torch.uint8))
    return {name: torch.stack(row) for name, row in rows.items()}, digest.hexdigest()


def make_tier(source, ids, device, *, mapped, write_combined=True):
    layout, size = [], 0
    for name, value in source.items():
        shape = (len(ids), *value.shape[1:])
        layout.append((name, size, shape))
        size = align(size + math.prod(shape))
    owner = (
        MappedHostAllocation(
            (size,), torch.uint8, device, write_combined=write_combined
        )
        if mapped
        else None
    )
    slab = (
        owner.device_view
        if mapped
        else torch.empty(size, dtype=torch.uint8, device=device)
    )
    destination = owner.host_view if mapped else slab
    fields = {}
    for name, offset, shape in layout:
        end = offset + math.prod(shape)
        destination[offset:end].view(shape).copy_(source[name][list(ids)])
        fields[name] = slab[offset:end].view(shape)
    return TierStorage(slab, fields, owner)


def declare(tier, capacity, topk, total):
    f = tier.fields
    e, h, i = f["w13"].shape[0], f["w2"].shape[1], f["w13"].shape[1] // 2
    weight_plan = moe.plan_weights(
        source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
        activation=moe.ActivationSpec(
            mode="a16", nonlinearity="silu", io_dtype=torch.bfloat16
        ),
        geometry=moe.MoEGeometry(num_experts=e, hidden_size=h, intermediate_size=i),
        constraints=moe.WeightPlanConstraints(required_packing="source_native"),
    )
    experts = moe.prepare_weights(
        plan=weight_plan,
        weights=moe.PackedWeights(
            w13=f["w13"],
            w2=f["w2"],
            w13_block_scales=f["s13"],
            w2_block_scales=f["s2"],
            w13_global_scales=f["g13"].view(torch.float32).reshape(e),
            w2_global_scales=f["g2"].view(torch.float32).reshape(e),
        ),
    )
    plan = moe.plan_execution(
        experts=experts,
        capacity=moe.ExecutionCapacity(
            max_tokens=capacity,
            top_k=topk,
            warmup_token_counts=(capacity,),
            route_num_experts=total,
        ),
        override=moe.MoeDecodeConfig(
            backend="w4a16",
            route_planner="internal",
            max_active_clusters=None,
            w4a16_route_mode="packed",
        ),
    )
    return experts, plan


class Experiment:
    """Own one serialized experimental lane; close only after graph release."""

    def __init__(
        self,
        source,
        *,
        hot,
        capacity,
        topk,
        id_dtype=torch.int64,
        cache_dir=None,
        backing_write_combined=True,
        journal_write_combined=True,
    ):
        self.tiers, self.graphs, self.journal_owner, self.session = [], [], None, None
        try:
            self._initialize(
                source,
                hot=hot,
                capacity=capacity,
                topk=topk,
                id_dtype=id_dtype,
                cache_dir=cache_dir,
                backing_write_combined=backing_write_combined,
                journal_write_combined=journal_write_combined,
            )
        except BaseException:
            self.close()
            raise

    def _initialize(
        self,
        source,
        *,
        hot,
        capacity,
        topk,
        id_dtype,
        cache_dir,
        backing_write_combined,
        journal_write_combined,
    ):
        from .sm120_residency_support import compile_support, invoke

        self.invoke = invoke
        self.device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.get_device_capability(self.device) != (12, 0):
            raise RuntimeError("this experiment requires a physical SM120 device")
        from cuda.bindings import runtime as cuda

        error, supported = cuda.cudaDeviceGetAttribute(
            cuda.cudaDeviceAttr.cudaDevAttrCanMapHostMemory, self.device.index
        )
        if error != cuda.cudaError_t.cudaSuccess or not supported:
            raise RuntimeError("the selected CUDA device cannot map host memory")
        self.e, self.h = source["w2"].shape[:2]
        if not 0 < hot < self.e or capacity < 1 or topk < 1:
            raise ValueError(
                "require nonempty hot/cold tiers and positive capacity/top-k"
            )
        self.placement = ExpertPlacement(
            total_experts=self.e,
            resident_expert_ids=tuple(range(hot)),
            backing_expert_ids=tuple(range(hot, self.e)),
        )
        for t, ids in enumerate(
            (self.placement.resident_expert_ids, self.placement.backing_expert_ids)
        ):
            self.tiers.append(
                make_tier(
                    source,
                    ids,
                    self.device,
                    mapped=t == 1,
                    write_combined=backing_write_combined,
                )
            )
        self.control_tier = make_tier(source, range(self.e), self.device, mapped=False)
        self.mapping = torch.tensor(
            self.placement.expert_map, dtype=torch.int32, device=self.device
        )
        self.maps = torch.empty(2, self.e, dtype=torch.int32, device=self.device)
        self.identity = torch.arange(self.e, dtype=torch.int32, device=self.device)
        self.ids = torch.zeros(capacity, topk, dtype=id_dtype, device=self.device)
        self.safe_ids = torch.empty_like(self.ids, dtype=torch.int32)
        self.a = (
            torch.randn(capacity, self.h, device=self.device, dtype=torch.bfloat16)
            * 0.1
        )
        self.weights = torch.full((capacity, topk), 1 / topk, device=self.device)
        self.output = torch.empty_like(self.a)
        self.remap, self.reduce = compile_support(self.e, self.h, topk, id_dtype)
        self.refresh(capacity)
        journal_bytes = sum(align(2 * v[0].numel()) for v in source.values())
        self.journal_owner = MappedHostAllocation(
            (journal_bytes + 2 * align(self.e * 8),),
            torch.uint8,
            self.device,
            write_combined=journal_write_combined,
        )
        host, offset, journal = self.journal_owner.host_view, 0, {}
        for name, value in source.items():
            n = 2 * value[0].numel()
            journal[name] = host[offset : offset + n].view(2, *value.shape[1:])
            offset += align(n)
        before = host[offset : offset + self.e * 8].view(torch.int32).view(self.e, 2)
        after = (
            host[offset + align(self.e * 8) : offset + align(self.e * 8) + self.e * 8]
            .view(torch.int32)
            .view(self.e, 2)
        )
        self.updates = _SlotUpdates(
            tiers=self.tiers,
            mapping=self.mapping,
            expert_map=self.placement.expert_map,
            journal=journal,
            before=before,
            after=after,
            transfer=_CudaTransfer(self.device),
            owner=self.journal_owner,
        )
        self.row_bytes = sum(v[0].numel() for v in source.values())
        self.session = PreparationSession(
            device=self.device, autotune=False, compile_workers=0, cache_dir=cache_dir
        )
        self.prepared, self.scratch, self.graphs = [], [], []
        requests = []
        for number, tier in enumerate((*self.tiers, self.control_tier)):
            experts, plan = declare(tier, capacity, topk, self.e)
            out = torch.empty_like(self.a)
            route_map = self.maps[number] if number < 2 else self.identity
            kwargs = dict(
                a=self.a,
                experts=experts,
                topk_weights=self.weights,
                topk_ids=self.safe_ids,
                output=out,
                input_scales_static=True,
                route_expert_map=route_map,
            )
            requests.append(
                request_for_capacity(
                    plan,
                    name=f"tier-{number}",
                    calls={
                        capacity: prepared_call(
                            output=out,
                            bind=lambda state, scratch, kwargs=kwargs: state.bind(
                                scratch=scratch, **kwargs
                            ),
                        )
                    },
                )
            )
            self.prepared.append((plan, kwargs))
        self.query = moe.RoutingProfileQuery(
            layers=(("experiment", self.e),), max_tokens=capacity, max_top_k=topk
        )
        self.counter = moe.plan_routing_profile(self.query)

        def counter_call(state):
            binding = state.bind(layer="experiment", phase="decode", topk_ids=self.ids)
            return PreparedCall(run=binding.run, owners=(state, binding))

        requests.append(self.counter.request(name="counter", prepare_call=counter_call))
        self.session.prepare(tuple(requests))
        self.controls = moe.routing_profile_state(self.counter)
        # Binding the native payload must not migrate any mapped expert field.
        for plan, _ in self.prepared:
            child = getattr(plan, "variants", {capacity: plan})[capacity]
            scratch = scratch_for(child)
            self.scratch.append(scratch)
        self.bindings = self.bind(capacity)
        for binding, tier in zip(
            self.bindings, (*self.tiers, self.control_tier), strict=True
        ):
            native = binding.experts.representation_for("w4a16")
            for attr, field in (
                ("w13", "w13"),
                ("w2", "w2"),
                ("w13_scale", "s13"),
                ("w2_scale", "s2"),
                ("w13_global_scale", "g13"),
                ("w2_global_scale", "g2"),
            ):
                if getattr(native, attr).data_ptr() != tier.fields[field].data_ptr():
                    raise AssertionError(
                        f"native preparation copied the {field} slot storage"
                    )
        self.session.freeze()

    def refresh(self, live):
        self.invoke(
            self.remap,
            (self.mapping, self.maps, self.ids, self.safe_ids),
            (live * self.ids.shape[1],),
        )

    def bind(self, live):
        return [
            moe.bind(
                plan,
                scratch=scratch,
                **(
                    kwargs
                    | dict(
                        a=self.a[:live],
                        topk_ids=self.safe_ids[:live],
                        topk_weights=self.weights[:live],
                        output=kwargs["output"][:live],
                    )
                ),
            )
            for (plan, kwargs), scratch in zip(self.prepared, self.scratch, strict=True)
        ]

    def capture(self, live):
        bindings = self.bind(live)
        observer = moe.bind_routing_profile(
            self.counter, layer="experiment", phase="decode", topk_ids=self.ids[:live]
        )

        def run():
            self.refresh(live)
            for binding in bindings[:2]:
                moe.run(binding=binding)
            self.invoke(
                self.reduce,
                (
                    bindings[0].intermediate_cache13,
                    bindings[1].intermediate_cache13,
                    self.ids,
                    self.mapping,
                    self.output,
                ),
                (live,),
            )
            observer.run()

        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with self.session.capture(), torch.cuda.graph(graph):
            run()
        self.graphs.append(graph)
        return graph, bindings[2]

    def pointers(self):
        buffers = [
            self.mapping,
            self.maps,
            self.ids,
            self.safe_ids,
            self.a,
            self.weights,
            self.output,
            self.controls.storage,
            self.journal_owner.host_view,
            self.control_tier.slab,
            *(t.slab for t in self.tiers),
            *(x for row in self.scratch for x in row),
        ]
        return tuple(t.data_ptr() for t in buffers)

    def close(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for graph in self.graphs:
            graph.reset()
        if self.session is not None:
            self.session.close()
        if self.journal_owner is not None:
            self.journal_owner.close()
        for tier in self.tiers:
            if tier.owner is not None:
                tier.owner.close()


def exercise(experiment, *, live_counts):
    """Validate current cold execution, policy exchange and subsequent replay."""
    receipts = []
    for live in live_counts:
        graph, control = experiment.capture(live)
        experiment.controls.reset(quiescent=True)
        policy = ResidencyCacheController(
            config=ResidencyCacheConfig(
                max_pairs=1,
                minimum_cold_selections=2,
                minimum_score_gain=2,
                minimum_residency_windows=1,
            ),
            observations=RoutingObservationSpec(
                layer="experiment",
                experts=experiment.e,
                phase="decode",
                max_top_k=experiment.ids.shape[1],
            ),
            exchange=ResidencyExchangeSpec(
                backend="sm120-nvfp4-pcie-poc",
                direct_backing_execution=True,
                fixed_address_quiescent_exchange=True,
                payload_copy_bytes_per_pair=4 * experiment.row_bytes,
                map_copy_bytes_per_transaction=16 * experiment.e,
            ),
            slots=experiment.updates.snapshot(),
            baseline=experiment.controls.snapshot(quiescent=True),
        )
        pointers = experiment.pointers()
        # Two repeated cold workloads followed by returning to the initial hot expert.
        cold = [
            e
            for e, (tier, _) in enumerate(experiment.updates.snapshot().expert_map)
            if tier
        ]
        targets = (cold[0], cold[0], cold[-1], cold[-1], 0, 0)
        for expert in targets:
            experiment.ids.fill_(expert)
            experiment.a.mul_(0.9375)
            # Duplicate, sentinel and oversized IDs are tested by the same graph.
            if experiment.ids.shape[1] > 1:
                experiment.ids[:live, -1] = -1
            if experiment.ids.shape[1] > 2 and experiment.ids.dtype == torch.int64:
                experiment.ids[:live, -2] = 2**40
            gc.collect()
            before = torch.cuda.memory_stats()
            with kernel_resolution_guard("SM120 cache graph replay"):
                graph.replay()
                graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for key in (
                "allocation.all.allocated",
                "allocation.all.freed",
                "allocated_bytes.all.current",
            ):
                if before[key] != after[key]:
                    raise AssertionError(f"graph replay allocator event: {key}")
            reference = moe.run(binding=control)
            torch.cuda.synchronize()
            torch.testing.assert_close(
                experiment.output[:live], reference, atol=0, rtol=0
            )
            if not torch.isfinite(reference).all() or not torch.count_nonzero(
                reference
            ):
                raise AssertionError("native output must be finite and nonzero")
            slots = experiment.updates.snapshot()
            decision = policy.observe(
                experiment.controls.snapshot(quiescent=True), slots=slots
            )
            start = time.perf_counter()
            if decision.pairs:
                with kernel_resolution_guard("SM120 cache exchange"):
                    slots = experiment.updates.exchange(
                        decision.pairs, expected=decision.expected, quiescent=True
                    )
            pause_ms = (time.perf_counter() - start) * 1000 if decision.pairs else None
            outcome = policy.finish(decision, slots=slots)
            if pointers != experiment.pointers():
                raise AssertionError("captured pointer changed")
            receipts.append(
                dict(
                    kind="policy",
                    live=live,
                    target=expert,
                    decision=asdict(decision),
                    outcome=asdict(outcome),
                    exchange_wall_ms=pause_ms,
                    bitwise_equal=True,
                    replay_allocator_events=0,
                )
            )
        if outcome.promotions == 0:
            raise AssertionError("workload produced no policy exchange")
        for mode in ("mixed", "hot", "cold", "invalid"):
            slots = experiment.updates.snapshot()
            available = [
                e
                for e, (tier, _) in enumerate(slots.expert_map)
                if mode == "mixed"
                or (mode == "hot" and tier == 0)
                or (mode == "cold" and tier == 1)
            ]
            pattern = [
                available[n % len(available)] if available else -1
                for n in range(live * experiment.ids.shape[1])
            ]
            experiment.ids[:live].copy_(torch.tensor(pattern).view(live, -1))
            # Exact cancellation, negative and zero route weights use the same recipe.
            experiment.weights[:live].copy_(
                torch.linspace(-1, 1, experiment.weights[:live].numel()).view(live, -1)
            )
            with kernel_resolution_guard("SM120 mixed route replay"):
                graph.replay()
            torch.cuda.synchronize()
            actual = experiment.output[:live]
            reference = moe.run(binding=control)
            # Split expert groups can change native split-K rounding before the
            # BF16 route boundary. Validate that boundary separately from sum.
            bindings = experiment.bind(live)
            rows = [
                binding.intermediate_cache13[
                    : live * experiment.ids.shape[1] * experiment.h
                ].view(live, experiment.ids.shape[1], experiment.h)
                for binding in bindings[:2]
            ]
            ordered = torch.zeros_like(actual, dtype=torch.float32)
            for rank in range(experiment.ids.shape[1]):
                for token in range(live):
                    expert = pattern[token * experiment.ids.shape[1] + rank]
                    if expert >= 0:
                        ordered[token] += rows[slots.expert_map[expert][0]][
                            token, rank
                        ].float()
            torch.testing.assert_close(actual, ordered.bfloat16(), atol=0, rtol=0)
            af, rf = actual.float(), reference.float()
            relative_l2 = float((af - rf).norm() / rf.norm().clamp_min(1e-20))
            cosine = (
                float(
                    torch.nn.functional.cosine_similarity(
                        af.flatten(), rf.flatten(), dim=0
                    )
                )
                if rf.norm()
                else 1.0
            )
            if (
                not torch.isfinite(actual).all()
                or relative_l2 > 0.005
                or cosine < 0.9999
            ):
                raise AssertionError(
                    f"{mode=} {live=}: native W4A16 parity {relative_l2=}, {cosine=}"
                )
            receipts.append(
                dict(
                    kind="route-boundary",
                    live=live,
                    mode=mode,
                    bitwise_equal=torch.equal(actual, reference),
                    ordered_sum_exact=True,
                    relative_l2=relative_l2,
                    cosine=cosine,
                    max_abs=float((af - rf).abs().max()),
                )
            )
        experiment.weights.fill_(1 / experiment.ids.shape[1])
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prefix", default="model.language_model.layers.0.mlp.experts")
    parser.add_argument(
        "--experts",
        type=int,
        default=16,
        help="load canonical IDs [0, count) from one layer",
    )
    parser.add_argument("--hot-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--live", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--source-revision",
        required=True,
        help="source commit plus dirty/export identity",
    )
    args = parser.parse_args()
    if min(args.live) < 1:
        parser.error("live token counts must be positive")
    torch.manual_seed(731)
    source, source_hash = load_layer(args.checkpoint, args.prefix, args.experts)
    experiment = Experiment(
        source, hot=args.hot_experts, capacity=max(args.live), topk=args.top_k
    )
    try:
        receipt = dict(
            kind="one-layer-native-sm120-cache-poc",
            source_revision=args.source_revision,
            checkpoint_fields_sha256=source_hash,
            checkpoint=str(args.checkpoint),
            prefix=args.prefix,
            torch=torch.__version__,
            cuda=torch.version.cuda,
            toolchain={
                name: importlib.metadata.version(name)
                for name in ("nvidia-cutlass-dsl", "triton", "cuda-bindings")
            },
            arguments={
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            gpu=subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=uuid,name,driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
            ),
            geometry=dict(
                experts=args.experts,
                hidden=experiment.h,
                intermediate=source["w13"].shape[1] // 2,
                top_k=args.top_k,
            ),
            resident_slab_bytes=experiment.tiers[0].slab.numel(),
            backing_slab_bytes=experiment.tiers[1].slab.numel(),
            journal_bytes=experiment.journal_owner.host_view.numel(),
            records=exercise(experiment, live_counts=args.live),
        )
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: receipt[k]
                    for k in (
                        "kind",
                        "geometry",
                        "resident_slab_bytes",
                        "backing_slab_bytes",
                    )
                }
            )
        )
        print(
            f"PASS: {len(receipt['records'])} same-graph policy/route cases; receipt={args.receipt}"
        )
    finally:
        experiment.close()


if __name__ == "__main__":
    main()
