"""Plan capacity-specific learned placements from immutable calibration counts.

Uses the serving backend's memory formulas and profile constructor. The report
is a preflight estimate using a completed reference run's non-cache reservations;
the loader still performs authoritative admission against live memory.
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from b12x.integration.vllm.expert_cache import digest
from b12x.integration.vllm.residency_epoch import ResidencyServingMemory
from b12x.moe import fused_moe as moe
from b12x.moe.fused_moe._cache_preparation import ExpertCacheQuery, memory_for
from b12x.moe.fused_moe.residency import profile_from_counts
from b12x.testing.artifacts import sha256
from scripts._sm103_source import source_identity


def calibration_counts(profile, receipt):
    """Require the original calibrated artifact and its recorded observation."""
    payload = {k: v for k, v in profile.items() if k != "hash"}
    if digest(payload) != profile.get("hash"):
        raise ValueError("calibration profile hash mismatch")
    records = [r for r in receipt if r["kind"] == "profile"]
    if len(records) != 1 or receipt[-1]["kind"] != "complete":
        raise ValueError("calibration requires one completed profile receipt")
    results = records[0]["result"]
    if isinstance(results, list):
        world = profile["identity"].get("tp_size", 1)
        if len(results) != world:
            raise ValueError("calibration must include every TP participant")
        if world > 1:
            if {r["snapshot"]["rank"] for r in results} != set(range(world)):
                raise ValueError("calibration TP ranks are missing or duplicated")
            owner = next(r for r in results if r["snapshot"]["rank"] == 0)
            for result in results:
                if (
                    result["hash"] != owner["hash"]
                    or result["snapshot"]["layers"] != owner["snapshot"]["layers"]
                ):
                    raise ValueError("TP calibration counts or profile disagree")
            results = [owner]
        results = results[0]
    if results["hash"] != profile["hash"]:
        raise ValueError("calibration receipt belongs to a different profile")
    counts = {
        r["layer"]: tuple(r["counts"])
        for r in results["snapshot"]["layers"]
        if r.get("phase", "decode") == "decode"
    }
    identity = profile["identity"]
    if (
        set(counts) != set(identity["layers"])
        or set(counts) != set(profile["placements"])
        or identity["recipe"] != "nvfp4_w4a16_whole_k_weighted_bf16_ordered_sum"
    ):
        raise ValueError("calibration layers or numerical recipe mismatch")
    for name, value in profile["placements"].items():
        p = moe.ExpertResidencyPlan.from_dict(value)
        if (
            p.layer != name
            or p.model_fingerprint != identity["checkpoint"]
            or p.workload != identity["workload"]
            or p.phase != "decode"
            or p.total_experts != identity["layers"][name]["num_experts"]
            or tuple(p.selection_counts) != counts[name]
            or not sum(counts[name])
        ):
            raise ValueError("calibration identity or canonical counts mismatch")
    return counts


def allocate(queries, envelope, counter_bytes, memory):
    """Use calibration's fair incremental allocation with adaptive overhead."""
    counts = dict.fromkeys(queries, 1)
    rows = {n: memory(q, 1) for n, q in queries.items()}
    used = sum(m.hbm_total_bytes for m in rows.values()) + counter_bytes
    if used > envelope:
        raise ValueError(
            "expert envelope cannot admit one slot per layer and workspace"
        )
    while True:
        changed = False
        for name, q in queries.items():
            if counts[name] == q.experts:
                continue
            candidate = memory(q, counts[name] + 1)
            delta = candidate.hbm_total_bytes - rows[name].hbm_total_bytes
            if used + delta <= envelope:
                counts[name] += 1
                rows[name] = candidate
                used += delta
                changed = True
        if not changed:
            return counts, rows, used


def runtime_reservations(baseline, reference):
    """Charge observed engine storage beyond the pre-preparation reservation.

    CUDA free-memory readings include Torch pools and native allocations. Only
    their excess over already charged device storage is added; safety remains
    an additional fixed reserve rather than paying for known engine storage.
    """
    model = ResidencyServingMemory(**baseline)
    observations = [
        r
        for event in reference
        if event["kind"] == "resources"
        for r in event["result"]
        if r["stage"] in ("graphs_ready", "serving_finished")
    ]
    if not observations or any(
        r["device_total"] != model.device_capacity for r in observations
    ):
        raise ValueError("reference needs matching graph/serving resource checkpoints")
    observed = max(r["device_total"] - r["device_free"] for r in observations)
    additional = max(0, observed - (model.device_bytes - model.device_safety))
    adjusted = replace(model, other_device=model.other_device + additional)
    return asdict(adjusted), dict(
        observed_device_used_bytes=observed,
        additional_engine_reservation_bytes=additional,
        declared_model_memory=baseline,
        adjusted_model_memory=asdict(adjusted),
    )


def maximum_envelope(baseline):
    """Keep observed non-cache storage and every fixed reserve intact."""
    model = ResidencyServingMemory(**baseline)
    fixed = model.device_bytes - (
        model.resident_experts + model.workspace + model.metadata
    )
    return model.device_capacity - fixed


def plan(profile, counts, baseline, envelope, *, device, capacity=64, layer_pairs=2):
    identity = profile["identity"]
    queries = {
        name: ExpertCacheQuery(
            experts=g["num_experts"],
            resident=1,
            hidden=g["hidden_size"],
            intermediate=g["intermediate_size"],
            max_tokens=capacity,
            top_k=identity["top_k"],
            w13_layout=g["w13_layout"],
            checkpoint_fingerprint=identity["checkpoint"],
            profile_hash="0" * 64,
        )
        for name, g in sorted(identity["layers"].items())
    }
    observer = moe.RoutingProfileQuery(
        layers=tuple((n, q.experts) for n, q in queries.items()),
        max_tokens=capacity,
        max_top_k=identity["top_k"],
        runtime_token_limit=True,
        health_summary=True,
    )
    memo = {}

    def memory(q, resident):
        q = replace(
            q,
            resident=resident,
            max_pairs=min(layer_pairs, resident, q.experts - resident),
        )
        if q not in memo:
            memo[q] = memory_for(q, device, 0)
        return memo[q]

    slots, rows, used = allocate(
        queries, envelope, observer.storage_bytes + observer.health_device_bytes, memory
    )
    model = replace(
        ResidencyServingMemory(**baseline),
        resident_experts=sum(m.resident_bytes for m in rows.values()),
        backing_experts=sum(m.backing_bytes for m in rows.values()),
        workspace=sum(m.workspace_bytes for m in rows.values()),
        metadata=sum(m.metadata_bytes for m in rows.values())
        + observer.storage_bytes
        + observer.health_device_bytes,
        host_staging=sum(m.update_host_bytes for m in rows.values())
        + observer.health_host_bytes,
    )
    placements = {
        name: profile_from_counts(
            counts=counts[name],
            hot_count=slots[name],
            layer=name,
            model_fingerprint=identity["checkpoint"],
            workload=identity["workload"],
            provenance="capacity projection of immutable calibration "
            + profile["hash"],
            phase="decode",
        ).to_dict()
        for name in queries
    }
    return dict(
        expert_envelope_bytes=envelope,
        cache_device_bytes=used,
        counts=slots,
        expert_counts={n: q.experts for n, q in queries.items()},
        resident_payload_fraction=model.resident_experts / model.backing_experts,
        memory=asdict(model),
        device_reserved_bytes=model.device_bytes,
        device_reserved_headroom_bytes=model.device_capacity - model.device_bytes,
        host_reserved_bytes=model.host_bytes,
        all_resident=all(slots[n] == q.experts for n, q in queries.items()),
        adaptive_supported=any(slots[n] < q.experts for n, q in queries.items()),
    ), placements


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibrated-profile", type=Path, required=True)
    p.add_argument("--calibration-receipt", type=Path, required=True)
    p.add_argument("--reference-receipt", type=Path, required=True)
    capacity = p.add_mutually_exclusive_group(required=True)
    capacity.add_argument("--cache-gib", type=int, nargs="+")
    capacity.add_argument(
        "--maximum",
        action="store_true",
        help="Use the largest envelope after observed engine storage and fixed reserves",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--capacity",
        type=int,
        help="Prepared token capacity; defaults to the reference receipt's capacity",
    )
    args = p.parse_args()
    if args.capacity is not None and args.capacity < 1:
        p.error("capacity must be positive")
    import torch

    if torch.cuda.get_device_capability(args.device) != (12, 0):
        raise ValueError(
            "capacity preflight requires the physical SM120 reference device"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    profile = json.loads(args.calibrated_profile.read_text())
    calibration = [
        json.loads(r) for r in args.calibration_receipt.read_text().splitlines()
    ]
    counts = calibration_counts(profile, calibration)
    reference = [json.loads(r) for r in args.reference_receipt.read_text().splitlines()]
    if reference[-1]["kind"] != "complete":
        raise ValueError("reference must complete before using its reservations")
    status = next(r["status"] for r in reference if r["kind"] == "prepared")
    config = next(r for r in reference if r["kind"] == "configuration")
    prepared_capacity = (
        args.capacity if args.capacity is not None else config["arguments"]["capacity"]
    )
    if status["checkpoint"] != profile["identity"]["checkpoint"]:
        raise ValueError("reference checkpoint differs from calibration")
    baseline, accounting = runtime_reservations(status["memory"], reference)
    provenance = dict(
        source=source_identity(Path(__file__).resolve().parents[2]),
        device=str(torch.cuda.get_device_properties(args.device)),
        calibrated_profile_sha256=sha256(args.calibrated_profile),
        calibration_receipt_sha256=sha256(args.calibration_receipt),
        reference_receipt_sha256=sha256(args.reference_receipt),
        canonical_counts_sha256=digest(counts),
        calibration_arguments=calibration[0]["arguments"],
        reference_arguments=config["arguments"],
        prepared_capacity=prepared_capacity,
        reference_memory_accounting=accounting,
    )
    results = []
    envelopes = (
        [("maximum", maximum_envelope(baseline), None)]
        if args.maximum
        else [(f"{gib}gib", gib << 30, gib) for gib in args.cache_gib]
    )
    for label, envelope, gib in envelopes:
        result = {"cache_gib": gib, "capacity_selection": label}
        try:
            row, placements = plan(
                profile,
                counts,
                baseline,
                envelope,
                device=args.device,
                capacity=prepared_capacity,
                layer_pairs=config["arguments"]["layer_pairs"],
            )
            artifact = dict(
                identity=profile["identity"],
                placements=placements,
                termination=profile["termination"],
                converged=profile["converged"],
                construction=provenance,
            )
            artifact["hash"] = digest(artifact)
            path = args.output / f"profile-{label}.json"
            with path.open("x") as stream:
                json.dump(artifact, stream, indent=2)
            result.update(
                status="preflight_admitted",
                **row,
                profile=str(path),
                profile_hash=artifact["hash"],
                profile_sha256=sha256(path),
            )
        except ValueError as error:
            result.update(status="admission_rejected", reason=str(error))
        results.append(result)
    (args.output / "capacity-plan.json").write_text(
        json.dumps(
            dict(
                schema="b12x-capacity-plan/v1",
                provenance=provenance,
                capacities=results,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
