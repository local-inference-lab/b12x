"""Repeated lifecycle acceptance keeps rank identity and release boundaries."""

import json

import pytest

from scripts import check_expert_cache_lifecycle as report


def evidence(tmp_path, monkeypatch, world):
    cycles = tmp_path / "cycles"
    cycles.mkdir()
    workers, clients = [], []
    for cycle in range(4):
        start = cycle * 100
        rows = [dict(kind="configuration", time_ns=start),
                dict(kind="shutdown", time_ns=start + 90)]
        (cycles / f"cycle-{cycle}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows))
        # Deliberately reverse arrival order. Different ranks may have
        # different stable runtime pools without leaking across cycles.
        for rank in reversed(range(world)):
            workers.append(dict(
                stage="after_worker_shutdown", time_ns=start + 80 + rank,
                rank=rank, tp_size=world, mapped_bytes=0, cpu_source_bytes=0,
                graph_owners=0, pending_health=False, health_host_bytes=0,
                allocated=(rank + 1) << 28, reserved=(rank + 1) << 29,
                device_free=(rank + 1) << 30, status={"VmRSS": f"{rank * 1024 + 100} kB"},
            ))
        clients.append(dict(status={"VmRSS": "100 kB"}, descriptors=12, children=["tracker"]))
    (cycles / "client-resources.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in clients))
    resources = tmp_path / "resources.jsonl"
    monkeypatch.setattr(report, "summarize", lambda _: dict(
        arguments={"tp_size": world}, epochs={"promotions": 2}))
    return cycles, resources, workers


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.mark.parametrize("world", [1, 2, 3, 4])
def test_lifecycle_checks_each_rank_across_cycles(tmp_path, monkeypatch, world):
    cycles, path, workers = evidence(tmp_path, monkeypatch, world)
    if world == 1:
        for row in workers:
            row.pop("rank")
            row.pop("tp_size")
    write(path, workers)
    result = report.check(cycles, path)
    assert result["tp_size"] == world and result["status"] == "passed"
    assert all(len(samples) == 4 for samples in result["samples"].values())


@pytest.mark.parametrize("fault", ["duplicate", "missing", "unidentified", "outside", "owner", "growth"])
def test_arbitrary_rank_failure_is_not_hidden_by_group_totals(tmp_path, monkeypatch, fault):
    cycles, path, workers = evidence(tmp_path, monkeypatch, 4)
    target = workers[-2]  # Rank 1 in the final cycle.
    if fault == "duplicate":
        target["rank"] = 2
    elif fault == "missing":
        workers.remove(target)
    elif fault == "unidentified":
        target.pop("rank")
    elif fault == "outside":
        target["time_ns"] = 9999
    elif fault == "owner":
        target["health_host_bytes"] = 128
    elif fault == "growth":
        target["allocated"] += 128 << 20
    write(path, workers)
    with pytest.raises(ValueError):
        report.check(cycles, path)
