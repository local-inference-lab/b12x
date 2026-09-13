from dataclasses import replace

import pytest
import torch

from b12x._lib.platform import PlatformCapabilities
from b12x.comm.roce._transport import select_transport
from b12x._lib.architecture import UnsupportedArchitectureError

GRACE = PlatformCapabilities((10, 3), False, True, True, True, "aarch64")
SPARK = PlatformCapabilities((12, 1), True, True, True, True, "aarch64")


def test_station_transport_is_explicit_and_tp2_only():
    assert GRACE.grace_coherent
    assert select_transport(SPARK, world_size=4).backend == "spark_mapped"
    with pytest.raises(UnsupportedArchitectureError):
        select_transport(GRACE, world_size=2)
    with pytest.raises(UnsupportedArchitectureError):
        select_transport(GRACE, world_size=2, requested="grace_mapped")
    selection = select_transport(
        GRACE, world_size=2, requested="grace_mapped", experimental=True
    )
    assert selection.experimental and selection.memory == "mapped_host"
    with pytest.raises(UnsupportedArchitectureError):
        select_transport(
            GRACE, world_size=4, requested="grace_mapped", experimental=True
        )
    with pytest.raises(UnsupportedArchitectureError, match="not implemented"):
        select_transport(GRACE, world_size=2, requested="hbm_gdr", experimental=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("host_native_atomics", False),
        ("pageable_memory_access", False),
        ("uses_host_page_tables", False),
        ("cpu_architecture", "x86_64"),
        ("compute_capability", (12, 0)),
    ],
)
def test_coherency_does_not_follow_from_sm103_name(field, value):
    caps = replace(GRACE, **{field: value})
    assert not caps.grace_coherent
    with pytest.raises(UnsupportedArchitectureError):
        select_transport(
            caps, world_size=2, requested="grace_mapped", experimental=True
        )


def test_engram_storage_uses_existing_lookup_api_and_retains_owner():
    from b12x.sequence import engram
    from tests.sequence.test_engram import _plan

    plan = _plan(torch.device("cpu"), tokens=1)
    storage = engram.allocate_storage(plan)
    storage.weight_load_view.zero_()
    storage.scales_load_view.fill_(127)
    binding = engram.bind_lookup(
        plan,
        storage=storage,
        hash_ids=torch.zeros((1, 24), dtype=torch.int64),
        num_tokens=torch.ones(1, dtype=torch.int32),
        out=torch.empty((1, 6144), dtype=torch.bfloat16),
    )
    assert binding.storage is storage
    assert binding.weight is storage.weight
    assert storage.stats()["logical_bytes_per_token"] == 24 * 264
    assert storage.stats()["mapped_host_bytes"] == 0
    storage.close()
    with pytest.raises(RuntimeError, match="storage is closed"):
        engram.run_lookup(binding)
    with pytest.raises(ValueError, match="remain open"):
        engram.bind_lookup(
            plan,
            storage=storage,
            hash_ids=binding.hash_ids,
            num_tokens=binding.num_tokens,
            out=binding.out,
        )


def test_grace_engram_allocation_fails_before_allocating_without_coherency(monkeypatch):
    from b12x.sequence.engram import _storage
    from tests.sequence.test_engram import _plan

    plan = _plan(torch.device("cpu"), tokens=1)
    monkeypatch.setattr(
        _storage,
        "probe_platform",
        lambda device: replace(GRACE, host_native_atomics=False),
    )
    monkeypatch.setattr(
        _storage,
        "MappedHostAllocation",
        lambda *args: pytest.fail("allocated unqualified memory"),
    )
    with pytest.raises(NotImplementedError, match="coherent"):
        _storage.allocate_storage(plan, memory="grace")


@pytest.mark.parametrize(
    "factory,group_key",
    [
        ("from_exchange_group", "exchange_group"),
        ("from_process_group", "process_group"),
    ],
)
def test_roce_factories_preserve_explicit_station_selection(factory, group_key):
    from b12x.comm.roce.roce_oneshot import RoceOneshotAllReduce

    class ConstructorProbe(RoceOneshotAllReduce):
        def __init__(self, **kwargs):
            self.arguments = kwargs

    group = object()
    runtime = getattr(ConstructorProbe, factory)(
        **{group_key: group},
        device="cuda:0",
        transport="grace_mapped",
        experimental=True,
    )
    assert runtime.arguments["transport"] == "grace_mapped"
    assert runtime.arguments["experimental"] is True
    assert runtime.arguments["exchange_group"] is group
