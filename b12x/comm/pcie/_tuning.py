"""Singleton declarations for caller-owned PCIe communication surfaces.

Each surface retains its existing communicator, transport precision, launch
settings and channel ownership. This contract does not select interchangeable
collective backends or qualify a topology from a visible-device count.
"""

from dataclasses import dataclass, replace

from b12x.preparation import BackendConfig, FrozenMapping, make_fixed_contract


# Native execution surfaces after the all-reduce manager's plan-time routing.
# Values are supported worlds, never a set of transports to race.
SURFACES = {
    "OneshotAllReduce.all_reduce": (2, 4, 6, 8, 10),
    "OneshotAllReduce.all_reduce_fused_add_rms_norm": (2, 4, 6, 8, 10),
    "OneshotAllReducePool.all_reduce": (2, 4, 6, 8, 10),
    "OneshotAllReducePool.all_reduce_fused_add_rms_norm": (2, 4, 6, 8, 10),
    "DmaAllReduce.all_reduce": (2, 4, 6, 8, 10),
    "PCIeTwoShotBF16.all_reduce": (4,),
    "PCIeTwoShotBF16.reduce_scatter": (4,),
    "PCIeTwoShotBF16.all_gather": (4,),
    "TwoShotReduceScatter.reduce_scatter_fp8": (2, 4, 8),
    "TwoShotReduceScatter.all_gather_fp8": (2, 4, 8),
    "DcpAllToAll.lse_reduce_scatter": (2, 4, 8, 16),
    "DcpAllToAll.all_gather_heads": (2, 4, 8, 16),
    "DcpAllToAll.all_gather_pair": (2, 4, 8, 16),
    "DcpAllToAll.all_gather_pair_kimi_topk": (2, 4, 8, 16),
    "DcpAllToAll.kimi_topk16": (2, 4, 8, 16),
    "DcpAllToAllPool.lse_reduce_scatter": (2, 4, 8, 16),
    "DcpAllToAllPool.all_gather_heads": (2, 4, 8, 16),
    "DcpAllToAllPool.all_gather_pair": (2, 4, 8, 16),
    "DcpAllToAllPool.all_gather_pair_kimi_topk": (2, 4, 8, 16),
    "DcpAllToAllPool.kimi_topk16": (2, 4, 8, 16),
    "DcpTopKOwnerExchange.stage_candidates": (2, 3, 4, 6, 8),
    "PagedKvReplica.replicate": (4,),
    "VocabParallelArgmax.fused_add_argmax": (8, 12, 16),
    "PCIeHierarchicalAllReduce.all_reduce": (12, 16),
    "PCIeIslandRSAllReduce.all_reduce": (16,),
    "kimi_topk16": (1,),
}


@dataclass(frozen=True, kw_only=True)
class PcieQuery:
    surface: str
    world_size: int
    rank: int
    topology: str
    # Capacity, tensor geometry/dtypes, codec, channel, and existing host launch
    # overrides are concrete caller metadata, never alternative tuning labels.
    call: FrozenMapping
    setup: FrozenMapping


PcieConfig = BackendConfig
TUNING = make_fixed_contract(
    component_id="comm.pcie",
    query_type=PcieQuery,
    backend="native",
)
_validate_fixed_query = TUNING.validate_query


def _validate_query(query: PcieQuery, device) -> None:
    _validate_fixed_query(query, device)
    worlds = SURFACES.get(query.surface)
    if worlds is None:
        raise ValueError(f"unknown PCIe execution surface {query.surface!r}")
    if query.world_size not in worlds:
        raise ValueError(f"{query.surface} supports world sizes {worlds}")
    if not 0 <= query.rank < query.world_size:
        raise ValueError("rank must belong to the caller's process group")
    if query.surface == "kimi_topk16":
        if query.topology != "local":
            raise ValueError("stateless kimi_topk16 uses one local device")
    elif query.topology != "pcie_ipc":
        raise ValueError(
            "PCIe collectives require caller-established CUDA IPC topology"
        )
    if query.surface == "PagedKvReplica.replicate":
        call = query.call
        if (call["ratio"] not in (1, 2) or call["page_size"] not in (128, 256)
                or type(call["stripe"]) is not int or call["stripe"] <= 0
                or call["page_size"] % call["stripe"]):
            raise ValueError("unsupported DS4.1 replica compression/page/stripe geometry")
        if not 0 < call["max_tokens"] <= query.setup["max_tokens"]:
            raise ValueError("replica declaration exceeds the channel's token capacity")
        cycles, tail = divmod(call["max_tokens"], call["stripe"] * query.world_size)
        max_owned = cycles * call["stripe"] + min(tail, call["stripe"])
        if max_owned > query.setup["local_capacity"]:
            raise ValueError(
                "replica stripe geometry exceeds the channel's owner capacity"
            )
    if query.surface.endswith("all_gather_heads"):
        push = query.call.get("peer_write", False)
        if type(push) is not bool:
            raise ValueError("DCP head gather peer_write must be a bool")
        if push and (
            query.world_size != 4
            or query.call["dtype"] != "torch.bfloat16"
            or query.setup["total_heads"] != 64
            or query.setup["query_head_dim"] != 512
        ):
            raise ValueError("posted-write DCP head gather requires TP4 BF16 64x512 heads")


TUNING = replace(TUNING, query_schema_version=5, validate_query=_validate_query)

__all__ = ["PcieQuery", "PcieConfig", "SURFACES", "TUNING"]
