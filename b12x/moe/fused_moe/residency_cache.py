"""SM103 native MXFP4 adapter for the shared host residency policy."""
from ..residency import (
    ResidencyCacheConfig, ResidencyCacheDecision, ResidencyCacheOutcome,
    ResidencyCacheController as _SharedController, ResidencyExchangeSpec,
    RoutingObservationSpec,
)
from .automatic import ResidencyLayerSpec
from ._routing_profile_tuning import RoutingProfileQuery


class ResidencyCacheController(_SharedController):
    """Preserve the fused-MoE constructor with backend-specific accounting.

    This adapter declares the existing exclusive HBM/Grace exchange mechanism.
    Preparation still verifies hardware, storage, recipe and journal admission.
    """
    def __init__(self, *, spec, config, counter_query, slots, baseline):
        if not isinstance(spec, ResidencyLayerSpec) or not isinstance(config, ResidencyCacheConfig):
            raise TypeError("cache policy requires typed layer and configuration contracts")
        if not isinstance(counter_query, RoutingProfileQuery):
            raise TypeError("cache policy requires a routing counter query")
        if (dict(counter_query.layers).get(spec.layer) != spec.experts
                or config.phase not in counter_query.phases):
            raise ValueError("cache policy requires the authoritative rank and declared layer/phase")
        self.spec, self.counter_query = spec, counter_query
        super().__init__(config=config,
            observations=RoutingObservationSpec(layer=spec.layer, experts=spec.experts,
                phase=config.phase, max_top_k=counter_query.max_top_k,
                sample_every=counter_query.sample_every, rank=counter_query.rank,
                owner_rank=counter_query.owner_rank),
            exchange=ResidencyExchangeSpec(backend="sm103_mxfp4_exclusive",
                direct_backing_execution=True, fixed_address_quiescent_exchange=True,
                # Journal both rows, then restore each payload into the opposite
                # tier. Read and publish the complete int32[E,2] map once each.
                payload_copy_bytes_per_pair=4*spec.expert_bytes,
                map_copy_bytes_per_transaction=2*spec.experts*8),
            slots=slots, baseline=baseline)
