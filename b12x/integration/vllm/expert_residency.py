"""Explicit worker hooks for engine-owned calibration and restart coordination.

This module imports no vLLM internals and installs no patches. A worker supplies
canonical weight plans, memory reservations, phase labels and TP ownership. It
submits the returned request to its existing PreparationSession and records
bound observers immediately after routing in its calibration graphs.
"""
from __future__ import annotations

from b12x.moe.fused_moe.automatic import PHASES, ResidencyController
from b12x.moe.fused_moe.routing_profile import (
    bind_routing_profile, plan_routing_profile, routing_profile_state,
)


class ExpertResidencyWorker:
    """One model lane with one authoritative replicated-TP routing observer."""
    def __init__(self, controller: ResidencyController, *, rank=0, tp_size=1):
        self.controller, self.rank, self.tp_size = controller, rank, tp_size
        self.counter_plan = None

    def startup(self):
        progress = self.controller.startup()
        query = self.controller.profiler_query(rank=self.rank, tp_size=self.tp_size)
        if query is not None:
            self.counter_plan = plan_routing_profile(query)
        return progress

    def preparation_request(self, *, name="moe.routing_profile"):
        """Return a real counter primer, or None when the serving lane is off."""
        if self.counter_plan is None:
            return None
        from b12x.preparation import PreparedCall
        import torch
        query = self.counter_plan.query
        def prime(state):
            bindings = []
            for layer, _ in query.layers:
                for phase in query.phases:
                    for dtype in (torch.int32, torch.int64):
                        ids = torch.zeros((query.max_tokens, query.max_top_k), dtype=dtype, device=state.device)
                        bindings.append(state.bind(layer=layer, phase=phase, topk_ids=ids))
            def run():
                for binding in bindings:
                    binding.run()
            return PreparedCall(run=run, owners=(state, *bindings))
        return self.counter_plan.request(name=name, prepare_call=prime)

    def bind_routes(self, *, layer, phase, topk_ids):
        """Bind at graph construction; None means no observer node is recorded."""
        if self.counter_plan is None:
            return None
        if phase not in PHASES:
            raise ValueError("the engine must supply an explicit supported routing phase")
        if layer not in dict(self.counter_plan.query.layers):
            raise ValueError("routing layer was not declared for this worker")
        if phase not in self.counter_plan.query.phases:
            return None
        return bind_routing_profile(self.counter_plan, layer=layer, phase=phase, topk_ids=topk_ids)

    def begin_calibration(self, *, quiescent=False):
        if self.counter_plan is None:
            return False
        if self.controller.progress.state not in ("calibrating", "monitoring"):
            raise RuntimeError("completed calibration requires a new controller and preparation lifecycle")
        state = routing_profile_state(self.counter_plan)
        state.set_enabled(False, quiescent=quiescent)
        state.reset(quiescent=quiescent)
        state.set_enabled(True, quiescent=quiescent)
        return self.rank == self.controller.owner_rank

    def snapshot_counters(self, *, quiescent=False):
        if self.counter_plan is None:
            raise RuntimeError("worker has no prepared routing observer")
        return routing_profile_state(self.counter_plan).snapshot(quiescent=quiescent)

    def poll(self, *, request_count=0, token_count=0, quiescent=False):
        """Engine control-plane hook; callers pause all graph producers first."""
        if self.counter_plan is None or self.rank != self.controller.owner_rank:
            return None
        if self.controller.progress.state not in ("calibrating", "monitoring"):
            return self.controller.progress
        snapshot = self.snapshot_counters(quiescent=quiescent)
        if self.controller.progress.state == "monitoring":
            return self.controller.monitor(snapshot)
        progress = self.controller.observe(snapshot, request_count=request_count, token_count=token_count)
        if progress.state != "calibrating":
            routing_profile_state(self.counter_plan).set_enabled(False, quiescent=quiescent)
        return progress

    def end_calibration(self, *, quiescent=False):
        if self.counter_plan is not None:
            routing_profile_state(self.counter_plan).set_enabled(False, quiescent=quiescent)

    def inspect(self):
        """Return immutable state; activation/restart remains the engine's job."""
        return self.controller.progress
