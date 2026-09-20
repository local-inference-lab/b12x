"""Research-only counter reads at controlled, drained request boundaries.

Selecting this worker does not instrument graph replay. The benchmark must pause
submission before each read. Receipts containing these synchronous diagnostics
are observation evidence, not serving performance measurements.
"""

from dataclasses import asdict

from b12x.integration.vllm.residency_epoch import ResidencyEpochWorkerExtension


class RoutingSnapshotWorker(ResidencyEpochWorkerExtension):
    def diagnostic_routing_snapshot(self, *, quiescent=False):
        """Read existing counters without consuming a policy observation window."""
        if quiescent is not True:
            raise ValueError("routing diagnostic requires a completed engine pause")
        import torch

        runtime = self._b12x_epoch_runtime()
        runtime._validate()
        if runtime._stage != "idle":
            raise RuntimeError("routing diagnostic requires an idle transaction")
        before = {name: binding.snapshot() for name, binding in runtime.bindings.items()}
        torch.cuda.synchronize()
        snapshot = runtime.snapshot_counters(quiescent=True)
        runtime._validate()
        if before != {name: binding.snapshot() for name, binding in runtime.bindings.items()}:
            raise RuntimeError("residency changed during routing diagnostic")
        return {"snapshot": asdict(snapshot),
                "slots": {name: asdict(slots) for name, slots in before.items()}}
