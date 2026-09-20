"""Explicit research worker for scheduler/model-input causality diagnostics.

This module wraps the V2 runner only when selected as the worker extension.
CPU-only mode reads no device tensors and adds no graph nodes. Explicit
request selection also copies inputs, hidden states and logits; selected
module hooks record eager prefill intermediates after graph capture.
It is not imported by serving integration, and traced runs are not performance
evidence. Begin/end RPCs bound the recording independently of maintenance.
"""

from functools import wraps
from copy import deepcopy
import json
import hashlib
from pathlib import Path

from b12x.integration.vllm.residency_epoch import ResidencyEpochWorkerExtension


_active = False
_records = []
_limit = 0
_overflow = False
_device_records = []
_request_prefixes = ()
_output_limit = 0
_pending = None
_probe = None
_hooks = []
_module_names = ()
_initial_residency = None
_maintenance_records = []


def _copy_tree(value, *, host=False):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu() if host else value.detach().clone()
    if isinstance(value, dict):
        return {k: _copy_tree(v, host=host) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_copy_tree(v, host=host) for v in value]
    return (
        value
        if isinstance(value, (str, int, float, bool, type(None)))
        else type(value).__name__
    )


def _install():
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    original = GPUModelRunner.execute_model

    @wraps(original)
    def execute(self, scheduler_output, *args, **kwargs):
        global _overflow, _probe, _pending
        _pending = None
        _probe = (
            {}
            if (
                _active
                and len(_records) < _limit
                and any(n > 1 for n in scheduler_output.num_scheduled_tokens.values())
                and any(
                    req.startswith(prefix)
                    for req in scheduler_output.num_scheduled_tokens
                    for prefix in _request_prefixes
                )
            )
            else None
        )
        result = original(self, scheduler_output, *args, **kwargs)
        if not _active or kwargs.get("dummy_run", False):
            return result
        state = self.execute_model_state
        if state is None or not scheduler_output.total_num_scheduled_tokens:
            return result
        if len(_records) >= _limit:
            _overflow = True
            return result
        batch = state.input_batch
        record = {
            "step": len(_records),
            "requests": list(batch.req_ids),
            "state_indices": batch.idx_mapping_np.tolist(),
            "scheduled": batch.num_scheduled_tokens.tolist(),
            "computed": batch.num_computed_tokens_np.tolist(),
            "prefill_lengths": batch.prefill_len_np.tolist(),
            "prefilling": batch.is_prefilling_np.tolist(),
            "query_start": batch.query_start_loc_np.tolist(),
            "tokens": batch.num_tokens,
            "padded_tokens": batch.num_tokens_after_padding,
            "padded_requests": batch.num_reqs_after_padding,
            "full_graph": self.step_timing._batch[0],
            "new_requests": [
                {
                    "request": r.req_id,
                    "tokens": r.prompt_token_ids,
                    "blocks": r.block_ids,
                }
                for r in scheduler_output.scheduled_new_reqs
            ],
            "cached_requests": list(scheduler_output.scheduled_cached_reqs.req_ids),
            "new_blocks": scheduler_output.scheduled_cached_reqs.new_block_ids,
            "finished": sorted(scheduler_output.finished_req_ids),
        }
        _records.append(deepcopy(record))
        if any(
            any(req.startswith(prefix) for prefix in _request_prefixes)
            and computed - prompt < _output_limit
            for req, computed, prompt in zip(
                batch.req_ids,
                record["computed"],
                record["prefill_lengths"],
                strict=True,
            )
        ):
            _pending = {
                "step": record["step"],
                "input_ids": batch.input_ids.clone(),
                "positions": batch.positions.clone(),
                "seq_lens": batch.seq_lens.clone(),
            }
            if _probe is not None:
                _pending["modules"] = _probe
                metadata = state.attn_metadata
                if isinstance(metadata, dict) and metadata:
                    metadata = next(iter(metadata.values()))
                _pending["attention_metadata"] = (
                    _copy_tree(vars(metadata))
                    if hasattr(metadata, "__dict__")
                    else type(metadata).__name__
                )
                _pending["slot_mappings"] = _copy_tree(state.slot_mappings_by_layer)
            _device_records.append(_pending)
        _probe = None
        return result

    GPUModelRunner.execute_model = execute
    original_sample = GPUModelRunner.sample

    @wraps(original_sample)
    def sample(self, *args, **kwargs):
        if not _active or _pending is None:
            return original_sample(self, *args, **kwargs)
        compute = self.model.compute_logits

        def observed(hidden):
            logits = compute(hidden)
            _pending["hidden"] = hidden.clone()
            _pending["logits"] = logits.clone()
            return logits

        self.model.compute_logits = observed
        try:
            return original_sample(self, *args, **kwargs)
        finally:
            self.model.compute_logits = compute

    GPUModelRunner.sample = sample
    from vllm.model_executor.layers.fused_moe.b12x_cache import ModelOptNvFp4CacheMoE

    original_apply = ModelOptNvFp4CacheMoE.apply

    @wraps(original_apply)
    def apply(self, layer, x, topk_weights, topk_ids, *args, **kwargs):
        result = original_apply(self, layer, x, topk_weights, topk_ids, *args, **kwargs)
        if _probe is not None and any(
            self.prefix == n or self.prefix.startswith(n + ".") for n in _module_names
        ):
            _probe[self.prefix + "/routes"] = _copy_tree(
                {
                    "input": x,
                    "ids": topk_ids,
                    "weights": topk_weights,
                    "output": result,
                }
            )
        return result

    ModelOptNvFp4CacheMoE.apply = apply


def _residency(runner):
    model = getattr(runner, "b12x_expert_cache", None)
    if model is None:
        return None
    result = {}
    for name, plan in model.plans.items():
        state = plan.prepared.state
        mapping = state.mapping.cpu().numpy().tobytes()
        result[name] = {
            "map_sha256": hashlib.sha256(mapping).hexdigest(),
            "pointers": state.pointers(),
            "generation": state.updates.snapshot().generation if state.updates else 0,
        }
    return result


class ExecutionTraceWorker(ResidencyEpochWorkerExtension):
    def b12x_residency_maintenance(self, config):
        runtime = self._b12x_epoch_runtime()
        maintenance = getattr(runtime, "_local_maintenance", None)
        previous = {}
        if _active and maintenance is not None and maintenance.coordinator is not None:
            previous = {
                n: (c._baseline.counts, c._slots.expert_map)
                for n, c in maintenance.coordinator.controllers.items()
            }
        result = super().b12x_residency_maintenance(config)
        if previous:
            layers = {}
            for name, controller in maintenance.coordinator.controllers.items():
                counts, mapping = previous[name]
                delta = tuple(
                    b - a
                    for a, b in zip(counts, controller._baseline.counts, strict=True)
                )
                cold = [e for e, (tier, _) in enumerate(mapping) if tier == 1]
                hot = [e for e, (tier, _) in enumerate(mapping) if tier == 0]
                cold_count, total = sum(delta[e] for e in cold), sum(delta)
                layers[name] = {
                    "selections": total,
                    "cold_selections": cold_count,
                    "cold_fraction": cold_count / total if total else None,
                    "cold_experts_selected": sum(delta[e] > 0 for e in cold),
                    "cold_experts_selected_repeatedly": sum(delta[e] > 1 for e in cold),
                    "unprotected_score_gap": (
                        max(controller._scores[e] for e in cold)
                        - min(controller._scores[e] for e in hot)
                    )
                    if cold and hot
                    else None,
                }
            _maintenance_records.append(
                {
                    "after_step": len(_records),
                    "health": result["health"],
                    "layers": layers,
                }
            )
        return result

    def measure_idle_readback(self, samples=20):
        """Diagnostic lower bounds after traffic drains; no health decision is made."""
        from time import perf_counter_ns
        import torch

        counters = self.model_runner.b12x_expert_cache._counters
        if counters is None or not 1 <= samples <= 100:
            raise ValueError(
                "readback diagnostic requires prepared counters and 1..100 samples"
            )
        torch.cuda.synchronize()
        scalar = next(iter(counters.rows.values()))[:1]
        rows = []
        for _ in range(samples):
            start = perf_counter_ns()
            scalar.cpu()
            copied = perf_counter_ns()
            counters.snapshot(quiescent=True)
            rows.append(
                {
                    "scalar_read_ns": copied - start,
                    "snapshot_ns": perf_counter_ns() - copied,
                    "snapshot_stages_ns": counters.last_snapshot_timings_ns,
                }
            )
        return {
            "samples": rows,
            "scalar_bytes": 8,
            "snapshot_bytes": counters.storage.numel(),
            "idle_only": True,
            "scalar_is_health_summary": False,
        }

    def begin_execution_trace(
        self, limit=4096, request_prefixes=(), output_limit=0, modules=()
    ):
        global _active, _limit, _overflow, _request_prefixes, _output_limit
        global _module_names, _initial_residency
        if type(limit) is not int or limit <= 0:
            raise ValueError("execution trace capacity must be positive")
        if type(output_limit) is not int or not 0 <= output_limit <= limit:
            raise ValueError("selected output limit must fit the trace capacity")
        if _active:
            raise RuntimeError("execution trace already active")
        available = dict(self.model_runner.model.named_modules())
        for name in modules:
            if name not in available:
                raise ValueError(f"unknown diagnostic module: {name}")
        _records.clear()
        _device_records.clear()
        _maintenance_records.clear()
        _request_prefixes, _output_limit = tuple(request_prefixes), output_limit
        _module_names = tuple(modules)
        _initial_residency = _residency(self.model_runner)
        _limit, _overflow, _active = limit, False, True
        for name in modules:

            def before(module, args, kwargs, name=name):
                if _probe is not None:
                    _probe[name] = {
                        "args": _copy_tree(args),
                        "kwargs": _copy_tree(kwargs),
                    }

            def after(module, args, kwargs, output, name=name):
                if _probe is not None:
                    _probe[name]["output"] = _copy_tree(output)

            _hooks.append(
                available[name].register_forward_pre_hook(before, with_kwargs=True)
            )
            _hooks.append(
                available[name].register_forward_hook(after, with_kwargs=True)
            )
        return {
            "capacity": limit,
            "device_observation": bool(request_prefixes),
            "modules": {n: type(available[n]).__name__ for n in modules},
            "residency": _initial_residency,
        }

    def end_execution_trace(self, destination):
        global _active
        _active = False
        for hook in _hooks:
            hook.remove()
        _hooks.clear()
        path = Path(destination)
        with path.open("x") as stream:
            json.dump(
                {
                    "schema": 1,
                    "overflow": _overflow,
                    "steps": _records,
                    "initial_residency": _initial_residency,
                    "final_residency": _residency(self.model_runner),
                    "maintenance_pressure": _maintenance_records,
                },
                stream,
            )
        if _device_records:
            import torch

            torch.cuda.synchronize()
            with Path(str(path) + ".pt").open("xb") as stream:
                torch.save(_copy_tree(_device_records, host=True), stream)
        return {"path": str(path), "steps": len(_records), "overflow": _overflow}


_install()
