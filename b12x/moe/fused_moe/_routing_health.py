"""Prepared, single-producer health readback with one retained result slot.

The owner serializes launches with counter writers and placement updates on the
same stream. Event polling never drains unrelated streams. Health baselines are
independent of policy snapshots and must be rebased after map changes.
"""

from time import perf_counter_ns
import torch


class RoutingHealthState:
    def __init__(self, counters):
        self.counters = counters
        q, device = counters.query, counters.device
        self.layers = tuple(n for n, _ in q.layers)
        self.anchor = None
        self.anchor_mask = (torch.zeros(sum(e for _, e in q.layers),
                                       dtype=torch.uint8, device=device)
                            if q.anchor_summary else None)
        self.previous = torch.zeros(
            sum(e for _, e in q.layers), dtype=torch.uint64, device=device
        )
        self.dummy_map = torch.zeros(
            (self.previous.numel(), 2), dtype=torch.int32, device=device
        )
        self.descriptors = torch.empty(
            (len(self.layers), 5 if q.anchor_summary else 4), dtype=torch.int64, device=device
        )
        self.output = torch.empty(
            (len(self.layers), 7 if q.anchor_summary else 6), dtype=torch.uint64, device=device
        )
        self.host = torch.empty_like(self.output, device="cpu", pin_memory=True)
        self.started = torch.cuda.Event(enable_timing=True)
        self.reduced = torch.cuda.Event(enable_timing=True)
        self.done = torch.cuda.Event(enable_timing=True)
        self.pending = False
        self.epoch = None
        self.generation = None
        self.stream = None
        self.owners = ()
        maps, offset = {}, 0
        for name, experts in q.layers:
            maps[name] = self.dummy_map[offset : offset + experts]
            offset += experts
        self.bind_maps(maps)

    def bind_maps(self, maps):
        if self.pending:
            raise RuntimeError("health readback is pending")
        if set(maps) != set(self.layers):
            raise ValueError("health maps must cover the declared layers")
        rows, offset = [], 0
        for name, experts in self.counters.query.layers:
            mapping = maps[name]
            if (
                mapping.shape != (experts, 2)
                or mapping.dtype != torch.int32
                or mapping.device != self.counters.device
                or not mapping.is_contiguous()
            ):
                raise ValueError(
                    "health map differs from prepared geometry/device/layout"
                )
            rows.append(
                (
                    self.counters.rows[name, "decode"].data_ptr(),
                    mapping.data_ptr(),
                    self.previous.data_ptr() + offset * 8,
                    experts,
                ) + ((self.anchor_mask.data_ptr() + offset,)
                     if self.anchor_mask is not None else ())
            )
            offset += experts
        self.owners = tuple(maps[n] for n in self.layers)
        self.descriptors.copy_(
            torch.tensor(rows, dtype=torch.int64), non_blocking=False
        )
        self.epoch = None
        self.stream = None

    def bind_anchor(self, anchor):
        """Install one validated learned profile before serving; never replace it."""
        from b12x.moe.residency.anchor import ResidencyAnchor
        if self.anchor_mask is None or self.pending or self.anchor is not None:
            raise RuntimeError("anchor is unprepared, pending or already bound")
        if not isinstance(anchor, ResidencyAnchor):
            raise TypeError("health anchor requires a validated learned profile")
        placements = dict(anchor.placements)
        if set(placements) != set(self.layers) or any(
            placements[n].total_experts != e for n, e in self.counters.query.layers
        ):
            raise ValueError("anchor geometry differs from routing counters")
        mask = []
        for name, experts in self.counters.query.layers:
            resident = set(placements[name].resident_expert_ids)
            mask.extend(int(e not in resident) for e in range(experts))
        self.anchor_mask.copy_(torch.tensor(mask, dtype=torch.uint8))
        self.anchor = anchor
        self.epoch = None

    def _launch(self, baseline):
        import cutlass
        import cuda.bindings.driver as cuda
        from b12x.moe._shared.kernels.sm103.launch import pointer

        stream = torch.cuda.current_stream(self.counters.device)
        if self.stream is not None and stream.cuda_stream != self.stream:
            raise RuntimeError(
                "health reduction requires its serialized producer stream"
            )
        self.stream = stream.cuda_stream
        self.counters.programs["health"](
            pointer(cutlass.Int64, self.descriptors),
            pointer(cutlass.Uint64, self.output),
            cutlass.Int32(len(self.layers)),
            cutlass.Int32(baseline),
            cuda.CUstream(stream.cuda_stream),
        )

    def rebase(self, generation):
        if self.pending:
            raise RuntimeError("consume pending health result before rebasing")
        self._launch(1)
        self.epoch, self.generation = self.counters.epoch, generation

    def start(self, generation):
        if self.pending:
            raise RuntimeError("health readback already pending")
        self._validate(generation)
        if self.anchor_mask is not None and self.anchor is None:
            raise RuntimeError("learned anchor has not been bound")
        self.wall_started = perf_counter_ns()
        self.started.record()
        self._launch(0)
        self.reduced.record()
        self.host.copy_(self.output, non_blocking=True)
        self.done.record()
        self.pending = True
        self.enqueue_wall_ns = perf_counter_ns() - self.wall_started

    def _validate(self, generation):
        if self.epoch != self.counters.epoch or self.generation != generation:
            raise RuntimeError(
                "stale health baseline: rebase after counter reset or map change"
            )

    def poll(self, generation):
        self._validate(generation)
        if not self.pending:
            raise RuntimeError("health readback has not been submitted")
        if not self.done.query():
            return None
        rows = self.host.tolist()
        self.pending = False
        if any(r[-1] for r in rows):
            self.epoch = None
            raise ValueError(
                "invalid health counters/map: reset, overflow or stale data"
            )
        layers = {
            n: dict(
                selections=r[0],
                cold_selections=r[1],
                cold_experts_once=r[2],
                cold_experts_repeated=r[3],
                repeated_cold_selections=r[4],
                **({"anchor_cold_selections": r[5]} if self.anchor is not None else {}),
            )
            for n, r in zip(self.layers, rows, strict=True)
        }
        total = sum(r[0] for r in rows)
        cold = sum(r[1] for r in rows)
        return dict(
            layers=layers,
            **({"anchor_profile": self.anchor.profile_id,
                "anchor_cold_selections": sum(r[5] for r in rows)}
               if self.anchor is not None else {}),
            selections=total,
            cold_selections=cold,
            cold_fraction=cold / total if total else None,
            generation=self.generation,
            counter_epoch=self.epoch,
            reduction_us=self.started.elapsed_time(self.reduced) * 1000,
            copy_us=self.reduced.elapsed_time(self.done) * 1000,
            enqueue_wall_ns=self.enqueue_wall_ns,
            completion_wall_ns=perf_counter_ns() - self.wall_started,
            readback_bytes=self.host.numel() * self.host.element_size(),
        )
