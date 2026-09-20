"""Bounded cumulative checkpoints on the routing-counter producer stream.

Checkpoints record evidence only. A quiescent consumer decodes retained cuts;
policy replay and placement publication remain outside this storage object.
"""

from time import perf_counter_ns
import torch


class RoutingHistoryState:
    def __init__(self, counters):
        self.counters = counters
        self.depth = counters.query.history_depth
        self.storage = torch.empty(
            (self.depth, counters.query.storage_bytes),
            dtype=torch.uint8,
            device=counters.device,
        )
        self.host = torch.empty_like(self.storage, device="cpu", pin_memory=True)
        self.slots = tuple(self.storage.unbind())
        self.host_slots = tuple(self.host.unbind())
        self.events = tuple(
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(self.depth)
        )
        self.read_started = torch.cuda.Event(enable_timing=True)
        self.read_done = torch.cuda.Event(enable_timing=True)
        self.epoch = self.generation = self.stream = None
        self.sequence = 0
        self.timestamps = [0] * self.depth

    def _stream(self):
        stream = torch.cuda.current_stream(self.counters.device)
        if self.stream is not None and self.stream != stream.cuda_stream:
            raise RuntimeError(
                "routing history requires its serialized producer stream"
            )
        self.stream = stream.cuda_stream
        return stream

    def _validate(self, generation):
        if self.epoch != self.counters.epoch or generation != self.generation:
            raise RuntimeError(
                "stale routing history: counter reset or placement generation changed"
            )

    def rebase(self, generation):
        self._stream()
        self.epoch, self.generation = self.counters.epoch, generation
        self.sequence = 0

    def checkpoint(self, generation):
        self._validate(generation)
        stream = self._stream()
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "history checkpoints require an explicit observation boundary"
            )
        index = self.sequence % self.depth
        started, done = self.events[index]
        started.record(stream)
        self.slots[index].copy_(self.counters.storage, non_blocking=True)
        done.record(stream)
        self.timestamps[index] = perf_counter_ns()
        self.sequence += 1
        return {
            "checkpoint": self.sequence,
            "retained": min(self.depth, self.sequence),
            "coalesced_checkpoints": max(0, self.sequence - self.depth),
        }

    def read(self, generation, *, quiescent=False):
        """Copy retained cuts after the engine has drained all counter writers.

        On wrap, the first retained cumulative cut includes the omitted prefix.
        Counts are conserved; omitted decay boundaries are explicitly coalesced.
        A full snapshot closes the final observation, so callers replay all but
        the newest checkpoint before evaluating that final snapshot.
        """
        if not quiescent or torch.cuda.is_current_stream_capturing():
            raise RuntimeError("routing history read requires engine quiescence")
        self._validate(generation)
        start = perf_counter_ns()
        count = min(self.depth, self.sequence)
        first = self.sequence - count
        if not count:
            return (), {
                "checkpoints": 0,
                "coalesced_checkpoints": 0,
                "readback_bytes": 0,
                "wall_ns": perf_counter_ns() - start,
            }
        stream = torch.cuda.current_stream(self.counters.device)
        with torch.cuda.stream(stream):
            self.read_started.record(stream)
            self.host.copy_(self.storage, non_blocking=True)
            self.read_done.record(stream)
        self.read_done.synchronize()
        copied = perf_counter_ns()
        snapshots, observations = [], []
        for sequence in range(first, self.sequence):
            index = sequence % self.depth
            snapshots.append(self.counters.decode_snapshot(self.host_slots[index]))
            started, done = self.events[index]
            observations.append(
                {
                    "checkpoint": sequence + 1,
                    "enqueued_ns": self.timestamps[index],
                    "copy_us": started.elapsed_time(done) * 1000,
                }
            )
        from b12x.moe.residency.contracts import validate_routing_progress

        validate_routing_progress(snapshots)
        return tuple(snapshots), {
            "checkpoints": self.sequence,
            "retained": count,
            "coalesced_checkpoints": max(0, self.sequence - self.depth),
            "observations": observations,
            "device_copy_bytes_per_checkpoint": self.counters.query.storage_bytes,
            "readback_bytes": self.host.numel(),
            "readback_us": self.read_started.elapsed_time(self.read_done) * 1000,
            "decode_ns": perf_counter_ns() - copied,
            "wall_ns": perf_counter_ns() - start,
        }
