"""Opt-in bounded phase timing; device events resolve after serving drains."""

from time import perf_counter_ns


def classify_iteration(scheduled, computed, prompt_lengths):
    if not len(scheduled) == len(computed) == len(prompt_lengths):
        raise ValueError("phase metadata has inconsistent request lengths")
    prompt, decode, chunks = 0, 0, []
    for count, before, length in zip(scheduled, computed, prompt_lengths, strict=True):
        count, before, length = int(count), int(before), int(length)
        if min(count, before, length) < 0:
            raise ValueError("phase token counts must be nonnegative")
        prefill = min(count, max(0, length - before))
        prompt += prefill
        decode += count - prefill
        chunks.append(prefill)
    phase = "mixed" if prompt and decode else "prefill" if prompt else "decode" if decode else "empty"
    return {"phase": phase, "prompt_tokens": prompt, "decode_tokens": decode,
            "prompt_chunks": chunks}


def observation_ranges(scheduled, computed, prompt_lengths):
    """Map scheduler rows to phases, including a request straddling its boundary."""
    classification = classify_iteration(scheduled, computed, prompt_lengths)
    ranges, offset = [], 0
    for count, prefill in zip(scheduled, classification["prompt_chunks"], strict=True):
        count = int(count)
        if prefill:
            ranges.append((offset, offset + prefill, 2))
        if count > prefill:
            ranges.append((offset + prefill, offset + count, 1))
        offset += count
    return tuple(ranges)


class PhaseTiming:
    """Preallocated events bracket actual model execution outside captured graphs.

    Mixed iteration durations remain indivisible. Rates use pure-prefill CUDA
    elapsed time, not client TTFT or a proportional allocation of mixed time.
    """

    def __init__(self, capacity):
        import torch
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("phase timing capacity must be positive")
        self.events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(capacity)]
        for pair in self.events:
            for event in pair:
                event.record()
        self.events[-1][-1].synchronize()
        self.records, self.pending, self.overflow = [], None, False

    def start(self, batch):
        if self.pending is not None:
            raise RuntimeError("model iteration did not complete its timing boundary")
        if len(self.records) == len(self.events):
            self.overflow = True
            return
        record = classify_iteration(batch.num_scheduled_tokens, batch.num_computed_tokens_np, batch.prefill_len_np)
        record.update(requests=list(batch.req_ids), scheduled=batch.num_scheduled_tokens.tolist(),
            computed=batch.num_computed_tokens_np.tolist(), host_submit_start_ns=perf_counter_ns())
        self.events[len(self.records)][0].record()
        self.pending = record

    def end(self):
        if self.pending is not None:
            self.events[len(self.records)][1].record()
            self.pending["host_submit_end_ns"] = perf_counter_ns()
            self.records.append(self.pending)
            self.pending = None

    def finish(self):
        if self.pending is not None or self.overflow:
            raise RuntimeError("phase timing is incomplete or exceeded its prepared capacity")
        if self.records:
            self.events[len(self.records) - 1][1].synchronize()
        for record, (start, end) in zip(self.records, self.events, strict=False):
            record["model_device_ms"] = start.elapsed_time(end)
        result = {"schema": 1, "scope": "model CUDA elapsed time; not client TTFT or scheduler wall time",
                  "iterations": self.records}
        self.events.clear()
        return result
