# Deferred expert-cache policy history

Status: **implemented; research-only**. Prepared routing history retains cumulative counter
cuts between full maintenance operations. Recording a cut performs one
device-to-device copy; it does not run policy, drain serving, copy counters to
the host, or change placement. Static learned placement remains the default.
The BF16 router and placement-invariant whole-K MoE recipe are unchanged.
The [serving and depth measurements](expert-cache-history-results.md) distinguish
useful ranking information from measured throughput benefit.

## Observation, history and mutation

Continuous routing counters, history checkpoints, health probes and maintenance
have distinct responsibilities. Counters observe canonical expert IDs. History
records evidence. Health detects routing pressure. At an engine-owned maintenance
boundary, host policy interprets retained observations and the backend performs
bounded, quiescent movement.

The shared controller already expresses an observation without movement:

```python
decision = coordinator.observe(cut, slots=slots, allow_movement=False)
coordinator.finish(decision, slots=slots)
```

This path updates decayed scores, counts hits earned by already-resident promoted
experts, and ages residency guards for nonempty windows. It creates no candidate
pairs and publishes no generation. Empty windows neither decay scores nor age
guards. There is no second policy implementation or device replacement policy.
When maintenance has a pressure threshold, it evaluates the complete interval
before replay advances any policy baseline. A cold final subwindow does not
silently replace that full-interval gate.

Full maintenance consumes all retained cuts except the newest through this path.
The newest cut and the trailing work completed before maintenance form one final
window. That final observation can propose movement under the existing model-wide
pair and copy-byte budgets. Combining the tail avoids adding a tiny artificial
decay window merely because a utility response arrived after more model work.

Deferred replay is not equivalent to executing fixed maintenance at every cut.
Earlier windows cannot promote experts, earn hypothetical resident hits, or
change the map used to classify later selections. Policy scores and eligibility
can improve while fewer movement opportunities still limit adaptation.

## Bounded storage and wrap

`RoutingProfileQuery(history_depth=D)` declares a ring of D complete cumulative
counter slabs and an equally sized pinned readback ring. Zero is the default and
allocates no history. History requires owner-rank, decode-only observations; it
does not require a health reduction. The counter query schema is version 4.

For layer expert counts `E_l`, the existing counter slab size is:

```text
S = 16 + sum(8 * (2 * floor((E_l + 6) / 2)))
history device payload = D * S
history pinned-host payload = D * S
```

Slab rows and ring strides are aligned to 16 bytes. The model admission charges
both rings once, in addition to ordinary counters, health buffers and backend
storage. Allocator rounding, CUDA event objects and host bookkeeping still need
reserved headroom. At 48 layers with 128 experts, S is 51,472 bytes; depth 8 owns
411,776 bytes on each side. This geometry is an example, not an API contract.
Adding history can make an otherwise valid placement exceed its declared
envelope. Admission rejects that configuration; it does not silently reduce the
saved profile's resident set.

Wrap explicitly coalesces the omitted prefix. The oldest retained cumulative cut
includes every selection since the preceding maintenance baseline, but internal
decay boundaries in the omitted prefix are unavailable. Receipts report
`coalesced_checkpoints`; they do not report those missing boundaries as replayed
windows. No selections are dropped. Increasing depth preserves more recent
boundaries at the cost of storage and host replay time.

This is a bounded approximation, not archival tracing or an exact reconstruction
of arbitrary long-term LFU history. The maximum full-snapshot interval remains a
separate experimental control. Depth 1 preserves no extra policy boundary when
the final checkpoint is combined with the maintenance tail.

## Ownership and synchronization

Preparation owns the rings, fixed views, copy events, counter programs and source
counter slab. Priming executes the counter operation and every checkpoint slot's
copy/event path. Recording uses the same serialized producer stream as counter
writers. It adds no kernel specialization, per-route atomic or graph node.
Checkpoints are explicit control operations outside CUDA graph capture/replay.

The ring belongs to one counter reset epoch and a tuple of preparation IDs and
placement generations. A checkpoint rejects a stale identity. Full maintenance
drains writers before copying the ring to its prepared pinned buffer. The host
reads that buffer only after its copy event completes. Every retained cumulative
cut, including the final checkpoint and maintenance tail, must advance
monotonically. Counter overflow, a reset, changed owner/layer geometry or a stale
generation fails closed.

Successful maintenance rebases the ring before submissions resume, including
maintenance that moves no experts. Existing uncertain-update, stale-pointer,
victim-restoration and poisoned-preparation rules remain unchanged. History is
not a second mutation protocol. Concurrent producers sharing a ring, distributed
history aggregation and independent serving lanes sharing a counter plan are
unsupported.

## Serving integration

`ExpertCacheServingConfig(history_depth=D)` enables admitted history storage in
adaptive mode. Static/profile modes reject a nonzero depth. The existing worker
utility `b12x_residency_checkpoint` can record a cut independently of health.
The engine owner supplies its cadence and serializes it with maintenance.

For a shared health/history trigger, prepare both options and use:

```python
health = VllmResidencyHealth(engine, record_history=True)
await maintenance.run()  # Establish policy, health and history baselines.
async with control_lock:
    result = await health.probe()
    if thresholds.assess(result["summary"])["health"] == "pressure":
        await maintenance.run()
```

This adds the checkpoint to the existing start utility; no additional round trip
is needed. Recording history does not make a health probe advance policy state.
Health without history continues to allocate no ring. The serving experiment
shares the periodic trigger for convenience; the public storage contract does
not require equal checkpoint and health intervals.

The benchmark enables history explicitly:

```bash
python -m benchmarks.moe.expert_cache_serving \
  --model /models/Qwen3-30B-A3B-NVFP4 \
  --mode adaptive --control health --history-depth 8 \
  --profile placement.json \
  --prompts benchmarks/moe/fixtures/expert_health_prose_math.jsonl \
  --output history-serving.jsonl --admission together \
  --concurrency 8 --tokens 128 --epoch-tokens 32 \
  --health-max-tokens 1024 --cold-threshold 0.15 \
  --epoch-pairs 16 --epoch-mib 64
```

Depth, cadence and thresholds remain experiment settings. `--policy-diagnostics`
retains full counts, scores, proposed/selected pairs and hit accounting at each
logical observation. Its serialization cost belongs in diagnostic runs, separate
from timing arms without that option. Maintenance receipts distinguish full
interval selection totals from the shorter windows replayed into policy.

## Qualification boundaries

The maintained companion's serialized worker utilities and scheduler maintenance
boundary provide the implemented single-worker lifecycle. No companion scheduler
or completion-output changes are required. Source-built paired serving uses
controlled admission and exact token equality; differing ordinary BF16 execution
shapes are not interchangeable numerical controls.

History storage uses portable CUDA copies and the existing counter programs.
SM103 cross-compilation does not qualify physical Grace-backed execution. B300
acceptance still starts with all-HBM, all-Grace and mixed native correctness,
Grace TMA legality, same-graph updates and native sanitizers before adaptive
serving. PCIe Gen4 SM120 measurements do not predict Gen5 or Grace transport.
