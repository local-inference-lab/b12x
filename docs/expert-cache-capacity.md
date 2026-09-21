# SM120 resident-capacity experiment

This experiment measures an intentionally constrained expert-memory envelope for
the supported [SM120 reference](expert-cache-reference.md). It holds the local
checkpoint, whole-K BF16/W4A16 arithmetic, KV reservation, context, graph capacity,
controlled admission and adaptive settings fixed. An envelope includes private
workspace and metadata; its size is not the fraction of expert payload resident.
No benefit from spending freed memory on additional KV or concurrency is measured.

Use the verified companion artifacts and explicit paths from the reference
guide. Run its bounded serving smoke before planning, and serialize timed runs
on an otherwise idle GPU. Preserve the calibration receipt, original profile and
calibration prompts together. Calibration and held-out evaluation must be disjoint.

The capacity planner verifies the calibrated artifact against its recorded
canonical counter snapshot. It calls the existing `profile_from_counts`
constructor with a different admitted resident count; it does not edit a profile
budget or reuse evaluation routes. Every capacity uses identical rankings and
counts, including deterministic expert-ID tie breaking. Profile construction
records the calibration and reference-receipt hashes. Non-cache reservations and
retained CPU-source bytes come from the completed reference run. The report also
charges observed graph/serving device use beyond the loader's pre-preparation
reservation. It adds only the unaccounted increment, including native allocations
and allocator pools, while retaining the fixed safety reserve. Live loader
admission remains authoritative, including changes in free memory.

```bash
python -m benchmarks.moe.expert_cache_capacity \
  --calibrated-profile "$CALIBRATED_PROFILE" \
  --calibration-receipt "$CALIBRATION_RECEIPT" \
  --reference-receipt "$REFERENCE_ADAPTIVE_RECEIPT" \
  --cache-gib 5 8 13 17 --output "$CAPACITY_PLANS"
```

Inspect `capacity-plan.json` before running anything. It reports per-layer
counts, payload fraction, device and host reservations, and rejected envelopes.
The fair incremental allocator uses the serving memory formulas and reserves
adaptive observation/health storage even for the paired static placement. It
does not spend hypothetical shared-workspace savings. A larger envelope stops
adding slots when all experts are resident; unused envelope bytes are not an
allocation or manufactured memory pressure.

For each admitted constrained capacity, run the existing acceptance runner:

```bash
python scripts/qualify_expert_cache.py --tier serving \
  --device-uuid "$GPU_UUID" --model "$MODEL" \
  --profile "$CAPACITY_PLANS/profile-${CACHE_GIB}gib.json" \
  --build-manifest "$BUILD_MANIFEST" \
  --prompts benchmarks/moe/fixtures/expert_health_chat_code.jsonl \
  --concurrency 4 --tokens 256 --pairs 3 --cache-gib "$CACHE_GIB" \
  --host-gib 40 --kv-gib 2 --output "$CAPACITY_RECEIPTS"
```

Each pair starts fresh engines and alternates arm order. Static has no observer.
Adaptive uses health-16, cold threshold 0.15, maximum interval 1024, decayed LFU,
32 pairs/128 MiB, two prepared pairs per layer, and no anchor recovery, routing
history or specialist protection. These are fixed research settings. A saturated
movement budget is an observation, not permission to retune a capacity cell.

For an admitted all-resident placement, add `--all-resident-static`. This scope
runs repeated static engines, checks that every expert is resident with no update
capacity, and requires identical output IDs across trials. It does **not** claim
adaptive acceptance: the serving adapter rejects adaptive models with no
nonresident layers. The all-resident cache still retains canonical mapped
backing, CPU sources and both tier workspaces. It is not an ordinary native
whole-K serving backend. The runner's ordinary ModelOpt smoke uses different
W4A4 arithmetic and is a lifecycle regression check, not a matched throughput
baseline.

At reference capacity only, `expert_cache_calibration_mixed.jsonl` supplies eight
independently authored general, code, math and multilingual prompts. Use
`--workload mixed-calibration --calibration-prompts ...` with a new profile path.
The runner generates 128 tokens per calibration prompt, matching the general
calibration effort. Retain both initializations and their different counts;
neither may consume held-out evaluation routes.

```bash
python -m benchmarks.moe.summarize_expert_cache_capacity \
  "$LOW_RECEIPTS/acceptance.json" "$REFERENCE_RECEIPTS/acceptance.json" \
  "$HIGH_RECEIPTS/acceptance.json" "$ALL_RESIDENT_RECEIPTS/acceptance.json" \
  --output "$CAPACITY_SUMMARY"
```

The summary retains individual engine trials, paired differences, stable and
specialist rates, latency distributions, lifecycle memory checkpoints, policy
backlog, control cost and final control tails. Its replication unit is an engine
trial. Static cold rates remain absent unless separately measured. Nested timers
and drains are not added again to serving wall time.

An optional counters-only run uses the existing harness with `--mode adaptive
--control observe --routing-diagnostics --admission together`. Keep the same
requests and execution settings; this diagnostic does not enter policy or move
experts. Pass its completed receipt as `--routing-receipt` and the exact measured
profile files as `--profiles` to the capacity summarizer. It requires matching
output IDs, immutable generation-zero maps, complete request boundaries and
monotonic canonical counts. The resulting cold fractions are counterfactuals
for that recorded route trace, separate from uninstrumented static timing.

Physical B300 acceptance remains the separate [native qualification](sm103-qualification.md).
Ripper's measured link is PCIe Gen4 x16. Its movement and direct-host costs do
not establish a Gen5 or Grace transport ceiling.
