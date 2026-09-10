# GDN prefill

Status: **research-only**. The public API, shared CuTe implementation, independent
oracle, qualification tests, policy generators, and FlashInfer race are
implemented. The segment-parallel GDN path has GPU oracle and graph-replay
coverage on Max-Q RTX PRO 6000. GB10 qualification of that path and measured
embedded profile integration are incomplete. The prefill generators are not
registered in the catalog until every embedded device profile has measured
entries for both prefill components.

`b12x.sequence.gdn_prefill` computes the scalar-gated delta-rule recurrence over
packed sequences and a caller-owned recurrent-state pool. It consumes Q/K/V
after projection and convolution; output RMSNorm and output gating are external.
The recurrence output and state layout are suitable for continuation through
`b12x.sequence.gdn_decode`.

## Tensor contract

| Argument | Shape | Dtype / layout |
| --- | --- | --- |
| `q`, `k` | `[token_capacity, key_heads, 128]` | BF16, contiguous within each token |
| `v`, `output` | `[token_capacity, value_heads, 128]` | BF16, contiguous within each token |
| `a` | `[token_capacity, value_heads]` | Raw scalar decay projection; BF16, contiguous within each token |
| `b` | `[token_capacity, value_heads]` | Raw update projection; BF16, positive strides |
| `A_log`, `dt_bias` | `[value_heads]` | BF16 or FP32, contiguous |
| `recurrent_state` | `[max_state_slots, value_heads, 128, 128]` | FP32; `[slot, head, value, key]`; padded slot stride allowed |
| `cu_seqlens` | `[sequence_capacity + 1]` | Contiguous int32 |
| `initial_state_indices`, `final_state_indices`, `checkpoint_state_indices` | `[sequence_capacity]` | All int32 or all int64; final indices may have a positive stride |
| `checkpoint_offsets` | `[sequence_capacity]` | Contiguous int32; positive offsets must be multiples of 16 |
| `num_tokens`, `num_seqs` | `[1]` each | Contiguous int32 device scalars |

Require `value_heads == 3 * key_heads`. Input Q/K tensors are not expanded:
the prepare kernel maps value head `h` to key head `h // 3`. Positive token
strides permit views into packed projection buffers. Bound token and sequence
capacities must fit the plan; live counts come from the device scalars.

For a value head, the recurrence is

```text
alpha_t = exp(-exp(A_log) * softplus(a_t + dt_bias))
beta_t  = float32(bfloat16(sigmoid(b_t)))
S       = alpha_t * S
u       = beta_t * (v_t - S @ k_t)
S       = S + outer(u, k_t)
o_t     = S @ (scale * q_t)
```

Q/K normalization is `x / sqrt(sum(x*x) + eps)` when `qk_l2norm=True`.
`run` defaults to `scale=128**-0.5` and `eps=1e-6`. State math uses an FP32
master and BF16 matrix operands. The reference module contains both an
independent sequential oracle and a chunk mirror with the production rounding
points; neither is a production fallback.

Softplus uses `log1p(exp(a + dt_bias))` below its linear threshold, retaining
small positive values before multiplying by `exp(A_log)`. For example,
`a=-20`, `A_log=20`, and zero bias produce decay approximately `exp(-1)`.
Rounding `1 + exp(-20)` to one would incorrectly preserve the preceding state.
Prefill and both grouped and ungrouped decode state updates use this calculation.

## Segment-parallel recurrence

The `chunk_parallel` algorithm partitions sequences into capacity-planned
segments of 128, 256, 512, or 1024 tokens. Every segment retains the 16-token
BF16-operand/FP32-state recurrence. Separate transfer and zero-initial-state
summaries supply the incoming state for each segment; boundary propagation
keeps the transfer's diagonal center in FP32 and uses BF16 matrix operands
for its residual. The output pass starts from those propagated states.

With unsplit key columns, the zero-initial-state pass also computes output
and saves an FP32 state checkpoint at token 128 of each segment. During output
correction, a value group may reuse the remaining output only when every state
bit matches that checkpoint, including signed zero. A mismatch retains the
complete correction recurrence. This is an exact state comparison, not a decay
threshold, tolerance, or decay floor. Checkpoints before convergence are
corrected; checkpoints and final states after convergence come from the
identical local recurrence. All convergence states occupy preplanned scratch.

A transfer summary that becomes exactly zero can stop only if all remaining
transfer operands are finite. Boundary propagation skips an exactly zero
transfer product only when doing so preserves nonfinite-state behavior.
Pipeline termination drains already issued stages before releasing a CTA.
The sequential algorithm remains selectable for geometries where segment
summary and boundary overhead exceed their parallelism benefit.

## Planning, binding, and replay

Construct `Caps` with device, head geometry, `max_tokens`, `max_seqs`, and
`max_state_slots`, then call `plan(caps, policy=...)`. Allocate the returned
scratch specification and bind it with all input, metadata, output, and state
tensors. Call `prewarm(binding)` and a warm `run(binding)` before graph capture.
`run(binding, *, scale=None, eps=1e-6, max_live_tokens=None,
max_live_seqs=None)` returns the bound output tensor.

Plans resolve policy once and retain `policy_resolution`. Compile keys contain
recipe, static model geometry, planned capacity, and device identity. Live
counts, strides, window indices, and scalar math parameters are runtime launch
arguments. GDN and KDA have separate prepare specializations and share prologue,
workspace, and recurrence implementations under
`b12x/sequence/_shared/delta_prefill/`.

Each sequence reads one initial slot and writes one final slot. With
`checkpoint_export=True`, it may also write the state after a chunk-aligned
token offset. A null initial slot means zero state. Null output slots suppress
stores; a sequence spanning workspace windows requires a non-null final slot
to retain intermediate state. Empty sequences copy initial to final state.

Transactional metadata validation detects conflicting slot ownership,
malformed sequence intervals/counts, invalid slots, and invalid checkpoints.
An invalid invocation poisons live output and preserves the state pool; callers
inspect `binding.error_code`. Trusted validation mode requires valid metadata.
Mutable output, scratch, and state must not overlap each other or read-only
inputs. Pooled offsets multiply slot indices in int64.

## Qualification and performance evidence

`tests/sequence/test_gdn_prefill.py` covers the sequential oracle, strong decay
without cumulative-subtraction cancellation, chunk boundaries, 32K-token
sequences, packed views, checkpoint/null/in-place slots, device metadata replay,
frozen kernel resolution, poisoning, input immutability, decode continuation,
and state offsets beyond the signed int32 boundary. CPU checks alone do not
qualify the production GPU path. The large-slot test needs slightly more than
8 GiB of free device memory for a mostly uninitialized FP32 pool.

After selecting an assigned physical GPU, use the existing GDN benchmark:

```bash
.venv/bin/python benchmarks/benchmark_gdn_decode.py \
  --operation prefill --race flashinfer --mode both \
  --warmup 10 --iterations 100 --json /absolute/evidence/gdn-prefill.json
```

Decode remains the default operation. Prefill races all four head pairs
`(2,6)`, `(4,12)`, `(8,24)`, and `(16,48)` across single sequences of
16/64/256/1024/4096/8192/32768 tokens, batches `4*1024` and `8*512`, and ragged
lengths `[4096,1024,127,1]`. Checkpoint export is disabled in this shared race.

The timed boundary is raw projections and pooled initial state to recurrence
output and pooled final state. FlashInfer arms include benchmark-local CuTe
normalization, activation, metadata conversion, state gather, and state scatter.
The installed FlashInfer package remains unchanged. Separate arms call
`use_cp="auto"` and `use_cp=False`; evidence includes loaded source hashes and
the selected Python dispatch functions. The non-CP implementation's
`alpha + 1e-10` logarithm conversion is recorded without changing b12x math.

Measurements alternate complete arm order and reverse. The full initial pool
is restored before every invocation, with restoration timed separately.
Graph and eager modes collect warm and L2-flushed samples; the flush precedes
restoration. Thus the flushed result includes any cache warming from restoring
state. Each arm must pass the independent output/state oracle, replay poison,
address stability, allocation, and input immutability checks before timing.
Failed arms remain in JSON and produce a nonzero exit status. The ratio is
`FlashInfer_us / b12x_us`; larger than one favors b12x. JSON files are never
overwritten.

Use `--policy-profile /absolute/evidence/profile.json` to race a generated
profile before embedding it. This accepts a standard profile or generator
artifact, including gzip, records the file hash, and requires
`PREPLANNED_ONLY` coverage for every b12x case. Device mismatches and uncovered
queries fail instead of using the heuristic.

`benchmarks/benchmark_delta_prefill_regression.py --source-root TREE --json FILE`
uses the same input contract to run the public KDA path from an explicitly
selected source tree. Run it on a pristine comparison tree before collecting
the specialized tree's timings. An archive without Git metadata also requires
`--source-revision COMMIT`. Add `--compare BASELINE.json` to the second run;
every per-case median slowdown above 5% is flagged and requires a repeat on
the same GPU. Source hashes, GPU identity/mode, toolchain, correctness, and raw
graph samples accompany the result.

The policy generators enumerate 33 combinations of `v_split` 16/32/64/128,
`k_split` 1/2/4, and 2/3/4 stages, excluding configurations above 1024 threads.
Sequential candidates use half, base, and double prepared-record window
budgets, deduplicated at capacity boundaries. GDN also races all four segment
lengths for each launch geometry. Its config schema is version 2; both
generators use candidate contract version 5.
Planning also validates the exact recurrence shared-memory footprint against
SM12x's 99 KiB per-block limit, including bounded segment-band storage and
pipeline termination counters. The 128-row, two-way key split with four stages
exceeds that limit and remains visible as a rejected candidate in sweep evidence.
Window capacity uses the shared prepared-record byte budget. Both final-only
and checkpoint cases are measured, with distinct GDN/KDA component identities
and resumable candidate contracts. Measured profiles must cover each exact GPU
identity independently before catalog registration and embedded profile updates.
