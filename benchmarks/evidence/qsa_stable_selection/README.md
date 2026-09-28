# Exact QSA selection performance receipt

Status: **research-only**. This historical receipt records complete-QSA component
and Qwen3.8-Flash-Next serving measurements on NVIDIA DGX Spark. It supports the
selection-fusion performance claims under the stated conditions; it does not
GPU-qualify the complete standalone PR-head/master combination.

## Inspect the recorded measurements

Run from the B12X repository root; no GPU or model is needed:

```sh
python3 benchmarks/evidence/qsa_stable_selection/verify.py
```

The verifier checks artifact hashes, recomputes every component median and
serving throughput, reconstructs all 21 prompt files byte-for-byte, and prints
the comparison. It validates recorded arithmetic, not GPU correctness anew.

[receipt.json](receipt.json) identifies the exact commands, paths, source
revisions, worktrees, source-archive hashes, container images, physical GPU UUIDs,
operating-mode snapshots, runtime configuration, and correctness scope.

| Measurement | Raw evidence | Meaning |
| --- | --- | --- |
| TP2 complete QSA | [component_tp2.json](component_tp2.json) | Five cases, 30 samples per arm per case, in microseconds |
| TP4 complete QSA | [component_tp4.json](component_tp4.json) | Five cases, 30 samples per arm per case, in microseconds |
| TP4 serving control | [tp4_control.jsonl](tp4_control.jsonl) | Three repetitions of seven workloads |
| TP4 serving candidate | [tp4_candidate.jsonl](tp4_candidate.jsonl) | Three repetitions after a cached restart |
| TP2 serving candidate | [tp2_candidate.jsonl](tp2_candidate.jsonl) | Three repetitions; no matched TP2 serving control |
| Initial TP4 partial run | [tp4_initial_partial.jsonl](tp4_initial_partial.jsonl) | Every completed measured batch; excluded from restart means |

Each serving JSONL row retains the measured batch statistics, every request's
stream-event timestamps and token counts, server prefill duration, draft
acceptance counters, preemption counters, and source-result hash. Generated text
and token identities are omitted; no timing sample is dropped. Warmups are
excluded from the reported sample sets.

Component latency reduction is
`100 * (1 - median(candidate_us) / median(control_us))`; positive means faster.
Serving throughput change is
`100 * (mean(candidate_tokens_per_second) / mean(control_tokens_per_second) - 1)`;
positive means faster. Prefill divides input tokens by the server's prefill
duration. Decode divides emitted tokens by the common interval during which all
requests are decoding. C8 is aggregate throughput.

## Measured implementation and correctness

The serving comparison uses B12X control
`72baebbda2a200762f37223ef25aa64e8a5cb734` and candidate
`d2d5368d6c8a5cc43d79bfc8a4a5c58db584fc47`, both with vLLM
`76061de4bff2adc741cb25018ca79991263228be`. The measured B12X commits are reachable
from `codex/qwen-qsa-selection-20260917`. Immutable source archives exclude
uncommitted worktree contents. The selector production files in PR code commit
`e0ed66d89a1ee3b0f8d9108cf7ed5591550915d6` match the measured candidate exactly.

The paired component scripts call the real prepared QSA transaction through
`_contract._run`, using the same binding, scorer, attention program, and inputs
for both arms. The control decode function and stabilization kernels are loaded
from the control image; the candidate implementation is loaded from its source
archive. Both captured graphs include identical state restoration. Before any
timing, both arms must produce exact finite nonzero attention output, selected
positions, and persistent state without allocating during graph replay. Timing
uses CUDA events in A/B/B/A order with 15 samples per segment.

The [image test log](image-qsa-tests.log) records 17 passing targeted tests.
Six scalar-reference transaction cases also pass. Serving TP2 and TP4 each pass
26 exact shared-prefix/history-edit checks at 64K/C4 and 128K/C8. These are
bounded checks, not general model-quality evidence. TP4 component timing uses
candidate `7adadf451945cad662fedc83d92b6954431b0e82`; TP2 component timing and all
serving candidate measurements use `d2d5368d6c8a5cc43d79bfc8a4a5c58db584fc47`.
The component revisions share the selection algorithm; the latter also removes
unused helpers and supplies explicit tensor-device context handling.

## Reproduce complete-QSA measurements

The executed scripts are reproduced as [component_tp2.py](component_tp2.py)
and [component_tp4.py](component_tp4.py), with semicolon-separated statements
expanded onto separate lines. Their Python ASTs are identical to the executed
scripts; both original and formatted file hashes are recorded.
The full historical Docker commands are in `receipt.json` under
`component_cases.*.command_argv`; its mount paths identify the host workspaces.

Use an idle DGX Spark with the recorded ARM64 CUDA/PyTorch/B12X control image.
The control checkout must be available inside that image at
`/opt/spark-vllm/b12x`; `/candidate` must contain a clean archive of the selected
candidate commit. Its `benchmarks/benchmark_qsa.py` supplies the prepared
transaction harness. The image dependency and model stack are prerequisites;
this receipt does not redistribute either.

For TP2, mount an empty writable result directory at `/evidence`, the candidate
archive at `/candidate:ro`, and this receipt directory at `/receipt:ro`. Execute
the following command in that container, retaining the environment and cache
settings from the recorded Docker command:

```sh
PYTHONPATH=/candidate CUTE_DSL_ARCH=sm_121a VLLM_PLUGINS= \
  /opt/spark-vllm/.venv/bin/python -u /receipt/component_tp2.py
```

For TP4 geometry use `component_tp4.py` and its recorded candidate revision.
These are single-GPU component measurements of each topology's local geometry,
not distributed serving measurements. Output is `/evidence/fusion-results.json`.

## Reproduce serving measurements

[measure.py](measure.py) retains the original measurement functions verbatim.
Only workspace/output/endpoint configuration and prompt-helper loading differ;
the separate post-performance correctness subprocess is omitted. The helper
functions and constants in [prompt_helpers.py](prompt_helpers.py) are extracted
verbatim from the measured prompt generator. Their source file hash is recorded
because that external benchmark worktree contained local modifications.

Start the recorded model stack and settings before running this script. The
profile JSON files specify the expected image and rank containers for health
checks; they are measurement inputs, not deployment launchers. On another
installation, supply an equivalent profile with its actual host/container names
and immutable image identity. The original commands and working directory are
recorded in `receipt.json`; the reusable invocation is:

```sh
mkdir -p /tmp/qsa-serving/benchmark-results
python3 benchmarks/evidence/qsa_stable_selection/verify.py \
  --write-prompts /tmp/qsa-serving/matched-prompts
QSA_BENCH_ROOT=/tmp/qsa-serving QSA_BENCH_OUT=/tmp/qsa-serving \
QSA_BENCH_URL=http://maxwell:8000 \
  python3 benchmarks/evidence/qsa_stable_selection/measure.py \
  --depth 3 \
  --profile benchmarks/evidence/qsa_stable_selection/tp4_candidate_profile.json \
  --label candidate --repeats 3 --skip-shared
```

The script checks served input-token hashes against the restored prompts, uses
distinct cache salts, and checks fixed output lengths and rank health. Use the
corresponding recorded profile for the control or TP2. The historical control
command is reconstructed from its saved driver's invocation; candidate commands
have direct command artifacts.

## Limits of the claims

The three-repeat TP4 comparison gives +0.15% / +2.23% / +3.96% prefill throughput
at 8K / 64K / 128K. Complete-QSA component latency falls about 9–11% at 64K/128K.
These are different measurement scopes. TP2 serving values are absolute because
there is no matched serving control. Decode token-rate changes include different
draft acceptance and do not establish an isolated decode benefit.

The initial partial TP4 run has an intermittent lead-rank slowdown in unchanged
kernels. Its samples remain published separately. The condition clears after an
unchanged cached restart, but its cause and independence from this patch remain
unresolved. Mode/clock telemetry is sampled, not continuous; no fixed-clock
guarantee is claimed. Other GPU architectures are not performance-qualified.
