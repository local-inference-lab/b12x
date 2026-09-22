# SM120 experimental reference

Status: **experimental**. Learned-static placement is the deployment baseline.
Adaptive residency is an explicit experiment; its benefit depends on routing
locality and workload drift. This guide qualifies the cache and engine lifecycle.
It does not select another replacement policy.

## Supported lane

| Configuration | Status and boundary |
|---|---|
| Qwen3-30B-A3B ModelOpt NVFP4, single SM120 GPU, V2 runner, TP/DP/PP=1 | Physically tested; use the supplied local checkpoint and checkpoint-bound profile |
| BF16 activations, native W4A16 whole-K, actual top-k IDs/weights, greedy decode | Required explicit recipe; not W4A4-equivalent |
| CPU routed source parameters, fixed VRAM slots, full cacheable pinned/mapped canonical backing | Implemented and physically tested; no full-expert GPU load peak |
| Learned static; health-triggered decayed-LFU adaptation; optional anchor recovery | Physically tested experimental modes; static allocates no adaptive counters or control state |
| Short routing history and specialist retention | Research-only; disabled in reference arms |
| EP, DP/PP expansion, sequence/context parallelism, DBO, LoRA, speculation, non-SiLU/bias variants, input route weighting | Unsupported by this lane and rejected |
| Qwen3-Next-80B NVFP4, pinned revision, SM120 TP1/TP2 | [Hybrid qualification](hybrid-inference-results.md); separate rank-local host envelopes, learned profiles and admitted resident geometry |
| Other ModelOpt MoE checkpoints satisfying declared geometry/scales | Loader contracts exist; no general model-family qualification |
| SM103 HBM/Grace native expert execution | Separate implemented prototype; requires physical B300 gates |
| Native SM103 CPU-source full-model cache serving | Unsupported by the SM120 loader; an MXFP4/MXFP8 source/preparation adapter remains necessary |
| Pageable canonical storage plus bounded pinned staging | Not implemented; direct-host miss execution requires mapped backing |

The profile is immutable and bound to checkpoint contents, geometry, resident
capacity, workload identity, and the numerical recipe
`nvfp4_w4a16_whole_k_weighted_bf16_ordered_sum`. Unequal gate/up global scales and
incompatible profiles fail rather than being silently converted. See
[storage and admission](expert-cache-serving.md) for exact accounting.

## Source-matched environment

Use isolated environments and explicit paths. `MODEL` is an existing local
checkpoint; these commands do not download or redistribute weights. Choose new
artifact and receipt directories. The companion branch is
`local-inference-lab/vllm:codex/b12x-expert-cache`; record both exact commits with
`git rev-parse HEAD` and retain `git status --porcelain` before building.

Build prerequisites are Python 3.12, a CUDA development toolkit compatible with
Torch 2.13, a C++ toolchain, CMake/Ninja, Rust/Cargo, and libdw/elfutils development
headers. The recorded environment uses CUDA 13.3, CUDA bindings 13.0.3, and
CUTLASS DSL 4.6.2. Install companion runtime and build requirements into the
isolated environment, then b12x with its pinned CUTLASS packages. Do not overlay
Python sources onto a wheel from another companion revision.

```bash
uv venv --python 3.12 "$ENV_DIR"
. "$ENV_DIR/bin/activate"
uv pip install -r "$VLLM_SOURCE/requirements/cuda.txt" \
  -r "$VLLM_SOURCE/requirements/build/cuda.txt"
uv pip install -e "$B12X_SOURCE[dev]" "cuda-python==13.0.3"
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=16 NVCC_THREADS=2
bash "$B12X_SOURCE/scripts/build_expert_cache_companion.sh" \
  "$VLLM_SOURCE" "$WHEEL_DIR" "$BUILD_LOG"
uv pip install --no-deps "$VLLM_WHEEL"
python "$B12X_SOURCE/scripts/expert_cache_artifact.py" \
  --wheel "$VLLM_WHEEL" --source "$VLLM_SOURCE" --build-log "$BUILD_LOG" \
  --output "$BUILD_MANIFEST" --verify-installed
```

The build disables precompiled CUDA and Rust downloads. Its log retains package
versions and toolchains. Artifact verification compares packaged Python against
source, installed files against the wheel, and loaded modules/native libraries
against their wheel hashes. The acceptance worker repeats loaded-library checks
after model initialization. A version suffix alone is not build identity.
Retain the source export or clean checkout alongside the build log and wheel;
the manifest does not independently attest a third-party compiler invocation.

The environment must pass import checks before loading a checkpoint. If an
installed dependency's CUDA bindings differ from the qualified environment,
resolve that version mismatch explicitly and preserve the dependency lock; do
not silently mount a different library into a headline qualification run.

## Acceptance

Run from the b12x checkout. All output directories must be new. Host acceptance
needs no GPU or companion installation:

```bash
uv venv --python 3.12 "$HOST_ENV"
uv pip sync --python "$HOST_ENV/bin/python" --torch-backend cpu --require-hashes \
  ci/expert_cache/requirements-host.lock
"$HOST_ENV/bin/python" scripts/qualify_expert_cache.py \
  --tier host --output "$HOST_RECEIPTS"

python scripts/qualify_expert_cache.py --tier gpu --device-uuid "$GPU_UUID" \
  --model "$MODEL" --output "$GPU_RECEIPTS"

python scripts/qualify_expert_cache.py --tier serving --device-uuid "$GPU_UUID" \
  --model "$MODEL" --profile "$PROFILE" --build-manifest "$BUILD_MANIFEST" \
  --calibration-prompts benchmarks/moe/fixtures/expert_cache_calibration.jsonl \
  --prompts benchmarks/moe/fixtures/expert_health_chat_code.jsonl \
  --concurrency 4 --tokens 256 --pairs 3 --output "$SERVING_RECEIPTS"
```

Calibration requires a new profile path and prompts disjoint from evaluation.
Omit `--calibration-prompts` to use an explicitly supplied validated profile.
The serving runner checks ordinary non-cache serving, then alternates static and
adaptive order across three pairs. Exact paired token equality, captured graph
and cache addresses, at least one actual promotion, explicit shutdown, and
released cache owners are acceptance gates. GPU skips fail GPU acceptance.
Host-only runs retain individual GPU requirements and documented contract
exclusions in JUnit; skips do not qualify GPU work. Registry failures are never
excluded.

The Qwen3-30 reference settings are 8 GiB cache envelope, 40 GiB host envelope,
2 GiB BF16 KV, context 2048, prepared capacity 64, two prepared pairs per layer,
health checks every 16 delivered tokens, cold threshold 0.15, maximum full
snapshot interval 1024 tokens, decayed LFU, and at most 32 pairs / 128 MiB per
maintenance. History and specialist protection are zero. These are reproducible
experiment settings, not production recommendations. Override memory envelopes
explicitly; admission must account for dense weights, private workspaces, graphs,
KV, source parameters, full mapped backing, and safety reserves.

For optional return-to-anchor experiments, use the existing serving harness with
`--anchor-advantage 0.02 --anchor-breadth 0.75 --recenter-pairs 64
--recenter-mib 256`. Keep other inputs fixed. Report stable, specialist, and return
intervals separately. The summarizer includes final control tails in serving
wall time and retains nested timers separately; do not sum those timers or count
draining useful work twice.

## Lifecycle and resource checks

Applications stop and await request/control producers before calling
`AsyncLLM.shutdown_async()`. The harness uses `close_serving()` to retain cleanup
through caller cancellation. The engine aborts further submissions, drains
readers, releases runner graphs, closes prepared storage and CPU owners, and
acknowledges worker release before terminating the engine process. Its existing
forced process timeout is unchanged. Uncertain control outcomes remain paused;
teardown never resumes them to simplify cleanup.

Use diagnostic runs separately from timing. The harness accepts
`--resources FILE --repeat-lifecycle 3` to reconstruct engines in one client
process. `--shutdown-case` additionally exercises `health-pending`,
`health-completed`, `health-cancelled`, or `maintenance-cancelled` with adaptive
health control. Zero `--epoch-pairs` and `--epoch-mib` provide an observation-only
shutdown control. A cancellation receipt states whether cancellation actually
arrived before completion.

```bash
python -m benchmarks.moe.expert_cache_serving --model "$MODEL" \
  --profile "$PROFILE" --prompts benchmarks/moe/fixtures/expert_health_chat_code.jsonl \
  --mode adaptive --control health --cold-threshold 0.15 --epoch-tokens 16 \
  --epoch-pairs 32 --epoch-mib 128 --admission together --concurrency 4 \
  --tokens 64 --repeat-lifecycle 3 --resources "$RESOURCE_RECEIPT" \
  --output "$CYCLE_DIRECTORY"
python scripts/check_expert_cache_lifecycle.py --cycles "$CYCLE_DIRECTORY" \
  --resources "$RESOURCE_RECEIPT" --output "$RESOURCE_CHECK"
```

Resource observations cover pre-load, CPU expert ownership, cache preparation,
graph/KV readiness, serving completion, and worker release. They distinguish
explicit mapped owners and CPU sources from Torch live/reserved pools, device
free memory, RSS, descriptors and children. Linux `VmPin` and Torch allocator
statistics alone do not measure CUDA pinning. The default plateau check allows
64 MiB post-warmup variation and requires stable client descriptor/child counts;
retain raw samples and justify any changed bound. Worker reconstruction uses new
processes; prepared-component tests separately exercise in-process close/rebuild.

## CI and native hardware

`.github/workflows/expert-cache-host.yml` runs hash-locked host checks on disposable
GitHub-hosted runners with read-only repository permission and no credentials
retained in checkout. It does not expose personal-network GPUs to fork code.
GPU qualification is a trusted manual path: approve an exact source pair, fetch
those exact commits, verify loaded artifacts, and retain the acceptance manifest.
Do not run untrusted checkout through `pull_request_target` or a persistent
privileged GPU runner. A workflow file is not a passing workflow run.

Use the existing [SM103 qualification guide](sm103-qualification.md) and
[HBM/Grace runbook](expert-residency.md#qualification-commands) for Station work.
SM120 source-built serving does not supply native SM103 loading, tcgen05/TMEM,
or Grace TMA acceptance. Ripper measurements describe **PCIe Gen4 x16**; record
negotiated link state under load. Gen5 transport and B300 performance require
separate measurements.

See the [source-bound qualification report](expert-cache-reference-results.md)
for measured lifecycle bounds, serving samples, retained failures and deferred
physical gates.

The [resident-capacity experiment](expert-cache-capacity.md) reuses these gates
with capacity-specific profiles derived from immutable calibration counts.
It separates intentionally constrained expert envelopes from feasible
all-resident execution while keeping KV and graph reservations fixed.
