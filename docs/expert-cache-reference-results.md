# Expert-cache reference qualification, September 2026

This pass hardens the demonstrated single-GPU SM120 experiment. It changes
shutdown ownership, source verification and acceptance tooling; it does not
change cache policy, expert arithmetic, transport, or production defaults.
Use the [reference guide](expert-cache-reference.md) to reproduce the supported
configuration without following the chronological research ledger.

## Source and artifact boundary

The initial live branches were b12x `84d82af329b32b7ac9bc9575a735b71f8a9d59ef`
and vLLM `3e45b530e58186046383e7294e611c2f6bf5cfb8`. The inspected default branches
were b12x master `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68` and vLLM main
`47ccf6c57d92f03630ebcbad3809450545825488`. Neither default branch was changed.

| Evidence | b12x revision | Companion revision |
|---|---|---|
| Full 38-case SM120 suite and initial bounded sanitizers | `604d4ae865c8abc188870fc79f60018eadcfe10b` | `c3efbdf25b9fdcd9adf5ae3888ced9c7e0a906ca` |
| Serving, repeated lifecycle and fresh calibration | `b33d457f65e734366937d053ca87de449afc107e` | same |
| Stronger physically pending read test and sanitizers | `3d547aec77c75acce4d6f49e4f53695b4c35832f` | same |
| Independent hosted host acceptance | `3d547aec77c75acce4d6f49e4f53695b4c35832f` | host tier does not install vLLM |

`b33d457` adds rejection of the actual process-manager forced-kill warning.
`3d547ae` changes the pending-read test to use the existing explicit stream gate.
Neither changes the runtime measured at `604d4ae`. These remain separate source
receipts, not measurements relabeled as the final documentation commit.

The complete companion wheel was built from source, including CUDA extensions
and Rust, with precompiled downloads disabled. Its SHA-256 is
`9b4b73f9c57cad647b42fa535790dfc566141d21ee0ba1ed6eb46732fd5ebc99`.
Verification covers 2,761 packaged Python/native files, 2,749 matching source
Python files, installed artifacts and libraries loaded by the serving worker.
The environment used Torch 2.13.0, CUDA 13.3, CUTLASS DSL 4.6.2, Triton 3.7.1
and driver 580.173.02. CUDA bindings 13.0.3 were explicitly supplied from a
separate dependency directory; the receipt records the loaded path and version.
This is a declared dependency override, not an overlay of old vLLM extensions.

Serving used GPU `GPU-47363510-b87a-13a5-4824-2542e97df76c`, an RTX PRO 4000
Blackwell on ripper, with **PCIe Gen4 x16** observed under load. The stronger
pending-read diagnostics used the second physical GPU,
`GPU-cc109c01-9756-d0db-21ea-f1825d3f963f`. GPU diagnostics were separate from
timed serving. Raw receipts retain power/clocks, topology, affinity, exact
commands, source exports, package identities, prompts and output token IDs.

The supplied Qwen3-30B-A3B-NVFP4 checkpoint fingerprint is
`bdde460712efe6ea50175eac186c27aa983348baf9faf2ab28a917ed93182609`.
The retained learned profile ID is
`20006a3fa40680307aa6d1f72816ded8fe91c4d7297c44410e585031a28fc4fa`.
No weights were downloaded or redistributed.

## Shutdown defect and fix

The unchanged baseline reproduced two different symptoms. Ordinary non-cache
serving exited its worker normally but later printed an ignored
`AsyncLLM.__del__` exception: interpreter finalization had already cleared
`shutdown_prometheus`. Static cache serving also exceeded the existing five-second
process-exit timeout. Captured worker stacks showed mapped backing release in
`cudaFreeHost`, followed by native teardown, inside that process-exit interval.

The fix provides an explicit asynchronous close before process termination.
Application producers stop first. Engine-owned controls finish despite caller
cancellation; the scheduler retires readers; runner graphs, preparation state,
mapped owners and CPU sources are released; the worker acknowledges release;
then existing process teardown runs. Completed shutdown is idempotent, so the
destructor has no remaining work. The process kill timeout is unchanged.
Failed or uncertain maintenance remains paused and teardown never resumes it.

Completed probes, cancelled probes and cancelled maintenance were physically
exercised. The cancellation receipts confirm that cancellation arrived before
completion. Their explicit shutdown took approximately 6.83–6.86 seconds,
including worker release and process exit, with no forced kill or destructor
exception. Zero-movement diagnostic controls and actual-promotion cycles are
retained separately. Host tests cover repeated cancellation and failed or timed-out
close; physical cancellation is not a claim that every worker-failure path was
fault-injected on hardware.

## Repeated lifecycle and memory

Six consecutive C4 engines were reconstructed in one client process. Each
served 1,024 tokens, performed 320 promotions and completed shutdown. Every
worker reported zero cache-owned mapped bytes, retained checkpoint-source bytes,
graph owners and pending health reads after release. A separate prepared-component
test closes and reconstructs four times in-process, holds an actual GPU read
pending behind a stream gate, and checks a flat live-allocation plateau.

| Post-shutdown resource, six engine cycles | Observation |
|---|---|
| Worker live Torch memory | 459 MiB each cycle |
| Worker reserved Torch pool | 508 MiB each cycle |
| Device free memory after first-cycle warmup | 0.06 MiB range |
| Worker RSS after warmup | 5.88 MiB range |
| Client RSS after warmup | 41.86 MiB range |
| Client descriptors | 38 each cycle; worker descriptors settle at 94 |
| Client children | One persistent resource tracker |

The default acceptance bound is a 64 MiB post-warmup range plus stable descriptor
and child counts. The ordinary non-cache worker leaves the same 459 MiB live,
with a 486 MiB reserved pool before process exit. That residual is recorded as
process-lifetime engine storage, not attributed entirely to CUDA context or declared reclaimed
in-process. These finite runs establish bounded observed behavior; they do not
prove arbitrary-duration leak freedom or full in-process vLLM reconstruction.

The loader observations preserve the central storage contract:

| Stage, static cache arm | Live Torch device allocation | Retained CPU expert sources | Mapped canonical backing |
|---|---:|---:|---:|
| Before model loading | 0 | 0 | 0 |
| Dense loading and CPU expert ownership | 1.670 GiB | 15.188 GiB | 0 |
| Cache materialization | 9.668 GiB | 15.188 GiB | 15.188 GiB |
| Graph/KV ready | 12.540 GiB | 15.188 GiB | 15.188 GiB |
| Worker release | 0.448 GiB | 0 | 0 |

The cache loader's recorded Torch peak at the end of loading was also 1.670 GiB.
The ordinary non-cache loader reached 16.858 GiB after weight loading. The cache
loader never creates that complete routed-expert GPU representation. Device
free-memory observations accompany Torch statistics;
explicit owner accounting measures mapped storage. Neither Linux `VmPin` nor
Torch pools alone represents all host pinning/native allocation. Full canonical
backing and retained CPU sources are both charged: bounded pageable staging has
not been implemented.

## Acceptance and retained failures

- Independent GitHub-hosted acceptance at `3d547ae`: **1,083 passed, 63 visible
  skips** for physical GPU requirements and documented compile-contract
  exclusions. [Actual workflow run](https://github.com/local-inference-lab/b12x/actions/runs/35545564066).
  This runs real host tests on a disposable hosted runner with read-only
  permissions; no personal-network GPU or secrets are exposed to fork code.
- SM120 prepared arithmetic, checkpoint layer, routing/health and model-wide
  epochs: **38 passed, zero skips**.
- Final pending-read/native-cache memcheck and synccheck: **two passed each,
  zero sanitizer errors**. The pending-read case also passed without sanitizer.
- Source-built companion loader/control/lifecycle tests: **18 passed**.
- Ordinary non-cache graph serving and source/artifact checks run in serving
  acceptance, not merely as checks for old receipt files.

The old registry failure was reproduced: five failed and four passed on the
initial source. Lazy FP6 API registration and stale/omitted exports were repaired;
all nine registry tests now pass. No blanket exclusion was added.

The documentation-commit [host run](https://github.com/local-inference-lab/b12x/actions/runs/35547240067)
subsequently exposed a polling assumption in
`test_two_ranks_agree_on_cached_choices_and_shard_remaining_races`. It expected
both ranks to finish a bounded preparation advance in the same poll. Forcing
either rank to use a shorter advance quantum reproduced the failure. The test
now waits for both contributions, verifies their common tuning key, and exercises
all cache combinations with either rank delayed. All 64 preparation-session
tests pass locally. Preparation runtime code is unchanged; the failed hosted log
and deterministic reproduction remain in the evidence bundle.

Other retained failures include a missing Triton dependency in the first fresh
CPU-only environment, a local temporary-directory quota failure, companion pytest
importing the source checkout instead of its installed wheel, and the first
stronger pending-read test. A fixed-duration GPU sleep did not guarantee an
in-flight read under sanitizer instrumentation; both runs failed that assertion
without sanitizer memory/synchronization errors. Reusing the explicit stream
gate made the condition deterministic and both sanitizers passed. Failed logs
and source identities remain alongside successful retries. An optional unused
MXFP4 Triton import warning remains visible; this qualification does not cover
that backend.

## Serving comparison

The retained-profile acceptance passed three alternating-order pairs at C4:
static/adaptive, adaptive/static, static/adaptive. Each arm generated 4,096 tokens
from `benchmarks/moe/fixtures/expert_health_chat_code.jsonl`, with 256 tokens per
request, controlled admission, identical profile, BF16 whole-K recipe and graph
configuration. Every pair has exact token equality; all six output hashes are
`b610b6c6e4ef` (prefix). Graph and cache addresses remain unchanged within each
arm. Static contains no routing observer. History, protection and anchor recovery
are disabled; adaptive uses health-16 and 32 pairs / 128 MiB.

| Interval | Static tok/s, mean ± sample SD | Adaptive tok/s, mean ± sample SD | Adaptive / static − 1 |
|---|---:|---:|---:|
| Complete traffic sequence | 85.122 ± 0.068 | 121.567 ± 0.034 | +42.82% |
| Stable general | 173.377 ± 0.013 | 172.677 ± 0.031 | −0.40% |
| Code transition | 56.477 ± 0.059 | 94.044 ± 0.038 | +66.52% |

Overall static samples are 85.098, 85.069 and 85.198 tok/s; adaptive samples are
121.593, 121.581 and 121.528. These measurements describe this authored fixture
and token length, not a universal improvement or a matched before/after estimate
of the lifecycle patch. They preserve the demonstrated stable-versus-drift tradeoff
without selecting new policy settings. No return interval was added to this
bounded baseline; anchor recovery has its retained, separately sourced research
evidence.

Each adaptive arm performs 34 maintenance operations after the preparation
baseline, 1,088 promotions and 2.691 GiB of API-accounted copies. Aggregate blocked
scheduling intervals are 2.800–2.816 seconds. In the first pair, one operation and
32 promotions occur during stable traffic, and 33 operations / 1,056 promotions
during code traffic. Median maintenance wall time is 82.52–83.89 ms across arms;
p95 is 101.33–101.57 ms. Median nested stages include approximately 44–45 ms
scheduler drain, 38 ms worker RPC, 9.3 ms policy and 13.1 ms apply. The worker
stages are inside the RPC, and the drain includes useful already-submitted work:
these timers must not be added together or added again to measured serving wall.

The adaptive final-control tail is 5.03–7.01 ms and is included in overall
throughput. Startup and explicit shutdown are reported separately; timed-arm
shutdown takes 9.89–10.01 seconds and all six workers exit without forced cleanup.
The ordinary non-cache graph smoke also passes. It uses ordinary ModelOpt W4A4,
so it is a regression smoke, not a numerically matched W4A16 operator benchmark.

| Client latency statistic, range across three arms | Static | Adaptive |
|---|---:|---:|
| Median TTFT | 349.03–349.68 ms | 286.13–287.21 ms |
| p95 TTFT | 460.23–462.54 ms | 427.76–429.35 ms |
| Median token delivery gap | 39.83–40.13 ms | 27.09–27.27 ms |
| p95 token delivery gap | 89.52–89.78 ms | 57.64–57.92 ms |
| p99 token delivery gap | 99.47–99.82 ms | 84.64–85.57 ms |

Raw request events retain per-phase distributions, maintenance-adjacent gaps and
per-request decode rates. Client delivery gaps are not device iteration timings.
The alternating order and narrow observed variation support this local baseline;
they do not remove topology, prompt, concurrency or controlled-admission limits.

Fresh-profile acceptance also passes, separately from the timing baseline.
Eight calibration prompts generate 1,024 tokens and produce profile
`cf39df99c86608b4d780b482d8cbc0488ab7a3b82a1e1b1e7e0e809a8237d17a`.
The profile is checkpoint/recipe/geometry-bound, terminates at an explicit finite
calibration boundary and does not claim convergence. Evaluation text is disjoint
from calibration. Ordinary non-cache serving completes, then static and adaptive
each generate 1,024 tokens; adaptive performs 320 promotions. Their exact output
hash is `aa10bf56f714` (prefix), with fixed graph/cache addresses, immutable profile
and released owners after clean shutdown. This verifies the complete calibration
to serving path rather than only reuse of a development profile.

## Native SM103 and remaining scope

No accessible physical B300/GB300 endpoint was identified in this session.
The existing SM103 launcher now prepares and executes the ordered residency
gates: all HBM, all coherent Grace, mixed parity, same-graph updates, and native
cache control. It rejects non-SM103 hardware and fails required skips. The
[existing runbook](sm103-qualification.md#native-hierarchical-residency-execution)
contains exact native and sanitizer commands; production geometry and per-stage
measurement remain behind those gates.

There are **no new physical SM103 results or matched native performance numbers**.
The CPU-source full-model adapter is still SM120-specific; native SM103 needs a
qualified MXFP4/MXFP8 source/preparation adapter. Keep all-HBM operator quality,
hierarchical tier cost and adaptive serving benefit as separate comparisons.
SM120 portable byte access cannot qualify Grace TMA or tcgen05/TMEM execution.

Physical B300 correctness is the next required target gate. After that and the
reference lifecycle are stable, the smallest useful generalization is a resident
capacity sweep on the same checkpoint: it tests the admission and cache economics
without introducing a new loader, model family or numerical contract. Gen5
transport requires its own fixed-route/fixed-promotion measurements; no Gen4
bandwidth extrapolation is made here.

## Receipt locations

Raw artifacts remain outside the repository at
`ripper:/home/jasonc/b12x-reference-results-20260920`, with a local copy under
`/home/jasonc/b12x-reference-evidence-20260920`. The complete wheel and build are
retained under `ripper:/models/b12x-reference-build-20260920`.
These are historical evidence locations, not required reproduction paths.
The reference commands accept explicit checkpoint, profile, artifact and output
locations. Earlier policy experiments and failure receipts were not modified.
`evidence-index.json` in the local bundle binds raw files by SHA-256;
`serving-analysis.json`, `serving-summaries.json`, `fresh-profile-summaries.json`
and `lifecycle-six-check.json` retain the derived tables and their underlying
receipt identities.
