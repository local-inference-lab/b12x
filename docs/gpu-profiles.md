# GPU preparation and startup autotuning

b12x has one execution lifecycle:

```text
Plan (declaration) -> PreparationSession fills the plan -> bind/run from the plan
```

A `Plan` is a declaration. Constructing it allocates no CUDA storage, compiles
no kernel, loads no GPU module, and warms nothing. A `PreparationSession`
configures the declaration, compiles missing programs, materializes the
component's private state, primes the operation, and installs the result in
the plan. Families bind and run from the plan they were given; nothing else
carries executable state.

## Declaring and preparing work

Each component owns its typed query and config, a `TuningContract` in
`_tuning.py`, metadata-only compile jobs, memory formulas, and private
materialized state in `_preparation.py`. The inventory is
`b12x.preparation.catalog.list_tuning_components()`.

```python
from b12x.preparation import PreparationSession, PreparedCall

plan = component.plan(caps_or_query, invocation=metadata, override=pin)
request = plan.request(
    name="decoder.attention",
    prepare_call=make_real_preparation_call,
    benchmark_call=make_representative_trial,
)
with PreparationSession(device=device) as session:
    session.prepare((request,))
    binding = component.bind(plan, actual_tensors_and_workspace...)
    component.run(binding)
```

The callbacks are integration-owned. They receive the component's private
materialized state, bind real parameters and inputs, and return
`PreparedCall(run=..., output=..., produce=..., reset=..., restore=...,
owners=..., close=...)`. Priming runs reset, activation production, and the
operation under a guard that fails if the operation launches a program the
declaration did not list. A no-op primer does not qualify an operation.

A prepare callback's `owners` and `close` are retained until the plan is
released. Include only resources needed by the prepared execution. Temporary
activation, output and scratch buffers belong to the call's closures, which
are released after priming. `restore` runs before readiness; it restores
borrowed inputs and finishes cleanup of call-only resources.

A benchmark callback is required when an unpinned declaration has more than one
eligible configuration. It uses representative state, valid nonzero inputs and
the serving numerical recipe. Producers write activations and their scales,
never model weights. Stateful trials restore the state they touch. Trial
scratch belongs to the trial call and is freed after the race; the caller's
serving workspace is never used for trials.

Fused MoE declares a composite plan whose `variants` map planned token counts
to child plans. The maximum count is the prefill chunk capacity; other counts
retain exact decode specializations. Requests take `prepare_calls` and
`benchmark_calls` mappings keyed by count. Runtime selects an exact planned
count when present and otherwise uses the prefill capacity, passing live M
to the native launch. A live count within capacity does not declare another
plan. Kernels with a static-M ABI still require their exact specialization.

WO projection defaults to exact-M preparation. Its optional invocation
`dynamic_tokens=True` permits live rows within a prepared capacity.
The ordinary native GEMMs and packing use the live count and group strides;
small fused decode declarations remain exact. Preparing one large capacity
therefore covers arbitrary prefix-replay suffixes without padding, changing
precision, or declaring/JIT-compiling a serving-time plan.

## Selection and cache semantics

With autotuning enabled, selection precedence is:

1. A valid explicit per-query config pin.
2. A validated completed choice from the selection cache.
3. A race over the complete eligible configuration set on a cache miss.

With autotuning disabled or cancelled, unprepared plans use a valid explicit
pin or the component's validated heuristic default. Already prepared plans
remain ready.

A fixed or effective-singleton declaration primes without a race. Defaults,
pins, fixed choices and partial races are never saved as measured winners.
Malformed matching cache data fails closed. Compiler artifact availability is
checked independently of selection-cache presence.

`B12X_COMPILE_WORKERS` limits compiler processes per preparation session
(default: 8). An explicit `compile_workers` argument takes precedence. Lower
this on unified-memory systems where compiler processes share RAM with model
weights and KV caches. This changes compilation concurrency, not autotuning.
`session.configure_compile_workers()` can change the budget between jobs
without releasing prepared plans. vLLM's `B12X_WEIGHTS_COMPILE_WORKERS` and
`B12X_STATE_COMPILE_WORKERS` override the common budget for their respective
preparation stages. State tuning uses disposable pools with the final KV layout
before allocating serving KV. `B12X_BIND_COMPILE_WORKERS` controls final binding
and priming. The Spark TP2 launchers default to 16, 16, and 4 respectively.

`autotune=False` uses a serial warmup through the same session's materialize,
prime and resource-ownership hooks. It skips selection-cache lookup, candidate
enumeration, compilation planning, compiler worker processes and measurement
graphs. Missing kernels compile in the calling process. Set `B12X_AUTOTUNE=0`
before launch on every rank to force this behavior for every batch.

During vLLM startup, press ESC in the foreground terminal to stop autotuning
across ranks and use that warmup path for remaining plans. The panel advertises
the key when terminal input is available. Cancellation persists across both
preparation stages; an active compile or GPU step must return before its rank
can switch. Each rank warms its own defaults, without candidate sharding or
winner consolidation. The coordinator retains collective ordering, error
propagation and complete-world completion. Incomplete races are not cached.

`cache_only=True` refuses search and compilation. A complete cached restart
loads and primes process-local state without starting compiler workers.

The decision cache uses schema 5 and an explicit positive integer version,
`B12X_TUNING_CACHE_VERSION`, which defaults to `1`. Set the same version on
every rank. Increment it before launch to discard prior tuning decisions:

```bash
export B12X_TUNING_CACHE_VERSION=2
```

Decision identity includes that version, the model namespace and reported
CUDA device name. GPUs with the same name share choices regardless of UUID or
visible ordinal. Source edits and compiler/toolchain changes do not invalidate
decisions automatically. CuTe and Triton artifacts retain their source-based
keys; the tuning-cache version is excluded from the compiler environment key.
A cached decision still validates its assignment and configuration, then
compiles any missing artifacts for that selected configuration. Increment the
decision version when implementation or toolchain changes warrant retuning.
Schema-4 files are left intact and are not read by schema 5.

A selection key digests the component, the contract versions, the encoded
query, the invocation, the pin and the dependencies. GDN decode, GDN and KDA
prefill, QSA and PLE encode their query without the pool's size (state slot
counts, cache page counts), and none of their programs is specialized on it,
so a restart against a pool of another size neither measures nor compiles
them. Paged attention keeps the pool in its key: its configuration carries
capacities derived from the page count.
Explicit codegen and numerical controls are captured in declaration metadata
and used consistently for configuration, memory and compilation.

Races run in bounded batches. The session instantiates candidates up to
`race_batch` at a time or until the measured resident bytes reach
`race_budget` (half of free device memory by default), times the batch, keeps
the best candidate, and carries it into the next batch so every candidate is
compared head to head with the running best. Losers are restored and closed,
and their trial storage is released before the following batch is materialized.

Race memory accounting reads the native allocator's aggregate allocated-byte
counter without converting historical CUDA graph-pool statistics to Python.
The first race lazily builds a small C++ extension through PyTorch; this
requires a host C++ compiler and CUDA headers located through `CUDA_HOME`.
The extension uses PyTorch's extension cache (`TORCH_EXTENSIONS_DIR` when
set). Preparation that performs no race does not build it.

Each candidate is primed once. A batch times two representative samples
per candidate by default, with L2 eviction and activation production before
each timed invocation. The first scored invocation also sizes a 256-microsecond
kernel-time budget per round; there are no unscored calibration replays.
Every sample queues a CUDA stream memory wait before its start event and
releases the wait through mapped host memory after its end event is queued.
Device execution therefore excludes Python launch gaps without graph capture
or CUPTI. The interval includes device dispatch between kernels; no per-kernel
marker cost is subtracted. Benchmark calls must enqueue asynchronous work on
the current stream; host synchronization and unjoined side streams are not
supported inside a scored call. Producers and resets execute before the gate.
Groups wait on their launch stream before reading or reusing events, and
the gate is drained before its host memory is freed. The L2 buffer is reused
across batches. Tuning cache identities include `stream_gated_events_v1` so
decisions measured with another method are retained separately.

Within a batch every candidate is timed with stream-gated events for at least one
round. After each round a candidate whose best round trails the batch leader
by more than 10% stops being re-timed and keeps the median of the rounds it
completed; the carried champion is re-timed in every round; survivors
complete three rounds. The winner is the lowest median over every candidate,
so an eliminated candidate is never barred from winning. `measured_count`
counts every candidate of the race, and the sharded winner exchange between
TP ranks is unaffected. The exhaustive race in `benchmarks/startup_quality.py`
does not eliminate.

A family's candidate space is its knobs' cartesian product under the
contract's predicates and per-query value ladders. Predicates omit wide a16
GEMM tiles for outputs of at most 64 columns and chunk-parallel GDN schedules
for at most 128 tokens. Ladders sample GDN pipeline-window sizes and MoE
resident-grid widths, retaining the default GDN window that fits the L2
budget and the MoE task-queue clamp. These ranges do not establish a
performance bound on omitted configurations; selection quality requires
measured comparisons. Changing a space bumps the family's
`candidate_contract_version`, which invalidates its cached selections.

Dense GEMM, mHC, KDA/GDN prefill, dense MLA, the DSA indexer, contiguous attention and paged GQA
separate correctness constraints from efficiency predicates. `B12X_AUTOTUNE_EXHAUSTIVE=1`, set
before declaration, bypasses the efficiency group while retaining TMA alignment,
MMA divisibility, thread-block and shared-memory limits, and scratch bounds.
The default search removes padded M tiles with an equivalent smaller-tile grid,
excess N coverage, inactive N warps, surplus pipeline stages, and projection grids
smaller than one eighth of the device's SM count. These are efficiency heuristics,
not correctness requirements. The query captures the setting, so exhaustive and
pruned searches have distinct selection-cache identities. Explicit valid pins
remain valid in either mode. Dense GEMM restricts NVFP4 cp.async to unswapped K<=256, bounds oversized
row tiles, restricts MXFP8 decode swapping to N<=K/2, and omits wide NVFP4
K512 prefill tiles outside short-K or narrow-output cases. MXFP8 retains
16-row tiles through M128 and the legal BK64 row-tile exception. Explicit
launch constraints bypass these search heuristics.

mHC prefill at M>=384 couples M warp groups, single N warps and 128–256
buffered K elements. At M>=2048 it keeps N tiles covering all 24 projections
and at least 2048 K elements per split. Beyond eight K splits, the grid is
limited to four times the device's SM count. These predicates preserve the
declared cartesian axes; explicit valid pins and exhaustive search bypass
them. The DSA indexer omits scalar scoring for MXFP4 prefill with 32 query
heads and at least 64 query rows.

KDA samples power-of-two windows plus capacity and its L2-derived default.
GDN retains its capacity/half/quarter/default window ladder and covering
segment restrictions in ordinary mode. Dense MLA samples quotient and
power-of-two split caps plus its default; these are capacity heuristics,
not equivalence classes for dynamic lengths. Contiguous attention limits
M tiles to 128 while preserving every legal N tile. Paged GQA samples
power-of-two residencies plus 3, 6, saturation and its default, preserving
explicit residency and split-KV controls. Exhaustive mode restores each
complete declared axis. These restrictions do not change kernel math,
launch specialization, trial timing or elimination. Other components retain
their declared predicates.

Equal declarations reuse a completed candidate list and coverage counts when
they share the same contract object. During the configuration pass, memory
requirements are checked once per unprepared request, using its selected
configuration or the default while selection is pending.

## Plans, sharing, release, and the fallback

`session.prepare` fills each request's plan in place. A plan that is already
prepared is skipped, so a second `prepare` call over a superset of plans is
incremental. Preparation with tuning enabled upgrades a plan whose selection
came from the default.

A family sets `shared=True` on its `Plan` when the materialized state depends
only on the declaration, never on the caller's weights, pools or
communicators, and when the caller's prepare call has no per-owner side
effects, because an aliasing plan's prepare call never runs. Equal shared
declarations then share one prepared state, which the session prepares once.
Shared plans cannot declare dependencies or collectives. Pool-dependent
families (paged and QSA attention, PLE state, GDN and KDA prefill) are not
shared.

`session.release(plan)` drops a plan's prepared state; composites release
their children. Destroy graphs that replay the plan first. `session.close()`
releases every plan it prepared. When a preparation finishes, the session
evicts every compiled program that no prepared plan retains from the
families' registered kernel memos and the compiler's memory cache, so a
race's losing candidates do not stay resident. After a race it also
collects the evicted executables so their CUDA libraries unload, and resets
the device's per-thread local memory limit to what the retained kernels
need, since the driver grows that limit for any launched kernel and never
shrinks it; a cold start therefore leaves the device in the state a cached
start reaches.

An unprepared plan that is bound or run is materialized with its default
configuration on first use, without a priming call, and a warning names it;
its kernels compile on first use, in the calling process. Compile planning
leaves deferred programs in the families' kernel memos and in Triton's
per-function caches; the in-process compile drops the ones it is about to
compile first, so the factories lower them instead of reusing the planning
artifact. This is the whole treatment of a serving
shape the calling layer did not declare: a heuristic plan and its scratch,
as in an integration without startup preparation. Runtime lengths within an
existing declaration's capacity reuse that plan; they are not undeclared
shapes. A different capacity or static kernel ABI requires its own declaration.
An integration that
wants such a plan primed before serving calls `prepare_default(request)`
with its own prepare call during startup. After `session.freeze()`, or
during a capture, an unprepared plan is an error.

A plan pickles as its declaration only. Compiler caches serialize the guard
values of compiled code, which can include a plan closed over by that code;
the prepared payload never travels, and the unpickled copy keeps the handle
number without being registered, so it does not resolve through
`plan_from_handle`.

## Memory

`MemoryRequirements` combines caller-provided scratch specifications and
`PersistentMemory(key, required_nbytes, resident_nbytes)` entries. Inside
memory and materialize callbacks, `current_plan()` is the allocation-owner
key and `current_prepared_state()` is the plan's installed state, if any.
`plan.memory_requirements()` and `plan.scratch_specs()` report the prepared
configuration's memory once prepared and the default configuration's memory
before that, so a caller can size scratch from the plan at any time.

Block-scaled GEMM declarations carry a `workspace_form`. `provided` takes the
caller's workspace at run time; `owned` allocates the workspace and any
packed-activation buffers once at materialize and declares them as persistent
memory. No prepared execution allocates per call. The caller's capacity in
`workspace_nbytes` prunes configurations whose workspace would exceed it.

## Capture and freeze

`session.capture()` and `session.freeze()` hold a process-wide kernel
resolution guard. Every compile, JIT miss, and planned module load raises
while the guard is held. `freeze()` is permanent for the session and rejects
new obligations; captured graphs replay prepared launchers only.

## Cooperative startup

`session.begin(requests)` returns a `PreparationJob`; `job.advance()` runs
local metadata, GPU preparation and measurement steps within a 100 ms time
slice. An in-flight step finishes before the deadline is checked. Pending
compilation, collective readiness and winner exchange return control
immediately, so a driver can coordinate ranks before admitting dependent work.
`configure_tuning_shard(rank, ranks)` assigns the process a disjoint share of
every race; ranks exchange their local winners through `TuningRequirement`
and install the global winner. Collective declarations are never raced; their
priming is preceded by a `CollectiveRequirement` barrier that a distributed
driver authorizes only when every participant is ready. `cancel_tuning()` is
sticky: it stops optional work, preserves completed winners, and still
prepares required defaults.

## vLLM

vLLM prepares plans at two lifecycle points. The weights stage runs after
model load and before memory profiling and prepares every plan that depends
only on parameters or communicators, with timing. The state stage runs after
the KV and state pools exist, before graph capture, and prepares the plans
that depend on those pools, with timing. Both stages declare the same token
counts, including the decode-graph counts the runner rounds up for
speculative decoding, so a weight-only plan serves every captured graph. The
state stage also collects the weights-stage units again: prepared plans are
skipped, and a plan a layer declared for a count it first saw in the state
stage is prepared before graph capture. During graph-memory profiling the
pool-dependent plans are prepared against the throwaway profiling pool with
their default configurations and released afterward; plans that were already
prepared are left installed.

Each native layer exposes one unit per stage through
`layer.b12x_preparation_provider.get_b12x_preparation_units(layer, workload)`.
A layer that is executed for a shape its units did not declare declares that
shape on first use; the plan materializes its default configuration when it
is first bound, with a warning. vLLM does not freeze the session after
startup; `session.capture()` guards graph capture.
The driver collects units, hands their requests to one session per worker,
and drives the world-coordinated rounds. Graph capture runs under
`session.capture()`; the session is never frozen, so a shape that appears
only in serving compiles on first use. There is no install step: the layer
holds its plans. Custom ops carry tensors, primitives and a layer name; their
bodies resolve the layer and run the family's plain-Python path. Workspace is
drawn from vLLM's workspace manager inside op bodies, sized during the
profile run, and locked before capture; growth under capture is an error.
The shared experts of a MoE layer run on a side stream while the routed
experts hold arena views, so their linears report their scratch requirement
through `get_workspace_size` and run inside the scratch the MoE runner
reserves for them instead of drawing an overlapping view.

Autotuning cache misses is enabled by default. Disable optional search while
retaining mandatory preparation with:

```text
--kernel-config '{"enable_b12x_autotune":false}'
```

The compact loading display is rank-zero only and is driven by real
preparation phases, candidate counts, completed timing rounds and cache and
compiler activity. The progress bar counts measured candidates against the
planned candidate races across the participating ranks; request readiness is a separate
count. Candidate counts describe search work, not an elapsed-time estimate.
Race status distinguishes batch size, rank-local candidates and global
candidate count. Latency histories reset between batches and do not combine
different ranks.
Redirected output uses plain milestone lines.

Set `B12X_PREPARATION_TRACE_DIR` before startup to collect timing JSONL:

```bash
export B12X_PREPARATION_TRACE_DIR=/tmp/b12x-preparation-timing
```

Each worker writes `job-<pid>.jsonl` with request queries, completed
selection sources and coverage, batch candidate indices, carried champions,
latencies, and cumulative wall-time counters. The vLLM driver also writes
`coordinator-<pid>.jsonl`. A `begin` record starts a job or coordinator
segment; multiple preparation stages append to the same per-process file.
Counters are inclusive: `compile_plan`, `materialize_prime`,
`memory_accounting` and cleanup are contained in the job's phase and
`advance` totals. Job `between_advances` measures time outside the job.
Coordinator `local` includes `compiler_wait`; `control_exchange` measures
rank coordination, and its `between_advances` measures the external driver.
These nested totals must not be added together.

## Native benchmark consumers

`B12X_PCIE_DCP_HEAD_GATHER_PUSH=1` opts into the experimental native posted-write
decode-context-parallel head gather for four ranks, BF16, 64 global heads, and a
512-element head dimension. The default `0` selects the pull transport.
`query_from_runtime(..., call={..., "peer_write": True})` can pin the posted-write
transport explicitly. The control is captured in declaration metadata and the
compiled launcher identity; changing it after preparation does not change binding
or graph replay. Other push geometry fails closed. The posted-write DCP channel
uses its IPC slab, system-scope barrier, graph slots, and L1-bypassing incoming loads.
Serialized launch and replay remain required; this transport is not a qualified
default.

`B12X_PCIE_TP4_REMOTE_PUSH=1` enables the experimental native plain BF16
all-reduce for four-rank tensor parallelism, DeepSeek V4.1 widths 5120 and 1280,
and one through eight rows within the declared eager-slot capacity. Other plain
shapes use the pull transport, and the default remains off. The transport uses
four source shards, alternating graph slots, and a system-scope barrier.
Destinations are rank-staggered, incoming loads bypass L1, and accumulation uses
rotating-rank FP32 addition followed by one BF16 rounding operation. The
four-rank tensor-parallel preparation path snapshots the flag and checks rank
agreement before allocating IPC; binding and replay do not reread it. Fused
four-rank tensor-parallel eligibility is unchanged.

`benchmarks/benchmark_startup_autotuner.py` uses the same declarations and
session with explicit retained benchmark calls. Its optional group subsets are
diagnostics, not full-model serving acceptance. `benchmarks/startup_quality.py`
independently checks complete eligible sets, numerical correctness and
selection regret through cache-only pinned sessions. Fixed paths receive
numerical qualification rather than fictitious races.
