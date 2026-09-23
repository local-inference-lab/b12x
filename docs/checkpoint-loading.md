# Checkpoint loader

The `--load-format b12x` adapter loads selected safetensors ranges into ordinary
PyTorch CUDA tensors. It never installs an allocator or requires model weight
factories. `read_mode` selects transport only:

| Setting | Checkpoint payload transport |
| --- | --- |
| `auto` (default) | Bounce on GPU host-page-table devices such as GB10; GDS elsewhere |
| `bounce` | O_DIRECT → 16 MiB io_uring host ring → asynchronous CUDA copies |
| `gds` | Shared cuFile reads and CUDA IPC scatter across host-local TP ranks |

GDS registers ordinary CUDA allocations. Both paths retain tensor owners through
completion, and both leave allocation to PyTorch. `B12X_DISK_BACKEND` separately
controls Engram/PLE row fetching. Bounce mode does not load cuFile.

## Integration and ownership

Install b12x and a vLLM checkout containing its weight-transfer hooks, then enable
the loader plugin:

```sh
VLLM_PLUGINS=b12x_loader vllm serve MODEL --load-format b12x
```

`weight_transfer` scopes checkpoint writes and completion fences only. Weights,
outputs and workspaces all use their normal allocation paths, including MTP and
draft modules. Numerical preparation can reuse weights in place or allocate new
tensors without consulting the loader.

The adapter preserves index, prefix, expert and file-backed-weight filtering,
including draft-model loading. File-backed embedding tables retain their separate
storage contract. The rank-zero `b12x / weight loading` panel shares the kernel
autotuning display's palette and stage rail. It honors `use_tqdm_on_load`, uses
plain milestones for redirected output, and remains active through read completion.
Terminal output from existing log handlers, native code and child processes is
serialized above the panel through the vLLM startup-output capture.
TP ranks finish model construction before the panel starts. The panel completes
and stops before vLLM emits its weight-loading timing log.
The bar measures shards routed, with selected payload bytes on rank zero;
completing routing does not imply completed I/O. Progress reporting never forces
a flush at shard boundaries. Setup, planning, reading and synchronization use an
animated activity bar with elapsed phase time; they do not show a shard percentage.

After completion, shared loading reports selected payload and physical reads
summed across the TP group. Total loading time includes routing, planning, reads,
scatter and synchronization. The phase breakdown reports routing, shared-reader
setup, planning, reads and synchronization, including collective waits in the
reporting rank's wall time. Shared read-and-scatter time is reported separately
using the slowest reader's native interval in each epoch. Neither interval includes
subsequent weight preparation or kernel compilation. The estimated transfer rate
divides physical bytes read by the shared readers across all TP ranks by that
shared read-and-scatter interval, using decimal GB/s. It includes GPU scatter and
excludes header reads and rank-local numerical dependency reads.
Detailed per-rank I/O and
transfer counters are logged at debug level and retained on the model.

Loaded tensors remain valid after the session closes. The loader retains queued
source/destination tensors until I/O and CUDA copies finish; callers retain graph
inputs for replay. No custom MemPool, allocator lifetime registry, or temporary
allocator-setting changes are used.

## Routing and completion

`b12x/loader/_checkpoint.py` parses headers and routes metadata-only tensor views
through model sharding. A meta view carries a source range, shape, dtype and
strides; unregistered arithmetic on that view fails explicitly. Numerical callbacks
use `materialize_weight` to load owned values before computing with them.

Each queued transfer has eight unsigned 64-bit fields:

```text
(fd, offset, row_bytes, destination, operation, rows, source_stride, destination_stride)
```

Strides are bytes. Operations are unchanged-byte reads, BF16-to-FP32 expansion,
and owned CPU control-metadata copies. The metadata operation uses `offset` as a
source address. Native validation checks file bounds, CUDA allocation bounds,
device identity and disjoint destination extents before issuing reads. File,
row, stride and pointer arithmetic uses 64-bit values. Source files must remain
immutable until the session closes.

The executor sorts jobs by file position, divides large transfers into 64 MiB jobs,
and executes them through the selected native transport. `io_threads` defaults to
8 and accepts 1–16 through vLLM's model-loader extra configuration. Python retains
source descriptors and destination owners until completion; no Python callback
runs per read.

`flush()` waits for all queued writes. The caller's loading stream is synchronized
before native writes, and GPU copy streams complete before their scratch can be
reused. Numerical consumers and weight preparation require a completion fence.
Errors drain worker activity before being reported. Validation errors leave
weights untouched; an execution failure can leave partial destinations and poisons
the GDS executor, so model startup must fail.

`finish()` marks an explicit model-routing boundary. vLLM calls
`finish_weight_transfers()` once after `model.load_weights` and before numerical
weight preparation. Every rank in a shared read group must enter the same boundary,
including ranks with no queued destinations. Ordinary `flush()` and materialization
remain rank-local and complete all accepted writes. An eager flush may therefore
read a source range independently before the remaining ranks share their reads.

The device loader initializes its rank-local shared reader on a background thread
after model construction, while checkpoint routing continues. The completion boundary
waits for initialization and reports failures collectively before any shared read.
Session cleanup joins initialization and releases its resources even if routing
fails before reaching that boundary. Reported setup time includes only work and
waits remaining at completion; initialization overlapped with routing is not added
again to the wall-time breakdown.

## Asynchronous bounce reads

On Spark the default transport uses sixteen 1 MiB host slots registered with
io_uring and CUDA (16 MiB per rank). It issues `READ_FIXED` requests against
`O_DIRECT` descriptors. `io_threads` sets maximum outstanding reads, from 1 to 16,
defaulting to 8:

```sh
VLLM_PLUGINS=b12x_loader vllm serve MODEL --load-format b12x \
  --model-loader-extra-config '{"read_mode":"bounce","io_threads":8}'
```

A native copy thread submits `cudaMemcpy2DAsync` into the final CUDA tensors while
other reads remain outstanding. A slot is recycled only after its copy stream
completes. The caller's loading stream is synchronized before the batch; `flush()`
waits for all submitted reads and CUDA writes. No CPU writes target model weights.

Headers and control metadata use a separate 64 KiB direct-read buffer. BF16-to-FP32
expansion compacts and widens backwards within a ring slot, preserving every bit
pattern before the CUDA copy. Other contiguous casts can use an additional bounded
8 MiB GPU input buffer. TP slices preserve destination padding. There is no Python
callback per read, page-cache fallback, or tensor-sized host staging allocation.

This transport requires liburing development headers and pkg-config when building
the native helper, sufficient locked-memory allowance, and permission to use
io_uring in the host/container policy. Missing support fails explicitly. Both
normal and expandable PyTorch CUDA allocations are accepted. I/O failure drains
outstanding work; CUDA or ring submission failures retire the executor. All ring
resources are released when the loading session closes.

The audit records `read_mode`, `bounce_buffer_bytes`, `bounced_bytes`,
`io_uring_submits`, and `max_inflight_reads`.

## GPUDirect Storage transport

Bulk loading uses the shared reader described below. Rank-local numerical
consumers still require immediate completion; `b12x/loader/_gds_checkpoint.c`
serves those dependency boundaries with synchronous `cuFileRead` calls from native
worker threads. Payloads enter GPU memory without a CPU payload buffer. Each
worker owns an 8 MiB registered device scratch buffer, up to 64 KiB of allocation
alignment padding, and a CUDA stream. Header reads use a separate 64 KiB CPU
`O_DIRECT` reader. At the default eight workers, reserved native GPU scratch is
64.5 MiB if a rank-local consumer needs this executor. It is allocated lazily and
released at session close. Ordinary bulk completion uses the shared reader's
17.8125 MiB allocation instead.

A file range aligned to 4 KiB with a 64 KiB-aligned destination and at least
64 KiB of payload can read directly into a registered final-weight range. Final
registrations cover disjoint 64 KiB GPU pages. Other layouts read aligned file
windows into registered GPU scratch, then copy the selected bytes to final
storage. Small strided TP rows are coalesced and scattered on the GPU. Final EOF
reads require the exact available byte count. Alignment and TP layout can therefore
cause GPU copies and extra physical reads, but never a full host payload copy.

`b12x/loader/_gds_kernels.py` compiles byte-copy and BF16-expansion kernels before
loading. Expansion writes the exact FP32 representation `bf16_bits << 16`, including
signed zeros, subnormals and NaN payloads. Kernel arguments use 64-bit runtime
counts and strides; live tensor sizes do not enter compile keys. Native launch
validation requires the compiled parameter ABI and a scratch-free ordinary CUDA
launch. The Python session retains compiled module owners throughout execution.

The optional C extension requires cuFile >= 1.14 development files, CUDA runtime
and driver libraries, and a C compiler. `CUDA_HOME` or cuFile's `pkg-config` entry
selects build files. Build caching includes C/header source, compiler flags and
library identity. Importing `b12x.loader` neither builds the helper nor initializes
CUDA.

Before loading cuFile, b12x derives configuration from the caller's
`CUFILE_ENV_PATH_JSON`, `/etc/cufile.json`, or cuFile defaults. It defaults
`properties.allow_compat_mode` to `true`, preserves explicit settings, and selects
a cached private JSON file in the process environment when a change is needed.
Source configuration files are not modified. cuFile uses GDS when available and
CPU compatibility reads otherwise; destination allocations and GPU scatter retain
the same contracts in either mode.

Explicit cuFile settings remain authoritative, including a driver initialized by
another library. `CUFILE_FORCE_COMPAT_MODE=true` forces CPU compatibility, while
`CUFILE_ALLOW_COMPAT_MODE=false` disables fallback. Missing cuFile development
files, registration failures and failed reads remain fatal. The `gds_enabled`
counter identifies the cuFile backend, not the transport used by an individual
read; cuFile's transport statistics distinguish direct GPU and CPU operations.

## Shared checkpoint reads

Status: implemented for host-local groups using device weight allocations. The
vLLM adapter always uses this backend on those devices, including TP1. Selecting
the b12x load format is sufficient:

```sh
VLLM_PLUGINS=b12x_loader vllm serve MODEL --load-format b12x \
  --tensor-parallel-size 4
```

The vLLM adapter supplies its TP CPU process group. b12x validates host identity
and distinct physical device UUIDs, gathers routed descriptors, and partitions the
aligned union of their source envelopes among ranks. File identity includes the
resolved pathname, device/inode, size, and modification/change timestamps. Strided
gaps can contribute physical reads; only routed rows are written. Source identity
is retained when metadata is opened and checked before completion and after
execution. File descriptors and virtual addresses are never
used as cross-process identities.

Each allocation is exported only after validating its complete CUDA extent and
device identity. Peers import handles on the GPU that launches the scatter.
All ranks finish descriptor, file-handle and mapping validation before shared
payload writes start. Earlier rank-local metadata copies and eager transformations
are already complete and are not rolled back on failure. The native executor in
`b12x/loader/_gds_owner.c` uses four registered 4,653,056-byte slots and reserves
18,677,760 bytes (17.8125 MiB) including alignment padding. The default eight
workers issue two synchronous reads per slot. `io_threads` must be 4, 8, 12, or 16.
Completed reads feed byte-copy or BF16-expansion kernels; CUDA events gate slot
reuse. Registered staging, streams, workers and file handles persist until the
loading session closes. Python performs planning and control collectives, without
per-read polling or per-fragment launches during execution.

Completion compares aggregate written bytes with the routed destination count.
Every rank drains its reads and scatter streams before acknowledging completion.
Peers close IPC mappings and acknowledge retirement before destination references
can be released or replaced by quantization preparation. A failed rank or control
collective fails the load. If peer retirement cannot be proven, destination owners
are retained until process exit. The supplied process group controls communication
timeout; no rank-local fallback is attempted after a shared-read failure.

Managed allocations, multi-host groups, absent CUDA peer access, and overlapping
destination envelopes are unsupported by this backend. Disjoint views whose
bounding envelopes overlap require rank-local completion before bulk routing
finishes. Small loads can cost more because metadata exchange and IPC mapping add
fixed work. Offloaded embedding tables continue to use their separate storage path.

`io.shared_reads` records preparation, planning, execution, mapping retirement,
physical reads, destination and peer bytes, and reserved staging. Read-task seconds
sum concurrent calls and are not wall time; native idle seconds measure waits for
read or scatter completion. The enclosing completion time includes collective
waits. Destination and peer counters describe writes issued by that reader rank
across all destinations. These startup costs do not enter graph capture or replay.

## Metadata, conversions and accounting

Scalar and explicitly declared control values are coalesced into owned CPU spans,
with a 64 MiB aggregate span limit per session, including intervening bytes. These
values may be copied to GPU parameters. Signed FP4 payloads and E8M0 scale views
preserve their raw bytes, including TP slices; they do not undergo numeric FP8
conversion. Other supported contiguous casts reuse one 8 MiB input allocation
and execute their numerical conversion on the GPU. Unsupported strided casts
fail explicitly.

Logs separate physical input bytes, selected payload bytes, direct destination
bytes, alignment/strided copies, BF16 expansion, other casts, metadata copies,
reserved scratch and final parameter bytes. `gds_physical_bytes` excludes CPU
header/metadata reads; `physical_bytes` includes both. `gds_version` records the
runtime cuFile version. Optional cuFile statistics expose NVFS, P2P, POSIX and
error counters for transport qualification.

These bounds cover loader scratch and explicitly declared metadata. Existing
quantization/model callbacks can allocate full materialized tensors, so loader
counters do not establish a bound on aggregate transform memory. A shared
target/draft transform budget is unsupported. Disk I/O and blocking waits belong
to startup, outside CUDA graph capture and replay.

## Qualification

`tests/loader/test_gds_checkpoint.py` exercises exact bytes, all BF16 bit patterns,
strided TP rows and destination padding, aligned final reads, unaligned EOF,
file and device-storage offsets beyond 4 GiB, bounded casts, metadata copies,
invalid CUDA allocation bounds, overlaps, truncation, transport counters, and graph
replay after session close. Run it on a GDS filesystem with the assigned GPU:

```sh
CUDA_VISIBLE_DEVICES=GPU_UUID CUDA_HOME=/path/to/cuda \
python -m pytest tests/loader/test_gds_checkpoint.py \
  --basetemp=/gds-filesystem/checkpoint-tests
```

`tests/loader/test_shared_read_plan.py` compares the collective planner with an
independent byte oracle. `tests/loader/test_shared_checkpoint.py` uses four owner
processes and covers delayed and empty ranks, rank-local completion, repeated
epochs, offsets beyond 4 GiB, BF16 expansion, poisoned padding, validation failure
propagation, source identity changes, truncation after preflight, and graph replay
after IPC and reader cleanup. Select exactly four assigned GPUs:

```sh
CUDA_VISIBLE_DEVICES=GPU_UUID_0,GPU_UUID_1,GPU_UUID_2,GPU_UUID_3 \
CUDA_HOME=/path/to/cuda python -m pytest \
  tests/loader/test_shared_read_plan.py tests/loader/test_shared_checkpoint.py \
  --basetemp=/gds-filesystem/shared-checkpoint-tests
```

`benchmarks/gds_layer_exchange.py --methods shared_native` exercises the production
collective reader on original routed-expert checkpoint slices. Add
`--all-expert-layers` for the complete expert corpus. Its timed boundary includes
reader initialization, collective planning, IPC mapping, reading, scattering and
cleanup. Destination allocation, source routing, process setup and the byte oracle
are outside that interval; the command records source and native-library hashes.

`tests/loader/test_vllm.py` qualifies model routing using ordinary PyTorch
allocations. `tests/loader/test_direct.py` covers the asynchronous ring
transport and normal CUDA tensor lifetime. Full serving qualification must
also exercise the selected model's actual launch script, preparation, CUDA graph
capture/replay, repeated requests and cache hits. Transport unit tests alone do
not qualify a model for serving or establish startup performance.
