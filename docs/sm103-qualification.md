# SM103 preparation and hardware qualification

Status: **implemented prototype; physical SM103 execution unqualified**.
Use the [feature map](sm103-change-summary.md) for supported operators and the
[readiness report](sm103-readiness-report.md) for evidence limits. Offline
compilation cannot establish correctness, graph behavior, occupancy or speed
on B300. Portable SM120 tests cannot execute SM103-specific cubins.

## Preparation lifecycle

Follow [GPU preparation and startup autotuning](gpu-profiles.md). Declare
immutable model geometry, numerical recipe, capacity, layouts and optional
operand presence. Submit real preparation callbacks; retain the resulting
plans and storage for the graph lifetime. Binding and replay use retained
programs. First-use fallback preparation is a debugging convenience and is
forbidden after freezing or during capture.

A minimal portable projection example is:

```python
import torch
from b12x.gemm import bf16_gemv
from b12x.preparation import PreparationSession, PreparedCall


def main():
    source = torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
    plan = bf16_gemv.plan(bf16_gemv.query_from_call(source, weight))
    request = plan.request(
        name="projection",
        prepare_call=lambda state: PreparedCall(run=lambda: state.run(source, weight)),
    )
    with PreparationSession(device=source.device, autotune=False) as session:
        session.prepare((request,))
        output = bf16_gemv.mm(source, weight, plan=plan)
        torch.cuda.synchronize()
        assert output.isfinite().all() and output.count_nonzero()


if __name__ == "__main__":
    main()
```

For multi-candidate tuning, supply representative benchmark callbacks and
restore any state modified by trials. A heuristic/default choice is not a
measured winner. Integrations supply metadata and capacities; b12x owns tuning
and dispatch. See component tests for scratch binding and graph examples.

## Checks without a B300

Use an isolated environment with the repository's declared CUTLASS DSL version.
Run each compiler with an empty output directory outside the repository.

```bash
python -m pytest tests/architecture tests/preparation -q
python scripts/compile_sm103_prepared.py --workers 4 --output-dir /tmp/b12x-sm103-prepared
python scripts/compile_sm103.py --component all --output-dir /tmp/b12x-sm103-native
python scripts/qualify_sm103.py --output-dir /tmp/b12x-sm103-qualification
```

`compile_sm103_prepared.py` extracts the production program list for each typed
declaration and compiles it through the offline worker contract. Its manifest
records the source hash, toolchain, cases and target. `cases.jsonl` records
program keys and scratch requirements; `native/` contains hashed PTX/cubin
pairs. Trellis cases declare compiler metadata; GPU checkpoint preparation is
qualified separately. Success requires an unchanged package and no CUDA
initialization.

`compile_sm103.py` preserves the broader kernel corpus. Add `--nvdisasm` and
`--cuobjdump` with absolute tool paths to retain SASS and resource reports.
Use `audit_sm103_resources.py --help` for artifact/resource accounting and
`audit_sm103_packing_ptxas.py --help` for the common-assembler diagnostic.
Retain every resource flag; an offline flag is not a measured performance loss.

The qualification command without `--execute` writes a source-bound list of
GPU tests. It launches no GPU work and provides no runtime acceptance. Retain
all failures and skips in the evidence. Host preparation baseline failures
are documented in the validation record rather than counted as passes.

Portable regressions can run on a physically verified SM120/SM121 GPU. Select
its UUID and compile for its actual architecture; do not set an SM103 target
on an SM12x device. Useful component suites include:

```bash
python -m pytest tests/attention/test_sm103_sparse_mla.py tests/attention/test_sm103_compressed_mla.py tests/attention/test_sm103_dsa_indexer.py -q
python -m pytest tests/norm/test_sm103_mhc.py tests/norm/test_sm103_hyperconnection.py tests/sequence/test_gdn_decode_kda_cute.py tests/sequence/test_mtp_feedback.py -q
python -m pytest tests/gemm/test_sm103_fp8.py tests/gemm/test_sm103_block_fp8_linear.py tests/gemm/test_bf16_vocab_projection.py -q
```

Native SM103 tests skip on other architectures; skipped cases are not qualified.
Large-offset tests need several GiB of free memory. Run memory-heavy suites in
separate processes when necessary, retaining the failed combined-run record.

## Physical SM103 acceptance

Verify the actual GPU UUID, capability, SM count, opt-in shared-memory limits,
CUDA mode, driver and toolchain before choosing architecture settings. The
compiler target is `sm_103a` for SM103; it is architecture-specific. Preserve
physical identity and before/after GPU snapshots with every run.

```bash
python scripts/qualify_sm103.py --execute --device-uuid GPU-REPLACE-WITH-ACTUAL-UUID --compile-manifest /tmp/b12x-sm103-native/manifest.json --output-dir /tmp/b12x-sm103-runtime
```

The launcher refuses execution on a non-SM103 device. It verifies matching
source and native artifact hashes when a compile manifest is supplied. A
required test failure or skip fails acceptance. Repeat relevant suites with
`--sanitizer /absolute/path/to/compute-sanitizer --sanitizer-tool memcheck`,
then synchronization/race diagnostics for kernels whose ownership requires it.

Correctness must establish finite, nonzero outputs, numerical oracles,
quantization and rounding semantics, top-k agreement and boundary behavior.
Paged and recurrent pools must include high recycled IDs past the signed
32-bit scaled-offset boundary. Cover zero and multiple live request sizes
under frozen program resolution, stable addresses, fixed workspace capacity,
input mutation and replay without allocation.

Only then measure the real production path. Record command, commit, package
and artifact hashes, worktree, GPU UUID/mode, correctness result, warmup,
raw samples, power/clock/throttle state and ratio direction. Use graph replay
where serving uses it. Do not infer SM103 speed from SM120 timing or a
compile-only resource census.

## Companion, Grace and Station boundaries

The companion vLLM branch needs adaptation to preparation sessions before
full-model tests. Preserve its GLM selection compaction, speculative pool
history and unmapped-slot guards. Run eager, graph and speculative modes on
the same checkpoint and requests, recording accuracy and repeatability.
Historical fixed-request passes do not waive the recorded failed GLM gates.

Engram device/mapped-host owners must outlive every binding and graph. Grace
placement requires live coherence/capability probes. Disk lookups are
synchronous transactions; the compatibility prefetch methods do not implement
an asynchronous prefetch pipeline. Verify data visibility and storage lifetime
on the physical platform.

Experimental Grace TP2 transport requires actual memory registration, NIC and
peer visibility, epoch/reset behavior, failure handling and ordering tests.
Direct HBM RDMA remains unsupported. Operator compilation does not qualify
Station communication or a complete serving deployment.
