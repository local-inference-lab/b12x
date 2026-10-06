# b12x has moved to FlashInfer

This repository is archived. Development, bug reports, and pull requests belong
in [FlashInfer](https://github.com/flashinfer-ai/flashinfer). The b12x inference
kernels and their preparation, autotuning, compilation, workspace, and checkpoint
loading infrastructure live under
[`flashinfer/experimental/b12x`](https://github.com/flashinfer-ai/flashinfer/tree/main/flashinfer/experimental/b12x).

The integration was merged in
[FlashInfer #5767](https://github.com/flashinfer-ai/flashinfer/pull/5767), commit
[`fc8fdc17f702371302a0ea780c604b099b927196`](https://github.com/flashinfer-ai/flashinfer/commit/fc8fdc17f702371302a0ea780c604b099b927196).
The standalone `b12x` releases and this repository remain available as historical
artifacts; further development is in FlashInfer.

- [Migrate an application](#migrate-an-application)
- [Technical layout and contracts](#technical-layout-and-contracts)
- [Port a b12x pull request](#port-a-b12x-pull-request)
- [Instructions for coding agents](#instructions-for-coding-agents)

## Migrate an application

### Install FlashInfer with the b12x extra

The distribution name changes from `b12x` to `flashinfer-python[b12x]`.
**Python imports remain `b12x`.** FlashInfer supplies the compatibility package;
the standalone distribution is no longer needed.

Use a FlashInfer revision containing the merge commit above. The source install
below pins that commit so it does not depend on a wheel release containing b12x.
For subsequent changes, select a tested FlashInfer commit that descends from it.

Use Python 3.10+ and an environment with a CUDA-compatible PyTorch installation
meeting the selected revision's requirements. The imported b12x subsystem targets
SM120/SM121 GPUs, including RTX 5090, RTX PRO 6000 Blackwell, and DGX Spark;
individual operations may have narrower hardware and configuration support.
At the merge commit, the `b12x` extra requires PyTorch 2.12+ and CUTLASS DSL 4.7.1,
with `rich` and `safetensors` for loading and progress display. Let the selected
revision's `pyproject.toml` define the dependency versions.

Run these commands outside the archived b12x checkout:

```bash
git clone --recursive https://github.com/flashinfer-ai/flashinfer.git
cd flashinfer
git checkout fc8fdc17f702371302a0ea780c604b099b927196
git submodule update --init --recursive

python -m pip uninstall -y b12x
python -m pip install 'setuptools>=77' 'packaging>=24' 'apache-tvm-ffi>=0.1.11,<0.2'
python -m pip install --no-build-isolation '.[b12x]'
```

Remove standalone `b12x` **before** installing FlashInfer: both distributions
provide the same top-level package, so installing or uninstalling one over the
other can overwrite or remove files. If both were installed together, uninstall
`b12x`, then reinstall the selected FlashInfer build. Remove archived b12x
checkouts from `PYTHONPATH` and editable-install configuration as well; running
Python from this repository can shadow the installed package.

Replace standalone `b12x` requirements in application manifests, lockfiles, and
container builds with the selected FlashInfer distribution/revision and its
`b12x` extra. A dependency on the distribution named `b12x` is not satisfied by
FlashInfer merely providing the same import, and would reinstall the archived
package. Regenerate the affected lockfiles.

### Keep the compatibility imports

Existing imports continue to use the compatibility namespace:

```python
import b12x
from b12x.preparation import PreparationSession
from b12x.loader import DirectWeightSession
```

`flashinfer.experimental.b12x` is an implementation location, not a supported
consumer import path. Do not rewrite application imports to that namespace.
The compatibility package loads the embedded implementation with shared module
identity, registries, and runtime state.

Check which installation Python actually uses:

```bash
python - <<'PY'
from importlib.metadata import packages_distributions, version
import b12x

print('FlashInfer version:', version('flashinfer-python'))
print('b12x implementation:', b12x.__file__)
print('b12x distribution providers:', packages_distributions().get('b12x'))
PY
```

The implementation path should end in `flashinfer/experimental/b12x/_api.py`,
and the package provider should be `flashinfer-python`. Version checks that
previously queried distribution metadata for `b12x` must query
`flashinfer-python` instead. Pin the FlashInfer revision together with the
serving framework revision used to validate it.

The compatibility API remains experimental. Its location inside FlashInfer does
not make it a stable FlashInfer API or extend b12x kernels to every GPU FlashInfer
supports. Installing the extra does not automatically route stable FlashInfer
APIs through b12x. Existing direct `b12x` calls do not require
`FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS`; that variable governs automatic
selection of experimental backends by FlashInfer APIs.

### vLLM and checkpoint loading

FlashInfer supplies the framework-independent `b12x.loader` interfaces:
`DirectWeightSession`, `SharedReadGroup`, `CheckpointDisplay`, and `capabilities`.
The serving framework owns model-loader registration, model-specific routing,
and its integration with preparation and execution.

FlashInfer does **not** install the `b12x_loader` or `b12x_fp6` vLLM plugin entry
points, or the archived `b12x.integration.vllm` package. Remove those plugin names
from explicit plugin lists and use a vLLM revision that contains the required
native adapter. `--load-format b12x` works only when that vLLM checkout registers
the loader; installing FlashInfer alone does not add the option to stock vLLM.
The [local-inference-lab vLLM repository](https://github.com/local-inference-lab/vllm)
contains the integration development branches. Check the adapter and its tests
on the branch you deploy.

Validate the actual model, checkpoint format, tensor/expert parallel layout,
warmup, and CUDA graph capture/replay after changing the installation. A successful
import or kernel unit test does not establish serving compatibility.

## Technical layout and contracts

Paths below are relative to the respective repository roots.

| Archived b12x location | FlashInfer location |
| --- | --- |
| `b12x/<group>/...` | `flashinfer/experimental/b12x/<group>/...` |
| `b12x/_lib/`, `b12x/preparation/`, `b12x/loader/` | The corresponding directories under `flashinfer/experimental/b12x/` |
| `b12x/__init__.py` public registry and exports | `flashinfer/experimental/b12x/_api.py` |
| Standalone package installation | `flashinfer-python` supplies the shim in `flashinfer/experimental/b12x/_compat/b12x/__init__.py` |
| `tests/<group>/...` | `tests/experimental/b12x/<group>/...` |
| `benchmarks/...` | `benchmarks/experimental/b12x/...` |
| `b12x/integration/vllm/...` and plugin registration | Framework-owned implementation in vLLM; no FlashInfer package mapping |
| `docs/`, `validation/`, release scripts and evidence artifacts | No blanket transfer; add only what the FlashInfer change requires |

The source layout changes; ordinary `b12x.*` imports inside kernels and tests
do not. Imports of benchmark helpers must account for the benchmark package
move, for example `benchmarks.experimental.b12x.common`. Tests that locate the
repository root or fixture files must account for their deeper directory.

Preserve the execution boundary: an operation declares a plan;
`PreparationSession` selects, compiles, allocates, and primes its execution;
binding and replay use prepared state. Keep live request counts out of compile
and tuning keys, and preserve fixed workspace capacity, allocation stability,
and collective preparation ordering. Keep shared machinery in the embedded
subsystem instead of duplicating it for individual kernels.

Port against the implementation in FlashInfer. For example, preparation's
allocator integration uses PyTorch facilities there; copying the archived
preparation C++ extension back into FlashInfer would undo that change. vLLM
plugins, the `transformers` dependency, benchmark tests in the unit-test tree,
and migration evidence machinery were also deliberately excluded. Native
sources that a feature actually needs must be reviewed and packaged explicitly.

FlashInfer's [experimental policy](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/experimental/README.md)
governs containment and graduation. Keep kernel logic, compilation, and optional
dependencies out of eager core imports, and do not register experimental kernels
for AOT builds. A native FlashInfer API or stable-API backend requires its own
reviewed interface and opt-in behavior; it is not created by renaming an import.

## Port a b12x pull request

Open the port against `flashinfer-ai/flashinfer:main`. The archived PR remains a
source of the proposal, discussion, and authorship, not a place to land code.

1. **Inspect the source PR and its dependencies.** Read the full diff, review
   threads, follow-up commits, and any prerequisite PRs. Record the b12x base
   and head SHAs. For a PR based on an integration branch, separate its actual
   change from unrelated branch history.
2. **Compare with FlashInfer before porting.** Some b12x PRs were integrated
   with fixes, some were rejected, and b12x also received changes after the
   imported revision. Neither an open/merged PR badge nor the tip of archived
   `master` proves that a change is present in FlashInfer. Inspect the destination
   code and tests, and port only the missing behavior. Reassess correctness,
   architectural fit, and complexity versus benefit.
3. **Work on a branch from FlashInfer `main`.** Read its
   [contribution guide](https://github.com/flashinfer-ai/flashinfer/blob/main/CONTRIBUTING.md),
   [root agent guidance](https://github.com/flashinfer-ai/flashinfer/blob/main/CLAUDE.md),
   and [experimental agent guidance](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/experimental/CLAUDE.md).
   Use an isolated worktree if another checkout has work in progress. For
   development, use `python -m pip install --no-build-isolation -e '.[b12x]'`.
4. **Adapt the change at the mapped paths.** A plain cherry-pick can recreate
   `b12x/`, `tests/`, or `benchmarks/` at the wrong locations. Review the resulting
   tree, imports, packaging, fixtures, preparation contracts, and call sites.
   Keep framework changes in the framework repository. Preserve original commit
   authorship when replaying commits, and link the source PR in the port.
5. **Validate the changed path.** Bring focused correctness tests using b12x's
   own references. Exercise supported shapes, dtypes, quantization semantics,
   graph replay, and relevant multi-GPU behavior. For performance changes,
   measure the FlashInfer base and the port with the same benchmark, hardware,
   shapes, and timing method. Record absolute timings, commands, revisions,
   correctness results, and untested configurations. Use the subsystem's
   preparation/timing infrastructure rather than adding a parallel harness.
6. **Submit with FlashInfer's PR template.** Describe the resulting behavior
   and technical reason, link the original b12x PR and dependencies, complete
   the Experimental Track section, and supply the required tracking issue.
   Retain the `experimental-tests` fence and list the affected test files or
   directories under `tests/experimental/b12x/`; CI does not accept pytest
   `::test_name` selectors in that fence. Run the required pre-commit checks.

From the FlashInfer checkout, install test dependencies and explicitly select
the experimental tests; the ordinary `pytest tests/` traversal excludes them:

```bash
python -m pip install pytest -r requirements-test.txt
python -m pytest --collect-only -q tests/experimental/b12x/
python -m pytest -q tests/experimental/b12x/test_registry.py
```

The registry test is a small installation check. Run the affected component's
tests as well; collecting the suite is not executing it. Benchmark modules are
invoked from the FlashInfer root with their mapped package name, for example:

```bash
python -m benchmarks.experimental.b12x.benchmark_moe --help
```

Check each benchmark's arguments before choosing the workload. Preserve the
reference and timing semantics when comparing a port with its base.

## Instructions for coding agents

Give the agent the original PR URL, a FlashInfer checkout, and any required
hardware, model, or serving configuration. The following prompt can be copied
and filled in:

```text
Port b12x PR <URL> into my FlashInfer checkout at <PATH>.
Target hardware and serving configuration: <DETAILS>.

Read the source PR, all review threads, and prerequisite changes. Read the
FlashInfer checkout's AGENTS.md, CLAUDE.md, flashinfer/experimental/CLAUDE.md,
CONTRIBUTING.md, and PR template. Follow destination-repository instructions;
the archived b12x repository's release and validation workflows are historical.

Inspect the worktree and environment before changing them. Start from FlashInfer
main in a separate worktree if necessary. Identify source and destination SHAs,
check whether the behavior is already present, and list actual prerequisites.
Do not copy the archived branch wholesale or assume the source PR is correct.

Map b12x/ to flashinfer/experimental/b12x/, tests/ to
tests/experimental/b12x/, and benchmarks/ to benchmarks/experimental/b12x/.
The public root exports live in _api.py; the compatibility shim is under
_compat/b12x/. Keep consumer imports as b12x. Adapt benchmark imports and
fixture paths. Do not restore standalone packaging, vLLM plugins, transformers,
the preparation C++ allocator extension, or migration evidence machinery.

Review correctness, architectural fit, and complexity versus benefit. Preserve
preparation/bind/replay boundaries, runtime live counts, workspace stability,
quantization contracts, and collective ordering. Keep framework adapters in
the framework repository and stable FlashInfer behavior unchanged by default.

Implement only the missing change and necessary fixes. Validate using the
component's own references and the real affected execution path. For performance
claims, benchmark the FlashInfer base first and preserve shapes and timing
semantics. Report exact commands, revisions, hardware, raw results, failures,
skips, and any serving or hardware coverage still missing. Do not present test
collection or a proxy workload as execution of the requested acceptance case.

Prepare the diff and a FlashInfer PR description using its template, including
the source PR link, authorship, dependencies, Experimental Track, and specific
experimental-tests paths. Summarize what was already upstream, what changed,
why it belongs, and the validation limits. Do not publish or merge unless asked.
```
