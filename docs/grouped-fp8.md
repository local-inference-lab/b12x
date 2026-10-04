# Standalone grouped FP8 GEMM

`b12x.gemm.mgroup_fp8_gemm` is a prepared CuTe DSL operation for
DeepGEMM-style masked decode and contiguous-label prefill. It accepts E4M3
values and plain FP32, positive power-of-two scales representable in UE8M0,
accumulates in FP32, and writes caller-owned BF16 output. K must divide by 128.
Scale values are a caller contract, not generally range-validated.

## Tensor and lifetime contracts

- B is contiguous `(G, N, K)`, with SFB `(G, N, K/128)` in both modes.
- Masked A is `(G, M, K)`, SFA `(G, M, K/128)`, output `(G, M, N)`.
  Device-int32 `masked_m[G]` contains live counts in `[0, M]`; counts are not
  range-validated. Rows beyond each count are undefined, not guaranteed zero.
- Contiguous A is `(M, K)`, SFA `(M, K/32)`, output `(M, N)`.
  Device-int32 `labels[M]` contains `-1` padding or group IDs `[0, G)`.
  Every non-padding run must start at a multiple of **128**, including a
  repeated group's later run. Padding output is zero. Invalid labels/run
  starts raise an asynchronous CUDA device error, also during replay.
- Values, scales, output and activity tensors must be contiguous and on the
  prepared device. Joint execution also requires 128-byte B base alignment.
- Query geometry and capacity are fixed. Live masks/labels are not compile
  keys; `expected_m` is a fixed planning hint, not a new live count at execution.
  Eager row views may vary within capacity; captured tensor shapes/addresses
  stay fixed while their contents change.
- Prepare before capture, keep operands/plans alive, and destroy graphs before
  releasing the session. Both modes own private, mutable, fixed-capacity scale
  workspaces: serialize calls/replays or use separate plans and outputs.
  Masked calls repack into the existing buffers using the current tensor row
  stride; eager packing and graph replay do not allocate scale storage.

## Prepared public API example

Run this complete block in a b12x environment on a supported GPU. It uses
exact small integer FP8 operands and non-unit scales so an independent
matrix-product oracle can check both modes exactly. Autotuning is disabled
for this usage smoke; normal applications may provide representative producers
and enable it. The preparation callback's state is the session-provided object;
execution uses the public functions, without private imports.

```python
import torch
from b12x.gemm import mgroup_fp8_gemm as mgg
from b12x.preparation import PreparationSession, PreparedCall


def example(mode):
    torch.manual_seed(7)
    device = torch.device("cuda")
    groups, n, k = 2, 128, 256
    rows = 32 if mode == "masked" else 256
    shape = (groups, rows, k) if mode == "masked" else (rows, k)
    gran = 128 if mode == "masked" else 32
    a = torch.randint(-2, 3, shape, device=device).to(torch.float8_e4m3fn)
    b = torch.randint(-2, 3, (groups, n, k), device=device).to(torch.float8_e4m3fn)
    sa = torch.full((*shape[:-1], k // gran), 0.5, device=device)
    sb = torch.full((groups, n, k // 128), 2.0, device=device)
    lhs, rhs = (a, sa), (b, sb)
    d = torch.empty((*shape[:-1], n), device=device, dtype=torch.bfloat16)
    if mode == "masked":
        activity = torch.tensor([7, 0], device=device, dtype=torch.int32)
    else:
        activity = torch.full((rows,), -1, device=device, dtype=torch.int32)
        activity[:64] = 0
        activity[128:200] = 1
    plan = mgg.plan(mgg.query_from_call(lhs, rhs, d))

    def prepare(state):
        run = state.run_masked if mode == "masked" else state.run_contiguous
        return PreparedCall(run=lambda: run(lhs, rhs, d, activity))

    def run():
        verb = mgg.masked_mm if mode == "masked" else mgg.contiguous_mm
        return verb(lhs, rhs, d, activity, plan=plan)

    def check():
        ad = a.float() * sa.repeat_interleave(gran, dim=-1)
        bd = b.float() * sb.repeat_interleave(128, dim=-1)
        if mode == "masked":
            for g, live in enumerate(activity.cpu().tolist()):
                ref = (ad[g, :live].double() @ bd[g].double().T).to(d.dtype)
                torch.testing.assert_close(d[g, :live], ref, rtol=0, atol=0)
        else:
            ref = torch.zeros_like(d)
            for g in range(groups):
                selected = activity == g
                ref[selected] = (ad[selected].double() @ bd[g].double().T).to(d.dtype)
            torch.testing.assert_close(d, ref, rtol=0, atol=0)

    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare((plan.request(name=mode, prepare_call=prepare),))
        session.freeze()
        run()
        check()
        graph = torch.cuda.CUDAGraph()
        try:
            with session.capture(), torch.cuda.graph(graph):
                run()
            if mode == "masked":
                activity.copy_(torch.tensor([1, 17], device=device, dtype=torch.int32))
            else:
                activity.fill_(-1)
                activity[:32] = 1
                activity[128:256] = 0
            d.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            check()
        finally:
            graph.reset()
    print(mode, "eager and live-activity graph oracle passed")


if __name__ == "__main__":
    example("masked")
    example("contiguous")
```

## Qualification and evidence limits

These bounded grouped FP8 measurements compare with a clean DeepGEMM fork base
`d5e3bbfb91ebb83ea8d4aff4befacbba61d711e7`. Ratios are b12x median / DeepGEMM
median (lower is better) on NVIDIA RTX PRO 6000 Blackwell Server Edition
(SM120, 188 SMs). Each round used one physical card; its run-local `gpu-1`
pseudonym does not establish whether both rounds used the same card.

The measured cases cover DeepSeek-V4-Flash, DeepSeek-V4.1-Flash, GLM-5.3 and
GLM-5.3-Flash expert GEMMs (w13/w2), plus small contract cases. Model geometries
include EP1/4/8; legacy expert lengths are synthetic. The separate rank-local
route-tp-v1 corpus includes TP1/4/8, TP rank 0, representative EP ranks and global
token counts 1/128 for decode or 1024/8192 for prefill. These are neither
production routing traces nor TP collective benchmarks.

| Measured tree (contract 8) | Paired cases | Initial ratio geomean | Per-case worst-phase geomean | Worst phase |
|---|---:|---:|---:|---:|
| Semantic 3: prefill + decode | 819 (373 + 446) | 0.8213204489 | 0.8215264918 | 1.0055542348 |
| Semantic 4: decode only | 446 | 0.7318103372 | 0.7318186491 | 0.9903273810 |

Separate sanitized observations are available for
[semantic 3 raw cases](benchmarks/grouped-fp8-semantic3-cases.jsonl.gz) and
[provenance](benchmarks/grouped-fp8-semantic3-provenance.json), and for
[semantic 4 decode raw cases](benchmarks/grouped-fp8-semantic4-decode-cases.jsonl.gz)
and [provenance](benchmarks/grouped-fp8-semantic4-decode-provenance.json).
They retain raw latencies, phase medians, block ordering and selected correctness
summaries, not full private lifecycle bindings, mutation tensors or memory
traces; they cannot independently replay the complete private correctness gate.

Every measured phase passed the <=1.01 threshold with pre/post correctness
checks. Semantic 3 includes 216 confirmed cases, 1,251 paired phases and 221,940
raw samples; semantic 4 includes six confirmations, 458 phases and 31,560 raw
samples. Interleaved-v1 uses cold-L2 events around the full eager API for prefill
and retained CUDA-graph replay for decode. After five warmups per side, the
initial five-sample ABBA x3 schedule yields 30 samples per side. **Fixed
risk/history OR initial ratio >=0.99** triggers both ten-sample ABBA x10 and
BAAB x10 confirmations (200 samples per side per phase). Medians include every
sample; no drift or trimming exemption applies.

At measurement, semantic 3 used b12x Git base
`a489f972e0dde54fedd5f83bf73a7d3754fc60d6`, and semantic 4 used
`56c00e61a86195cac0c3505502ec1095fe8faa5b`, each with dirty/untracked inputs;
the harness root was dirty too. Neither has an exact public measurement
revision. Results apply **only to the measured trees**: semantic 3 prefill and
semantic 4 decode cannot be combined into a same-source 819-case claim.
Reproducing either round exactly from a clean public commit is not supported
by this evidence.

Preparation checks reported Python 3.12.14, PyTorch 2.14.0+cu130 and CUDA
**runtime 13.0**; task results reported driver versions and snapshot clocks/P-state.
The workspace lock associated with these exports lists CUDA **toolkit/nvcc
13.3.73**, CUTLASS DSL 4.7.1 and Triton 3.8.0; those lock values are not
independent historical runtime observations. Throttling-reason telemetry was
not recorded for these two rounds. Only the pinned CUTLASS DSL 4.7.1 is qualified
for this op. These results do not establish SM121 or whole-upstream-suite
qualification, all live distributions, or performance of another source tree.

A separate [PR review regression export](benchmarks/pr451-review-provenance.json)
contains [10 nongrouped dense cases](benchmarks/pr451-review-nongrouped-cases.jsonl.gz)
and [20 grouped representatives](benchmarks/pr451-review-grouped-cases.jsonl.gz).
The nongrouped comparison uses clean base `e4084d2e` versus a dirty tree based on
`56c00e61`; the grouped comparison uses the specified DeepGEMM fork. These are
bounded observations of that measured tree, not a new full grouped matrix.
Positive allocated-GPR deltas remain visible in the provenance alongside the
per-case phase results. EP1/EP8 scaling is not independently established here.

From a configured b12x checkout, run its existing tests with:

```sh
python -m pytest tests/gemm/test_mgroup*.py -q
python -m pytest tests/test_registry.py tests/preparation -q
```

GPU tests require suitable hardware and dependencies; repeating these paired
measurements additionally requires the external corpus, formal drivers, matched
DeepGEMM fork and corresponding measured sources. Public commit IDs alone do not
reconstruct those dirty trees.
