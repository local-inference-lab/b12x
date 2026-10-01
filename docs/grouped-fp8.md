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
  releasing the session. Contiguous plans own mutable, fixed-capacity scale
  workspaces: serialize calls/replays or use separate plans and outputs.
  Masked eager calls allocate scale-packing buffers; graph replay reuses the
  captured allocations. No general allocation-free eager claim is made.

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

## Qualification, not a universal performance promise

The upstream-adapted active-group candidate (semantic version 3, candidate
contract 8) was measured with the mega-repository's xcheck and its specified
DeepGEMM fork (`d5e3bbfb91ebb83ea8d4aff4befacbba61d711e7`), not an arbitrary
upstream DeepGEMM installation. The tested combination is CUDA toolkit 13.3,
CUTLASS DSL 4.7.1, and RTX PRO 6000 Blackwell Server Edition (SM120, 188 SMs).
Dependencies now pin DSL 4.7.1, matching that tested compiler. SM121 and
other compiler versions are **not qualified**. The 819-case evidence describes
the 2026-09-30 candidate before the upstream 1.5 merge; it does not qualify
all newly merged upstream runtime paths.

The independent 2026-09-30 round passed **819/819** cases: 373 prefill and 446
decode across 47/56 slices. All phases satisfy b12x/DeepGEMM <=1.01. Initial
geometric mean is 0.8213204489; geometric mean of each case's worst phase is
0.8215264918; worst phase is 1.0055542348. The
[819-row performance table](benchmarks/wi004-grouped-fp8-upstream-interleaved-20260930.csv)
includes geometry, initial/confirmation ratios and task/attempt references.
It is a summary, not the raw samples or a standalone benchmark harness.

`interleaved-v1` uses cold-L2 CUDA events (retained graph replay for decode):
A=DeepGEMM, B=b12x; initial five-sample ABBA blocks repeated three times give
30 samples per side. Fixed risk/history or initial ratio >=0.99 requires
separate ten-sample ABBA x10 and BAAB x10 confirmations, 200 samples per side
per phase. Each phase has pre/post correctness checks and five warmups per
side, without block rewarmup. All raw samples per side are pooled for the
median; every phase, including initial, must pass without trimming or drift
exemptions. This round contains 1,251 paired phases, 2,502 records and 221,940
raw samples. Representative/A/A diagnostics are excluded.

Earlier sequential-protocol failures remain failures: task 599's GLM disjoint
case reached 1.0137543356; task 617's DSV4.1 disjoint case reached 1.0225888823.
The new approved-protocol round does not retroactively qualify those runs.

Validation also includes 202 grouped tests and targeted memcheck, racecheck
and synccheck on two active-marker/joint graph cases each. Those sanitizer
checks do not cover all 819 cases. Existing GPU regression evidence covers
additional dense paths, but is not qualification of the entire upstream suite.
The final combined working tree was tested; intermediate commits in the local
series were not separately GPU-qualified. EP1/EP8 scaling acceptance remains
unchanged and is not independently concluded by this performance table.

From a configured b12x checkout, run its existing tests with:

```sh
python -m pytest tests/gemm/test_mgroup*.py -q
python -m pytest tests/test_registry.py tests/preparation -q
```

GPU cases need the target hardware and supported dependencies. Full 819
reproduction additionally needs the mega-repository's xcheck corpus, formal
prefill/decode drivers, frozen b12x candidate and the specified DeepGEMM fork;
it cannot be reproduced by an independent b12x-only command. Optimized defaults
are identity/geometry-scoped; availability alone is not performance evidence
for other devices, configurations or live distributions.
