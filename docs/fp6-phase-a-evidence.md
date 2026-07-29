# Phase A — FP6 vs FP8 evidence runbook (Behemoth-R1-123B, 2x RTX PRO 6000)

Companion to the performance-recovery plan. This is the **serving-box**
checklist; the Windows workspace only authors the tooling.

## What Phase A must produce

1. **GEMM microbench JSON** from
   [`benchmarks/benchmark_dense_gemm_fp6.py`](../benchmarks/benchmark_dense_gemm_fp6.py)
   covering Behemoth TP=2 shards × M-sweep × `{fp6_expanded, fp6_packed, fp8_cutlass}`.
2. **Serving torch-profile attribution** for FP6 and FP8:
   - one ctx≈8192 prefill (`max_tokens=1`)
   - one decode window (short prompt, `max_tokens=128`)
   summarized by [`scripts/summarize_vllm_trace.py`](../scripts/summarize_vllm_trace.py)
   into GEMM / act-quant / host GS / allreduce / attention budgets.

Do **not** claim Phase-B wins until both artifacts exist with the evidence
header (command, commit, GPU UUID/mode, raw timings, ratio direction).

## 1. GEMM microbench

On the serving box, from a checkout of this branch, with the sparkinfer +
vLLM env that can import `torch.ops._C.cutlass_scaled_mm` (or
`vllm._custom_ops.cutlass_scaled_mm`):

```bash
cd /path/to/b12x
export CUDA_VISIBLE_DEVICES=0   # single GPU microbench is enough
python benchmarks/benchmark_dense_gemm_fp6.py \
  --preset behemoth-tp2 \
  --warmup 10 --iters 50 \
  --json-out /tmp/fp6_phase_a_gemm.json
```

Reading the output:

| Column | Meaning |
|---|---|
| `fp6_expanded` | byte-container B (production narrow-N / losing packed regime) |
| `fp6_packed` | 3:4 packed-B + in-smem expand (production wide-N path) |
| `fp8_cutlass` | vLLM SM120 blockwise FP8 (the serving baseline) |
| `vs_exp` / ratio summary | `>1` = slower than `fp6_expanded` |

Decisions this matrix feeds:

- If `fp6_packed / fp6_expanded > 1` at M≥2048 → packed-B loses on prefill (Phase B item 6).
- If `fp6_packed / fp6_expanded > 1` at M≤16 for qkv (N=7168) → TP-shard packed heuristic is wrong (Phase B item 6a).
- If `fp6_* / fp8_cutlass ≫ 1` at M≥2048 → large-M unroll + tile ladder are the budget (Phase B items 3–4).
- If decode M≤16 FP6 is still behind FP8 after bandwidth adjustment → CTA/tile/split-K (Phase C).

If the CUTLASS arm prints `SKIP: vLLM cutlass_scaled_mm not importable`, rerun
inside the vLLM serving venv (not a bare sparkinfer venv).

## 2. Serving profile (both quants)

Launch both servers with profiler endpoints enabled (`PROFILE=1` is already
wired in the PhaeDawg launch scripts):

```bash
# Terminal A — FP6
PROFILE=1 PROFILE_DIR=/tmp/vllm_prof_fp6 \
  CUDA_VISIBLE_DEVICES=0,1 TP_SIZE=2 \
  ./behemoth123b-r1-v2-fp6.sh "$API_KEY"

# Terminal B — FP8
PROFILE=1 PROFILE_DIR=/tmp/vllm_prof_fp8 \
  CUDA_VISIBLE_DEVICES=0,1 TP_SIZE=2 \
  ./behemoth123b-r1-v2-fp8.sh "$API_KEY"
```

Then from the sparkinfer checkout:

```bash
export VLLM_API_KEY=...
export PROFILE_DIR=/tmp/vllm_prof_fp6   # capture script also copies newer traces
BASE_FP6=http://127.0.0.1:8000 \
BASE_FP8=http://127.0.0.1:8001 \
  ./scripts/phase_a_profile_behemoth.sh "$VLLM_API_KEY"
```

Or attribute an existing PROFILE_DIR manually:

```bash
python scripts/summarize_vllm_trace.py /tmp/vllm_prof_fp6 /tmp/vllm_prof_fp8 \
  --json-out /tmp/fp6_phase_a_attr.json
```

Buckets that matter for Phase B:

| Bucket | Phase-B lever if oversized on FP6 |
|---|---|
| `fp6_gemm` | large_m_unroll, tile ladder, packed-B policy |
| `fp6_act_quant` | TMA quant cost (usually small) |
| `host_gs_chain` | fuse large-M per-row GS into TMA quant |
| `allreduce` | shared with FP8 (NCCL); not an FP6 kernel bug |
| `attention` | shared; KV growth with ctx — not Phase B |

## 3. Hand-back artifacts

Copy these off the serving box before starting Phase B:

- `/tmp/fp6_phase_a_gemm.json`
- profile `OUT_ROOT/` (includes `attribution.json` + `evidence_header.txt`)
- optional: the raw `*.pt.trace.json.gz` files

Paste the ratio-summary table and the side-by-side bucket ms table into the
PR / plan notes so Phase B item selection is evidence-tied.
