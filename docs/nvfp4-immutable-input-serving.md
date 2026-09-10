# Immutable NVFP4 input-scale serving evidence

Status: **implemented**, with bounded GLM TP4 and Qwen TP1 prefill qualification.
The equality proof enables one input quantization per token when all routed
experts use exactly the same immutable input scale. It preserves the canonical
scale vectors; it neither averages scales nor requantizes checkpoint weights.

The serving gains below require both the separate FC1/FC2 kernels from
[B12X #353](https://github.com/local-inference-lab/b12x/pull/353) and the
immutable inference-weight declaration in
[vLLM #727](https://github.com/local-inference-lab/vllm/pull/727).
They are not measurements of this preparation change alone.

The launch predicate reads prepared metadata and, for versioned tensors, the
version counter. It performs no scale-value inspection, policy lookup, device
allocation or reduction. CUDA graph replay does not execute that Python
predicate. Eager mutation invalidates the proof until weights are prepared
again; a long-lived cached binding boolean must not bypass this invalidation.
Inference-tensor owners guarantee immutability for the graph lifetime.

## Hardware, artifacts and source boundaries

Every comparison uses RTX PRO 6000 Blackwell Workstation GPUs at a 600 W power
limit, graphics offset zero and **VRAM offset +6000**. These are not stock-clock
measurements. GLM runs sequentially on host `aiserver`, physical GPUs 4–7;
Qwen uses physical GPU 4 alone. GPU UUIDs, immutable image identities and
configuration are recorded in [the conditions receipt](evidence/nvfp4-immutable-input/conditions.json).

The reference image is
`localinferencelab/vllm@sha256:c9ad4a6ef4aa55232df9ed1a37e85d94eb8c7d5349561a6cbe828e72b61de83c`,
with B12X `3edbcbce70f491741b82f5eab9c1b30b39447228` and vLLM
`5576927057cf71b6ec61d120932338b333efa089`.
The measured candidate uses B12X `8e6427240e37304f9cdfca23d38923b31b8e9c84`
and vLLM `ae89131442359dc332d9c46009be3c1f8cdee0b4`.
The candidate B12X source includes #353 through `87bde512`, the immutable-scale
proof, and explicit direct-scale setup in a mutable-owner test.

The complete attributed histories are published in
[the B12X integration branch](https://github.com/voipmonitor/b12x/tree/release/jovian-nvfp4-split-r33-20260910)
and [the vLLM integration branch](https://github.com/voipmonitor/vllm/tree/codex/jovian-nvfp4-split-r33-20260910).
Check out the measured revisions above to reproduce this table; moving branch
heads are not its measurement boundary. The physical worktrees were
`/root/vllm/worktrees/b12x-glm-nvfp4-split-prefill-20260910` and
`/root/vllm/worktrees/vllm-glm-nvfp4-scale-sharing-20260910`.

GLM checkpoint: `local-inference-lab/GLM-5.3-Flash-NVFP4`, revision
`46aaae8a82032f77100f2f03e9cc11b391df3b4d`. Qwen checkpoint:
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4`, revision
`b797d2e1160b9596b2570e56c1d3590faa09d4ed`. No checkpoint tensors change.

## Real serving route and commands

The GLM launcher and client are retained beside the receipts:
[launch_glm_nvfp4_split_local.sh](evidence/nvfp4-immutable-input/launch_glm_nvfp4_split_local.sh),
[measure_exact_cold_prefill.py](evidence/nvfp4-immutable-input/measure_exact_cold_prefill.py).
The launcher mounts only the candidate B12X package and vLLM MoE adapter over
the immutable reference. The control has no source mounts. The recorded order
is reference, shared-input monolithic, separate-FC1/FC2, reference.

Commands were executed from `/root/vllm/glm53f`, stopping only the owned arm
between measurements:

```bash
bash diagnostics/launch_glm_nvfp4_split_local.sh reference a1
uv run --no-project --with httpx python diagnostics/measure_exact_cold_prefill.py \
  --base-url http://127.0.0.1:5241 --tokens 32768 --duration 30 \
  --output results/nvfp4-split-prefill-local-20260910/reference-a1-prefill32k.json
```

Substitute `shared b1`, `split c1`, and `reference a2` with distinct output
paths for the other arms. The copied launcher and client specify TP4/DCP1,
no speculation, FP8 target KV, a 4096-token scheduler budget, OMP1,
16 NCCL channels, a 2 MiB NCCL buffer and full-and-piecewise graphs.
Prefill uses temperature 0, one excluded warmup, at least 30 measured seconds,
32768 input token IDs and one output token. Each request has a fresh leading
nonce; all prompt tokens must be locally computed. The metric is input tokens
divided by client time to first token, not a sum of isolated kernel times.

Qwen uses the retained
[TP1 launcher](evidence/nvfp4-immutable-input/run_qwen_cookbook_source_qualification.sh)
with `PHYSICAL_GPU=4`, `MTP_NVFP4_LM_HEAD=1`, `VLLM_LM_HEAD_A16=1`,
`MAX_NUM_SEQS=16` and `QUALIFICATION_CAPTURE_SIZE=64`.
Its reference image is identical to GLM's; the candidate immutable source-overlay
image has ID `sha256:07043bc67615e67d2e476e7342a4ba35ad8cab78b078834852f23e99e66e495a`.
Each image contains its serving source; neither Qwen launch mounts source files.
The Qwen client command, from `/root/vllm/qwen38next`, is:

```bash
uv run --no-project python tools/measure_token_id_prefill.py \
  --base-url http://127.0.0.1:5242 --model Qwen3.8-Flash-Next \
  --tokens 32768 --warmups 2 --samples 5 --seed 20260908 --temperature 1 \
  --output prefill32k.json
```

The client is [included here](evidence/nvfp4-immutable-input/measure_token_id_prefill.py).
Qwen uses TP1/MTP3, CPU PLE offload, FP8 KV, OMP2, a 6019-token scheduler
budget, temperature 1/top-p 0.95/top-k 20. The target head remains BF16;
the private NVFP4 draft head and W4A16 draft MoE are unchanged.

## Measurements and limits

| Real serving measurement | Reference input tok/s | Split input tok/s | Candidate/reference − 1 |
|---|---:|---:|---:|
| GLM 32K, bracketed controls | 15842.49 / 15737.30 | 17152.43 | +8.27% / +8.99% |
| Qwen 32K, HTTP wall time | 15707.42 | 17211.35 | +9.57% |

Decode concurrency means simultaneous active clients: **C1 is one client**
and **C8 is eight clients**, with C8 rates aggregated across clients.
The decode workload uses context argument zero, temperature 1, respected EOS,
a ten-second warmup and a thirty-second measured cell.

Every measured and excluded warmup latency is retained in the adjacent JSON
receipts. The [serving summary](evidence/nvfp4-immutable-input/serving-summary.json)
also preserves every GLM C1 observation, including the approximately 177 tok/s
state on both reference and candidate. No decode speedup is established.
The Qwen C1/C8 verifier rates differ by less than 1%; output-rate changes include
stochastic proposal acceptance and are not a statistical decode-equivalence test.
Their complete receipts are [the reference decode](evidence/nvfp4-immutable-input/qwen-reference-decode.json)
and [the split decode](evidence/nvfp4-immutable-input/qwen-split-decode.json).
Artifact fields in the GLM summary identify the locally recorded origin files;
the summary contains all aggregate observations, not every generated transcript.

The included clients preserve the measured request and timer semantics on the
qualified domain. Input validation additionally rejects invalid durations and
absent cache evidence; those cases were not accepted by the serving qualification.
The GLM client includes only its prompt-counter dependency, not an unrelated
external-cache restore command. The Qwen command here qualifies vLLM TP1 only.

The actual GLM rank-0 prefill trace records 168 calls each to routing, FC1 and
FC2 kernels (42 layers × four target forwards). Qwen records 288 each
(48 layers × six forwards). These traces establish real serving dispatch.
The unprofiled client latencies above, not the trace sums, define the gain.

Independent operation oracles pass the established mean-cosine ≥0.999 and
normalized-RMSE ≤0.03 bounds. GLM split cosine/RMSE are 0.999981/0.00603;
Qwen split results are 0.9999785/0.006455. Both improve on the corresponding
monolithic oracle errors. **Strict cross-kernel cosine 0.9999 fails** (about
0.999889 GLM and 0.999874 Qwen): no bit-exact logits, identical token sequence,
or target-distribution equivalence is claimed. Retain this limitation when
reusing the performance results.

Graph replay, live-row capacity, caller-owned scratch, nonzero output, immutable
proof invalidation and mutable-owner behavior have focused tests. GLM passes
three 32K literal lookups (nine values); Qwen passes 68 cold/repeated requests,
six instruction-prefix cases and 38 post-prefill logprob requests. This is bounded
correctness coverage, not a general model-quality evaluation. TP2 Qwen has
intermediate width 320 and is not eligible for the split tile's 128-element
constraint. No TP2 speedup is claimed.
