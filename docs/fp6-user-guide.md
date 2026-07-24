# SparkInfer MX-FP6: making and serving FP6 quants

End-to-end guide: how to produce an MX-FP6 checkpoint from a BF16 HF model,
and how to serve it with SparkInfer + nightly vLLM. Reference numbers
(classification, KLD, performance) live in `fp6-results.md`; kernel-level
format details in `mxfp6-w6a8.md`; plugin internals in
`mxfp6-vllm-integration.md`.

Requirements: NVIDIA Blackwell GPU (sm_120 validated; the GEMM uses the
block-scaled `mxf8f6f4` MMA), CUDA 13 stack, Python >= 3.12.

---

## 1. Making an FP6 quant

### 1.1 The converter

One script handles both dense and MoE models:

```bash
python scripts/quantize_model_fp6.py \
  --model /path/to/bf16-hf-model \
  --out   /path/to/output-fp6 \
  --arch  auto
```

It walks the HF checkpoint (`config.json` + `*.safetensors`), quantizes the
eligible 2-D Linear weights to packed MX-FP6 (E2M3, 6-bit codes + UE8M0
block scale per 32 values), and writes a full loadable HF checkpoint:
sharded `model-*.safetensors` + index, a patched `config.json` carrying
`quantization_config = {"quant_method": "modelopt", "quant_algo": "W6A6"}`
with ModelOpt-mirror tensor keys (`.weight` / `.weight_scale` /
`.weight_scale_2` / `.input_scale`), and copies of the tokenizer/chat
template files. Routed MoE experts are written per-expert.

**What gets quantized (the "golden rule" walk):** MLP projections,
attention q/k/v/o, MoE routed experts (gate/up/down) and the shared expert.
**Kept BF16:** norms, embeddings, router gates, `lm_head`, the MTP head,
linear-attention (SSM/DeltaNet) projections and the vision tower — the last
two can be opted in (below). Exclusions are recorded in
`quantization_config.exclude_modules`.

Preview what a run will do without writing anything:

```bash
python scripts/quantize_model_fp6.py --model <dir> --out /tmp/x --arch auto --dry-run
```

### 1.2 Quant-time knobs

| Flag | Default | What it controls |
|---|---|---|
| `--arch {auto,moe,dense}` | `auto` | Model family; `auto` detects routed experts from the checkpoint. |
| `--source-format` | `mxfp6_w6a8` | Weight/activation format pairing. `mxfp6_w6a8`: E2M3 weights + FP8 E4M3 *runtime* activations (best KLD, the validated default). `mxfp6_default`: E2M3 + E3M2 activations (legacy true-W6A6). `mxfp6_e2m3` / `mxfp6_e3m2`: symmetric pairings. |
| `--block-scale-rule {mse,ceil}` | `mse` | UE8M0 weight block-exponent selection. `mse` tries per-block ceil vs ceil-1 and keeps the lower error; `ceil` is amax containment only. Weights only — runtime activation quant is unaffected. |
| `--no-attention` | off | Skip attention q/k/v/o (weight-only MLP/expert quant). |
| `--include-linear-attn` | off | Also quantize linear-attention (DeltaNet/SSM) `in_proj_*`/`out_proj` matmuls. Shrinks hybrid multimodal checkpoints below FP8 size; quality trade-off — validate KLD downstream. (The published Qwen3.6-27B "`_la`" build uses this.) |
| `--include-vision` | off | Also quantize vision-tower block linears. |
| `--activation {silu,relu2}` | `silu` | MoE only: expert activation function baked into the fused kernel selection. |
| `--gate-up-order {gate_up,up_gate}` | `gate_up` | MoE only: row order of the source fused FC1 weight. |
| `--skip-experts` | off | MoE sensitivity diagnostic: keep routed experts BF16, quantize everything else. Not for production. |
| `--skip-shared-expert` | off | MoE sensitivity diagnostic: keep shared-expert linears BF16. |
| `--report-error` | off | Dequantize every emitted tensor and print a per-group relative-RMSE table (experts gate/up/down, shared_expert, self_attn, linear_attn, ...). |
| `--limit-layers N` | all | Convert only the first N layers (bring-up/testing). |
| `--no-gpu` | off | MoE only: slow torch reference quantizer instead of the GPU path. |
| `--format {safetensors,pt}` | `safetensors` | `safetensors` = full HF checkpoint (what vLLM serves). `pt` = per-layer kernel-validation artifacts (name is historical — output is still safetensors; pickle is never used). |
| `--inspect LAYER` | — | Print every checkpoint key + shape under `.layers.{LAYER}.` and exit. |
| `--dry-run` | off | Print the conversion plan; write nothing. |

Practical guidance: leave `--source-format` and `--block-scale-rule` at
their defaults — `mxfp6_w6a8` + `mse` is the combination all published KLD
numbers were measured with. The knobs that meaningfully change the
size/quality trade are `--include-linear-attn` and `--include-vision`; use
`--report-error` and a downstream KLD run to judge them per model.

---

## 2. Serving with SparkInfer + nightly vLLM

### 2.1 Install (serving venv — KLD scoring uses its own venv, see §2.6)

vLLM nightly pins its own torch; sparkinfer declares `torch>=2.12`. Keep
the venv on **vLLM's pinned stack** and install sparkinfer with
`--no-deps` so it cannot upgrade torch out from under vLLM's precompiled
binaries:

```bash
python3.12 -m venv venv && source venv/bin/activate

# 1. torch stack (cu130) at vLLM's pin:
uv pip install --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0

# 2. nightly vLLM from source with precompiled kernels:
cd vllm && VLLM_USE_PRECOMPILED=1 uv pip install -e . && cd ..

# 3. sparkinfer — --no-deps is MANDATORY (see above):
uv pip install -e /path/to/sparkinfer --no-deps

python -c "import torchvision, vllm, sparkinfer; print('stack OK')"
```

If torch 2.13.x ever appears in an install log, the pin was broken —
re-run steps 1-3 in order.

Installing sparkinfer registers the vLLM plugin automatically via the
`vllm.general_plugins` entry point
(`sparkinfer.integration.vllm.plugin:register_sparkinfer_fp6`); it runs in
every vLLM process (front end, engine core, TP workers). `sparkinfer_fp6`
is **not** a stock vLLM quant method — without the package installed,
`--quantization sparkinfer_fp6` will not resolve.

### 2.2 Environment gates

| Variable | Purpose |
|---|---|
| `SPARKINFER_ENABLE_FP6=1` | Master gate. Unset/0 = the plugin stays inert and vLLM uses its native paths. |
| `SPARKINFER_FP6_MODEL_DIR=<checkpoint dir>` | Where the plugin indexes FP6 tensors. Export **unconditionally** per launch: a stale value from a previous launch mis-classifies layers and trips loader shape asserts. |
| `SPARKINFER_ENABLE_FP6_MICRO=0` | Keep the fast fused dense kernel (the BS1 micro kernel is slower for these workloads). |
| `SPARKINFER_DENSE_PER_ROW_IN_KERNEL` | Default 1 (fused in-kernel per-row activation quant). `0` = host-chain A/B fallback, bit-identical but ~1.7 ms/token slower at decode. |

Never leak KLD-scoring-only variables into serving: unset
`TORCH_COMPILE_DISABLE` and `SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT`
(the launch scripts below do this).

### 2.3 Minimal launch

```bash
export SPARKINFER_ENABLE_FP6=1
export SPARKINFER_FP6_MODEL_DIR=/path/to/fp6-checkpoint

vllm serve /path/to/fp6-checkpoint \
  --quantization sparkinfer_fp6 \
  --served-model-name my-model-fp6 \
  --trust-remote-code
```

`--quantization sparkinfer_fp6` is optional — the plugin claims any
checkpoint whose `config.json` carries `quant_method=modelopt` +
`quant_algo=W6A6` when the enable gate is set. At TP>1 on Blackwell add
`--disable-custom-all-reduce` (vLLM's custom all-reduce kernel crashes
during CUDA-graph capture on sm_120; the NCCL fallback costs ~1-3%).

Verify the server end to end:

```bash
curl -s -H "Authorization: Bearer $OPENAI_API_KEY" \
  http://localhost:8001/v1/models
```

The production launch scripts below encode all of the hard-won defaults
(MTP speculative config, CUDA-graph capture sizes synced to the MTP verify
batch, vision toggles, profiler hooks) and are the recommended starting
point.

### 2.4 Example launch script — Qwen3.6-27B dense (TP=1/2)

Usage: `./qwen3.6-27b-fp6.sh API-KEY-HERE` (or `VLLM_API_KEY=... ./qwen3.6-27b-fp6.sh`).
Override any `${VAR:-default}` at launch, e.g.
`MODEL_DIR=/path TP_SIZE=2 ./qwen3.6-27b-fp6.sh KEY`.

```bash
#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# TheHouseOfTheDude/Qwen3.6-27B-FP6 on 1x RTX PRO 6000 (Blackwell) via vLLM
# ============================================================
#
# Model:
#   - Base:    Qwen/Qwen3.6-27B  (arch tag: qwen3_5, ~28B params)
#   - Quant:   SparkInfer MX-FP6 (W6A6 schema tag): 6-bit weights on disk,
#              activations quantized at runtime to FP8 E4M3 (W6A8 execution).
#              quantization_config = {"quant_method":"modelopt","quant_algo":"W6A6"}
#              Golden-rule Linear coverage: MLP + full attention + linear_attn
#              projections quantized to FP6 (this is the "_la" build that drops
#              below the FP8 size). IGNORED / kept BF16: lm_head, visual tower,
#              mtp head, embeddings, router gates, norms.
#              Served by the SparkInfer fused dense kernel (not the micro kernel).
#
# Architecture notes (from the upstream Qwen/Qwen3.6-27B card):
#   - Hybrid 64-layer stack:
#       16 x (3 x (Gated DeltaNet -> FFN) + 1 x (Gated Attention -> FFN))
#     i.e. 48 linear-attention (DeltaNet) layers + 16 full-attention layers.
#     KV cache is only allocated for the 16 Gated Attention layers.
#   - Gated Attention: 24 Q heads, 4 KV heads, head_dim=256, partial RoPE.
#   - Gated DeltaNet: 48 V heads, 16 QK heads, head_dim=128 (no KV cache).
#   - Vision encoder present (VL model); MTP head trained-in.
#   - Native ctx: 262,144 tokens (so 131,072 here needs no YaRN override).
#
# ------------------------------------------------------------
# PREREQUISITE -- register the SparkInfer FP6 quantization in vLLM
# ------------------------------------------------------------
#   "sparkinfer_fp6" is NOT a stock vLLM quant method. vLLM's native ModelOpt
#   path does not implement W6A6, so the sparkinfer package must be installed
#   in the serving venv: its "vllm.general_plugins" entry point
#   (sparkinfer.integration.vllm.plugin:register_sparkinfer_fp6) registers the
#   SparkInferFp6Config in every vLLM process (front, engine-core, TP workers).
#   Until then, `--quantization sparkinfer_fp6` will not resolve.
#
#   The two env gates below are what the FP6 serving layer keys off of:
#     SPARKINFER_ENABLE_FP6=1        -> select the FP6 path (else native fallback)
#     SPARKINFER_ENABLE_FP6_MICRO=0  -> use the fast fused kernel, not the BS1
#                                       micro kernel (slower for this workload).
#
# VRAM fit on 1x 96 GiB Blackwell:
#   - FP6 weights for the quantized Linears (~16B params @ ~6.25 bit) plus the
#     BF16 residue (visual/lm_head/embeddings/mtp) resident ~= 26 GiB.
#   - KV cache (Gated Attention layers only, BF16, TP=1):
#       per-token KV = 16 layers * 2 (K+V) * 4 KV heads * 256 = 64 KiB/token
#       128K ctx, 1 seq = 8 GiB ; 4 seqs = 32 GiB
#   - 26 GiB weights + 32 GiB KV + graphs/vision still fits 96 GiB with margin.
#
# This script does not store API keys. Pass the key as the first argument, or
# set VLLM_API_KEY in the environment.
# ============================================================

# --- Memory Management ---
export PYTORCH_ALLOC_CONF=expandable_segments:True

# --- GPU Selection (single Blackwell card) ---
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# --- SparkInfer FP6 gates (see PREREQUISITE above) ---
export SPARKINFER_ENABLE_FP6="${SPARKINFER_ENABLE_FP6:-1}"
export SPARKINFER_ENABLE_FP6_MICRO="${SPARKINFER_ENABLE_FP6_MICRO:-0}"
# The vLLM plugin (sparkinfer.integration.vllm.plugin) reads the FP6 checkpoint
# dir from SPARKINFER_FP6_MODEL_DIR in every process, including spawned TP
# workers. Set below once MODEL_DIR is known.
# Make sure KLD-scoring-only vars never leak into serving:
unset TORCH_COMPILE_DISABLE SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT 2>/dev/null || true

# --- vLLM / CUDA behavior ---
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_SLEEP_WHEN_IDLE=1
export VLLM_USE_FLASHINFER_SAMPLER=0

# --- Loading / CPU threading ---
export SAFETENSORS_FAST_GPU=1
export OMP_NUM_THREADS=8

# --- Torch profiler (registers POST /start_profile and /stop_profile) ---
# Recent vLLM nightlies gate these endpoints on the --profiler-config CLI arg
# (vllm/entrypoints/serve/profile/api_router.py); the old VLLM_TORCH_PROFILER_DIR
# env var no longer registers them. PROFILE=1 enables; traces (.json.gz) land in
# PROFILE_DIR; summarize with sparkinfer scripts/summarize_vllm_trace.py.
PROFILE="${PROFILE:-0}"
PROFILE_DIR="${PROFILE_DIR:-/tmp/vllm_prof}"
if [[ "$PROFILE" == "1" ]]; then
  mkdir -p "$PROFILE_DIR"
  # Kept for older builds that still read the env var; harmless on new ones.
  export VLLM_TORCH_PROFILER_DIR="$PROFILE_DIR"
fi

# --- Long-context (YaRN) escape hatch ---
# 131072 is within the native 262144 window, so no override is needed. Only set
# this if you push MAX_MODEL_LEN past 262144 with an --hf-overrides rope config.
# export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# ============================================================
# Model / serving configuration
# ============================================================

API_KEY="${1:-${VLLM_API_KEY:-}}"

if [[ -z "$API_KEY" ]]; then
  echo "Usage: $0 API-KEY-HERE" >&2
  echo "       or: VLLM_API_KEY=API-KEY-HERE $0" >&2
  exit 2
fi

# Local SparkInfer MX-FP6 (W6A6) checkpoint. Override at launch time if needed:
#   MODEL_DIR=/path/to/model ./qwen3.6-27b-fp6.sh
MODEL_DIR="${MODEL_DIR:-/media/fmodels/TheHouseOfTheDude/qwen3-6_27B_dense_fp6_la}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.6-27B-FP6-W6A6}"
# Tell the FP6 vLLM plugin where the weights live (inherited by workers).
# Set UNCONDITIONALLY: a stale SPARKINFER_FP6_MODEL_DIR from a previous launch
# makes the plugin index the wrong checkpoint -> wrong FP6/BF16 classification
# -> loader shape asserts (this exact failure hit the 35B MoE KLD run).
export SPARKINFER_FP6_MODEL_DIR="$MODEL_DIR"
unset B12X_FP6_MODEL_DIR 2>/dev/null || true  # legacy name must not shadow it
# The FP6 export copies tokenizer.json/tokenizer_config.json and
# chat_template.jinja, so keep tokenizer resolution local by default.
TOKENIZER="${TOKENIZER:-$MODEL_DIR}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"

# Single GPU -> no tensor parallelism.
TP_SIZE="${TP_SIZE:-1}"

# ------------------------------------------------------------
# Capacity sizing for 1x 96 GiB Blackwell at full VLM (vision enabled).
# 128K context fits comfortably; bump MAX_NUM_SEQS for more concurrency
# (each 128K sequence costs ~8 GiB of KV cache).
# ------------------------------------------------------------
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

# Blackwell handles fp8 KV cache well; auto (BF16) already fits 128K here, so
# keep auto unless you want to trade a little accuracy for more concurrency.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"

TOKENIZER_MODE="${TOKENIZER_MODE:-hf}"
CONFIG_FORMAT="${CONFIG_FORMAT:-auto}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"

# SparkInfer MX-FP6 (W6A6). Requires the sparkinfer package installed in the
# serving venv (see PREREQUISITE above). Set QUANTIZATION="" to let vLLM
# auto-detect from config.json (works because the plugin overrides modelopt).
QUANTIZATION="${QUANTIZATION:-sparkinfer_fp6}"

# Qwen3.6 ships a custom modeling file (qwen3_5 / qwen3_next family).
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"

# ============================================================
# Qwen3.6 reasoning / tool-call / MTP / VLM toggles
# ============================================================

# Qwen3.6 thinks by default; qwen3 reasoning parser covers the whole family.
REASONING_PARSER="${REASONING_PARSER:-qwen3}"

# Tool calling: qwen3_coder is the model-card parser for OpenAI-style tool use.
ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"

# Multi-Token Prediction (trained-in MTP head, kept BF16 by the FP6 export).
# Disable (ENABLE_MTP=0) if you hit a Mamba/DeltaNet cudagraph assert during
# MTP draft capture, or when benchmarking non-spec throughput.
ENABLE_MTP="${ENABLE_MTP:-1}"
# NOTE: JSON defaults must NOT live inside ${VAR:-...} — bash closes the
# expansion at the FIRST '}' in the default, appending a stray literal '}'
# to any env-provided value (breaks vllm's JSON parsing).
if [[ -z "${MTP_SPEC:-}" ]]; then
  MTP_SPEC='{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
fi

# Vision. VL model; the FP6 export left the visual tower in BF16 so it works
# out of the box. Set TEXT_ONLY=1 to skip vision profiling and free its
# workspace for additional KV cache (text-only traffic).
TEXT_ONLY="${TEXT_ONLY:-0}"

# Optional: lets clients drive video frame sampling via
# extra_body={"mm_processor_kwargs": {"fps": ...}}. Harmless when no video.
if [[ -z "${MEDIA_IO_KWARGS:-}" ]]; then
  MEDIA_IO_KWARGS='{"video":{"num_frames":-1}}'
fi

# Prefix caching is generally a win for chat/agent workloads.
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"

# CUDA graph setup:
# - full_decode_only keeps CUDA graphs on the decode path and avoids the
#   highest-memory full-prefill captures.
# - Qwen3.6's Gated DeltaNet layers occasionally trip a cudagraph cache assert
#   ("Mamba/DeltaNet cuda-graph cache" path). If you see it, set
#   MAX_CUDAGRAPH_CAPTURE_SIZE=4 (or 1) below as a fallback.
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-full_decode_only}"
# Default capture sizes are derived from the MTP spec: vLLM pads decode
# batches to the next captured size, so the list must reach at least the
# MTP verify batch (1 + num_speculative_tokens) and ideally contain it
# exactly (padding 5 -> 8 wastes ~37% of every verify GEMM). Users only
# set MTP_SPEC; this stays in sync automatically.
if [[ -z "${CUDAGRAPH_CAPTURE_SIZES:-}" ]]; then
  NSPEC=0
  if [[ "$ENABLE_MTP" == "1" && "$MTP_SPEC" =~ \"num_speculative_tokens\"[[:space:]]*:[[:space:]]*([0-9]+) ]]; then
    NSPEC="${BASH_REMATCH[1]}"
  fi
  VERIFY=$((NSPEC + 1))
  CUDAGRAPH_CAPTURE_SIZES="[$(printf '%s\n' 1 2 4 8 "$VERIFY" | sort -un | paste -sd, -)]"
fi
if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
  COMPILATION_CONFIG="{\"mode\":3,\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\",\"cudagraph_capture_sizes\":${CUDAGRAPH_CAPTURE_SIZES}}"
fi

# Fallback for older vLLM versions that do not understand cudagraph_capture_sizes
# inside --compilation-config, OR the DeltaNet cudagraph cache assert above.
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"

# Eager mode escape hatch. The FP6 linear is an opaque torch custom op
# (sparkinfer::fp6_dense_linear), so mode-3 compile + CUDA graphs work
# normally; set ENFORCE_EAGER=1 only to isolate kernel issues from
# compile/capture issues during bring-up.
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

# ============================================================
# Assemble the argument list
# ============================================================

VLLM_ARGS=(
  "$MODEL_DIR"
  --served-model-name "$SERVED_MODEL_NAME"
  --api-key "$API_KEY"
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size "$TP_SIZE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --tokenizer "$TOKENIZER"
  --tokenizer-mode "$TOKENIZER_MODE"
  --config-format "$CONFIG_FORMAT"
  --load-format "$LOAD_FORMAT"
  --dtype auto
)

# Eager mode disables torch.compile + CUDA graphs, so the compilation-config is
# both unnecessary and conflicting there; only pass it when compiling.
if [[ "$ENFORCE_EAGER" == "1" ]]; then
  VLLM_ARGS+=(--enforce-eager)
else
  VLLM_ARGS+=(--compilation-config "$COMPILATION_CONFIG")
fi

# Blackwell sm_120: vLLM's custom all-reduce kernel crashes during CUDA graph
# capture at TP>1 (custom_all_reduce.cuh 'invalid argument'). Force the NCCL
# fallback (~1-3% slower at TP=2) whenever tensor parallelism is active.
if [[ "$TP_SIZE" -gt 1 ]]; then
  VLLM_ARGS+=(--disable-custom-all-reduce)
fi

if [[ "$PROFILE" == "1" ]]; then
  VLLM_ARGS+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROFILE_DIR}\"}")
fi

if [[ -n "$QUANTIZATION" ]]; then
  VLLM_ARGS+=(--quantization "$QUANTIZATION")
fi

if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  VLLM_ARGS+=(--trust-remote-code)
fi

if [[ -n "$REASONING_PARSER" ]]; then
  VLLM_ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi

if [[ "$ENABLE_TOOL_CALLING" == "1" ]]; then
  VLLM_ARGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
fi

if [[ "$ENABLE_MTP" == "1" ]]; then
  VLLM_ARGS+=(--speculative-config "$MTP_SPEC")
fi

if [[ "$TEXT_ONLY" == "1" ]]; then
  VLLM_ARGS+=(--language-model-only)
elif [[ -n "$MEDIA_IO_KWARGS" ]]; then
  VLLM_ARGS+=(--media-io-kwargs "$MEDIA_IO_KWARGS")
fi

if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
  VLLM_ARGS+=(--enable-prefix-caching)
fi

if [[ -n "$MAX_CUDAGRAPH_CAPTURE_SIZE" ]]; then
  VLLM_ARGS+=(--max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE")
fi

echo "Launching vLLM with:"
echo "  MODEL_DIR=${MODEL_DIR}"
echo "  SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "  TOKENIZER=${TOKENIZER}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  SPARKINFER_ENABLE_FP6=${SPARKINFER_ENABLE_FP6} (micro=${SPARKINFER_ENABLE_FP6_MICRO})"
echo "  SPARKINFER_FP6_MODEL_DIR=${SPARKINFER_FP6_MODEL_DIR}"
echo "  TP_SIZE=${TP_SIZE}"
echo "  MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "  KV_CACHE_DTYPE=${KV_CACHE_DTYPE}"
echo "  QUANTIZATION=${QUANTIZATION}"
echo "  REASONING_PARSER=${REASONING_PARSER}"
echo "  TOOL_CALLING=${ENABLE_TOOL_CALLING} (parser=${TOOL_CALL_PARSER})"
echo "  MTP=${ENABLE_MTP} (${MTP_SPEC})"
echo "  TEXT_ONLY=${TEXT_ONLY}"
echo "  PREFIX_CACHING=${ENABLE_PREFIX_CACHING}"
if [[ "$PROFILE" == "1" ]]; then
  echo "  PROFILER=torch -> ${PROFILE_DIR} (POST /start_profile + /stop_profile)"
else
  echo "  PROFILER=disabled (PROFILE=1 to enable)"
fi
if [[ "$ENFORCE_EAGER" == "1" ]]; then
  echo "  ENFORCE_EAGER=1 (torch.compile + CUDA graphs DISABLED)"
else
  echo "  COMPILATION_CONFIG=${COMPILATION_CONFIG}"
fi
echo

vllm serve "${VLLM_ARGS[@]}"
```

### 2.5 Example launch script — Qwen3.6-35B-A3B MoE (TP=1/2)

Same usage pattern. MoE expert sharding works through vLLM's loader, and
TP=2 has been validated (the script adds `--disable-custom-all-reduce`
automatically at TP>1). At single-stream loads TP=2 buys capacity more
than latency (~8% TPOT gain; see `fp6-results.md` §3.4-3.5).

```bash
#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# TheHouseOfTheDude/Qwen3.6-35B-A3B-FP6 on 1x RTX PRO 6000 (Blackwell) via vLLM
# ============================================================
#
# Model:
#   - Base:    Qwen/Qwen3.6-35B-A3B  (arch tag: qwen3_5 MoE family, ~35B total
#              / ~3B active params)
#   - Quant:   SparkInfer MX-FP6 (W6A6 schema tag): 6-bit weights on disk,
#              activations quantized at runtime to FP8 E4M3 (W6A8 execution).
#              quantization_config = {"quant_method":"modelopt","quant_algo":"W6A6"}
#              Coverage: all 256 routed experts per layer (gate/up/down), the
#              shared expert, MLP + full attention + linear_attn projections.
#              IGNORED / kept BF16: lm_head, visual tower, mtp head, embeddings,
#              router gates (mlp.gate / shared_expert_gate), norms, GDN aux.
#              Routed experts run through the SparkInfer fused MoE kernel
#              (FusedMoE binding); everything else uses the dense FP6 path.
#
# Architecture notes:
#   - 40-layer hybrid stack: 30 Gated DeltaNet (linear_attn) layers + 10 full
#     Gated Attention layers. KV cache only for the 10 full-attention layers.
#   - MoE: 256 routed experts, top-8, + 1 shared expert per layer.
#     Expert dims 2048 (hidden) -> 512 (intermediate).
#   - Vision encoder present (VL model); MTP head trained-in (kept BF16).
#
# ------------------------------------------------------------
# PREREQUISITE -- register the SparkInfer FP6 quantization in vLLM
# ------------------------------------------------------------
#   "sparkinfer_fp6" is NOT a stock vLLM quant method; the sparkinfer package
#   (sparkinfer.integration.vllm.plugin, wired via the vllm.general_plugins
#   entry point in sparkinfer's pyproject) must be installed in the serving
#   venv. The env gates:
#     SPARKINFER_ENABLE_FP6=1        -> select the FP6 path (else native fallback)
#     SPARKINFER_ENABLE_FP6_MICRO=0  -> dense linears use the fused kernel. The
#                                       MoE expert path picks its own backend
#                                       per shape.
#
# VRAM fit on 1x 96 GiB Blackwell:
#   - FP6 checkpoint is ~28.8 GiB on disk; plus BF16 residue (visual/lm_head/
#     embeddings/mtp/router) resident ~= 32 GiB.
#   - KV cache only for the 10 Gated Attention layers -> 128K context is cheap
#     relative to the dense 27B build.
#
# This script does not store API keys. Pass the key as the first argument, or
# set VLLM_API_KEY in the environment.
# ============================================================

# --- Memory Management ---
export PYTORCH_ALLOC_CONF=expandable_segments:True

# --- GPU Selection (single Blackwell card) ---
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# --- SparkInfer FP6 gates (see PREREQUISITE above) ---
export SPARKINFER_ENABLE_FP6="${SPARKINFER_ENABLE_FP6:-1}"
export SPARKINFER_ENABLE_FP6_MICRO="${SPARKINFER_ENABLE_FP6_MICRO:-0}"
# Make sure KLD-scoring-only vars never leak into serving:
unset TORCH_COMPILE_DISABLE SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT 2>/dev/null || true

# --- vLLM / CUDA behavior ---
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_SLEEP_WHEN_IDLE=1
export VLLM_USE_FLASHINFER_SAMPLER=0

# --- Loading / CPU threading ---
export SAFETENSORS_FAST_GPU=1
export OMP_NUM_THREADS=8

# --- Torch profiler (registers POST /start_profile and /stop_profile) ---
PROFILE="${PROFILE:-0}"
PROFILE_DIR="${PROFILE_DIR:-/tmp/vllm_prof}"
if [[ "$PROFILE" == "1" ]]; then
  mkdir -p "$PROFILE_DIR"
  export VLLM_TORCH_PROFILER_DIR="$PROFILE_DIR"
fi

# ============================================================
# Model / serving configuration
# ============================================================

API_KEY="${1:-${VLLM_API_KEY:-}}"

if [[ -z "$API_KEY" ]]; then
  echo "Usage: $0 API-KEY-HERE" >&2
  echo "       or: VLLM_API_KEY=API-KEY-HERE $0" >&2
  exit 2
fi

# Local SparkInfer MX-FP6 (W6A6) checkpoint. Override at launch time if needed:
#   MODEL_DIR=/path/to/model ./qwen3.6-35b-a3b-fp6.sh
MODEL_DIR="${MODEL_DIR:-/media/fmodels/TheHouseOfTheDude/qwen3-6_35B-A3B_moe_fp6}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.6-35B-A3B-FP6-W6A6}"
# Tell the FP6 vLLM plugin where the weights live (inherited by workers).
# Set UNCONDITIONALLY: a stale SPARKINFER_FP6_MODEL_DIR from a previous launch
# makes the plugin index the wrong checkpoint -> wrong FP6/BF16 classification
# -> loader shape asserts / silent BF16 fallback (the Cydonia lesson; also hit
# the 35B MoE KLD run when the 27B dir was still exported).
export SPARKINFER_FP6_MODEL_DIR="$MODEL_DIR"
unset B12X_FP6_MODEL_DIR 2>/dev/null || true  # legacy name must not shadow it
# The FP6 export copies tokenizer.json/tokenizer_config.json and
# chat_template.jinja, so keep tokenizer resolution local by default.
TOKENIZER="${TOKENIZER:-$MODEL_DIR}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"

# Default 1 GPU. MoE experts shard through vLLM's loader, so TP=2 works;
# the script adds --disable-custom-all-reduce automatically at TP>1.
TP_SIZE="${TP_SIZE:-1}"

# ------------------------------------------------------------
# Capacity sizing for 1x 96 GiB Blackwell at full VLM (vision enabled).
# Only 10 of 40 layers carry KV cache, so long context is cheap here.
# ------------------------------------------------------------
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"

TOKENIZER_MODE="${TOKENIZER_MODE:-hf}"
CONFIG_FORMAT="${CONFIG_FORMAT:-auto}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"

# SparkInfer MX-FP6 (W6A6). Set QUANTIZATION="" to let vLLM auto-detect from
# config.json (works because the plugin overrides the modelopt method).
QUANTIZATION="${QUANTIZATION:-sparkinfer_fp6}"

# Qwen3.6 ships a custom modeling file (qwen3_5 / qwen3_next family).
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"

# ============================================================
# Qwen3.6 reasoning / tool-call / MTP / VLM toggles
# ============================================================

# Qwen3.6 thinks by default; qwen3 reasoning parser covers the whole family.
REASONING_PARSER="${REASONING_PARSER:-qwen3}"

# Tool calling: qwen3_coder is the model-card parser for OpenAI-style tool use.
ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"

# Multi-Token Prediction (trained-in MTP head, kept BF16 by the FP6 export —
# including its packed MoE experts). Disable (ENABLE_MTP=0) for non-spec
# benchmarking or if MTP draft capture asserts.
ENABLE_MTP="${ENABLE_MTP:-1}"
# NOTE: JSON defaults must NOT live inside ${VAR:-...} — bash closes the
# expansion at the FIRST '}' in the default, appending a stray literal '}'
# to any env-provided value (breaks vllm's JSON parsing).
if [[ -z "${MTP_SPEC:-}" ]]; then
  MTP_SPEC='{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
fi

# Vision. VL model; the FP6 export left the visual tower in BF16 so it works
# out of the box. Set TEXT_ONLY=1 to skip vision profiling and free its
# workspace for additional KV cache (text-only traffic).
TEXT_ONLY="${TEXT_ONLY:-0}"

# Optional: lets clients drive video frame sampling via
# extra_body={"mm_processor_kwargs": {"fps": ...}}. Harmless when no video.
if [[ -z "${MEDIA_IO_KWARGS:-}" ]]; then
  MEDIA_IO_KWARGS='{"video":{"num_frames":-1}}'
fi

# Prefix caching is generally a win for chat/agent workloads.
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"

# CUDA graph setup. The FP6 MoE binding warm-runs the resolved capture sizes
# at weight-load time automatically (it reads vLLM's compilation config;
# SPARKINFER_MOE_WARM_MS overrides) — no manual sync with the plugin is needed.
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-full_decode_only}"
# Default capture sizes are derived from the MTP spec: vLLM pads decode
# batches to the next captured size, so the list must reach at least the
# MTP verify batch (1 + num_speculative_tokens) and ideally contain it
# exactly (padding 5 -> 8 wastes ~37% of every verify GEMM). Users only
# set MTP_SPEC; this stays in sync automatically. The FP6 plugin warms
# whatever sizes vLLM resolves, so no manual sync is needed there either.
if [[ -z "${CUDAGRAPH_CAPTURE_SIZES:-}" ]]; then
  NSPEC=0
  if [[ "$ENABLE_MTP" == "1" && "$MTP_SPEC" =~ \"num_speculative_tokens\"[[:space:]]*:[[:space:]]*([0-9]+) ]]; then
    NSPEC="${BASH_REMATCH[1]}"
  fi
  VERIFY=$((NSPEC + 1))
  CUDAGRAPH_CAPTURE_SIZES="[$(printf '%s\n' 1 2 4 8 "$VERIFY" | sort -un | paste -sd, -)]"
fi
if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
  COMPILATION_CONFIG="{\"mode\":3,\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\",\"cudagraph_capture_sizes\":${CUDAGRAPH_CAPTURE_SIZES}}"
fi

# Fallback for older vLLM versions, OR the DeltaNet cudagraph cache assert.
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"

# Eager mode escape hatch for first bring-up of the MoE binding: if mode-3
# compile or graph capture trips on the FusedMoE path, relaunch with
# ENFORCE_EAGER=1 to isolate kernel correctness from compile/capture issues.
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

# ============================================================
# Assemble the argument list
# ============================================================

VLLM_ARGS=(
  "$MODEL_DIR"
  --served-model-name "$SERVED_MODEL_NAME"
  --api-key "$API_KEY"
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size "$TP_SIZE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --tokenizer "$TOKENIZER"
  --tokenizer-mode "$TOKENIZER_MODE"
  --config-format "$CONFIG_FORMAT"
  --load-format "$LOAD_FORMAT"
  --dtype auto
)

# Eager mode disables torch.compile + CUDA graphs, so the compilation-config is
# both unnecessary and conflicting there; only pass it when compiling.
if [[ "$ENFORCE_EAGER" == "1" ]]; then
  VLLM_ARGS+=(--enforce-eager)
else
  VLLM_ARGS+=(--compilation-config "$COMPILATION_CONFIG")
fi

# Blackwell sm_120: vLLM's custom all-reduce kernel crashes during CUDA graph
# capture at TP>1 (custom_all_reduce.cuh 'invalid argument'). Force the NCCL
# fallback (~1-3% slower at TP=2) whenever tensor parallelism is active.
if [[ "$TP_SIZE" -gt 1 ]]; then
  VLLM_ARGS+=(--disable-custom-all-reduce)
fi

if [[ "$PROFILE" == "1" ]]; then
  VLLM_ARGS+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROFILE_DIR}\"}")
fi

if [[ -n "$QUANTIZATION" ]]; then
  VLLM_ARGS+=(--quantization "$QUANTIZATION")
fi

if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  VLLM_ARGS+=(--trust-remote-code)
fi

if [[ -n "$REASONING_PARSER" ]]; then
  VLLM_ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi

if [[ "$ENABLE_TOOL_CALLING" == "1" ]]; then
  VLLM_ARGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
fi

if [[ "$ENABLE_MTP" == "1" ]]; then
  VLLM_ARGS+=(--speculative-config "$MTP_SPEC")
fi

if [[ "$TEXT_ONLY" == "1" ]]; then
  VLLM_ARGS+=(--language-model-only)
elif [[ -n "$MEDIA_IO_KWARGS" ]]; then
  VLLM_ARGS+=(--media-io-kwargs "$MEDIA_IO_KWARGS")
fi

if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
  VLLM_ARGS+=(--enable-prefix-caching)
fi

if [[ -n "$MAX_CUDAGRAPH_CAPTURE_SIZE" ]]; then
  VLLM_ARGS+=(--max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE")
fi

echo "Launching vLLM with:"
echo "  MODEL_DIR=${MODEL_DIR}"
echo "  SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "  TOKENIZER=${TOKENIZER}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  SPARKINFER_ENABLE_FP6=${SPARKINFER_ENABLE_FP6} (micro=${SPARKINFER_ENABLE_FP6_MICRO})"
echo "  SPARKINFER_FP6_MODEL_DIR=${SPARKINFER_FP6_MODEL_DIR}"
echo "  TP_SIZE=${TP_SIZE}"
echo "  MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "  KV_CACHE_DTYPE=${KV_CACHE_DTYPE}"
echo "  QUANTIZATION=${QUANTIZATION}"
echo "  REASONING_PARSER=${REASONING_PARSER}"
echo "  TOOL_CALLING=${ENABLE_TOOL_CALLING} (parser=${TOOL_CALL_PARSER})"
echo "  MTP=${ENABLE_MTP} (${MTP_SPEC})"
echo "  TEXT_ONLY=${TEXT_ONLY}"
echo "  PREFIX_CACHING=${ENABLE_PREFIX_CACHING}"
if [[ "$PROFILE" == "1" ]]; then
  echo "  PROFILER=torch -> ${PROFILE_DIR} (POST /start_profile + /stop_profile)"
else
  echo "  PROFILER=disabled (PROFILE=1 to enable)"
fi
if [[ "$ENFORCE_EAGER" == "1" ]]; then
  echo "  ENFORCE_EAGER=1 (torch.compile + CUDA graphs DISABLED)"
else
  echo "  COMPILATION_CONFIG=${COMPILATION_CONFIG}"
fi
echo

vllm serve "${VLLM_ARGS[@]}"
```

### 2.6 Validating a quant (KLD scoring)

The KLD scorer (`examples/offline_inference/score_mode_kld.py`) is **not**
part of upstream vLLM — it lives on the feature branch
[`phaelon74/vllm@feature/score-mode-ppl-kld`](https://github.com/phaelon74/vllm/tree/feature/score-mode-ppl-kld),
an off-branch from vLLM main. Because it diverges from the nightly you
serve with, install it in its **own dedicated venv** — do not mix it into
the serving venv from §2.1:

```bash
git clone -b feature/score-mode-ppl-kld https://github.com/phaelon74/vllm.git vllm-kld
python3.12 -m venv venv-kld && source venv-kld/bin/activate

# Same torch pin + install order as §2.1, inside THIS venv:
uv pip install --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
cd vllm-kld && VLLM_USE_PRECOMPILED=1 uv pip install -e . && cd ..
uv pip install -e /path/to/sparkinfer --no-deps
```

Scoring runs offline (not against a server) with deterministic eager
execution, from the `vllm-kld` checkout with `venv-kld` active:

```bash
export SPARKINFER_ENABLE_FP6=1
export SPARKINFER_FP6_MODEL_DIR=/path/to/fp6-checkpoint
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_COMPILE_DISABLE=1   # mandatory for reproducible KLD

python examples/offline_inference/score_mode_kld.py \
  --model "$SPARKINFER_FP6_MODEL_DIR" \
  --reference-logits /path/to/ref_logits_<base>_ctx2048_s512 \
  --dataset wikitext --dataset-config wikitext-2-raw-v1 \
  --context-length 2048 --stride 512 \
  --gpu-memory-utilization 0.90
```

The mean KLD is bit-deterministic under this configuration: run it twice
and expect the identical value (any difference is a bug). Reference values
for the validated models are in `fp6-results.md`. `TORCH_COMPILE_DISABLE`
is for scoring only — never leave it set when serving (the launch scripts
unset it).

### 2.7 Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `--quantization sparkinfer_fp6` "invalid choice" | sparkinfer not installed in the serving venv; the plugin entry point never ran. |
| Loader `AssertionError` in `vllm/model_executor/parameter.py` | `SPARKINFER_FP6_MODEL_DIR` points at a different checkpoint than the one being served (stale export). Export it unconditionally per launch. |
| `torchvision::nms` missing after installing sparkinfer | torch got upgraded past vLLM's pin — reinstall per §2.1 (sparkinfer with `--no-deps`). |
| CUDA error `custom_all_reduce.cuh ... invalid argument` at TP>1 | Blackwell sm_120 custom all-reduce vs CUDA graphs; pass `--disable-custom-all-reduce` (scripts do it automatically). |
| `--speculative-config ... cannot be converted` | JSON default embedded in `${VAR:-...}` — bash truncates at the first `}`. Assign JSON via a plain `if [[ -z ... ]]` block (scripts already do). |
| Mamba/DeltaNet cudagraph cache assert during MTP capture | Set `MAX_CUDAGRAPH_CAPTURE_SIZE=4` (or 1), or `ENABLE_MTP=0` to isolate. |
| `pytest` fails with `ModuleNotFoundError: torch` | The system pytest resolved instead of the venv's; run `python -m pytest`. |
