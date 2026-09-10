#!/usr/bin/env bash
# Compare immutable R32 and NVFP4 MoE overlays on aiserver's physical GPU4–7.
set -euo pipefail
if [[ $(hostname) != aiserver || $# != 2 ]]; then
  printf 'Usage on aiserver: %s <reference|shared|split> RUN_ID\n' "$0" >&2
  exit 2
fi
arm=$1
run_id=$2
[[ ${arm} =~ ^(reference|shared|split)$ && ${run_id} =~ ^[a-z0-9-]+$ ]]
readonly image=sha256:84a2c85d9f889bf0a4b0d08f33630544e1366deb3546d2b26046b64e3e1f364a
readonly b12x_root=/root/vllm/worktrees/b12x-glm-nvfp4-split-prefill-20260910
readonly vllm_root=/root/vllm/worktrees/vllm-glm-nvfp4-scale-sharing-20260910
readonly model_root=/root/.cache/huggingface/hub/models--local-inference-lab--GLM-5.3-Flash-NVFP4
readonly revision=46aaae8a82032f77100f2f03e9cc11b391df3b4d
readonly container="glm53-nvfp4-split-local-${arm}-${run_id}"
[[ -f ${model_root}/snapshots/${revision}/config.json ]]
if docker container inspect "${container}" >/dev/null 2>&1; then
  printf 'Container exists; preserve its evidence: %s\n' "${container}" >&2
  exit 2
fi
if [[ -n $(ss -ltnH 'sport = :5241') ]]; then
  printf 'Port 5241 is already in use\n' >&2
  exit 2
fi
while IFS= read -r memory_used; do
  if (( memory_used > 64 )); then
    printf 'Physical GPUs4–7 must be idle before starting the comparison\n' >&2
    exit 2
  fi
done < <(nvidia-smi -i 4,5,6,7 --query-gpu=memory.used --format=csv,noheader,nounits)
overlay=()
split=0
if [[ ${arm} != reference ]]; then
  [[ -f ${b12x_root}/b12x/moe/_shared/kernels/nvfp4_phase1.py ]]
  [[ -f ${vllm_root}/vllm/model_executor/layers/fused_moe/b12x.py ]]
  overlay+=(
    --mount "type=bind,src=${b12x_root}/b12x,dst=/opt/glm53-flash/b12x/b12x,readonly"
    --mount "type=bind,src=${vllm_root}/vllm/model_executor/layers/fused_moe/b12x.py,dst=/opt/glm53-flash/vllm/vllm/model_executor/layers/fused_moe/b12x.py,readonly"
  )
fi
[[ ${arm} != split ]] || split=1
docker run -d --name "${container}" --init \
  --gpus '"device=4,5,6,7"' --network host --ipc host \
  --mount "type=bind,src=${model_root},dst=/model-cache,readonly" \
  --mount type=volume,src=glm53-nvfp4-split-local-jit,dst=/cache \
  "${overlay[@]}" \
  -e MODEL="/model-cache/snapshots/${revision}" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e TP=4 -e DCP=1 -e PORT=5241 \
  -e SPECULATOR=mtp -e MTP_DEPTH=0 -e VLLM_LM_HEAD_A16=1 \
  -e CACHE_MODE=vram -e KV_CACHE_QUANT=fp8_ds_mla \
  -e GPU_MEMORY_UTILIZATION=0.93 -e MAX_MODEL_LEN=1048576 \
  -e MAX_NUM_SEQS=32 -e MAX_NUM_BATCHED_TOKENS=4096 \
  -e OMP_NUM_THREADS=1 -e CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
  -e NCCL_MIN_NCHANNELS=16 -e NCCL_MAX_NCHANNELS=16 -e NCCL_BUFFSIZE=2097152 \
  -e B12X_NVFP4_DYNAMIC_MATERIALIZED="${split}" \
  "${image}" \
  --profiler-config.profiler=torch \
  --profiler-config.torch_profiler_dir="/cache/profiles/${container}" \
  --profiler-config.torch_profiler_with_stack=false \
  --profiler-config.torch_profiler_dump_cuda_time_total=false \
  --profiler-config.ignore_frontend=true \
  --profiler-config.max_iterations=4
