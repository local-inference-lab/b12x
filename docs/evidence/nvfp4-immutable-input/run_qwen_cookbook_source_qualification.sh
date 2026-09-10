#!/usr/bin/env bash
set -euo pipefail

# TP1 or TP2 with CPU PLE offload and a BF16 target vocabulary head.
# TP2 uses TP_GPU_IDS (default 2,3); TP1 uses PHYSICAL_GPU.
# MTP_TOKENS=0 disables speculation; the default is three draft tokens.
# The draft shares BF16 by default; MTP_NVFP4_LM_HEAD=1 selects a private head.
# LANGUAGE_MODEL_ONLY=0 enables up to four images, resized to at most 4 MiPixels
# each. Video inputs are disabled in that configuration.
# Source code is supplied by the image; model and compiler caches are mounts.
physical_gpu=${PHYSICAL_GPU:?Set PHYSICAL_GPU to an explicitly assigned GPU index}
[[ $physical_gpu =~ ^(0|[1-9][0-9]*)$ ]] || exit 2
nvidia-smi -i "$physical_gpu" --query-gpu=uuid --format=csv,noheader >/dev/null
tensor_parallel_size=${TENSOR_PARALLEL_SIZE:-1}
case $tensor_parallel_size in
    1) gpu_request="device=$physical_gpu" ;;
    2)
        tp_gpu_ids=${TP_GPU_IDS:-2,3}
        case $tp_gpu_ids in
            0,1|1,2|2,3) gpu_request="\"device=$tp_gpu_ids\"" ;;
            *) echo "TP_GPU_IDS must be 0,1 or 1,2 or 2,3" >&2; exit 2 ;;
        esac
        ;;
    *) echo "TENSOR_PARALLEL_SIZE must be 1 or 2" >&2; exit 2 ;;
esac
port=${PORT:?Set PORT to an unused port}
container_name=${CONTAINER_NAME:?Set a unique CONTAINER_NAME}
image=${IMAGE:?Set IMAGE to a complete-source qualification image}
cache_key=${CACHE_KEY:?Set CACHE_KEY to an isolated compiler-cache directory}
mtp_nvfp4_head=${MTP_NVFP4_LM_HEAD:-0}
[[ $mtp_nvfp4_head == 0 || $mtp_nvfp4_head == 1 ]] || exit 2
# A shared runtime image may carry another model's activation defaults.
# Preserve BF16 inputs to the private NVFP4 draft head unless explicitly tested.
head_activation_a16=${VLLM_LM_HEAD_A16:-${LM_HEAD_A16:-1}}
[[ $head_activation_a16 == 0 || $head_activation_a16 == 1 ]] || exit 2
head_activation_environment=(-e "VLLM_LM_HEAD_A16=$head_activation_a16")
mtp_tokens=${MTP_TOKENS:-3}
multimodal_options=()
case ${LANGUAGE_MODEL_ONLY:-1} in
    1) multimodal_options=(--language-model-only) ;;
    0) multimodal_options=(
        --limit-mm-per-prompt '{"image":4,"video":0}'
        --mm-processor-kwargs '{"max_pixels":4194304}'
    ) ;;
    *) echo "LANGUAGE_MODEL_ONLY must be 0 or 1" >&2; exit 2 ;;
esac
speculative_options=()
case $mtp_tokens in
    3) speculative_options=(--speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"b12x"}') ;;
    0) [[ $mtp_nvfp4_head == 0 ]] || { echo "An NVFP4 draft head requires MTP" >&2; exit 2; } ;;
    *) echo "MTP_TOKENS must be 0 or 3" >&2; exit 2 ;;
esac
profile_options=()
profile_mounts=()
graph_options=()
if [[ -n ${QUALIFICATION_CAPTURE_SIZE:-} ]]; then
    [[ $QUALIFICATION_CAPTURE_SIZE =~ ^[1-9][0-9]*$ ]] || exit 2
    graph_options=(--max-cudagraph-capture-size "$QUALIFICATION_CAPTURE_SIZE")
fi
checkpoint_options=()
if [[ -n ${RECURRENT_CHECKPOINT_POLICY:-} ]]; then
    case $RECURRENT_CHECKPOINT_POLICY in
        auto|aligned|request_boundaries) ;;
        *) echo "Invalid recurrent checkpoint policy" >&2; exit 2 ;;
    esac
    checkpoint_options=(--recurrent-checkpoint-policy "$RECURRENT_CHECKPOINT_POLICY")
fi
if [[ -n ${PROFILE_DIR:-} ]]; then
    mkdir -p "$PROFILE_DIR"
    profile_mounts=(-v "$PROFILE_DIR:/tmp/vllm-profile")
    profile_options=(
        --profiler-config.profiler=torch
        --profiler-config.torch_profiler_dir=/tmp/vllm-profile/run
        --profiler-config.torch_profiler_with_stack=true
        --profiler-config.torch_profiler_record_shapes=false
        --profiler-config.torch_profiler_with_memory=false
        --profiler-config.torch_profiler_with_flops=false
        --profiler-config.torch_profiler_use_gzip=true
        --profiler-config.torch_profiler_dump_cuda_time_total=false
        --profiler-config.ignore_frontend=true
        --profiler-config.delay_iterations=0
        --profiler-config.max_iterations=16
        --profiler-config.warmup_iterations=0
        --profiler-config.active_iterations=17
        --profiler-config.wait_iterations=0
    )
fi
if docker container inspect "$container_name" >/dev/null 2>&1; then
    echo "Container $container_name already exists; choose another name" >&2
    exit 1
fi
if ss -H -ltn "sport = :$port" | rg -q .; then
    echo "TCP port $port is already in use" >&2
    exit 1
fi
docker run -d --name "$container_name" \
    --gpus "$gpu_request" --network host --ipc host --shm-size 32g --init \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e OMP_NUM_THREADS=2 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_USE_V2_MODEL_RUNNER=1 \
    -e VLLM_COMPUTE_NANS_IN_LOGITS="${VLLM_COMPUTE_NANS_IN_LOGITS:-0}" \
    -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e VLLM_SSM_CONV_STATE_LAYOUT=DS -e VLLM_PLE_CPU_OFFLOAD=1 \
    -e VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel \
    -e VLLM_CAUSAL_CONV1D_UPDATE_HOIST=1 \
    -e VLLM_ENABLE_PCIE_ALLREDUCE=1 -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
    -e VLLM_MXFP8_LM_HEAD=0 -e VLLM_MTP_NVFP4_LM_HEAD="$mtp_nvfp4_head" \
    "${head_activation_environment[@]}" \
    -e VLLM_QWEN3_8_FLASH_NEXT_MTP_COMPACT=1 \
    -e VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH=1 \
    -e VLLM_QWEN3_8_FLASH_NEXT_OVERLAP="${QWEN_OVERLAP:-1}" \
    -e VLLM_B12X_DENSE_ACTIVATION_MODE="${DENSE_ACTIVATION_MODE:-auto}" \
    -e B12X_DYNAMIC_SPLIT_ROUTE_COMPUTE=1 \
    -e B12X_DYNAMIC_DIRECT_EXPERT_SCALES=1 \
    -e B12X_DYNAMIC_SPLIT_LOW_SMEM=1 \
    -e B12X_DYNAMIC_SKIP_SPLIT_BARRIER_RESET=1 \
    -e B12X_DYNAMIC_SPLIT_FAST_PREPARE=1 \
    -e B12X_DYNAMIC_WORK_SOURCE=persistent_grid \
    -e B12X_DYNAMIC_DETERMINISTIC_OUTPUT=0 \
    -e B12X_DYNAMIC_SPLIT_COMPUTE_MAC=224 \
    -e B12X_DENSE_SPLITK_TURBO=1 \
    -e B12X_PCIE_ONESHOT_THREADS=512 -e B12X_PCIE_ONESHOT_BLOCK_LIMIT=4 \
    -e B12X_PCIE_ONESHOT_PDL=1 -e B12X_MHC_PDL=1 \
    -e NCCL_IB_DISABLE=1 -e NCCL_P2P_LEVEL=SYS -e NCCL_CUMEM_ENABLE=0 \
    -e NCCL_PROTO=LL,LL128,Simple -e NCCL_MIN_NCHANNELS=16 -e NCCL_MAX_NCHANNELS=16 \
    -e NCCL_BUFFSIZE=2097152 \
    -e XDG_CACHE_HOME="/cache/jit/$cache_key" \
    -e VLLM_CACHE_ROOT="/cache/jit/$cache_key/vllm" \
    -e VLLM_CACHE_DIR="/cache/jit/$cache_key/vllm" \
    -e TRITON_CACHE_DIR="/cache/jit/$cache_key/triton" \
    -e CUTE_DSL_CACHE_DIR="/cache/jit/$cache_key/cute-dsl" \
    -e B12X_CUTE_COMPILE_CACHE_DIR="/cache/jit/$cache_key/b12x/cute" \
    -e B12X_COMPILE_CACHE_DIR="/cache/jit/$cache_key/b12x/compile" \
    -e SPARKINFER_COMPILE_CACHE_DIR="/cache/jit/$cache_key/b12x/compile" \
    -e TORCHINDUCTOR_CACHE_DIR="/cache/jit/$cache_key/torchinductor" \
    -e CUDA_CACHE_PATH="/cache/jit/$cache_key/cuda" \
    -v qwen38next-trusted-ple-cache:/cache \
    -v /root/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4:/model-repo:ro \
    "${profile_mounts[@]}" \
    --entrypoint /bin/bash "$image" \
    -lc 'unset NCCL_GRAPH_FILE; exec "$@"' -- \
    /opt/venv/bin/python -m vllm.entrypoints.cli.main serve \
    /model-repo/snapshots/b797d2e1160b9596b2570e56c1d3590faa09d4ed \
    --served-model-name Qwen3.8-Flash-Next --host 0.0.0.0 --port "$port" \
    --tensor-parallel-size "$tensor_parallel_size" --pipeline-parallel-size 1 \
    --mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill \
    --mamba-ssm-cache-dtype auto --async-scheduling \
    --dtype bfloat16 --kv-cache-dtype fp8 --quantization modelopt_mixed \
    --block-size 64 --load-format instanttensor \
    --gpu-memory-utilization 0.96 --max-model-len 262144 \
    --max-num-seqs "${MAX_NUM_SEQS:-4}" --max-num-batched-tokens 6019 \
    --mm-encoder-tp-mode data --mm-processor-cache-gb 0 \
    "${multimodal_options[@]}" \
    "${speculative_options[@]}" \
    --gdn-decode-kernel b12x --linear-backend b12x --moe-backend b12x \
    --no-enable-flashinfer-autotune \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
    "${graph_options[@]}" \
    --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice \
    "${checkpoint_options[@]}" \
    "${profile_options[@]}"
