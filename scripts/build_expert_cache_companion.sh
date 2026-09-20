#!/usr/bin/env bash
# Build every companion extension; installation and source attestation are separate.
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo 'Usage: build_expert_cache_companion.sh VLLM_SOURCE WHEEL_OUTPUT BUILD_LOG' >&2
  exit 2
fi
source_dir=$(realpath "$1")
wheel_output=$(realpath -m "$2")
build_log=$(realpath -m "$3")
test -f "$source_dir/vllm/model_executor/layers/fused_moe/b12x_cache.py"
test ! -e "$build_log"
mkdir -p "$wheel_output" "$(dirname "$build_log")"
if find "$wheel_output" -maxdepth 1 -name '*.whl' -print -quit | read -r _; then
  echo 'Wheel output already contains a build; choose a new directory.' >&2
  exit 2
fi
: "${TORCH_CUDA_ARCH_LIST:?Set the physical target architecture, e.g. 12.0 for SM120}"
export VLLM_USE_PRECOMPILED=0 VLLM_USE_PRECOMPILED_RUST=0
export CMAKE_BUILD_TYPE=Release
(
  python -c 'import torch; print(torch.__version__, torch.version.cuda)'
  nvcc --version
  rustc --version
  uv pip freeze
  uv build --wheel --no-build-isolation --out-dir "$wheel_output" "$source_dir"
) 2>&1 | tee "$build_log"
