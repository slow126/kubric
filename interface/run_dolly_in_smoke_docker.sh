#!/usr/bin/env bash
set -euo pipefail

cuda_device="${KUBRIC_CUDA_DEVICE:-1}"
gpu_args=()
if [[ "${KUBRIC_DOCKER_USE_GPUS_FLAG:-0}" == "1" ]]; then
  gpu_args+=(--gpus "device=${cuda_device}")
fi

docker run --rm --interactive \
  "${gpu_args[@]}" \
  --env "CUDA_VISIBLE_DEVICES=${cuda_device}" \
  --env "NVIDIA_VISIBLE_DEVICES=${cuda_device}" \
  --user "$(id -u):$(id -g)" \
  --volume "$(pwd):/kubric" \
  kubricdockerhub/kubruntu \
  /usr/bin/python3 interface/run_dolly_in_smoke.py \
    --output-dir interface/output/dolly_in_smoke \
    --scratch-dir interface/output/dolly_in_smoke_scratch \
    --resolution 192x192 \
    --frame-count 4 \
    --samples-per-pixel 32 \
    "$@"
