#!/usr/bin/env bash
# Run render_intervention_scenelets.py inside the kubruntu container.
# All arguments are forwarded to the python driver, e.g.:
#   interface/run_intervention_scenelets_docker.sh \
#     --theta-json /kubric/theta.json --output-dir /kubric/out --n-pairs 5
# Paths must be container paths (the repo is mounted at /kubric).
#
# Env knobs (all optional):
#   KUBRIC_CUDA_DEVICE          GPU index to expose            (default 1)
#   KUBRIC_DOCKER_USE_GPUS_FLAG pass --gpus device=<idx>       (default 0)
#   KUBRIC_USE_GPU              let Blender use the GPU         (default 0/CPU)
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
  --env "KUBRIC_USE_GPU=${KUBRIC_USE_GPU:-0}" \
  --user "$(id -u):$(id -g)" \
  --volume "$(pwd):/kubric" \
  kubricdockerhub/kubruntu \
  /usr/bin/python3 interface/render_intervention_scenelets.py \
    "$@"
