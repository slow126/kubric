#!/usr/bin/env bash
set -euo pipefail

docker run --rm --interactive \
  --user "$(id -u):$(id -g)" \
  --volume "$(pwd):/kubric" \
  kubricdockerhub/kubruntu \
  /usr/bin/python3 interface/preview_movi_tfds.py \
    "$@"
