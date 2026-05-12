#!/usr/bin/env bash
set -euo pipefail

# Mirror the public pre-rendered MOVi-F TFDS files locally.
#
# Default:
#   dataset/config: movi_f/512x512
#   source root:    gs://kubric-public/tfds
#   local root:     ./data/kubric_tfds
#
# After this finishes, load with:
#   tfds.load("movi_f/512x512", data_dir="./data/kubric_tfds", split="train")

config="${MOVI_F_CONFIG:-512x512}"
source_root="${KUBRIC_TFDS_SOURCE:-gs://kubric-public/tfds}"
target_root="${KUBRIC_TFDS_TARGET:-data/kubric_tfds}"
version="${MOVI_F_VERSION:-1.0.0}"

source_path="${source_root}/movi_f/${config}/${version}"
target_path="${target_root}/movi_f/${config}/${version}"

mkdir -p "${target_path}"

if command -v gcloud >/dev/null 2>&1; then
  gcloud storage cp --recursive "${source_path}/*" "${target_path}/"
elif command -v gsutil >/dev/null 2>&1; then
  gsutil -m cp -r "${source_path}/*" "${target_path}/"
else
  echo "Neither gcloud nor gsutil is available." >&2
  echo "Install Google Cloud CLI, then rerun this script." >&2
  exit 1
fi

echo "Downloaded MOVi-F ${config} TFDS files to ${target_path}"
echo "Use: tfds.load(\"movi_f/${config}\", data_dir=\"${target_root}\", split=\"train\")"

