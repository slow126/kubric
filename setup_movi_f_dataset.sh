#!/usr/bin/env bash
set -euo pipefail

# Download the public pre-rendered MOVi-F TFDS dataset to local storage.
#
# Cluster default:
#   ./setup_movi_f_dataset.sh
# writes:
#   /home/slow1/Data/movi_f/512x512/1.0.0
#
# Override the destination:
#   ./setup_movi_f_dataset.sh /path/to/tfds_root
#   ./setup_movi_f_dataset.sh --target-root /path/to/tfds_root
#
# After this finishes, load with:
#   tfds.load("movi_f/512x512", data_dir="/home/slow1/Data", split="train")

usage() {
  cat <<'EOF'
Usage:
  setup_movi_f_dataset.sh [TARGET_ROOT]
  setup_movi_f_dataset.sh [options]

Options:
  --target-root PATH   TFDS data root. Default: /home/slow1/Data
  --config NAME        MOVi-F config. Default: 512x512
  --version VERSION    Dataset version. Default: 1.0.0
  --source-root URI    Source TFDS root. Default: gs://kubric-public/tfds
  -h, --help           Show this help.

Environment overrides:
  KUBRIC_TFDS_TARGET   Same as --target-root
  MOVI_F_CONFIG        Same as --config
  MOVI_F_VERSION       Same as --version
  KUBRIC_TFDS_SOURCE   Same as --source-root
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${script_dir}"

target_root="${KUBRIC_TFDS_TARGET:-/home/slow1/Data}"
config="${MOVI_F_CONFIG:-512x512}"
version="${MOVI_F_VERSION:-1.0.0}"
source_root="${KUBRIC_TFDS_SOURCE:-gs://kubric-public/tfds}"

while (($#)); do
  case "$1" in
    --target-root)
      target_root="${2:?Missing value for --target-root}"
      shift 2
      ;;
    --config)
      config="${2:?Missing value for --config}"
      shift 2
      ;;
    --version)
      version="${2:?Missing value for --version}"
      shift 2
      ;;
    --source-root)
      source_root="${2:?Missing value for --source-root}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      target_root="$1"
      shift
      ;;
  esac
done

source_path="${source_root}/movi_f/${config}/${version}"
target_path="${target_root}/movi_f/${config}/${version}"

echo "Repository: ${repo_root}"
echo "Source:     ${source_path}"
echo "Target:     ${target_path}"

mkdir -p "${target_path}"

if command -v gcloud >/dev/null 2>&1; then
  gcloud storage cp --recursive "${source_path}/*" "${target_path}/"
elif command -v gsutil >/dev/null 2>&1; then
  gsutil -m cp -r "${source_path}/*" "${target_path}/"
else
  echo "Neither gcloud nor gsutil is available." >&2
  echo "Install Google Cloud CLI on the cluster, then rerun this script." >&2
  exit 1
fi

echo
echo "Downloaded MOVi-F ${config} TFDS files to:"
echo "  ${target_path}"
echo
echo "Use with TensorFlow Datasets:"
echo "  tfds.load(\"movi_f/${config}\", data_dir=\"${target_root}\", split=\"train\")"
