#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 <workdir> [subset=p10] [workers=4]" >&2
  exit 1
fi

workdir=$1
subset=${2:-p10}
workers=${3:-4}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

bash "$script_dir/start_parallel_manifest_download.sh" \
  --workdir "$workdir" \
  --base-url "https://physionet.org/files/mimic-cxr-jpg/2.1.0/" \
  --manifest "IMAGE_FILENAMES" \
  --filter "^files/${subset}/" \
  --workers "$workers" \
  --run-name "${subset//\//_}"
