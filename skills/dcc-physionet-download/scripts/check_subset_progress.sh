#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <workdir> [subset=p10]" >&2
  exit 1
fi

workdir=$1
subset=${2:-p10}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

bash "$script_dir/check_manifest_progress.sh" \
  --workdir "$workdir" \
  --base-url "https://physionet.org/files/mimic-cxr-jpg/2.1.0/" \
  --manifest "IMAGE_FILENAMES" \
  --filter "^files/${subset}/" \
  --run-name "${subset//\//_}"
