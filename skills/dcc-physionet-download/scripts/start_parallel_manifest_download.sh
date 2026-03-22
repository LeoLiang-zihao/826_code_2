#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  start_parallel_manifest_download.sh --workdir DIR --base-url URL --manifest MANIFEST [options]

Options:
  --filter REGEX      grep -E regex applied to the manifest (default: .*)
  --workers N         number of parallel workers (default: 4)
  --run-name NAME     label used for part files and logs
  --cut-dirs N        wget --cut-dirs value (default: 1)
EOF
  exit 1
}

workdir=""
base_url=""
manifest=""
filter=".*"
workers="4"
run_name=""
cut_dirs="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workdir) workdir=${2:-}; shift 2 ;;
    --base-url) base_url=${2:-}; shift 2 ;;
    --manifest) manifest=${2:-}; shift 2 ;;
    --filter) filter=${2:-}; shift 2 ;;
    --workers) workers=${2:-}; shift 2 ;;
    --run-name) run_name=${2:-}; shift 2 ;;
    --cut-dirs) cut_dirs=${2:-}; shift 2 ;;
    *) usage ;;
  esac
done

if [[ -z "$workdir" || -z "$base_url" || -z "$manifest" ]]; then
  usage
fi

if ! [[ "$workers" =~ ^[1-9][0-9]*$ ]]; then
  echo "workers must be a positive integer" >&2
  exit 1
fi

if ! [[ "$cut_dirs" =~ ^[0-9]+$ ]]; then
  echo "cut-dirs must be a non-negative integer" >&2
  exit 1
fi

if [[ ! -f "$HOME/.netrc" ]]; then
  echo "~/.netrc not found" >&2
  exit 1
fi

mkdir -p "$workdir/logs"
cd "$workdir"

manifest_basename=$(basename "$manifest")
manifest_local="$workdir/$manifest_basename"

if [[ "$manifest" == http://* || "$manifest" == https://* ]]; then
  wget -N -c -O "$manifest_local" "$manifest"
elif [[ -f "$manifest" ]]; then
  if [[ "$(cd "$(dirname "$manifest")" && pwd)/$(basename "$manifest")" != "$manifest_local" ]]; then
    cp "$manifest" "$manifest_local"
  fi
elif [[ -f "$workdir/$manifest" ]]; then
  manifest_local="$workdir/$manifest"
else
  candidate_url="${base_url}${manifest}"
  wget -N -c -O "$manifest_local" "$candidate_url"
fi

if [[ ! -s "$manifest_local" ]]; then
  echo "Manifest is missing or empty: $manifest_local" >&2
  exit 1
fi

if [[ -z "$run_name" ]]; then
  run_name=$(printf '%s\n' "${manifest_basename}_${filter}" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/_+/_/g; s/^_//; s/_$//')
  run_name=${run_name:0:60}
  if [[ -z "$run_name" ]]; then
    run_name="download"
  fi
fi

list_file="$workdir/${run_name}.list"
part_prefix="$workdir/${run_name}.part."

grep -E "$filter" "$manifest_local" > "$list_file"

if [[ ! -s "$list_file" ]]; then
  echo "No entries matched filter '$filter' in $manifest_local" >&2
  exit 1
fi

rm -f "${part_prefix}"*
split -n "l/${workers}" "$list_file" "$part_prefix"

for part in "${part_prefix}"*; do
  label=$(basename "$part")
  log_file="$workdir/logs/${label}.log"
  nohup wget --netrc -r -N -c -np -nH --cut-dirs="$cut_dirs" -i "$part" --base="$base_url" > "$log_file" 2>&1 &
  echo "started $label -> $log_file"
done

echo "manifest: $manifest_local"
echo "filter: $filter"
echo "workers: $workers"
echo "run-name: $run_name"
echo "check: bash scripts/check_manifest_progress.sh --workdir \"$workdir\" --base-url \"$base_url\" --manifest \"$manifest_local\" --filter \"$filter\" --run-name \"$run_name\" --cut-dirs \"$cut_dirs\""
