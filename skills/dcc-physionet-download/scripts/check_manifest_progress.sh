#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  check_manifest_progress.sh --workdir DIR --base-url URL --manifest MANIFEST [options]

Options:
  --filter REGEX      grep -E regex applied to the manifest (default: .*)
  --run-name NAME     label used for part files and logs
  --cut-dirs N        wget --cut-dirs value (default: 1)
EOF
  exit 1
}

workdir=""
base_url=""
manifest=""
filter=".*"
run_name=""
cut_dirs="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workdir) workdir=${2:-}; shift 2 ;;
    --base-url) base_url=${2:-}; shift 2 ;;
    --manifest) manifest=${2:-}; shift 2 ;;
    --filter) filter=${2:-}; shift 2 ;;
    --run-name) run_name=${2:-}; shift 2 ;;
    --cut-dirs) cut_dirs=${2:-}; shift 2 ;;
    *) usage ;;
  esac
done

if [[ -z "$workdir" || -z "$base_url" || -z "$manifest" ]]; then
  usage
fi

if ! [[ "$cut_dirs" =~ ^[0-9]+$ ]]; then
  echo "cut-dirs must be a non-negative integer" >&2
  exit 1
fi

cd "$workdir"

if [[ -f "$manifest" ]]; then
  manifest_local="$manifest"
elif [[ -f "$workdir/$manifest" ]]; then
  manifest_local="$workdir/$manifest"
else
  echo "Manifest not found: $manifest" >&2
  exit 1
fi

if [[ -z "$run_name" ]]; then
  run_name=$(printf '%s\n' "$(basename "$manifest_local")_${filter}" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/_+/_/g; s/^_//; s/_$//')
  run_name=${run_name:0:60}
  if [[ -z "$run_name" ]]; then
    run_name="download"
  fi
fi

filtered_list="$workdir/${run_name}.list"
if [[ -f "$filtered_list" ]]; then
  source_list="$filtered_list"
else
  source_list="$workdir/.${run_name}.tmp.list"
  grep -E "$filter" "$manifest_local" > "$source_list"
fi

expected=$(wc -l < "$source_list" | tr -d ' ')

base_path=$(printf '%s\n' "$base_url" | sed -E 's#^[A-Za-z]+://[^/]+/##; s#/$##')
local_root=$(printf '%s\n' "$base_path" | cut -d/ -f$((cut_dirs + 1))-)
if [[ -z "$local_root" || "$local_root" == "$base_path" && "$cut_dirs" -gt 0 && "$base_path" != */* ]]; then
  local_root="."
fi

current=0
while IFS= read -r relpath; do
  [[ -z "$relpath" ]] && continue
  relpath=${relpath#./}
  local_path="$workdir"
  if [[ "$local_root" != "." ]]; then
    local_path="$local_path/$local_root"
  fi
  local_path="$local_path/$relpath"
  if [[ -f "$local_path" ]]; then
    current=$((current + 1))
  fi
done < "$source_list"

dataset_dir="$workdir"
if [[ "$local_root" != "." ]]; then
  dataset_dir="$workdir/$local_root"
fi
if [[ -d "$dataset_dir" ]]; then
  size=$(du -sh "$dataset_dir" | awk '{print $1}')
else
  size=0
fi

awk -v current="$current" -v expected="$expected" 'BEGIN {
  pct = (expected == 0 ? 0 : 100 * current / expected)
  printf("Progress: %d / %d (%.2f%%)\n", current, expected, pct)
}'
echo "Manifest: $manifest_local"
echo "Filter: $filter"
echo "Dataset root: $dataset_dir"
echo "Disk usage: $size"
echo "Active workers:"
pgrep -af "wget.*${run_name}\.part\.|wget.*$(printf '%s' "$base_url" | sed 's/[.[\\*^$()+?{}|]/\\&/g')" || true

if [[ "$source_list" == "$workdir/.${run_name}.tmp.list" ]]; then
  rm -f "$source_list"
fi
