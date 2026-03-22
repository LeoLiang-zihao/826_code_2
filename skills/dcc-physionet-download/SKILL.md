---
name: dcc-physionet-download
description: Set up, accelerate, resume, and verify PhysioNet dataset downloads on Duke DCC with `wget`, `nohup`, `~/.netrc`, manifest files such as `IMAGE_FILENAMES` or `RECORDS`, and multi-process filtered downloads. Use when Codex needs to help a student download any PhysioNet subset or manifest-listed file set on DCC, especially when downloads must survive disconnects, run in parallel, or be checked and restarted safely.
---

# DCC PhysioNet Download

Use this skill to download any PhysioNet file set that can be described by a manifest file and a filter regex.

## Quick Start

1. Prefer a CPU compute node over a login node for long downloads.
2. Download into `/work/<netid>` or `/cwork/<netid>`.
3. Configure `~/.netrc` for `physionet.org` and protect it with `chmod 600`.
4. Test authentication with one file before launching background workers.
5. Launch parallel workers with [`start_parallel_manifest_download.sh`](./scripts/start_parallel_manifest_download.sh).
6. Check progress with [`check_manifest_progress.sh`](./scripts/check_manifest_progress.sh).

## Prerequisites

- Assume the user already has PhysioNet access approval for the target dataset.
- Use `~/.netrc` instead of `--password` on the command line so credentials stay out of shell history and process listings.
- Treat the manifest as the source of truth. The manifest can be `IMAGE_FILENAMES`, `RECORDS`, or any line-based file list where each line is a relative path under the dataset base URL.

Create `~/.netrc` like this:

```bash
cat > ~/.netrc <<'EOF'
machine physionet.org
login YOUR_PHYSIONET_USERNAME
password YOUR_PHYSIONET_PASSWORD
EOF
chmod 600 ~/.netrc
```

## Authentication Test

After creating a filtered worker file, test one line before launching background workers:

```bash
head -n 1 p10.part.aa | wget --netrc -r -N -c -np -nH --cut-dirs=1 -i - --base=https://physionet.org/files/mimic-cxr-jpg/2.1.0/
```

`401 Unauthorized` followed by `Authentication selected` and then `200 OK` or `304 Not Modified` is normal.

## Generic Parallel Download

Use the generic launcher:

```bash
bash scripts/start_parallel_manifest_download.sh \
  --workdir /work/<netid>/mimic-cxr-jpg-2.1.0 \
  --base-url https://physionet.org/files/mimic-cxr-jpg/2.1.0/ \
  --manifest IMAGE_FILENAMES \
  --filter '^files/p10/' \
  --workers 4 \
  --run-name p10
```

Arguments:

- `--workdir`: download workspace on DCC
- `--base-url`: dataset root URL ending with `/`
- `--manifest`: local manifest path, manifest filename inside `workdir`, or direct manifest URL
- `--filter`: `grep -E` regex applied to the manifest. Use `.*` to download everything listed.
- `--workers`: number of parallel `wget` processes
- `--run-name`: short label used for list files and logs
- `--cut-dirs`: optional override for `wget --cut-dirs`, default `1`

The launcher will:

- fetch or reuse the manifest
- build a filtered manifest
- split it into worker parts
- start `nohup wget` workers
- write logs under `logs/<run-name>.part.*.log`

## Generic Progress Check

Use the generic checker:

```bash
bash scripts/check_manifest_progress.sh \
  --workdir /work/<netid>/mimic-cxr-jpg-2.1.0 \
  --base-url https://physionet.org/files/mimic-cxr-jpg/2.1.0/ \
  --manifest IMAGE_FILENAMES \
  --filter '^files/p10/' \
  --run-name p10
```

This prints:

- expected file count from the filtered manifest
- current existing file count derived from the same manifest entries
- completion percentage
- dataset-root disk usage
- active workers

Because the checker maps manifest entries to expected local paths, it works for arbitrary filters, not just `p10`.

## Common Patterns

Download one MIMIC-CXR-JPG subset:

```bash
bash scripts/start_parallel_manifest_download.sh \
  --workdir /work/<netid>/mimic-cxr-jpg-2.1.0 \
  --base-url https://physionet.org/files/mimic-cxr-jpg/2.1.0/ \
  --manifest IMAGE_FILENAMES \
  --filter '^files/p11/' \
  --workers 4 \
  --run-name p11
```

Download everything listed in a manifest:

```bash
bash scripts/start_parallel_manifest_download.sh \
  --workdir /work/<netid>/some-dataset \
  --base-url https://physionet.org/files/some-dataset/1.0/ \
  --manifest RECORDS \
  --filter '.*' \
  --workers 4 \
  --run-name full
```

Download a custom subset from a local list:

```bash
bash scripts/start_parallel_manifest_download.sh \
  --workdir /work/<netid>/mimic-cxr-jpg-2.1.0 \
  --base-url https://physionet.org/files/mimic-cxr-jpg/2.1.0/ \
  --manifest /work/<netid>/my-curated-files.txt \
  --filter 'study1|study2' \
  --workers 3 \
  --run-name curated
```

## Resume Or Repair

- If a worker dies, rerun the same launcher command. `wget -c -N` will resume or skip existing files.
- If an older single-threaded `wget` is still running, stop it before launching parallel workers so the same files are not requested twice.
- If DNS resolution fails temporarily, wait a few minutes and retry. This is separate from authentication failures.
- If the user only wants the legacy MIMIC-CXR-JPG `p10` workflow, use the compatibility wrappers [`start_parallel_subset_download.sh`](./scripts/start_parallel_subset_download.sh) and [`check_subset_progress.sh`](./scripts/check_subset_progress.sh).

## Scope

This skill is optimized for PhysioNet datasets served over HTTP Basic auth where a manifest file lists relative paths under the dataset base URL.
