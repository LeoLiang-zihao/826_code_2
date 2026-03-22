from __future__ import annotations

import argparse
from pathlib import Path

from cxr_project.data.manifest import build_mimic_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a manifest for a MIMIC-CXR-JPG subset.")
    parser.add_argument("--labels-path", required=True)
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--pathology", default="Pleural Effusion")
    parser.add_argument("--negative-ratio", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=826)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_mimic_manifest(
        labels_path=args.labels_path,
        metadata_path=args.metadata_path,
        image_root=args.image_root,
        pathology=args.pathology,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
    )
    output_path = Path(args.output_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    print(f"Saved manifest with {len(manifest)} rows to {output_path}")


if __name__ == "__main__":
    main()

