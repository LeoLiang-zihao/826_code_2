from __future__ import annotations

import argparse

from cxr_project.data.synthetic import generate_synthetic_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a synthetic chest X-ray style dataset.")
    parser.add_argument("--output-dir", default="data/synthetic")
    parser.add_argument("--num-subjects", type=int, default=36)
    parser.add_argument("--positives-fraction", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=826)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_synthetic_dataset(
        output_dir=args.output_dir,
        num_subjects=args.num_subjects,
        positives_fraction=args.positives_fraction,
        image_size=args.image_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

