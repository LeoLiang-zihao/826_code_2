from pathlib import Path

from cxr_project.data.synthetic import generate_synthetic_dataset


def test_train_manifest_created(tmp_path: Path) -> None:
    manifest = generate_synthetic_dataset(tmp_path, num_subjects=18, positives_fraction=0.5, seed=2)
    assert Path(manifest.iloc[0]["image_path"]).exists()
