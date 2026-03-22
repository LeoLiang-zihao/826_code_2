from pathlib import Path

from cxr_project.data.synthetic import generate_synthetic_dataset


def test_generate_synthetic_dataset(tmp_path: Path) -> None:
    manifest = generate_synthetic_dataset(tmp_path, num_subjects=12, positives_fraction=0.5, seed=1)
    assert len(manifest) == 12
    assert set(manifest["split"]) == {"train", "val", "test"}
    assert (tmp_path / "manifest.csv").exists()

