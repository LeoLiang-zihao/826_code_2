from pathlib import Path

from cxr_project.data.dataset import ChestXrayDataset
from cxr_project.data.synthetic import generate_synthetic_dataset
from cxr_project.data.transforms import build_eval_transforms
from cxr_project.models.attribution import GradCAM, save_cam_figure
from cxr_project.models.classifier import LightningBinaryClassifier


def test_gradcam_generates_figure(tmp_path: Path) -> None:
    manifest = generate_synthetic_dataset(tmp_path, num_subjects=12, positives_fraction=0.5, seed=5)
    dataset = ChestXrayDataset(manifest.iloc[:1].copy(), transform=build_eval_transforms(64))
    sample = dataset[0]

    model = LightningBinaryClassifier(
        model_name="resnet18",
        pretrained=False,
        fine_tune_mode="head_only",
        learning_rate=1e-3,
        weight_decay=1e-4,
    )
    cam = GradCAM(model, model.target_layer)
    try:
        image_tensor = sample["image"].unsqueeze(0)
        heatmap = cam.generate(image_tensor)
        output_path = tmp_path / "gradcam.png"
        save_cam_figure(
            original_image_path=sample["image_path"],
            normalized_tensor=sample["image"],
            cam=heatmap,
            probability=0.5,
            label=int(sample["label"]),
            output_path=output_path,
        )
    finally:
        cam.close()

    assert output_path.exists()
