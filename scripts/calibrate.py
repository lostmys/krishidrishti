"""Fit temperature scaling calibration on validation data and report ECE before/after."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from krishidrishti_ai.calibration import apply_temperature_scaling, fit_temperature_scaling, save_calibration
from krishidrishti_ai.config import load_config
from krishidrishti_ai.data.transforms import build_transforms
from krishidrishti_ai.metrics import expected_calibration_error
from krishidrishti_ai.models.classifier import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate classifier using validation data.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and config["training"]["device"] == "auto" else "cpu")
    processed_dir = Path(config["paths"]["processed_data_dir"])
    image_size = config["data"]["image_size"]

    val_dataset = ImageFolder(processed_dir / "val", build_transforms(image_size, training=False))
    test_dataset = ImageFolder(processed_dir / "test", build_transforms(image_size, training=False))

    checkpoint_path = Path(config["paths"]["checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = checkpoint["classes"]

    model = build_model(checkpoint.get("model_name", config["model"]["name"]), len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 1. Collect validation logits (STRICTLY used for fitting calibration)
    val_logits_list: list[torch.Tensor] = []
    val_targets_list: list[torch.Tensor] = []
    for images, labels in DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False):
        with torch.inference_mode():
            val_logits_list.append(model(images.to(device)).cpu())
            val_targets_list.append(labels)

    val_logits = torch.cat(val_logits_list)
    val_targets = torch.cat(val_targets_list)

    # 2. Fit optimal temperature T* on validation set
    optimal_temperature = fit_temperature_scaling(val_logits, val_targets)

    # 3. Evaluate validation ECE before / after
    val_raw_probs = torch.softmax(val_logits, dim=1).numpy()
    val_cal_probs = apply_temperature_scaling(val_logits.numpy(), optimal_temperature)
    val_ece_raw = expected_calibration_error(val_raw_probs, val_targets.numpy())
    val_ece_cal = expected_calibration_error(val_cal_probs, val_targets.numpy())

    # 4. Evaluate untouched test set ECE before / after
    test_logits_list: list[torch.Tensor] = []
    test_targets_list: list[torch.Tensor] = []
    for images, labels in DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False):
        with torch.inference_mode():
            test_logits_list.append(model(images.to(device)).cpu())
            test_targets_list.append(labels)

    test_logits = torch.cat(test_logits_list)
    test_targets = torch.cat(test_targets_list)

    test_raw_probs = torch.softmax(test_logits, dim=1).numpy()
    test_cal_probs = apply_temperature_scaling(test_logits.numpy(), optimal_temperature)
    test_ece_raw = expected_calibration_error(test_raw_probs, test_targets.numpy())
    test_ece_cal = expected_calibration_error(test_cal_probs, test_targets.numpy())

    # 5. Save calibration artifact alongside checkpoint
    calibration_file = checkpoint_path.parent / "calibration.json"
    calibration_data = {
        "experiment": config.get("experiment", {}).get("name", checkpoint_path.stem),
        "temperature": optimal_temperature,
        "validation_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "validation_ece_before": round(val_ece_raw, 6),
        "validation_ece_after": round(val_ece_cal, 6),
        "test_ece_before": round(test_ece_raw, 6),
        "test_ece_after": round(test_ece_cal, 6),
    }
    save_calibration(calibration_file, calibration_data)

    print("=" * 60)
    print(f"Calibration Complete for {checkpoint_path.name}")
    print(f"Artifact Saved: {calibration_file}")
    print("=" * 60)
    print(json.dumps(calibration_data, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()
