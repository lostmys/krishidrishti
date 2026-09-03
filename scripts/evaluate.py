"""Evaluate a trained experiment, including post-hoc calibration and selective decision metrics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from krishidrishti_ai.calibration import apply_temperature_scaling, load_calibration
from krishidrishti_ai.config import load_config
from krishidrishti_ai.data.transforms import build_transforms
from krishidrishti_ai.metrics import benchmark_latency_ms, classification_metrics, model_size_mb
from krishidrishti_ai.models.classifier import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and config["training"]["device"] == "auto" else "cpu")
    dataset = ImageFolder(
        Path(config["paths"]["processed_data_dir"]) / "test",
        build_transforms(config["data"]["image_size"], False),
    )

    checkpoint_path = Path(config["paths"]["checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if dataset.classes != checkpoint["classes"]:
        raise ValueError("Test class mapping differs from checkpoint")

    model = build_model(checkpoint.get("model_name", config["model"]["name"]), len(dataset.classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load calibrated temperature
    inf_cfg = config.get("inference", {})
    temperature = float(inf_cfg.get("temperature", 1.0))
    cal_file = checkpoint_path.parent / "calibration.json"
    cal_data = load_calibration(cal_file)
    if cal_data and "temperature" in cal_data:
        temperature = float(cal_data["temperature"])

    predictions: list[int] = []
    targets: list[int] = []
    raw_prob_batches: list[np.ndarray] = []
    cal_prob_batches: list[np.ndarray] = []

    for images, labels in DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False):
        with torch.inference_mode():
            logits = model(images.to(device)).cpu()
            raw_probs = torch.softmax(logits, dim=1).numpy()
            cal_probs = apply_temperature_scaling(logits.numpy(), temperature)

        raw_prob_batches.append(raw_probs)
        cal_prob_batches.append(cal_probs)
        predictions.extend(cal_probs.argmax(axis=1).tolist())
        targets.extend(labels.tolist())

    raw_probabilities = np.concatenate(raw_prob_batches, axis=0)
    cal_probabilities = np.concatenate(cal_prob_batches, axis=0)

    report_dir = Path(config["paths"]["reports_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics = classification_metrics(
        targets=targets,
        predictions=predictions,
        probabilities=raw_probabilities,
        classes=dataset.classes,
        calibrated_probabilities=cal_probabilities,
        confident_threshold=float(inf_cfg.get("confident_threshold", 0.75)),
        margin_threshold=float(inf_cfg.get("ambiguity_margin_threshold", 0.30)),
        max_entropy_threshold=float(inf_cfg.get("max_entropy_threshold", 0.55)),
    )

    metrics["temperature"] = temperature
    metrics["model_size_mb"] = model_size_mb(model)
    metrics["latency_ms_batch_1"] = benchmark_latency_ms(
        model,
        config["data"]["image_size"],
        device,
        config["evaluation"]["latency_warmup"],
        config["evaluation"]["latency_iterations"],
    )

    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (report_dir / "per_class_report.json").write_text(
        json.dumps(metrics["per_class"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    matrix = confusion_matrix(targets, predictions)
    plt.figure(figsize=(max(8, len(dataset.classes)), max(6, len(dataset.classes))))
    sns.heatmap(matrix, annot=True, fmt="d", xticklabels=dataset.classes, yticklabels=dataset.classes, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(report_dir / "confusion_matrix.png", dpi=180)
    plt.close()

    print(json.dumps({key: value for key, value in metrics.items() if key != "per_class"}, indent=2))


if __name__ == "__main__":
    main()
