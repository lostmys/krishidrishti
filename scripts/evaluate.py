"""Evaluate a trained experiment, including calibration and latency metrics."""
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

from krishidrishti_ai.config import load_config
from krishidrishti_ai.data.transforms import build_transforms
from krishidrishti_ai.metrics import benchmark_latency_ms, classification_metrics, model_size_mb
from krishidrishti_ai.models.classifier import build_model


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args(); config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and config["training"]["device"] == "auto" else "cpu")
    dataset = ImageFolder(Path(config["paths"]["processed_data_dir"]) / "test", build_transforms(config["data"]["image_size"], False))
    checkpoint = torch.load(config["paths"]["checkpoint_path"], map_location=device, weights_only=False)
    if dataset.classes != checkpoint["classes"]: raise ValueError("Test class mapping differs from checkpoint")
    model = build_model(checkpoint.get("model_name", config["model"]["name"]), len(dataset.classes), pretrained=False).to(device); model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    predictions: list[int] = []; targets: list[int] = []; probability_batches: list[np.ndarray] = []
    for images, labels in DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False):
        with torch.inference_mode(): probabilities = torch.softmax(model(images.to(device)), dim=1).cpu().numpy()
        probability_batches.append(probabilities); predictions.extend(probabilities.argmax(axis=1).tolist()); targets.extend(labels.tolist())
    report_dir = Path(config["paths"]["reports_dir"]); report_dir.mkdir(parents=True, exist_ok=True)
    metrics = classification_metrics(targets, predictions, np.concatenate(probability_batches), dataset.classes)
    metrics["model_size_mb"] = model_size_mb(model); metrics["latency_ms_batch_1"] = benchmark_latency_ms(model, config["data"]["image_size"], device, config["evaluation"]["latency_warmup"], config["evaluation"]["latency_iterations"])
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (report_dir / "per_class_report.json").write_text(json.dumps(metrics["per_class"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    matrix = confusion_matrix(targets, predictions); plt.figure(figsize=(max(8, len(dataset.classes)), max(6, len(dataset.classes))))
    sns.heatmap(matrix, annot=True, fmt="d", xticklabels=dataset.classes, yticklabels=dataset.classes, cmap="Blues"); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout(); plt.savefig(report_dir / "confusion_matrix.png", dpi=180); plt.close()
    print(json.dumps({key: value for key, value in metrics.items() if key != "per_class"}, indent=2))


if __name__ == "__main__": main()
