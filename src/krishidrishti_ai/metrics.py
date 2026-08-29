from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support


def expected_calibration_error(probabilities: np.ndarray, targets: np.ndarray, bins: int = 15) -> float:
    """Compute top-label expected calibration error; lower is better."""
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    ece = 0.0
    for lower, upper in zip(np.linspace(0.0, 1.0, bins, endpoint=False), np.linspace(1.0 / bins, 1.0, bins)):
        mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            ece += mask.mean() * abs((predictions[mask] == targets[mask]).mean() - confidences[mask].mean())
    return float(ece)


def classification_metrics(targets: list[int], predictions: list[int], probabilities: np.ndarray, classes: list[str]) -> dict[str, Any]:
    macro = precision_recall_fscore_support(targets, predictions, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(targets, predictions, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": float(macro[0]), "macro_recall": float(macro[1]), "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]), "weighted_recall": float(weighted[1]), "weighted_f1": float(weighted[2]),
        "expected_calibration_error": expected_calibration_error(probabilities, np.asarray(targets)),
        "per_class": classification_report(targets, predictions, target_names=classes, output_dict=True, zero_division=0),
    }


def model_size_mb(model: torch.nn.Module) -> float:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()) / (1024 * 1024)


def benchmark_latency_ms(model: torch.nn.Module, image_size: int, device: torch.device, warmup: int = 10, iterations: int = 50) -> float:
    model.eval(); batch = torch.zeros((1, 3, image_size, image_size), device=device)
    with torch.inference_mode():
        for _ in range(warmup): model(batch)
        if device.type == "cuda": torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(iterations): model(batch)
        if device.type == "cuda": torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000 / iterations
