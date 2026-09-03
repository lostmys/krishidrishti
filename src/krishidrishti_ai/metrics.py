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


def shannon_entropy(probabilities: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Compute Shannon entropy for each probability distribution across classes.

    If normalize=True, scales to [0, 1] by dividing by ln(num_classes).
    """
    eps = 1e-12
    clipped = np.clip(probabilities, eps, 1.0)
    ent = -np.sum(clipped * np.log(clipped), axis=-1)
    if normalize and probabilities.shape[-1] > 1:
        ent = ent / np.log(probabilities.shape[-1])
    return np.asarray(ent, dtype=float)


def prediction_margin(probabilities: np.ndarray) -> np.ndarray:
    """Compute margin between top-1 and top-2 probabilities for each sample."""
    if probabilities.shape[-1] < 2:
        return np.ones(probabilities.shape[0], dtype=float)
    sorted_probs = np.sort(probabilities, axis=-1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    return np.asarray(margin, dtype=float)


def selective_classification_metrics(
    targets: np.ndarray | list[int],
    predictions: np.ndarray | list[int],
    probabilities: np.ndarray,
    confident_threshold: float = 0.75,
    margin_threshold: float = 0.30,
    max_entropy_threshold: float = 0.55,
) -> dict[str, float]:
    """Compute selective classification performance under a conservative decision policy."""
    targets_arr = np.asarray(targets)
    preds_arr = np.asarray(predictions)
    confidences = probabilities.max(axis=1)
    margins = prediction_margin(probabilities)
    entropies = shannon_entropy(probabilities, normalize=True)

    # Conservative acceptance criteria
    accepted_mask = (
        (confidences >= confident_threshold)
        & (margins >= margin_threshold)
        & (entropies <= max_entropy_threshold)
    )

    total_samples = len(targets_arr)
    accepted_count = int(accepted_mask.sum())
    coverage = float(accepted_count / total_samples) if total_samples > 0 else 0.0
    review_rate = 1.0 - coverage

    if accepted_count > 0:
        accuracy_on_accepted = float((preds_arr[accepted_mask] == targets_arr[accepted_mask]).mean())
    else:
        accuracy_on_accepted = 0.0

    return {
        "coverage": round(coverage, 4),
        "accuracy_on_accepted": round(accuracy_on_accepted, 4),
        "review_rate": round(review_rate, 4),
        "accepted_count": accepted_count,
        "total_count": total_samples,
    }


def classification_metrics(
    targets: list[int],
    predictions: list[int],
    probabilities: np.ndarray,
    classes: list[str],
    calibrated_probabilities: np.ndarray | None = None,
    confident_threshold: float = 0.75,
    margin_threshold: float = 0.30,
    max_entropy_threshold: float = 0.55,
) -> dict[str, Any]:
    macro = precision_recall_fscore_support(targets, predictions, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(targets, predictions, average="weighted", zero_division=0)

    raw_confidences = probabilities.max(axis=1)
    raw_margins = prediction_margin(probabilities)
    raw_entropies = shannon_entropy(probabilities, normalize=True)
    raw_ece = expected_calibration_error(probabilities, np.asarray(targets))

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "expected_calibration_error": raw_ece,
        "raw_expected_calibration_error": raw_ece,
        "confidence_distribution": {
            "mean": float(raw_confidences.mean()),
            "std": float(raw_confidences.std()),
            "min": float(raw_confidences.min()),
            "max": float(raw_confidences.max()),
        },
        "margin_distribution": {
            "mean": float(raw_margins.mean()),
            "std": float(raw_margins.std()),
            "min": float(raw_margins.min()),
            "max": float(raw_margins.max()),
        },
        "entropy_distribution": {
            "mean": float(raw_entropies.mean()),
            "std": float(raw_entropies.std()),
            "min": float(raw_entropies.min()),
            "max": float(raw_entropies.max()),
        },
        "per_class": classification_report(targets, predictions, target_names=classes, output_dict=True, zero_division=0),
    }

    eval_probs = probabilities
    if calibrated_probabilities is not None:
        cal_ece = expected_calibration_error(calibrated_probabilities, np.asarray(targets))
        metrics["calibrated_expected_calibration_error"] = cal_ece
        metrics["expected_calibration_error"] = cal_ece  # Primary ECE is calibrated if available
        eval_probs = calibrated_probabilities

    # Selective classification metrics
    metrics["selective_classification"] = selective_classification_metrics(
        targets, predictions, eval_probs, confident_threshold, margin_threshold, max_entropy_threshold
    )

    return metrics


def model_size_mb(model: torch.nn.Module) -> float:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()) / (1024 * 1024)


def benchmark_latency_ms(model: torch.nn.Module, image_size: int, device: torch.device, warmup: int = 10, iterations: int = 50) -> float:
    model.eval()
    batch = torch.zeros((1, 3, image_size, image_size), device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(iterations):
            model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000 / iterations
