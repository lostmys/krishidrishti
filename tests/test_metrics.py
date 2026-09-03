from __future__ import annotations

import numpy as np

from krishidrishti_ai.metrics import (
    classification_metrics,
    expected_calibration_error,
    prediction_margin,
    selective_classification_metrics,
    shannon_entropy,
)


def test_expected_calibration_error_is_zero_for_perfectly_calibrated_predictions() -> None:
    probabilities = np.array([[0.8, 0.2]] * 5)
    targets = np.array([0, 0, 0, 0, 1])
    assert abs(expected_calibration_error(probabilities, targets, bins=2)) < 1e-12


def test_shannon_entropy_calculation() -> None:
    # Uniform distribution over 2 classes -> max normalized entropy 1.0
    uniform_probs = np.array([[0.5, 0.5]])
    assert np.isclose(shannon_entropy(uniform_probs, normalize=True)[0], 1.0)

    # Deterministic distribution -> min entropy 0.0
    certain_probs = np.array([[1.0, 0.0]])
    assert np.isclose(shannon_entropy(certain_probs, normalize=True)[0], 0.0)


def test_prediction_margin_calculation() -> None:
    probs = np.array([[0.7, 0.2, 0.1], [0.55, 0.45, 0.0]])
    margins = prediction_margin(probs)
    assert np.isclose(margins[0], 0.5)
    assert np.isclose(margins[1], 0.1)


def test_selective_classification_metrics() -> None:
    targets = np.array([0, 1, 0, 1])
    predictions = np.array([0, 1, 1, 1])
    # Samples:
    # 0: conf=0.95, margin=0.90, ent=0.286 -> ACCEPTED, correct
    # 1: conf=0.90, margin=0.80, ent=0.469 -> ACCEPTED, correct
    # 2: conf=0.55, margin=0.10, ent=0.993 -> REJECTED / REVIEW
    # 3: conf=0.60, margin=0.20, ent=0.971 -> REJECTED / REVIEW
    probs = np.array([
        [0.95, 0.05],
        [0.10, 0.90],
        [0.55, 0.45],
        [0.40, 0.60],
    ])
    res = selective_classification_metrics(
        targets, predictions, probs, confident_threshold=0.75, margin_threshold=0.30, max_entropy_threshold=0.55
    )
    assert res["accepted_count"] == 2
    assert res["coverage"] == 0.5
    assert res["accuracy_on_accepted"] == 1.0
    assert res["review_rate"] == 0.5


def test_classification_metrics_include_macro_metrics_and_per_class_report() -> None:
    probabilities = np.array([[0.9, 0.1], [0.7, 0.3], [0.1, 0.9], [0.2, 0.8]])
    cal_probabilities = np.array([[0.85, 0.15], [0.65, 0.35], [0.15, 0.85], [0.25, 0.75]])
    metrics = classification_metrics(
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        probabilities,
        ["a", "b"],
        calibrated_probabilities=cal_probabilities,
    )
    assert metrics["accuracy"] == 0.5
    assert "macro_f1" in metrics
    assert "selective_classification" in metrics
    assert "confidence_distribution" in metrics
    assert "margin_distribution" in metrics
    assert "entropy_distribution" in metrics
    assert set(metrics["per_class"]).issuperset({"a", "b"})
