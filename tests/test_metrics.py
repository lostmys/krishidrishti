import numpy as np

from krishidrishti_ai.metrics import classification_metrics, expected_calibration_error


def test_expected_calibration_error_is_zero_for_perfectly_calibrated_predictions() -> None:
    probabilities = np.array([[0.8, 0.2]] * 5)
    targets = np.array([0, 0, 0, 0, 1])
    assert abs(expected_calibration_error(probabilities, targets, bins=2)) < 1e-12


def test_classification_metrics_include_macro_metrics_and_per_class_report() -> None:
    probabilities = np.array([[0.9, 0.1], [0.7, 0.3], [0.1, 0.9], [0.2, 0.8]])
    metrics = classification_metrics([0, 1, 1, 0], [0, 0, 1, 1], probabilities, ["a", "b"])
    assert metrics["accuracy"] == 0.5
    assert "macro_f1" in metrics
    assert set(metrics["per_class"]).issuperset({"a", "b"})
