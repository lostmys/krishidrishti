from __future__ import annotations

import numpy as np
import pytest
import torch

from krishidrishti_ai.calibration import (
    TemperatureScaler,
    apply_temperature_scaling,
    fit_temperature_scaling,
    load_calibration,
    save_calibration,
)
from krishidrishti_ai.metrics import expected_calibration_error


def test_temperature_scaler_forward() -> None:
    scaler = TemperatureScaler(init_temperature=2.0)
    logits = torch.tensor([[4.0, 2.0], [1.0, 3.0]])
    scaled = scaler(logits)
    assert torch.allclose(scaled, torch.tensor([[2.0, 1.0], [0.5, 1.5]]))
    assert scaler.get_temperature() == 2.0


def test_fit_temperature_scaling_on_overconfident_logits() -> None:
    # Synthetic overconfident logits
    # True accuracy is 50%, but logits are huge -> softmax confidences ~ 1.0
    torch.manual_seed(42)
    num_samples = 200
    logits = torch.randn(num_samples, 2) * 5.0  # Large spread
    labels = (torch.rand(num_samples) > 0.5).long()

    raw_probs = torch.softmax(logits, dim=1).numpy()
    raw_ece = expected_calibration_error(raw_probs, labels.numpy())

    temp = fit_temperature_scaling(logits, labels)
    assert temp > 1.0  # Temperature should scale down overconfident logits

    cal_probs = apply_temperature_scaling(logits.numpy(), temp)
    cal_ece = expected_calibration_error(cal_probs, labels.numpy())

    assert cal_ece < raw_ece, f"Calibrated ECE ({cal_ece}) should be less than Raw ECE ({raw_ece})"


def test_apply_temperature_scaling_preserves_probability_properties() -> None:
    logits = np.array([[2.5, 0.1, -1.2], [1.0, 3.0, 0.5]])
    cal_probs = apply_temperature_scaling(logits, temperature=1.2)
    assert cal_probs.shape == (2, 3)
    assert np.allclose(cal_probs.sum(axis=1), np.ones(2))
    assert (cal_probs >= 0.0).all() and (cal_probs <= 1.0).all()


def test_save_and_load_calibration(tmp_path) -> None:
    cal_file = tmp_path / "calibration.json"
    data = {
        "experiment": "tomato_mobilenetv3",
        "temperature": 1.181,
        "test_ece_before": 0.00548,
        "test_ece_after": 0.00790,
    }
    save_calibration(cal_file, data)
    assert cal_file.is_file()

    loaded = load_calibration(cal_file)
    assert loaded is not None
    assert loaded["temperature"] == 1.181
    assert loaded["experiment"] == "tomato_mobilenetv3"
