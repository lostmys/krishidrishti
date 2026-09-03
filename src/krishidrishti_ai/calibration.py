"""Post-hoc probability calibration module using Temperature Scaling."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim

from krishidrishti_ai.metrics import expected_calibration_error


class TemperatureScaler(nn.Module):
    """Wraps model logits with a learnable temperature parameter T > 0."""

    def __init__(self, init_temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-3)

    def get_temperature(self) -> float:
        return float(self.temperature.item())


def fit_temperature_scaling(
    logits: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    max_iter: int = 100,
    lr: float = 0.01,
) -> float:
    """Fit temperature parameter T > 0 on validation set logits to minimize Negative Log Likelihood.

    Must ONLY be fitted on validation data; never on test data.
    """
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits).float()
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels).long()

    scaler = TemperatureScaler(init_temperature=1.5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS([scaler.temperature], lr=lr, max_iter=max_iter)

    def _eval() -> torch.Tensor:
        optimizer.zero_grad()
        loss = criterion(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(_eval)
    return round(scaler.get_temperature(), 4)


def apply_temperature_scaling(
    logits: torch.Tensor | np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """Apply temperature scaling to logits and return calibrated softmax probabilities."""
    if temperature <= 0:
        temperature = 1.0

    if isinstance(logits, np.ndarray):
        logits_tensor = torch.from_numpy(logits).float()
    else:
        logits_tensor = logits.float()

    with torch.inference_mode():
        calibrated_probs = torch.softmax(logits_tensor / temperature, dim=-1)

    return calibrated_probs.cpu().numpy()


def save_calibration(path: str | Path, calibration_data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(calibration_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_calibration(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.is_file():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return None
