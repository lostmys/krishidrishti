"""Tests for Phase 3C Tomato Robustness hardening."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision import transforms

from krishidrishti_ai.config import load_config
from krishidrishti_ai.data.transforms import build_transforms
from krishidrishti_ai.models.classifier import build_model
from krishidrishti_ai.services.inference import DiagnosisService

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_build_transforms_robust_includes_expected_stages():
    t_std = build_transforms(224, training=True, robust=False)
    t_rob = build_transforms(224, training=True, robust=True)

    # Standard has HFlip, Rotation, ColorJitter, ToTensor, Normalize
    assert len(t_std.transforms) == 6
    # Robust has Resize, HFlip, VFlip, Rotation, ColorJitter, ToTensor, Normalize, RandomErasing (8 total)
    assert len(t_rob.transforms) == 8

    # Verify VFlip and RandomErasing present
    types = [type(t) for t in t_rob.transforms]
    assert transforms.RandomVerticalFlip in types
    assert transforms.RandomErasing in types


def test_robust_config_inherits_and_resolves():
    cfg_path = PROJECT_ROOT / "configs" / "experiments" / "tomato_mobilenetv3_robust.yaml"
    assert cfg_path.is_file(), "Robust config file must exist"

    cfg = load_config(cfg_path)
    assert cfg["model"]["name"] == "mobilenet_v3_small"
    assert cfg["training"]["robust_augmentation"] is True
    assert "tomato_robust" in cfg["paths"]["processed_data_dir"]
    assert Path(cfg["paths"]["checkpoint_path"]).name == "mobilenetv3_robust.pt"


def test_realworld_manifest_integrity():
    manifest_path = PROJECT_ROOT / "data" / "raw" / "tomato_realworld" / "manifest.json"
    assert manifest_path.is_file(), "Real-world dataset manifest must exist"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "splits" in manifest
    assert "totals" in manifest
    assert manifest["totals"]["realworld_holdout"] == 70
    assert manifest["totals"]["train"] == 577
    assert manifest["totals"]["val"] == 97


def test_robust_checkpoint_exists_and_matches_10_classes():
    robust_ckpt_path = PROJECT_ROOT / "artifacts" / "checkpoints" / "tomato" / "mobilenetv3_robust.pt"
    baseline_ckpt_path = PROJECT_ROOT / "artifacts" / "checkpoints" / "tomato" / "mobilenetv3_baseline.pt"

    assert robust_ckpt_path.is_file(), "Robust checkpoint must exist"
    assert baseline_ckpt_path.is_file(), "Baseline checkpoint must remain untouched"

    r_ckpt = torch.load(robust_ckpt_path, map_location="cpu", weights_only=False)
    b_ckpt = torch.load(baseline_ckpt_path, map_location="cpu", weights_only=False)

    assert len(r_ckpt["classes"]) == 10
    assert r_ckpt["classes"] == b_ckpt["classes"], "Class list and order must be 100% identical"


def test_robust_calibration_artifact_valid():
    cal_path = PROJECT_ROOT / "artifacts" / "checkpoints" / "tomato" / "calibration_robust.json"
    assert cal_path.is_file(), "Robust calibration artifact must exist"

    cal_data = json.loads(cal_path.read_text(encoding="utf-8"))
    assert cal_data["temperature"] > 0
    assert "validation_ece_before" in cal_data
    assert "validation_ece_after" in cal_data


def test_robust_model_predicts_healthy_for_failure_case():
    healthy_jpg = Path(r"C:\Users\Vaibhav Sharma\Downloads\healthy.jpg")
    if not healthy_jpg.is_file():
        pytest.skip("healthy.jpg not found on machine")

    cfg = load_config(PROJECT_ROOT / "configs" / "experiments" / "tomato_mobilenetv3_robust.yaml")
    service = DiagnosisService(cfg)

    result = service.diagnose(healthy_jpg.read_bytes())
    # Crucial assertion: the robust model must diagnose the healthy leaf as healthy!
    assert result["diagnosis"] == "healthy", f"Expected 'healthy', got '{result['diagnosis']}'"
    assert result["crop"] == "Tomato"
