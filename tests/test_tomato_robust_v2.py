"""Tests for Phase 3D Tomato Robust V2."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from krishidrishti_ai.config import load_config
from krishidrishti_ai.services.inference import DiagnosisService

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_robust_v2_config_inherits_and_resolves():
    cfg_path = PROJECT_ROOT / "configs" / "experiments" / "tomato_mobilenetv3_robust_v2.yaml"
    assert cfg_path.is_file(), "Robust V2 config file must exist"

    cfg = load_config(cfg_path)
    assert cfg["model"]["name"] == "mobilenet_v3_small"
    assert cfg["training"]["robust_augmentation"] is True
    assert "tomato_robust_v2" in cfg["paths"]["processed_data_dir"]
    assert Path(cfg["paths"]["checkpoint_path"]).name == "mobilenetv3_robust_v2.pt"


def test_tomatoleaves_manifest_integrity():
    manifest_path = PROJECT_ROOT / "data" / "raw" / "tomatoleaves_realworld" / "manifest.json"
    assert manifest_path.is_file(), "TomatoLeaves manifest must exist"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "splits" in manifest
    assert sum(manifest["splits"]["train"].values()) == 4794
    assert sum(manifest["splits"]["val"].values()) == 1020
    assert sum(manifest["splits"]["test"].values()) == 1020


def test_robust_v2_checkpoint_matches_10_classes_and_baseline():
    v2_path = PROJECT_ROOT / "artifacts" / "checkpoints" / "tomato" / "mobilenetv3_robust_v2.pt"
    v1_path = PROJECT_ROOT / "artifacts" / "checkpoints" / "tomato" / "mobilenetv3_robust.pt"
    base_path = PROJECT_ROOT / "artifacts" / "checkpoints" / "tomato" / "mobilenetv3_baseline.pt"

    assert v2_path.is_file(), "Robust V2 checkpoint must exist"
    assert v1_path.is_file(), "Robust V1 checkpoint must be untouched"
    assert base_path.is_file(), "Baseline checkpoint must be untouched"

    v2_ckpt = torch.load(v2_path, map_location="cpu", weights_only=False)
    v1_ckpt = torch.load(v1_path, map_location="cpu", weights_only=False)
    b_ckpt = torch.load(base_path, map_location="cpu", weights_only=False)

    assert len(v2_ckpt["classes"]) == 10
    assert v2_ckpt["classes"] == b_ckpt["classes"] == v1_ckpt["classes"]


def test_robust_v2_calibration_artifact_valid():
    cal_path = PROJECT_ROOT / "artifacts" / "checkpoints" / "tomato" / "calibration_robust_v2.json"
    assert cal_path.is_file(), "Robust V2 calibration artifact must exist"

    cal_data = json.loads(cal_path.read_text(encoding="utf-8"))
    assert cal_data["temperature"] > 0
    assert "validation_ece_before" in cal_data
    assert "validation_ece_after" in cal_data
    assert cal_data["validation_ece_after"] <= cal_data["validation_ece_before"]


def test_robust_v2_predicts_healthy_for_failure_case():
    healthy_jpg = Path(r"C:\Users\Vaibhav Sharma\Downloads\healthy.jpg")
    if not healthy_jpg.is_file():
        pytest.skip("healthy.jpg not found on machine")

    cfg = load_config(PROJECT_ROOT / "configs" / "experiments" / "tomato_mobilenetv3_robust_v2.yaml")
    service = DiagnosisService(cfg)

    result = service.diagnose(healthy_jpg.read_bytes())
    assert result["diagnosis"] == "healthy", f"Expected 'healthy', got '{result['diagnosis']}'"
    assert result["confidence"] > 0.90, f"Expected high confidence, got {result['confidence']}"
