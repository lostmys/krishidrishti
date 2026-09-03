from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image
import pytest
import torch

from krishidrishti_ai.services.inference import DiagnosisService
from krishidrishti_ai.services.quality import QualityResult


def test_prediction_parsing() -> None:
    service = DiagnosisService.__new__(DiagnosisService)

    # Tomato-style label: separator present
    service.config = {"data": {"class_separator": "___"}}
    pred1 = service._prediction("Tomato___Early_blight", 0.8523456)
    assert pred1["crop"] == "Tomato"
    assert pred1["diagnosis"] == "Early_blight"
    assert pred1["confidence"] == 0.852346

    # Soyabean-style label: no separator -> fallback to crop_name
    service.config = {"data": {"class_separator": "___", "crop_name": "Soyabean"}}
    pred2 = service._prediction("Rust", 0.9123456)
    assert pred2["crop"] == "Soyabean"
    assert pred2["diagnosis"] == "Rust"
    assert pred2["confidence"] == 0.912346

    # Plain label with no crop_name configured -> "Unknown"
    service.config = {"data": {"class_separator": "___"}}
    pred3 = service._prediction("Healthy", 0.999)
    assert pred3["crop"] == "Unknown"
    assert pred3["diagnosis"] == "Healthy"
    assert pred3["confidence"] == 0.999


def test_temperature_scaling_and_conservative_decision_policy() -> None:
    service = DiagnosisService.__new__(DiagnosisService)
    service.config = {
        "data": {"class_separator": "___", "crop_name": "Tomato", "image_size": 224},
        "training": {"device": "cpu"},
        "quality": {
            "min_width": 64,
            "min_height": 64,
            "min_blur_score": 10.0,
            "min_brightness_score": 10.0,
            "max_brightness_score": 250.0,
        },
        "inference": {
            "top_k": 3,
            "temperature": 1.2,
            "confident_threshold": 0.75,
            "review_threshold": 0.45,
            "ambiguity_margin_threshold": 0.30,
            "max_entropy_threshold": 0.55,
        },
    }
    service.classes = ["Tomato___Early_blight", "Tomato___Septoria_leaf_spot", "Tomato___healthy"]
    service.device = "cpu"
    service.temperature = 1.2
    service.confident_threshold = 0.75
    service.review_threshold = 0.45
    service.ambiguity_margin_threshold = 0.30
    service.max_entropy_threshold = 0.55

    service.knowledge_service = MagicMock()
    service.knowledge_service.get_display_name.side_effect = lambda c, d, language="en": d
    service.knowledge_service.generate_analysis.return_value = {
        "summary": "Summary",
        "confidence_explanation": "Explanation",
        "uncertainty_note": "Note",
        "visual_limitations_note": "Limits",
    }
    service.knowledge_service.get_advice.return_value = {
        "display_name": "Early Blight",
        "description": "Desc",
        "typical_symptoms": [],
        "immediate_actions": [],
        "prevention": [],
        "severity_guidance": "Medium",
        "source": "ICAR",
    }

    img = Image.new("RGB", (128, 128), color="green")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    service.transform = MagicMock(return_value=torch.zeros((3, 224, 224)))

    # Case 1: High raw score but narrow margin -> REVIEW_RECOMMENDED (no automatic AI_CONFIDENT)
    mock_model1 = MagicMock()
    mock_model1.return_value = torch.tensor([[4.0, 3.8, 1.0]])  # Top 2 are close
    service.model = mock_model1

    with patch("krishidrishti_ai.services.inference.assess_image_quality", return_value=QualityResult(True, 150.0, 100.0)):
        res1 = service.diagnose(img_bytes, language="en")
    assert res1["status"] == "REVIEW_RECOMMENDED"
    assert res1["is_ambiguous"] is True
    assert res1["margin"] < 0.30

    # Case 2: Decisive prediction -> AI_CONFIDENT
    mock_model2 = MagicMock()
    mock_model2.return_value = torch.tensor([[8.0, 1.0, 0.5]])  # Dominant class
    service.model = mock_model2

    with patch("krishidrishti_ai.services.inference.assess_image_quality", return_value=QualityResult(True, 150.0, 100.0)):
        res2 = service.diagnose(img_bytes, language="en")
    assert res2["status"] == "AI_CONFIDENT"
    assert res2["is_ambiguous"] is False
    assert res2["confidence"] >= 0.75
    assert res2["margin"] >= 0.30

    # Case 3: Flat logits -> LOW_CONFIDENCE
    mock_model3 = MagicMock()
    mock_model3.return_value = torch.tensor([[1.0, 1.0, 0.9]])  # High uncertainty
    service.model = mock_model3

    with patch("krishidrishti_ai.services.inference.assess_image_quality", return_value=QualityResult(True, 150.0, 100.0)):
        res3 = service.diagnose(img_bytes, language="en")
    assert res3["status"] == "LOW_CONFIDENCE"
    assert res3["is_ambiguous"] is True


# ---------------------------------------------------------------------------
# CropRegistry tests
# ---------------------------------------------------------------------------


def test_crop_registry_unknown_slug_raises_key_error() -> None:
    from krishidrishti_ai.services.registry import CropRegistry

    registry = CropRegistry({"tomato": Path("/fake/tomato.yaml")})
    with pytest.raises(KeyError):
        registry.get("wheat")


def test_crop_registry_available_crops_is_sorted() -> None:
    from krishidrishti_ai.services.registry import CropRegistry

    registry = CropRegistry(
        {"soyabean": Path("/a.yaml"), "tomato": Path("/b.yaml"), "cotton": Path("/c.yaml")}
    )
    assert registry.available_crops == ["cotton", "soyabean", "tomato"]


def test_crop_registry_slug_normalised_to_lowercase() -> None:
    from krishidrishti_ai.services.registry import CropRegistry

    registry = CropRegistry({"TOMATO": Path("/fake/tomato.yaml")})
    assert "tomato" in registry.available_crops
    assert "TOMATO" not in registry.available_crops

    with pytest.raises(KeyError):
        registry.get("Wheat")


def test_crop_registry_resolves_known_crop_via_mock(tmp_path: Path) -> None:
    from krishidrishti_ai.services.registry import CropRegistry

    fake_config = tmp_path / "tomato.yaml"
    fake_config.touch()

    stub_service = MagicMock(spec=DiagnosisService)
    registry = CropRegistry({"tomato": fake_config})

    with (
        patch("krishidrishti_ai.services.registry.load_config", return_value={}),
        patch("krishidrishti_ai.services.registry.DiagnosisService", return_value=stub_service),
    ):
        service = registry.get("tomato")

    assert service is stub_service


def test_crop_registry_caches_service_instance(tmp_path: Path) -> None:
    from krishidrishti_ai.services.registry import CropRegistry

    fake_config = tmp_path / "soyabean.yaml"
    fake_config.touch()

    stub_service = MagicMock(spec=DiagnosisService)
    registry = CropRegistry({"soyabean": fake_config})

    with (
        patch("krishidrishti_ai.services.registry.load_config", return_value={}),
        patch("krishidrishti_ai.services.registry.DiagnosisService", return_value=stub_service),
    ):
        first = registry.get("soyabean")
        second = registry.get("soyabean")

    assert first is second is stub_service
