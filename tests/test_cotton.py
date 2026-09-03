"""Unit tests for Cotton support across config, registry, API, and multilingual knowledge."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import krishidrishti_ai.api as _api_module
from krishidrishti_ai.api import app
from krishidrishti_ai.config import load_config
from krishidrishti_ai.services.knowledge import DiseaseKnowledgeService

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COTTON_CLASSES = [
    "Bacterial Blight",
    "Curl Virus",
    "Healthy Leaf",
    "Herbicide Growth Damage",
    "Leaf Hopper Jassids",
    "Leaf Redding",
    "Leaf Variegation",
]


def _jpeg_bytes(width: int = 256, height: int = 256) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (width, height), color=(100, 150, 80)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def test_cotton_config_exists_and_valid() -> None:
    config_path = PROJECT_ROOT / "configs" / "experiments" / "cotton_mobilenetv3.yaml"
    assert config_path.is_file(), "cotton_mobilenetv3.yaml does not exist"
    config = load_config(config_path)
    assert config["data"]["crop_name"] == "Cotton"
    assert config["data"]["class_prefix"] is None
    assert "cotton" in config["paths"]["processed_data_dir"]
    assert "cotton" in config["paths"]["checkpoint_path"]
    assert config["model"]["name"] == "mobilenet_v3_small"


def test_cotton_in_default_yaml_crop_registry() -> None:
    default_config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    assert "crops" in default_config
    assert "cotton" in default_config["crops"]


def test_crops_endpoint_includes_cotton(client: TestClient) -> None:
    resp = client.get("/crops")
    assert resp.status_code == 200
    crops = resp.json()["crops"]
    assert "cotton" in crops
    assert "tomato" in crops
    assert "soyabean" in crops


def test_diagnose_cotton_mocked(client: TestClient) -> None:
    expected = {
        "crop": "Cotton",
        "diagnosis": "Bacterial Blight",
        "confidence": 0.95,
        "is_ambiguous": False,
        "margin": 0.90,
        "status": "AI_CONFIDENT",
        "image_quality": {"passed": True, "blur_score": 280.0, "brightness_score": 115.0},
        "top_predictions": [
            {"crop": "Cotton", "diagnosis": "Bacterial Blight", "confidence": 0.95}
        ],
        "advice": {
            "display_name": "Cotton Bacterial Blight",
            "description": "Bacterial infection",
            "symptoms": ["Angular spots"],
            "immediate_actions": ["Avoid working in wet foliage"],
            "prevention": ["Acid-delinted seeds"],
            "severity_guidance": "Moderate to High",
            "source": "ICAR-CICR",
        },
    }
    registry_mock = MagicMock()
    registry_mock.available_crops = ["cotton", "soyabean", "tomato"]
    stub_service = MagicMock()
    stub_service.diagnose.return_value = expected
    registry_mock.get.return_value = stub_service

    with patch.object(_api_module, "_get_registry", return_value=registry_mock):
        resp = client.post(
            "/diagnose/cotton?language=en",
            files={"image": ("cotton_leaf.jpg", _jpeg_bytes(), "image/jpeg")},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["crop"] == "Cotton"
    assert data["diagnosis"] == "Bacterial Blight"
    assert data["status"] == "AI_CONFIDENT"


def test_all_7_cotton_classes_have_complete_knowledge() -> None:
    service = DiseaseKnowledgeService()
    for class_name in COTTON_CLASSES:
        for lang in ["en", "hi", "mr"]:
            advice = service.get_advice("Cotton", class_name, "AI_CONFIDENT", language=lang)
            assert advice["display_name"], f"Missing display_name for Cotton {class_name} in {lang}"
            assert advice["description"], f"Missing description for Cotton {class_name} in {lang}"
            assert len(advice["typical_symptoms"]) > 0, f"Missing symptoms for Cotton {class_name} in {lang}"
            assert len(advice["immediate_actions"]) > 0, f"Missing actions for Cotton {class_name} in {lang}"
            assert len(advice["prevention"]) > 0, f"Missing prevention for Cotton {class_name} in {lang}"
            assert advice["severity_guidance"], f"Missing severity for Cotton {class_name} in {lang}"
            assert advice["source"], f"Missing source for Cotton {class_name} in {lang}"


def test_cotton_healthy_crop_has_no_pesticide_advice() -> None:
    service = DiseaseKnowledgeService()
    for lang in ["en", "hi", "mr"]:
        advice = service.get_advice("Cotton", "Healthy Leaf", "AI_CONFIDENT", language=lang)
        for action in advice["immediate_actions"]:
            assert "pesticide" not in action.lower()
            assert "कीटनाशक" not in action
            assert "कीटकनाशक" not in action


def test_cotton_multilingual_analysis_generation() -> None:
    service = DiseaseKnowledgeService()
    top_preds = [
        {"crop": "Cotton", "diagnosis": "Curl Virus", "confidence": 0.65},
        {"crop": "Cotton", "diagnosis": "Leaf Hopper Jassids", "confidence": 0.25},
    ]

    analysis_mr = service.generate_analysis(
        "Cotton", "Curl Virus", 0.65, "REVIEW_RECOMMENDED", "mr", is_ambiguous=True, top_predictions=top_preds
    )
    assert "कापूस" in analysis_mr["summary"]
    assert "65.0%" in analysis_mr["summary"]
    assert "चुरडा-मुरडा" in analysis_mr["summary"] or "पर्णगुच्छ" in analysis_mr["summary"]
    assert "तुडतुडे" in analysis_mr["uncertainty_note"] or "जॅसिड" in analysis_mr["uncertainty_note"]

    analysis_hi = service.generate_analysis(
        "Cotton", "Curl Virus", 0.65, "REVIEW_RECOMMENDED", "hi", is_ambiguous=True, top_predictions=top_preds
    )
    assert "कपास" in analysis_hi["summary"]
    assert "65.0%" in analysis_hi["summary"]
    assert "पत्ती मरोड़" in analysis_hi["summary"]
