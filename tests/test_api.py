"""API-layer tests using FastAPI TestClient.

All tests mock _get_registry so no real checkpoints are needed.
The DiagnosisService itself is already covered by test_inference_logic.py.
"""
from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import krishidrishti_ai.api as _api_module
from krishidrishti_ai.api import app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jpeg_bytes(width: int = 256, height: int = 256) -> bytes:
    """Return bytes of a small, valid JPEG image."""
    buf = BytesIO()
    Image.new("RGB", (width, height), color=(100, 150, 80)).save(buf, format="JPEG")
    return buf.getvalue()


def _make_registry(
    available: list[str],
    *,
    diagnose_return: dict | None = None,
    unknown_crop: bool = False,
    checkpoint_ready: bool = True,
) -> MagicMock:
    """Build a MagicMock CropRegistry."""
    registry = MagicMock()
    registry.available_crops = sorted(available)
    registry.checkpoint_ready.return_value = checkpoint_ready

    if unknown_crop:
        registry.get.side_effect = KeyError("wheat")
    else:
        stub_service = MagicMock()
        stub_service.diagnose.return_value = diagnose_return or {
            "crop": "Tomato",
            "diagnosis": "Early_blight",
            "confidence": 0.91,
            "status": "AI_CONFIDENT",
            "image_quality": {"passed": True, "blur_score": 250.0, "brightness_score": 110.0},
            "top_predictions": [
                {"crop": "Tomato", "diagnosis": "Early_blight", "confidence": 0.91}
            ],
            "advice": {
                "display_name": "Early Blight",
                "description": "Fungal infection",
                "symptoms": ["Dark spots"],
                "immediate_actions": ["Prune leaves"],
                "prevention": ["Crop rotation"],
                "severity_guidance": "Moderate",
                "source": "Test Source",
            },
        }
        registry.get.return_value = stub_service

    return registry


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# GET / (Index UI)
# ---------------------------------------------------------------------------


def test_index_ui_returns_200(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "KrishiDrishti" in resp.text


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_all_crops_ready(client: TestClient) -> None:
    mock = _make_registry(["soyabean", "tomato"], checkpoint_ready=True)
    with patch.object(_api_module, "_get_registry", return_value=mock):
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "READY"
    assert set(body["crops_ready"]) == {"tomato", "soyabean"}
    assert body["crops_not_ready"] == []


def test_health_partial_ready(client: TestClient) -> None:
    """PARTIAL when some checkpoints are present but not all."""
    registry = MagicMock()
    registry.available_crops = ["soyabean", "tomato"]
    registry.checkpoint_ready.side_effect = lambda slug: slug == "tomato"

    with patch.object(_api_module, "_get_registry", return_value=registry):
        resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "PARTIAL"
    assert body["crops_ready"] == ["tomato"]
    assert body["crops_not_ready"] == ["soyabean"]


# ---------------------------------------------------------------------------
# /crops
# ---------------------------------------------------------------------------


def test_list_crops_returns_sorted_slugs(client: TestClient) -> None:
    mock = _make_registry(["tomato", "soyabean"])
    with patch.object(_api_module, "_get_registry", return_value=mock):
        resp = client.get("/crops")
    assert resp.status_code == 200
    assert resp.json() == {"crops": ["soyabean", "tomato"]}


# ---------------------------------------------------------------------------
# POST /diagnose/{crop}
# ---------------------------------------------------------------------------


def test_diagnose_unknown_crop_returns_404(client: TestClient) -> None:
    mock = _make_registry(["tomato", "soyabean"], unknown_crop=True)
    with patch.object(_api_module, "_get_registry", return_value=mock):
        resp = client.post(
            "/diagnose/wheat",
            files={"image": ("leaf.jpg", _jpeg_bytes(), "image/jpeg")},
        )
    assert resp.status_code == 404
    assert "wheat" in resp.json()["detail"].lower()


def test_diagnose_empty_payload_returns_400(client: TestClient) -> None:
    mock = _make_registry(["tomato", "soyabean"])
    with patch.object(_api_module, "_get_registry", return_value=mock):
        resp = client.post(
            "/diagnose/tomato",
            files={"image": ("leaf.jpg", b"", "image/jpeg")},
        )
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_diagnose_large_payload_returns_400(client: TestClient) -> None:
    mock = _make_registry(["tomato", "soyabean"])
    large_bytes = b"0" * (11 * 1024 * 1024)  # 11MB > 10MB limit
    with patch.object(_api_module, "_get_registry", return_value=mock):
        resp = client.post(
            "/diagnose/tomato",
            files={"image": ("large.jpg", large_bytes, "image/jpeg")},
        )
    assert resp.status_code == 400
    assert "exceeds maximum limit" in resp.json()["detail"].lower()


def test_diagnose_non_image_content_type_returns_415(client: TestClient) -> None:
    mock = _make_registry(["tomato", "soyabean"])
    with patch.object(_api_module, "_get_registry", return_value=mock):
        resp = client.post(
            "/diagnose/tomato",
            files={"image": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
        )
    assert resp.status_code == 415


def test_diagnose_returns_structured_result_with_language(client: TestClient) -> None:
    expected = {
        "crop": "Soyabean",
        "diagnosis": "Rust",
        "confidence": 0.93,
        "is_ambiguous": False,
        "margin": 0.90,
        "alternative_prediction": None,
        "status": "AI_CONFIDENT",
        "image_quality": {"passed": True, "blur_score": 300.0, "brightness_score": 120.0},
        "top_predictions": [{"crop": "Soyabean", "diagnosis": "Rust", "display_name": "सोयाबीन तांबेरा (Soybean Rust)", "confidence": 0.93}],
        "analysis": {
            "summary": "मॉडेलने ९३.०% खात्रीसह या सोयाबीनच्या फोटोचे वर्गीकरण सोयाबीन तांबेरा (Soybean Rust) असे केले आहे.",
            "confidence_explanation": "प्रशिक्षित रोग लक्षणांवर आधारित उच्च मॉडेल खात्री.",
            "uncertainty_note": "",
            "visual_limitations_note": "सूचना: ही एआय प्रणाली केवळ प्रशिक्षित पानांच्या प्रतिमांवर आधारित स्वयंचलित प्राथमिक तपासणी करते.",
        },
        "advice": {
            "display_name": "सोयाबीन तांबेरा (Soybean Rust)",
            "description": "फॅकोप्सोरा बुरशीमुळे होतो.",
            "symptoms": ["पानांच्या मागे बारीक तांबूस उभरे फोड."],
            "immediate_actions": ["झाडाच्या खालच्या पानांची बारकाईने पाहणी करा."],
            "prevention": ["लवकर येणाऱ्या वाणांची पेरणी करा."],
            "severity_guidance": "अत्यंत तीव्र.",
            "source": "ICAR - IISR",
        },
        "reference_knowledge": {
            "display_name": "सोयाबीन तांबेरा (Soybean Rust)",
            "description": "फॅकोप्सोरा बुरशीमुळे होतो.",
            "symptoms": ["पानांच्या मागे बारीक तांबूस उभरे फोड."],
            "immediate_actions": ["झाडाच्या खालच्या पानांची बारकाईने पाहणी करा."],
            "prevention": ["लवकर येणाऱ्या वाणांची पेरणी करा."],
            "severity_guidance": "अत्यंत तीव्र.",
            "source": "ICAR - IISR",
        },
    }
    mock = _make_registry(["soyabean", "tomato"], diagnose_return=expected)

    with patch.object(_api_module, "_get_registry", return_value=mock):
        resp = client.post(
            "/diagnose/soyabean?language=mr",
            files={"image": ("leaf.jpg", _jpeg_bytes(), "image/jpeg")},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["crop"] == "Soyabean"
    assert body["diagnosis"] == "Rust"
    assert "analysis" in body
    assert "advice" in body
    assert "reference_knowledge" in body
    assert body["advice"]["display_name"] == "सोयाबीन तांबेरा (Soybean Rust)"



def test_diagnose_slug_is_case_insensitive(client: TestClient) -> None:
    mock = _make_registry(["tomato", "soyabean"])
    with patch.object(_api_module, "_get_registry", return_value=mock):
        resp = client.post(
            "/diagnose/TOMATO",
            files={"image": ("leaf.jpg", _jpeg_bytes(), "image/jpeg")},
        )
    assert resp.status_code == 200


def test_diagnose_missing_checkpoint_returns_503(client: TestClient) -> None:
    registry = MagicMock()
    registry.available_crops = ["tomato"]
    registry.get.side_effect = FileNotFoundError("checkpoint missing")

    with patch.object(_api_module, "_get_registry", return_value=registry):
        resp = client.post(
            "/diagnose/tomato",
            files={"image": ("leaf.jpg", _jpeg_bytes(), "image/jpeg")},
        )
    assert resp.status_code == 503
