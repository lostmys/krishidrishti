import os
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

from krishidrishti_ai.api import app
from krishidrishti_ai.communication.whatsapp_client import (
    WhatsAppClient,
    MetaWhatsAppClient,
    MockWhatsAppClient,
    get_whatsapp_client,
)
from krishidrishti_ai.communication.voice_service import (
    FarmerVoiceService,
    RealSTTProvider,
    MockSTTProvider,
)
from krishidrishti_ai.services.satellite import SatelliteService


@pytest.fixture
def client():
    return TestClient(app)


def test_whatsapp_mock_provider_fallback():
    with patch.dict(os.environ, {}, clear=True):
        client = WhatsAppClient()
        assert not client.is_live
        assert client.status_label == "🟡 DEMO / STAGED"
        msg = client.send_message("+919822012345", "Test message")
        assert msg.status == "DELIVERED_STAGED"
        assert msg.is_demo is True
        chk = client.check_connection()
        assert chk["configured"] is False
        assert chk["status"] == "DEMO / STAGED"


def test_whatsapp_meta_provider_live_instantiation():
    with patch.dict(os.environ, {
        "WHATSAPP_ACCESS_TOKEN": "EAABtesttoken123",
        "WHATSAPP_PHONE_NUMBER_ID": "1234567890",
    }):
        client = WhatsAppClient()
        assert client.is_live is True
        assert client.status_label == "🟢 LIVE"
        assert isinstance(client._provider, MetaWhatsAppClient)


def test_voice_mock_provider_fallback():
    with patch.dict(os.environ, {}, clear=True):
        svc = FarmerVoiceService()
        assert not svc.is_live
        assert svc.is_demo is True
        assert svc.mode == "🟡 DEMO / STAGED"
        res = svc.process_voice_note(b"dummy_bytes", crop="tomato", language="mr")
        assert res["status"] == "PROCESSED"
        assert "पानांवर डाग" in res["transcript_mr"]
        assert "Spots are appearing" in res["transcript_en"]
        chk = svc.get_status()
        assert chk["configured"] is False


def test_voice_real_provider_live_instantiation():
    with patch.dict(os.environ, {
        "STT_API_KEY": "gsk_test_api_key_12345",
        "STT_PROVIDER": "groq",
    }):
        svc = FarmerVoiceService()
        assert svc.is_live is True
        assert svc.is_demo is False
        assert svc.mode == "🟢 LIVE"
        assert isinstance(svc._provider, RealSTTProvider)
        chk = svc.get_status()
        assert chk["configured"] is True
        assert chk["status"] == "LIVE"


def test_satellite_status_reporting():
    with patch.dict(os.environ, {}, clear=True):
        status = SatelliteService.get_status()
        assert status["status"] == "DEMO / PRECOMPUTED"
        assert status["badge"] == "🟡 DEMO / PRECOMPUTED"
        assert status["configured"] is False
        assert status["is_live"] is False


def test_system_status_api_endpoint(client):
    resp = client.get("/system/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "OPERATIONAL"
    comps = data["components"]
    required_keys = [
        "image_ai",
        "cases_api",
        "officer_dashboard",
        "risk_fusion",
        "whatsapp_delivery",
        "voice_processing",
        "satellite_sensor",
    ]
    for k in required_keys:
        assert k in comps, f"Component {k} missing in /system/status"
        assert "status" in comps[k]
        assert "badge" in comps[k]


def test_satellite_status_api_endpoint(client):
    resp = client.get("/satellite/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "badge" in data


def test_voice_status_api_endpoint(client):
    resp = client.get("/whatsapp/voice/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "languages" in data
    assert "mr" in data["languages"]
