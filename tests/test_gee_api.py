from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from krishidrishti_ai.api import app
from krishidrishti_ai.gee_api import EarthEngineSettings, _normalize_payload, compute_local_abnormality


def test_compute_local_abnormality_creates_clustered_score() -> None:
    grid = [
        [0.1, 0.1, 0.1, 0.1],
        [0.1, 4.2, 4.4, 0.1],
        [0.1, 4.1, 4.3, 0.1],
        [0.1, 0.1, 0.1, 0.1],
    ]
    result = compute_local_abnormality(
        grid,
        percentile_threshold=90.0,
        zscore_threshold=2.0,
        min_cluster_size=2,
        bounds=(72.0, 72.1, 18.0, 18.1),
    )
    assert result["label"] == "abnormal"
    assert result["connected_clusters"] >= 1
    assert result["farm_local_abnormal_score"] > 0.0
    assert result["geojson"]["type"] == "FeatureCollection"
    anomaly = next(feature for feature in result["geojson"]["features"] if feature["properties"]["label"] == "anomaly")
    assert anomaly["properties"]["latitude"] == 18.05
    assert 72.0 < anomaly["properties"]["longitude"] < 72.1
    assert anomaly["properties"]["radius_meters"] > 0
    assert anomaly["geometry"]["type"] == "MultiPolygon"


def test_compute_local_abnormality_labels_unreachable_pixels() -> None:
    result = compute_local_abnormality(
        [[None, None, 0.1], [None, 0.1, 0.1]],
        min_cluster_size=2,
        bounds=(72.0, 72.1, 18.0, 18.1),
    )
    unreachable = [feature for feature in result["geojson"]["features"] if feature["properties"]["label"] == "unreachable"]
    assert result["unreachable_zones"] == 1
    assert unreachable[0]["properties"]["radius_meters"] > 0


def test_normalize_payload_validates_india_and_area() -> None:
    with pytest.raises(ValueError, match="India"):
        _normalize_payload({"latitude": 12.0, "longitude": 65.0, "radius_meters": 5000})
    with pytest.raises(ValueError, match="4 acres"):
        _normalize_payload({"latitude": 19.076, "longitude": 72.877, "radius_meters": 10})


def test_earth_engine_settings_require_service_account_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ["GEE_SERVICE_ACCOUNT_EMAIL", "GEE_PRIVATE_KEY", "GEE_PROJECT_ID"]:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="GEE_SERVICE_ACCOUNT_EMAIL"):
        EarthEngineSettings.from_env()


def test_analyze_farm_endpoint_returns_result(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeService:
        def analyze(self, payload):
            return {
                "label": "abnormal",
                "farm_label": "abnormal",
                "farm_local_abnormal_score": 0.74,
                "summary": {"area_m2": 245000.0, "connected_clusters": 2, "label": "abnormal", "farm_label": "abnormal"},
                "geojson": {"type": "FeatureCollection", "features": []},
                "data_availability": {"sentinel_2_sr": True, "sentinel_1_grd": True},
            }

    monkeypatch.setattr("krishidrishti_ai.api.get_gee_service", lambda: FakeService())
    client = TestClient(app)
    response = client.post(
        "/analyze-farm",
        json={
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[72.0, 18.0], [72.1, 18.0], [72.1, 18.1], [72.0, 18.1], [72.0, 18.0]]],
            }
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] == "abnormal"
    assert payload["farm_local_abnormal_score"] == 0.74
    assert payload["summary"]["area_m2"] == 245000.0
    assert payload["summary"]["farm_label"] == "abnormal"
    assert payload["data_availability"]["sentinel_2_sr"] is True
