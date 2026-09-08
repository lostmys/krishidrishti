import math
import sys
from datetime import date
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from krishidrishti_ai.api import app
from krishidrishti_ai.gee_api import (
    DEFAULT_POINT_RADIUS_METERS,
    MIN_FARM_AREA_ACRES,
    MIN_FARM_AREA_M2,
    EarthEngineAnalysisService,
    EarthEngineSettings,
    FarmAnalysisService,
    _grid_from_anomaly_image,
    _normalize_payload,
    compute_local_abnormality,
)
from krishidrishti_ai.services.fusion import RiskFusionService
from krishidrishti_ai.services.satellite import SatelliteService


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
    with pytest.raises(ValueError, match="0.5 acre"):
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
                "region_image": "data:image/png;base64,AAAA",
                "region_image_mime_type": "image/png",
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
    assert payload["region_image_mime_type"] == "image/png"
    assert payload["region_image"].startswith("data:image/png;base64,")


def test_grid_from_anomaly_image_batches_reduce_regions() -> None:
    mock_ee = mock.MagicMock()
    with mock.patch.dict(sys.modules, {"ee": mock_ee}):
        region = mock.MagicMock()
        region.bounds.return_value.getInfo.return_value = {
            "coordinates": [[[72.0, 18.0], [72.1, 18.0], [72.1, 18.1], [72.0, 18.1], [72.0, 18.0]]]
        }
        anomaly_image = mock.MagicMock()
        features = []
        for r in range(8):
            for c in range(8):
                val = None if (r, c) == (0, 0) else 0.35 + (r + c) * 0.01
                features.append({"properties": {"row": r, "col": c, "ANOMALY": val}})
        anomaly_image.reduceRegions.return_value.getInfo.return_value = {"features": features}

        grid = _grid_from_anomaly_image(anomaly_image, region, rows=8, cols=8)

        # Verified: Exactly 1 single batch call to reduceRegions
        anomaly_image.reduceRegions.assert_called_once()
        anomaly_image.reduceRegion.assert_not_called()
        assert len(grid) == 8
        assert all(len(row) == 8 for row in grid)
        assert grid[0][0] is None
        assert grid[1][1] == pytest.approx(0.37)
        # Verify row/col metadata ordering survived
        assert grid[7][7] == pytest.approx(0.49)


def test_mask_s2_sr_masks_required_scl_classes() -> None:
    mock_ee = mock.MagicMock()
    with mock.patch.dict(sys.modules, {"ee": mock_ee}):
        service = EarthEngineAnalysisService.__new__(EarthEngineAnalysisService)
        image = mock.MagicMock()
        scl = mock.MagicMock()
        image.select.return_value = scl
        scl_cond = mock.MagicMock()
        scl.neq.return_value = scl_cond
        scl_cond.And.return_value = scl_cond

        service._mask_s2_sr(image)

        image.select.assert_called_once_with("SCL")
        # Check neq was called with classes 0, 1, 3, 8, 9, 10, 11
        neq_calls = [c.args[0] for c in scl.neq.call_args_list]
        for required_class in [0, 1, 3, 8, 9, 10, 11]:
            assert required_class in neq_calls
        image.updateMask.assert_called_once()


def test_area_validation_0_5_acre_threshold() -> None:
    # 0.5 acre = 2023.43 m2
    exact_radius = math.sqrt(MIN_FARM_AREA_M2 / math.pi)
    accepted = _normalize_payload({"latitude": 19.0, "longitude": 73.0, "radius_meters": exact_radius})
    assert accepted["area_m2"] == pytest.approx(MIN_FARM_AREA_M2, abs=0.01)

    above_radius = 30.0  # ~2827.4 m2
    accepted_above = _normalize_payload({"latitude": 19.0, "longitude": 73.0, "radius_meters": above_radius})
    assert accepted_above["area_m2"] > MIN_FARM_AREA_M2

    below_radius = 20.0  # ~1256.6 m2 < 2023.43 m2
    with pytest.raises(ValueError, match="0.5 acre"):
        _normalize_payload({"latitude": 19.0, "longitude": 73.0, "radius_meters": below_radius})


def test_point_radius_configuration_and_geometry() -> None:
    assert DEFAULT_POINT_RADIUS_METERS == 100.0
    assert SatelliteService.DEFAULT_RADIUS_METERS == 100.0

    # Test payload without radius_meters defaults to DEFAULT_POINT_RADIUS_METERS
    payload = _normalize_payload({"latitude": 19.0, "longitude": 73.0})
    assert payload["radius_meters"] == 100.0
    assert payload["area_m2"] == pytest.approx(math.pi * 100.0**2)

    # Test _coordinates_to_geojson produces closed proxy polygon
    geojson = SatelliteService._coordinates_to_geojson(19.0, 73.0, radius_meters=100.0)
    assert geojson["type"] == "Feature"
    assert geojson["geometry"]["type"] == "Polygon"
    coords = geojson["geometry"]["coordinates"][0]
    assert len(coords) == 33  # 32 vertices + closed loop
    assert coords[0] == coords[-1]
    assert geojson["properties"]["proxy_radius_meters"] == 100.0
    assert geojson["properties"]["is_approximate_proxy"] is True


def test_sentinel_2_unavailable_falls_back_to_sentinel_1() -> None:
    mock_ee = mock.MagicMock()
    with mock.patch.dict(sys.modules, {"ee": mock_ee}):
        service = EarthEngineAnalysisService.__new__(EarthEngineAnalysisService)
        service.initialize = mock.MagicMock()
        service._to_ee_geometry = mock.MagicMock()

        # Mock S2 has 0 scenes, S1 has 2 scenes
        mock_s2_col = mock.MagicMock()
        mock_s2_col.filterBounds.return_value = mock_s2_col
        mock_s2_col.filterDate.return_value = mock_s2_col
        mock_s2_col.size.return_value.getInfo.return_value = 0

        mock_s1_col = mock.MagicMock()
        mock_s1_col.filterBounds.return_value = mock_s1_col
        mock_s1_col.filterDate.return_value = mock_s1_col
        mock_s1_col.filter.return_value = mock_s1_col
        mock_s1_col.map.return_value = mock_s1_col
        mock_s1_col.size.return_value.getInfo.return_value = 2
        mock_s1_mean = mock.MagicMock()
        mock_s1_col.median.return_value.clip.return_value = mock_s1_mean
        mock_vv = mock.MagicMock()
        mock_vh = mock.MagicMock()
        mock_s1_mean.select.side_effect = lambda b: mock_vv if b == "VV" else mock_vh
        mock_vv.reduceRegion.return_value.getInfo.return_value = -12.5
        mock_vh.reduceRegion.return_value.getInfo.return_value = -18.2

        def mock_image_collection(col_id):
            if "S2" in col_id:
                return mock_s2_col
            return mock_s1_col

        mock_ee.ImageCollection.side_effect = mock_image_collection
        mock_ee.Image.constant.return_value.pow.return_value.divide.return_value.rename.return_value = mock.MagicMock()
        mock_ee.Image.cat.return_value.reduceRegion.return_value = mock.MagicMock()

        farm = {"type": "circle", "latitude": 19.0, "longitude": 73.0, "radius_meters": 100.0}
        with mock.patch("krishidrishti_ai.gee_api._grid_from_anomaly_image", return_value=[[0.2]*8 for _ in range(8)]):
            res = service.fetch_farm_metrics(farm, date(2026, 9, 1), date(2026, 9, 7))

        assert res["modality"] == "SAR_ONLY"
        assert res["data_availability"]["sentinel_2_sr"] is False
        assert res["data_availability"]["sentinel_1_grd"] is True
        assert res["sentinel2"]["ndvi_mean"] is None
        assert res["sentinel2"]["ndmi_mean"] is None
        assert res["sentinel2"]["ndre_mean"] is None
        assert res["sentinel1"]["vv_mean_db"] == -12.5


def test_both_sensors_unavailable_fails_gracefully() -> None:
    mock_ee = mock.MagicMock()
    with mock.patch.dict(sys.modules, {"ee": mock_ee}):
        service = EarthEngineAnalysisService.__new__(EarthEngineAnalysisService)
        service.initialize = mock.MagicMock()
        service._to_ee_geometry = mock.MagicMock()

        mock_col = mock.MagicMock()
        mock_col.size.return_value.getInfo.return_value = 0
        mock_col.filterBounds.return_value = mock_col
        mock_col.filterDate.return_value = mock_col
        mock_col.filter.return_value = mock_col
        mock_ee.ImageCollection.return_value = mock_col

        farm = {"type": "circle", "latitude": 19.0, "longitude": 73.0, "radius_meters": 100.0}
        with pytest.raises(ValueError, match="No Sentinel-2 or Sentinel-1 imagery is available"):
            service.fetch_farm_metrics(farm, date(2026, 9, 1), date(2026, 9, 7))

    # SatelliteService wraps this as UNAVAILABLE
    sat_service = SatelliteService()
    with mock.patch.object(SatelliteService, "is_configured", return_value=True):
        with mock.patch("krishidrishti_ai.services.satellite.get_gee_service") as mock_get:
            mock_analyzer = mock.MagicMock()
            mock_analyzer.analyze.side_effect = ValueError("No Sentinel-2 or Sentinel-1 imagery is available for this farm and date window.")
            mock_get.return_value = mock_analyzer
            result = sat_service.analyze_farm_safely({"latitude": 19.0, "longitude": 73.0, "radius_meters": 100.0})
            assert result["status"] == "UNAVAILABLE"
            assert result["available"] is False


def test_sentinel_1_temporal_change_handles_missing_previous_observation() -> None:
    mock_ee = mock.MagicMock()
    with mock.patch.dict(sys.modules, {"ee": mock_ee}):
        service = EarthEngineAnalysisService.__new__(EarthEngineAnalysisService)
        service.initialize = mock.MagicMock()
        service._to_ee_geometry = mock.MagicMock()

        mock_s2 = mock.MagicMock()
        mock_s2.filterBounds.return_value = mock_s2
        mock_s2.filterDate.return_value = mock_s2
        mock_s2.size.return_value.getInfo.return_value = 0

        mock_s1 = mock.MagicMock()
        mock_s1.size.return_value.getInfo.return_value = 1
        mock_s1.filterBounds.return_value = mock_s1
        mock_s1.filter.return_value = mock_s1
        mock_s1.map.return_value = mock_s1
        mock_s1_mean = mock.MagicMock()
        mock_s1.median.return_value.clip.return_value = mock_s1_mean
        mock_s1_mean.select.return_value.reduceRegion.return_value.getInfo.return_value = -15.0

        mock_recent = mock.MagicMock()
        mock_recent.size.return_value.getInfo.return_value = 1
        mock_latest = mock.MagicMock()
        mock_latest.get.return_value.getInfo.return_value = "DESCENDING"
        mock_recent.sort.return_value.first.return_value = mock_latest
        mock_recent_matched = mock.MagicMock()
        mock_recent_matched.size.return_value.getInfo.return_value = 1
        mock_recent.filter.return_value = mock_recent_matched

        mock_prev = mock.MagicMock()
        mock_prev.size.return_value.getInfo.return_value = 0
        mock_prev_matched = mock.MagicMock()
        mock_prev_matched.size.return_value.getInfo.return_value = 0
        mock_prev.filter.return_value = mock_prev_matched

        date_calls = []
        def track_filter_date(start_d, end_d):
            date_calls.append((start_d, end_d))
            if len(date_calls) <= 2:
                return mock_s1
            elif len(date_calls) == 3:
                return mock_recent
            else:
                return mock_prev

        mock_s1.filterDate.side_effect = track_filter_date
        mock_ee.ImageCollection.side_effect = lambda cid: mock_s2 if "S2" in cid else mock_s1
        mock_ee.Image.constant.return_value.pow.return_value.divide.return_value.rename.return_value = mock.MagicMock()
        mock_ee.Image.cat.return_value.reduceRegion.return_value = mock.MagicMock()

        farm = {"type": "circle", "latitude": 19.0, "longitude": 73.0, "radius_meters": 100.0}
        with mock.patch("krishidrishti_ai.gee_api._grid_from_anomaly_image", return_value=[[0.2]*8 for _ in range(8)]):
            res = service.fetch_farm_metrics(farm, date(2026, 9, 1), date(2026, 9, 7))

        assert res["data_availability"]["sar_change_available"] is False
        assert res["sentinel1"]["temporal_change"] is None


def test_sentinel_1_orbit_consistency_skips_incompatible_passes() -> None:
    mock_ee = mock.MagicMock()
    with mock.patch.dict(sys.modules, {"ee": mock_ee}):
        service = EarthEngineAnalysisService.__new__(EarthEngineAnalysisService)
        service.initialize = mock.MagicMock()
        service._to_ee_geometry = mock.MagicMock()

        mock_s2 = mock.MagicMock()
        mock_s2.filterBounds.return_value = mock_s2
        mock_s2.filterDate.return_value = mock_s2
        mock_s2.size.return_value.getInfo.return_value = 0

        mock_s1 = mock.MagicMock()
        mock_s1.size.return_value.getInfo.return_value = 1
        mock_s1.filterBounds.return_value = mock_s1
        mock_s1.filter.return_value = mock_s1
        mock_s1.map.return_value = mock_s1
        mock_s1_mean = mock.MagicMock()
        mock_s1.median.return_value.clip.return_value = mock_s1_mean
        mock_s1_mean.select.return_value.reduceRegion.return_value.getInfo.return_value = -15.0

        mock_recent = mock.MagicMock()
        mock_recent.size.return_value.getInfo.return_value = 1
        mock_latest = mock.MagicMock()
        mock_latest.get.return_value.getInfo.return_value = "DESCENDING"
        mock_recent.sort.return_value.first.return_value = mock_latest
        mock_recent_matched = mock.MagicMock()
        mock_recent_matched.size.return_value.getInfo.return_value = 1
        mock_recent.filter.return_value = mock_recent_matched

        mock_prev = mock.MagicMock()
        mock_prev.size.return_value.getInfo.return_value = 1  # Prev exists
        mock_prev_matched = mock.MagicMock()
        mock_prev_matched.size.return_value.getInfo.return_value = 0  # But 0 images with DESCENDING pass!
        mock_prev.filter.return_value = mock_prev_matched

        date_calls = []
        def track_filter_date(start_d, end_d):
            date_calls.append((start_d, end_d))
            if len(date_calls) <= 2:
                return mock_s1
            elif len(date_calls) == 3:
                return mock_recent
            else:
                return mock_prev

        mock_s1.filterDate.side_effect = track_filter_date
        mock_ee.ImageCollection.side_effect = lambda cid: mock_s2 if "S2" in cid else mock_s1
        mock_ee.Image.constant.return_value.pow.return_value.divide.return_value.rename.return_value = mock.MagicMock()
        mock_ee.Image.cat.return_value.reduceRegion.return_value = mock.MagicMock()

        farm = {"type": "circle", "latitude": 19.0, "longitude": 73.0, "radius_meters": 100.0}
        with mock.patch("krishidrishti_ai.gee_api._grid_from_anomaly_image", return_value=[[0.2]*8 for _ in range(8)]):
            res = service.fetch_farm_metrics(farm, date(2026, 9, 1), date(2026, 9, 7))

        assert res["data_availability"]["sar_change_available"] is False
        assert res["sentinel1"]["temporal_change"] is None


def test_sentinel_1_speckle_filter_presence() -> None:
    mock_ee = mock.MagicMock()
    with mock.patch.dict(sys.modules, {"ee": mock_ee}):
        service = EarthEngineAnalysisService.__new__(EarthEngineAnalysisService)
        image = mock.MagicMock()
        service._apply_sar_speckle_filter(image)
        image.focal_mean.assert_called_once_with(radius=1.5, kernelType="square", units="pixels")


def test_ndre_uses_b8a_narrow_nir_band() -> None:
    mock_ee = mock.MagicMock()
    with mock.patch.dict(sys.modules, {"ee": mock_ee}):
        service = EarthEngineAnalysisService.__new__(EarthEngineAnalysisService)
        service.initialize = mock.MagicMock()
        service._to_ee_geometry = mock.MagicMock()
        service._mask_s2_sr = mock.MagicMock()

        mock_s2 = mock.MagicMock()
        mock_s2.filterBounds.return_value = mock_s2
        mock_s2.filterDate.return_value = mock_s2
        mock_s2.size.return_value.getInfo.return_value = 1
        mock_s2.map.return_value = mock_s2
        s2_img = mock.MagicMock()
        mock_s2.median.return_value.clip.return_value = s2_img

        norm_diff_calls = []
        def track_norm_diff(bands):
            norm_diff_calls.append(bands)
            m = mock.MagicMock()
            m.reduceRegion.return_value.getInfo.return_value = 0.5
            return m

        s2_img.normalizedDifference.side_effect = track_norm_diff

        mock_s1 = mock.MagicMock()
        mock_s1.filterBounds.return_value = mock_s1
        mock_s1.filterDate.return_value = mock_s1
        mock_s1.filter.return_value = mock_s1
        mock_s1.size.return_value.getInfo.return_value = 0

        mock_ee.ImageCollection.side_effect = lambda cid: mock_s2 if "S2" in cid else mock_s1
        mock_ee.Image.cat.return_value.reduceRegion.return_value = mock.MagicMock()

        farm = {"type": "circle", "latitude": 19.0, "longitude": 73.0, "radius_meters": 100.0}
        with mock.patch("krishidrishti_ai.gee_api._grid_from_anomaly_image", return_value=[[0.2]*8 for _ in range(8)]):
            res = service.fetch_farm_metrics(farm, date(2026, 9, 1), date(2026, 9, 7))

        assert res["modality"] == "OPTICAL_ONLY"
        assert ["B8A", "B5"] in norm_diff_calls


def test_true_color_empty_collection_returns_none_safely() -> None:
    mock_ee = mock.MagicMock()
    with mock.patch.dict(sys.modules, {"ee": mock_ee}):
        service = EarthEngineAnalysisService.__new__(EarthEngineAnalysisService)
        service.initialize = mock.MagicMock()
        service._to_ee_geometry = mock.MagicMock()

        mock_s2 = mock.MagicMock()
        mock_s2.size.return_value.getInfo.return_value = 0  # 0 images!
        mock_s2.filterBounds.return_value = mock_s2
        mock_s2.filterDate.return_value = mock_s2
        mock_ee.ImageCollection.return_value = mock_s2

        farm = {"type": "circle", "latitude": 19.0, "longitude": 73.0, "radius_meters": 100.0}
        img = service.fetch_region_image(farm, date(2026, 9, 1), date(2026, 9, 7))
        assert img is None


def test_fusion_service_distinguishes_satellite_modalities() -> None:
    fusion = RiskFusionService()

    # 1. SAR_ONLY modality
    sar_only_result = {
        "status": "SUCCESS",
        "modality": "SAR_ONLY",
        "farm_local_abnormal_score": 0.82,
        "summary": {"connected_clusters": 2, "modality": "SAR_ONLY"},
    }
    risk_sar = fusion.assess_risk("cotton", diagnosis_result=None, satellite_result=sar_only_result)
    sat_sig = next(s for s in risk_sar.signals if s["name"] == "satellite_anomaly")
    assert "SAR radar" in sat_sig["value"]
    assert "radar backscatter/moisture/structure variation" in sat_sig["explanation"]
    assert "pathogen confirmation" in sat_sig["explanation"]

    # 2. OPTICAL_ONLY modality
    optical_result = {
        "status": "SUCCESS",
        "modality": "OPTICAL_ONLY",
        "farm_local_abnormal_score": 0.65,
        "summary": {"connected_clusters": 1, "modality": "OPTICAL_ONLY"},
    }
    risk_opt = fusion.assess_risk("cotton", diagnosis_result=None, satellite_result=optical_result)
    sat_sig_opt = next(s for s in risk_opt.signals if s["name"] == "satellite_anomaly")
    assert "Optical" in sat_sig_opt["value"]
    assert "chlorophyll" in sat_sig_opt["explanation"]

    # 3. COMBINED modality
    combined_result = {
        "status": "SUCCESS",
        "modality": "COMBINED",
        "farm_local_abnormal_score": 0.70,
        "summary": {"connected_clusters": 1, "modality": "COMBINED"},
    }
    risk_comb = fusion.assess_risk("cotton", diagnosis_result=None, satellite_result=combined_result)
    sat_sig_comb = next(s for s in risk_comb.signals if s["name"] == "satellite_anomaly")
    assert "Sentinel-1/2" in sat_sig_comb["explanation"]
