from __future__ import annotations

import base64
import math
import os
import urllib.request
from collections import deque
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

EARTH_RADIUS_M = 6_371_000.0
MIN_FARM_AREA_M2 = 16_187.4
INDIA_MIN_LAT = 6.0
INDIA_MAX_LAT = 38.0
INDIA_MIN_LON = 68.0
INDIA_MAX_LON = 97.5
DEFAULT_DATE_WINDOW_DAYS = 7
PREFERRED_MIN_DATE_WINDOW_DAYS = 5
PREFERRED_MAX_DATE_WINDOW_DAYS = 10
MAX_DATE_WINDOW_DAYS = 30

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _coerce_float(value: Any, field_name: str) -> float:
    if value is None:
        raise ValueError(f"{field_name} is required.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite.")
    return numeric


def _coerce_int(value: Any, field_name: str) -> int:
    if value is None:
        raise ValueError(f"{field_name} is required.")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc
    if numeric <= 0:
        raise ValueError(f"{field_name} must be positive.")
    return numeric


def _trim_private_key(raw: str | None) -> str:
    if raw is None:
        return ""
    return raw.strip().replace("\\n", "\n")


def _point_in_india(latitude: float, longitude: float) -> bool:
    return INDIA_MIN_LAT <= latitude <= INDIA_MAX_LAT and INDIA_MIN_LON <= longitude <= INDIA_MAX_LON


def _polygon_bounds(points: list[list[float]]) -> tuple[float, float, float, float]:
    lons = [float(point[0]) for point in points]
    lats = [float(point[1]) for point in points]
    return min(lons), max(lons), min(lats), max(lats)


def _polygon_area_m2(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for index, point in enumerate(points):
        longitude, latitude = point
        next_point = points[(index + 1) % len(points)]
        next_lon, next_lat = next_point
        total += math.radians(longitude) * math.radians(next_lat) - math.radians(next_lon) * math.radians(latitude)
    area = abs(total) / 2.0
    return area * EARTH_RADIUS_M**2


def _extract_polygon_coordinates(geometry: Any) -> list[list[float]]:
    if not isinstance(geometry, dict):
        raise ValueError("Geometry must be a GeoJSON object.")
    geometry_type = geometry.get("type")
    if geometry_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError("Geometry must be a GeoJSON Polygon or MultiPolygon.")
    if geometry_type == "Polygon":
        rings = geometry.get("coordinates")
        if not isinstance(rings, list) or not rings or not isinstance(rings[0], list):
            raise ValueError("Polygon coordinates are missing.")
        if len(rings[0]) < 4:
            raise ValueError("A polygon needs at least 4 coordinate pairs.")
        coordinates = [list(map(float, coord)) for coord in rings[0]]
        if coordinates[0] != coordinates[-1]:
            raise ValueError("Polygon linear ring must be closed.")
        return coordinates
    polygons = geometry.get("coordinates")
    if not isinstance(polygons, list) or not polygons or not isinstance(polygons[0], list):
        raise ValueError("MultiPolygon coordinates are missing.")
    rings = [ring for polygon in polygons if isinstance(polygon, list) for ring in polygon[:1]]
    if not rings or len(rings[0]) < 4:
        raise ValueError("A polygon needs at least 4 coordinate pairs.")
    coordinates = [list(map(float, coord)) for coord in rings[0]]
    if coordinates[0] != coordinates[-1]:
        raise ValueError("Polygon linear ring must be closed.")
    return coordinates


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    geometry = payload.get("geometry") or payload.get("polygon")
    latitude = payload.get("latitude")
    longitude = payload.get("longitude")
    radius_meters = payload.get("radius_meters")

    if geometry is not None:
        polygon_points = _extract_polygon_coordinates(geometry)
        min_lon, max_lon, min_lat, max_lat = _polygon_bounds(polygon_points)
        if any(not _point_in_india(lat, lon) for lon, lat in polygon_points):
            raise ValueError("Farm geometry must fall within India only.")
        area_m2 = _polygon_area_m2(polygon_points)
        if area_m2 < MIN_FARM_AREA_M2:
            raise ValueError("Farm area must be at least 4 acres (~16,187.4 m²).")
        return {
            "type": "polygon",
            "coordinates": polygon_points,
            "area_m2": area_m2,
            "latitude_min": min_lat,
            "latitude_max": max_lat,
            "longitude_min": min_lon,
            "longitude_max": max_lon,
        }

    if latitude is None or longitude is None:
        raise ValueError("Provide either a GeoJSON Polygon in 'geometry' or a 'latitude'/'longitude' pair with 'radius_meters'.")

    latitude_value = _coerce_float(latitude, "latitude")
    longitude_value = _coerce_float(longitude, "longitude")
    if not _point_in_india(latitude_value, longitude_value):
        raise ValueError("Farm centroid must fall within India only.")
    if radius_meters is None:
        raise ValueError("radius_meters is required when using latitude and longitude.")
    radius_value = _coerce_float(radius_meters, "radius_meters")
    if radius_value <= 0:
        raise ValueError("radius_meters must be positive.")
    area_m2 = math.pi * radius_value**2
    if area_m2 < MIN_FARM_AREA_M2:
        raise ValueError("Farm area must be at least 4 acres (~16,187.4 m²).")
    return {
        "type": "circle",
        "latitude": latitude_value,
        "longitude": longitude_value,
        "radius_meters": radius_value,
        "area_m2": area_m2,
        "latitude_min": latitude_value - radius_value / 111_000.0,
        "latitude_max": latitude_value + radius_value / 111_000.0,
        "longitude_min": longitude_value - radius_value / (111_000.0 * max(math.cos(math.radians(latitude_value)), 1e-6)),
        "longitude_max": longitude_value + radius_value / (111_000.0 * max(math.cos(math.radians(latitude_value)), 1e-6)),
    }


def _coerce_date(value: Any, field_name: str) -> date:
    if value is None:
        return date.today()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an ISO date string.") from exc
    raise ValueError(f"{field_name} must be a date or ISO date string.")


def _window_days(payload: dict[str, Any]) -> int:
    default_window = int(os.getenv("GEE_ANALYSIS_DATE_WINDOW_DAYS", str(DEFAULT_DATE_WINDOW_DAYS)))
    candidate = payload.get("date_window_days", default_window)
    if candidate is None:
        candidate = default_window
    window = _coerce_int(candidate, "date_window_days")
    if window > MAX_DATE_WINDOW_DAYS:
        raise ValueError("The farm analysis date window must be 30 days or less.")
    if window < 1:
        raise ValueError("date_window_days must be at least 1.")
    return window


def _resolve_window(payload: dict[str, Any]) -> tuple[date, date]:
    window_days = _window_days(payload)
    end_date = _coerce_date(payload.get("end_date"), "end_date")
    start_date = end_date - timedelta(days=window_days - 1)
    if window_days > PREFERRED_MAX_DATE_WINDOW_DAYS or window_days < PREFERRED_MIN_DATE_WINDOW_DAYS:
        return start_date, end_date
    return start_date, end_date


def _extract_numeric_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, dict):
        for nested in value.values():
            numeric = _extract_numeric_value(nested)
            if numeric is not None:
                return numeric
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, str):
        try:
            numeric = float(value)
        except ValueError:
            return None
        return numeric if math.isfinite(numeric) else None
    return None


def _clean_sampled_grid(value: Any) -> list[list[float | None]]:
    if value is None:
        return [[None]]
    if not isinstance(value, list) or not value or not all(isinstance(row, list) for row in value):
        raise ValueError("Earth Engine returned no anomaly pixels for this farm.")
    return [
        [
            None if item is None or (isinstance(item, (int, float)) and item <= -9998) else float(item)
            for item in row
        ]
        for row in value
    ]


def _grid_from_anomaly_image(anomaly_image: Any, region: Any, rows: int = 8, cols: int = 8) -> list[list[float | None]]:
    import ee  # type: ignore

    coords = region.bounds().getInfo()["coordinates"][0]
    lon_values = [point[0] for point in coords]
    lat_values = [point[1] for point in coords]
    lon_min, lon_max = min(lon_values), max(lon_values)
    lat_min, lat_max = min(lat_values), max(lat_values)
    lon_step = (lon_max - lon_min) / max(cols, 1)
    lat_step = (lat_max - lat_min) / max(rows, 1)

    grid: list[list[float | None]] = []
    for row_index in range(rows):
        row_values: list[float | None] = []
        for col_index in range(cols):
            cell_lon_min = lon_min + col_index * lon_step
            cell_lon_max = lon_min + (col_index + 1) * lon_step
            cell_lat_min = lat_min + row_index * lat_step
            cell_lat_max = lat_min + (row_index + 1) * lat_step
            cell_geometry = ee.Geometry.Rectangle([cell_lon_min, cell_lat_min, cell_lon_max, cell_lat_max])
            reduction = anomaly_image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=cell_geometry,
                scale=10,
                maxPixels=1e9,
            ).getInfo()
            row_values.append(_extract_numeric_value(reduction.get("ANOMALY")))
        grid.append(row_values)
    return grid


class EarthEngineSettings:
    def __init__(
        self,
        service_account_email: str,
        private_key: str,
        project_id: str,
        date_window_days: int = DEFAULT_DATE_WINDOW_DAYS,
        percentile_threshold: float = 95.0,
        zscore_threshold: float = 2.5,
        min_cluster_size: int = 3,
    ) -> None:
        self.service_account_email = service_account_email
        self.private_key = private_key
        self.project_id = project_id
        self.date_window_days = date_window_days
        self.percentile_threshold = percentile_threshold
        self.zscore_threshold = zscore_threshold
        self.min_cluster_size = min_cluster_size

    @classmethod
    def from_env(cls) -> "EarthEngineSettings":
        service_account_email = os.getenv("GEE_SERVICE_ACCOUNT_EMAIL")
        if not service_account_email:
            raise ValueError("Set GEE_SERVICE_ACCOUNT_EMAIL to the Earth Engine service account email.")
        private_key = _trim_private_key(os.getenv("GEE_PRIVATE_KEY"))
        if not private_key:
            key_path = os.getenv("GEE_PRIVATE_KEY_PATH")
            if key_path:
                private_key = Path(key_path).read_text(encoding="utf-8")
        if not private_key:
            raise ValueError("Set GEE_PRIVATE_KEY or GEE_PRIVATE_KEY_PATH to the service account private key.")
        project_id = os.getenv("GEE_PROJECT_ID")
        if not project_id:
            raise ValueError("Set GEE_PROJECT_ID to the Earth Engine project identifier.")
        date_window_days = _coerce_int(os.getenv("GEE_ANALYSIS_DATE_WINDOW_DAYS", str(DEFAULT_DATE_WINDOW_DAYS)), "GEE_ANALYSIS_DATE_WINDOW_DAYS")
        if date_window_days > MAX_DATE_WINDOW_DAYS:
            raise ValueError("GEE_ANALYSIS_DATE_WINDOW_DAYS must be 30 days or less.")
        percentile_threshold = _coerce_float(os.getenv("GEE_PERCENTILE_THRESHOLD", "95.0"), "GEE_PERCENTILE_THRESHOLD")
        zscore_threshold = _coerce_float(os.getenv("GEE_ZSCORE_THRESHOLD", "2.5"), "GEE_ZSCORE_THRESHOLD")
        min_cluster_size = _coerce_int(os.getenv("GEE_MIN_CLUSTER_SIZE", "3"), "GEE_MIN_CLUSTER_SIZE")
        return cls(
            service_account_email=service_account_email,
            private_key=_trim_private_key(private_key),
            project_id=project_id,
            date_window_days=date_window_days,
            percentile_threshold=percentile_threshold,
            zscore_threshold=zscore_threshold,
            min_cluster_size=min_cluster_size,
        )


class EarthEngineAnalysisService:
    def __init__(self, settings: EarthEngineSettings | None = None) -> None:
        self.settings = settings or EarthEngineSettings.from_env()

    def initialize(self) -> None:
        try:
            import ee  # type: ignore
        except ImportError as exc:
            raise RuntimeError("The 'earthengine-api' package is required for Google Earth Engine analysis.") from exc
        credentials = ee.ServiceAccountCredentials(
            self.settings.service_account_email,
            key_data=self.settings.private_key,
        )
        
        ee.Initialize(credentials=credentials, project=self.settings.project_id)

    def _to_ee_geometry(self, farm: dict[str, Any]) -> Any:
        import ee  # type: ignore

        if farm["type"] == "polygon":
            polygon = [[float(lon), float(lat)] for lon, lat in farm["coordinates"]]
            return ee.Geometry.Polygon([polygon])
        return ee.Geometry.Point([farm["longitude"], farm["latitude"]]).buffer(farm["radius_meters"])

    def fetch_region_image(self, farm: dict[str, Any], start_date: date, end_date: date, dimensions: int = 900) -> str:
        self.initialize()
        import ee  # type: ignore

        region = self._to_ee_geometry(farm).bounds()
        image = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(ee.Date(start_date.isoformat()), ee.Date(end_date.isoformat()))
            .map(lambda image: image.updateMask(self._mask_s2_sr(image)))
            .median()
            .clip(region)
        )
        rgb = image.select(["B4", "B3", "B2"]).multiply(0.0001).clamp(0, 1)
        visualized = rgb.visualize(min=0, max=0.35, gamma=1.4)
        thumb_url = visualized.getThumbURL({
            "region": region,
            "dimensions": dimensions,
            "format": "png",
            "crs": "EPSG:4326",
        })
        if not thumb_url:
            raise ValueError("Earth Engine could not produce a regional satellite image for this farm.")
        with urllib.request.urlopen(thumb_url) as response:
            payload = response.read()
        return _encode_image_bytes_to_data_url(payload, "image/png")

    def _mask_s2_sr(self, image: Any) -> Any:
        scl = image.select("SCL")
        return scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))

    def fetch_farm_metrics(self, farm: dict[str, Any], start_date: date, end_date: date) -> dict[str, Any]:
        self.initialize()
        import ee  # type: ignore

        region = self._to_ee_geometry(farm)
        start = ee.Date(start_date.isoformat())
        end = ee.Date(end_date.isoformat())

        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(start, end)
        )
        if s2.size().getInfo() == 0:
            raise ValueError("No Sentinel-2 imagery is available for this farm and date window.")
        s2_image = s2.map(lambda image: image.updateMask(self._mask_s2_sr(image))).median().clip(region)
        mask = self._mask_s2_sr(s2_image)
        ndvi = s2_image.normalizedDifference(["B8", "B4"]).updateMask(mask)
        ndmi = s2_image.normalizedDifference(["B8", "B11"]).updateMask(mask)
        ndre = s2_image.normalizedDifference(["B8", "B5"]).updateMask(mask)

        s1 = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(region)
            .filterDate(start, end)
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        )
        if s1.size().getInfo() == 0:
            raise ValueError("No Sentinel-1 VV/VH imagery is available for this farm and date window.")
        s1_mean = s1.median().clip(region)
        vv = s1_mean.select("VV")
        vh = s1_mean.select("VH")
        vh_vv_ratio = ee.Image.constant(10).pow(vh.divide(10)).divide(ee.Image.constant(10).pow(vv.divide(10))).rename("VH_VV_RATIO")

        s2_ndvi = ndvi.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=10, maxPixels=1e9)
        s2_ndmi = ndmi.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=10, maxPixels=1e9)
        s2_ndre = ndre.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=10, maxPixels=1e9)
        vv_stats = vv.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=10, maxPixels=1e9)
        vh_stats = vh.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=10, maxPixels=1e9)
        vh_vv_ratio_stats = vh_vv_ratio.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=10, maxPixels=1e9)
        midpoint = start.advance(end.difference(start, "day").divide(2), "day")
        previous = s1.filterDate(start, midpoint).median().select(["VV", "VH"])
        recent = s1.filterDate(midpoint, end).median().select(["VV", "VH"])
        temporal_change = recent.subtract(previous).abs().reduce(ee.Reducer.mean()).rename("SAR_CHANGE")
        metrics_image = ee.Image.cat([
            ndvi.rename("NDVI"),
            ndmi.rename("NDMI"),
            ndre.rename("NDRE"),
            vh_vv_ratio,
            temporal_change,
        ])
        stats = metrics_image.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
            geometry=region,
            scale=10,
            maxPixels=1e9,
        )
        mean_image = ee.Image.constant([stats.get("NDVI_mean"), stats.get("NDMI_mean"), stats.get("NDRE_mean"), stats.get("VH_VV_RATIO_mean"), stats.get("SAR_CHANGE_mean")]).rename(metrics_image.bandNames())
        std_image = ee.Image.constant([stats.get("NDVI_stdDev"), stats.get("NDMI_stdDev"), stats.get("NDRE_stdDev"), stats.get("VH_VV_RATIO_stdDev"), stats.get("SAR_CHANGE_stdDev")]).rename(metrics_image.bandNames())
        low_health = mean_image.subtract(metrics_image).divide(std_image.max(0.0001))
        anomaly_image = low_health.select(["NDVI", "NDMI", "NDRE", "VH_VV_RATIO"]).reduce(ee.Reducer.mean()).add(
            metrics_image.select("SAR_CHANGE").subtract(mean_image.select("SAR_CHANGE")).abs().divide(std_image.select("SAR_CHANGE").max(0.0001))
        ).divide(2).rename("ANOMALY")
        anomaly_grid = _grid_from_anomaly_image(anomaly_image, region, rows=8, cols=8)

        return {
            "data_availability": {
                "sentinel_2_sr": True,
                "sentinel_1_grd": True,
                "sentinel_2_acquisitions": s2.size().getInfo(),
                "sentinel_1_acquisitions": s1.size().getInfo(),
            },
            "sentinel2": {
                "ndvi_mean": _extract_numeric_value(s2_ndvi.getInfo()),
                "ndmi_mean": _extract_numeric_value(s2_ndmi.getInfo()),
                "ndre_mean": _extract_numeric_value(s2_ndre.getInfo()),
            },
            "sentinel1": {
                "vv_mean_db": _extract_numeric_value(vv_stats.getInfo()),
                "vh_mean_db": _extract_numeric_value(vh_stats.getInfo()),
                "vh_vv_ratio_mean": _extract_numeric_value(vh_vv_ratio_stats.getInfo()),
                "temporal_change": _extract_numeric_value(
                    temporal_change.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=region,
                        scale=10,
                        maxPixels=1e9,
                    ).getInfo()
                ),
            },
            "anomaly_grid": anomaly_grid,
        }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * (percentile / 100.0)
    lower_index = int(math.floor(index))
    upper_index = int(math.ceil(index))
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    if lower_value == upper_value:
        return lower_value
    return lower_value + (upper_value - lower_value) * (index - lower_index)


def _constrained_neighbors(row_index: int, col_index: int, rows: int, cols: int) -> list[tuple[int, int]]:
    neighbors: list[tuple[int, int]] = []
    for delta_row in (-1, 0, 1):
        for delta_col in (-1, 0, 1):
            if delta_row == 0 and delta_col == 0:
                continue
            current_row = row_index + delta_row
            current_col = col_index + delta_col
            if 0 <= current_row < rows and 0 <= current_col < cols:
                neighbors.append((current_row, current_col))
    return neighbors


def _cluster_grid(mask: list[list[bool]]) -> list[dict[str, Any]]:
    rows = len(mask)
    cols = len(mask[0]) if rows else 0
    visited: set[tuple[int, int]] = set()
    clusters: list[dict[str, Any]] = []
    for row_index in range(rows):
        for col_index in range(cols):
            if not mask[row_index][col_index] or (row_index, col_index) in visited:
                continue
            queue: deque[tuple[int, int]] = deque([(row_index, col_index)])
            visited.add((row_index, col_index))
            cells: list[tuple[int, int]] = []
            while queue:
                current_row, current_col = queue.popleft()
                cells.append((current_row, current_col))
                for next_row, next_col in _constrained_neighbors(current_row, current_col, rows, cols):
                    if mask[next_row][next_col] and (next_row, next_col) not in visited:
                        visited.add((next_row, next_col))
                        queue.append((next_row, next_col))
            clusters.append({"cells": cells})
    return clusters


def _grid_to_geojson(cluster_cells: list[tuple[int, int]], lat_min: float, lat_max: float, lon_min: float, lon_max: float, rows: int, cols: int) -> dict[str, Any]:
    if not cluster_cells:
        return {"type": "FeatureCollection", "features": []}
    lat_step = (lat_max - lat_min) / max(1, rows)
    lon_step = (lon_max - lon_min) / max(1, cols)
    polygons = []
    for row, col in cluster_cells:
        polygons.append([[
            [lon_min + col * lon_step, lat_min + row * lat_step],
            [lon_min + (col + 1) * lon_step, lat_min + row * lat_step],
            [lon_min + (col + 1) * lon_step, lat_min + (row + 1) * lat_step],
            [lon_min + col * lon_step, lat_min + (row + 1) * lat_step],
            [lon_min + col * lon_step, lat_min + row * lat_step],
        ]])
    return {
        "type": "Feature",
        "geometry": {"type": "MultiPolygon", "coordinates": polygons},
        "properties": {"cluster_size": len(cluster_cells)},
    }


def _encode_image_bytes_to_data_url(image_bytes: bytes, mime_type: str = "image/png") -> str:
    return f"data:{mime_type};base64," + base64.b64encode(image_bytes).decode("ascii")


def _cluster_properties(
    cells: list[tuple[int, int]],
    label: str,
    rows: int,
    cols: int,
    bounds: tuple[float, float, float, float] | None,
    cluster_score: float | None = None,
    cluster_index: int | None = None,
) -> dict[str, Any]:
    if not bounds:
        return {"label": label, "cluster_index": cluster_index, "pixel_count": len(cells), "cluster_score": cluster_score}

    lon_min, lon_max, lat_min, lat_max = bounds
    lat_step = (lat_max - lat_min) / max(1, rows)
    lon_step = (lon_max - lon_min) / max(1, cols)
    center_lon = sum(lon_min + (col + 0.5) * lon_step for _, col in cells) / max(len(cells), 1)
    center_lat = sum(lat_min + (row + 0.5) * lat_step for row, _ in cells) / max(len(cells), 1)
    extreme_distances = []
    for row, col in cells:
        corners = [
            (lon_min + col * lon_step, lat_min + row * lat_step),
            (lon_min + (col + 1) * lon_step, lat_min + row * lat_step),
            (lon_min + (col + 1) * lon_step, lat_min + (row + 1) * lat_step),
            (lon_min + col * lon_step, lat_min + (row + 1) * lat_step),
        ]
        for lon, lat in corners:
            dx = (lon - center_lon) * 111_000.0 * max(math.cos(math.radians(center_lat)), 1e-6)
            dy = (lat - center_lat) * 111_000.0
            extreme_distances.append(math.hypot(dx, dy))
    radius_m = max(extreme_distances, default=0.0)
    geometry = _grid_to_geojson(cells, lat_min, lat_max, lon_min, lon_max, rows, cols)["geometry"]
    result = {
        "label": label,
        "cluster_index": cluster_index,
        "pixel_count": len(cells),
        "cluster_score": cluster_score,
        "latitude": round(center_lat, 7),
        "longitude": round(center_lon, 7),
        "radius_meters": round(radius_m, 2),
    }
    if geometry is not None:
        result["geometry"] = geometry
    return result


def _farm_health_label(
    connected_clusters: int,
    healthy_zones: int,
    unreachable_zones: int,
    anomalous_pixels: int,
) -> str:
    if connected_clusters > 0 or anomalous_pixels > 0:
        return "abnormal"
    if healthy_zones > 0:
        return "healthy"
    if unreachable_zones > 0:
        return "unreachable"
    return "healthy"


def compute_local_abnormality(
    grid: list[list[float | None]],
    percentile_threshold: float = 95.0,
    zscore_threshold: float = 2.5,
    min_cluster_size: int = 3,
    bounds: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    if not 0 < percentile_threshold <= 100:
        raise ValueError("percentile_threshold must be greater than 0 and at most 100.")
    if zscore_threshold <= 0 or min_cluster_size <= 0:
        raise ValueError("zscore_threshold and min_cluster_size must be positive.")

    flattened = [float(value) for row in grid for value in row if value is not None and math.isfinite(float(value))]
    if not flattened:
        return {
            "farm_local_abnormal_score": 0.0,
            "connected_clusters": 0,
            "healthy_zones": 0,
            "unreachable_zones": 0,
            "anomalous_pixels": 0,
            "max_cluster_score": 0.0,
            "thresholds": {"percentile": percentile_threshold, "zscore": zscore_threshold},
            "geojson": {"type": "FeatureCollection", "features": []},
        }

    mean = sum(flattened) / len(flattened)
    variance = sum((value - mean) ** 2 for value in flattened) / len(flattened)
    std_dev = math.sqrt(variance)
    cut_point = _percentile(flattened, percentile_threshold)

    abnormal_mask: list[list[bool]] = []
    reachable_mask: list[list[bool]] = []
    score_grid: list[list[float]] = []
    for row in grid:
        abnormal_row: list[bool] = []
        score_row: list[float] = []
        reachable_row: list[bool] = []
        for value in row:
            if value is None or not math.isfinite(float(value)):
                abnormal_row.append(False)
                score_row.append(0.0)
                reachable_row.append(False)
                continue
            numeric = float(value)
            zscore = 0.0 if std_dev == 0 else (numeric - mean) / std_dev
            flagged = numeric >= cut_point or abs(zscore) >= zscore_threshold
            abnormal_row.append(flagged)
            score_row.append(max(0.0, max(numeric - cut_point, 0.0) / max(abs(cut_point), 1e-6), abs(zscore) / max(zscore_threshold, 1e-6)))
            reachable_row.append(True)
        abnormal_mask.append(abnormal_row)
        reachable_mask.append(reachable_row)
        score_grid.append(score_row)

    anomaly_clusters = [cluster for cluster in _cluster_grid(abnormal_mask) if len(cluster["cells"]) >= min_cluster_size]
    reachable_clusters = [cluster for cluster in _cluster_grid(reachable_mask) if len(cluster["cells"]) >= min_cluster_size]
    unreachable_clusters = [cluster for cluster in _cluster_grid([[not cell for cell in row] for row in reachable_mask]) if len(cluster["cells"]) >= min_cluster_size]
    healthy_clusters = []
    for row_index, row in enumerate(reachable_mask):
        for col_index, is_reachable in enumerate(row):
            if not is_reachable:
                continue
            if abnormal_mask[row_index][col_index]:
                continue
            # collect a connected healthy cluster through the same helper logic
            pass
    healthy_mask = [[reachable and not abnormal for reachable, abnormal in zip(row_reachable, row_abnormal)] for row_reachable, row_abnormal in zip(reachable_mask, abnormal_mask)]
    healthy_clusters = [cluster for cluster in _cluster_grid(healthy_mask) if len(cluster["cells"]) >= min_cluster_size]

    features: list[dict[str, Any]] = []
    cluster_scores: list[float] = []
    for cluster_index, cluster in enumerate(anomaly_clusters):
        cells = cluster["cells"]
        cluster_score = sum(score_grid[row][col] for row, col in cells) / max(len(cells), 1)
        cluster_scores.append(cluster_score)
        props = _cluster_properties(cells, "anomaly", len(grid), len(grid[0]) if grid else 0, bounds, cluster_score, cluster_index)
        geometry = props.pop("geometry", None)
        features.append({"type": "Feature", "properties": props, "geometry": geometry})

    for cluster_index, cluster in enumerate(healthy_clusters):
        cells = cluster["cells"]
        props = _cluster_properties(cells, "healthy", len(grid), len(grid[0]) if grid else 0, bounds, None, cluster_index)
        geometry = props.pop("geometry", None)
        features.append({"type": "Feature", "properties": props, "geometry": geometry})

    for cluster_index, cluster in enumerate(unreachable_clusters):
        cells = cluster["cells"]
        props = _cluster_properties(cells, "unreachable", len(grid), len(grid[0]) if grid else 0, bounds, None, cluster_index)
        geometry = props.pop("geometry", None)
        features.append({"type": "Feature", "properties": props, "geometry": geometry})

    farm_local_abnormal_score = sum(cluster_scores) / max(len(cluster_scores), 1) if cluster_scores else 0.0
    anomalous_pixels = sum(1 for row in abnormal_mask for value in row if value)
    farm_label = _farm_health_label(len(anomaly_clusters), len(healthy_clusters), len(unreachable_clusters), anomalous_pixels)
    geojson = {"type": "FeatureCollection", "features": features}
    return {
        "farm_local_abnormal_score": round(farm_local_abnormal_score, 6),
        "label": farm_label,
        "connected_clusters": len(anomaly_clusters),
        "healthy_zones": len(healthy_clusters),
        "unreachable_zones": len(unreachable_clusters),
        "anomalous_pixels": anomalous_pixels,
        "max_cluster_score": max(cluster_scores) if cluster_scores else 0.0,
        "thresholds": {"percentile": percentile_threshold, "zscore": zscore_threshold},
        "geojson": geojson,
    }


class FarmAnalysisService:
    def __init__(self, settings: EarthEngineSettings | None = None) -> None:
        self.settings = settings or EarthEngineSettings.from_env()
        self.gee = EarthEngineAnalysisService(self.settings)

    def analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        farm = _normalize_payload(payload)
        start_date, end_date = _resolve_window(payload)
        metrics = self.gee.fetch_farm_metrics(farm, start_date, end_date)
        grid = metrics.get("anomaly_grid")
        if grid is None:
            raise ValueError("Earth Engine returned no anomaly pixels for this farm.")
        field_scores = compute_local_abnormality(
            grid,
            percentile_threshold=float(payload.get("percentile_threshold", self.settings.percentile_threshold)),
            zscore_threshold=float(payload.get("zscore_threshold", self.settings.zscore_threshold)),
            min_cluster_size=int(payload.get("min_cluster_size", self.settings.min_cluster_size)),
            bounds=(
                farm["longitude_min"],
                farm["longitude_max"],
                farm["latitude_min"],
                farm["latitude_max"],
            ),
        )
        farm_label = _farm_health_label(
            field_scores["connected_clusters"],
            field_scores["healthy_zones"],
            field_scores["unreachable_zones"],
            field_scores["anomalous_pixels"],
        )
        summary = {
            "area_m2": round(farm["area_m2"], 4),
            "date_window_days": (end_date - start_date).days + 1,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "label": farm_label,
            "farm_label": farm_label,
            "farm_local_abnormal_score": field_scores["farm_local_abnormal_score"],
            "connected_clusters": field_scores["connected_clusters"],
            "healthy_zones": field_scores["healthy_zones"],
            "unreachable_zones": field_scores["unreachable_zones"],
            "anomalous_pixels": field_scores["anomalous_pixels"],
            "max_cluster_score": field_scores["max_cluster_score"],
            "sentinel2": {
                "ndvi_mean": metrics.get("sentinel2", {}).get("ndvi_mean"),
                "ndmi_mean": metrics.get("sentinel2", {}).get("ndmi_mean"),
                "ndre_mean": metrics.get("sentinel2", {}).get("ndre_mean"),
            },
            "sentinel1": {
                "vv_mean_db": metrics.get("sentinel1", {}).get("vv_mean_db"),
                "vh_mean_db": metrics.get("sentinel1", {}).get("vh_mean_db"),
                "vh_vv_ratio_mean": metrics.get("sentinel1", {}).get("vh_vv_ratio_mean"),
                "temporal_change": metrics.get("sentinel1", {}).get("temporal_change"),
            },
            "score_thresholds": field_scores["thresholds"],
        }
        region_image = self.gee.fetch_region_image(farm, start_date, end_date)
        return {
            "label": farm_label,
            "farm_label": farm_label,
            "farm_local_abnormal_score": field_scores["farm_local_abnormal_score"],
            "summary": summary,
            "geojson": field_scores["geojson"],
            "data_availability": metrics.get("data_availability", {}),
            "region_image": region_image,
            "region_image_mime_type": "image/png",
        }


def get_gee_service() -> FarmAnalysisService:
    return FarmAnalysisService()
