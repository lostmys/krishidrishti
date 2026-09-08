from __future__ import annotations

import math
import os
import urllib.request
from collections import deque
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv

EARTH_RADIUS_M = 6_371_000.0
MIN_FARM_AREA_ACRES = 0.5
MIN_FARM_AREA_M2 = 2_023.43  # ~0.5 acre (conservative threshold supporting smallholders)
DEFAULT_POINT_RADIUS_METERS = 100.0  # ~7.76 acres proxy area for smallholders (not exact cadastral boundary)
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
        if area_m2 < MIN_FARM_AREA_M2 - 1e-4:
            raise ValueError(f"Farm area ({area_m2:.1f} m²) must be at least {MIN_FARM_AREA_ACRES} acre (~{MIN_FARM_AREA_M2:.1f} m²).")
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
        radius_meters = DEFAULT_POINT_RADIUS_METERS
    radius_value = _coerce_float(radius_meters, "radius_meters")
    if radius_value <= 0:
        raise ValueError("radius_meters must be positive.")
    area_m2 = math.pi * radius_value**2
    if area_m2 < MIN_FARM_AREA_M2 - 1e-4:
        raise ValueError(f"Farm area ({area_m2:.1f} m²) must be at least {MIN_FARM_AREA_ACRES} acre (~{MIN_FARM_AREA_M2:.1f} m²).")
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
    """Sample the anomaly image over an 8x8 grid within the farm region's bounding box.

    Uses a single server-side reduceRegions() call over an ee.FeatureCollection rather than
    64 sequential reduceRegion() calls to prevent latency spikes and API rate limit exhaustion.
    """
    import ee  # type: ignore

    coords = region.bounds().getInfo()["coordinates"][0]
    lon_values = [point[0] for point in coords]
    lat_values = [point[1] for point in coords]
    lon_min, lon_max = min(lon_values), max(lon_values)
    lat_min, lat_max = min(lat_values), max(lat_values)
    lon_step = (lon_max - lon_min) / max(cols, 1)
    lat_step = (lat_max - lat_min) / max(rows, 1)

    features = []
    for row_index in range(rows):
        for col_index in range(cols):
            cell_lon_min = lon_min + col_index * lon_step
            cell_lon_max = lon_min + (col_index + 1) * lon_step
            cell_lat_min = lat_min + row_index * lat_step
            cell_lat_max = lat_min + (row_index + 1) * lat_step
            cell_geometry = ee.Geometry.Rectangle([cell_lon_min, cell_lat_min, cell_lon_max, cell_lat_max])
            features.append(ee.Feature(cell_geometry, {"row": row_index, "col": col_index}))

    feature_collection = ee.FeatureCollection(features)
    reduced = anomaly_image.reduceRegions(
        collection=feature_collection,
        reducer=ee.Reducer.mean(),
        scale=10,
    ).getInfo()

    grid: list[list[float | None]] = [[None for _ in range(cols)] for _ in range(rows)]
    for feature in reduced.get("features", []):
        props = feature.get("properties", {})
        r = props.get("row")
        c = props.get("col")
        if r is not None and c is not None and 0 <= r < rows and 0 <= c < cols:
            val = props.get("ANOMALY")
            if val is None:
                val = props.get("mean")
            grid[r][c] = _extract_numeric_value(val)
    return grid


def _sample_anomaly_points(anomaly_image: Any, region: Any, scale: int = 10) -> list[dict[str, float]]:
    samples = anomaly_image.sample(
        region=region,
        scale=scale,
        geometries=True,
        # Earth Engine limits server-side aggregations to 5,000 elements.
        # Keep a safety margin because geometries add response payload.
        numPixels=4_000,
        seed=7,
    ).getInfo()
    points: list[dict[str, float]] = []
    for feature in samples.get("features", []):
        geometry = feature.get("geometry", {})
        coordinates = geometry.get("coordinates", [])
        value = _extract_numeric_value(feature.get("properties", {}).get("ANOMALY"))
        if value is None or not isinstance(coordinates, list) or len(coordinates) != 2:
            continue
        points.append({
            "longitude": round(float(coordinates[0]), 7),
            "latitude": round(float(coordinates[1]), 7),
            "value": value,
        })
    return points


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

<<<<<<< HEAD
    def fetch_region_image(
        self,
        farm: dict[str, Any],
        start_date: date,
        end_date: date,
        dimensions: int = 900,
        anomaly_geojson: dict[str, Any] | None = None,
        anomaly_points: list[dict[str, Any]] | None = None,
    ) -> str:
=======
    def fetch_region_image(self, farm: dict[str, Any], start_date: date, end_date: date, dimensions: int = 900) -> str | None:
        """Fetch true-color Sentinel-2 preview image for the farm parcel.

        Returns None safely if no clear Sentinel-2 scenes exist in the date window (P1-6),
        preventing crashes and avoiding fake image generation.
        """
>>>>>>> dcc12f8cbf742f8390e1320bd097f65d9f314692
        self.initialize()
        import ee  # type: ignore

        region = self._to_ee_geometry(farm).bounds()
        s2_collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(ee.Date(start_date.isoformat()), ee.Date(end_date.isoformat()))
        )
        if s2_collection.size().getInfo() == 0:
            return None

        # Mask individual scenes prior to compositing (P0-2)
        image = (
            s2_collection
            .map(self._mask_s2_sr)
            .median()
            .clip(region)
        )
<<<<<<< HEAD
        # Sentinel-2 SR stores reflectance as scaled integers (scale factor
        # 10,000). Visualize the native values with a per-band stretch so the
        # RGB thumbnail retains its natural color instead of looking grayscale.
        rgb = image.select(["B4", "B3", "B2"])
        visualized = rgb.visualize(min=200, max=3500, gamma=1.25)
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
        if anomaly_geojson is not None:
            payload = _overlay_farm_and_anomalies(
                payload,
                farm,
                anomaly_geojson,
                (
                    farm["longitude_min"],
                    farm["longitude_max"],
                    farm["latitude_min"],
                    farm["latitude_max"],
                ),
                anomaly_points=anomaly_points,
            )
        output_dir = Path(os.getenv("GEE_IMAGE_OUTPUT_DIR", "artifacts/gee_images")).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"farm_analysis_{date.today().isoformat()}_{uuid4().hex}.png"
        output_path.write_bytes(payload)
        return str(output_path.resolve())
=======
        rgb = image.select(["B4", "B3", "B2"]).multiply(0.0001).clamp(0, 1)
        visualized = rgb.visualize(min=0, max=0.35, gamma=1.4)
        try:
            thumb_url = visualized.getThumbURL({
                "region": region,
                "dimensions": dimensions,
                "format": "png",
                "crs": "EPSG:4326",
            })
            if not thumb_url:
                return None
            with urllib.request.urlopen(thumb_url) as response:
                payload = response.read()
            return _encode_image_bytes_to_data_url(payload, "image/png")
        except Exception:
            return None
>>>>>>> dcc12f8cbf742f8390e1320bd097f65d9f314692

    def _mask_s2_sr(self, image: Any) -> Any:
        """Mask cloud, cloud shadow, cirrus, and invalid pixels from Sentinel-2 SR using SCL band (P0-2).

        Excluded SCL classes:
          0: NO_DATA
          1: SATURATED_OR_DEFECTIVE
          3: CLOUD_SHADOW
          8: CLOUD_MEDIUM_PROBABILITY
          9: CLOUD_HIGH_PROBABILITY
          10: THIN_CIRRUS
          11: SNOW_ICE
        Preserves valid vegetation (4) and bare soil (5), water (6), dark area (2), and unclassified (7).
        Note: SCL provides scene-quality cloud screening; it does not remove all atmospheric artifacts.
        """
        scl = image.select("SCL")
        valid_mask = (
            scl.neq(0)
            .And(scl.neq(1))
            .And(scl.neq(3))
            .And(scl.neq(8))
            .And(scl.neq(9))
            .And(scl.neq(10))
            .And(scl.neq(11))
        )
        return image.updateMask(valid_mask)

    def _apply_sar_speckle_filter(self, image: Any) -> Any:
        """Apply a conservative 3x3 spatial focal mean filter to reduce SAR speckle noise (P1-4).

        Note: Spatial filtering reduces speckle variance but also softens fine spatial detail.
        It does not eliminate radar noise completely.
        """
        return image.focal_mean(radius=1.5, kernelType="square", units="pixels")

    def fetch_farm_metrics(self, farm: dict[str, Any], start_date: date, end_date: date) -> dict[str, Any]:
        """Fetch multispectral and SAR metrics for a farm parcel.

        Implements S2->S1 fallback (P1-1), 12-24d S1 temporal change with orbit consistency (P1-2, P1-3),
        speckle filtering (P1-4), NDRE narrow-NIR B8A (P1-5), and batch grid reduction (P0-1).
        """
        self.initialize()
        import ee  # type: ignore

        region = self._to_ee_geometry(farm)
        start = ee.Date(start_date.isoformat())
        end = ee.Date(end_date.isoformat())

        # 1. Query Sentinel-2 Collection
        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(start, end)
        )
        s2_count = int(s2.size().getInfo())

        # 2. Query Sentinel-1 Collection
        s1 = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(region)
            .filterDate(start, end)
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        )
        s1_count = int(s1.size().getInfo())

<<<<<<< HEAD
        s2_ndvi = ndvi.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=10, maxPixels=1e9)
        s2_ndmi = ndmi.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=10, maxPixels=1e9)
        s2_ndre = ndre.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=10, maxPixels=1e9)
        vv_stats = vv.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=10, maxPixels=1e9)
        vh_stats = vh.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=10, maxPixels=1e9)
        vh_vv_ratio_stats = vh_vv_ratio.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=10, maxPixels=1e9)
        midpoint = start.advance(end.difference(start, "day").divide(2), "day")
        previous_collection = s1.filterDate(start, midpoint)
        recent_collection = s1.filterDate(midpoint, end)
        previous_count = previous_collection.size().getInfo()
        recent_count = recent_collection.size().getInfo()
        if previous_count and recent_count:
            previous = previous_collection.median().select(["VV", "VH"])
            recent = recent_collection.median().select(["VV", "VH"])
            temporal_change = recent.subtract(previous).abs().reduce(ee.Reducer.mean()).rename("SAR_CHANGE")
        else:
            # A short date window can contain Sentinel-1 scenes in only one
            # half. Keep the analysis usable without fabricating a change signal.
            temporal_change = ee.Image.constant(0).rename("SAR_CHANGE").clip(region)
        metrics_image = ee.Image.cat([
            ndvi.rename("NDVI"),
            ndmi.rename("NDMI"),
            ndre.rename("NDRE"),
            vh_vv_ratio,
            temporal_change,
        ])
=======
        # Fallback Check: Both sensors unavailable?
        if s2_count == 0 and s1_count == 0:
            raise ValueError("No Sentinel-2 or Sentinel-1 imagery is available for this farm and date window.")

        # Determine Analysis Modality
        if s2_count > 0 and s1_count > 0:
            modality = "COMBINED"
        elif s2_count > 0:
            modality = "OPTICAL_ONLY"
        else:
            modality = "SAR_ONLY"

        bands_to_combine: list[Any] = []
        band_names: list[str] = []

        # 3. Process Optical (Sentinel-2) if available
        s2_metrics: dict[str, float | None] = {"ndvi_mean": None, "ndmi_mean": None, "ndre_mean": None}
        if s2_count > 0:
            # Mask individual scenes before temporal compositing (P0-2)
            s2_masked = s2.map(self._mask_s2_sr)
            s2_image = s2_masked.median().clip(region)

            ndvi = s2_image.normalizedDifference(["B8", "B4"]).rename("NDVI")
            ndmi = s2_image.normalizedDifference(["B8", "B11"]).rename("NDMI")
            # NDRE using narrow NIR B8A (P1-5)
            ndre = s2_image.normalizedDifference(["B8A", "B5"]).rename("NDRE")

            s2_ndvi = ndvi.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=10, maxPixels=1e9)
            s2_ndmi = ndmi.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=10, maxPixels=1e9)
            s2_ndre = ndre.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=10, maxPixels=1e9)

            s2_metrics["ndvi_mean"] = _extract_numeric_value(s2_ndvi.getInfo())
            s2_metrics["ndmi_mean"] = _extract_numeric_value(s2_ndmi.getInfo())
            s2_metrics["ndre_mean"] = _extract_numeric_value(s2_ndre.getInfo())

            bands_to_combine.extend([ndvi, ndmi, ndre])
            band_names.extend(["NDVI", "NDMI", "NDRE"])

        # 4. Process SAR (Sentinel-1) if available
        s1_metrics: dict[str, float | None] = {
            "vv_mean_db": None,
            "vh_mean_db": None,
            "vh_vv_ratio_mean": None,
            "temporal_change": None,
        }
        sar_change_available = False
        temporal_change_image: Any = None

        if s1_count > 0:
            # Apply 3x3 speckle filter to each S1 scene before compositing (P1-4)
            s1_speckled = s1.map(self._apply_sar_speckle_filter)
            s1_mean = s1_speckled.median().clip(region)
            vv = s1_mean.select("VV")
            vh = s1_mean.select("VH")
            vh_vv_ratio = (
                ee.Image.constant(10)
                .pow(vh.divide(10))
                .divide(ee.Image.constant(10).pow(vv.divide(10)))
                .rename("VH_VV_RATIO")
            )

            vv_stats = vv.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=10, maxPixels=1e9)
            vh_stats = vh.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=10, maxPixels=1e9)
            vh_vv_ratio_stats = vh_vv_ratio.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=10, maxPixels=1e9)

            s1_metrics["vv_mean_db"] = _extract_numeric_value(vv_stats.getInfo())
            s1_metrics["vh_mean_db"] = _extract_numeric_value(vh_stats.getInfo())
            s1_metrics["vh_vv_ratio_mean"] = _extract_numeric_value(vh_vv_ratio_stats.getInfo())

            bands_to_combine.append(vh_vv_ratio)
            band_names.append("VH_VV_RATIO")

            # Temporal Change Analysis: 12-24 day baseline with orbit consistency (P1-2, P1-3)
            s1_baseline = (
                ee.ImageCollection("COPERNICUS/S1_GRD")
                .filterBounds(region)
                .filterDate(end.advance(-24, "day"), end)
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            )
            recent_s1 = s1_baseline.filterDate(end.advance(-12, "day"), end)
            prev_s1 = s1_baseline.filterDate(end.advance(-24, "day"), end.advance(-12, "day"))

            recent_count = int(recent_s1.size().getInfo())
            if recent_count > 0:
                # Find orbit pass of most recent observation
                latest_image = recent_s1.sort("system:time_start", False).first()
                pass_val = latest_image.get("orbitProperties_pass").getInfo()
                if pass_val:
                    recent_matched = recent_s1.filter(ee.Filter.eq("orbitProperties_pass", pass_val))
                    prev_matched = prev_s1.filter(ee.Filter.eq("orbitProperties_pass", pass_val))
                    if int(recent_matched.size().getInfo()) > 0 and int(prev_matched.size().getInfo()) > 0:
                        recent_img = recent_matched.map(self._apply_sar_speckle_filter).median().select(["VV", "VH"])
                        prev_img = prev_matched.map(self._apply_sar_speckle_filter).median().select(["VV", "VH"])
                        temporal_change_image = recent_img.subtract(prev_img).abs().reduce(ee.Reducer.mean()).rename("SAR_CHANGE")
                        tc_stats = temporal_change_image.reduceRegion(
                            reducer=ee.Reducer.mean(),
                            geometry=region,
                            scale=10,
                            maxPixels=1e9,
                        ).getInfo()
                        s1_metrics["temporal_change"] = _extract_numeric_value(tc_stats)
                        sar_change_available = True
                        bands_to_combine.append(temporal_change_image)
                        band_names.append("SAR_CHANGE")

        # 5. Build Combined Anomaly Image
        metrics_image = ee.Image.cat(bands_to_combine)
>>>>>>> dcc12f8cbf742f8390e1320bd097f65d9f314692
        stats = metrics_image.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
            geometry=region,
            scale=10,
            maxPixels=1e9,
        )
<<<<<<< HEAD
        mean_image = ee.Image.constant([stats.get("NDVI_mean"), stats.get("NDMI_mean"), stats.get("NDRE_mean"), stats.get("VH_VV_RATIO_mean"), stats.get("SAR_CHANGE_mean")]).rename(metrics_image.bandNames())
        std_image = ee.Image.constant([stats.get("NDVI_stdDev"), stats.get("NDMI_stdDev"), stats.get("NDRE_stdDev"), stats.get("VH_VV_RATIO_stdDev"), stats.get("SAR_CHANGE_stdDev")]).rename(metrics_image.bandNames())
        low_health = mean_image.subtract(metrics_image).divide(std_image.max(0.0001)).max(0)
        anomaly_image = low_health.select(["NDVI", "NDMI", "NDRE", "VH_VV_RATIO"]).reduce(ee.Reducer.mean()).add(
            metrics_image.select("SAR_CHANGE").subtract(mean_image.select("SAR_CHANGE")).abs().divide(std_image.select("SAR_CHANGE").max(0.0001))
        ).divide(2).rename("ANOMALY")
=======

        mean_constants = [stats.get(f"{name}_mean") for name in band_names]
        std_constants = [stats.get(f"{name}_stdDev") for name in band_names]
        mean_image = ee.Image.constant(mean_constants).rename(band_names)
        std_image = ee.Image.constant(std_constants).rename(band_names)

        low_health = mean_image.subtract(metrics_image).divide(std_image.max(0.0001))

        static_bands = [b for b in band_names if b != "SAR_CHANGE"]
        if static_bands and sar_change_available and temporal_change_image is not None:
            anomaly_image = (
                low_health.select(static_bands).reduce(ee.Reducer.mean())
                .add(
                    metrics_image.select("SAR_CHANGE")
                    .subtract(mean_image.select("SAR_CHANGE"))
                    .abs()
                    .divide(std_image.select("SAR_CHANGE").max(0.0001))
                )
                .divide(2)
                .rename("ANOMALY")
            )
        elif static_bands:
            anomaly_image = low_health.select(static_bands).reduce(ee.Reducer.mean()).rename("ANOMALY")
        elif sar_change_available and temporal_change_image is not None:
            anomaly_image = (
                metrics_image.select("SAR_CHANGE")
                .subtract(mean_image.select("SAR_CHANGE"))
                .abs()
                .divide(std_image.select("SAR_CHANGE").max(0.0001))
                .rename("ANOMALY")
            )
        else:
            anomaly_image = ee.Image.constant(0.0).rename("ANOMALY")

        # Batch grid reduction (P0-1)
>>>>>>> dcc12f8cbf742f8390e1320bd097f65d9f314692
        anomaly_grid = _grid_from_anomaly_image(anomaly_image, region, rows=8, cols=8)
        anomaly_points = _sample_anomaly_points(anomaly_image, region)

        return {
            "modality": modality,
            "data_availability": {
                "sentinel_2_sr": s2_count > 0,
                "sentinel_1_grd": s1_count > 0,
                "sar_change_available": sar_change_available,
                "sentinel_2_acquisitions": s2_count,
                "sentinel_1_acquisitions": s1_count,
            },
            "sentinel2": s2_metrics,
            "sentinel1": s1_metrics,
            "anomaly_grid": anomaly_grid,
            "anomaly_points": anomaly_points,
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


def _farm_boundary_coordinates(farm: dict[str, Any]) -> list[tuple[float, float]]:
    if farm["type"] == "polygon":
        return [(float(lon), float(lat)) for lon, lat in farm["coordinates"]]
    latitude = float(farm["latitude"])
    longitude = float(farm["longitude"])
    radius = float(farm["radius_meters"])
    coordinates = []
    for index in range(65):
        angle = 2 * math.pi * index / 64
        coordinates.append((
            longitude + radius * math.cos(angle) / (111_000.0 * max(math.cos(math.radians(latitude)), 1e-6)),
            latitude + radius * math.sin(angle) / 111_000.0,
        ))
    return coordinates


def _point_in_farm(farm: dict[str, Any], longitude: float, latitude: float) -> bool:
    if farm["type"] == "circle":
        dx = (longitude - farm["longitude"]) * 111_000.0 * max(math.cos(math.radians(farm["latitude"])), 1e-6)
        dy = (latitude - farm["latitude"]) * 111_000.0
        return math.hypot(dx, dy) <= farm["radius_meters"]
    inside = False
    polygon = farm["coordinates"]
    for index, (point_lon, point_lat) in enumerate(polygon):
        next_lon, next_lat = polygon[(index + 1) % len(polygon)]
        intersects = (point_lat > latitude) != (next_lat > latitude)
        if intersects and longitude < (next_lon - point_lon) * (latitude - point_lat) / (next_lat - point_lat) + point_lon:
            inside = not inside
    return inside


def _overlay_farm_and_anomalies(
    image_bytes: bytes,
    farm: dict[str, Any],
    anomaly_geojson: dict[str, Any],
    bounds: tuple[float, float, float, float],
    anomaly_points: list[dict[str, Any]] | None = None,
) -> bytes:
    try:
        from io import BytesIO

        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required to render farm and anomaly overlays.") from exc

    image = Image.open(BytesIO(image_bytes)).convert("RGBA")
    draw = ImageDraw.Draw(image)
    lon_min, lon_max, lat_min, lat_max = bounds
    width, height = image.size

    def to_pixel(longitude: float, latitude: float) -> tuple[int, int]:
        x = round((longitude - lon_min) / max(lon_max - lon_min, 1e-12) * (width - 1))
        y = round((lat_max - latitude) / max(lat_max - lat_min, 1e-12) * (height - 1))
        return max(0, min(width - 1, x)), max(0, min(height - 1, y))

    boundary = [to_pixel(lon, lat) for lon, lat in _farm_boundary_coordinates(farm)]
    if len(boundary) >= 2:
        draw.line(boundary, fill=(255, 255, 255, 230), width=8, joint="curve")
        draw.line(boundary, fill=(0, 102, 255, 255), width=4, joint="curve")

    points_to_draw = anomaly_points or []
    if not points_to_draw:
        for feature in anomaly_geojson.get("features", []):
            properties = feature.get("properties", {})
            if properties.get("label") == "anomaly":
                points_to_draw.extend(
                    {"longitude": point[0], "latitude": point[1]}
                    for point in properties.get("cell_centers", [])
                )
    for point in points_to_draw:
        longitude = point.get("longitude")
        latitude = point.get("latitude")
        if longitude is None or latitude is None:
            continue
        x, y = to_pixel(float(longitude), float(latitude))
        draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill=(255, 255, 255, 235))
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(220, 0, 0, 255), outline=(120, 0, 0, 255), width=2)

    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


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
    cell_centers = [
        [
            round(lon_min + (col + 0.5) * lon_step, 7),
            round(lat_min + (row + 0.5) * lat_step, 7),
        ]
        for row, col in cells
    ]
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
        "cell_centers": cell_centers,
    }
    if geometry is not None:
        result["geometry"] = geometry
    return result


def _farm_health_label(
    anomalous_pixels: int,
    reachable_pixels: int,
) -> str:
    if anomalous_pixels == 0:
        return "healthy"
    anomaly_ratio = anomalous_pixels / max(reachable_pixels, 1)
    return "critical" if anomaly_ratio > 0.20 else "abnormal"


def compute_local_abnormality(
    grid: list[list[float | None]],
    percentile_threshold: float = 95.0,
    zscore_threshold: float = 2.5,
    min_cluster_size: int = 3,
    bounds: tuple[float, float, float, float] | None = None,
    farm: dict[str, Any] | None = None,
    sampled_points: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    if not 0 < percentile_threshold <= 100:
        raise ValueError("percentile_threshold must be greater than 0 and at most 100.")
    if zscore_threshold <= 0 or min_cluster_size <= 0:
        raise ValueError("zscore_threshold and min_cluster_size must be positive.")

    flattened = [float(value) for row in grid for value in row if value is not None and math.isfinite(float(value))]
    if not flattened:
        return {
            "farm_local_abnormal_score": 0.0,
            "label": "healthy",
            "connected_clusters": 0,
            "healthy_zones": 0,
            "unreachable_zones": 0,
            "anomalous_pixels": 0,
            "anomalies": [],
            "anomaly_ratio": 0.0,
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
            # The anomaly image is a one-sided low-health score. High values are
            # suspicious; low values are healthy, so they must not be flagged.
            flagged = numeric > cut_point or zscore >= zscore_threshold
            abnormal_row.append(flagged)
            score_row.append(max(0.0, max(numeric - cut_point, 0.0) / max(abs(cut_point), 1e-6), zscore / max(zscore_threshold, 1e-6)))
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

    anomalous_pixels = sum(1 for row in abnormal_mask for value in row if value)
    reachable_pixels = sum(value for row in reachable_mask for value in row)
    anomalies = []
    lon_min, lon_max, lat_min, lat_max = bounds or (None, None, None, None)
    if bounds:
        lat_step = (lat_max - lat_min) / max(len(grid), 1)
        lon_step = (lon_max - lon_min) / max(len(grid[0]), 1)
        if sampled_points:
            for point in sampled_points:
                longitude = float(point["longitude"])
                latitude = float(point["latitude"])
                if not (lon_min <= longitude <= lon_max and lat_min <= latitude <= lat_max):
                    continue
                if farm is not None and not _point_in_farm(farm, longitude, latitude):
                    continue
                row_index = min(len(grid) - 1, max(0, int((latitude - lat_min) / max(lat_step, 1e-12))))
                col_index = min(len(grid[0]) - 1, max(0, int((longitude - lon_min) / max(lon_step, 1e-12))))
                if not abnormal_mask[row_index][col_index]:
                    continue
                anomalies.append({
                    "latitude": round(latitude, 7),
                    "longitude": round(longitude, 7),
                    "score": round(float(point.get("value", score_grid[row_index][col_index])), 6),
                    "grid_row": row_index,
                    "grid_column": col_index,
                })
        else:
            for row_index, row in enumerate(abnormal_mask):
                for col_index, is_anomaly in enumerate(row):
                    if not is_anomaly:
                        continue
                    latitude = lat_min + (row_index + 0.5) * lat_step
                    longitude = lon_min + (col_index + 0.5) * lon_step
                    if farm is not None and not _point_in_farm(farm, longitude, latitude):
                        continue
                    anomalies.append({
                        "latitude": round(latitude, 7),
                        "longitude": round(longitude, 7),
                        "score": round(score_grid[row_index][col_index], 6),
                        "grid_row": row_index,
                        "grid_column": col_index,
                    })
    anomaly_count = len(anomalies) if farm is not None else anomalous_pixels
    anomaly_scores = [
        score_grid[row_index][col_index]
        for row_index, row in enumerate(abnormal_mask)
        for col_index, is_anomaly in enumerate(row)
        if is_anomaly and (
            farm is None
            or _point_in_farm(
                farm,
                (bounds[0] + (col_index + 0.5) * (bounds[1] - bounds[0]) / max(len(grid[0]), 1)),
                (bounds[2] + (row_index + 0.5) * (bounds[3] - bounds[2]) / max(len(grid), 1)),
            )
        )
    ]
    farm_local_abnormal_score = sum(anomaly_scores) / max(len(anomaly_scores), 1)
    anomaly_ratio = anomaly_count / max(reachable_pixels, 1)
    farm_label = _farm_health_label(anomaly_count, reachable_pixels)
    geojson = {"type": "FeatureCollection", "features": features}
    return {
        "farm_local_abnormal_score": round(farm_local_abnormal_score, 6),
        "label": farm_label,
        "connected_clusters": len(anomaly_clusters),
        "healthy_zones": len(healthy_clusters),
        "unreachable_zones": len(unreachable_clusters),
        "anomalous_pixels": anomalous_pixels,
        "anomaly_ratio": round(anomaly_ratio, 6),
        "anomalies": anomalies,
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
            farm=farm,
            sampled_points=metrics.get("anomaly_points"),
        )
        farm_label = {"healthy": "Healthy", "abnormal": "Abnormal", "critical": "Critical"}[field_scores["label"]]
        if farm["type"] == "polygon":
            farm_coordinates: dict[str, Any] = {
                "type": "Polygon",
                "coordinates": [farm["coordinates"]],
            }
        else:
            farm_coordinates = {
                "type": "Circle",
                "center": [farm["longitude"], farm["latitude"]],
                "radius_meters": farm["radius_meters"],
            }
        mean_metrics = {
            "ndvi": metrics.get("sentinel2", {}).get("ndvi_mean"),
            "ndmi": metrics.get("sentinel2", {}).get("ndmi_mean"),
            "ndre": metrics.get("sentinel2", {}).get("ndre_mean"),
            "vv_db": metrics.get("sentinel1", {}).get("vv_mean_db"),
            "vh_db": metrics.get("sentinel1", {}).get("vh_mean_db"),
            "vh_vv_ratio": metrics.get("sentinel1", {}).get("vh_vv_ratio_mean"),
            "sar_temporal_change": metrics.get("sentinel1", {}).get("temporal_change"),
        }
        summary = {
            "area_m2": round(farm["area_m2"], 4),
            "date_window_days": (end_date - start_date).days + 1,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "label": farm_label,
            "farm_label": farm_label,
            "modality": metrics.get("modality", "COMBINED"),
            "farm_local_abnormal_score": field_scores["farm_local_abnormal_score"],
            "connected_clusters": field_scores["connected_clusters"],
            "healthy_zones": field_scores["healthy_zones"],
            "unreachable_zones": field_scores["unreachable_zones"],
            "anomalous_pixels": field_scores["anomalous_pixels"],
            "anomaly_ratio": field_scores["anomaly_ratio"],
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
<<<<<<< HEAD
        region_image = self.gee.fetch_region_image(
            farm,
            start_date,
            end_date,
            anomaly_geojson=field_scores["geojson"],
            anomaly_points=field_scores["anomalies"],
        )
=======
        try:
            region_image = self.gee.fetch_region_image(farm, start_date, end_date)
        except Exception:
            region_image = None
>>>>>>> dcc12f8cbf742f8390e1320bd097f65d9f314692
        return {
            "label": farm_label,
            "farm": farm_coordinates,
            "mean_metrics": mean_metrics,
            "anomalies": field_scores["anomalies"],
            "farm_local_abnormal_score": field_scores["farm_local_abnormal_score"],
            "summary": summary,
            "data_availability": metrics.get("data_availability", {}),
<<<<<<< HEAD
            "region_image_path": region_image,
            "region_image_mime_type": "image/png",
=======
            "region_image": region_image,
            "region_image_mime_type": "image/png" if region_image else None,
            "modality": metrics.get("modality", "COMBINED"),
>>>>>>> dcc12f8cbf742f8390e1320bd097f65d9f314692
        }


def get_gee_service() -> FarmAnalysisService:
    return FarmAnalysisService()
