from __future__ import annotations

import os
from typing import Any

from krishidrishti_ai.gee_api import (
    DEFAULT_POINT_RADIUS_METERS,
    _normalize_payload,
    _resolve_window,
    get_gee_service,
)


class SatelliteService:
    """Resilient adapter for Google Earth Engine (GEE) farm analysis.

    Ensures that KrishiDrishti applications and case workflows never fail
    due to missing Earth Engine credentials or transient satellite query errors.
    """

    DEFAULT_RADIUS_METERS: float = DEFAULT_POINT_RADIUS_METERS

    @staticmethod
    def coordinates_to_payload(
        latitude: float,
        longitude: float,
        radius_meters: float = DEFAULT_POINT_RADIUS_METERS,
    ) -> dict[str, Any]:
        """Generate a point/radius payload for satellite screening.

        Note: Point radius (default 100m ~ 7.76 acres) is an approximate farm-area proxy
        for smallholder plots when cadastral boundaries are unavailable.
        It does NOT represent exact property boundaries.
        Explicit GeoJSON parcel geometry should be preferred whenever available.
        """
        return {
            "latitude": latitude,
            "longitude": longitude,
            "radius_meters": radius_meters,
        }

    @staticmethod
    def _coordinates_to_geojson(
        latitude: float,
        longitude: float,
        radius_meters: float = DEFAULT_POINT_RADIUS_METERS,
    ) -> dict[str, Any]:
        """Convert a center coordinate and radius into a circular GeoJSON polygon proxy.

        Note: Point radius is an approximate farm proxy (default 100m ~ 7.76 acres).
        It does NOT represent exact cadastral boundary information.
        Explicit GeoJSON parcel geometry should be preferred whenever available.
        """
        import math

        points = []
        num_vertices = 32
        for i in range(num_vertices):
            angle = 2.0 * math.pi * i / num_vertices
            dx = radius_meters * math.cos(angle)
            dy = radius_meters * math.sin(angle)
            lat = latitude + dy / 111_000.0
            lon = longitude + dx / (111_000.0 * max(math.cos(math.radians(latitude)), 1e-6))
            points.append([round(lon, 7), round(lat, 7)])
        points.append(points[0])
        return {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [points],
            },
            "properties": {
                "proxy_radius_meters": radius_meters,
                "is_approximate_proxy": True,
            },
        }

    @staticmethod
    def is_configured() -> bool:
        """Return True if Earth Engine environment credentials are present."""
        has_email = bool(os.getenv("GEE_SERVICE_ACCOUNT_EMAIL"))
        has_key = bool(os.getenv("GEE_PRIVATE_KEY") or os.getenv("GEE_PRIVATE_KEY_PATH"))
        has_project = bool(os.getenv("GEE_PROJECT_ID"))
        return has_email and has_key and has_project

    @classmethod
    def get_status(cls) -> dict[str, Any]:
        """Report truthful configuration and connectivity status of GEE."""
        configured = cls.is_configured()
        if not configured:
            return {
                "status": "DEMO / PRECOMPUTED",
                "badge": "🟡 DEMO / PRECOMPUTED",
                "is_live": False,
                "provider": "Google Earth Engine (Precomputed Benchmark Fallback)",
                "configured": False,
                "authenticated": False,
                "reason": "GEE credentials not configured. Using precomputed satellite anomaly indices.",
            }
        try:
            service = get_gee_service()
            service.gee.initialize()
            return {
                "status": "LIVE",
                "badge": "🟢 LIVE",
                "is_live": True,
                "provider": "Google Earth Engine (Live Sentinel-2 / Sentinel-1)",
                "configured": True,
                "authenticated": True,
                "project_id": os.getenv("GEE_PROJECT_ID"),
                "service_account": os.getenv("GEE_SERVICE_ACCOUNT_EMAIL"),
            }
        except Exception as exc:
            return {
                "status": "ERROR",
                "badge": "🔴 ERROR",
                "is_live": False,
                "provider": "Google Earth Engine",
                "configured": True,
                "authenticated": False,
                "error": str(exc),
            }

    def analyze_farm_safely(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Attempt farm anomaly analysis, degrading gracefully if GEE is unavailable.

        Returns a standardized payload with 'status' ('SUCCESS', 'UNAVAILABLE', 'INVALID_GEOMETRY', or 'ERROR').
        """
        # 1. Quick check for coordinates or geometry
        has_geom = bool(payload.get("geometry") or payload.get("polygon"))
        has_coords = payload.get("latitude") is not None and payload.get("longitude") is not None
        if not has_geom and not has_coords:
            return {
                "status": "NOT_REQUESTED",
                "available": False,
                "reason": "No farm geometry or coordinates provided in request.",
            }

        # 2. Check credentials configuration
        if not self.is_configured():
            return {
                "status": "UNAVAILABLE",
                "available": False,
                "reason": "Google Earth Engine service account credentials are not configured.",
                "note": "To enable satellite screening, set GEE_SERVICE_ACCOUNT_EMAIL, GEE_PROJECT_ID, and GEE_PRIVATE_KEY in .env.",
            }

        # 3. Attempt execution
        try:
            service = get_gee_service()
            result = service.analyze(payload)
            # Ensure thumbnail image is present
            if "region_image" not in result and hasattr(service, "gee"):
                try:
                    normalized = _normalize_payload(payload)
                    start_date, end_date = _resolve_window(payload)
                    result["region_image"] = service.gee.fetch_region_image(normalized, start_date, end_date)
                    result["region_image_mime_type"] = "image/png"
                except Exception:
                    pass
            result["status"] = "SUCCESS"
            result["available"] = True
            if "modality" not in result:
                result["modality"] = result.get("summary", {}).get("modality", "COMBINED")
            return result
        except ValueError as exc:
            msg = str(exc)
            if "No Sentinel-2 or Sentinel-1 imagery is available" in msg or "No imagery is available" in msg:
                return {
                    "status": "UNAVAILABLE",
                    "available": False,
                    "reason": msg,
                }
            return {
                "status": "INVALID_GEOMETRY",
                "available": False,
                "reason": msg,
            }
        except RuntimeError as exc:
            return {
                "status": "UNAVAILABLE",
                "available": False,
                "reason": str(exc),
            }
        except Exception as exc:
            return {
                "status": "ERROR",
                "available": False,
                "reason": f"Satellite analysis failed: {exc}",
            }
