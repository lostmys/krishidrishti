"""
HTTP client used by frontend.py / dashboard.py to talk to Cases API & Core AI.

Env:
  KRISHI_CASES_API=http://127.0.0.1:8002   (default)
  IMAGE_AI_URL=http://127.0.0.1:8001       (default)
"""

from __future__ import annotations

import os
from typing import Any

import requests

CASES_API_BASE = os.getenv("KRISHI_CASES_API", "http://127.0.0.1:8002").rstrip("/")
IMAGE_AI_BASE = os.getenv("IMAGE_AI_URL", "http://127.0.0.1:8001").rstrip("/")
TIMEOUT = float(os.getenv("KRISHI_API_TIMEOUT", "10"))


class CasesAPIError(Exception):
    pass


def _url(path: str) -> str:
    if not path.startswith("/"):
        path = "/" + path
    return f"{CASES_API_BASE}{path}"


def health() -> dict[str, Any]:
    r = requests.get(_url("/health"), timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def image_ai_health() -> dict[str, Any]:
    """Check readiness of Core AI / ML service (Port 8001)."""
    r = requests.get(f"{IMAGE_AI_BASE}/health", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_system_status() -> dict[str, Any]:
    """Retrieve comprehensive truthful subsystem status from Core AI (Port 8001)."""
    try:
        r = requests.get(f"{IMAGE_AI_BASE}/system/status", timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def get_available_crops() -> list[str]:
    """Retrieve supported crop slugs from Core AI."""
    try:
        r = requests.get(f"{IMAGE_AI_BASE}/crops", timeout=TIMEOUT)
        r.raise_for_status()
        return r.json().get("crops", ["tomato", "soyabean", "cotton"])
    except Exception:
        return ["tomato", "soyabean", "cotton"]


def diagnose_image(
    crop: str,
    image_bytes: bytes,
    filename: str = "leaf.jpg",
    language: str = "en",
) -> dict[str, Any]:
    """
    Send leaf image to real production MobileNetV3 model endpoint:
    POST http://127.0.0.1:8001/diagnose/{crop}?language={language}
    """
    url = f"{IMAGE_AI_BASE}/diagnose/{crop.strip().lower()}"
    files = {"image": (filename, image_bytes, "image/jpeg")}
    params = {"language": language}
    r = requests.post(url, files=files, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def analyze_satellite(
    lat: float,
    lon: float,
    radius_meters: float = 100.0,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """
    Send coordinates to Core AI satellite analysis endpoint:
    POST http://127.0.0.1:8001/analyze-farm
    """
    payload: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "radius_meters": radius_meters,
    }
    if start_date and end_date:
        payload["start_date"] = start_date
        payload["end_date"] = end_date
    url = f"{IMAGE_AI_BASE}/analyze-farm"
    r = requests.post(url, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_cases(
    status: str | None = None,
    district: str | None = None,
) -> list[dict[str, Any]]:
    params = {}
    if status:
        params["status"] = status
    if district:
        params["district"] = district
    r = requests.get(_url("/api/cases"), params=params, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data.get("cases", [])


def get_case(case_id: str) -> dict[str, Any]:
    r = requests.get(_url(f"/api/cases/{case_id}"), timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def post_action(case_id: str, action: str, officer_id: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"action": action}
    if officer_id:
        payload["officer_id"] = officer_id
    r = requests.post(
        _url(f"/api/cases/{case_id}/action"),
        json=payload,
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def ingest_case(case_dict: dict[str, Any]) -> dict[str, Any]:
    """Persist/update a case in Cases API."""
    payload = {"case": case_dict}
    r = requests.post(_url("/api/cases/ingest"), json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def reset_demo() -> dict[str, Any]:
    r = requests.post(_url("/api/cases/demo/reset"), timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def ingest_demo_case() -> dict[str, Any]:
    r = requests.post(_url("/api/cases/demo/new"), timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def try_live_cases() -> tuple[bool, list[dict[str, Any]] | str]:
    """
    Returns (ok, cases_or_error_message).
    ok=False → caller should use mock data.
    """
    try:
        cases = get_cases()
        return True, cases
    except Exception as exc:
        return False, str(exc)

