"""
HTTP client used by frontend.py / dashboard.py to talk to Cases API.

Env:
  KRISHI_CASES_API=http://127.0.0.1:8000   (default)
"""

from __future__ import annotations

import os
from typing import Any

import requests

CASES_API_BASE = os.getenv("KRISHI_CASES_API", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = float(os.getenv("KRISHI_API_TIMEOUT", "8"))


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
