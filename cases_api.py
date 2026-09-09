"""
KrishiDrishti — Cases API (Officer Dashboard backend)

WHY THIS FILE EXISTS
--------------------
src/krishidrishti_ai/api.py          → Image diagnose + GEE /analyze-farm  (Vaibhav/Oishik)
src/krishidrishti_ai/api_vaibhav.py  → Crop-aware /diagnose/{crop}         (Vaibhav)

Neither stores officer cases or expert actions.
This service is the §10 case store your frontend.py / dashboard reads & writes.

Run (from repo root, after placing this file):
  uvicorn cases_api:app --host 0.0.0.0 --port 8000 --reload

Optional env:
  CASES_API_PORT=8000
  IMAGE_AI_URL=http://127.0.0.1:8001    # if Vaibhav API runs elsewhere
  GEE_API_URL=http://127.0.0.1:8001     # if combined api.py hosts GEE
"""

from __future__ import annotations

import copy
import os
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="KrishiDrishti Cases API",
    description="Officer/Expert dashboard case store + expert actions (SIH 26131)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_lock = Lock()

# Upstream module URLs (optional — used only by helper proxy routes)
IMAGE_AI_URL = os.getenv("IMAGE_AI_URL", "http://127.0.0.1:8001").rstrip("/")
GEE_API_URL = os.getenv("GEE_API_URL", "http://127.0.0.1:8001").rstrip("/")


# ---------------------------------------------------------------------------
# Seed data — CASE-001 matches handbook demo journey
# ---------------------------------------------------------------------------

def _seed_cases() -> dict[str, dict[str, Any]]:
    now = datetime.now()
    t0 = now - timedelta(hours=6)

    def iso(dt: datetime) -> str:
        return dt.isoformat(timespec="seconds")

    case_001 = {
        "case_id": "CASE-001",
        "field_id": "FIELD-001",
        "farmer_name": "Ramesh Patil",
        "crop": "Soybean",
        "district": "Yavatmal",
        "taluka": "Pusad",
        "village": "Digras Wadi",
        "lat": 19.908,
        "lon": 77.564,
        "status": "PENDING",
        "advisory_sent": False,
        "risk_level": "HIGH",
        "risk_score": 81,
        "image_prediction": "Cercospora Leaf Blight",
        "image_confidence": 0.87,
        "image_quality": "GOOD",
        "ndvi_current": 0.41,
        "ndvi_historical": 0.62,
        "anomaly_score": 0.78,
        "anomaly_detected": True,
        "weather_risk": 0.72,
        "nearby_reports": 3,
        "needs_expert": True,
        "reported_at": iso(t0),
        "satellite": {
            "anomaly_detected": True,
            "anomaly_score": 0.78,
            "centroid": {"lat": 19.908, "lng": 77.564},
            "zone": "north-east",
            "ndvi_current": 0.41,
            "ndvi_historical": 0.62,
        },
        "image_ai": {
            "crop": "soyabean",
            "condition": "Cercospora Leaf Blight",
            "confidence": 0.87,
            "quality": "GOOD",
            "diseaseDetected": True,
        },
        "fusion": {
            "risk": "HIGH",
            "score": 0.81,
            "expert_required": True,
            "reasons": [
                "Disease detected from crop image",
                "Satellite vegetation anomaly detected",
                "Weather conditions indicate elevated risk",
                "Similar nearby reports found",
            ],
        },
        "farmer_report": {
            "received": True,
            "channel": "WhatsApp voice",
            "language": "mr",
            "summary": "पाने पिवळी पडत आहेत, काही ठिकाणी ठिपके दिसत आहेत.",
        },
        "reasons": [
            "Disease detected from crop image",
            "Satellite vegetation anomaly detected",
            "Weather conditions indicate elevated risk",
            "Similar nearby reports found",
        ],
        "contribution": {
            "Image AI (40%)": 34.8,
            "Satellite (35%)": 27.3,
            "Outbreak context (15%)": 10.8,
            "Farmer urgency (10%)": 8.1,
        },
        "log": [
            f"{t0.strftime('%H:%M')} — Satellite anomaly flagged",
            f"{t0.strftime('%H:%M')} — WhatsApp alert sent to farmer",
            f"{(t0 + timedelta(minutes=12)).strftime('%H:%M')} — Farmer voice report received",
            f"{(t0 + timedelta(minutes=18)).strftime('%H:%M')} — Crop image received",
            f"{(t0 + timedelta(minutes=19)).strftime('%H:%M')} — Image AI → Cercospora Leaf Blight (87%)",
            f"{(t0 + timedelta(minutes=20)).strftime('%H:%M')} — Fusion risk = HIGH (0.81)",
        ],
        "whatsapp_advisory_mr": (
            "⚠️ सूचना: आपल्या सोयाबीन शेतात (Digras Wadi) असामान्य बदल व रोगाची लक्षणे आढळली आहेत. "
            "अधिक माहितीसाठी कृषी सहाय्यकांशी संपर्क साधा। — KrishiDrishti"
        ),
        "whatsapp_advisory_en": (
            "⚠️ Alert: Unusual stress and disease symptoms detected in your Soybean field (Digras Wadi). "
            "Please contact your agriculture assistant. — KrishiDrishti"
        ),
        # Demo visuals (senior: use demo pics — not live GEE/WhatsApp fetch)
        "satellite_image": "assets/demo/case001_satellite.jpg",
        "crop_image": "assets/demo/case001_leaf.jpg",
        "images_source": "DEMO",
    }

    # A few extra cases so the map/KPIs aren't empty
    extras = [
        {
            "case_id": "CASE-002",
            "field_id": "FIELD-002",
            "farmer_name": "Suresh Jadhav",
            "crop": "Soybean",
            "district": "Yavatmal",
            "taluka": "Pusad",
            "village": "Shembalpimpri",
            "lat": 19.86,
            "lon": 77.52,
            "status": "PENDING",
            "advisory_sent": False,
            "risk_level": "MEDIUM",
            "risk_score": 58,
            "image_prediction": "Leaf Spot (Fungal)",
            "image_confidence": 0.74,
            "image_quality": "GOOD",
            "ndvi_current": 0.48,
            "ndvi_historical": 0.61,
            "anomaly_score": 0.55,
            "anomaly_detected": True,
            "weather_risk": 0.50,
            "nearby_reports": 1,
            "needs_expert": True,
            "reported_at": iso(now - timedelta(hours=10)),
            "satellite": {
                "anomaly_detected": True,
                "anomaly_score": 0.55,
                "centroid": {"lat": 19.86, "lng": 77.52},
                "zone": "south",
                "ndvi_current": 0.48,
                "ndvi_historical": 0.61,
            },
            "image_ai": {
                "crop": "soyabean",
                "condition": "Leaf Spot (Fungal)",
                "confidence": 0.74,
                "quality": "GOOD",
                "diseaseDetected": True,
            },
            "fusion": {
                "risk": "MEDIUM",
                "score": 0.58,
                "expert_required": True,
                "reasons": [
                    "Disease detected from crop image",
                    "Satellite vegetation anomaly detected",
                ],
            },
            "farmer_report": {
                "received": True,
                "channel": "WhatsApp voice",
                "language": "mr",
                "summary": "काही पानांवर ठिपके दिसत आहेत.",
            },
            "reasons": [
                "Disease detected from crop image",
                "Satellite vegetation anomaly detected",
            ],
            "contribution": {
                "Image AI (40%)": 29.6,
                "Satellite (25%)": 13.8,
                "Weather (15%)": 7.5,
                "Nearby reports (20%)": 5.0,
            },
            "log": [f"{(now - timedelta(hours=10)).strftime('%H:%M')} — Case created from pipeline"],
            "whatsapp_advisory_mr": "⚠️ सूचना: Shembalpimpri — तपासणी आवश्यक. — KrishiDrishti",
            "whatsapp_advisory_en": "⚠️ Alert: Inspection needed at Shembalpimpri. — KrishiDrishti",
            "satellite_image": None,
            "crop_image": None,
            "images_source": "PENDING",
        },
        {
            "case_id": "CASE-003",
            "field_id": "FIELD-003",
            "farmer_name": "Vandana Rathod",
            "crop": "Cotton",
            "district": "Amravati",
            "taluka": "Achalpur",
            "village": "Paratwada",
            "lat": 21.27,
            "lon": 77.51,
            "status": "PENDING",
            "advisory_sent": False,
            "risk_level": "LOW",
            "risk_score": 28,
            "image_prediction": "Healthy",
            "image_confidence": 0.91,
            "image_quality": "GOOD",
            "ndvi_current": 0.64,
            "ndvi_historical": 0.66,
            "anomaly_score": 0.22,
            "anomaly_detected": False,
            "weather_risk": 0.30,
            "nearby_reports": 0,
            "needs_expert": False,
            "reported_at": iso(now - timedelta(hours=20)),
            "satellite": {
                "anomaly_detected": False,
                "anomaly_score": 0.22,
                "centroid": {"lat": 21.27, "lng": 77.51},
                "zone": "east",
                "ndvi_current": 0.64,
                "ndvi_historical": 0.66,
            },
            "image_ai": {
                "crop": "cotton",
                "condition": "Healthy",
                "confidence": 0.91,
                "quality": "GOOD",
                "diseaseDetected": False,
            },
            "fusion": {
                "risk": "LOW",
                "score": 0.28,
                "expert_required": False,
                "reasons": ["No strong multi-signal risk"],
            },
            "farmer_report": {
                "received": True,
                "channel": "WhatsApp text",
                "language": "en",
                "summary": "Crop looks mostly fine; routine check.",
            },
            "reasons": ["No strong multi-signal risk"],
            "contribution": {
                "Image AI (40%)": 9.0,
                "Satellite (25%)": 5.5,
                "Weather (15%)": 4.5,
                "Nearby reports (20%)": 0.0,
            },
            "log": [f"{(now - timedelta(hours=20)).strftime('%H:%M')} — Case created"],
            "whatsapp_advisory_mr": "माहिती: सध्या कमी जोखीम. — KrishiDrishti",
            "whatsapp_advisory_en": "Info: Low risk currently. — KrishiDrishti",
            "satellite_image": None,
            "crop_image": None,
            "images_source": "PENDING",
        },
    ]

    store = {"CASE-001": case_001}
    for c in extras:
        store[c["case_id"]] = c
    return store


CASES: dict[str, dict[str, Any]] = _seed_cases()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ActionBody(BaseModel):
    action: Literal["CONFIRM", "REJECT", "FIELD_VISIT_REQUIRED"]
    officer_id: str | None = None
    note: str | None = None


class CaseIngest(BaseModel):
    """Optional: other modules push a fused case here."""
    case: dict[str, Any] = Field(..., description="Full case dict matching dashboard contract")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_reported_at(case: dict[str, Any]) -> datetime:
    raw = case.get("reported_at")
    if isinstance(raw, datetime):
        return raw
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw.replace("Z", ""))
        except ValueError:
            pass
    return datetime.now()


def _public_case(case: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy (reported_at as ISO string)."""
    out = copy.deepcopy(case)
    if isinstance(out.get("reported_at"), datetime):
        out["reported_at"] = out["reported_at"].isoformat(timespec="seconds")
    return out


# ---------------------------------------------------------------------------
# Routes — what frontend.py should call
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "READY",
        "service": "cases-api",
        "cases": len(CASES),
        "image_ai_url": IMAGE_AI_URL,
        "gee_api_url": GEE_API_URL,
    }


@app.get("/api/cases")
def list_cases(
    status: str | None = None,
    district: str | None = None,
    risk_level: str | None = None,
) -> dict[str, Any]:
    with _lock:
        items = [_public_case(c) for c in CASES.values()]
    if status:
        items = [c for c in items if c.get("status") == status]
    if district:
        items = [c for c in items if c.get("district") == district]
    if risk_level:
        items = [c for c in items if c.get("risk_level") == risk_level]
    items.sort(key=lambda c: c.get("risk_score", 0), reverse=True)
    return {"cases": items, "count": len(items)}


@app.get("/api/cases/{case_id}")
def get_case(case_id: str) -> dict[str, Any]:
    with _lock:
        if case_id not in CASES:
            raise HTTPException(status_code=404, detail=f"Unknown case_id {case_id}")
        return _public_case(CASES[case_id])


@app.post("/api/cases/{case_id}/action")
def case_action(case_id: str, body: ActionBody) -> dict[str, Any]:
    """
    Expert actions (handbook §08):
      CONFIRM              → VERIFIED + advisory_sent=True
      REJECT               → REJECTED
      FIELD_VISIT_REQUIRED → FIELD_VISIT_REQUIRED
    """
    with _lock:
        if case_id not in CASES:
            raise HTTPException(status_code=404, detail=f"Unknown case_id {case_id}")
        case = CASES[case_id]
        ts = datetime.now().strftime("%H:%M")

        if body.action == "CONFIRM":
            case["status"] = "VERIFIED"
            case["advisory_sent"] = True
            case.setdefault("log", []).append(f"{ts} — Expert CONFIRM → VERIFIED")
            case["log"].append(f"{ts} — Farmer advisory SENT (status flag; WhatsApp hook optional)")
        elif body.action == "REJECT":
            case["status"] = "REJECTED"
            case.setdefault("log", []).append(f"{ts} — Expert REJECT")
        else:
            case["status"] = "FIELD_VISIT_REQUIRED"
            case.setdefault("log", []).append(f"{ts} — Expert FIELD VISIT REQUIRED")

        if body.note:
            case["log"].append(f"{ts} — Note: {body.note}")
        if body.officer_id:
            case["log"].append(f"{ts} — Officer: {body.officer_id}")

        return _public_case(case)


@app.post("/api/cases/ingest")
def ingest_case(body: CaseIngest) -> dict[str, Any]:
    """Fusion / WhatsApp / other modules can POST a full case here."""
    case = body.case
    cid = case.get("case_id")
    if not cid:
        raise HTTPException(status_code=400, detail="case.case_id is required")
    with _lock:
        CASES[cid] = copy.deepcopy(case)
        return _public_case(CASES[cid])


@app.post("/api/cases/demo/reset")
def reset_demo() -> dict[str, Any]:
    global CASES
    with _lock:
        CASES = _seed_cases()
    return {"ok": True, "count": len(CASES)}


@app.post("/api/cases/demo/new")
def demo_new_case() -> dict[str, Any]:
    """Sidebar 'ingest mock' equivalent on the server."""
    with _lock:
        nums = []
        for cid in CASES:
            try:
                nums.append(int(cid.split("-")[-1]))
            except ValueError:
                pass
        n = (max(nums) if nums else 0) + 1
        cid = f"CASE-{n:03d}"
        base = copy.deepcopy(CASES.get("CASE-002") or next(iter(CASES.values())))
        base["case_id"] = cid
        base["field_id"] = f"FIELD-{n:03d}"
        base["status"] = "PENDING"
        base["advisory_sent"] = False
        base["reported_at"] = datetime.now().isoformat(timespec="seconds")
        base["farmer_name"] = f"Demo Farmer {n}"
        base.setdefault("log", []).append(f"{datetime.now().strftime('%H:%M')} — Demo case ingested")
        CASES[cid] = base
        return _public_case(base)


# ---------------------------------------------------------------------------
# Optional proxies — call Vaibhav / GEE APIs without mixing ports in the UI
# ---------------------------------------------------------------------------

@app.get("/api/upstream/image-health")
def upstream_image_health() -> dict[str, Any]:
    try:
        r = httpx.get(f"{IMAGE_AI_URL}/health", timeout=5.0)
        return {"ok": r.is_success, "status_code": r.status_code, "body": r.json()}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "url": f"{IMAGE_AI_URL}/health"}


@app.get("/api/upstream/gee-ping")
def upstream_gee_ping() -> dict[str, Any]:
    """Best-effort: hits same host health (GEE lives on api.py in your monorepo)."""
    try:
        r = httpx.get(f"{GEE_API_URL}/health", timeout=5.0)
        return {"ok": r.is_success, "status_code": r.status_code, "body": r.json()}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "url": f"{GEE_API_URL}/health"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("CASES_API_PORT", "8000"))
    uvicorn.run("cases_api:app", host="0.0.0.0", port=port, reload=True)
