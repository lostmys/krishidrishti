from __future__ import annotations

import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse

from krishidrishti_ai.config import load_config
from krishidrishti_ai.gee_api import _normalize_payload, _resolve_window, get_gee_service
from krishidrishti_ai.services.cases import CaseService
from krishidrishti_ai.services.fusion import RiskFusionService
from krishidrishti_ai.services.inference import DiagnosisService
from krishidrishti_ai.services.knowledge import DiseaseKnowledgeService
from krishidrishti_ai.services.notifications import LocalNotificationAdapter, format_case_alert
from krishidrishti_ai.services.registry import CropRegistry
from krishidrishti_ai.services.satellite import DEFAULT_POINT_RADIUS_METERS, SatelliteService

app = FastAPI(title="KrishiDrishti AI Agricultural Platform")

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB limit

# ---------------------------------------------------------------------------
# Project root resolver
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Singletons & Service Factories
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_registry() -> CropRegistry:
    """Build and cache the CropRegistry from default.yaml crops block."""
    config_path = os.getenv(
        "KRISHIDRISHTI_CONFIG",
        str(_PROJECT_ROOT / "configs" / "default.yaml"),
    )
    config = load_config(config_path)
    crops_raw: dict[str, str] = config.get("crops", {})
    crops_resolved = {
        slug: (_PROJECT_ROOT / path).resolve()
        for slug, path in crops_raw.items()
    }
    return CropRegistry(crops_resolved)


@lru_cache(maxsize=1)
def _get_case_service() -> CaseService:
    return CaseService()


@lru_cache(maxsize=1)
def _get_satellite_service() -> SatelliteService:
    return SatelliteService()


@lru_cache(maxsize=1)
def _get_fusion_service() -> RiskFusionService:
    return RiskFusionService()


@lru_cache(maxsize=1)
def _get_notification_service() -> LocalNotificationAdapter:
    return LocalNotificationAdapter()


@lru_cache(maxsize=1)
def _get_knowledge_service() -> DiseaseKnowledgeService:
    return DiseaseKnowledgeService()


@lru_cache(maxsize=1)
def _get_legacy_service() -> DiagnosisService:
    config_path = os.getenv(
        "KRISHIDRISHTI_CONFIG",
        str(_PROJECT_ROOT / "configs" / "default.yaml"),
    )
    return DiagnosisService(load_config(config_path))


def get_service() -> DiagnosisService:
    """Legacy alias used by initial API tests."""
    return _get_legacy_service()


# ---------------------------------------------------------------------------
# Core Endpoints: Web UI, Health, Crops
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the farmer-facing web application UI."""
    html_path = Path(__file__).resolve().parent / "static" / "index.html"
    if html_path.is_file():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>KrishiDrishti AI Platform</h1>", status_code=200)


@app.get("/health")
def health() -> dict:
    """Report readiness for crop models, satellite module, and database."""
    registry = _get_registry()
    ready = [s for s in registry.available_crops if registry.checkpoint_ready(s)]
    not_ready = [s for s in registry.available_crops if not registry.checkpoint_ready(s)]
    if not not_ready:
        overall = "READY"
    elif ready:
        overall = "PARTIAL"
    else:
        overall = "MODEL_NOT_READY"

    sat_service = _get_satellite_service()
    return {
        "status": overall,
        "crops_ready": ready,
        "crops_not_ready": not_ready,
        "satellite_service_configured": sat_service.is_configured(),
        "database": "CONNECTED",
    }


@app.get("/crops")
def list_crops() -> dict:
    """List all registered crop slugs."""
    return {"crops": _get_registry().available_crops}


# ---------------------------------------------------------------------------
# Diagnosis Endpoints
# ---------------------------------------------------------------------------


@app.post("/diagnose/{crop}")
async def diagnose(
    crop: str,
    image: UploadFile = File(...),
    language: str = Query("en", description="Language code (en, hi, mr)"),
) -> dict:
    """Diagnose a plant-disease image for the specified crop with multilingual advice."""
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file")

    registry = _get_registry()
    try:
        service = registry.get(crop)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown crop '{crop}'. Available crops: {registry.available_crops}",
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model checkpoint for '{crop}' is not available",
        ) from exc

    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    if len(payload) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds maximum limit of 10MB")

    try:
        return service.diagnose(payload, language=language)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/diagnose")
async def diagnose_legacy(
    image: UploadFile = File(...),
    language: str = Query("en", description="Language code (en, hi, mr)"),
) -> dict:
    """Deprecated — use POST /diagnose/{crop} instead."""
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file")

    try:
        service = _get_legacy_service()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model checkpoint is not available") from exc

    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    if len(payload) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds maximum limit of 10MB")

    try:
        return service.diagnose(payload, language=language)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Satellite Farm Analysis Endpoint
# ---------------------------------------------------------------------------


@app.post("/analyze-farm")
async def analyze_farm(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Perform standalone Earth Engine satellite anomaly analysis on farm bounds."""
    try:
        service = get_gee_service()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        result = service.analyze(payload)
        if "region_image_path" not in result and hasattr(service, "gee"):
            normalized = _normalize_payload(payload)
            start_date, end_date = _resolve_window(payload)
            result["region_image_path"] = service.gee.fetch_region_image(normalized, start_date, end_date)
            result["region_image_mime_type"] = "image/png"
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Case & Incident Management Endpoints
# ---------------------------------------------------------------------------


@app.post("/cases")
async def report_case(
    crop: str = Form(...),
    farmer_contact: str | None = Form(None),
    farmer_name: str | None = Form(None),
    latitude: float | str | None = Form(None),
    longitude: float | str | None = Form(None),
    description: str | None = Form(None),
    language: str = Form("en"),
    image: UploadFile | None = File(None),
) -> dict[str, Any]:
    """Farmer Incident Report: registers a case, executes leaf AI diagnosis (if photo supplied),

    safely queries satellite anomaly screening, and fuses signals into an explainable risk score.
    """
    case_service = _get_case_service()
    registry = _get_registry()
    crop_slug = crop.strip().lower()

    if crop_slug not in registry.available_crops:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown crop '{crop}'. Available crops: {registry.available_crops}",
        )

    # Safely parse coordinates (handles empty strings from web forms)
    parsed_lat: float | None = None
    parsed_lon: float | None = None
    if latitude is not None and str(latitude).strip():
        try:
            parsed_lat = float(latitude)
        except (ValueError, TypeError):
            parsed_lat = None
    if longitude is not None and str(longitude).strip():
        try:
            parsed_lon = float(longitude)
        except (ValueError, TypeError):
            parsed_lon = None

    # 1. Leaf AI Diagnosis (if image provided)
    diagnosis_result: dict[str, Any] | None = None
    saved_image_path: str | None = None
    image_filename: str | None = None

    if image is not None and image.filename:
        image_bytes = await image.read()
        if image_bytes:
            if len(image_bytes) > MAX_UPLOAD_SIZE:
                raise HTTPException(status_code=400, detail="File size exceeds maximum limit of 10MB")
            if image.content_type and not image.content_type.startswith("image/"):
                raise HTTPException(status_code=415, detail="Upload an image file")

            # Save uploaded image locally
            upload_dir = _PROJECT_ROOT / "data" / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            image_filename = f"{crop_slug}_{Path(image.filename).name}"
            target_path = upload_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{image_filename}"
            target_path.write_bytes(image_bytes)
            saved_image_path = str(target_path)

            try:
                diag_service = registry.get(crop_slug)
                diagnosis_result = diag_service.diagnose(image_bytes, language=language)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=503, detail=f"Model for '{crop}' unavailable") from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    # 2. Resilient Satellite Screening
    satellite_service = _get_satellite_service()
    sat_payload: dict[str, Any] = {}
    if parsed_lat is not None and parsed_lon is not None:
        sat_payload = {
            "latitude": parsed_lat,
            "longitude": parsed_lon,
            "radius_meters": DEFAULT_POINT_RADIUS_METERS,  # ~100m farm proxy area (~7.76 acres)
        }
    satellite_result = satellite_service.analyze_farm_safely(sat_payload)

    # 3. Localized Outbreak Context
    has_coords = parsed_lat is not None and parsed_lon is not None
    nearby_cases = case_service.count_nearby_recent_cases(
        crop=crop_slug,
        latitude=parsed_lat,
        longitude=parsed_lon,
        days=14,
    )

    # 4. Deterministic Multi-Signal Risk Fusion
    fusion_service = _get_fusion_service()
    risk = fusion_service.assess_risk(
        crop=crop,
        diagnosis_result=diagnosis_result,
        satellite_result=satellite_result,
        farmer_description=description,
        nearby_cases_count=nearby_cases,
        coordinates_available=has_coords,
    )

    # 5. Localized Advisory
    knowledge_service = _get_knowledge_service()
    advisory: dict[str, Any] | None = None
    if diagnosis_result and diagnosis_result.get("diagnosis"):
        advisory = knowledge_service.get_advice(
            crop=diagnosis_result.get("crop", crop),
            diagnosis=diagnosis_result.get("diagnosis"),
            status=diagnosis_result.get("status", "AI_CONFIDENT"),
            language=language,
        )

    # 6. Persist Case in SQLite
    case = case_service.create_case(
        crop=crop_slug,
        farmer_contact=farmer_contact,
        farmer_name=farmer_name,
        latitude=parsed_lat,
        longitude=parsed_lon,
        location_details={"latitude": parsed_lat, "longitude": parsed_lon} if parsed_lat else None,
        description=description,
        image_path=saved_image_path,
        image_filename=image_filename,
        diagnosis_result=diagnosis_result,
        satellite_result=satellite_result,
        risk_assessment=risk.to_dict(),
        advisory=advisory,
    )

    # 7. Local Notification Alert
    if farmer_contact:
        notif_service = _get_notification_service()
        alert_text = format_case_alert(
            case_id=case["case_id"],
            crop=crop,
            event_type="CREATED",
            language=language,
        )
        notif_service.send(farmer_contact, alert_text, {"case_id": case["case_id"]})

    return case


@app.get("/cases")
def list_cases(
    status: str | None = Query(None, description="Filter by case status"),
    crop: str | None = Query(None, description="Filter by crop slug"),
    risk_level: str | None = Query(None, description="Filter by risk level"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict:
    """List agricultural incident cases with optional filtering and pagination."""
    case_service = _get_case_service()
    cases = case_service.list_cases(status=status, crop=crop, risk_level=risk_level, limit=limit, offset=offset)
    return {"cases": cases, "count": len(cases)}


@app.get("/cases/due-followup")
def get_due_followups(hours: int = Query(48, ge=1, description="Threshold hours since case registration")) -> dict:
    """List open or active cases where the recommended follow-up period has elapsed."""
    case_service = _get_case_service()
    due = case_service.get_due_followups(hours=hours)
    return {"due_cases": due, "count": len(due)}


@app.get("/cases/{case_id}")
def get_case(
    case_id: str,
    language: str = Query("en", description="Re-translate advisory into specified language (en, hi, mr)"),
) -> dict:
    """Retrieve full case details, audit history, follow-ups, and localized advisory."""
    case_service = _get_case_service()
    case = case_service.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")

    # If requested language differs, update localized advisory
    if language and case.get("ai_diagnosis"):
        knowledge_service = _get_knowledge_service()
        case["advisory"] = knowledge_service.get_advice(
            crop=case["crop"],
            diagnosis=case["ai_diagnosis"],
            status=case.get("ai_status", "AI_CONFIDENT"),
            language=language,
        )

    return case


@app.post("/cases/{case_id}/satellite")
def run_case_satellite(case_id: str) -> dict:
    """Trigger or refresh Google Earth Engine satellite screening for an existing case."""
    case_service = _get_case_service()
    case = case_service.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")

    if not case.get("latitude") or not case.get("longitude"):
        raise HTTPException(status_code=400, detail="Case lacks geographic coordinates for satellite analysis.")

    satellite_service = _get_satellite_service()
    sat_result = satellite_service.analyze_farm_safely({
        "latitude": case["latitude"],
        "longitude": case["longitude"],
        "radius_meters": DEFAULT_POINT_RADIUS_METERS,
    })

    # Re-fuse risk
    fusion_service = _get_fusion_service()
    lat = case.get("latitude")
    lon = case.get("longitude")
    has_coords = lat is not None and lon is not None
    nearby_cases = case_service.count_nearby_recent_cases(
        crop=case["crop"],
        latitude=lat,
        longitude=lon,
        days=14,
        exclude_case_id=case_id,
    )
    risk = fusion_service.assess_risk(
        crop=case["crop"],
        diagnosis_result=case.get("ai_details"),
        satellite_result=sat_result,
        farmer_description=case.get("description"),
        nearby_cases_count=nearby_cases,
        coordinates_available=has_coords,
    )

    return case_service.update_case_satellite(case_id, sat_result, risk.to_dict())


@app.post("/cases/{case_id}/review")
def expert_review(
    case_id: str,
    action: str = Body(..., embed=True, description="CONFIRM, REJECT, REQUEST_FIELD_VISIT, or NOTE"),
    reviewer_name: str = Body("Agricultural Officer", embed=True),
    notes: str = Body(..., embed=True),
    corrected_diagnosis: str | None = Body(None, embed=True),
    language: str = Body("en", embed=True),
) -> dict:
    """Expert / Officer review: confirm or correct diagnosis, request field visits, and add notes."""
    case_service = _get_case_service()
    try:
        updated_case = case_service.add_expert_review(
            case_id=case_id,
            action=action,
            reviewer_name=reviewer_name,
            notes=notes,
            corrected_diagnosis=corrected_diagnosis,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Dispatch farmer notification alert
    contact = updated_case.get("farmer_contact")
    if contact:
        notif_service = _get_notification_service()
        event = "FIELD_VISIT" if action.upper() == "REQUEST_FIELD_VISIT" else "REVIEWED"
        msg = format_case_alert(
            case_id=case_id,
            crop=updated_case["crop"],
            event_type=event,
            diagnosis=corrected_diagnosis or updated_case.get("ai_diagnosis"),
            language=language,
        )
        notif_service.send(contact, msg, {"case_id": case_id, "action": action})

    if language and updated_case.get("ai_diagnosis"):
        knowledge_service = _get_knowledge_service()
        diag = corrected_diagnosis or updated_case.get("ai_diagnosis")
        updated_case["advisory"] = knowledge_service.get_advice(
            crop=updated_case["crop"],
            diagnosis=diag,
            status=updated_case.get("ai_status", "AI_CONFIDENT"),
            language=language,
        )

    return updated_case


@app.post("/cases/{case_id}/follow-up")
def record_follow_up(
    case_id: str,
    condition: str = Body(..., embed=True, description="IMPROVED, SAME, or WORSE"),
    observed_by: str = Body("Farmer", embed=True),
    notes: str | None = Body(None, embed=True),
    language: str = Body("en", embed=True),
) -> dict:
    """Farmer or officer follow-up record to monitor problem progression."""
    case_service = _get_case_service()
    try:
        updated = case_service.add_follow_up(
            case_id=case_id,
            condition=condition,
            observed_by=observed_by,
            notes=notes,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if language and updated.get("ai_diagnosis"):
        knowledge_service = _get_knowledge_service()
        updated["advisory"] = knowledge_service.get_advice(
            crop=updated["crop"],
            diagnosis=updated["ai_diagnosis"],
            status=updated.get("ai_status", "AI_CONFIDENT"),
            language=language,
        )

    return updated
