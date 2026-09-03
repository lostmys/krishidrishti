from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse

from krishidrishti_ai.config import load_config
from krishidrishti_ai.services.inference import DiagnosisService
from krishidrishti_ai.services.registry import CropRegistry

app = FastAPI(title="KrishiDrishti AI Image Diagnosis")

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB limit

# ---------------------------------------------------------------------------
# Project root — walk up from this file to find pyproject.toml.
# Used to resolve relative crop config paths from default.yaml.
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = next(
    (
        c
        for c in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents)
        if (c / "pyproject.toml").is_file()
    ),
    Path(__file__).resolve().parents[3],
)


# ---------------------------------------------------------------------------
# Crop-aware registry (preferred path)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_registry() -> CropRegistry:
    """Build and cache the CropRegistry from the main config's crops: block."""
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the farmer-facing web application UI."""
    html_path = Path(__file__).resolve().parent / "static" / "index.html"
    if html_path.is_file():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>KrishiDrishti AI Diagnosis API</h1>", status_code=200)


@app.get("/health")
def health() -> dict:
    """Report readiness for every registered crop."""
    registry = _get_registry()
    ready = [s for s in registry.available_crops if registry.checkpoint_ready(s)]
    not_ready = [s for s in registry.available_crops if not registry.checkpoint_ready(s)]
    if not not_ready:
        overall = "READY"
    elif ready:
        overall = "PARTIAL"
    else:
        overall = "MODEL_NOT_READY"
    return {"status": overall, "crops_ready": ready, "crops_not_ready": not_ready}


@app.get("/crops")
def list_crops() -> dict:
    """List all registered crop slugs."""
    return {"crops": _get_registry().available_crops}


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


# ---------------------------------------------------------------------------
# Legacy single-model endpoint — kept for backward compatibility.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_legacy_service() -> DiagnosisService:
    config_path = os.getenv(
        "KRISHIDRISHTI_CONFIG",
        str(_PROJECT_ROOT / "configs" / "default.yaml"),
    )
    return DiagnosisService(load_config(config_path))


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
        raise HTTPException(
            status_code=503, detail="Model checkpoint is not available"
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
