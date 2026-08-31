from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from fastapi import Body, FastAPI, File, HTTPException, UploadFile

from krishidrishti_ai.config import load_config
from krishidrishti_ai.gee_api import _normalize_payload, _resolve_window, get_gee_service
from krishidrishti_ai.services.inference import DiagnosisService

app = FastAPI(title="KrishiDrishti AI Image Diagnosis")


@lru_cache
def get_service() -> DiagnosisService:
    return DiagnosisService(load_config(os.getenv("KRISHIDRISHTI_CONFIG", "configs/default.yaml")))


@app.get("/health")
def health() -> dict[str, str]:
    try:
        get_service()
    except FileNotFoundError:
        return {"status": "MODEL_NOT_READY"}
    return {"status": "READY"}


@app.post("/diagnose")
async def diagnose(image: UploadFile = File(...)) -> dict:
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file")
    try:
        service = get_service()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model checkpoint is not available") from exc
    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")
    try:
        return service.diagnose(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/analyze-farm")
async def analyze_farm(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        service = get_gee_service()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        result = service.analyze(payload)
        if "region_image" not in result and hasattr(service, "gee"):
            normalized = _normalize_payload(payload)
            start_date, end_date = _resolve_window(payload)
            result["region_image"] = service.gee.fetch_region_image(normalized, start_date, end_date)
            result["region_image_mime_type"] = "image/png"
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
