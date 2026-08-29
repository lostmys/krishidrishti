from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile

from krishidrishti_ai.config import load_config
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
