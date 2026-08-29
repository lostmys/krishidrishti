from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class QualityResult:
    passed: bool
    blur_score: float
    brightness_score: float
    reason: str | None = None


def assess_image_quality(image_bytes: bytes, settings: dict) -> QualityResult:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        return QualityResult(False, 0.0, 0.0, "Unable to decode image")
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness_score = float(gray.mean())
    if width < settings["min_width"] or height < settings["min_height"]:
        return QualityResult(False, blur_score, brightness_score, "Image resolution is too low")
    if blur_score < settings["min_blur_score"]:
        return QualityResult(False, blur_score, brightness_score, "Image is too blurry")
    if not settings["min_brightness_score"] <= brightness_score <= settings["max_brightness_score"]:
        return QualityResult(False, blur_score, brightness_score, "Image brightness is unsuitable")
    return QualityResult(True, blur_score, brightness_score)
