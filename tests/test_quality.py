import cv2
import numpy as np

from krishidrishti_ai.services.quality import assess_image_quality


SETTINGS = {"min_width": 128, "min_height": 128, "min_blur_score": 1.0, "min_brightness_score": 35.0, "max_brightness_score": 220.0}


def test_quality_rejects_invalid_bytes() -> None:
    result = assess_image_quality(b"not-an-image", SETTINGS)
    assert not result.passed
    assert result.reason == "Unable to decode image"


def test_quality_rejects_small_image() -> None:
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    _, encoded = cv2.imencode(".png", image)
    result = assess_image_quality(encoded.tobytes(), SETTINGS)
    assert not result.passed
    assert result.reason == "Image resolution is too low"
