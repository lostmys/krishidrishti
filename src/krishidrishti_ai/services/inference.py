from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from krishidrishti_ai.data.transforms import build_transforms
from krishidrishti_ai.models.classifier import build_model
from krishidrishti_ai.services.quality import assess_image_quality


class DiagnosisService:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.device = self._device(config["training"]["device"])
        checkpoint_path = Path(config["paths"]["checkpoint_path"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Trained checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.classes: list[str] = checkpoint["classes"]
        model_name = checkpoint.get("model_name", config["model"]["name"])
        self.model = build_model(model_name, len(self.classes), pretrained=False).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.transform = build_transforms(config["data"]["image_size"], training=False)

    @staticmethod
    def _device(value: str) -> torch.device:
        return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value)

    def diagnose(self, image_bytes: bytes) -> dict[str, Any]:
        quality = assess_image_quality(image_bytes, self.config["quality"])
        quality_payload = {"passed": quality.passed, "blur_score": quality.blur_score, "brightness_score": quality.brightness_score}
        if not quality.passed:
            return {"crop": None, "diagnosis": None, "confidence": 0.0, "image_quality": quality_payload, "status": "IMAGE_QUALITY_REJECTED", "top_predictions": []}
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise ValueError("Uploaded file is not a readable RGB image") from exc
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor)[0], dim=0)
        count = min(self.config["inference"]["top_k"], len(self.classes))
        scores, indices = torch.topk(probabilities, count)
        predictions = [self._prediction(self.classes[index.item()], score.item()) for score, index in zip(scores, indices)]
        best = predictions[0]
        return {"crop": best["crop"], "diagnosis": best["diagnosis"], "confidence": best["confidence"], "image_quality": quality_payload, "status": self._status(best["confidence"]), "top_predictions": predictions}

    def _prediction(self, label: str, confidence: float) -> dict[str, Any]:
        crop, diagnosis = label.split(self.config["data"]["class_separator"], maxsplit=1)
        return {"crop": crop, "diagnosis": diagnosis, "confidence": round(confidence, 6)}

    def _status(self, confidence: float) -> str:
        rules = self.config["inference"]
        if confidence >= rules["confident_threshold"]:
            return "AI_CONFIDENT"
        if confidence >= rules["review_threshold"]:
            return "REVIEW_RECOMMENDED"
        return "LOW_CONFIDENCE"
