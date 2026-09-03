from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from krishidrishti_ai.calibration import load_calibration
from krishidrishti_ai.data.transforms import build_transforms
from krishidrishti_ai.metrics import prediction_margin, shannon_entropy
from krishidrishti_ai.models.classifier import build_model
from krishidrishti_ai.services.knowledge import DiseaseKnowledgeService
from krishidrishti_ai.services.quality import assess_image_quality


class DiagnosisService:
    """Production-quality crop diagnosis service with post-hoc temperature calibration,

    multi-signal uncertainty detection, and conservative abstention logic.
    """

    def __init__(self, config: dict[str, Any], knowledge_service: DiseaseKnowledgeService | None = None):
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
        self.knowledge_service = knowledge_service or DiseaseKnowledgeService()

        # Calibration & Decision policy parameters
        inf_cfg = config.get("inference", {})
        self.temperature = float(inf_cfg.get("temperature", 1.0))
        # If calibration file exists alongside checkpoint or configured, prefer it
        cal_path = inf_cfg.get("calibration_path") or (checkpoint_path.parent / "calibration.json")
        cal_data = load_calibration(cal_path)
        if cal_data and "temperature" in cal_data:
            self.temperature = float(cal_data["temperature"])

        self.confident_threshold = float(inf_cfg.get("confident_threshold", 0.75))
        self.review_threshold = float(inf_cfg.get("review_threshold", 0.45))
        self.ambiguity_margin_threshold = float(inf_cfg.get("ambiguity_margin_threshold", 0.30))
        self.max_entropy_threshold = float(inf_cfg.get("max_entropy_threshold", 0.55))

    @staticmethod
    def _device(value: str) -> torch.device:
        return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value)

    def diagnose(self, image_bytes: bytes, language: str = "en") -> dict[str, Any]:
        # 1. Image Quality Gate
        quality = assess_image_quality(image_bytes, self.config["quality"])
        quality_payload = {
            "passed": quality.passed,
            "blur_score": quality.blur_score,
            "brightness_score": quality.brightness_score,
        }
        if not quality.passed:
            status = "IMAGE_QUALITY_REJECTED"
            analysis = self.knowledge_service.generate_analysis(None, None, 0.0, status, language=language)
            advice = self.knowledge_service.get_advice(None, None, status, language=language)
            return {
                "crop": None,
                "diagnosis": None,
                "confidence": 0.0,
                "raw_confidence": 0.0,
                "temperature": self.temperature,
                "margin": 0.0,
                "entropy": 0.0,
                "is_ambiguous": False,
                "alternative_prediction": None,
                "image_quality": quality_payload,
                "status": status,
                "top_predictions": [],
                "analysis": analysis,
                "advice": advice,
                "reference_knowledge": advice,
            }

        # 2. Decode and Preprocess Image
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise ValueError("Uploaded file is not a readable RGB image") from exc

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 3. Model Forward Pass & Logits
        with torch.inference_mode():
            logits = self.model(tensor)[0]
            raw_probs = torch.softmax(logits, dim=0).cpu().numpy()
            cal_probs = torch.softmax(logits / self.temperature, dim=0).cpu().numpy()

        raw_p1 = float(raw_probs.max())
        num_classes = len(self.classes)
        count = min(self.config["inference"].get("top_k", 3), num_classes)

        # 4. Calibrated Top-k Predictions & Metrics
        top_indices = np.argsort(cal_probs)[::-1][:count]
        predictions: list[dict[str, Any]] = []
        for idx in top_indices:
            pred_item = self._prediction(self.classes[idx], float(cal_probs[idx]))
            pred_item["display_name"] = self.knowledge_service.get_display_name(
                pred_item["crop"], pred_item["diagnosis"], language=language
            )
            predictions.append(pred_item)

        best = predictions[0]
        p1 = float(best["confidence"])
        p2 = float(predictions[1]["confidence"]) if len(predictions) > 1 else 0.0
        margin = round(p1 - p2, 6)

        # Normalized Shannon entropy
        entropy = round(float(shannon_entropy(cal_probs[np.newaxis, :], normalize=True)[0]), 6)

        # 5. Multi-Signal Decision Policy (Conservative Abstention / Review)
        if p1 < self.review_threshold:
            status = "LOW_CONFIDENCE"
            is_ambiguous = True
        elif (p1 < self.confident_threshold) or (margin < self.ambiguity_margin_threshold) or (entropy > self.max_entropy_threshold):
            status = "REVIEW_RECOMMENDED"
            is_ambiguous = True
        else:
            status = "AI_CONFIDENT"
            is_ambiguous = False

        alt_pred = predictions[1] if len(predictions) > 1 else None

        # 6. Honest Diagnostic Analysis & ICAR Reference Knowledge
        analysis = self.knowledge_service.generate_analysis(
            best["crop"],
            best["diagnosis"],
            p1,
            status,
            language=language,
            is_ambiguous=is_ambiguous,
            top_predictions=predictions,
        )

        advice = self.knowledge_service.get_advice(
            best["crop"],
            best["diagnosis"],
            status,
            language=language,
            is_ambiguous=is_ambiguous,
            top_predictions=predictions,
        )

        return {
            "crop": best["crop"],
            "diagnosis": best["diagnosis"],
            "confidence": round(p1, 6),
            "raw_confidence": round(raw_p1, 6),
            "temperature": self.temperature,
            "margin": margin,
            "entropy": entropy,
            "is_ambiguous": is_ambiguous,
            "alternative_prediction": alt_pred,
            "image_quality": quality_payload,
            "status": status,
            "top_predictions": predictions,
            "analysis": analysis,
            "advice": advice,
            "reference_knowledge": advice,
        }

    def _prediction(self, label: str, confidence: float) -> dict[str, Any]:
        separator = self.config["data"].get("class_separator", "___")
        if separator in label:
            crop, diagnosis = label.split(separator, maxsplit=1)
        else:
            crop = self.config["data"].get("crop_name", "Unknown")
            diagnosis = label
        return {"crop": crop, "diagnosis": diagnosis, "confidence": round(confidence, 6)}
