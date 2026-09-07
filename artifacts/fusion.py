from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from krishidrishti_ai.services.fusion import RiskFusionEngine

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

WEIGHT_CONFIDENCE = 0.50
WEIGHT_TRACK_RECORD = 0.25
WEIGHT_AGREEMENT = 0.25

TRUST_THRESHOLD = 0.70   # p >= 0.70  -> LIKELY_CORRECT
DOUBT_THRESHOLD = 0.40   # p <= 0.40  -> LIKELY_INCORRECT (else UNCERTAIN)

# A confident disease diagnosis that ALL other evidence contradicts is
# flagged LIKELY_INCORRECT even if the blended probability looks okay.
CONTRADICTION_CONFIDENCE_MIN = 0.75
CONTRADICTION_OTHERS_MAX = 0.30
CONTRADICTION_AGREEMENT_MAX = 0.40

# Caps applied when the image model itself signals uncertainty.
# REVIEW_STATUS_CAP sits just below TRUST_THRESHOLD on purpose: if the model
# flags uncertainty (or no status was provided), the best verdict it can ever
# earn is UNCERTAIN — never LIKELY_CORRECT.
LOW_CONFIDENCE_CAP = 0.45
REVIEW_STATUS_CAP = 0.69

OTHER_SIGNALS = ("satellite", "weather", "nearby_reports")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_unit(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"'{name}' must be a number between 0 and 1, got {value!r}.")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"'{name}' must be between 0 and 1, got {value!r}.")
    return numeric


def _normalize_label(value: Any) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(value).lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    root = next(
        (c for c in (Path.cwd(), *Path.cwd().parents) if (c / "pyproject.toml").is_file()),
        Path.cwd(),
    )
    project_file = root / candidate
    return project_file if project_file.is_file() else candidate


def load_track_record(path: str | Path = "data/knowledge/model_track_record.yaml") -> dict[str, Any]:
    """Load per-class test scores: {crop: {"macro_f1": x, "classes": {label: f1}}}."""
    import yaml  # PyYAML is already a project dependency

    resolved = _resolve_project_path(path)
    with resolved.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict) or "track_record" not in raw:
        raise ValueError(f"Track-record file {resolved} must contain a top-level 'track_record:' mapping.")
    if not isinstance(raw["track_record"], dict):
        raise ValueError(f"Track-record file {resolved} has a malformed 'track_record:' block.")
    return raw["track_record"]


def _safe_unit(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        return None
    return numeric


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class PredictionVerifier:
    """Judge whether an image-model prediction should be trusted."""

    def __init__(
        self,
        fusion_engine: RiskFusionEngine | None = None,
        track_record: Mapping[str, Any] | None = None,
        trust_threshold: float = TRUST_THRESHOLD,
        doubt_threshold: float = DOUBT_THRESHOLD,
    ) -> None:
        self.fusion = fusion_engine or RiskFusionEngine()
        self.track_record = dict(track_record) if track_record else {}
        _require_unit("trust_threshold", trust_threshold)
        _require_unit("doubt_threshold", doubt_threshold)
        if not 0.0 < doubt_threshold < trust_threshold < 1.0:
            raise ValueError("Require 0 < doubt_threshold < trust_threshold < 1.")
        self.trust_threshold = float(trust_threshold)
        self.doubt_threshold = float(doubt_threshold)

    @classmethod
    def from_config(
        cls,
        fusion_config: str | Path = "configs/fusion.yaml",
        track_record_path: str | Path = "data/knowledge/model_track_record.yaml",
    ) -> "PredictionVerifier":
        try:
            engine = RiskFusionEngine.from_config(fusion_config)
        except (FileNotFoundError, ValueError, OSError):
            engine = RiskFusionEngine()
        try:
            record = load_track_record(track_record_path)
        except (FileNotFoundError, ValueError, OSError):
            record = {}
        return cls(fusion_engine=engine, track_record=record)

    # -- history lookup ----------------------------------------------------

    def lookup_track_record(self, crop: Any, diagnosis_label: Any) -> tuple[float, str]:
        """Return (historical score, human-readable source note)."""
        crop_key = _normalize_label(crop) if crop else ""
        label_key = _normalize_label(diagnosis_label) if diagnosis_label else ""
        crop_block = self.track_record.get(crop_key)
        if isinstance(crop_block, Mapping):
            classes = crop_block.get("classes", {})
            if isinstance(classes, Mapping) and label_key in classes:
                value = _safe_unit(classes[label_key])
                if value is not None:
                    return value, f"test F1 for '{diagnosis_label}' ({value:.0%})"
            macro = _safe_unit(crop_block.get("macro_f1"))
            if macro is not None:
                return macro, f"class not in test records — {crop} average ({macro:.0%})"
        return 0.5, "no test record for this crop/class — neutral 50%"

    # -- main verdict ------------------------------------------------------

    def verify(
        self,
        diagnosis: Mapping[str, Any],
        signals: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Verify one image-model prediction against all available evidence.

        Parameters
        ----------
        diagnosis: the ``/diagnose`` response (or a minimal mapping with at
            least ``diagnosis`` + ``confidence``; ``status``/``crop`` optional
            but recommended).
        signals: the four fusion signals (each 0..1).
        context: optional free-form dict passed through to the risk card.
        """
        if not isinstance(diagnosis, Mapping):
            raise ValueError("diagnosis must be a mapping like the /diagnose response.")
        if "confidence" not in diagnosis:
            raise ValueError("diagnosis must include 'confidence' (the model's confidence).")
        confidence = _require_unit("diagnosis.confidence", diagnosis["confidence"])
        status = str(diagnosis.get("status", "UNKNOWN")).upper()
        label = diagnosis.get("diagnosis")
        crop = diagnosis.get("crop")
        context = context if isinstance(context, Mapping) else {}

        # 1. Fuse the four evidence sources (validates the signals too).
        risk_card = self.fusion.fuse(signals, context=context)
        breakdown = risk_card["breakdown"]

        # 2. Agreement: does the rest of the evidence back the image verdict?
        image_signal = breakdown["image_ai"]["signal"]
        others_avg = round(
            sum(breakdown[name]["signal"] for name in OTHER_SIGNALS) / len(OTHER_SIGNALS), 4
        )
        agreement = round(1.0 - abs(image_signal - others_avg), 4)

        # 3. Historical track record for this exact predicted class.
        historical, historical_note = self.lookup_track_record(crop, label)

        # 4. Blend into a correctness probability, honouring the model's
        # own uncertainty flags.
        reasons: list[str] = []
        label_text = f"'{label}'" if label is not None else "no diagnosis"
        crop_text = f" ({crop})" if crop else ""
        reasons.append(
            f"Model predicted {label_text}{crop_text} with {confidence:.0%} calibrated confidence [{status}]"
        )
        reasons.append(f"Model's history: {historical_note}")
        reasons.append(
            f"Other evidence agrees {agreement:.0%} with the image "
            f"(other sources average {others_avg:.2f})"
        )

        if status == "IMAGE_QUALITY_REJECTED" or label is None:
            probability = 0.0
            verdict = "UNCERTAIN"
            reasons.append("Cannot verify — the image was rejected or unreadable, so there is no prediction to judge.")
        elif status == "LOW_CONFIDENCE":
            probability = round(min(
                WEIGHT_CONFIDENCE * confidence + WEIGHT_TRACK_RECORD * historical + WEIGHT_AGREEMENT * agreement,
                LOW_CONFIDENCE_CAP,
            ), 3)
            verdict = "UNCERTAIN"
            reasons.append("Model abstained with low confidence — verdict stays UNCERTAIN until an expert checks.")
        else:
            probability = round(
                WEIGHT_CONFIDENCE * confidence + WEIGHT_TRACK_RECORD * historical + WEIGHT_AGREEMENT * agreement, 3
            )
            if status in ("REVIEW_RECOMMENDED", "UNKNOWN"):
                if probability > REVIEW_STATUS_CAP:
                    reasons.append(
                        f"Model flagged uncertainty ({status}) — correctness capped at {REVIEW_STATUS_CAP:.0%}."
                    )
                    probability = REVIEW_STATUS_CAP
            is_disease = "healthy" not in str(label).lower()
            contradicted = (
                is_disease
                and confidence >= CONTRADICTION_CONFIDENCE_MIN
                and others_avg <= CONTRADICTION_OTHERS_MAX
                and agreement <= CONTRADICTION_AGREEMENT_MAX
            )
            if contradicted:
                verdict = "LIKELY_INCORRECT"
                reasons.append(
                    "Contradiction: confident disease diagnosis, but satellite/weather/nearby evidence "
                    "shows a calm field — the image prediction looks wrong."
                )
            elif probability >= self.trust_threshold:
                verdict = "LIKELY_CORRECT"
                reasons.append("Verdict: LIKELY_CORRECT — confidence, history and other evidence all agree.")
            elif probability <= self.doubt_threshold:
                verdict = "LIKELY_INCORRECT"
                reasons.append("Verdict: LIKELY_INCORRECT — combined evidence is too weak to trust this prediction.")
            else:
                verdict = "UNCERTAIN"
                reasons.append("Verdict: UNCERTAIN — evidence is mixed, expert review needed.")

        is_trusted = verdict == "LIKELY_CORRECT"
        needs_expert = (not is_trusted) or (risk_card["risk"] == "HIGH")
        if is_trusted and risk_card["risk"] == "HIGH":
            reasons.append("Prediction is trusted AND risk is HIGH — an expert should still confirm treatment.")

        return {
            "verdict": verdict,
            "is_trusted": is_trusted,
            "correctness_probability": probability,
            "predicted_diagnosis": label,
            "crop": crop,
            "confidence": confidence,
            "needs_expert": needs_expert,
            "reasons": reasons,
            "factors": {
                "confidence": confidence,
                "confidence_weight": WEIGHT_CONFIDENCE,
                "historical_score": historical,
                "historical_note": historical_note,
                "historical_weight": WEIGHT_TRACK_RECORD,
                "agreement": agreement,
                "agreement_weight": WEIGHT_AGREEMENT,
                "others_average": others_avg,
                "image_signal": image_signal,
            },
            "risk_assessment": risk_card,
            "note": (
                "correctness_probability is a statistical estimate from confidence, test history and "
                "cross-source agreement — not ground truth. Definitive confirmation needs a field expert."
            ),
        }


def verify_prediction(
    diagnosis: Mapping[str, Any],
    signals: Mapping[str, Any],
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One-line convenience API using defaults (no track-record file)."""
    return PredictionVerifier().verify(diagnosis, signals, context=context)
