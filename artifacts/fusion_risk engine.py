"""Fusion / Risk Engine.

Turns multiple independent signals into one transparent risk assessment.

Inputs (any subset; missing signals simply do not contribute):

- ``image_ai``       — result of the crop-image diagnosis, e.g.
                       ``{"confidence": 0.87, "diseaseDetected": true}``
- ``satellite``      — vegetation anomaly from the GEE satellite module, e.g.
                       ``{"anomaly": true, "score": 0.78}`` (or the
                       ``farm_label`` / ``farm_local_abnormal_score`` fields
                       returned by ``POST /analyze-farm``)
- ``weather``        — weather/context risk in 0..1, e.g. ``{"risk": 0.72}``
- ``nearby_reports`` — similar reports nearby, e.g. ``{"count": 3}``
- ``farmer_report``  — contextual signal from the farmer's voice/report, e.g.
                       ``{"severity": "high"}`` (acts as a bounded modifier,
                       not a weighted signal)

Each of the four core signals may also be passed as a bare number in 0..1
(already-normalised signal strength) — this is what
``artifacts/fusion.py::PredictionVerifier`` does.

Output ("risk card"):

.. code-block:: json

    {
      "risk": "HIGH",
      "score": 0.851,
      "expert_required": true,
      "reasons": [
        "Disease detected from crop image",
        "Satellite vegetation anomaly detected",
        "Weather conditions indicate elevated risk",
        "Similar nearby reports found"
      ],
      "breakdown": { "...per-signal arithmetic..." : {} },
      "whatsapp": { "en": "...", "hi": "...", "mr": "..." }
    }

The score is a weighted sum of the available signals.  When a signal is
missing, its weight is redistributed proportionally over the signals that
are available, so the score always stays in 0..1 and every contribution is
explainable.

IMPORTANT: the default weights are PROTOTYPE WEIGHTS — chosen for a
transparent, explainable demo, not scientifically validated final weights.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

# ---------------------------------------------------------------------------
# Prototype weights (see configs/fusion.yaml)
# ---------------------------------------------------------------------------

PROTOTYPE_WEIGHTS: dict[str, float] = {
    "image_ai": 0.40,
    "satellite": 0.25,
    "weather": 0.15,
    "nearby_reports": 0.20,
}

PROTOTYPE_WEIGHTS_NOTE = (
    "PROTOTYPE WEIGHTS — chosen for a transparent, explainable demo. "
    "They are NOT scientifically validated final weights."
)

DEFAULT_THRESHOLDS: dict[str, float] = {"high": 0.70, "medium": 0.40}
DEFAULT_NEARBY_SATURATION = 3       # count >= 3 -> full nearby-reports signal
DEFAULT_WEATHER_ELEVATED = 0.50     # weather signal at/above this = "elevated"
DEFAULT_CONTEXT_CAP = 0.05          # max score nudge from farmer-reported severity
DEFAULT_CONTRADICTION_MIN = 0.60    # one signal >= this vs LOW overall = conflict

# Farmer-reported severity -> 0..1 strength (drives a bounded +/- adjustment).
FARMER_SEVERITY_LEVELS: dict[str, float] = {
    "none": 0.0,
    "low": 0.25,
    "medium": 0.5,
    "high": 1.0,
}

CORE_SIGNALS = ("image_ai", "satellite", "weather", "nearby_reports")
CONTEXTUAL_SIGNALS = ("farmer_report",)
KNOWN_SIGNALS = CORE_SIGNALS + CONTEXTUAL_SIGNALS

# ---------------------------------------------------------------------------
# Reason strings (English is the canonical card language; WhatsApp messages
# are localised via _REASON_TRANSLATIONS below).
# ---------------------------------------------------------------------------

REASON_IMAGE_DISEASE = "Disease detected from crop image"
REASON_IMAGE_HEALTHY = "No disease detected in crop image"
REASON_IMAGE_WEAK = "Crop image signal is weak (low confidence)"
REASON_SATELLITE_ANOMALY = "Satellite vegetation anomaly detected"
REASON_SATELLITE_CLEAR = "No satellite vegetation anomaly detected"
REASON_WEATHER_ELEVATED = "Weather conditions indicate elevated risk"
REASON_WEATHER_CALM = "Weather conditions do not indicate elevated risk"
REASON_NEARBY_FOUND = "Similar nearby reports found"
REASON_NEARBY_FEW = "Few similar nearby reports found"
REASON_NEARBY_NONE = "No similar nearby reports found"
REASON_FARMER_HIGH = "Farmer reports severe symptoms in the field"
REASON_FARMER_MEDIUM = "Farmer reports moderate symptoms in the field"
REASON_FARMER_LOW = "Farmer reports mild symptoms in the field"
REASON_FARMER_NONE = "Farmer reports no symptoms in the field"
REASON_CONFLICT = (
    "Evidence is conflicting — one signal flags a problem while the fused "
    "score is low, so expert verification is recommended"
)
REASON_SINGLE_SIGNAL = (
    "Only one signal was available for this assessment — treat it with caution"
)

# ---------------------------------------------------------------------------
# WhatsApp advisory strings (curated translations, no LLM — repo convention)
# ---------------------------------------------------------------------------

_ADVISORY_TEXT = {
    "title": {
        "en": "KrishiDrishti Farm Advisory",
        "hi": "कृषिदृष्टि फ़ार्म सलाह",
        "mr": "कृषिदृष्टि शेती सल्ला",
    },
    "risk_line": {
        "en": "Risk level: {level} (score {score}/1.00)",
        "hi": "जोखिम स्तर: {level} (स्कोर {score}/1.00)",
        "mr": "धोक्याची पातळी: {level} (स्कोअर {score}/1.00)",
    },
    "why": {"en": "Why:", "hi": "कारण:", "mr": "कारण:"},
    "expert": {
        "en": "Expert verification recommended — please contact your local Krishi Vigyan Kendra (KVK) or agricultural officer.",
        "hi": "विशेषज्ञ सत्यापन की सिफारिश — कृपया अपने स्थानीय कृषि विज्ञान केंद्र (KVK) या कृषि अधिकारी से संपर्क करें।",
        "mr": "तज्ज्ञ तपासणीची शिफारस — कृपया तुमच्या स्थानिक कृषी विज्ञान केंद्राशी (KVK) किंवा कृषी अधिकाऱ्याशी संपर्क करा.",
    },
    "no_expert": {
        "en": "No expert visit needed right now. Keep monitoring your field.",
        "hi": "अभी किसी विशेषज्ञ की आवश्यकता नहीं है। अपने खेत की निगरानी जारी रखें।",
        "mr": "सध्या तज्ज्ञाची गरज नाही. तुमच्या शेताचे निरीक्षण सुरू ठेवा.",
    },
    "note": {
        "en": "(Prototype assessment for demonstration — not a final scientific diagnosis.)",
        "hi": "(प्रदर्शन हेतु प्रोटोटाइप मूल्यांकन — अंतिम वैज्ञानिक निदान नहीं।)",
        "mr": "(प्रात्यक्षिकासाठी प्रोटोटाइप मूल्यांकन — अंतिम वैज्ञानिक निदान नाही.)",
    },
}

_RISK_LEVEL_TEXT = {
    "HIGH": {"en": "HIGH", "hi": "उच्च (HIGH)", "mr": "उच्च (HIGH)"},
    "MEDIUM": {"en": "MEDIUM", "hi": "मध्यम (MEDIUM)", "mr": "मध्यम (MEDIUM)"},
    "LOW": {"en": "LOW", "hi": "कमी (LOW)", "mr": "कमी (LOW)"},
}

_REASON_TRANSLATIONS: dict[str, dict[str, str]] = {
    REASON_IMAGE_DISEASE: {
        "hi": "फ़सल की तस्वीर में रोग के लक्षण मिले",
        "mr": "पिकाच्या फोटोमध्ये रोगाची लक्षणे आढळली",
    },
    REASON_IMAGE_HEALTHY: {
        "hi": "फ़सल की तस्वीर में कोई रोग नहीं मिला",
        "mr": "पिकाच्या फोटोमध्ये रोग आढळला नाही",
    },
    REASON_IMAGE_WEAK: {
        "hi": "फ़सल की तस्वीर में रोग के कमज़ोर संकेत",
        "mr": "पिकाच्या फोटोमध्ये रोगाचे हलके संकेत",
    },
    REASON_SATELLITE_ANOMALY: {
        "hi": "उपग्रह इमेज में फ़सल में विसंगति दिखी",
        "mr": "उपग्रह चित्रात पिकात विसंगती दिसली",
    },
    REASON_SATELLITE_CLEAR: {
        "hi": "उपग्रह इमेज में कोई विसंगति नहीं दिखी",
        "mr": "उपग्रह चित्रात विसंगती दिसली नाही",
    },
    REASON_WEATHER_ELEVATED: {
        "hi": "मौसम की स्थितियाँ बढ़ा हुआ जोखिम दर्शाती हैं",
        "mr": "हवामान परिस्थिती वाढलेला धोका दर्शवतात",
    },
    REASON_WEATHER_CALM: {
        "hi": "मौसम की स्थितियाँ बढ़ा हुआ जोखिम नहीं दर्शातीं",
        "mr": "हवामान परिस्थिती वाढलेला धोका दर्शवत नाहीत",
    },
    REASON_NEARBY_FOUND: {
        "hi": "आस-पास के खेतों से ऐसी ही रिपोर्ट मिलीं",
        "mr": "जवळच्या शेतांमधून अशाच अहवाल मिळाले",
    },
    REASON_NEARBY_FEW: {
        "hi": "आस-पास से कुछ समान रिपोर्ट मिलीं",
        "mr": "जवळपास काही समान अहवाल आढळले",
    },
    REASON_NEARBY_NONE: {
        "hi": "आस-पास से कोई समान रिपोर्ट नहीं मिली",
        "mr": "जवळपासून समान अहवाल आढळले नाहीत",
    },
    REASON_FARMER_HIGH: {
        "hi": "किसान ने गंभीर लक्षण बताए हैं",
        "mr": "शेतकऱ्याने गंभीर लक्षणे सांगितली आहेत",
    },
    REASON_FARMER_MEDIUM: {
        "hi": "किसान ने मध्यम लक्षण बताए हैं",
        "mr": "शेतकऱ्याने मध्यम लक्षणे सांगितली आहेत",
    },
    REASON_FARMER_LOW: {
        "hi": "किसान ने हल्के लक्षण बताए हैं",
        "mr": "शेतकऱ्याने हलकी लक्षणे सांगितली आहेत",
    },
    REASON_FARMER_NONE: {
        "hi": "किसान ने कोई लक्षण नहीं बताए",
        "mr": "शेतकऱ्याने कोणतीही लक्षणे सांगितली नाहीत",
    },
    REASON_CONFLICT: {
        "hi": "संकेत आपस में मेल नहीं खाते — निर्णय से पहले विशेषज्ञ सल्ला लें",
        "mr": "संकेत एकमेकांशी जुळत नाहीत — निर्णयापूर्वी तज्ज्ञ सल्ला घ्या",
    },
    REASON_SINGLE_SIGNAL: {
        "hi": "केवल एक ही संकेत उपलब्ध था — सावधानी बरतें",
        "mr": "केवळ एकच संकेत उपलब्ध होता — सावधगिरी ठेवा",
    },
}


# ---------------------------------------------------------------------------
# Small validation helpers (kept in sync with artifacts/fusion.py conventions)
# ---------------------------------------------------------------------------


def _require_unit(name: str, value: Any) -> float:
    """Validate ``value`` is a real number in [0, 1] and return it as float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"'{name}' must be a number between 0 and 1, got {value!r}.")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"'{name}' must be between 0 and 1, got {value!r}.")
    return numeric


def _optional_unit(name: str, mapping: Mapping[str, Any], *keys: str) -> float | None:
    """Return the first present unit-scale key from ``keys``, else None."""
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return _require_unit(f"{name}.{key}", mapping[key])
    return None


def _require_count(name: str, value: Any) -> int:
    """Validate ``value`` is a non-negative integer count."""
    if isinstance(value, bool):
        raise ValueError(f"'{name}' must be a non-negative integer count, got {value!r}.")
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer() or value < 0:
            raise ValueError(f"'{name}' must be a non-negative integer count, got {value!r}.")
        return int(value)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"'{name}' must be a non-negative integer count, got {value!r}.")
    return value


def _resolve_project_path(path: str | Path) -> Path:
    """Resolve ``path`` against the project root (nearest pyproject.toml)."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    root = next(
        (c for c in (Path.cwd(), *Path.cwd().parents) if (c / "pyproject.toml").is_file()),
        Path.cwd(),
    )
    project_file = root / candidate
    return project_file if project_file.is_file() else candidate


def _unavailable(note: str = "not provided") -> dict[str, Any]:
    return {"available": False, "signal": 0.0, "note": note, "reason": None, "detail": {}}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class RiskFusionEngine:
    """Fuse independent signals into one transparent LOW / MEDIUM / HIGH risk card.

    Parameters
    ----------
    weights:
        Mapping of the four core signals to weights in [0, 1].  Defaults to
        the PROTOTYPE WEIGHTS (Image AI 40%, Satellite 25%, Weather 15%,
        Nearby reports 20%).  When some signals are missing at fusion time
        the weights of the available signals are renormalised to sum to 1.
    thresholds:
        ``{"high": 0.70, "medium": 0.40}`` — fused score >= high -> HIGH,
        >= medium -> MEDIUM, else LOW (same 70/40 cut-offs the dashboard
        uses on its 0-100 scale).
    nearby_saturation:
        Report count at which the nearby-reports signal reaches full
        strength (default 3).
    weather_elevated_threshold:
        Weather signal at/above this value counts as "elevated risk"
        (default 0.5).
    context_adjustment_cap:
        Maximum positive/negative score adjustment from farmer-reported
        severity (default +/-0.05).  Farmer reports are contextual — they
        nudge the score but carry no prototype weight of their own.
    contradiction_signal_min:
        A single available signal at/above this value while the overall
        risk is LOW flags conflicting evidence (default 0.6).
    """

    def __init__(
        self,
        weights: Mapping[str, float] | None = None,
        thresholds: Mapping[str, float] | None = None,
        nearby_saturation: int = DEFAULT_NEARBY_SATURATION,
        weather_elevated_threshold: float = DEFAULT_WEATHER_ELEVATED,
        context_adjustment_cap: float = DEFAULT_CONTEXT_CAP,
        contradiction_signal_min: float = DEFAULT_CONTRADICTION_MIN,
    ) -> None:
        resolved_weights = dict(weights) if weights is not None else dict(PROTOTYPE_WEIGHTS)
        unknown = set(resolved_weights) - set(CORE_SIGNALS)
        if unknown:
            raise ValueError(f"Unknown weight keys {sorted(unknown)}; expected {list(CORE_SIGNALS)}.")
        missing = set(CORE_SIGNALS) - set(resolved_weights)
        if missing:
            raise ValueError(f"Missing weights for {sorted(missing)}.")
        self.weights: dict[str, float] = {
            name: _require_unit(f"weights.{name}", resolved_weights[name]) for name in CORE_SIGNALS
        }
        if sum(self.weights.values()) <= 0:
            raise ValueError("At least one weight must be greater than zero.")

        resolved_thresholds = dict(thresholds) if thresholds is not None else dict(DEFAULT_THRESHOLDS)
        unknown = set(resolved_thresholds) - {"high", "medium"}
        if unknown:
            raise ValueError(f"Unknown threshold keys {sorted(unknown)}; expected ['high', 'medium'].")
        if "high" not in resolved_thresholds or "medium" not in resolved_thresholds:
            raise ValueError("Thresholds must include both 'high' and 'medium'.")
        self.thresholds: dict[str, float] = {
            "high": _require_unit("thresholds.high", resolved_thresholds["high"]),
            "medium": _require_unit("thresholds.medium", resolved_thresholds["medium"]),
        }
        if not 0.0 < self.thresholds["medium"] < self.thresholds["high"] <= 1.0:
            raise ValueError("Require 0 < thresholds.medium < thresholds.high <= 1.")

        if isinstance(nearby_saturation, bool) or not isinstance(nearby_saturation, int) or nearby_saturation < 1:
            raise ValueError(f"nearby_saturation must be an integer >= 1, got {nearby_saturation!r}.")
        self.nearby_saturation = nearby_saturation
        self.weather_elevated_threshold = _require_unit(
            "weather_elevated_threshold", weather_elevated_threshold
        )
        self.context_adjustment_cap = _require_unit("context_adjustment_cap", context_adjustment_cap)
        self.contradiction_signal_min = _require_unit(
            "contradiction_signal_min", contradiction_signal_min
        )

    # -- construction from config ------------------------------------------

    @classmethod
    def from_config(cls, path: str | Path = "configs/fusion.yaml") -> "RiskFusionEngine":
        """Build the engine from a YAML config file (default: configs/fusion.yaml).

        The file must contain a top-level ``fusion:`` mapping (a flat mapping
        with the same keys is also accepted).  Raises ``FileNotFoundError`` if
        the file is missing and ``ValueError`` if its contents are invalid.
        """
        import yaml  # PyYAML is a project dependency; imported lazily

        resolved = _resolve_project_path(path)
        with resolved.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        if not isinstance(raw, dict):
            raise ValueError(f"Fusion config {resolved} must contain a YAML mapping.")
        block = raw.get("fusion", raw if "weights" in raw else None)
        if not isinstance(block, dict):
            raise ValueError(
                f"Fusion config {resolved} must contain a top-level 'fusion:' mapping "
                "with keys: weights, thresholds, nearby_reports_saturation, "
                "weather_elevated_threshold, context_adjustment_cap, contradiction_signal_min."
            )
        allowed = {
            "weights",
            "thresholds",
            "nearby_reports_saturation",
            "weather_elevated_threshold",
            "context_adjustment_cap",
            "contradiction_signal_min",
        }
        unknown = set(block) - allowed
        if unknown:
            raise ValueError(f"Unknown fusion config keys {sorted(unknown)}; allowed: {sorted(allowed)}.")
        return cls(
            weights=block.get("weights"),
            thresholds=block.get("thresholds"),
            nearby_saturation=block.get("nearby_reports_saturation", DEFAULT_NEARBY_SATURATION),
            weather_elevated_threshold=block.get("weather_elevated_threshold", DEFAULT_WEATHER_ELEVATED),
            context_adjustment_cap=block.get("context_adjustment_cap", DEFAULT_CONTEXT_CAP),
            contradiction_signal_min=block.get("contradiction_signal_min", DEFAULT_CONTRADICTION_MIN),
        )

    # -- per-signal normalisation -------------------------------------------

    def _normalize_image_ai(self, value: Any) -> dict[str, Any]:
        """Image AI result -> signal in 0..1 plus a human-readable reason."""
        if isinstance(value, Mapping):
            detail = dict(value)
            status = str(detail.get("status", "")).upper()
            if status == "IMAGE_QUALITY_REJECTED":
                return _unavailable("image rejected by the quality gate")

            confidence = _optional_unit("image_ai", detail, "confidence", "score", "signal")
            if confidence is None:
                raise ValueError(
                    "image_ai payload must include 'confidence' (0..1), "
                    f"got keys {sorted(detail)}."
                )

            detected = detail.get("diseaseDetected", detail.get("disease_detected"))
            if detected is None:
                label = detail.get("diagnosis", detail.get("label", detail.get("predicted_class")))
                detected = "healthy" not in str(label).lower() if label is not None else True
            if not isinstance(detected, bool):
                raise ValueError(
                    f"image_ai.diseaseDetected must be true or false, got {detected!r}."
                )

            if not detected:
                return {
                    "available": True,
                    "signal": 0.0,
                    "note": f"model is confident the crop is healthy ({confidence:.0%})",
                    "reason": REASON_IMAGE_HEALTHY,
                    "detail": detail,
                }
            return {
                "available": True,
                "signal": round(confidence, 4),
                "note": f"disease detected with {confidence:.0%} confidence"
                + (f" [{status}]" if status else ""),
                "reason": REASON_IMAGE_DISEASE,
                "detail": detail,
            }

        # Bare number: pre-normalised signal strength (PredictionVerifier style).
        signal = _require_unit("image_ai", value)
        if signal >= 0.5:
            reason, note = REASON_IMAGE_DISEASE, "pre-normalised image signal"
        elif signal > 0.0:
            reason, note = REASON_IMAGE_WEAK, "pre-normalised image signal"
        else:
            reason, note = REASON_IMAGE_HEALTHY, "pre-normalised image signal"
        return {"available": True, "signal": round(signal, 4), "note": note, "reason": reason, "detail": {}}

    def _normalize_satellite(self, value: Any) -> dict[str, Any]:
        """Satellite anomaly -> signal in 0..1 plus a human-readable reason."""
        if isinstance(value, Mapping):
            detail = dict(value)
            label = str(detail.get("farm_label", detail.get("label", ""))).lower()
            if label == "unreachable":
                return _unavailable("satellite data unreachable for this farm")

            score = _optional_unit(
                "satellite",
                detail,
                "score",
                "anomaly_score",
                "risk",
                "farm_local_abnormal_score",
                "abnormality_score",
            )
            anomaly = detail.get("anomaly")
            if anomaly is None:
                if label:
                    anomaly = label == "abnormal"
                elif score is not None:
                    anomaly = True
                else:
                    raise ValueError(
                        "satellite payload must include 'anomaly' (bool) or a score "
                        f"(score/anomaly_score/farm_local_abnormal_score), got keys {sorted(detail)}."
                    )
            if not isinstance(anomaly, bool):
                raise ValueError(f"satellite.anomaly must be true or false, got {anomaly!r}.")

            if not anomaly:
                return {
                    "available": True,
                    "signal": 0.0,
                    "note": "no vegetation anomaly on the satellite pass",
                    "reason": REASON_SATELLITE_CLEAR,
                    "detail": detail,
                }
            signal = score if score is not None else 1.0
            return {
                "available": True,
                "signal": round(signal, 4),
                "note": f"vegetation anomaly with strength {signal:.0%}"
                + (f" [farm_label={label}]" if label else ""),
                "reason": REASON_SATELLITE_ANOMALY,
                "detail": detail,
            }

        signal = _require_unit("satellite", value)
        reason = REASON_SATELLITE_ANOMALY if signal >= 0.5 else REASON_SATELLITE_CLEAR
        return {
            "available": True,
            "signal": round(signal, 4),
            "note": "pre-normalised satellite signal",
            "reason": reason,
            "detail": {},
        }

    def _normalize_weather(self, value: Any) -> dict[str, Any]:
        """Weather / context risk -> signal in 0..1 plus a human-readable reason."""
        if isinstance(value, Mapping):
            detail = dict(value)
            risk = _optional_unit("weather", detail, "risk", "score", "signal")
            if risk is None:
                raise ValueError(
                    f"weather payload must include 'risk' (0..1), got keys {sorted(detail)}."
                )
        else:
            detail = {}
            risk = _require_unit("weather", value)

        elevated = risk >= self.weather_elevated_threshold
        return {
            "available": True,
            "signal": round(risk, 4),
            "note": f"weather risk {risk:.0%}"
            + (" (elevated)" if elevated else " (not elevated)"),
            "reason": REASON_WEATHER_ELEVATED if elevated else REASON_WEATHER_CALM,
            "detail": detail,
        }

    def _normalize_nearby_reports(self, value: Any) -> dict[str, Any]:
        """Nearby similar reports -> signal in 0..1 plus a human-readable reason.

        A bare integer is treated as a report count; a bare float in [0, 1] is
        treated as a pre-normalised signal strength.
        """
        if isinstance(value, Mapping):
            detail = dict(value)
            count = None
            for key in ("count", "reports", "similar_reports", "nearby_count"):
                if key in detail and detail[key] is not None:
                    count = _require_count(f"nearby_reports.{key}", detail[key])
                    break
            if count is None:
                pre = _optional_unit("nearby_reports", detail, "signal", "risk")
                if pre is None:
                    raise ValueError(
                        "nearby_reports payload must include 'count' (non-negative "
                        f"integer), got keys {sorted(detail)}."
                    )
                return self._nearby_from_signal(pre)
            note = f"{count} similar nearby report(s)"
            if count == 0:
                return {
                    "available": True,
                    "signal": 0.0,
                    "note": note,
                    "reason": REASON_NEARBY_NONE,
                    "detail": detail,
                }
            signal = min(1.0, count / self.nearby_saturation)
            reason = REASON_NEARBY_FOUND if signal >= 0.5 else REASON_NEARBY_FEW
            return {
                "available": True,
                "signal": round(signal, 4),
                "note": f"{note} (saturates at {self.nearby_saturation})",
                "reason": reason,
                "detail": detail,
            }

        if isinstance(value, bool):
            raise ValueError(f"'nearby_reports' must be a count or a 0..1 signal, got {value!r}.")
        if isinstance(value, int):
            return self._normalize_nearby_reports({"count": value})
        return self._nearby_from_signal(_require_unit("nearby_reports", value))

    def _nearby_from_signal(self, signal: float) -> dict[str, Any]:
        if signal >= 0.5:
            reason = REASON_NEARBY_FOUND
        elif signal > 0.0:
            reason = REASON_NEARBY_FEW
        else:
            reason = REASON_NEARBY_NONE
        return {
            "available": True,
            "signal": round(signal, 4),
            "note": "pre-normalised nearby-reports signal",
            "reason": reason,
            "detail": {},
        }

    def _normalize_farmer_report(self, value: Any) -> dict[str, Any]:
        """Farmer voice / report -> bounded context modifier (not weighted)."""
        if isinstance(value, str):
            severity = value
            detail: dict[str, Any] = {"severity": value}
        elif isinstance(value, Mapping):
            detail = dict(value)
            severity = detail.get("severity", detail.get("reported_severity", detail.get("farmer_severity")))
        else:
            raise ValueError(
                "farmer_report must be a mapping like {'severity': 'low' | 'medium' | 'high' | 'none'}, "
                f"got {value!r}."
            )
        if not isinstance(severity, str) or severity.strip().lower() not in FARMER_SEVERITY_LEVELS:
            raise ValueError(
                "farmer_report.severity must be one of "
                f"{sorted(FARMER_SEVERITY_LEVELS)}, got {severity!r}."
            )
        severity = severity.strip().lower()
        strength = FARMER_SEVERITY_LEVELS[severity]
        reason = {
            "none": REASON_FARMER_NONE,
            "low": REASON_FARMER_LOW,
            "medium": REASON_FARMER_MEDIUM,
            "high": REASON_FARMER_HIGH,
        }[severity]
        return {
            "available": True,
            "severity": severity,
            "strength": strength,
            "reason": reason,
            "detail": detail,
        }

    # -- fusion --------------------------------------------------------------

    def fuse(
        self,
        signals: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fuse the given signals into one transparent risk assessment.

        Parameters
        ----------
        signals:
            Mapping with any of the keys in ``KNOWN_SIGNALS``.  Values are
            either raw payload dicts (see module docstring) or pre-normalised
            0..1 numbers.  A ``context`` key inside ``signals`` is treated the
            same as the ``context`` parameter.
        context:
            Optional free-form mapping, echoed back on the risk card.

        Returns
        -------
        The risk card (see module docstring).  Raises ``ValueError`` for
        invalid input, including when no usable core signal is provided.
        """
        if not isinstance(signals, Mapping):
            raise ValueError(f"signals must be a mapping, got {type(signals).__name__}.")

        merged_context: dict[str, Any] = {}
        inline_context = signals.get("context")
        if inline_context is not None and not isinstance(inline_context, Mapping):
            raise ValueError("context must be a mapping (free-form dict).")
        if inline_context:
            merged_context.update(inline_context)
        if context:
            if not isinstance(context, Mapping):
                raise ValueError("context must be a mapping (free-form dict).")
            merged_context.update(context)

        payloads = {key: value for key, value in signals.items() if key != "context"}
        unknown = set(payloads) - set(KNOWN_SIGNALS)
        if unknown:
            raise ValueError(
                f"Unknown signal(s) {sorted(unknown)}; valid signals: {list(KNOWN_SIGNALS)}."
            )
        if not payloads:
            raise ValueError(
                "Provide at least one signal; valid signals: "
                f"{list(KNOWN_SIGNALS)} (plus optional 'context')."
            )

        normalizers = {
            "image_ai": self._normalize_image_ai,
            "satellite": self._normalize_satellite,
            "weather": self._normalize_weather,
            "nearby_reports": self._normalize_nearby_reports,
            "farmer_report": self._normalize_farmer_report,
        }
        normalized: dict[str, dict[str, Any]] = {}
        for name in KNOWN_SIGNALS:
            if name not in payloads:
                normalized[name] = _unavailable()
            else:
                try:
                    normalized[name] = normalizers[name](payloads[name])
                except ValueError as exc:
                    raise ValueError(f"Invalid {name} signal: {exc}") from exc

        core_available = [name for name in CORE_SIGNALS if normalized[name]["available"]]
        if not core_available:
            problems = "; ".join(
                f"{name}: {normalized[name].get('note', 'unusable')}"
                for name in KNOWN_SIGNALS
                if name in payloads
            )
            raise ValueError(
                "Risk assessment needs at least one usable signal among "
                f"{list(CORE_SIGNALS)}. Details: {problems or 'no signals provided'}."
            )

        # Weighted sum over available signals, renormalised to sum to 1.
        total_weight = sum(self.weights[name] for name in core_available)
        score = 0.0
        effective: dict[str, float] = {}
        contributions: dict[str, float] = {}
        for name in core_available:
            weight = self.weights[name] / total_weight
            contribution = weight * normalized[name]["signal"]
            effective[name] = round(weight, 4)
            contributions[name] = round(contribution, 4)
            score += contribution

        # Contextual adjustment from the farmer's own report (bounded, +/- cap).
        farmer = normalized["farmer_report"]
        adjustment = 0.0
        adjustment_source = None
        if farmer["available"]:
            adjustment = self.context_adjustment_cap * (2.0 * farmer["strength"] - 1.0)
            adjustment_source = f"farmer_report severity={farmer['severity']}"
        score = min(1.0, max(0.0, score + adjustment))

        risk = (
            "HIGH"
            if score >= self.thresholds["high"]
            else ("MEDIUM" if score >= self.thresholds["medium"] else "LOW")
        )

        # Conflicting evidence: one loud signal drowned by calm ones.
        loudest = max(core_available, key=lambda name: normalized[name]["signal"])
        conflict = (
            normalized[loudest]["signal"] >= self.contradiction_signal_min and risk == "LOW"
        )

        expert_required = (
            risk in ("HIGH", "MEDIUM")
            or conflict
            or (farmer["available"] and farmer["severity"] == "high")
        )

        # Reasons in the canonical (weights-table) order: image, satellite,
        # weather, nearby reports — then contextual / safety notes.
        reasons: list[str] = [
            normalized[name]["reason"]
            for name in CORE_SIGNALS
            if normalized[name]["available"] and normalized[name]["reason"]
        ]
        if farmer["available"]:
            reasons.append(farmer["reason"])
        if conflict:
            reasons.append(REASON_CONFLICT)
        if len(core_available) == 1:
            reasons.append(REASON_SINGLE_SIGNAL)

        breakdown: dict[str, dict[str, Any]] = {}
        for name in CORE_SIGNALS:
            entry = normalized[name]
            breakdown[name] = {
                "available": entry["available"],
                "signal": round(entry["signal"], 4),
                "weight": effective.get(name, 0.0),
                "configured_weight": self.weights[name],
                "contribution": contributions.get(name, 0.0),
                "note": entry["note"],
            }
        breakdown["farmer_report"] = {
            "available": farmer["available"],
            "severity": farmer.get("severity"),
            "adjustment": round(adjustment, 4) if farmer["available"] else 0.0,
            "note": (
                "farmer-reported severity nudges the score by at most "
                f"+/-{self.context_adjustment_cap:.2f} (contextual signal, no prototype weight)"
                if farmer["available"]
                else "not provided"
            ),
        }

        explain_terms = " + ".join(
            f"{effective[name]:.2f}x{normalized[name]['signal']:.2f} ({name})" for name in core_available
        )
        explain = (
            f"{explain_terms}"
            + (f" {'+' if adjustment >= 0 else '-'} {abs(adjustment):.2f} (farmer report)" if adjustment else "")
            + f" = {score:.3f} -> {risk} (HIGH >= {self.thresholds['high']:.2f}, "
            f"MEDIUM >= {self.thresholds['medium']:.2f}, else LOW)"
        )
        if len(core_available) < len(CORE_SIGNALS):
            explain += f"; weights renormalised over available signals: {', '.join(core_available)}"

        card: dict[str, Any] = {
            "risk": risk,
            "score": round(score, 3),
            "score_percent": int(round(score * 100)),
            "expert_required": expert_required,
            "reasons": reasons,
            "breakdown": breakdown,
            "weights": dict(self.weights),
            "weights_used": {name: effective[name] for name in core_available},
            "weights_note": PROTOTYPE_WEIGHTS_NOTE,
            "signals_used": list(core_available),
            "signals_missing": [name for name in CORE_SIGNALS if name not in core_available],
            "context_adjustment": {
                "applied": round(adjustment, 4),
                "cap": self.context_adjustment_cap,
                "source": adjustment_source,
            },
            "thresholds": dict(self.thresholds),
            "explain": explain,
            "context": merged_context,
        }
        card["whatsapp"] = {lang: whatsapp_advisory(card, language=lang) for lang in ("en", "hi", "mr")}
        return card


# ---------------------------------------------------------------------------
# WhatsApp advisory formatting
# ---------------------------------------------------------------------------


def whatsapp_advisory(risk_card: Mapping[str, Any], language: str = "en") -> str:
    """Format a risk card as a short WhatsApp advisory message.

    ``language`` is one of ``en`` (default), ``hi`` or ``mr``; unknown
    languages fall back to English.  Known reason strings are translated;
    anything else is passed through unchanged.
    """
    lang = language if language in _ADVISORY_TEXT["title"] else "en"
    level = str(risk_card.get("risk", "LOW"))
    score = risk_card.get("score", 0.0)
    level_text = _RISK_LEVEL_TEXT.get(level, {}).get(lang, level)

    lines = [
        f"\U0001F33E {_ADVISORY_TEXT['title'][lang]}",
        _ADVISORY_TEXT['risk_line'][lang].format(level=level_text, score=f"{float(score):.2f}"),
    ]
    reasons = list(risk_card.get("reasons") or [])
    if reasons:
        lines.append(_ADVISORY_TEXT['why'][lang])
        lines.extend(f"- {translate_reason(reason, lang)}" for reason in reasons)
    if risk_card.get("expert_required"):
        lines.append(_ADVISORY_TEXT['expert'][lang])
    else:
        lines.append(_ADVISORY_TEXT['no_expert'][lang])
    lines.append(_ADVISORY_TEXT['note'][lang])
    return "\n".join(lines)


def translate_reason(reason: str, language: str) -> str:
    """Translate a known English reason string; unknown strings pass through."""
    translations = _REASON_TRANSLATIONS.get(reason)
    if translations and language in translations:
        return translations[language]
    return reason