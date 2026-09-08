from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

URGENCY_KEYWORDS = [
    "spread",
    "fast",
    "rapid",
    "dying",
    "wilt",
    "wilting",
    "severe",
    "heavy loss",
    "destroy",
    "पसर",
    "वाळ",
    "नुकसान",
    "करपा",
    "सुख",
    "मर",
    "खराब",
]


@dataclass(frozen=True)
class RiskSignal:
    name: str
    value: str
    weight: float
    contribution: float
    explanation: str


@dataclass(frozen=True)
class RiskAssessment:
    risk_level: str
    risk_score: float
    signals: list[dict[str, Any]]
    reasoning: str
    disclaimer: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "signals": self.signals,
            "reasoning": self.reasoning,
            "disclaimer": self.disclaimer,
        }


class RiskFusionService:
    """Transparent, deterministic multi-signal risk fusion service.

    Fuses leaf AI diagnosis, satellite anomaly detections, localized outbreak density,
    and farmer description urgency into a clear, explainable triage score.
    Strictly rule-based and deterministic — no LLM hallucinations or black-box predictions.
    """

    PROTOTYPE_DISCLAIMER = (
        "PROTOTYPE_ASSESSMENT: Deterministic rule-based index for triage prioritization. "
        "Not clinically or agronomically validated. Always consult local KVK or extension officers."
    )

    WEIGHTS = {
        "leaf_diagnosis": 0.40,
        "satellite_anomaly": 0.35,
        "outbreak_context": 0.15,
        "symptom_urgency": 0.10,
    }

    def assess_risk(
        self,
        crop: str,
        diagnosis_result: dict[str, Any] | None = None,
        satellite_result: dict[str, Any] | None = None,
        farmer_description: str | None = None,
        nearby_cases_count: int | None = None,
        coordinates_available: bool | None = None,
        monitored_radius_km: float = 10.0,
    ) -> RiskAssessment:
        signals: list[RiskSignal] = []

        # Resolve geographic coordinate availability and nearby cases count
        if coordinates_available is None:
            if nearby_cases_count is not None and nearby_cases_count > 0:
                coords_ok = True
                nearby_count = int(nearby_cases_count)
            else:
                coords_ok = False
                nearby_count = 0
        else:
            coords_ok = bool(coordinates_available)
            nearby_count = max(0, int(nearby_cases_count)) if nearby_cases_count is not None else 0

        radius_str = (
            f"{monitored_radius_km:.0f} km"
            if monitored_radius_km == int(monitored_radius_km)
            else f"{monitored_radius_km} km"
        )

        # 1. Leaf AI Signal (Weight: 0.40)
        w_leaf = self.WEIGHTS["leaf_diagnosis"]
        leaf_diag_name: str | None = None
        is_healthy_leaf = False
        has_leaf_diagnosis = False
        leaf_quality_rejected = False
        leaf_conf = 0.0
        leaf_status: str | None = None

        if not diagnosis_result or not diagnosis_result.get("diagnosis"):
            if diagnosis_result and diagnosis_result.get("status") == "IMAGE_QUALITY_REJECTED":
                c_leaf = 0.35
                val_leaf = "Image quality check failed"
                exp_leaf = "Photo unreadable (excessive blur, inadequate resolution, or poor lighting); leaf evidence is unverified."
                leaf_quality_rejected = True
            else:
                c_leaf = 0.20
                val_leaf = "Leaf evidence unavailable"
                exp_leaf = "No leaf photo or diagnosis provided; leaf evidence is unavailable/unverified."
        else:
            has_leaf_diagnosis = True
            diag = str(diagnosis_result.get("diagnosis"))
            leaf_diag_name = diag
            leaf_status = str(diagnosis_result.get("status", "AI_CONFIDENT"))
            conf = float(diagnosis_result.get("confidence", 0.0))
            leaf_conf = conf

            if "healthy" in diag.lower():
                is_healthy_leaf = True
                c_leaf = 0.05
                val_leaf = f"Healthy foliage detected ({conf * 100:.1f}% confidence)"
                exp_leaf = "Leaf model confirmed healthy foliage, substantially reducing disease risk."
            elif leaf_status == "LOW_CONFIDENCE":
                c_leaf = 0.50
                val_leaf = f"Low confidence prediction for {diag} ({conf * 100:.1f}%)"
                exp_leaf = f"Model uncertainty is high (< 45% confidence for {diag}); requires expert review."
            elif leaf_status == "REVIEW_RECOMMENDED":
                c_leaf = 0.65
                val_leaf = f"Ambiguous diagnosis: {diag} ({conf * 100:.1f}%)"
                exp_leaf = f"Ambiguous prediction for {diag} due to narrow probability margin or entropy; expert review recommended."
            else:  # AI_CONFIDENT
                advice = diagnosis_result.get("advice") or {}
                severity = str(advice.get("severity_guidance", "Moderate")).lower()
                if "high" in severity or "severe" in severity or "तीव्र" in severity:
                    sev_mult = 1.0
                elif "low" in severity or "कमी" in severity:
                    sev_mult = 0.6
                else:
                    sev_mult = 0.8
                c_leaf = round(min(1.0, max(0.20, conf * sev_mult)), 4)
                val_leaf = f"{diag} ({conf * 100:.1f}% confidence, severity: {advice.get('severity_guidance', 'Moderate')})"
                exp_leaf = f"AI leaf model predicted {diag} with {advice.get('severity_guidance', 'Moderate')} severity."

        signals.append(RiskSignal("leaf_diagnosis", val_leaf, w_leaf, c_leaf, exp_leaf))

        # 2. Satellite Anomaly Signal (Weight: 0.35)
        w_sat = self.WEIGHTS["satellite_anomaly"]
        sat_observed = False
        has_sat_anomaly = False

        if satellite_result and satellite_result.get("status") == "SUCCESS":
            sat_observed = True
            score = float(satellite_result.get("farm_local_abnormal_score", 0.0))
            summary = satellite_result.get("summary", {})
            clusters = int(summary.get("connected_clusters", satellite_result.get("connected_clusters", 0)))
            modality = str(satellite_result.get("modality", summary.get("modality", "COMBINED"))).upper()

            # Normalize anomaly score (0.0 to 1.5+ mapped into 0.0 - 1.0)
            norm_score = min(1.0, max(0.0, score / 1.5))
            if clusters >= 1:
                norm_score = min(1.0, norm_score * 1.15 + 0.1)
                has_sat_anomaly = True
            elif score >= 0.5:
                has_sat_anomaly = True
            c_sat = round(norm_score, 4)

            if modality == "SAR_ONLY":
                val_sat = f"SAR radar anomaly score {score:.2f} with {clusters} connected cluster(s)"
                exp_sat = (
                    f"Sentinel-1 SAR radar screening detected {clusters} spatial anomaly cluster(s) "
                    "(radar backscatter/moisture/structure variation due to cloud cover; not pathogen confirmation)."
                )
            elif modality == "OPTICAL_ONLY":
                val_sat = f"Optical anomaly score {score:.2f} with {clusters} connected hotspot cluster(s)"
                exp_sat = (
                    f"Sentinel-2 optical screening detected {clusters} spatial anomaly cluster(s) across plot "
                    "(canopy reflectance/chlorophyll variation; not pathogen confirmation)."
                )
            else:
                val_sat = f"Anomaly score {score:.2f} with {clusters} connected hotspot cluster(s)"
                exp_sat = (
                    f"Sentinel-1/2 Earth Engine analysis detected {clusters} spatial anomaly cluster(s) across plot "
                    "(canopy reflectance/moisture variation; not pathogen confirmation)."
                )
        else:
            c_sat = 0.25
            reason = (satellite_result or {}).get("reason", "satellite screening unperformed or unconfigured")
            val_sat = "Satellite evidence unavailable"
            exp_sat = f"Satellite evidence is unavailable/unobserved ({reason}); neutral baseline used without implying an observed anomaly."

        signals.append(RiskSignal("satellite_anomaly", val_sat, w_sat, c_sat, exp_sat))

        # 3. Local Outbreak Context (Weight: 0.15)
        w_hist = self.WEIGHTS["outbreak_context"]
        if not coords_ok:
            c_hist = 0.15
            val_hist = "Nearby cases could not be assessed"
            exp_hist = "Location coordinates were not provided; geographically-nearby case density could not be assessed."
        else:
            if nearby_count >= 3:
                c_hist = 0.85
                val_hist = f"{nearby_count} geographically-nearby cases within {radius_str} (last 14 days)"
                exp_hist = f"Active local cluster: {nearby_count} geographically-nearby same-crop cases documented within {radius_str} in the last 14 days."
            elif nearby_count in (1, 2):
                c_hist = 0.50
                val_hist = f"{nearby_count} geographically-nearby case(s) within {radius_str} (last 14 days)"
                exp_hist = f"Moderate local activity: {nearby_count} geographically-nearby same-crop case(s) documented within {radius_str} in the last 14 days."
            else:
                c_hist = 0.15
                val_hist = f"0 geographically-nearby cases within {radius_str}"
                exp_hist = f"No geographically-nearby cases were identified within {radius_str} in the last 14 days (does not rule out broader regional incidence)."

        signals.append(RiskSignal("outbreak_context", val_hist, w_hist, c_hist, exp_hist))

        # 4. Farmer Symptom Urgency (Weight: 0.10)
        w_desc = self.WEIGHTS["symptom_urgency"]
        desc_text = (farmer_description or "").lower()
        has_urgent_keywords = any(re.search(rf"\b{re.escape(k)}", desc_text) for k in URGENCY_KEYWORDS)
        if has_urgent_keywords:
            c_desc = 0.80
            val_desc = "Urgent symptoms reported in description"
            exp_desc = "Farmer narrative reports rapid spread, severe wilting, or heavy crop loss."
        elif farmer_description and farmer_description.strip():
            c_desc = 0.20
            val_desc = "Standard symptom description"
            exp_desc = "Farmer narrative does not contain emergency outbreak keywords."
        else:
            c_desc = 0.20
            val_desc = "No symptom narrative provided"
            exp_desc = "Farmer description was not provided; standard baseline applied."

        signals.append(RiskSignal("symptom_urgency", val_desc, w_desc, c_desc, exp_desc))

        # Composite Weighted Calculation
        total_risk = sum(sig.weight * sig.contribution for sig in signals)
        total_risk = round(min(1.0, max(0.0, total_risk)), 4)

        if total_risk >= 0.75:
            risk_level = "CRITICAL"
        elif total_risk >= 0.55:
            risk_level = "HIGH"
        elif total_risk >= 0.35:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        # Construct honest reasoning grounded strictly in available evidence
        drivers: list[str] = []
        if has_leaf_diagnosis and not is_healthy_leaf:
            if leaf_status == "AI_CONFIDENT":
                drivers.append(f"AI leaf prediction of {leaf_diag_name}")
            else:
                drivers.append(f"uncertain leaf prediction of {leaf_diag_name}")

        if sat_observed and has_sat_anomaly:
            drivers.append("satellite canopy anomaly/stress observed (not disease confirmation)")

        if coords_ok and nearby_count > 0:
            drivers.append(f"{nearby_count} geographically-nearby cases within {radius_str}")

        if has_urgent_keywords:
            drivers.append("urgent symptom progression reported by farmer")

        evidence_caveats: list[str] = []
        if not has_leaf_diagnosis:
            if leaf_quality_rejected:
                evidence_caveats.append("leaf photo quality check failed (unverified)")
            else:
                evidence_caveats.append("leaf evidence is unavailable/unverified")

        if not sat_observed:
            evidence_caveats.append("satellite evidence is unavailable/unobserved")

        if not coords_ok:
            evidence_caveats.append("nearby cases could not be assessed (missing coordinates)")
        elif nearby_count == 0:
            evidence_caveats.append(f"no geographically-nearby cases identified within {radius_str}")

        if risk_level == "CRITICAL":
            drivers_str = "; ".join(drivers) if drivers else "multiple elevated risk factors"
            caveats_str = f" Note: {'; '.join(evidence_caveats)}." if evidence_caveats else ""
            reasoning = (
                f"Critical risk: High-severity indicators detected ({drivers_str}). "
                f"Urgent field inspection recommended.{caveats_str}"
            )
        elif risk_level == "HIGH":
            drivers_str = "; ".join(drivers) if drivers else "elevated risk signals"
            caveats_str = f" Note: {'; '.join(evidence_caveats)}." if evidence_caveats else ""
            reasoning = (
                f"High risk: Elevated distress signals detected ({drivers_str}). "
                f"Prompt agronomist review recommended.{caveats_str}"
            )
        elif risk_level == "MODERATE":
            if drivers:
                drivers_str = "; ".join(drivers)
                caveats_str = f" Note: {'; '.join(evidence_caveats)}." if evidence_caveats else ""
                reasoning = (
                    f"Moderate risk: Noticeable indicators detected ({drivers_str}). "
                    f"Advisory review recommended.{caveats_str}"
                )
            else:
                caveats_str = f" ({'; '.join(evidence_caveats)})" if evidence_caveats else ""
                reasoning = (
                    f"Moderate risk: Baseline risk threshold reached, though specific pathogen signals remain unconfirmed{caveats_str}."
                )
        else:  # LOW
            if drivers:
                drivers_str = "; ".join(drivers)
                caveats_str = f" Note: {'; '.join(evidence_caveats)}." if evidence_caveats else ""
                reasoning = (
                    f"Low risk: Overall composite risk remains low, but elevated signal observed: {drivers_str}.{caveats_str}"
                )
            elif is_healthy_leaf:
                notes_str = f" ({'; '.join(evidence_caveats)})" if evidence_caveats else ""
                reasoning = f"Low risk: Leaf model diagnosed healthy foliage with no urgent stress signals reported.{notes_str}"
            elif not has_leaf_diagnosis:
                notes_str = "; ".join(evidence_caveats)
                reasoning = f"Low risk: Monitored indicators show no elevated distress, but {notes_str}."
            else:
                reasoning = "Low risk: Signals indicate negligible agricultural distress across monitored indicators."

        return RiskAssessment(
            risk_level=risk_level,
            risk_score=total_risk,
            signals=[asdict(sig) for sig in signals],
            reasoning=reasoning,
            disclaimer=self.PROTOTYPE_DISCLAIMER,
        )
