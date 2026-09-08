from __future__ import annotations

from krishidrishti_ai.services.fusion import RiskFusionService


def test_fusion_healthy_leaf_reduces_risk() -> None:
    service = RiskFusionService()
    diag = {
        "crop": "Tomato",
        "diagnosis": "healthy",
        "confidence": 0.98,
        "status": "AI_CONFIDENT",
        "advice": {"severity_guidance": "None"},
    }
    result = service.assess_risk("Tomato", diagnosis_result=diag)
    assert result.risk_score < 0.35
    assert result.risk_level == "LOW"
    assert "PROTOTYPE_ASSESSMENT" in result.disclaimer
    leaf_signal = next(s for s in result.signals if s["name"] == "leaf_diagnosis")
    assert leaf_signal["contribution"] == 0.05


def test_fusion_high_severity_disease_and_satellite_anomaly_yields_critical_or_high() -> None:
    service = RiskFusionService()
    diag = {
        "crop": "Soyabean",
        "diagnosis": "Rust",
        "confidence": 0.95,
        "status": "AI_CONFIDENT",
        "advice": {"severity_guidance": "Severe. Spreads rapidly in warm, humid weather."},
    }
    sat = {
        "status": "SUCCESS",
        "farm_local_abnormal_score": 1.2,
        "connected_clusters": 3,
    }
    result = service.assess_risk(
        crop="Soyabean",
        diagnosis_result=diag,
        satellite_result=sat,
        farmer_description="Foliage dying fast, yellowing spreads quickly across plot",
        nearby_cases_count=4,
    )
    assert result.risk_level in ("HIGH", "CRITICAL")
    assert result.risk_score >= 0.70


def test_fusion_unobserved_satellite_uses_neutral_baseline() -> None:
    service = RiskFusionService()
    result = service.assess_risk(
        crop="Cotton",
        diagnosis_result=None,
        satellite_result={"status": "UNAVAILABLE", "reason": "No credentials"},
    )
    sat_signal = next(s for s in result.signals if s["name"] == "satellite_anomaly")
    assert sat_signal["contribution"] == 0.25
    assert "neutral baseline" in sat_signal["explanation"].lower()


def test_fusion_urgency_keywords_multilingual() -> None:
    service = RiskFusionService()

    # Marathi urgency
    res_mr = service.assess_risk("Cotton", farmer_description="पाने करपून वाळत चालली आहेत, रोग पसरतोय")
    desc_signal_mr = next(s for s in res_mr.signals if s["name"] == "symptom_urgency")
    assert desc_signal_mr["contribution"] == 0.80

    # Hindi urgency
    res_hi = service.assess_risk("Tomato", farmer_description="पौधे सूख रहे हैं और नुकसान बढ़ रहा है")
    desc_signal_hi = next(s for s in res_hi.signals if s["name"] == "symptom_urgency")
    assert desc_signal_hi["contribution"] == 0.80

    # Normal description
    res_norm = service.assess_risk("Tomato", farmer_description="Minor yellow spots on lower leaves")
    desc_signal_norm = next(s for s in res_norm.signals if s["name"] == "symptom_urgency")
    assert desc_signal_norm["contribution"] == 0.20


def test_fusion_image_quality_rejected_handles_gracefully() -> None:
    service = RiskFusionService()
    diag = {
        "crop": None,
        "diagnosis": None,
        "confidence": 0.0,
        "status": "IMAGE_QUALITY_REJECTED",
    }
    result = service.assess_risk("Tomato", diagnosis_result=diag)
    leaf_signal = next(s for s in result.signals if s["name"] == "leaf_diagnosis")
    assert leaf_signal["contribution"] == 0.35
    assert "quality check failed" in leaf_signal["value"].lower()


def test_fusion_no_diagnosis_does_not_produce_healthy_foliage_language() -> None:
    service = RiskFusionService()
    result = service.assess_risk("Tomato", diagnosis_result=None)
    assert "healthy" not in result.reasoning.lower()
    assert "foliage appears healthy" not in result.reasoning.lower()
    assert "leaf evidence is unavailable/unverified" in result.reasoning.lower()
    leaf_signal = next(s for s in result.signals if s["name"] == "leaf_diagnosis")
    assert "leaf evidence is unavailable/unverified" in leaf_signal["explanation"].lower()
    assert "healthy" not in leaf_signal["value"].lower()


def test_fusion_unavailable_satellite_does_not_imply_observed_anomaly() -> None:
    service = RiskFusionService()
    result = service.assess_risk(
        "Tomato",
        satellite_result={"status": "UNAVAILABLE", "reason": "Credentials not configured"},
    )
    sat_signal = next(s for s in result.signals if s["name"] == "satellite_anomaly")
    assert "satellite evidence is unavailable/unobserved" in sat_signal["explanation"].lower()
    assert "without implying an observed anomaly" in sat_signal["explanation"].lower()
    assert "satellite evidence is unavailable/unobserved" in result.reasoning.lower()
    assert "anomaly observed" not in result.reasoning.lower()


def test_fusion_missing_coordinates_does_not_produce_false_nearby_claim() -> None:
    service = RiskFusionService()
    result = service.assess_risk("Tomato", coordinates_available=False, nearby_cases_count=0)
    outbreak_sig = next(s for s in result.signals if s["name"] == "outbreak_context")
    assert outbreak_sig["value"] == "Nearby cases could not be assessed"
    assert "0 nearby" not in outbreak_sig["value"].lower()
    assert "could not be assessed" in outbreak_sig["explanation"].lower()
    assert "nearby cases could not be assessed" in result.reasoning.lower()
    assert "0 geographically-nearby cases" not in result.reasoning.lower()


def test_fusion_zero_nearby_cases_identified_with_coordinates() -> None:
    service = RiskFusionService()
    result = service.assess_risk("Tomato", coordinates_available=True, nearby_cases_count=0)
    outbreak_sig = next(s for s in result.signals if s["name"] == "outbreak_context")
    assert "0 geographically-nearby cases" in outbreak_sig["value"]
    assert "no geographically-nearby cases were identified" in outbreak_sig["explanation"].lower()
    assert "no outbreak anywhere" not in outbreak_sig["explanation"].lower()
    assert "no geographically-nearby cases identified" in result.reasoning.lower()


def test_fusion_real_nearby_cases_described_as_geographically_nearby() -> None:
    service = RiskFusionService()
    result = service.assess_risk("Tomato", coordinates_available=True, nearby_cases_count=3)
    outbreak_sig = next(s for s in result.signals if s["name"] == "outbreak_context")
    assert "3 geographically-nearby cases" in outbreak_sig["value"]
    assert "geographically-nearby same-crop cases documented" in outbreak_sig["explanation"].lower()
    assert "3 geographically-nearby cases" in result.reasoning.lower()


def test_fusion_satellite_anomaly_not_described_as_disease_confirmation() -> None:
    service = RiskFusionService()
    sat = {
        "status": "SUCCESS",
        "farm_local_abnormal_score": 1.2,
        "connected_clusters": 2,
    }
    result = service.assess_risk("Tomato", satellite_result=sat)
    sat_sig = next(s for s in result.signals if s["name"] == "satellite_anomaly")
    assert "not pathogen confirmation" in sat_sig["explanation"].lower()
    assert "not disease confirmation" in result.reasoning.lower()

