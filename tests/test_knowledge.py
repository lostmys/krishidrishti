from __future__ import annotations

import pytest
from krishidrishti_ai.services.knowledge import DiseaseKnowledgeService


def test_knowledge_lookup_tomato_bacterial_spot() -> None:
    service = DiseaseKnowledgeService()
    advice = service.get_advice("Tomato", "Bacterial_spot", "AI_CONFIDENT", "en")
    assert advice["display_name"] == "Bacterial Spot"
    assert "Xanthomonas" in advice["description"]
    assert len(advice["typical_symptoms"]) > 0
    assert len(advice["immediate_actions"]) > 0
    assert len(advice["prevention"]) > 0
    assert advice["severity_guidance"] != ""


def test_knowledge_lookup_soyabean_rust() -> None:
    service = DiseaseKnowledgeService()
    advice = service.get_advice("Soyabean", "Rust", "AI_CONFIDENT", "en")
    assert advice["display_name"] == "Soybean Rust"
    assert "Phakopsora pachyrhizi" in advice["description"]
    assert len(advice["typical_symptoms"]) > 0
    assert len(advice["immediate_actions"]) > 0


def test_knowledge_lookup_healthy_crop() -> None:
    service = DiseaseKnowledgeService()
    for lang, expected_token in [("en", "healthy"), ("hi", "स्वस्थ"), ("mr", "निरोगी")]:
        advice = service.get_advice("Tomato", "healthy", "AI_CONFIDENT", lang)
        assert expected_token in advice["display_name"].lower()
        # Verify no pesticide prescriptions for healthy crops
        for action in advice["immediate_actions"]:
            assert "pesticide" not in action.lower()
            assert "कीटनाशक" not in action
            assert "कीटकनाशक" not in action


def test_knowledge_lookup_unknown_diagnosis() -> None:
    service = DiseaseKnowledgeService()
    # English
    adv_en = service.get_advice("Tomato", "NonExistentDisease", "AI_CONFIDENT", "en")
    assert "NonExistentDisease" in adv_en["display_name"]
    assert "Consult the local agricultural extension officer" in adv_en["immediate_actions"][0]

    # Hindi
    adv_hi = service.get_advice("Tomato", "NonExistentDisease", "AI_CONFIDENT", "hi")
    assert "अज्ञात रोग" in adv_hi["display_name"]
    assert "कृषि विस्तार अधिकारी" in adv_hi["immediate_actions"][0]

    # Marathi
    adv_mr = service.get_advice("Tomato", "NonExistentDisease", "AI_CONFIDENT", "mr")
    assert "अज्ञात रोग" in adv_mr["display_name"]
    assert "कृषी विस्तार अधिकारी" in adv_mr["immediate_actions"][0]


def test_multilingual_responses() -> None:
    service = DiseaseKnowledgeService()

    # English
    advice_en = service.get_advice("Soyabean", "Rust", "AI_CONFIDENT", "en")
    assert advice_en["display_name"] == "Soybean Rust"

    # Hindi
    advice_hi = service.get_advice("Soyabean", "Rust", "AI_CONFIDENT", "hi")
    assert advice_hi["display_name"] == "सोयाबीन गेरुई / तांबेरा (Soybean Rust)"
    assert "फेकोप्सोरा" in advice_hi["description"]

    # Marathi
    advice_mr = service.get_advice("Soyabean", "Rust", "AI_CONFIDENT", "mr")
    assert advice_mr["display_name"] == "सोयाबीन तांबेरा (Soybean Rust)"
    assert "फॅकोप्सोरा" in advice_mr["description"]


def test_analysis_generation_and_visual_limitations() -> None:
    service = DiseaseKnowledgeService()
    top_preds = [
        {"crop": "Tomato", "diagnosis": "Septoria_leaf_spot", "confidence": 0.697},
        {"crop": "Tomato", "diagnosis": "Early_blight", "confidence": 0.285},
    ]

    # English Analysis
    analysis_en = service.generate_analysis(
        "Tomato", "Septoria_leaf_spot", 0.697, "REVIEW_RECOMMENDED", "en", is_ambiguous=True, top_predictions=top_preds
    )
    assert "Septoria Leaf Spot" in analysis_en["summary"]
    assert "69.7%" in analysis_en["summary"]
    assert "Early Blight" in analysis_en["uncertainty_note"]
    assert "28.5%" in analysis_en["uncertainty_note"]
    assert "does not perform microscopic" in analysis_en["visual_limitations_note"]

    # Hindi Analysis
    analysis_hi = service.generate_analysis(
        "Tomato", "Septoria_leaf_spot", 0.697, "REVIEW_RECOMMENDED", "hi", is_ambiguous=True, top_predictions=top_preds
    )
    assert "सेप्टोरिया पत्ती धब्बा" in analysis_hi["summary"]
    assert "अगेती झुलसा" in analysis_hi["uncertainty_note"]
    assert "28.5%" in analysis_hi["uncertainty_note"]
    assert "सूक्ष्मदर्शीय" in analysis_hi["visual_limitations_note"]

    # Marathi Analysis
    analysis_mr = service.generate_analysis(
        "Tomato", "Septoria_leaf_spot", 0.697, "REVIEW_RECOMMENDED", "mr", is_ambiguous=True, top_predictions=top_preds
    )
    assert "सेप्टोरिया पानांवरील ठिपके" in analysis_mr["summary"]
    assert "लवकर येणारा करपा" in analysis_mr["uncertainty_note"]
    assert "28.5%" in analysis_mr["uncertainty_note"]
    assert "सूक्ष्मदर्शक" in analysis_mr["visual_limitations_note"]


def test_status_ai_confident() -> None:
    service = DiseaseKnowledgeService()
    advice = service.get_advice("Tomato", "Early_blight", "AI_CONFIDENT", "en")
    assert advice["display_name"] == "Early Blight"
    assert advice["status_guidance"] != ""
    assert not any(action.startswith("NOTICE:") for action in advice["immediate_actions"])


def test_status_low_confidence() -> None:
    service = DiseaseKnowledgeService()
    for lang in ["en", "hi", "mr"]:
        advice = service.get_advice("Tomato", "Early_blight", "LOW_CONFIDENCE", lang)
        assert advice["severity_guidance"] != ""
        assert len(advice["typical_symptoms"]) == 0
        assert len(advice["immediate_actions"]) > 0


def test_status_image_quality_rejected() -> None:
    service = DiseaseKnowledgeService()
    for lang in ["en", "hi", "mr"]:
        advice = service.get_advice(None, None, "IMAGE_QUALITY_REJECTED", lang)
        assert advice["severity_guidance"] != ""
        assert len(advice["immediate_actions"]) > 0


def test_all_15_diseases_translated_completely() -> None:
    """Verify that every single one of the 15 classes has full non-empty EN, HI, MR translations."""
    service = DiseaseKnowledgeService()
    assert len(service.knowledge_data) >= 15
    for entry in service.knowledge_data:
        crop = entry["crop"]
        diag = entry["diagnosis"]
        for lang in ["en", "hi", "mr"]:
            advice = service.get_advice(crop, diag, "AI_CONFIDENT", language=lang)
            assert advice["display_name"], f"Empty display_name for {crop} {diag} in {lang}"
            assert advice["description"], f"Empty description for {crop} {diag} in {lang}"
            assert len(advice["typical_symptoms"]) > 0, f"Empty symptoms for {crop} {diag} in {lang}"
            assert len(advice["immediate_actions"]) > 0, f"Empty actions for {crop} {diag} in {lang}"
            assert len(advice["prevention"]) > 0, f"Empty prevention for {crop} {diag} in {lang}"
            assert advice["severity_guidance"], f"Empty severity for {crop} {diag} in {lang}"
