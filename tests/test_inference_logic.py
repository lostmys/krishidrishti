from krishidrishti_ai.services.inference import DiagnosisService


def test_confidence_statuses() -> None:
    service = DiagnosisService.__new__(DiagnosisService)
    service.config = {"inference": {"confident_threshold": 0.75, "review_threshold": 0.45}}
    assert service._status(0.75) == "AI_CONFIDENT"
    assert service._status(0.5) == "REVIEW_RECOMMENDED"
    assert service._status(0.2) == "LOW_CONFIDENCE"
