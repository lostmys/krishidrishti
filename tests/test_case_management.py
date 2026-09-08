from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import krishidrishti_ai.api as _api_module
from krishidrishti_ai.api import app
from krishidrishti_ai.services.cases import CaseService


@pytest.fixture()
def temp_case_service(tmp_path: Path) -> CaseService:
    db_file = tmp_path / "test_krishidrishti.db"
    return CaseService(db_path=db_file)


@pytest.fixture()
def client(tmp_path: Path):
    test_cs = CaseService(db_path=tmp_path / "test_api_cases.db")
    with patch.object(_api_module, "_get_case_service", return_value=test_cs):
        yield TestClient(app, raise_server_exceptions=False)


def _jpeg_bytes(width: int = 128, height: int = 128) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color="green").save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Direct Service Unit Tests
# ---------------------------------------------------------------------------


def test_case_creation_and_retrieval(temp_case_service: CaseService) -> None:
    case = temp_case_service.create_case(
        crop="cotton",
        farmer_contact="+919988776655",
        farmer_name="Ramesh Patil",
        latitude=19.75,
        longitude=75.71,
        description="Foliage curl observed on upper canopy",
        initial_status="OPEN",
    )
    assert case["case_id"].startswith("KD-")
    assert case["crop"] == "cotton"
    assert case["current_status"] == "OPEN"
    assert case["farmer_name"] == "Ramesh Patil"

    retrieved = temp_case_service.get_case(case["case_id"])
    assert retrieved is not None
    assert retrieved["case_id"] == case["case_id"]
    assert retrieved["description"] == "Foliage curl observed on upper canopy"


def test_state_machine_valid_and_invalid_transitions(temp_case_service: CaseService) -> None:
    case = temp_case_service.create_case(crop="tomato", initial_status="OPEN")
    cid = case["case_id"]

    # Valid: OPEN -> FIELD_VISIT_REQUIRED
    rev1 = temp_case_service.add_expert_review(cid, "REQUEST_FIELD_VISIT", notes="Field inspection needed")
    assert rev1["current_status"] == "FIELD_VISIT_REQUIRED"
    assert len(rev1["expert_reviews_history"]) == 1

    # Valid: FIELD_VISIT_REQUIRED -> CONFIRMED
    rev2 = temp_case_service.add_expert_review(cid, "CONFIRM", notes="Confirmed in field")
    assert rev2["current_status"] == "CONFIRMED"

    # Valid: CONFIRMED -> RESOLVED
    fu1 = temp_case_service.add_follow_up(cid, "IMPROVED", notes="Leaves recovered after treatment")
    assert fu1["current_status"] == "RESOLVED"
    assert len(fu1["follow_up_history"]) == 1

    # Invalid: RESOLVED -> CONFIRMED (only OPEN allowed)
    with pytest.raises(ValueError, match="Illegal status transition"):
        temp_case_service.add_expert_review(cid, "CONFIRM", notes="Cannot re-confirm resolved case")


def test_followup_worse_condition_triggers_field_visit(temp_case_service: CaseService) -> None:
    case = temp_case_service.create_case(crop="soyabean", initial_status="OPEN")
    cid = case["case_id"]
    temp_case_service.add_expert_review(cid, "CONFIRM", notes="Initial confirmation")

    # WORSE condition on confirmed case triggers FIELD_VISIT_REQUIRED
    fu = temp_case_service.add_follow_up(cid, "WORSE", notes="Symptoms worsening rapidly")
    assert fu["current_status"] == "FIELD_VISIT_REQUIRED"


# ---------------------------------------------------------------------------
# API Integration Tests
# ---------------------------------------------------------------------------


def test_api_report_case_without_image(client: TestClient) -> None:
    resp = client.post(
        "/cases",
        data={
            "crop": "cotton",
            "farmer_name": "Suresh",
            "farmer_contact": "9876543210",
            "description": "Yellowing leaves",
            "language": "en",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["case_id"].startswith("KD-")
    assert body["crop"] == "cotton"
    assert body["current_status"] in ("OPEN", "REVIEW_RECOMMENDED")
    assert "risk_assessment" in body
    assert body["risk_assessment"]["risk_level"] in ("LOW", "MODERATE", "HIGH", "CRITICAL")


def test_api_report_case_with_image_and_multilingual_advisory(client: TestClient) -> None:
    mock_registry = MagicMock()
    mock_registry.available_crops = ["cotton", "soyabean", "tomato"]
    mock_diag_service = MagicMock()
    mock_diag_service.diagnose.return_value = {
        "crop": "Tomato",
        "diagnosis": "Early_blight",
        "confidence": 0.94,
        "status": "AI_CONFIDENT",
        "advice": {
            "display_name": "लवकर येणारा करपा",
            "description": "ऑल्टर्नेरिया सोलेनी बुरशीमुळे होतो",
            "symptoms": ["काळे-भुरे ठिपके"],
            "immediate_actions": ["खालची पाने कापा"],
            "prevention": ["पीकपालट करा"],
            "severity_guidance": "Moderate",
            "source": "ICAR-IARI",
        },
    }
    mock_registry.get.return_value = mock_diag_service

    with patch.object(_api_module, "_get_registry", return_value=mock_registry):
        resp = client.post(
            "/cases",
            data={
                "crop": "tomato",
                "farmer_name": "Anil",
                "farmer_contact": "9123456780",
                "description": "पानांवर काळे ठिपके पडत आहेत",
                "language": "mr",
            },
            files={"image": ("leaf.jpg", _jpeg_bytes(), "image/jpeg")},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["crop"] == "tomato"
    assert body["ai_diagnosis"] == "Early_blight"
    assert body["calibrated_confidence"] == 0.94
    assert "लवकर येणारा करपा" in body["advisory"]["display_name"]


def test_api_expert_review_and_followup_flow(client: TestClient) -> None:
    # 1. Create case
    c_resp = client.post(
        "/cases",
        data={"crop": "soyabean", "farmer_name": "Kisan", "description": "Leaf spots"},
    )
    assert c_resp.status_code == 200
    cid = c_resp.json()["case_id"]

    # 2. Expert review -> REQUEST_FIELD_VISIT
    rev_resp = client.post(
        f"/cases/{cid}/review",
        json={"action": "REQUEST_FIELD_VISIT", "reviewer_name": "Dr. Joshi", "notes": "Need to check field soil"},
    )
    assert rev_resp.status_code == 200
    assert rev_resp.json()["current_status"] == "FIELD_VISIT_REQUIRED"

    # 3. Follow-up
    fu_resp = client.post(
        f"/cases/{cid}/follow-up",
        json={"condition": "SAME", "notes": "No changes yet"},
    )
    assert fu_resp.status_code == 200
    assert len(fu_resp.json()["follow_up_history"]) == 1

    # 4. List cases
    list_resp = client.get("/cases?crop=soyabean")
    assert list_resp.status_code == 200
    assert any(c["case_id"] == cid for c in list_resp.json()["cases"])

    # 5. Get case with Hindi translation
    get_resp = client.get(f"/cases/{cid}?language=hi")
    assert get_resp.status_code == 200
    assert get_resp.json()["case_id"] == cid


def test_count_nearby_recent_cases_geographic_proximity(temp_case_service: CaseService) -> None:
    # Base location: Jalna, Maharashtra
    base_lat = 19.8410
    base_lon = 75.8863

    # a) Case inside radius (~1.5 km away, cotton, recent)
    c_inside = temp_case_service.create_case(
        crop="cotton",
        latitude=19.8500,
        longitude=75.8900,
        description="Inside radius case",
    )

    # b) Case outside radius (~90 km away in Aurangabad, cotton, recent)
    c_outside = temp_case_service.create_case(
        crop="cotton",
        latitude=19.8762,
        longitude=75.3433,
        description="Outside radius case",
    )

    # c) Case different crop (inside radius ~1.5 km, tomato, recent)
    c_diff_crop = temp_case_service.create_case(
        crop="tomato",
        latitude=19.8500,
        longitude=75.8900,
        description="Different crop inside radius",
    )

    # e) Case with missing coordinates (cotton, recent, no coordinates)
    c_no_coords = temp_case_service.create_case(
        crop="cotton",
        latitude=None,
        longitude=None,
        description="Missing coordinates case",
    )

    # d) Case older than 14 days (inside radius, cotton, created 20 days ago)
    old_date = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
    c_old = temp_case_service.create_case(
        crop="cotton",
        latitude=19.8500,
        longitude=75.8900,
        description="Old case inside radius",
    )
    with temp_case_service._get_connection() as conn:
        conn.execute("UPDATE cases SET created_at = ? WHERE case_id = ?", (old_date, c_old["case_id"]))

    # Test a: Case inside radius IS counted (only c_inside should match)
    count = temp_case_service.count_nearby_recent_cases(
        crop="cotton",
        latitude=base_lat,
        longitude=base_lon,
        radius_km=10.0,
    )
    assert count == 1

    # Test b: Outside radius not counted (narrow radius excludes even c_inside)
    count_narrow = temp_case_service.count_nearby_recent_cases(
        crop="cotton",
        latitude=base_lat,
        longitude=base_lon,
        radius_km=0.5,
    )
    assert count_narrow == 0

    # Expand radius to 100 km to include c_outside as well
    count_wide = temp_case_service.count_nearby_recent_cases(
        crop="cotton",
        latitude=base_lat,
        longitude=base_lon,
        radius_km=100.0,
    )
    assert count_wide == 2

    # Test c: Different crop is not counted
    count_tomato = temp_case_service.count_nearby_recent_cases(
        crop="tomato",
        latitude=base_lat,
        longitude=base_lon,
        radius_km=10.0,
    )
    assert count_tomato == 1  # only c_diff_crop

    # Test d: Older than 14 days is not counted
    # (c_old is cotton and inside radius, but was backdated to 20 days ago, so count is still 1)
    assert count == 1

    # Test e: Case with missing coordinates is not counted (c_no_coords ignored)
    assert count == 1

    # Test f: Current location with missing coordinates safely returns 0
    assert temp_case_service.count_nearby_recent_cases("cotton", latitude=None, longitude=None) == 0
    assert temp_case_service.count_nearby_recent_cases("cotton", latitude=base_lat, longitude=None) == 0
    assert temp_case_service.count_nearby_recent_cases("cotton", latitude=None, longitude=base_lon) == 0
    assert temp_case_service.count_nearby_recent_cases("cotton") == 0  # backward compatibility default


def test_api_case_creation_geographic_outbreak_reasoning(client: TestClient) -> None:
    # 1. Create first case at unique isolated coordinates where no previous cases exist
    resp_with_coords1 = client.post(
        "/cases",
        data={
            "crop": "cotton",
            "farmer_name": "Ganesh",
            "latitude": 15.1234,
            "longitude": 74.5678,
            "description": "Leaf curl",
        },
    )
    assert resp_with_coords1.status_code == 200
    b_coords1 = resp_with_coords1.json()
    outbreak_sig_c1 = next(s for s in b_coords1["risk_assessment"]["signals"] if s["name"] == "outbreak_context")
    assert "0 geographically-nearby cases" in outbreak_sig_c1["value"]
    assert "could not be assessed" not in outbreak_sig_c1["value"].lower()

    # 2. Create second case ~100m away at (15.1240, 74.5680) - should detect case 1 as geographically nearby
    resp_with_coords2 = client.post(
        "/cases",
        data={
            "crop": "cotton",
            "farmer_name": "Rao",
            "latitude": 15.1240,
            "longitude": 74.5680,
            "description": "Leaf curl spread",
        },
    )
    assert resp_with_coords2.status_code == 200
    b_coords2 = resp_with_coords2.json()
    outbreak_sig_c2 = next(s for s in b_coords2["risk_assessment"]["signals"] if s["name"] == "outbreak_context")
    assert "1 geographically-nearby case" in outbreak_sig_c2["value"]

    # 3. Create case without coordinates
    resp_no_coords = client.post(
        "/cases",
        data={
            "crop": "cotton",
            "farmer_name": "Sanjay",
            "description": "Leaf curl",
        },
    )
    assert resp_no_coords.status_code == 200
    b_no_coords = resp_no_coords.json()
    outbreak_sig_nc = next(s for s in b_no_coords["risk_assessment"]["signals"] if s["name"] == "outbreak_context")
    assert outbreak_sig_nc["value"] == "Nearby cases could not be assessed"
    assert "0 nearby" not in outbreak_sig_nc["value"].lower()
    assert "nearby cases could not be assessed" in b_no_coords["risk_assessment"]["reasoning"].lower()


# ---------------------------------------------------------------------------
# Phase 2: Complete Farmer Case Workflow Tests (Requirements A - J)
# ---------------------------------------------------------------------------


def test_phase2_requirement_a_diagnosis_confidence_preservation(client: TestClient) -> None:
    """Requirement A: AI Diagnosis preserves calibrated confidence score and AI status."""
    mock_registry = MagicMock()
    mock_registry.available_crops = ["soyabean"]
    mock_diag = MagicMock()
    mock_diag.diagnose.return_value = {
        "crop": "Soyabean",
        "diagnosis": "Bacterial_blight",
        "confidence": 0.885,
        "raw_confidence": 0.942,
        "status": "AI_CONFIDENT",
        "advice": {
            "display_name": "Bacterial Blight",
            "description": "Pseudomonas savastanoi pv. glycinea infection",
            "immediate_actions": ["Avoid working in wet fields"],
            "prevention": ["Use certified disease-free seeds"],
            "severity_guidance": "Moderate",
            "source": "ICAR-IISR",
        },
    }
    mock_registry.get.return_value = mock_diag

    with patch.object(_api_module, "_get_registry", return_value=mock_registry):
        resp = client.post(
            "/cases",
            data={
                "crop": "soyabean",
                "farmer_name": "Kavita",
                "farmer_contact": "+919123456789",
                "latitude": "20.1234",
                "longitude": "76.5432",
                "description": "Water-soaked lesions on leaves",
            },
            files={"image": ("soy_leaf.jpg", _jpeg_bytes(), "image/jpeg")},
        )

    assert resp.status_code == 200
    case = resp.json()
    assert case["case_id"].startswith("KD-")
    assert case["ai_diagnosis"] == "Bacterial_blight"
    assert case["calibrated_confidence"] == pytest.approx(0.885, abs=1e-3)
    assert case["ai_status"] == "AI_CONFIDENT"
    assert case["current_status"] == "OPEN"


def test_phase2_requirement_b_risk_assessment_storage(client: TestClient) -> None:
    """Requirement B: Multi-source risk assessment is fused and saved with score and breakdown."""
    resp = client.post(
        "/cases",
        data={
            "crop": "tomato",
            "farmer_name": "Mahesh",
            "latitude": 19.5,
            "longitude": 75.5,
            "description": "Rapidly spreading dark spots across entire parcel",
        },
    )
    assert resp.status_code == 200
    case = resp.json()
    assert "risk_assessment" in case
    risk = case["risk_assessment"]
    assert "risk_score" in risk
    assert "risk_level" in risk
    assert risk["risk_level"] in ("LOW", "MODERATE", "HIGH", "CRITICAL")
    assert "signals" in risk
    signal_names = [s["name"] for s in risk["signals"]]
    assert "leaf_diagnosis" in signal_names
    assert "outbreak_context" in signal_names
    assert "satellite_anomaly" in signal_names
    assert "symptom_urgency" in signal_names
    assert "reasoning" in risk


def test_phase2_requirement_c_missing_and_empty_coordinates_handling(client: TestClient) -> None:
    """Requirement C: Empty string coordinates from web forms are safely handled without crash."""
    resp = client.post(
        "/cases",
        data={
            "crop": "cotton",
            "farmer_name": "Vijay",
            "latitude": "",   # Form input with empty string
            "longitude": "   ",  # Whitespace-only string
            "description": "Slight yellowing",
        },
    )
    assert resp.status_code == 200
    case = resp.json()
    assert case["latitude"] is None
    assert case["longitude"] is None
    outbreak_sig = next(s for s in case["risk_assessment"]["signals"] if s["name"] == "outbreak_context")
    assert outbreak_sig["value"] == "Nearby cases could not be assessed"
    assert "could not be assessed" in case["risk_assessment"]["reasoning"].lower()


def test_phase2_requirement_d_honest_satellite_screening(client: TestClient) -> None:
    """Requirement D: Satellite unavailability is stored honestly without false disease claims."""
    resp = client.post(
        "/cases",
        data={
            "crop": "tomato",
            "farmer_name": "Sunita",
            "latitude": "19.88",
            "longitude": "75.33",
            "description": "Checking satellite screening honesty",
        },
    )
    assert resp.status_code == 200
    case = resp.json()
    sat_status = case["satellite_status"]
    assert sat_status in ("UNAVAILABLE", "NOT_REQUESTED", "ERROR")
    # Satellite screening must never assert that it proved or confirmed the plant disease
    reasoning = case["risk_assessment"]["reasoning"].lower()
    assert "confirmed bacterial" not in reasoning
    assert "confirmed early blight" not in reasoning


def test_phase2_requirement_e_case_retrieval_lifecycle_audit(client: TestClient) -> None:
    """Requirement E: GET /cases/{case_id} returns full lifecycle audit, risk, and timeline."""
    c_resp = client.post(
        "/cases",
        data={"crop": "soyabean", "farmer_name": "Balaji", "description": "Audit trail test"},
    )
    assert c_resp.status_code == 200
    cid = c_resp.json()["case_id"]

    get_resp = client.get(f"/cases/{cid}")
    assert get_resp.status_code == 200
    retrieved = get_resp.json()
    assert retrieved["case_id"] == cid
    assert "expert_reviews_history" in retrieved
    assert "follow_up_history" in retrieved
    assert "followup_deadline" in retrieved
    assert "followup_due" in retrieved
    assert "risk_assessment" in retrieved


def test_phase2_requirement_f_followup_condition_transitions(client: TestClient) -> None:
    """Requirement F: Follow-up condition transitions (IMPROVED -> RESOLVED, WORSE -> FIELD_VISIT)."""
    # 1. Test WORSE -> triggers FIELD_VISIT_REQUIRED
    c1 = client.post("/cases", data={"crop": "tomato", "farmer_name": "Farmer 1"}).json()
    cid1 = c1["case_id"]
    fu_worse = client.post(
        f"/cases/{cid1}/follow-up",
        json={"condition": "WORSE", "observed_by": "Farmer", "notes": "Symptoms spread to all plants"},
    )
    assert fu_worse.status_code == 200
    assert fu_worse.json()["current_status"] == "FIELD_VISIT_REQUIRED"

    # 2. Test SAME -> retains current status
    fu_same = client.post(
        f"/cases/{cid1}/follow-up",
        json={"condition": "SAME", "observed_by": "Farmer", "notes": "No changes observed today"},
    )
    assert fu_same.status_code == 200
    assert fu_same.json()["current_status"] == "FIELD_VISIT_REQUIRED"

    # 3. Test IMPROVED -> transitions CONFIRMED to RESOLVED
    c2 = client.post("/cases", data={"crop": "cotton", "farmer_name": "Farmer 2"}).json()
    cid2 = c2["case_id"]
    client.post(
        f"/cases/{cid2}/review",
        json={"action": "CONFIRM", "reviewer_name": "Officer", "notes": "Diagnosis confirmed"},
    )
    fu_improved = client.post(
        f"/cases/{cid2}/follow-up",
        json={"condition": "IMPROVED", "observed_by": "Farmer", "notes": "New green leaves emerging"},
    )
    assert fu_improved.status_code == 200
    assert fu_improved.json()["current_status"] == "RESOLVED"


def test_phase2_requirement_g_officer_review_actions_and_correction(client: TestClient) -> None:
    """Requirement G: Officer actions (CONFIRM, REQUEST_FIELD_VISIT, REJECT, NOTE) & correction."""
    c = client.post("/cases", data={"crop": "soyabean", "farmer_name": "Farmer G"}).json()
    cid = c["case_id"]

    # 1. REQUEST_FIELD_VISIT
    r1 = client.post(
        f"/cases/{cid}/review",
        json={"action": "REQUEST_FIELD_VISIT", "reviewer_name": "Officer Patel", "notes": "Inspection scheduled"},
    ).json()
    assert r1["current_status"] == "FIELD_VISIT_REQUIRED"

    # 2. NOTE does not change status
    r2 = client.post(
        f"/cases/{cid}/review",
        json={"action": "NOTE", "reviewer_name": "Officer Patel", "notes": "Called farmer before visiting"},
    ).json()
    assert r2["current_status"] == "FIELD_VISIT_REQUIRED"

    # 3. CONFIRM with corrected diagnosis
    r3 = client.post(
        f"/cases/{cid}/review",
        json={
            "action": "CONFIRM",
            "reviewer_name": "Officer Patel",
            "notes": "Field visit completed, verified Rust",
            "corrected_diagnosis": "Rust",
        },
    ).json()
    assert r3["current_status"] == "CONFIRMED"
    assert r3["expert_reviews_history"][-1]["corrected_diagnosis"] == "Rust"


def test_phase2_requirement_h_state_machine_illegal_transition_rejection(client: TestClient) -> None:
    """Requirement H: State machine enforces invalid transition rejection with HTTP 400."""
    c = client.post("/cases", data={"crop": "cotton", "farmer_name": "Farmer H"}).json()
    cid = c["case_id"]

    # Confirm case
    client.post(
        f"/cases/{cid}/review",
        json={"action": "CONFIRM", "reviewer_name": "Officer", "notes": "Confirmed"},
    )
    # Resolve case via follow-up
    client.post(
        f"/cases/{cid}/follow-up",
        json={"condition": "IMPROVED", "observed_by": "Farmer", "notes": "All clear"},
    )
    # Attempting to re-confirm a resolved case is an illegal transition
    err_resp = client.post(
        f"/cases/{cid}/review",
        json={"action": "CONFIRM", "reviewer_name": "Officer", "notes": "Cannot confirm resolved"},
    )
    assert err_resp.status_code == 400
    assert "illegal status transition" in err_resp.json()["detail"].lower()


def test_phase2_requirement_i_multilingual_advisory_retranslation(client: TestClient) -> None:
    """Requirement I: Multilingual advisory localized in Marathi and Hindi on case retrieval."""
    mock_registry = MagicMock()
    mock_registry.available_crops = ["tomato"]
    mock_diag = MagicMock()
    mock_diag.diagnose.return_value = {
        "crop": "Tomato",
        "diagnosis": "Early_blight",
        "confidence": 0.92,
        "status": "AI_CONFIDENT",
        "advice": {
            "display_name": "Early Blight",
            "description": "Fungal infection",
            "immediate_actions": ["Prune lower leaves"],
            "prevention": ["Crop rotation"],
            "severity_guidance": "Moderate",
            "source": "ICAR",
        },
    }
    mock_registry.get.return_value = mock_diag

    with patch.object(_api_module, "_get_registry", return_value=mock_registry):
        c = client.post(
            "/cases",
            data={"crop": "tomato", "farmer_name": "Shantaram", "language": "en"},
            files={"image": ("leaf.jpg", _jpeg_bytes(), "image/jpeg")},
        ).json()

    cid = c["case_id"]

    # Retrieve in Marathi
    mr_resp = client.get(f"/cases/{cid}?language=mr")
    assert mr_resp.status_code == 200
    mr_body = mr_resp.json()
    assert mr_body["advisory"] is not None
    assert "लवकर येणारा करपा" in mr_body["advisory"]["display_name"]

    # Retrieve in Hindi
    hi_resp = client.get(f"/cases/{cid}?language=hi")
    assert hi_resp.status_code == 200
    hi_body = hi_resp.json()
    assert hi_body["advisory"] is not None
    assert "अगेती झुलसा" in hi_body["advisory"]["display_name"]


def test_phase2_requirement_j_full_end_to_end_farmer_journey(client: TestClient) -> None:
    """Requirement J: End-to-end farmer case workflow from photo upload to resolution."""
    mock_registry = MagicMock()
    mock_registry.available_crops = ["soyabean"]
    mock_diag = MagicMock()
    mock_diag.diagnose.return_value = {
        "crop": "Soyabean",
        "diagnosis": "Rust",
        "confidence": 0.95,
        "raw_confidence": 0.97,
        "status": "AI_CONFIDENT",
        "advice": {
            "display_name": "तांबेरा (Rust)",
            "description": "Phakopsora pachyrhizi बुरशीमुळे होतो",
            "immediate_actions": ["शिफारस केलेले बुरशीनाशक फवारा"],
            "prevention": ["योग्य वेळी पेरणी करा"],
            "severity_guidance": "High",
            "source": "ICAR-IISR",
        },
    }
    mock_registry.get.return_value = mock_diag

    with patch.object(_api_module, "_get_registry", return_value=mock_registry):
        # 1. Farmer uploads leaf photo and registers case
        post_resp = client.post(
            "/cases",
            data={
                "crop": "soyabean",
                "farmer_name": "Eknath Shinde",
                "farmer_contact": "+919876543210",
                "latitude": "19.8765",
                "longitude": "75.3456",
                "description": "पानांवर तपकिरी ठिपके दिसत आहेत",
                "language": "mr",
            },
            files={"image": ("rust_leaf.jpg", _jpeg_bytes(), "image/jpeg")},
        )
    assert post_resp.status_code == 200
    case = post_resp.json()
    cid = case["case_id"]
    assert cid.startswith("KD-")
    assert case["ai_diagnosis"] == "Rust"
    assert case["current_status"] == "OPEN"

    # 2. Case lookup by ID
    lookup_resp = client.get(f"/cases/{cid}?language=mr")
    assert lookup_resp.status_code == 200
    assert lookup_resp.json()["case_id"] == cid

    # 3. Agricultural Officer reviews and confirms case
    review_resp = client.post(
        f"/cases/{cid}/review",
        json={
            "action": "CONFIRM",
            "reviewer_name": "AO Deshmukh",
            "notes": "Verified leaf photo, matches Phakopsora symptoms",
            "language": "mr",
        },
    )
    assert review_resp.status_code == 200
    assert review_resp.json()["current_status"] == "CONFIRMED"

    # 4. Farmer reports follow-up after 48h indicating improvement
    fu_resp = client.post(
        f"/cases/{cid}/follow-up",
        json={
            "condition": "IMPROVED",
            "observed_by": "Farmer",
            "notes": "फवारणीनंतर नवीन निरोगी पाने फुटली",
            "language": "mr",
        },
    )
    assert fu_resp.status_code == 200
    resolved_case = fu_resp.json()
    assert resolved_case["current_status"] == "RESOLVED"
    assert len(resolved_case["follow_up_history"]) == 1
    assert len(resolved_case["expert_reviews_history"]) == 1


def test_update_case_satellite_with_risk(temp_case_service: CaseService):
    """Verify that update_case_satellite correctly writes satellite and risk JSON without corrupting columns."""
    case = temp_case_service.create_case(
        crop="tomato",
        latitude=19.8762,
        longitude=75.3433,
        description="Testing satellite update with risk",
    )
    case_id = case["case_id"]

    sat_data = {
        "status": "SUCCESS",
        "modality": "OPTICAL_ONLY",
        "anomaly_score": 0.42,
        "mean_ndvi": 0.65,
    }
    risk_data = {
        "risk_level": "MODERATE",
        "risk_score": 0.48,
        "signals": [{"name": "optical", "value": 0.42}],
    }

    updated = temp_case_service.update_case_satellite(
        case_id=case_id,
        satellite_data=sat_data,
        risk_assessment=risk_data,
    )

    assert updated["satellite_status"] == "SUCCESS"
    assert updated["satellite_analysis"]["modality"] == "OPTICAL_ONLY"
    assert updated["satellite_analysis"]["anomaly_score"] == 0.42
    assert updated["risk_level"] == "MODERATE"
    assert updated["risk_score"] == 0.48
    assert updated["risk_assessment"]["signals"][0]["name"] == "optical"



