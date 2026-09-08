"""Tests for Phase 2 UI presentation and farmer/officer workspace separation.

Verifies:
- index.html structure contains role toggle buttons, officer details, separated actions.
- Clarified risk signal labels (Leaf evidence, Geographic cluster, Satellite evidence).
- Score out of 100 representation and weights footnote.
- Stepper logic derivation from backend current_status across all 6 lifecycle states.
- Multilingual translations in en, hi, mr for the new UI controls.
"""
from __future__ import annotations

import re
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from krishidrishti_ai.api import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_index_html_served_and_contains_workspace_toggle(client: TestClient) -> None:
    """Verify that root endpoint serves index.html with view switcher and officer controls."""
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text

    # View Mode Switcher
    assert 'id="btn-view-farmer"' in html
    assert 'id="btn-view-officer"' in html
    assert "setCaseViewMode('farmer')" in html
    assert "setCaseViewMode('officer')" in html

    # Officer-only sections
    assert 'id="officer-workspace-banner"' in html
    assert 'id="officer-details-card"' in html
    assert 'id="sec-officer-review-actions"' in html
    assert 'id="off-farmer-name"' in html
    assert 'id="off-farmer-phone"' in html
    assert 'id="off-farmer-location"' in html
    assert 'id="off-farmer-notes"' in html

    # Farmer follow-up form separation
    assert 'id="sec-case-followup-form"' in html
    assert 'id="sec-followup-history"' in html


def test_clarified_risk_signal_labels_and_footnote(client: TestClient) -> None:
    """Verify signal presentation as score out of 100 and explanatory footnote."""
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text

    # Clarified labels
    assert "Leaf evidence:" in html
    assert "Geographic cluster:" in html
    assert "Satellite evidence:" in html

    # Value placeholders / format
    assert "-- / 100" in html
    assert 'id="txt-risk-weight-note"' in html

    # Explanatory footnote text content
    assert "Leaf 40%" in html
    assert "Satellite 35%" in html
    assert "Geographic 15%" in html
    assert "Symptoms 10%" in html
    assert "do not sum to 100%" in html.lower()


def test_stepper_state_machine_logic_in_html() -> None:
    """Verify updateStepper JavaScript logic maps all 6 case statuses correctly."""
    html_path = Path(__file__).resolve().parent.parent / "src" / "krishidrishti_ai" / "static" / "index.html"
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")

    # Locate updateStepper function
    match = re.search(r"function updateStepper\(status\)\s*\{(.*?)function renderList", content, re.DOTALL)
    assert match is not None, "updateStepper function not found in index.html"
    fn_body = match.group(1)

    # Status OPEN: step 3 pending, step 4 pending (NOT active)
    assert "status === 'OPEN'" in fn_body
    open_block = re.search(r"status === 'OPEN'\) \{(.*?)\}", fn_body, re.DOTALL)
    assert open_block is not None
    assert "s3.classList.add('pending')" in open_block.group(1)
    assert "s4.classList.add('pending')" in open_block.group(1)
    assert "active" not in open_block.group(1)

    # Status REVIEW_RECOMMENDED: step 3 current, step 4 pending
    assert "status === 'REVIEW_RECOMMENDED'" in fn_body
    review_block = re.search(r"status === 'REVIEW_RECOMMENDED'\) \{(.*?)\}", fn_body, re.DOTALL)
    assert review_block is not None
    assert "s3.classList.add('current')" in review_block.group(1)
    assert "s4.classList.add('pending')" in review_block.group(1)

    # Status FIELD_VISIT_REQUIRED: step 3 flagged, step 4 pending
    assert "status === 'FIELD_VISIT_REQUIRED'" in fn_body
    visit_block = re.search(r"status === 'FIELD_VISIT_REQUIRED'\) \{(.*?)\}", fn_body, re.DOTALL)
    assert visit_block is not None
    assert "s3.classList.add('flagged')" in visit_block.group(1)
    assert "s4.classList.add('pending')" in visit_block.group(1)

    # Status CONFIRMED: step 3 completed, step 4 current
    assert "status === 'CONFIRMED'" in fn_body
    conf_block = re.search(r"status === 'CONFIRMED'\) \{(.*?)\}", fn_body, re.DOTALL)
    assert conf_block is not None
    assert "s3.classList.add('completed')" in conf_block.group(1)
    assert "s4.classList.add('current')" in conf_block.group(1)

    # Status REJECTED: step 3 completed, step 4 closed
    assert "status === 'REJECTED'" in fn_body
    rej_block = re.search(r"status === 'REJECTED'\) \{(.*?)\}", fn_body, re.DOTALL)
    assert rej_block is not None
    assert "s3.classList.add('completed')" in rej_block.group(1)
    assert "s4.classList.add('closed')" in rej_block.group(1)

    # Status RESOLVED: step 3 completed, step 4 completed
    assert "status === 'RESOLVED'" in fn_body
    res_block = re.search(r"status === 'RESOLVED'\) \{(.*?)\}", fn_body, re.DOTALL)
    assert res_block is not None
    assert "s3.classList.add('completed')" in res_block.group(1)
    assert "s4.classList.add('completed')" in res_block.group(1)


def test_multilingual_strings_present_for_ui_polish() -> None:
    """Verify en, hi, mr dictionaries contain all new labels."""
    html_path = Path(__file__).resolve().parent.parent / "src" / "krishidrishti_ai" / "static" / "index.html"
    content = html_path.read_text(encoding="utf-8")

    for lang in ["en", "hi", "mr"]:
        assert f"{lang}: {{" in content

    # Check key additions in UI_STRINGS
    required_keys = [
        "workspaceRole",
        "farmerViewBtn",
        "officerViewBtn",
        "officerBanner",
        "officerDetailsHdr",
        "offFarmerName",
        "offFarmerPhone",
        "offFarmerLoc",
        "offFarmerNotes",
        "leafEvidence",
        "geoCluster",
        "satEvidence",
        "riskWeightNote",
    ]
    for key in required_keys:
        assert f"{key}:" in content


def test_format_risk_explanation_function_logic() -> None:
    """Verify formatRiskExplanation cleanly distinguishes diagnostic confidence from overall triage risk."""
    html_path = Path(__file__).resolve().parent.parent / "src" / "krishidrishti_ai" / "static" / "index.html"
    content = html_path.read_text(encoding="utf-8")

    assert "function formatRiskExplanation(caseData, lang)" in content
    # Distinguishes confidence from triage risk across languages
    assert "triage risk" in content.lower()
    assert "ट्राइएज जोखिम" in content
    assert "ट्रायज जोखीम" in content
