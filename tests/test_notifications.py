from __future__ import annotations

from krishidrishti_ai.services.notifications import (
    LocalNotificationAdapter,
    format_case_alert,
)


def test_format_case_alert_multilingual() -> None:
    # English
    en_msg = format_case_alert("KD-101", "Tomato", "CREATED", language="en")
    assert "KD-101" in en_msg
    assert "registered" in en_msg

    # Marathi
    mr_msg = format_case_alert("KD-101", "कापूस", "FIELD_VISIT", language="mr")
    assert "KD-101" in mr_msg
    assert "शेत भेट" in mr_msg

    # Hindi
    hi_msg = format_case_alert("KD-101", "सोयाबीन", "REVIEWED", diagnosis="गेरुई", language="hi")
    assert "KD-101" in hi_msg
    assert "गेरुई" in hi_msg
    assert "विशेषज्ञ" in hi_msg


def test_local_notification_adapter_records_demo_alerts() -> None:
    adapter = LocalNotificationAdapter()
    receipt = adapter.send(
        recipient="+919876543210",
        message="Test alert message",
        metadata={"case_id": "KD-123"},
    )
    assert receipt.status == "DELIVERED"
    assert receipt.is_demo is True
    assert receipt.provider == "local_demo"
    assert len(adapter.sent_notifications) == 1
    assert adapter.sent_notifications[0].recipient == "+919876543210"

    adapter.clear()
    assert len(adapter.sent_notifications) == 0
