from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol


@dataclass
class NotificationReceipt:
    recipient: str
    status: str
    provider: str
    is_demo: bool
    timestamp: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)


class NotificationProvider(Protocol):
    def send(self, recipient: str, message: str, metadata: dict[str, Any] | None = None) -> NotificationReceipt:
        ...


class LocalNotificationAdapter:
    """Local demo notification provider that records dispatched alerts in memory.

    Clearly marks alerts with is_demo=True to prevent confusing simulated alerts
    with production external SMS/WhatsApp broadcasts.
    """

    def __init__(self) -> None:
        self.sent_notifications: list[NotificationReceipt] = []

    def send(self, recipient: str, message: str, metadata: dict[str, Any] | None = None) -> NotificationReceipt:
        now = datetime.now(timezone.utc).isoformat()
        receipt = NotificationReceipt(
            recipient=recipient,
            status="DELIVERED",
            provider="local_demo",
            is_demo=True,
            timestamp=now,
            message=message,
            metadata=metadata or {},
        )
        self.sent_notifications.append(receipt)
        return receipt

    def clear(self) -> None:
        self.sent_notifications.clear()


def format_case_alert(
    case_id: str,
    crop: str,
    event_type: str,
    diagnosis: str | None = None,
    language: str = "en",
) -> str:
    """Format a localized notification message for farmers."""
    lang = language.lower()
    if lang == "mr":
        if event_type == "CREATED":
            return f"कृषीदृष्टी: तुमची तक्रार #{case_id} ({crop}) नोंदवली गेली आहे. प्राथमिक तपासणी प्रगतीपथावर आहे."
        elif event_type == "REVIEWED":
            return f"कृषीदृष्टी: तज्ज्ञांनी केस #{case_id} ची पडताळणी केली आहे. निदान: {diagnosis or 'तपासणी पूर्ण'}. ॲपमध्ये सल्ला पहा."
        elif event_type == "FIELD_VISIT":
            return f"कृषीदृष्टी: केस #{case_id} साठी कृषी अधिकाऱ्यांची प्रत्यक्ष शेत भेट निश्चित करण्यात आली आहे."
        elif event_type == "FOLLOWUP_DUE":
            return f"कृषीदृष्टी: केस #{case_id} साठी ४८ तास पूर्ण झाले आहेत. कृपया पिकाच्या सद्यस्थितीची नोंद करा."
        return f"कृषीदृष्टी: केस #{case_id} संबंधी अद्यतन उपलब्ध आहे."

    if lang == "hi":
        if event_type == "CREATED":
            return f"कृषिदृष्टि: आपकी शिकायत #{case_id} ({crop}) दर्ज कर ली गई है। प्रारंभिक जांच जारी है।"
        elif event_type == "REVIEWED":
            return f"कृषिदृष्टि: विशेषज्ञ ने केस #{case_id} की समीक्षा की है। निदान: {diagnosis or 'समीक्षा पूर्ण'}। ऐप में सलाह देखें।"
        elif event_type == "FIELD_VISIT":
            return f"कृषिदृष्टि: केस #{case_id} के लिए कृषि अधिकारी के प्रक्षेत्र भ्रमण (Field Visit) का अनुरोध किया गया है।"
        elif event_type == "FOLLOWUP_DUE":
            return f"कृषिदृष्टि: केस #{case_id} के ४८ घंटे पूरे हो चुके हैं। कृपया अपनी फसल की वर्तमान स्थिति अपडेट करें।"
        return f"कृषिदृष्टि: केस #{case_id} के लिए नया अपडेट उपलब्ध है।"

    # Default: English
    if event_type == "CREATED":
        return f"KrishiDrishti: Your case #{case_id} for {crop} has been registered. Analysis is in progress."
    elif event_type == "REVIEWED":
        return f"KrishiDrishti: Expert review completed for case #{case_id}. Diagnosis: {diagnosis or 'Reviewed'}. View advisory in app."
    elif event_type == "FIELD_VISIT":
        return f"KrishiDrishti: An on-site field visit has been requested by an agricultural officer for case #{case_id}."
    elif event_type == "FOLLOWUP_DUE":
        return f"KrishiDrishti: 48 hours have passed for case #{case_id}. Please submit a follow-up condition update."
    return f"KrishiDrishti: Update available for case #{case_id}."
