"""WhatsApp Client adapter for Meta Cloud API & Staged Prototype Demonstration.

Adheres strictly to the truthful status protocol:
- If WHATSAPP_ACCESS_TOKEN and WHATSAPP_PHONE_NUMBER_ID are configured in environment:
  operates in 🟢 LIVE mode via Graph API v19.0.
- Otherwise:
  operates gracefully in 🟡 DEMO / STAGED mode with an in-memory outbox,
  structured delivery receipts, and verification logs.
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger("krishidrishti.whatsapp")


@dataclass
class OutboundWhatsAppMessage:
    message_id: str
    recipient: str
    body: str
    status: str
    is_demo: bool
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseWhatsAppClient(ABC):
    """Abstract base contract for WhatsApp delivery providers."""

    @property
    @abstractmethod
    def is_live(self) -> bool:
        """True if the provider is connected to live Meta Graph API."""
        ...

    @property
    @abstractmethod
    def status_label(self) -> str:
        """Display label, e.g. '🟢 LIVE' or '🟡 DEMO / STAGED'."""
        ...

    @abstractmethod
    def send_message(
        self,
        recipient: str,
        body: str,
        metadata: dict[str, Any] | None = None,
    ) -> OutboundWhatsAppMessage:
        """Send a text message to a farmer's WhatsApp number."""
        ...

    @abstractmethod
    def get_outbox(self, recipient: str | None = None) -> list[dict[str, Any]]:
        """Retrieve dispatched messages."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear recorded outbox messages."""
        ...

    @abstractmethod
    def check_connection(self) -> dict[str, Any]:
        """Verify API connectivity and configuration."""
        ...


class MetaWhatsAppClient(BaseWhatsAppClient):
    """Live Meta WhatsApp Cloud API client (Graph API v19.0)."""

    def __init__(
        self,
        token: str,
        phone_id: str,
        api_version: str = "v19.0",
    ) -> None:
        self.token = token
        self.phone_id = phone_id
        self.api_version = api_version
        self.graph_base = f"https://graph.facebook.com/{self.api_version}"
        self.outbox: list[OutboundWhatsAppMessage] = []

    @property
    def is_live(self) -> bool:
        return True

    @property
    def status_label(self) -> str:
        return "🟢 LIVE"

    def send_message(
        self,
        recipient: str,
        body: str,
        metadata: dict[str, Any] | None = None,
    ) -> OutboundWhatsAppMessage:
        now = datetime.now(timezone.utc).isoformat()
        msg_id = f"wamid.{datetime.now().strftime('%Y%m%d%H%M%S')}.{len(self.outbox) + 1:04d}"
        url = f"{self.graph_base}/{self.phone_id}/messages"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": recipient.lstrip("+").strip(),
            "type": "text",
            "text": {"preview_url": False, "body": body},
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                msg_id = data.get("messages", [{}])[0].get("id", msg_id)
                delivery_status = "SENT"
        except Exception as exc:
            logger.error(f"Failed to dispatch live WhatsApp message: {exc}")
            delivery_status = "FAILED_LIVE_FALLBACK_STAGED"

        record = OutboundWhatsAppMessage(
            message_id=msg_id,
            recipient=recipient,
            body=body,
            status=delivery_status,
            is_demo=False,
            timestamp=now,
            metadata=metadata or {},
        )
        self.outbox.append(record)
        logger.info(f"[{self.status_label}] Live WhatsApp to {recipient} [{delivery_status}]: {body[:60]}...")
        return record

    def get_outbox(self, recipient: str | None = None) -> list[dict[str, Any]]:
        records = self.outbox
        if recipient:
            records = [m for m in records if m.recipient == recipient]
        return [
            {
                "message_id": m.message_id,
                "recipient": m.recipient,
                "body": m.body,
                "status": m.status,
                "is_demo": False,
                "timestamp": m.timestamp,
                "metadata": m.metadata,
            }
            for m in records
        ]

    def clear(self) -> None:
        self.outbox.clear()

    def check_connection(self) -> dict[str, Any]:
        """Test reachability against Meta Graph API endpoint."""
        url = f"{self.graph_base}/{self.phone_id}"
        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(url, headers=headers)
                reachable = resp.status_code == 200
                return {
                    "provider": "Meta WhatsApp Cloud API (Graph v19.0)",
                    "configured": True,
                    "reachable": reachable,
                    "status": "LIVE" if reachable else "ERROR",
                    "badge": "🟢 LIVE" if reachable else "🔴 ERROR",
                    "phone_id": self.phone_id,
                    "status_code": resp.status_code,
                }
        except Exception as exc:
            return {
                "provider": "Meta WhatsApp Cloud API (Graph v19.0)",
                "configured": True,
                "reachable": False,
                "status": "OFFLINE",
                "badge": "🔴 OFFLINE",
                "error": str(exc),
            }


class MockWhatsAppClient(BaseWhatsAppClient):
    """In-memory staged mock client for prototype demonstrations and evaluators."""

    def __init__(self) -> None:
        self.outbox: list[OutboundWhatsAppMessage] = []

    @property
    def is_live(self) -> bool:
        return False

    @property
    def status_label(self) -> str:
        return "🟡 DEMO / STAGED"

    def send_message(
        self,
        recipient: str,
        body: str,
        metadata: dict[str, Any] | None = None,
    ) -> OutboundWhatsAppMessage:
        now = datetime.now(timezone.utc).isoformat()
        msg_id = f"wamid.demo.{datetime.now().strftime('%Y%m%d%H%M%S')}.{len(self.outbox) + 1:04d}"
        record = OutboundWhatsAppMessage(
            message_id=msg_id,
            recipient=recipient,
            body=body,
            status="DELIVERED_STAGED",
            is_demo=True,
            timestamp=now,
            metadata=metadata or {},
        )
        self.outbox.append(record)
        logger.info(f"[{self.status_label}] Staged WhatsApp to {recipient}: {body[:60]}...")
        return record

    def get_outbox(self, recipient: str | None = None) -> list[dict[str, Any]]:
        records = self.outbox
        if recipient:
            records = [m for m in records if m.recipient == recipient]
        return [
            {
                "message_id": m.message_id,
                "recipient": m.recipient,
                "body": m.body,
                "status": m.status,
                "is_demo": True,
                "timestamp": m.timestamp,
                "metadata": m.metadata,
            }
            for m in records
        ]

    def clear(self) -> None:
        self.outbox.clear()

    def check_connection(self) -> dict[str, Any]:
        return {
            "provider": "Mock WhatsApp Outbox (Staged Evaluation)",
            "configured": False,
            "reachable": True,
            "status": "DEMO / STAGED",
            "badge": "🟡 DEMO / STAGED",
            "reason": "Set WHATSAPP_ACCESS_TOKEN and WHATSAPP_PHONE_NUMBER_ID in .env to activate Meta Cloud API.",
        }


class WhatsAppClient(BaseWhatsAppClient):
    """Enterprise composite WhatsApp client switching between Meta and Mock gracefully."""

    def __init__(self) -> None:
        self.token = os.getenv("WHATSAPP_TOKEN") or os.getenv("WHATSAPP_ACCESS_TOKEN")
        self.phone_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
        self.api_version = os.getenv("WHATSAPP_API_VERSION", "v19.0")
        self.graph_base = f"https://graph.facebook.com/{self.api_version}"
        self._provider: BaseWhatsAppClient

        if self.token and self.phone_id and not self.token.startswith("demo_"):
            self._provider = MetaWhatsAppClient(
                token=self.token,
                phone_id=self.phone_id,
                api_version=self.api_version,
            )
        else:
            self._provider = MockWhatsAppClient()

    @property
    def is_live(self) -> bool:
        return self._provider.is_live

    @property
    def status_label(self) -> str:
        return self._provider.status_label

    @property
    def outbox(self) -> list[OutboundWhatsAppMessage]:
        if hasattr(self._provider, "outbox"):
            return self._provider.outbox
        return []

    def send_message(
        self,
        recipient: str,
        body: str,
        metadata: dict[str, Any] | None = None,
    ) -> OutboundWhatsAppMessage:
        return self._provider.send_message(recipient=recipient, body=body, metadata=metadata)

    def get_outbox(self, recipient: str | None = None) -> list[dict[str, Any]]:
        return self._provider.get_outbox(recipient=recipient)

    def clear(self) -> None:
        self._provider.clear()

    def check_connection(self) -> dict[str, Any]:
        return self._provider.check_connection()


# Global singleton client
_default_whatsapp_client: WhatsAppClient | None = None


def get_whatsapp_client() -> WhatsAppClient:
    global _default_whatsapp_client
    if _default_whatsapp_client is None:
        _default_whatsapp_client = WhatsAppClient()
    return _default_whatsapp_client
