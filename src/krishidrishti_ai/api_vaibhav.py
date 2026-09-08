"""Legacy alias module for backward compatibility.

All endpoints and services have been unified into `krishidrishti_ai.api`.
"""
from __future__ import annotations

from krishidrishti_ai.api import (
    _get_registry,
    app,
    diagnose,
    diagnose_legacy,
    health,
    index,
    list_crops,
)

__all__ = [
    "app",
    "index",
    "health",
    "list_crops",
    "diagnose",
    "diagnose_legacy",
    "_get_registry",
]
