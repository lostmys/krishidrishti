from __future__ import annotations

import json
import math
import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

VALID_STATUSES = {
    "OPEN",
    "REVIEW_RECOMMENDED",
    "CONFIRMED",
    "REJECTED",
    "FIELD_VISIT_REQUIRED",
    "RESOLVED",
}

VALID_TRANSITIONS: dict[str, set[str]] = {
    "OPEN": {"REVIEW_RECOMMENDED", "CONFIRMED", "REJECTED", "FIELD_VISIT_REQUIRED"},
    "REVIEW_RECOMMENDED": {"CONFIRMED", "REJECTED", "FIELD_VISIT_REQUIRED"},
    "CONFIRMED": {"RESOLVED", "FIELD_VISIT_REQUIRED"},
    "REJECTED": {"OPEN", "RESOLVED"},
    "FIELD_VISIT_REQUIRED": {"CONFIRMED", "REJECTED", "RESOLVED"},
    "RESOLVED": {"OPEN"},
}

VALID_FOLLOWUP_CONDITIONS = {"IMPROVED", "SAME", "WORSE"}

VALID_EXPERT_ACTIONS = {"CONFIRM", "REJECT", "REQUEST_FIELD_VISIT", "NOTE"}

DEFAULT_NEARBY_RADIUS_KM: float = 10.0
"""Default geographic radius (in kilometers) for identifying nearby crop disease cases.

Rationale:
In Maharashtra and broader Indian rural agricultural settings, a 10 km radius encompasses
a typical cluster of 4 to 8 adjoining villages (Gram Panchayat / sub-block cluster).
1. Epidemiological spread: Airborne fungal propagules (e.g. rusts, leaf blights) and winged
   vector insects (e.g. whiteflies, aphids, mites) frequently disperse across neighboring
   farms within a 10 km corridor over a 14-day incubation cycle under monsoon/kharif conditions.
2. Shared agricultural vectors: Farmers within a 10 km radius typically share local agricultural
   infrastructure (tillage/harvest machinery, custom hire centers, and input dealers at the local
   Krishi Seva Kendra), which can act as fomites for pathogen transmission.
3. Realistic triage scope: 10 km is localized enough to indicate an actionable neighborhood threat
   without diluting signal across an entire district or taluka.
"""


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two geographic coordinates in kilometers.

    Uses the Haversine formula with Earth's volumetric mean radius (6371.0 km).
    """
    R = 6371.0  # Earth's mean radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    a = min(1.0, max(0.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


class CaseService:
    """SQLite-backed repository and lifecycle service for farmer incident cases."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is not None:
            self.db_path = Path(db_path).resolve()
        else:
            env_path = os.getenv("KRISHIDRISHTI_DB_PATH")
            if env_path:
                self.db_path = Path(env_path).resolve()
            else:
                project_root = next(
                    (c for c in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents) if (c / "pyproject.toml").is_file()),
                    Path.cwd(),
                )
                self.db_path = (project_root / "data" / "krishidrishti.db").resolve()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    crop TEXT NOT NULL,
                    farmer_id TEXT,
                    farmer_contact TEXT,
                    farmer_name TEXT,
                    latitude REAL,
                    longitude REAL,
                    location_json TEXT,
                    description TEXT,
                    image_path TEXT,
                    image_filename TEXT,
                    ai_diagnosis TEXT,
                    raw_confidence REAL,
                    calibrated_confidence REAL,
                    ai_status TEXT,
                    ai_details_json TEXT,
                    satellite_status TEXT,
                    satellite_json TEXT,
                    risk_level TEXT,
                    risk_score REAL,
                    risk_json TEXT,
                    advisory_json TEXT,
                    current_status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS expert_reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL REFERENCES cases(case_id) ON DELETE CASCADE,
                    reviewer_name TEXT,
                    action TEXT NOT NULL,
                    notes TEXT,
                    previous_status TEXT NOT NULL,
                    new_status TEXT NOT NULL,
                    corrected_diagnosis TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS follow_ups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL REFERENCES cases(case_id) ON DELETE CASCADE,
                    observed_by TEXT,
                    condition TEXT NOT NULL,
                    notes TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_cases_crop ON cases(crop);
                CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(current_status);
                CREATE INDEX IF NOT EXISTS idx_cases_created ON cases(created_at);
                """
            )

    @staticmethod
    def generate_case_id() -> str:
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        short_id = uuid.uuid4().hex[:6].upper()
        return f"KD-{date_str}-{short_id}"

    def create_case(
        self,
        crop: str,
        farmer_contact: str | None = None,
        farmer_name: str | None = None,
        farmer_id: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        location_details: dict[str, Any] | None = None,
        description: str | None = None,
        image_path: str | None = None,
        image_filename: str | None = None,
        diagnosis_result: dict[str, Any] | None = None,
        satellite_result: dict[str, Any] | None = None,
        risk_assessment: dict[str, Any] | None = None,
        advisory: dict[str, Any] | None = None,
        initial_status: str = "OPEN",
    ) -> dict[str, Any]:
        case_id = self.generate_case_id()
        now = datetime.now(timezone.utc).isoformat()

        # Determine automatic initial status based on AI assessment
        if diagnosis_result:
            ai_status = diagnosis_result.get("status")
            if ai_status in ("REVIEW_RECOMMENDED", "LOW_CONFIDENCE", "IMAGE_QUALITY_REJECTED"):
                initial_status = "REVIEW_RECOMMENDED"

        if initial_status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {initial_status}. Must be one of {VALID_STATUSES}")

        ai_diag = diagnosis_result.get("diagnosis") if diagnosis_result else None
        raw_conf = float(diagnosis_result["raw_confidence"]) if diagnosis_result and "raw_confidence" in diagnosis_result else None
        cal_conf = float(diagnosis_result["confidence"]) if diagnosis_result and "confidence" in diagnosis_result else None
        ai_stat = diagnosis_result.get("status") if diagnosis_result else None

        sat_status = (satellite_result or {}).get("status", "NOT_REQUESTED")
        r_level = (risk_assessment or {}).get("risk_level")
        r_score = (risk_assessment or {}).get("risk_score")

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO cases (
                    case_id, crop, farmer_id, farmer_contact, farmer_name,
                    latitude, longitude, location_json, description,
                    image_path, image_filename,
                    ai_diagnosis, raw_confidence, calibrated_confidence, ai_status, ai_details_json,
                    satellite_status, satellite_json,
                    risk_level, risk_score, risk_json, advisory_json,
                    current_status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case_id,
                    crop,
                    farmer_id,
                    farmer_contact,
                    farmer_name,
                    latitude,
                    longitude,
                    json.dumps(location_details or {}),
                    description,
                    image_path,
                    image_filename,
                    ai_diag,
                    raw_conf,
                    cal_conf,
                    ai_stat,
                    json.dumps(diagnosis_result or {}),
                    sat_status,
                    json.dumps(satellite_result or {}),
                    r_level,
                    r_score,
                    json.dumps(risk_assessment or {}),
                    json.dumps(advisory or {}),
                    initial_status,
                    now,
                    now,
                ),
            )

        return self.get_case(case_id)  # type: ignore

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
            if not row:
                return None
            case_dict = self._row_to_case_dict(row)

            # Fetch expert reviews
            reviews = conn.execute(
                "SELECT * FROM expert_reviews WHERE case_id = ? ORDER BY id ASC",
                (case_id,),
            ).fetchall()
            case_dict["expert_reviews_history"] = [
                {
                    "id": r["id"],
                    "reviewer_name": r["reviewer_name"],
                    "action": r["action"],
                    "notes": r["notes"],
                    "previous_status": r["previous_status"],
                    "new_status": r["new_status"],
                    "corrected_diagnosis": r["corrected_diagnosis"],
                    "created_at": r["created_at"],
                }
                for r in reviews
            ]

            # Fetch follow-ups
            follow_ups = conn.execute(
                "SELECT * FROM follow_ups WHERE case_id = ? ORDER BY id ASC",
                (case_id,),
            ).fetchall()
            case_dict["follow_up_history"] = [
                {
                    "id": f["id"],
                    "observed_by": f["observed_by"],
                    "condition": f["condition"],
                    "notes": f["notes"],
                    "created_at": f["created_at"],
                }
                for f in follow_ups
            ]

            # Calculate 48-hour follow-up deadline
            try:
                created_dt = datetime.fromisoformat(case_dict["created_at"])
                deadline_dt = created_dt + timedelta(hours=48)
                case_dict["followup_deadline"] = deadline_dt.isoformat()
                case_dict["followup_due"] = datetime.now(timezone.utc) >= deadline_dt and case_dict["current_status"] != "RESOLVED"
            except Exception:
                case_dict["followup_deadline"] = None
                case_dict["followup_due"] = False

            return case_dict

    def list_cases(
        self,
        status: str | None = None,
        crop: str | None = None,
        risk_level: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM cases WHERE 1=1"
        params: list[Any] = []
        if status:
            query += " AND current_status = ?"
            params.append(status.upper())
        if crop:
            query += " AND LOWER(crop) = LOWER(?)"
            params.append(crop)
        if risk_level:
            query += " AND risk_level = ?"
            params.append(risk_level.upper())
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_case_dict(r) for r in rows]

    def add_expert_review(
        self,
        case_id: str,
        action: str,
        reviewer_name: str | None = None,
        notes: str | None = None,
        corrected_diagnosis: str | None = None,
    ) -> dict[str, Any]:
        action_upper = action.upper()
        if action_upper not in VALID_EXPERT_ACTIONS:
            raise ValueError(f"Invalid expert action '{action}'. Choose one of: {VALID_EXPERT_ACTIONS}")

        case = self.get_case(case_id)
        if not case:
            raise KeyError(f"Case '{case_id}' not found")

        curr_status = case["current_status"]

        if action_upper == "CONFIRM":
            new_status = "CONFIRMED"
        elif action_upper == "REJECT":
            new_status = "REJECTED"
        elif action_upper == "REQUEST_FIELD_VISIT":
            new_status = "FIELD_VISIT_REQUIRED"
        else:  # NOTE
            new_status = curr_status

        # Transition validation
        if new_status != curr_status:
            allowed = VALID_TRANSITIONS.get(curr_status, set())
            if new_status not in allowed:
                raise ValueError(
                    f"Illegal status transition from '{curr_status}' to '{new_status}'. Allowed transitions: {sorted(allowed)}"
                )

        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO expert_reviews (
                    case_id, reviewer_name, action, notes,
                    previous_status, new_status, corrected_diagnosis, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case_id,
                    reviewer_name or "Agricultural Officer",
                    action_upper,
                    notes,
                    curr_status,
                    new_status,
                    corrected_diagnosis,
                    now,
                ),
            )
            update_diag = corrected_diagnosis if corrected_diagnosis else case.get("ai_diagnosis")
            conn.execute(
                """
                UPDATE cases
                SET current_status = ?, ai_diagnosis = ?, updated_at = ?
                WHERE case_id = ?
                """,
                (new_status, update_diag, now, case_id),
            )

        return self.get_case(case_id)  # type: ignore

    def add_follow_up(
        self,
        case_id: str,
        condition: str,
        observed_by: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        cond_upper = condition.upper()
        if cond_upper not in VALID_FOLLOWUP_CONDITIONS:
            raise ValueError(f"Invalid condition '{condition}'. Must be one of: {VALID_FOLLOWUP_CONDITIONS}")

        case = self.get_case(case_id)
        if not case:
            raise KeyError(f"Case '{case_id}' not found")

        now = datetime.now(timezone.utc).isoformat()
        curr_status = case["current_status"]
        new_status = curr_status

        if cond_upper == "IMPROVED" and curr_status == "CONFIRMED":
            new_status = "RESOLVED"
        elif cond_upper == "WORSE" and curr_status in ("OPEN", "CONFIRMED", "REVIEW_RECOMMENDED"):
            new_status = "FIELD_VISIT_REQUIRED"

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO follow_ups (case_id, observed_by, condition, notes, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (case_id, observed_by or "Farmer", cond_upper, notes, now),
            )
            if new_status != curr_status:
                conn.execute(
                    "UPDATE cases SET current_status = ?, updated_at = ? WHERE case_id = ?",
                    (new_status, now, case_id),
                )

        return self.get_case(case_id)  # type: ignore

    def get_due_followups(self, hours: int = 48) -> list[dict[str, Any]]:
        threshold = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM cases
                WHERE current_status NOT IN ('RESOLVED', 'REJECTED')
                  AND created_at <= ?
                ORDER BY created_at ASC
                """,
                (threshold,),
            ).fetchall()
            return [self._row_to_case_dict(r) for r in rows]

    def count_nearby_recent_cases(
        self,
        crop: str,
        latitude: float | None = None,
        longitude: float | None = None,
        days: int = 14,
        radius_km: float = DEFAULT_NEARBY_RADIUS_KM,
        exclude_case_id: str | None = None,
    ) -> int:
        """Count recent active cases of the same crop within a geographic radius.

        Args:
            crop: Crop name or slug (case-insensitive).
            latitude: Current case latitude. If None, returns 0 because geographic proximity cannot be established.
            longitude: Current case longitude. If None, returns 0 because geographic proximity cannot be established.
            days: Time window in days (default: 14 days).
            radius_km: Geographic radius in kilometers (default: 10.0 km).
            exclude_case_id: Optional case_id to exclude (e.g. current case being updated).

        Returns:
            Number of matching same-crop cases within the radius created within the last `days` days.
            Returns 0 if coordinates are not provided, invalid, or if no cases match.
        """
        if latitude is None or longitude is None:
            return 0

        try:
            lat = float(latitude)
            lon = float(longitude)
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                return 0
        except (ValueError, TypeError):
            return 0

        threshold = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._get_connection() as conn:
            query = """
                SELECT case_id, latitude, longitude FROM cases
                WHERE LOWER(crop) = LOWER(?)
                  AND current_status IN ('OPEN', 'CONFIRMED', 'REVIEW_RECOMMENDED', 'FIELD_VISIT_REQUIRED')
                  AND created_at >= ?
                  AND latitude IS NOT NULL
                  AND longitude IS NOT NULL
            """
            params: list[Any] = [crop, threshold]
            if exclude_case_id:
                query += " AND case_id != ?"
                params.append(exclude_case_id)

            rows = conn.execute(query, params).fetchall()

            count = 0
            for row in rows:
                c_lat = row["latitude"]
                c_lon = row["longitude"]
                if c_lat is None or c_lon is None:
                    continue
                try:
                    dist = haversine_distance_km(lat, lon, float(c_lat), float(c_lon))
                    if dist <= radius_km:
                        count += 1
                except (ValueError, TypeError):
                    continue

            return count

    def update_case_satellite(
        self,
        case_id: str,
        satellite_data: dict[str, Any],
        risk_assessment: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            params: list[Any] = [
                satellite_data.get("status", "SUCCESS"),
                json.dumps(satellite_data),
            ]
            sql = "UPDATE cases SET satellite_status = ?, satellite_json = ?"
            if risk_assessment:
                sql += ", risk_level = ?, risk_score = ?, risk_json = ?"
                params.extend([
                    risk_assessment.get("risk_level"),
                    risk_assessment.get("risk_score"),
                    json.dumps(risk_assessment),
                ])
            sql += ", updated_at = ?"
            params.extend([now, case_id])
            conn.execute(sql + " WHERE case_id = ?", params)
        return self.get_case(case_id)  # type: ignore

    def _row_to_case_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "case_id": row["case_id"],
            "crop": row["crop"],
            "farmer_id": row["farmer_id"],
            "farmer_contact": row["farmer_contact"],
            "farmer_name": row["farmer_name"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "location": json.loads(row["location_json"] or "{}"),
            "description": row["description"],
            "image_path": row["image_path"],
            "image_filename": row["image_filename"],
            "ai_diagnosis": row["ai_diagnosis"],
            "raw_confidence": row["raw_confidence"],
            "calibrated_confidence": row["calibrated_confidence"],
            "ai_status": row["ai_status"],
            "ai_details": json.loads(row["ai_details_json"] or "{}"),
            "satellite_status": row["satellite_status"],
            "satellite_analysis": json.loads(row["satellite_json"] or "{}"),
            "risk_level": row["risk_level"],
            "risk_score": row["risk_score"],
            "risk_assessment": json.loads(row["risk_json"] or "{}"),
            "advisory": json.loads(row["advisory_json"] or "{}"),
            "current_status": row["current_status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
