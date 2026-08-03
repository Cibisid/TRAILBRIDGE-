"""
TrialBridge — Database Models

These are the SQLAlchemy ORM models — Python classes that map
directly to PostgreSQL tables. Alembic reads these to generate
database migrations. Every field is typed and documented.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.core.database import Base


# -----------------------------------------------
# Enums
# -----------------------------------------------
class TrialStatus(StrEnum):
    RECRUITING = "RECRUITING"
    NOT_YET_RECRUITING = "NOT_YET_RECRUITING"
    ACTIVE_NOT_RECRUITING = "ACTIVE_NOT_RECRUITING"
    COMPLETED = "COMPLETED"
    SUSPENDED = "SUSPENDED"
    TERMINATED = "TERMINATED"
    WITHDRAWN = "WITHDRAWN"
    UNKNOWN = "UNKNOWN"


class TrialPhase(StrEnum):
    PHASE_1 = "PHASE1"
    PHASE_2 = "PHASE2"
    PHASE_3 = "PHASE3"
    PHASE_4 = "PHASE4"
    EARLY_PHASE_1 = "EARLY_PHASE1"
    NOT_APPLICABLE = "NA"
    UNKNOWN = "UNKNOWN"


class MatchEligibility(StrEnum):
    ELIGIBLE = "eligible"
    LIKELY_ELIGIBLE = "likely_eligible"
    NEEDS_REVIEW = "needs_review"
    INELIGIBLE = "ineligible"


# -----------------------------------------------
# Mixins
# -----------------------------------------------
class TimestampMixin:
    """Adds created_at and updated_at to any model."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )


class UUIDMixin:
    """Adds a UUID primary key to any model."""

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )


# -----------------------------------------------
# Trial Model
# -----------------------------------------------
class Trial(UUIDMixin, TimestampMixin, Base):
    """
    Represents a single clinical trial from ClinicalTrials.gov.
    The embedding column stores the BioBERT vector for semantic search.
    pgvector handles this natively in PostgreSQL.
    """

    __tablename__ = "trials"

    # Core identifiers
    nct_id: Mapped[str] = mapped_column(String(20), unique=True, nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    brief_summary: Mapped[str | None] = mapped_column(Text)
    detailed_description: Mapped[str | None] = mapped_column(Text)

    # Status & Classification
    status: Mapped[TrialStatus] = mapped_column(Enum(TrialStatus), nullable=False, index=True)
    phase: Mapped[TrialPhase] = mapped_column(Enum(TrialPhase), nullable=False, index=True)

    # Eligibility criteria (raw text + parsed fields)
    eligibility_criteria_raw: Mapped[str | None] = mapped_column(Text)
    eligibility_parsed: Mapped[dict | None] = mapped_column(JSON)
    # Parsed structure example:
    # {
    #   "inclusion": ["Age 18-65", "Type 2 diabetes diagnosis"],
    #   "exclusion": ["Prior insulin therapy", "eGFR < 30"],
    #   "age_min": 18, "age_max": 65,
    #   "accepts_healthy_volunteers": false
    # }

    # Demographics
    minimum_age: Mapped[int | None] = mapped_column(Integer)
    maximum_age: Mapped[int | None] = mapped_column(Integer)
    gender: Mapped[str | None] = mapped_column(String(10))  # ALL | MALE | FEMALE
    accepts_healthy_volunteers: Mapped[bool] = mapped_column(Boolean, default=False)

    # Medical classification
    conditions: Mapped[list | None] = mapped_column(JSON)  # ["Type 2 Diabetes", ...]
    conditions_icd10: Mapped[list | None] = mapped_column(JSON)  # ["E11", ...]
    interventions: Mapped[list | None] = mapped_column(JSON)  # drug names, procedures
    keywords: Mapped[list | None] = mapped_column(JSON)

    # Study details
    sponsor: Mapped[str | None] = mapped_column(String(500))
    locations: Mapped[list | None] = mapped_column(JSON)  # [{city, state, country}, ...]
    start_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completion_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    enrollment_target: Mapped[int | None] = mapped_column(Integer)

    # BioBERT embedding for semantic search (768-dimensional vector)
    embedding: Mapped[Any | None] = mapped_column(Vector(768))
    embedding_text: Mapped[str | None] = mapped_column(Text)  # text that was embedded

    # Relationships
    match_results: Mapped[list[MatchResult]] = relationship(back_populates="trial", lazy="select")

    def __repr__(self) -> str:
        return f"<Trial {self.nct_id}: {self.title[:50]}>"


# -----------------------------------------------
# Patient Profile Model
# -----------------------------------------------
class PatientProfile(UUIDMixin, TimestampMixin, Base):
    """
    Structured patient profile extracted from an EHR note.
    We store both the original raw note and the extracted structured data.
    The raw note is hashed — we never store actual PHI in a queryable column.
    """

    __tablename__ = "patient_profiles"

    # Raw input (hashed for audit — never store real patient data unencrypted)
    raw_note_hash: Mapped[str | None] = mapped_column(String(64))  # SHA-256 hash
    source_system: Mapped[str | None] = mapped_column(String(100))  # "demo" | "epic" | "cerner"

    # Extracted demographics
    age: Mapped[int | None] = mapped_column(Integer)
    sex: Mapped[str | None] = mapped_column(String(10))  # M | F | OTHER | UNKNOWN
    race: Mapped[str | None] = mapped_column(String(100))
    ethnicity: Mapped[str | None] = mapped_column(String(100))

    # Medical conditions (primary + comorbidities)
    primary_diagnosis: Mapped[str | None] = mapped_column(String(500))
    primary_diagnosis_icd10: Mapped[str | None] = mapped_column(String(20))
    comorbidities: Mapped[list | None] = mapped_column(JSON)  # ["Hypertension", ...]
    comorbidities_icd10: Mapped[list | None] = mapped_column(JSON)  # ["I10", ...]

    # Medications
    current_medications: Mapped[list | None] = mapped_column(JSON)  # ["Metformin 500mg", ...]
    prior_treatments: Mapped[list | None] = mapped_column(JSON)  # ["Insulin therapy", ...]
    allergies: Mapped[list | None] = mapped_column(JSON)

    # Lab values (relevant for eligibility filtering)
    lab_values: Mapped[dict | None] = mapped_column(JSON)
    # Example: {"HbA1c": 8.2, "eGFR": 75, "ALT": 32, "creatinine": 1.1}

    # Clinical scores
    ecog_score: Mapped[int | None] = mapped_column(Integer)  # 0-4 performance status
    karnofsky_score: Mapped[int | None] = mapped_column(Integer)  # 0-100

    # Extraction metadata
    extraction_confidence: Mapped[float | None] = mapped_column(Float)
    extraction_model: Mapped[str | None] = mapped_column(String(100))
    extraction_warnings: Mapped[list | None] = mapped_column(JSON)

    # Relationships
    match_results: Mapped[list[MatchResult]] = relationship(back_populates="patient", lazy="select")

    def __repr__(self) -> str:
        return f"<PatientProfile {self.id}: age={self.age}, dx={self.primary_diagnosis}>"


# -----------------------------------------------
# Match Result Model
# -----------------------------------------------
class MatchResult(UUIDMixin, TimestampMixin, Base):
    """
    Records a single trial match for a patient.
    Stores scores, eligibility determination, and AI explanation.
    Every match is persisted — this creates an audit trail.
    """

    __tablename__ = "match_results"

    # Foreign keys
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patient_profiles.id"), nullable=False, index=True
    )
    trial_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("trials.id"), nullable=False, index=True
    )

    # Scoring
    semantic_score: Mapped[float] = mapped_column(Float, nullable=False)
    eligibility_score: Mapped[float] = mapped_column(Float, nullable=False)
    composite_score: Mapped[float] = mapped_column(Float, nullable=False, index=True)

    # Eligibility determination
    eligibility_status: Mapped[MatchEligibility] = mapped_column(
        Enum(MatchEligibility), nullable=False, index=True
    )
    inclusion_met: Mapped[list | None] = mapped_column(JSON)  # which criteria were met
    inclusion_unmet: Mapped[list | None] = mapped_column(JSON)  # which criteria weren't met
    exclusion_flags: Mapped[list | None] = mapped_column(JSON)  # triggered exclusions
    needs_verification: Mapped[list | None] = mapped_column(JSON)  # uncertain criteria

    # AI Explanation (generated by Claude)
    explanation: Mapped[str | None] = mapped_column(Text)
    explanation_model: Mapped[str | None] = mapped_column(String(100))

    # Session tracking (groups matches from the same query)
    session_id: Mapped[str | None] = mapped_column(String(36), index=True)
    rank: Mapped[int | None] = mapped_column(Integer)  # rank within session results

    # Relationships
    patient: Mapped[PatientProfile] = relationship(back_populates="match_results")
    trial: Mapped[Trial] = relationship(back_populates="match_results")

    def __repr__(self) -> str:
        return f"<MatchResult patient={self.patient_id} trial={self.trial_id} score={self.composite_score:.3f}>"


# -----------------------------------------------
# Audit Log Model
# -----------------------------------------------
class AuditLog(UUIDMixin, Base):
    """
    Immutable audit trail for all API operations.
    In healthcare, this is non-negotiable — you must be able to
    answer 'who queried what, when, and what did they see?'
    This table is append-only: never update or delete audit logs.
    """

    __tablename__ = "audit_logs"

    # When and who
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
        index=True,
    )
    user_id: Mapped[str | None] = mapped_column(String(100), index=True)
    ip_address: Mapped[str | None] = mapped_column(String(45))  # IPv6 max length
    user_agent: Mapped[str | None] = mapped_column(String(500))

    # What happened
    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    # e.g. "patient.parse", "match.query", "trial.view", "auth.login"

    resource_type: Mapped[str | None] = mapped_column(String(50))
    resource_id: Mapped[str | None] = mapped_column(String(100))

    # Request/Response details
    request_hash: Mapped[str | None] = mapped_column(String(64))  # SHA-256 of request body
    response_status: Mapped[int | None] = mapped_column(Integer)
    duration_ms: Mapped[int | None] = mapped_column(Integer)

    # Extra context
    extra_data: Mapped[dict | None] = mapped_column(JSON)

    def __repr__(self) -> str:
        return f"<AuditLog {self.action} by {self.user_id} at {self.timestamp}>"
