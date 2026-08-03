"""
TrialBridge — Day 9: Matching Engine
Combines semantic search + eligibility rules to produce ranked trial matches.

This is the core of TrialBridge. The full pipeline:
  1. Patient note → NLP extractor → PatientProfile
  2. PatientProfile → query string → sentence embedding
  3. Embedding → pgvector cosine search → top 50 candidates
  4. Candidates → eligibility rules engine → scored matches
  5. Scored matches → ranked list with explanations
"""

import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import asyncpg
from sentence_transformers import SentenceTransformer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.nlp.extractor_v2 import PatientProfile, PatientProfileExtractor

DB_URL = "postgresql://trialbridge_user:devpassword@localhost:5432/trialbridge"
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
SEMANTIC_CANDIDATES = 50  # How many trials to pull from vector search
FINAL_RESULTS = 10  # How many to return after rules filtering


# -----------------------------------------------
# Match Result Model
# -----------------------------------------------
@dataclass
class TrialMatch:
    """A single trial match result with scoring breakdown."""

    nct_id: str
    title: str
    status: str
    phase: str
    sponsor: str | None
    conditions: list[str]
    eligibility_criteria_raw: str | None

    # Scores
    semantic_score: float  # 0-1: how semantically similar
    eligibility_score: float  # 0-1: how well eligibility rules match
    composite_score: float  # 0-1: final combined score

    # Eligibility breakdown
    eligibility_status: str  # ELIGIBLE | LIKELY_ELIGIBLE | NEEDS_REVIEW | INELIGIBLE
    inclusion_met: list[str] = field(default_factory=list)
    exclusion_flags: list[str] = field(default_factory=list)
    needs_verification: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "nct_id": self.nct_id,
            "title": self.title,
            "status": self.status,
            "phase": self.phase,
            "sponsor": self.sponsor,
            "conditions": self.conditions,
            "scores": {
                "semantic": round(self.semantic_score, 3),
                "eligibility": round(self.eligibility_score, 3),
                "composite": round(self.composite_score, 3),
            },
            "eligibility_status": self.eligibility_status,
            "inclusion_met": self.inclusion_met,
            "exclusion_flags": self.exclusion_flags,
            "needs_verification": self.needs_verification,
        }


# -----------------------------------------------
# Eligibility Rules Engine
# -----------------------------------------------
class EligibilityRulesEngine:
    """
    Applies hard eligibility rules to filter and score candidates.

    Rules are applied in order of certainty:
    1. Hard exclusions (definite disqualifiers)
    2. Hard inclusions (definite qualifiers)
    3. Soft checks (need verification)
    """

    def score(
        self,
        patient: PatientProfile,
        trial: dict,
    ) -> tuple[float, str, list, list, list]:
        """
        Returns:
            (eligibility_score, status, inclusion_met, exclusion_flags, needs_verification)
        """
        inclusion_met = []
        exclusion_flags = []
        needs_verification = []

        # ---- Age Check ----
        min_age = trial.get("minimum_age")
        max_age = trial.get("maximum_age")

        if patient.age is not None:
            if min_age is not None and patient.age < min_age:
                exclusion_flags.append(f"Patient age {patient.age} below trial minimum {min_age}")
            elif max_age is not None and patient.age > max_age:
                exclusion_flags.append(f"Patient age {patient.age} above trial maximum {max_age}")
            else:
                if min_age or max_age:
                    inclusion_met.append(f"Age {patient.age} within range")
        else:
            needs_verification.append("Patient age not confirmed")

        # ---- Gender Check ----
        trial_gender = (trial.get("gender") or "ALL").upper()
        if trial_gender not in ("ALL", ""):
            if patient.sex:
                patient_sex = patient.sex.upper()
                if trial_gender == "MALE" and patient_sex != "MALE":
                    exclusion_flags.append("Trial requires male patients")
                elif trial_gender == "FEMALE" and patient_sex != "FEMALE":
                    exclusion_flags.append("Trial requires female patients")
                else:
                    inclusion_met.append(f"Gender match: {patient.sex}")
            else:
                needs_verification.append("Patient gender not confirmed")

        # ---- Condition Match ----
        trial_conditions = []
        if trial.get("conditions"):
            try:
                trial_conditions = (
                    json.loads(trial["conditions"])
                    if isinstance(trial["conditions"], str)
                    else trial["conditions"]
                )
            except Exception:
                pass

        if patient.primary_diagnosis and trial_conditions:
            patient_dx_lower = patient.primary_diagnosis.lower()
            matched_conditions = [
                c
                for c in trial_conditions
                if (
                    patient_dx_lower in c.lower()
                    or c.lower() in patient_dx_lower
                    or self._condition_overlap(patient_dx_lower, c.lower())
                )
            ]
            if matched_conditions:
                inclusion_met.append(f"Condition match: {matched_conditions[0]}")
            else:
                needs_verification.append(
                    f"Condition '{patient.primary_diagnosis}' may not match "
                    f"trial conditions: {', '.join(trial_conditions[:3])}"
                )

        # ---- Negated Conditions vs Eligibility ----
        if patient.negated_conditions and trial.get("eligibility_criteria_raw"):
            criteria_lower = trial["eligibility_criteria_raw"].lower()
            for neg_condition in patient.negated_conditions:
                if neg_condition.lower() in criteria_lower:
                    needs_verification.append(
                        f"Trial mentions '{neg_condition}' — verify relevance to negated condition"
                    )

        # ---- Prior Treatment Conflicts ----
        if patient.prior_treatments and trial.get("eligibility_criteria_raw"):
            criteria_lower = trial["eligibility_criteria_raw"].lower()
            for treatment in patient.prior_treatments:
                treatment_lower = treatment.lower()
                if treatment_lower in criteria_lower:
                    needs_verification.append(
                        f"Prior treatment '{treatment}' mentioned in eligibility — verify"
                    )

        # ---- Lab Value Checks ----
        if patient.lab_values and trial.get("eligibility_criteria_raw"):
            criteria_lower = trial["eligibility_criteria_raw"].lower()
            if "hba1c" in criteria_lower and "HbA1c" in patient.lab_values:
                inclusion_met.append(
                    f"HbA1c {patient.lab_values['HbA1c']} — verify against trial threshold"
                )
            if "egfr" in criteria_lower and "eGFR" in patient.lab_values:
                needs_verification.append(
                    f"eGFR {patient.lab_values['eGFR']} — verify against trial renal criteria"
                )

        # ---- Calculate Score ----
        if exclusion_flags:
            # Hard exclusion — definite disqualifier
            score = 0.1
            status = "INELIGIBLE"
        elif len(inclusion_met) >= 2:
            # Multiple criteria met — strong match
            score = 0.9 - (len(needs_verification) * 0.1)
            score = max(score, 0.5)
            status = "ELIGIBLE" if not needs_verification else "LIKELY_ELIGIBLE"
        elif len(inclusion_met) == 1:
            score = 0.6 - (len(needs_verification) * 0.1)
            score = max(score, 0.3)
            status = "LIKELY_ELIGIBLE"
        else:
            score = 0.3
            status = "NEEDS_REVIEW"

        return score, status, inclusion_met, exclusion_flags, needs_verification

    def _condition_overlap(self, dx1: str, dx2: str) -> bool:
        """Check if two condition strings share significant words."""
        stop_words = {"the", "a", "an", "of", "with", "and", "or", "type"}
        words1 = set(dx1.split()) - stop_words
        words2 = set(dx2.split()) - stop_words
        if not words1 or not words2:
            return False
        overlap = words1 & words2
        return len(overlap) / min(len(words1), len(words2)) > 0.5


# -----------------------------------------------
# Main Matching Engine
# -----------------------------------------------
class TrialMatchingEngine:
    """
    Orchestrates the full matching pipeline.
    Uses lazy loading — model loads on first use, not at import time.
    """

    def __init__(self):
        self._model = None
        self._extractor = PatientProfileExtractor()
        self._rules_engine = EligibilityRulesEngine()

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            print("Loading embedding model...")
            self._model = SentenceTransformer(MODEL_NAME)
        return self._model

    async def match_from_note(
        self,
        note: str,
        db_conn,
        n_results: int = FINAL_RESULTS,
    ) -> tuple[PatientProfile, list[TrialMatch]]:
        """
        Full pipeline: raw note → ranked trial matches.
        Returns (patient_profile, ranked_matches)
        """
        # Step 1: Extract patient profile
        profile = self._extractor.extract(note)

        # Step 2: Match using profile
        matches = await self.match_from_profile(profile, db_conn, n_results)

        return profile, matches

    async def match_from_profile(
        self,
        profile: PatientProfile,
        db_conn,
        n_results: int = FINAL_RESULTS,
    ) -> list[TrialMatch]:
        """
        Match from a pre-extracted patient profile.
        """
        # Step 1: Build query string from profile
        query_text = profile.to_query_string()
        if not query_text:
            return []

        # Step 2: Embed the query
        query_embedding = self.model.encode(
            [query_text],
            normalize_embeddings=True,
        )[0]

        # Step 3: Semantic search — get top 50 candidates from pgvector
        candidates = await db_conn.fetch(
            """
            SELECT
                nct_id, title, status, phase, sponsor,
                conditions, eligibility_criteria_raw,
                minimum_age, maximum_age, gender,
                1 - (embedding <=> $1::vector) as semantic_score
            FROM trials
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        """,
            str(query_embedding.tolist()),
            SEMANTIC_CANDIDATES,
        )

        if not candidates:
            return []

        # Step 4: Apply eligibility rules to each candidate
        matches = []
        for trial in candidates:
            trial_dict = dict(trial)
            semantic_score = float(trial_dict.pop("semantic_score"))

            (
                eligibility_score,
                status,
                inclusion_met,
                exclusion_flags,
                needs_verification,
            ) = self._rules_engine.score(profile, trial_dict)

            # Step 5: Calculate composite score
            # Semantic similarity: 60% weight
            # Eligibility rules: 40% weight
            composite = (semantic_score * 0.6) + (eligibility_score * 0.4)

            # Parse conditions
            conditions = []
            if trial_dict.get("conditions"):
                try:
                    conditions = (
                        json.loads(trial_dict["conditions"])
                        if isinstance(trial_dict["conditions"], str)
                        else trial_dict["conditions"]
                    )
                except Exception:
                    pass

            match = TrialMatch(
                nct_id=trial_dict["nct_id"],
                title=trial_dict["title"],
                status=trial_dict["status"],
                phase=trial_dict["phase"],
                sponsor=trial_dict.get("sponsor"),
                conditions=conditions[:5],
                eligibility_criteria_raw=trial_dict.get("eligibility_criteria_raw"),
                semantic_score=semantic_score,
                eligibility_score=eligibility_score,
                composite_score=composite,
                eligibility_status=status,
                inclusion_met=inclusion_met,
                exclusion_flags=exclusion_flags,
                needs_verification=needs_verification,
            )
            matches.append(match)

        # Step 6: Sort by composite score, filter out clear ineligibles
        matches.sort(key=lambda m: m.composite_score, reverse=True)

        # Return top N, excluding definite ineligibles unless we don't have enough
        eligible_matches = [m for m in matches if m.eligibility_status != "INELIGIBLE"]
        if len(eligible_matches) >= n_results:
            return eligible_matches[:n_results]
        else:
            return matches[:n_results]


# -----------------------------------------------
# Test the full pipeline
# -----------------------------------------------
async def run_test():
    print("=" * 65)
    print("TrialBridge — Day 9: Full Matching Pipeline Test")
    print("=" * 65)

    # Connect to database
    conn = await asyncpg.connect(DB_URL)
    engine = TrialMatchingEngine()

    test_cases = [
        {
            "label": "Type 2 Diabetes patient",
            "note": """
                58-year-old female with Type 2 Diabetes Mellitus.
                Currently on Metformin 1000mg. HbA1c 8.9%, eGFR 72.
                No prior insulin therapy. No history of cardiovascular disease.
                Blood pressure 138/82. BMI 31.2.
            """,
        },
        {
            "label": "Cancer patient",
            "note": """
                45-year-old male with breast cancer diagnosed 6 months ago.
                Previously treated with carboplatin. Currently receiving pembrolizumab.
                No history of diabetes or hypertension.
                ALT 42, AST 38, creatinine 0.9.
            """,
        },
        {
            "label": "Heart failure patient",
            "note": """
                72-year-old male with heart failure with reduced ejection fraction.
                History of hypertension and atrial fibrillation.
                Currently on lisinopril, warfarin, and metformin.
                HbA1c 7.2. eGFR 58. Denies history of stroke.
            """,
        },
    ]

    for test in test_cases:
        print(f"\n{'─' * 65}")
        print(f"PATIENT: {test['label']}")
        print(f"{'─' * 65}")

        profile, matches = await engine.match_from_note(test["note"], conn, n_results=5)

        print("Extracted Profile:")
        print(f"  Age: {profile.age} | Sex: {profile.sex}")
        print(f"  Diagnosis: {profile.primary_diagnosis}")
        print(f"  Medications: {profile.current_medications}")
        print(f"  Lab values: {profile.lab_values}")
        print(f"  Negated: {profile.negated_conditions}")

        print(f"\nTop {len(matches)} Matched Trials:")
        for i, match in enumerate(matches, 1):
            print(f"\n  [{i}] {match.nct_id} — {match.eligibility_status}")
            print(f"      Title: {match.title[:65]}")
            print(
                f"      Scores: semantic={match.semantic_score:.3f} | "
                f"eligibility={match.eligibility_score:.3f} | "
                f"composite={match.composite_score:.3f}"
            )
            if match.inclusion_met:
                print(f"      ✅ Met: {match.inclusion_met[0]}")
            if match.exclusion_flags:
                print(f"      ❌ Flag: {match.exclusion_flags[0]}")
            if match.needs_verification:
                print(f"      ⚠️  Verify: {match.needs_verification[0]}")

    await conn.close()
    print(f"\n{'=' * 65}")
    print("Day 9 complete. Full matching pipeline working end-to-end.")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(run_test())
