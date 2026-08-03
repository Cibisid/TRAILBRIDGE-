"""
TrialBridge — Patient Data Store

The agent never sees the patient note. It sees this.

Every value the agent wants must be requested through one of the lookup methods
below, and each one answers with an explicit presence status. When a note does
not contain a value, the store says so — it does not return an empty string, a
zero, or a plausible default. That is the whole mechanism: an agent that cannot
read a lab value cannot invent one, so "I don't know" becomes the structurally
available answer rather than something we have to prompt for and hope.

Every lookup is recorded. `audit()` returns what was asked and what came back,
which is what the API surfaces as the evidence trail behind a verdict.
"""

import re
from dataclasses import dataclass, field

from backend.nlp.extractor_v2 import PatientProfile

# -----------------------------------------------
# Presence Semantics
# -----------------------------------------------
PRESENT = "PRESENT"
NOT_PRESENT = "NOT_PRESENT"

# Conditions are tri-state on purpose. "The note doesn't mention diabetes" and
# "the note says the patient does not have diabetes" are different facts, and
# collapsing them is how a matcher ends up asserting something clinically false.
CONFIRMED_PRESENT = "CONFIRMED_PRESENT"
EXPLICITLY_ABSENT = "EXPLICITLY_ABSENT"
NOT_MENTIONED = "NOT_MENTIONED"

CURRENTLY_TAKING = "CURRENTLY_TAKING"
PRIOR_TREATMENT = "PRIOR_TREATMENT"

SCALAR_FIELDS = ("age", "sex", "primary_diagnosis", "ecog_score")
LIST_FIELDS = (
    "comorbidities",
    "negated_conditions",
    "current_medications",
    "prior_treatments",
    "allergies",
)

# Lab synonyms seen in both clinical notes and trial criteria. Deliberately
# short — every entry is a claim that two names mean the same measurement, and
# a wrong entry here produces a confidently wrong verdict.
_LAB_ALIASES = {
    "a1c": "hba1c",
    "hba1c": "hba1c",
    "hemoglobina1c": "hba1c",
    "glycatedhemoglobin": "hba1c",
    "egfr": "egfr",
    "gfr": "egfr",
    "estimatedgfr": "egfr",
    "creatinine": "creatinine",
    "scr": "creatinine",
    "hgb": "hemoglobin",
    "hb": "hemoglobin",
    "hemoglobin": "hemoglobin",
    "plt": "platelets",
    "platelets": "platelets",
    "platelet": "platelets",
    "wbc": "wbc",
    "anc": "anc",
    "bmi": "bmi",
    "ldl": "ldl",
    "hdl": "hdl",
}

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _normalize(name: str) -> str:
    return _NON_ALNUM.sub("", (name or "").lower())


def _canonical_lab(name: str) -> str:
    key = _normalize(name)
    return _LAB_ALIASES.get(key, key)


# -----------------------------------------------
# Lookup Result
# -----------------------------------------------
@dataclass
class LookupResult:
    status: str
    value: object = None
    note: str | None = None

    def to_dict(self) -> dict:
        out: dict = {"status": self.status}
        if self.value is not None:
            out["value"] = self.value
        if self.note:
            out["note"] = self.note
        return out


@dataclass
class PatientDataStore:
    """Read-only, audited access to a PatientProfile."""

    profile: PatientProfile
    _log: list[dict] = field(default_factory=list)

    def _record(self, tool: str, argument: str, result: LookupResult) -> LookupResult:
        self._log.append(
            {"tool": tool, "argument": argument, "status": result.status, "value": result.value}
        )
        return result

    # -------------------------------------------
    # Tool-backed lookups
    # -------------------------------------------
    def lookup_field(self, field_name: str) -> LookupResult:
        """Fetch a single structured field from the patient profile."""
        name = (field_name or "").strip().lower()

        if name not in SCALAR_FIELDS + LIST_FIELDS:
            return self._record(
                "lookup_patient_field",
                field_name,
                LookupResult(
                    NOT_PRESENT,
                    note=(
                        f"'{field_name}' is not a field this patient record has. "
                        f"Available: {', '.join(SCALAR_FIELDS + LIST_FIELDS)}."
                    ),
                ),
            )

        value = getattr(self.profile, name, None)

        # An empty list means the extractor found nothing of that kind in the
        # note — not that the patient has none. Report it as absent data.
        if value is None or (isinstance(value, list) and not value):
            return self._record(
                "lookup_patient_field",
                field_name,
                LookupResult(NOT_PRESENT, note=f"The note does not state the patient's {name}."),
            )

        return self._record("lookup_patient_field", field_name, LookupResult(PRESENT, value))

    def lookup_lab(self, lab_name: str) -> LookupResult:
        """
        Fetch a lab value by name.

        Returns the numeric value only. Notes in this corpus rarely carry draw
        dates, so a criterion requiring a value measured within a time window
        cannot be settled from the value alone — the caller must treat the
        missing date as missing data rather than assuming the value is current.
        """
        wanted = _canonical_lab(lab_name)
        for key, value in (self.profile.lab_values or {}).items():
            if _canonical_lab(key) == wanted:
                return self._record(
                    "lookup_lab_value",
                    lab_name,
                    LookupResult(PRESENT, value, note=f"Recorded in the note as '{key}'."),
                )

        return self._record(
            "lookup_lab_value",
            lab_name,
            LookupResult(NOT_PRESENT, note=f"No {lab_name} value appears in the note."),
        )

    def check_condition(self, condition: str) -> LookupResult:
        """
        Determine whether the patient has a condition.

        Three outcomes, and the third is the important one: a note that never
        mentions a condition is not evidence the patient lacks it.
        """
        target = _normalize(condition)
        if not target:
            return self._record(
                "check_condition", condition, LookupResult(NOT_MENTIONED, note="Empty condition.")
            )

        def _matches(candidate: str) -> bool:
            other = _normalize(candidate)
            return bool(other) and (target in other or other in target)

        for negated in self.profile.negated_conditions or []:
            if _matches(negated):
                return self._record(
                    "check_condition",
                    condition,
                    LookupResult(
                        EXPLICITLY_ABSENT,
                        negated,
                        note=f"The note explicitly rules out '{negated}'.",
                    ),
                )

        positives = list(self.profile.comorbidities or [])
        if self.profile.primary_diagnosis:
            positives.insert(0, self.profile.primary_diagnosis)

        for positive in positives:
            if _matches(positive):
                return self._record(
                    "check_condition",
                    condition,
                    LookupResult(CONFIRMED_PRESENT, positive),
                )

        return self._record(
            "check_condition",
            condition,
            LookupResult(
                NOT_MENTIONED,
                note=(
                    f"The note does not mention '{condition}' either way. "
                    "This is not evidence of absence."
                ),
            ),
        )

    def check_medication(self, medication: str) -> LookupResult:
        """Distinguish a current medication from a prior treatment from silence."""
        target = _normalize(medication)
        if not target:
            return self._record(
                "check_medication",
                medication,
                LookupResult(NOT_MENTIONED, note="Empty medication."),
            )

        def _matches(candidate: str) -> bool:
            other = _normalize(candidate)
            return bool(other) and (target in other or other in target)

        for current in self.profile.current_medications or []:
            if _matches(current):
                return self._record(
                    "check_medication", medication, LookupResult(CURRENTLY_TAKING, current)
                )

        for prior in self.profile.prior_treatments or []:
            if _matches(prior):
                return self._record(
                    "check_medication", medication, LookupResult(PRIOR_TREATMENT, prior)
                )

        for negated in self.profile.negated_conditions or []:
            if _matches(negated):
                return self._record(
                    "check_medication",
                    medication,
                    LookupResult(
                        EXPLICITLY_ABSENT,
                        negated,
                        note=f"The note explicitly rules out '{negated}'.",
                    ),
                )

        return self._record(
            "check_medication",
            medication,
            LookupResult(
                NOT_MENTIONED,
                note=f"The note does not mention '{medication}'. This is not evidence of absence.",
            ),
        )

    # -------------------------------------------
    # Audit
    # -------------------------------------------
    def audit(self) -> list[dict]:
        """Every lookup made, in order. This is the evidence trail for a verdict."""
        return list(self._log)

    def missing_requests(self) -> list[str]:
        """Arguments whose lookups came back without data — i.e. what the note lacks."""
        absent = {NOT_PRESENT, NOT_MENTIONED}
        seen: list[str] = []
        for entry in self._log:
            if entry["status"] in absent and entry["argument"] not in seen:
                seen.append(entry["argument"])
        return seen
