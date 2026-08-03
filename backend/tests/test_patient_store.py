"""
TrialBridge — Patient data store tests

The store is the reason the agent cannot invent a value, so these tests are
mostly about what it refuses to say. The important assertions are the negative
ones: a missing lab reports NOT_PRESENT rather than a default, and an unmentioned
condition reports NOT_MENTIONED rather than absent.
"""

import pytest

from backend.agent.patient_store import (
    CONFIRMED_PRESENT,
    CURRENTLY_TAKING,
    EXPLICITLY_ABSENT,
    NOT_MENTIONED,
    NOT_PRESENT,
    PRESENT,
    PRIOR_TREATMENT,
    PatientDataStore,
)
from backend.nlp.extractor_v2 import PatientProfile


@pytest.fixture
def store():
    profile = PatientProfile(
        age=58,
        sex="female",
        primary_diagnosis="Type 2 Diabetes",
        comorbidities=["hypertension"],
        negated_conditions=["cardiovascular disease"],
        current_medications=["Metformin"],
        prior_treatments=["sulfonylurea"],
        lab_values={"HbA1c": 8.9, "eGFR": 72.0},
    )
    return PatientDataStore(profile=profile)


# -----------------------------------------------
# Fields
# -----------------------------------------------
def test_present_field_returns_value(store):
    result = store.lookup_field("age")

    assert result.status == PRESENT
    assert result.value == 58


def test_absent_field_reports_not_present(store):
    """ECOG was never recorded. The store must not substitute a default."""
    result = store.lookup_field("ecog_score")

    assert result.status == NOT_PRESENT
    assert result.value is None


def test_empty_list_field_is_not_present(store):
    """An empty allergy list means the note was silent, not that there are none."""
    result = store.lookup_field("allergies")

    assert result.status == NOT_PRESENT


def test_unknown_field_name_is_rejected_with_guidance(store):
    result = store.lookup_field("hemoglobin")

    assert result.status == NOT_PRESENT
    assert "not a field" in result.note


# -----------------------------------------------
# Labs
# -----------------------------------------------
def test_lab_lookup_is_alias_aware(store):
    """Criteria say "A1c" or "Hemoglobin A1c"; notes say "HbA1c"."""
    for name in ("HbA1c", "hba1c", "A1c", "hemoglobin a1c"):
        result = store.lookup_lab(name)
        assert result.status == PRESENT, name
        assert result.value == 8.9


def test_missing_lab_reports_not_present(store):
    result = store.lookup_lab("platelets")

    assert result.status == NOT_PRESENT
    assert result.value is None


# -----------------------------------------------
# Conditions — the tri-state
# -----------------------------------------------
def test_diagnosis_is_confirmed_present(store):
    assert store.check_condition("Type 2 Diabetes").status == CONFIRMED_PRESENT


def test_comorbidity_is_confirmed_present(store):
    assert store.check_condition("hypertension").status == CONFIRMED_PRESENT


def test_negated_condition_is_explicitly_absent(store):
    """ "No history of cardiovascular disease" is a positive finding of absence."""
    assert store.check_condition("cardiovascular disease").status == EXPLICITLY_ABSENT


def test_unmentioned_condition_is_not_absent(store):
    """
    The single most important distinction in this module. The note never
    mentions asthma; that is not evidence the patient does not have it, and
    reporting it as absent would let a trial exclude or admit on a fact nobody
    established.
    """
    result = store.check_condition("asthma")

    assert result.status == NOT_MENTIONED
    assert result.status != EXPLICITLY_ABSENT
    assert "not evidence of absence" in result.note


def test_negation_is_checked_before_positives():
    """A term appearing in both lists must resolve as ruled out, not present."""
    profile = PatientProfile(
        primary_diagnosis="Breast Cancer",
        comorbidities=["diabetes"],
        negated_conditions=["diabetes"],
    )
    result = PatientDataStore(profile=profile).check_condition("diabetes")

    assert result.status == EXPLICITLY_ABSENT


# -----------------------------------------------
# Medications
# -----------------------------------------------
def test_current_medication(store):
    assert store.check_medication("Metformin").status == CURRENTLY_TAKING


def test_prior_treatment_is_distinct_from_current(store):
    """Trials routinely gate on prior exposure, so the two cannot collapse."""
    result = store.check_medication("sulfonylurea")

    assert result.status == PRIOR_TREATMENT
    assert result.status != CURRENTLY_TAKING


def test_unmentioned_medication_is_not_absent(store):
    assert store.check_medication("insulin").status == NOT_MENTIONED


# -----------------------------------------------
# Audit trail
# -----------------------------------------------
def test_audit_records_every_lookup_in_order(store):
    store.lookup_field("age")
    store.lookup_lab("HbA1c")
    store.check_condition("asthma")

    audit = store.audit()

    assert [entry["tool"] for entry in audit] == [
        "lookup_patient_field",
        "lookup_lab_value",
        "check_condition",
    ]
    assert audit[0]["argument"] == "age"


def test_missing_requests_lists_only_absent_lookups(store):
    store.lookup_field("age")
    store.lookup_lab("platelets")
    store.check_condition("asthma")

    assert store.missing_requests() == ["platelets", "asthma"]


def test_audit_is_a_copy(store):
    store.lookup_field("age")
    store.audit().clear()

    assert len(store.audit()) == 1
