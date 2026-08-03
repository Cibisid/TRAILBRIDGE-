"""
TrialBridge — Eligibility criteria parser tests

Fixtures use the real shape of ClinicalTrials.gov's
`eligibility/criteria/textblock`: leading indentation, a blank line after each
header, and two-space bullet markers.
"""

from backend.agent.criteria import (
    EXCLUSION,
    INCLUSION,
    UNSPECIFIED,
    parse_eligibility_criteria,
)

CTGOV_TEXTBLOCK = """
        Inclusion Criteria:

          -  Adults aged 18 to 75 years

          -  Diagnosis of Type 2 Diabetes Mellitus for at least 6 months

          -  HbA1c between 7.0% and 10.5% at screening

        Exclusion Criteria:

          -  Prior insulin therapy

          -  eGFR less than 45 mL/min/1.73m2
"""


def test_splits_inclusion_and_exclusion():
    criteria = parse_eligibility_criteria(CTGOV_TEXTBLOCK)

    assert [c.id for c in criteria] == ["INC-1", "INC-2", "INC-3", "EXC-1", "EXC-2"]
    assert [c.kind for c in criteria[:3]] == [INCLUSION] * 3
    assert [c.kind for c in criteria[3:]] == [EXCLUSION] * 2


def test_criterion_text_is_cleaned():
    criteria = parse_eligibility_criteria(CTGOV_TEXTBLOCK)

    assert criteria[0].text == "Adults aged 18 to 75 years"
    # Thresholds must survive parsing intact — they are what gets compared.
    assert criteria[2].text == "HbA1c between 7.0% and 10.5% at screening"
    assert criteria[4].text == "eGFR less than 45 mL/min/1.73m2"


def test_wrapped_bullets_stay_one_criterion():
    """A criterion that wraps across lines is one claim, not several."""
    raw = """
        Inclusion Criteria:

          -  Histologically confirmed adenocarcinoma of the colon or rectum
             with measurable disease per RECIST v1.1 criteria

          -  ECOG performance status 0 or 1
    """
    criteria = parse_eligibility_criteria(raw)

    assert len(criteria) == 2
    assert criteria[0].text.endswith("per RECIST v1.1 criteria")
    assert "measurable disease" in criteria[0].text


def test_numbered_and_alternate_bullet_markers():
    raw = """
        Inclusion Criteria:

          1. Age 18 or older
          2. Signed informed consent

        Exclusion Criteria:

          * Active infection
    """
    criteria = parse_eligibility_criteria(raw)

    assert [c.text for c in criteria] == [
        "Age 18 or older",
        "Signed informed consent",
        "Active infection",
    ]
    assert criteria[2].kind == EXCLUSION


def test_text_without_headers_is_unspecified_not_inclusion():
    """
    Polarity must not be guessed. Defaulting headerless text to "inclusion"
    would let the aggregator disqualify a patient on a rule it never actually
    read as a requirement.
    """
    criteria = parse_eligibility_criteria("  -  Age 18 or older\n  -  Able to provide consent\n")

    assert len(criteria) == 2
    assert all(c.kind == UNSPECIFIED for c in criteria)
    assert all(c.id.startswith("UNK-") for c in criteria)


def test_prose_without_bullets_still_splits():
    raw = """
        Inclusion Criteria:

        Patients must be at least 18 years of age. Patients must have a
        confirmed diagnosis. Written consent is required.
    """
    criteria = parse_eligibility_criteria(raw)

    assert len(criteria) >= 2
    assert any("18 years of age" in c.text for c in criteria)


def test_key_inclusion_criteria_header_variant():
    raw = """
        Key Inclusion Criteria:

          -  Age 18 or older

        Key Exclusion Criteria:

          -  Pregnancy
    """
    criteria = parse_eligibility_criteria(raw)

    assert criteria[0].kind == INCLUSION
    assert criteria[1].kind == EXCLUSION


def test_empty_and_missing_input_return_no_criteria():
    assert parse_eligibility_criteria(None) == []
    assert parse_eligibility_criteria("") == []
    assert parse_eligibility_criteria("   \n\n  ") == []


def test_ids_are_unique():
    criteria = parse_eligibility_criteria(CTGOV_TEXTBLOCK)
    ids = [c.id for c in criteria]

    assert len(ids) == len(set(ids))
