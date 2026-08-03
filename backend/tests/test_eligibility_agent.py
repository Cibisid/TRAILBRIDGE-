"""
TrialBridge — Verdict aggregation tests

`_aggregate` turns per-criterion verdicts into the trial-level status a
clinician reads. The property worth protecting is one-directional: a trial may
only come back QUALIFIES when every criterion was actually decided. Anything
undecided has to surface as UNKNOWN, with the missing data named.
"""

from backend.agent.criteria import EXCLUSION, INCLUSION, UNSPECIFIED
from backend.agent.eligibility_agent import (
    EXCLUDED,
    INDETERMINATE,
    MET,
    NOT_MET,
    QUALIFIES,
    UNKNOWN,
    CriterionVerdict,
    EligibilityAgent,
)

aggregate = EligibilityAgent._aggregate


def verdict(cid, kind, value, missing=None):
    return CriterionVerdict(
        criterion_id=cid,
        kind=kind,
        text=f"criterion {cid}",
        verdict=value,
        evidence="test",
        missing_data=missing,
    )


def test_all_inclusions_met_qualifies():
    status, missing, blocking = aggregate(
        [
            verdict("INC-1", INCLUSION, MET),
            verdict("INC-2", INCLUSION, MET),
            verdict("EXC-1", EXCLUSION, NOT_MET),
        ]
    )

    assert status == QUALIFIES
    assert missing == []
    assert blocking is None


def test_meeting_an_exclusion_disqualifies():
    """An exclusion criterion excludes when the patient MEETS it."""
    status, _, blocking = aggregate(
        [verdict("INC-1", INCLUSION, MET), verdict("EXC-1", EXCLUSION, MET)]
    )

    assert status == EXCLUDED
    assert blocking == "EXC-1"


def test_unmet_inclusion_disqualifies():
    status, _, blocking = aggregate([verdict("INC-2", INCLUSION, NOT_MET)])

    assert status == EXCLUDED
    assert blocking == "INC-2"


def test_unknown_criterion_blocks_qualification():
    """One undecided criterion is enough to withhold a pass."""
    status, missing, _ = aggregate(
        [
            verdict("INC-1", INCLUSION, MET),
            verdict("INC-2", INCLUSION, UNKNOWN, "eGFR draw date"),
        ]
    )

    assert status == INDETERMINATE
    assert missing == ["eGFR draw date"]


def test_disqualifier_wins_over_unknowns():
    """
    A known exclusion settles the trial. Reporting UNKNOWN here would send a
    coordinator chasing a lab value for a trial the patient cannot enter.
    """
    status, missing, blocking = aggregate(
        [
            verdict("INC-1", INCLUSION, UNKNOWN, "platelet count"),
            verdict("EXC-1", EXCLUSION, MET),
        ]
    )

    assert status == EXCLUDED
    assert blocking == "EXC-1"
    assert missing == []


def test_every_missing_data_point_is_reported():
    status, missing, _ = aggregate(
        [
            verdict("INC-1", INCLUSION, UNKNOWN, "eGFR draw date"),
            verdict("INC-2", INCLUSION, UNKNOWN, "ECOG performance status"),
            verdict("INC-3", INCLUSION, MET),
        ]
    )

    assert status == INDETERMINATE
    assert missing == ["eGFR draw date", "ECOG performance status"]


def test_unmet_criterion_of_unknown_polarity_does_not_disqualify():
    """
    Without an inclusion/exclusion header the trial text never said whether the
    rule is a requirement or a disqualifier, so an unmet criterion cannot be
    read as either. It is UNKNOWN, not EXCLUDED and not QUALIFIES.
    """
    status, missing, blocking = aggregate([verdict("UNK-1", UNSPECIFIED, NOT_MET)])

    assert status == INDETERMINATE
    assert blocking is None
    assert "UNK-1" in missing[0]


def test_unspecified_criterion_met_can_still_qualify():
    status, _, _ = aggregate([verdict("INC-1", INCLUSION, MET), verdict("UNK-1", UNSPECIFIED, MET)])

    assert status == QUALIFIES


def test_qualifies_requires_at_least_one_decided_criterion():
    """A trial whose criteria all came back UNKNOWN must never read as a pass."""
    status, _, _ = aggregate(
        [
            verdict("INC-1", INCLUSION, UNKNOWN, "age"),
            verdict("INC-2", INCLUSION, UNKNOWN, "diagnosis"),
        ]
    )

    assert status != QUALIFIES
    assert status == INDETERMINATE
