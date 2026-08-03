"""
TrialBridge — Evaluation dataset integrity

The eval labels are keyed by criterion id (INC-1, EXC-2, ...), and those ids
come out of the parser. If the parser ever splits or renumbers criteria, every
label silently points at the wrong claim and the published metric becomes
meaningless while still looking fine.

These tests run without AWS or a database, so CI catches that drift on the
commit that causes it rather than the next time someone runs the eval.
"""

import json
from pathlib import Path

import pytest

from backend.agent.criteria import parse_eligibility_criteria
from backend.agent.eligibility_agent import (
    EXCLUDED,
    INDETERMINATE,
    MET,
    NOT_MET,
    QUALIFIES,
    UNKNOWN,
)

CASES_PATH = Path(__file__).parents[2] / "evaluation" / "cases.json"

VALID_VERDICTS = {MET, NOT_MET, UNKNOWN}
VALID_STATUSES = {QUALIFIES, EXCLUDED, INDETERMINATE}


def load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]


CASES = load_cases()


def test_cases_file_exists_and_is_populated():
    assert CASES, "evaluation/cases.json has no cases"


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_labels_match_parsed_criterion_ids(case):
    """Every labeled id must be one the parser actually produces, and vice versa."""
    parsed = {c.id for c in parse_eligibility_criteria(case["criteria_text"])}
    labeled = set(case["expected_verdicts"])

    assert parsed == labeled, (
        f"{case['id']}: parser produced {sorted(parsed)} but labels cover {sorted(labeled)}"
    )


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_verdict_and_status_values_are_valid(case):
    for criterion_id, verdict in case["expected_verdicts"].items():
        assert verdict in VALID_VERDICTS, f"{case['id']}/{criterion_id}: bad verdict {verdict}"

    assert case["expected_status"] in VALID_STATUSES


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_expected_status_follows_from_expected_verdicts(case):
    """
    The labels must be internally consistent with the aggregation rules, or the
    eval would penalise the agent for correctly following them.
    """
    verdicts = case["expected_verdicts"]
    status = case["expected_status"]

    has_disqualifier = any(
        (cid.startswith("EXC") and v == MET) or (cid.startswith("INC") and v == NOT_MET)
        for cid, v in verdicts.items()
    )
    has_unknown = any(v == UNKNOWN for v in verdicts.values())
    has_unmet_unspecified = any(
        cid.startswith("UNK") and v == NOT_MET for cid, v in verdicts.items()
    )

    if has_disqualifier:
        assert status == EXCLUDED, f"{case['id']}: a disqualifier must settle the trial"
    elif has_unknown or has_unmet_unspecified:
        assert status == INDETERMINATE, f"{case['id']}: undecided criteria cannot QUALIFY"
    else:
        assert status == QUALIFIES


def test_dataset_covers_both_failure_modes():
    """
    The two headline rates each need a denominator. Without undecidable
    criteria the unsupported-decision rate is undefined; without decidable
    ones the over-abstention rate is.
    """
    all_verdicts = [v for c in CASES for v in c["expected_verdicts"].values()]

    assert sum(1 for v in all_verdicts if v == UNKNOWN) >= 3
    assert sum(1 for v in all_verdicts if v != UNKNOWN) >= 10


def test_case_ids_are_unique():
    ids = [c["id"] for c in CASES]

    assert len(ids) == len(set(ids))
