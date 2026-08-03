"""
TrialBridge — Eligibility agent evaluation

Scores the agent against hand-labeled criteria in evaluation/cases.json and
prints a reproducible report.

Two rates are reported together, and neither means anything alone:

  Unsupported-decision rate
      Criteria the agent decided MET or NOT_MET when the record could not
      support either. These are the failures that matter clinically — a
      confident answer built on a value nobody measured.

  Over-abstention rate
      Criteria the agent called UNKNOWN when the record did settle them. This
      is the price of the design.

Reporting only the first would be dishonest: any system can drive unsupported
decisions to zero by answering UNKNOWN to everything, and that system is
useless. The pair is the actual result.

Usage:
    python -m evaluation.run_eval
    python -m evaluation.run_eval --output evaluation/results.json
"""

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

from backend.agent.criteria import parse_eligibility_criteria
from backend.agent.eligibility_agent import UNKNOWN, EligibilityAgent
from backend.agent.patient_store import PatientDataStore
from backend.nlp.extractor_v2 import PatientProfile

CASES_PATH = Path(__file__).parent / "cases.json"
# Bedrock throughput is the limit here, not local CPU.
MAX_CONCURRENCY = 4


@dataclass
class CaseResult:
    case_id: str
    expected_status: str
    actual_status: str
    correct_verdicts: int = 0
    total_verdicts: int = 0
    unsupported_decisions: list[dict] = field(default_factory=list)
    over_abstentions: list[dict] = field(default_factory=list)
    mismatches: list[dict] = field(default_factory=list)
    error: str | None = None

    @property
    def status_correct(self) -> bool:
        return self.expected_status == self.actual_status


def build_profile(data: dict) -> PatientProfile:
    return PatientProfile(
        age=data.get("age"),
        sex=data.get("sex"),
        primary_diagnosis=data.get("primary_diagnosis"),
        comorbidities=data.get("comorbidities", []),
        negated_conditions=data.get("negated_conditions", []),
        current_medications=data.get("current_medications", []),
        prior_treatments=data.get("prior_treatments", []),
        allergies=data.get("allergies", []),
        lab_values=data.get("lab_values", {}),
        ecog_score=data.get("ecog_score"),
    )


async def run_case(agent: EligibilityAgent, case: dict, sem: asyncio.Semaphore) -> CaseResult:
    async with sem:
        criteria = parse_eligibility_criteria(case["criteria_text"])
        store = PatientDataStore(profile=build_profile(case["profile"]))
        expected = case["expected_verdicts"]

        result = CaseResult(
            case_id=case["id"],
            expected_status=case["expected_status"],
            actual_status="",
        )

        # A parser change that renumbers criteria would silently invalidate
        # every label, so fail loudly rather than score against wrong ids.
        parsed_ids = {c.id for c in criteria}
        if parsed_ids != set(expected):
            result.error = (
                f"Criterion ids drifted from the labels. "
                f"parsed={sorted(parsed_ids)} labeled={sorted(expected)}"
            )
            return result

        assessment = await agent.assess_trial(case["nct_id"], criteria, store)
        result.actual_status = assessment.status
        if assessment.error:
            result.error = assessment.error

        for verdict in assessment.verdicts:
            want = expected.get(verdict.criterion_id)
            got = verdict.verdict
            result.total_verdicts += 1

            if want == got:
                result.correct_verdicts += 1
                continue

            record = {
                "criterion_id": verdict.criterion_id,
                "criterion": verdict.text,
                "expected": want,
                "actual": got,
                "evidence": verdict.evidence,
            }
            result.mismatches.append(record)

            if want == UNKNOWN and got != UNKNOWN:
                # Decided something the record could not support.
                result.unsupported_decisions.append(record)
            elif want != UNKNOWN and got == UNKNOWN:
                result.over_abstentions.append(record)

        return result


def report(results: list[CaseResult]) -> dict:
    total_verdicts = sum(r.total_verdicts for r in results)
    correct_verdicts = sum(r.correct_verdicts for r in results)
    unsupported = sum(len(r.unsupported_decisions) for r in results)
    over_abstained = sum(len(r.over_abstentions) for r in results)
    status_correct = sum(1 for r in results if r.status_correct)
    errored = [r for r in results if r.error]

    # Denominators differ per rate: an unsupported decision is only possible on
    # a criterion that should have been UNKNOWN, an over-abstention only on one
    # that should have been decided. Dividing both by the total criterion count
    # would understate each of them.
    expected_unknown = _count_expected(results, UNKNOWN)
    expected_decided = _count_expected(results, None)

    summary = {
        "cases": len(results),
        "criteria": total_verdicts,
        "verdict_accuracy": _pct(correct_verdicts, total_verdicts),
        "trial_status_accuracy": _pct(status_correct, len(results)),
        "unsupported_decision_rate": _pct(unsupported, expected_unknown),
        "unsupported_decisions": unsupported,
        "undecidable_criteria": expected_unknown,
        "over_abstention_rate": _pct(over_abstained, expected_decided),
        "over_abstentions": over_abstained,
        "decidable_criteria": expected_decided,
        "errored_cases": [r.case_id for r in errored],
    }
    return summary


_LABELS: dict[str, dict] = {}


def _count_expected(results: list[CaseResult], want: str | None) -> int:
    """Count labeled criteria whose expected verdict is (or is not) `want`."""
    total = 0
    for r in results:
        labels = _LABELS.get(r.case_id, {})
        for expected in labels.values():
            if want is None:
                total += expected != UNKNOWN
            else:
                total += expected == want
    return total


def _pct(numerator: int, denominator: int) -> float | None:
    if not denominator:
        return None
    return round(100.0 * numerator / denominator, 1)


def print_report(results: list[CaseResult], summary: dict) -> None:
    print("=" * 72)
    print("TrialBridge — eligibility agent evaluation")
    print("=" * 72)

    for r in results:
        mark = "PASS" if r.status_correct and not r.mismatches else "FAIL"
        detail = f"{r.correct_verdicts}/{r.total_verdicts} criteria"
        status = f"{r.actual_status} (expected {r.expected_status})"
        print(f"  [{mark}] {r.case_id:<32} {detail:<18} {status}")
        if r.error:
            print(f"         error: {r.error}")
        for m in r.mismatches:
            print(f"         {m['criterion_id']}: expected {m['expected']}, got {m['actual']}")
            print(f"           criterion: {m['criterion']}")
            if m["evidence"]:
                print(f"           agent said: {m['evidence']}")

    print("-" * 72)
    print(f"  Cases                      {summary['cases']}")
    print(f"  Criteria                   {summary['criteria']}")
    print(f"  Verdict accuracy           {summary['verdict_accuracy']}%")
    print(f"  Trial status accuracy      {summary['trial_status_accuracy']}%")
    print()
    print(
        f"  Unsupported decisions      {summary['unsupported_decisions']}"
        f"/{summary['undecidable_criteria']}"
        f"  ({summary['unsupported_decision_rate']}%)"
    )
    print("    criteria the record could not settle, decided anyway")
    print(
        f"  Over-abstentions           {summary['over_abstentions']}"
        f"/{summary['decidable_criteria']}"
        f"  ({summary['over_abstention_rate']}%)"
    )
    print("    criteria the record did settle, called UNKNOWN")
    if summary["errored_cases"]:
        print(f"  Errored cases              {', '.join(summary['errored_cases'])}")
    print("=" * 72)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--effort", default=None, help="Override Bedrock effort level.")
    args = parser.parse_args()

    payload = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = payload["cases"]
    for case in cases:
        _LABELS[case["id"]] = case["expected_verdicts"]

    agent = EligibilityAgent(effort=args.effort)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    results = await asyncio.gather(*(run_case(agent, c, sem) for c in cases))

    summary = report(list(results))
    print_report(list(results), summary)

    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "cases": [
                        {
                            "case_id": r.case_id,
                            "expected_status": r.expected_status,
                            "actual_status": r.actual_status,
                            "correct_verdicts": r.correct_verdicts,
                            "total_verdicts": r.total_verdicts,
                            "mismatches": r.mismatches,
                            "error": r.error,
                        }
                        for r in results
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {args.output}")

    # Non-zero exit when the agent decided something it had no basis for.
    return 1 if summary["unsupported_decisions"] else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
