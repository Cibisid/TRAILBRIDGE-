"""
TrialBridge — Eligibility Agent

Decides a patient against one trial's criteria, one criterion at a time.

The rules engine in backend/matching/matcher.py scores a trial from structured
metadata (age range, gender, condition list) and substring-matches a couple of
lab names against the criteria text without ever comparing a value to a
threshold. This agent reads the criteria.

The design constraint that makes the output trustworthy is in
backend/agent/patient_store.py: the agent is never given the patient note. It
has to request each fact through a tool, and the store answers NOT_PRESENT when
the note doesn't contain one. So a criterion that hinges on an unrecorded lab
cannot be quietly resolved from a guessed value — the missing data is the tool's
literal response, and UNKNOWN is the only verdict available.

Runs on Claude through either the Anthropic API or Amazon Bedrock — set
`llm_provider`. Both expose the same messages surface, so the loop below is
identical either way.

Thinking is deliberately left on: with thinking disabled, this model can emit a
tool call as plain text, which in a loop like this one would look like a
completed turn where nothing actually ran.
"""

import asyncio
from dataclasses import dataclass, field

from anthropic import AsyncAnthropic, AsyncAnthropicBedrockMantle

from backend.agent.criteria import EXCLUSION, INCLUSION, Criterion
from backend.agent.patient_store import PatientDataStore
from backend.core.config import get_settings
from backend.core.logging import get_logger

logger = get_logger(__name__)

# -----------------------------------------------
# Verdicts
# -----------------------------------------------
MET = "MET"
NOT_MET = "NOT_MET"
UNKNOWN = "UNKNOWN"

QUALIFIES = "QUALIFIES"
EXCLUDED = "EXCLUDED"
INDETERMINATE = "UNKNOWN"

_VALID_VERDICTS = (MET, NOT_MET, UNKNOWN)


@dataclass
class CriterionVerdict:
    criterion_id: str
    kind: str
    text: str
    verdict: str
    evidence: str
    missing_data: str | None = None

    def to_dict(self) -> dict:
        return {
            "criterion_id": self.criterion_id,
            "kind": self.kind,
            "text": self.text,
            "verdict": self.verdict,
            "evidence": self.evidence,
            "missing_data": self.missing_data,
        }


@dataclass
class TrialAssessment:
    nct_id: str
    status: str
    verdicts: list[CriterionVerdict] = field(default_factory=list)
    missing_data: list[str] = field(default_factory=list)
    blocking_criterion: str | None = None
    audit: list[dict] = field(default_factory=list)
    iterations: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "nct_id": self.nct_id,
            "status": self.status,
            "verdicts": [v.to_dict() for v in self.verdicts],
            "missing_data": self.missing_data,
            "blocking_criterion": self.blocking_criterion,
            "audit": self.audit,
            "iterations": self.iterations,
            "error": self.error,
        }


# -----------------------------------------------
# Tools
# -----------------------------------------------
# strict=True guarantees the input validates against the schema, so the
# dispatcher below never has to defend against a malformed argument object.
def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


TOOLS = [
    _tool(
        "lookup_patient_field",
        "Read one structured field from the patient record. Returns PRESENT with the "
        "value, or NOT_PRESENT when the note does not state it.",
        {
            "field": {
                "type": "string",
                "enum": [
                    "age",
                    "sex",
                    "primary_diagnosis",
                    "ecog_score",
                    "comorbidities",
                    "negated_conditions",
                    "current_medications",
                    "prior_treatments",
                    "allergies",
                ],
                "description": "Which field to read.",
            }
        },
        ["field"],
    ),
    _tool(
        "lookup_lab_value",
        "Read a numeric lab value by name (e.g. HbA1c, eGFR, creatinine, platelets). "
        "Returns PRESENT with the value, or NOT_PRESENT. Values carry no draw date.",
        {"lab_name": {"type": "string", "description": "Name of the lab test."}},
        ["lab_name"],
    ),
    _tool(
        "check_condition",
        "Check whether the patient has a condition. Returns CONFIRMED_PRESENT, "
        "EXPLICITLY_ABSENT (the note rules it out), or NOT_MENTIONED (the note is "
        "silent — which is not evidence of absence).",
        {"condition": {"type": "string", "description": "Condition to check for."}},
        ["condition"],
    ),
    _tool(
        "check_medication",
        "Check the patient's relationship to a drug or therapy. Returns "
        "CURRENTLY_TAKING, PRIOR_TREATMENT, EXPLICITLY_ABSENT, or NOT_MENTIONED.",
        {"medication": {"type": "string", "description": "Drug or therapy name."}},
        ["medication"],
    ),
    _tool(
        "record_verdict",
        "Record the decision for one criterion. Call once per criterion.",
        {
            "criterion_id": {"type": "string", "description": "The criterion's ID, e.g. INC-3."},
            "verdict": {
                "type": "string",
                "enum": [MET, NOT_MET, UNKNOWN],
                "description": (
                    "MET if the patient satisfies this criterion, NOT_MET if they do not, "
                    "UNKNOWN if the record lacks a fact needed to decide."
                ),
            },
            "evidence": {
                "type": "string",
                "description": (
                    "One sentence citing the specific tool results behind this verdict."
                ),
            },
            "missing_data": {
                "type": "string",
                "description": (
                    "Required when verdict is UNKNOWN: the exact data point needed, phrased "
                    "so a coordinator knows what to go find. Empty string otherwise."
                ),
            },
        },
        ["criterion_id", "verdict", "evidence", "missing_data"],
    ),
]

SYSTEM_PROMPT = """\
You are adjudicating a patient against one clinical trial's eligibility criteria, \
the way a research coordinator would: one criterion at a time, against the record.

You do not have the patient's note. Every fact must come from a tool call. This is \
deliberate — it means you cannot supply a value the record does not contain.

Rules:

1. Decide each criterion separately and call record_verdict once for each. Work \
through every criterion you are given.
2. If a fact you need comes back NOT_PRESENT or NOT_MENTIONED, the verdict is \
UNKNOWN. Name the missing data point in missing_data. Do not estimate it, infer it \
from a related value, or treat a typical value as the patient's.
3. NOT_MENTIONED means the note is silent, not that the patient is negative. Only \
EXPLICITLY_ABSENT establishes that a condition has been ruled out.
4. When a criterion requires a measurement within a time window, note that lab \
values here carry no draw date. The value alone cannot settle it — that is UNKNOWN \
with the date as the missing data.
5. Compare numbers directly against the thresholds written in the criterion.
6. A criterion whose meaning you cannot determine is UNKNOWN, not MET.

Verdicts are read by clinicians. An honest UNKNOWN is useful to them; a confident \
wrong answer is not."""


# -----------------------------------------------
# Agent
# -----------------------------------------------
class EligibilityAgent:
    """
    Runs one tool-use loop per trial.

    Batching a trial's criteria into a single loop keeps the request count at
    one per candidate trial rather than one per criterion, while still letting
    the model gather facts incrementally as it works down the list.
    """

    def __init__(
        self,
        model: str | None = None,
        region: str | None = None,
        effort: str | None = None,
        max_iterations: int | None = None,
        provider: str | None = None,
    ):
        settings = get_settings()
        self.provider = provider or settings.llm_provider
        self.model = model or (
            settings.bedrock_model_id if self.provider == "bedrock" else settings.anthropic_model_id
        )
        self.region = region or settings.aws_region
        self.effort = effort or settings.bedrock_effort
        self.max_iterations = max_iterations or settings.agent_max_iterations
        self.max_tokens = settings.bedrock_max_tokens
        self._client: AsyncAnthropic | AsyncAnthropicBedrockMantle | None = None

    @property
    def client(self) -> AsyncAnthropic | AsyncAnthropicBedrockMantle:
        """
        Built on first use, not at import, so importing this module never
        requires credentials — which is what lets the unit tests run in CI
        with no key present.

        Both clients expose the same messages.create surface, so nothing below
        this property knows or cares which one it got.
        """
        if self._client is None:
            if self.provider == "bedrock":
                self._client = AsyncAnthropicBedrockMantle(aws_region=self.region)
            else:
                # Reads ANTHROPIC_API_KEY from the environment.
                self._client = AsyncAnthropic()
        return self._client

    # -------------------------------------------
    # Tool dispatch
    # -------------------------------------------
    def _dispatch(self, name: str, args: dict, store: PatientDataStore) -> dict:
        if name == "lookup_patient_field":
            return store.lookup_field(args["field"]).to_dict()
        if name == "lookup_lab_value":
            return store.lookup_lab(args["lab_name"]).to_dict()
        if name == "check_condition":
            return store.check_condition(args["condition"]).to_dict()
        if name == "check_medication":
            return store.check_medication(args["medication"]).to_dict()
        return {"error": f"Unknown tool '{name}'."}

    # -------------------------------------------
    # Assessment
    # -------------------------------------------
    async def assess_trial(
        self,
        nct_id: str,
        criteria: list[Criterion],
        store: PatientDataStore,
    ) -> TrialAssessment:
        """Decide every criterion for one trial and aggregate to a trial-level status."""
        if not criteria:
            # No parsed criteria means we could not read the trial's rules. That
            # is an unknown, never a pass.
            return TrialAssessment(
                nct_id=nct_id,
                status=INDETERMINATE,
                missing_data=["This trial's eligibility criteria could not be parsed."],
                error="no_criteria",
            )

        criteria_by_id = {c.id: c for c in criteria}
        rendered = "\n".join(f"[{c.id}] ({c.kind}) {c.text}" for c in criteria)
        messages: list[dict] = [
            {
                "role": "user",
                "content": (
                    f"Trial {nct_id}. Decide each criterion below.\n\n{rendered}\n\n"
                    f"Use the tools to gather what you need, then record a verdict for "
                    f"all {len(criteria)} criteria."
                ),
            }
        ]

        recorded: dict[str, CriterionVerdict] = {}
        iterations = 0

        while iterations < self.max_iterations:
            iterations += 1
            try:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                    tools=TOOLS,
                    output_config={"effort": self.effort},
                )
            except Exception as exc:  # noqa: BLE001 - surfaced to the caller as a status
                logger.error("agent_request_failed", nct_id=nct_id, error=str(exc))
                return TrialAssessment(
                    nct_id=nct_id,
                    status=INDETERMINATE,
                    verdicts=list(recorded.values()),
                    missing_data=["The eligibility agent could not complete this assessment."],
                    audit=store.audit(),
                    iterations=iterations,
                    error=str(exc),
                )

            # Check the stop reason before touching content: a refusal returns
            # HTTP 200 with empty or partial content.
            if response.stop_reason == "refusal":
                logger.warning("agent_refusal", nct_id=nct_id)
                return TrialAssessment(
                    nct_id=nct_id,
                    status=INDETERMINATE,
                    verdicts=list(recorded.values()),
                    missing_data=["This trial's criteria could not be assessed automatically."],
                    audit=store.audit(),
                    iterations=iterations,
                    error="refusal",
                )

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                break

            messages.append({"role": "assistant", "content": response.content})

            results = []
            for block in tool_uses:
                args = block.input or {}
                if block.name == "record_verdict":
                    verdict = self._record(block, args, criteria_by_id, recorded)
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": verdict,
                        }
                    )
                else:
                    payload = self._dispatch(block.name, args, store)
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(payload),
                        }
                    )

            # All results for one assistant turn go back in a single user
            # message; splitting them suppresses future parallel tool calls.
            messages.append({"role": "user", "content": results})

            if len(recorded) >= len(criteria):
                break

        # Criteria the model never got to are unknown, not satisfied.
        for criterion in criteria:
            if criterion.id not in recorded:
                recorded[criterion.id] = CriterionVerdict(
                    criterion_id=criterion.id,
                    kind=criterion.kind,
                    text=criterion.text,
                    verdict=UNKNOWN,
                    evidence="The agent did not reach a decision on this criterion.",
                    missing_data="Not assessed.",
                )

        ordered = [recorded[c.id] for c in criteria]
        status, missing, blocking = self._aggregate(ordered)

        return TrialAssessment(
            nct_id=nct_id,
            status=status,
            verdicts=ordered,
            missing_data=missing,
            blocking_criterion=blocking,
            audit=store.audit(),
            iterations=iterations,
        )

    def _record(
        self,
        block,
        args: dict,
        criteria_by_id: dict[str, Criterion],
        recorded: dict[str, CriterionVerdict],
    ) -> str:
        criterion_id = (args.get("criterion_id") or "").strip()
        criterion = criteria_by_id.get(criterion_id)
        if criterion is None:
            return f"No criterion with id '{criterion_id}'. Valid ids: {list(criteria_by_id)}"

        verdict = args.get("verdict")
        if verdict not in _VALID_VERDICTS:
            return f"Invalid verdict '{verdict}'. Use one of {list(_VALID_VERDICTS)}."

        missing = (args.get("missing_data") or "").strip() or None
        if verdict == UNKNOWN and not missing:
            return "An UNKNOWN verdict must name the missing data point in missing_data."

        recorded[criterion_id] = CriterionVerdict(
            criterion_id=criterion_id,
            kind=criterion.kind,
            text=criterion.text,
            verdict=verdict,
            evidence=(args.get("evidence") or "").strip(),
            missing_data=missing if verdict == UNKNOWN else None,
        )
        return f"Recorded {criterion_id}: {verdict}."

    @staticmethod
    def _aggregate(verdicts: list[CriterionVerdict]) -> tuple[str, list[str], str | None]:
        """
        Roll per-criterion verdicts into a trial-level status.

        Ordering matters. A disqualifier settles the trial regardless of what
        else is unknown, so exclusions are checked first. Anything still
        undecided after that leaves the trial UNKNOWN — never QUALIFIES.
        """
        for v in verdicts:
            if v.kind == EXCLUSION and v.verdict == MET:
                return EXCLUDED, [], v.criterion_id
            if v.kind == INCLUSION and v.verdict == NOT_MET:
                return EXCLUDED, [], v.criterion_id

        missing = [v.missing_data for v in verdicts if v.verdict == UNKNOWN and v.missing_data]
        if missing:
            return INDETERMINATE, missing, None

        # An unmet criterion whose polarity the source text never declared
        # can't be read as a disqualifier, but it isn't a pass either.
        unspecified = [
            v for v in verdicts if v.verdict == NOT_MET and v.kind not in (INCLUSION, EXCLUSION)
        ]
        if unspecified:
            return (
                INDETERMINATE,
                [
                    f"Criterion {v.criterion_id} is unmet, but the trial text does not say "
                    f"whether it is an inclusion or exclusion rule."
                    for v in unspecified
                ],
                None,
            )

        return QUALIFIES, [], None

    async def assess_trials(
        self,
        trials: list[tuple[str, list[Criterion]]],
        build_store,
    ) -> list[TrialAssessment]:
        """
        Assess several trials concurrently.

        Each trial gets its own PatientDataStore so the audit trails stay
        separate — the evidence behind one trial's verdict shouldn't include
        lookups made while working on another.
        """
        tasks = [self.assess_trial(nct_id, criteria, build_store()) for nct_id, criteria in trials]
        return await asyncio.gather(*tasks)
