"""
TrialBridge — Eligibility Criteria Parser

Splits a trial's free-text eligibility criteria into discrete, individually
decidable criteria.

The matching engine in backend/matching/matcher.py never actually reads this
text — it substring-matches a handful of lab names against it and defers
everything else to a human. This module is the first half of fixing that: turn
one wall of prose into a list of atomic claims that can each be decided,
one at a time.

Parsing is deterministic. ClinicalTrials.gov's bulk XML export puts criteria in
`eligibility/criteria/textblock` with a consistent shape:

        Inclusion Criteria:

          -  Age 18 years or older

          -  Histologically confirmed Stage II-IV disease

        Exclusion Criteria:

          -  Prior systemic chemotherapy

An LLM is not needed to split that, and using one would mean an API call per
trial per query. The corpus is 2,344 trials; deterministic parsing keeps this
free and testable.
"""

import re
from dataclasses import dataclass

# -----------------------------------------------
# Criterion Model
# -----------------------------------------------
INCLUSION = "inclusion"
EXCLUSION = "exclusion"
UNSPECIFIED = "unspecified"


@dataclass
class Criterion:
    """
    A single eligibility claim that can be decided independently.

    `kind` is UNSPECIFIED when the source text carried no inclusion/exclusion
    header. That distinction matters downstream: an unmet inclusion criterion
    disqualifies a patient, but an unmet criterion of unknown polarity only
    means we don't know. The verdict aggregator must not treat the two alike.
    """

    id: str
    kind: str
    text: str

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "text": self.text}


# -----------------------------------------------
# Section + Bullet Patterns
# -----------------------------------------------
# "Inclusion Criteria:", "EXCLUSION CRITERIA", "Key Inclusion Criteria -", etc.
_SECTION_RE = re.compile(
    r"^[ \t]*(?:key[ \t]+)?(?P<kind>inclusion|exclusion)[ \t]+criteria\b[ \t]*[:\-–]?[ \t]*",
    re.IGNORECASE | re.MULTILINE,
)

# Bullets: "-", "*", "•", "1.", "1)", "(1)", "a."
_BULLET_RE = re.compile(
    r"^[ \t]*(?:[-*•·]|\(?\d{1,2}[.)]|\(?[a-z][.)])[ \t]+",
    re.IGNORECASE,
)

# Collapses runs of whitespace inside a single criterion.
_WS_RE = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", text).strip(" \t-–—:;")


def _split_bullets(segment: str) -> list[str]:
    """
    Split one section into individual criteria.

    Bullet markers are the primary signal. A criterion frequently wraps across
    several indented lines, so a line without a marker continues the previous
    bullet rather than starting a new one.
    """
    items: list[str] = []
    current: list[str] = []

    for line in segment.splitlines():
        if not line.strip():
            continue
        if _BULLET_RE.match(line):
            if current:
                items.append(" ".join(current))
            current = [_BULLET_RE.sub("", line, count=1).strip()]
        elif current:
            current.append(line.strip())
        else:
            # Text before any bullet marker in this section.
            current = [line.strip()]

    if current:
        items.append(" ".join(current))

    # No bullet markers at all — fall back to sentence-ish splitting so a
    # prose-style criteria block still produces decidable units.
    if len(items) <= 1 and not _BULLET_RE.search(segment):
        blob = items[0] if items else segment
        items = re.split(r"(?<=[.;])\s+(?=[A-Z(])", blob)

    return [c for c in (_clean(i) for i in items) if len(c) > 3]


def parse_eligibility_criteria(raw: str | None) -> list[Criterion]:
    """
    Parse raw eligibility text into discrete criteria.

    Returns an empty list for missing or unparseable text. Callers must treat
    "no criteria parsed" as "cannot assess", never as "patient qualifies" —
    an empty list means we failed to read the trial, not that it has no rules.
    """
    if not raw or not raw.strip():
        return []

    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Locate section headers so each bullet inherits the right polarity.
    headers = [(m.start(), m.end(), m.group("kind").lower()) for m in _SECTION_RE.finditer(text)]

    segments: list[tuple[str, str]] = []
    if not headers:
        segments.append((UNSPECIFIED, text))
    else:
        # Any preamble before the first header has no declared polarity.
        preamble = text[: headers[0][0]].strip()
        if preamble:
            segments.append((UNSPECIFIED, preamble))
        for idx, (_, body_start, kind) in enumerate(headers):
            body_end = headers[idx + 1][0] if idx + 1 < len(headers) else len(text)
            segments.append((kind, text[body_start:body_end]))

    criteria: list[Criterion] = []
    counters = {INCLUSION: 0, EXCLUSION: 0, UNSPECIFIED: 0}
    prefixes = {INCLUSION: "INC", EXCLUSION: "EXC", UNSPECIFIED: "UNK"}

    for kind, segment in segments:
        for item in _split_bullets(segment):
            counters[kind] += 1
            criteria.append(
                Criterion(id=f"{prefixes[kind]}-{counters[kind]}", kind=kind, text=item)
            )

    return criteria
