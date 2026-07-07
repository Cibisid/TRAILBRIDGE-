"""
TrialBridge — Day 11: Claude API Explanation Engine

Takes a patient profile + trial match and generates a plain English
explanation that a doctor can actually act on.

This is what separates TrialBridge from a data tool.
A score of 0.673 means nothing to a clinician.
"This trial matches because..." is actionable.
"""

import os
import anthropic
import asyncio
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()

from backend.nlp.extractor_v2 import PatientProfile
from backend.matching.matcher import TrialMatch


# -----------------------------------------------
# Explanation Model
# -----------------------------------------------
@dataclass
class TrialExplanation:
    nct_id: str
    eligibility_status: str
    summary: str           # 2-3 sentence plain English explanation
    why_matches: list[str] # Specific reasons this trial fits
    flags: list[str]       # What to verify before enrolling
    recommendation: str    # One-line actionable recommendation


# -----------------------------------------------
# Explainer
# -----------------------------------------------
class MatchExplainer:
    """
    Uses Claude API to generate clinical explanations for trial matches.
    Falls back to rule-based explanations if API key is not set.
    """

    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self._has_api_key = bool(api_key and api_key != "your-anthropic-api-key")
        if self._has_api_key:
            self._client = anthropic.Anthropic(api_key=api_key)

    def explain(
        self,
        patient: PatientProfile,
        match: TrialMatch,
    ) -> TrialExplanation:
        """
        Generate explanation for a single trial match.
        Uses Claude API if available, falls back to rule-based.
        """
        if self._has_api_key:
            return self._explain_with_claude(patient, match)
        else:
            return self._explain_rule_based(patient, match)

    def explain_batch(
        self,
        patient: PatientProfile,
        matches: list[TrialMatch],
        max_explain: int = 5,
    ) -> list[TrialExplanation]:
        """Generate explanations for top N matches."""
        explanations = []
        for match in matches[:max_explain]:
            explanation = self.explain(patient, match)
            explanations.append(explanation)
        return explanations

    def _explain_with_claude(
        self,
        patient: PatientProfile,
        match: TrialMatch,
    ) -> TrialExplanation:
        """Call Claude API to generate clinical explanation."""

        prompt = f"""You are a clinical research coordinator reviewing a clinical trial match for a patient.

PATIENT PROFILE:
- Age: {patient.age}
- Sex: {patient.sex}
- Primary Diagnosis: {patient.primary_diagnosis}
- Comorbidities: {', '.join(patient.comorbidities) if patient.comorbidities else 'None'}
- Current Medications: {', '.join(patient.current_medications) if patient.current_medications else 'None'}
- Prior Treatments: {', '.join(patient.prior_treatments) if patient.prior_treatments else 'None'}
- Lab Values: {patient.lab_values if patient.lab_values else 'Not provided'}
- Conditions Patient Does NOT Have: {', '.join(patient.negated_conditions) if patient.negated_conditions else 'None'}

MATCHED TRIAL:
- NCT ID: {match.nct_id}
- Title: {match.title}
- Status: {match.status}
- Phase: {match.phase}
- Sponsor: {match.sponsor}
- Conditions Studied: {', '.join(match.conditions) if match.conditions else 'Not specified'}
- Eligibility Status: {match.eligibility_status}
- Inclusion Criteria Met: {', '.join(match.inclusion_met) if match.inclusion_met else 'None confirmed'}
- Exclusion Flags: {', '.join(match.exclusion_flags) if match.exclusion_flags else 'None'}
- Needs Verification: {', '.join(match.needs_verification) if match.needs_verification else 'None'}
- Match Scores: Semantic {match.semantic_score:.2f}, Eligibility {match.eligibility_score:.2f}, Composite {match.composite_score:.2f}

Write a clinical explanation in this exact JSON format:
{{
  "summary": "2-3 sentence plain English explanation of why this trial matches or doesn't match this patient",
  "why_matches": ["specific reason 1", "specific reason 2"],
  "flags": ["what to verify 1", "what to verify 2"],
  "recommendation": "one sentence actionable recommendation for the clinician"
}}

Be specific, clinical, and actionable. If the trial is INELIGIBLE, explain exactly why.
Return only valid JSON, no other text."""

        try:
            response = self._client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )

            import json
            content = response.content[0].text.strip()
            # Clean up any markdown code blocks
            content = content.replace("```json", "").replace("```", "").strip()
            data = json.loads(content)

            return TrialExplanation(
                nct_id=match.nct_id,
                eligibility_status=match.eligibility_status,
                summary=data.get("summary", ""),
                why_matches=data.get("why_matches", []),
                flags=data.get("flags", []),
                recommendation=data.get("recommendation", ""),
            )

        except Exception as e:
            # Fall back to rule-based if Claude call fails
            return self._explain_rule_based(patient, match)

    def _explain_rule_based(
        self,
        patient: PatientProfile,
        match: TrialMatch,
    ) -> TrialExplanation:
        """
        Rule-based explanation when Claude API is not available.
        Still useful and readable — just less nuanced.
        """
        why_matches = []
        flags = []

        # Build why_matches from inclusion criteria
        for criterion in match.inclusion_met:
            why_matches.append(criterion)

        # Add semantic relevance
        if match.semantic_score > 0.5:
            why_matches.append(
                f"Trial topic is highly relevant to patient's condition "
                f"(semantic similarity: {match.semantic_score:.0%})"
            )

        # Build flags from needs_verification
        for flag in match.needs_verification:
            flags.append(flag)

        # Build summary
        if match.eligibility_status == "ELIGIBLE":
            summary = (
                f"This {match.phase} trial by {match.sponsor or 'an unknown sponsor'} "
                f"appears to be a strong match for this patient. "
                f"The patient meets {len(match.inclusion_met)} confirmed eligibility criteria."
            )
            recommendation = (
                f"Review full eligibility criteria for {match.nct_id} "
                f"and consider enrolling if {len(flags)} remaining items are confirmed."
            ) if flags else (
                f"Strong candidate for enrollment — review full protocol for {match.nct_id}."
            )

        elif match.eligibility_status == "LIKELY_ELIGIBLE":
            summary = (
                f"This trial is a probable match but requires verification of "
                f"{len(flags)} item(s) before confirming eligibility. "
                f"The patient meets basic demographic and diagnostic criteria."
            )
            recommendation = (
                f"Verify {flags[0].split('—')[0].strip() if flags else 'eligibility details'} "
                f"before considering enrollment in {match.nct_id}."
            )

        elif match.eligibility_status == "INELIGIBLE":
            exclusion_reason = (
                match.exclusion_flags[0] if match.exclusion_flags
                else "one or more eligibility criteria"
            )
            summary = (
                f"This patient does not meet eligibility requirements for this trial. "
                f"Disqualifying factor: {exclusion_reason}."
            )
            recommendation = f"Do not enroll — patient is ineligible for {match.nct_id}."

        else:
            summary = (
                f"This trial requires manual review to determine eligibility. "
                f"Insufficient information to make an automated determination."
            )
            recommendation = f"Manual eligibility review required for {match.nct_id}."

        return TrialExplanation(
            nct_id=match.nct_id,
            eligibility_status=match.eligibility_status,
            summary=summary,
            why_matches=why_matches,
            flags=flags,
            recommendation=recommendation,
        )


# -----------------------------------------------
# Test
# -----------------------------------------------
async def run_test():
    import asyncpg
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    DB_URL = "postgresql://trialbridge_user:devpassword@localhost:5432/trialbridge"

    print("=" * 65)
    print("TrialBridge — Day 11: Explanation Engine Test")
    print("=" * 65)

    conn = await asyncpg.connect(DB_URL)
    from backend.matching.matcher import TrialMatchingEngine
    engine = TrialMatchingEngine()
    explainer = MatchExplainer()

    note = """
        58-year-old female with Type 2 Diabetes Mellitus.
        Currently on Metformin 1000mg. HbA1c 8.9%, eGFR 72.
        No prior insulin therapy. No history of cardiovascular disease.
    """

    print("\nRunning matching pipeline...")
    profile, matches = await engine.match_from_note(note, conn, n_results=3)

    print(f"Patient: {profile.age}yo {profile.sex}, {profile.primary_diagnosis}")
    print(f"Found {len(matches)} matches\n")

    api_status = "Claude API" if explainer._has_api_key else "Rule-based fallback"
    print(f"Explanation engine: {api_status}\n")

    explanations = explainer.explain_batch(profile, matches, max_explain=3)

    for i, (match, explanation) in enumerate(zip(matches, explanations), 1):
        print(f"{'─' * 65}")
        print(f"[{i}] {match.nct_id} — {explanation.eligibility_status}")
        print(f"    Title: {match.title[:60]}")
        print(f"    Composite score: {match.composite_score:.3f}")
        print(f"\n    EXPLANATION:")
        print(f"    {explanation.summary}")
        if explanation.why_matches:
            print(f"\n    WHY IT MATCHES:")
            for reason in explanation.why_matches:
                print(f"    ✅ {reason}")
        if explanation.flags:
            print(f"\n    VERIFY BEFORE ENROLLING:")
            for flag in explanation.flags:
                print(f"    ⚠️  {flag}")
        print(f"\n    RECOMMENDATION:")
        print(f"    → {explanation.recommendation}")

    await conn.close()
    print(f"\n{'=' * 65}")
    print("Day 11 complete. Explanation engine working.")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(run_test())