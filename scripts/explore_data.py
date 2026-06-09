"""
TrialBridge — Day 3: Data Exploration & Quality Report
Queries the database and produces a full report on data quality,
eligibility criteria patterns, and NLP parsing challenges.
This becomes the blueprint for the Week 2 NLP pipeline.
"""

import asyncio
import re
from pathlib import Path

import asyncpg

DB_URL = "postgresql://trialbridge_user:devpassword@localhost:5432/trialbridge"
REPORT_DIR = Path("reports")


# -----------------------------------------------
# Analysis Functions
# -----------------------------------------------
async def overview(conn) -> str:
    out = []
    out.append("=" * 60)
    out.append("TRIALBRIDGE — DATA QUALITY REPORT")
    out.append("=" * 60)

    total = await conn.fetchval("SELECT COUNT(*) FROM trials")
    out.append(f"\nTotal trials in database: {total:,}")

    # Missing fields analysis
    out.append("\n--- MISSING DATA ANALYSIS ---")
    fields = [
        ("eligibility_criteria_raw", "Eligibility criteria"),
        ("minimum_age", "Minimum age"),
        ("maximum_age", "Maximum age"),
        ("brief_summary", "Brief summary"),
        ("sponsor", "Sponsor"),
        ("enrollment_target", "Enrollment target"),
    ]
    for col, label in fields:
        missing = await conn.fetchval(
            f"SELECT COUNT(*) FROM trials WHERE {col} IS NULL"
        )
        pct = (missing / total * 100) if total else 0
        out.append(f"  {label:30} missing: {missing:,} ({pct:.1f}%)")

    return "\n".join(out)


async def status_phase_breakdown(conn) -> str:
    out = []
    out.append("\n--- BREAKDOWN BY STATUS ---")
    rows = await conn.fetch("""
        SELECT status, COUNT(*) as count,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as pct
        FROM trials GROUP BY status ORDER BY count DESC
    """)
    for row in rows:
        bar = "█" * int(row['pct'] / 2)
        out.append(f"  {row['status']:30} {row['count']:>5,} ({row['pct']:>5}%) {bar}")

    out.append("\n--- BREAKDOWN BY PHASE ---")
    rows = await conn.fetch("""
        SELECT phase, COUNT(*) as count,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as pct
        FROM trials GROUP BY phase ORDER BY count DESC
    """)
    for row in rows:
        bar = "█" * int(row['pct'] / 2)
        out.append(f"  {row['phase']:20} {row['count']:>5,} ({row['pct']:>5}%) {bar}")

    return "\n".join(out)


async def age_analysis(conn) -> str:
    out = []
    out.append("\n--- AGE RANGE ANALYSIS ---")

    stats = await conn.fetchrow("""
        SELECT
            MIN(minimum_age) as min_age,
            MAX(minimum_age) as max_min_age,
            AVG(minimum_age)::int as avg_min_age,
            MIN(maximum_age) as min_max_age,
            MAX(maximum_age) as max_age,
            AVG(maximum_age)::int as avg_max_age
        FROM trials
        WHERE minimum_age IS NOT NULL AND maximum_age IS NOT NULL
    """)

    out.append(f"  Minimum age range : {stats['min_age']} — {stats['max_min_age']} years")
    out.append(f"  Maximum age range : {stats['min_max_age']} — {stats['max_age']} years")
    out.append(f"  Avg minimum age   : {stats['avg_min_age']} years")
    out.append(f"  Avg maximum age   : {stats['avg_max_age']} years")

    # Trials with no upper age limit
    no_max = await conn.fetchval(
        "SELECT COUNT(*) FROM trials WHERE maximum_age IS NULL AND minimum_age IS NOT NULL"
    )
    out.append(f"  Trials with no upper age limit: {no_max:,}")

    # Age distribution buckets
    out.append("\n  Age group distribution (by minimum age):")
    buckets = [
        ("Children (0-17)", 0, 17),
        ("Adults (18-64)", 18, 64),
        ("Seniors (65+)", 65, 200),
    ]
    for label, low, high in buckets:
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM trials WHERE minimum_age >= $1 AND minimum_age <= $2",
            low, high
        )
        out.append(f"    {label:20} {count:,}")

    return "\n".join(out)


async def top_conditions(conn) -> str:
    out = []
    out.append("\n--- TOP 20 CONDITIONS IN DATABASE ---")

    rows = await conn.fetch("""
        SELECT condition, COUNT(*) as count
        FROM trials, jsonb_array_elements_text(conditions) as condition
        GROUP BY condition
        ORDER BY count DESC
        LIMIT 20
    """)

    for i, row in enumerate(rows, 1):
        bar = "█" * min(int(row['count'] / 5), 30)
        out.append(f"  {i:2}. {row['condition'][:40]:40} {row['count']:>4,} {bar}")

    return "\n".join(out)


async def eligibility_patterns(conn) -> str:
    out = []
    out.append("\n--- ELIGIBILITY CRITERIA PATTERN ANALYSIS ---")
    out.append("(These are the patterns the NLP parser must handle in Week 2)\n")

    rows = await conn.fetch("""
        SELECT eligibility_criteria_raw
        FROM trials
        WHERE eligibility_criteria_raw IS NOT NULL
        AND LENGTH(eligibility_criteria_raw) > 100
    """)

    criteria_texts = [row['eligibility_criteria_raw'] for row in rows]
    total = len(criteria_texts)
    out.append(f"  Trials with eligibility criteria: {total:,}")

    # Pattern detection
    patterns = {
        "Contains 'Inclusion Criteria'":
            lambda t: "inclusion criteria" in t.lower(),
        "Contains 'Exclusion Criteria'":
            lambda t: "exclusion criteria" in t.lower(),
        "Contains age reference":
            lambda t: bool(re.search(r'\b\d+\s*(year|month|age)', t.lower())),
        "Contains lab value (numbers with %)":
            lambda t: bool(re.search(r'\d+\.?\d*\s*%', t)),
        "Contains negation (no, not, without, except)":
            lambda t: bool(re.search(r'\b(no |not |without |except )', t.lower())),
        "Contains time constraint (within X months/years)":
            lambda t: bool(re.search(r'within\s+\d+\s*(month|year|week)', t.lower())),
        "Contains prior treatment reference":
            lambda t: bool(re.search(r'\b(prior|previous|history of|received)\b', t.lower())),
        "Contains OR logic":
            lambda t: bool(re.search(r'\bor\b', t.lower())),
        "Contains AND logic":
            lambda t: bool(re.search(r'\band\b', t.lower())),
        "Contains pregnancy exclusion":
            lambda t: bool(re.search(r'\b(pregnan|lactating|nursing)\b', t.lower())),
        "Contains lab test reference (HbA1c, eGFR, ALT etc)":
            lambda t: bool(re.search(r'\b(hba1c|egfr|alt|ast|creatinine|hemoglobin|wbc|platelet)\b', t.lower())),
        "Very long criteria (>2000 chars)":
            lambda t: len(t) > 2000,
    }

    for pattern_name, fn in patterns.items():
        count = sum(1 for t in criteria_texts if fn(t))
        pct = (count / total * 100) if total else 0
        out.append(f"  {pattern_name:50} {count:>4,} ({pct:.1f}%)")

    return "\n".join(out)


async def hardest_criteria(conn) -> str:
    out = []
    out.append("\n--- 10 HARDEST ELIGIBILITY CRITERIA EXAMPLES ---")
    out.append("(Real sentences from your database — this is what Week 2 must parse)\n")

    rows = await conn.fetch("""
        SELECT nct_id, eligibility_criteria_raw
        FROM trials
        WHERE eligibility_criteria_raw IS NOT NULL
        AND eligibility_criteria_raw ILIKE '%within%month%'
        AND eligibility_criteria_raw ILIKE '%not%'
        AND LENGTH(eligibility_criteria_raw) > 500
        ORDER BY LENGTH(eligibility_criteria_raw) DESC
        LIMIT 10
    """)

    for i, row in enumerate(rows, 1):
        # Extract just the most complex sentence
        text = row['eligibility_criteria_raw']
        sentences = [s.strip() for s in re.split(r'[\.\n]', text) if len(s.strip()) > 80]

        # Find sentence with most complexity markers
        def complexity_score(s):
            score = 0
            score += s.lower().count('not ') * 2
            score += s.lower().count('within') * 2
            score += s.lower().count('prior') * 2
            score += s.lower().count(' or ') * 1
            score += s.lower().count(' and ') * 1
            score += len(re.findall(r'\d+', s))
            return score

        if sentences:
            hardest = max(sentences, key=complexity_score)
            out.append(f"  [{i}] {row['nct_id']}")
            out.append(f"      \"{hardest[:300]}\"")
            out.append("")

    return "\n".join(out)


async def nlp_challenge_summary(conn) -> str:
    out = []
    out.append("\n--- NLP PARSING CHALLENGES — WEEK 2 BLUEPRINT ---")
    out.append("""
  Based on the data analysis above, here are the 6 core
  challenges the NLP pipeline must solve:

  1. NEGATION DETECTION
     "No history of X" must NOT flag X as a condition.
     Tool: negspaCy library on top of scispaCy.

  2. TEMPORAL CONSTRAINTS
     "Within 6 months of enrollment" requires date math.
     Pattern: within + number + time_unit.

  3. COMPOUND LOGIC
     Criteria use AND/OR with nested conditions.
     Must parse as a logic tree, not flat list.

  4. LAB VALUE THRESHOLDS
     "HbA1c > 7.5%" requires extracting entity + operator + value.
     Tool: custom regex + scispaCy NER.

  5. PRIOR TREATMENT REFERENCES
     "Prior platinum-based therapy" maps to drug class, not drug name.
     Requires medical ontology (SNOMED-CT) lookup.

  6. CONDITIONAL EXCEPTIONS
     "Unless disease progression was documented..."
     Requires dependency parsing to link exception to main clause.
    """)
    return "\n".join(out)


# -----------------------------------------------
# Main
# -----------------------------------------------
async def main():
    print("Connecting to database...")
    conn = await asyncpg.connect(DB_URL)
    print("Connected. Running analysis...\n")

    # Create reports directory
    REPORT_DIR.mkdir(exist_ok=True)

    # Run all analyses
    sections = [
        await overview(conn),
        await status_phase_breakdown(conn),
        await age_analysis(conn),
        await top_conditions(conn),
        await eligibility_patterns(conn),
        await hardest_criteria(conn),
        await nlp_challenge_summary(conn),
    ]

    full_report = "\n".join(sections)

    # Print to terminal
    print(full_report)

    # Save to file
    report_path = REPORT_DIR / "day3_data_quality_report.txt"
    report_path.write_text(full_report, encoding="utf-8")
    print(f"\nReport saved to: {report_path}")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())