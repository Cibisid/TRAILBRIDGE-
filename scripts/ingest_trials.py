"""
TrialBridge — Day 2: Bulk XML Ingestion Pipeline
Reads 400,000+ clinical trials from local XML files
and loads them into PostgreSQL. This is the same data
source used by hospitals and research institutions.
"""

import asyncio
import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import asyncpg

# -----------------------------------------------
# Config
# -----------------------------------------------
DB_URL = "postgresql://trialbridge_user:devpassword@localhost:5432/trialbridge"
DATA_DIR = Path("scripts/data")
MAX_TRIALS = 15000  # Load 15,000 trials today
BATCH_SIZE = 500    # Insert 500 at a time for speed


# -----------------------------------------------
# XML Parser
# -----------------------------------------------
def get_text(root, path: str) -> str | None:
    """Safely get text from an XML element."""
    el = root.find(path)
    return el.text.strip() if el is not None and el.text else None


def get_all_text(root, path: str) -> list[str]:
    """Get text from all matching XML elements."""
    return [
        el.text.strip()
        for el in root.findall(path)
        if el.text and el.text.strip()
    ]


def parse_age(age_str: str | None) -> int | None:
    """Convert '18 Years', '6 Months' etc to integer years."""
    if not age_str:
        return None
    age_str = age_str.strip().lower()
    try:
        if "year" in age_str:
            return int(age_str.split()[0])
        elif "month" in age_str:
            return max(0, int(age_str.split()[0]) // 12)
        elif "week" in age_str:
            return 0
        else:
            num = ''.join(filter(str.isdigit, age_str.split()[0]))
            return int(num) if num else None
    except (ValueError, IndexError):
        return None


def parse_date(date_str: str | None) -> str | None:
    """Parse date strings into ISO format."""
    if not date_str:
        return None
    for fmt in ["%B %d, %Y", "%B %Y", "%Y-%m-%d", "%Y"]:
        try:
            return datetime.strptime(date_str.strip(), fmt).isoformat()
        except ValueError:
            continue
    return None


def normalize_status(status: str | None) -> str:
    if not status:
        return "UNKNOWN"
    mapping = {
        "Recruiting": "RECRUITING",
        "Not yet recruiting": "NOT_YET_RECRUITING",
        "Active, not recruiting": "ACTIVE_NOT_RECRUITING",
        "Completed": "COMPLETED",
        "Suspended": "SUSPENDED",
        "Terminated": "TERMINATED",
        "Withdrawn": "WITHDRAWN",
    }
    return mapping.get(status.strip(), "UNKNOWN")


def normalize_phase(phase: str | None) -> str:
    if not phase:
        return "UNKNOWN"
    mapping = {
        "Phase 1": "PHASE1",
        "Phase 2": "PHASE2",
        "Phase 3": "PHASE3",
        "Phase 4": "PHASE4",
        "Early Phase 1": "EARLY_PHASE1",
        "Phase 1/Phase 2": "PHASE1",
        "Phase 2/Phase 3": "PHASE2",
        "N/A": "NA",
    }
    return mapping.get(phase.strip(), "UNKNOWN")


def parse_xml_file(filepath: Path) -> dict | None:
    """
    Parse a single ClinicalTrials.gov XML file into a clean dict.
    Returns None if the file is malformed or missing critical fields.
    """
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        nct_id = get_text(root, "id_info/nct_id")
        title = get_text(root, "brief_title")

        if not nct_id or not title:
            return None

        # Eligibility criteria
        eligibility_text = get_text(root, "eligibility/criteria/textblock")

        # Locations
        locations = []
        for facility in root.findall(".//location/facility"):
            loc = {
                "name": get_text(facility, "name"),
                "city": get_text(facility, "address/city"),
                "state": get_text(facility, "address/state"),
                "country": get_text(facility, "address/country"),
            }
            if any(loc.values()):
                locations.append(loc)

        # Interventions
        interventions = get_all_text(root, ".//intervention/intervention_name")

        return {
            "nct_id": nct_id,
            "title": title,
            "brief_summary": get_text(root, "brief_summary/textblock"),
            "detailed_description": get_text(root, "detailed_description/textblock"),
            "status": normalize_status(get_text(root, "overall_status")),
            "phase": normalize_phase(get_text(root, "phase")),
            "eligibility_criteria_raw": eligibility_text,
            "minimum_age": parse_age(get_text(root, "eligibility/minimum_age")),
            "maximum_age": parse_age(get_text(root, "eligibility/maximum_age")),
            "gender": get_text(root, "eligibility/gender") or "All",
            "accepts_healthy_volunteers": (
                get_text(root, "eligibility/healthy_volunteers") == "Accepts Healthy Volunteers"
            ),
            "conditions": get_all_text(root, "condition"),
            "interventions": interventions[:10],
            "sponsor": get_text(root, "sponsors/lead_sponsor/agency"),
            "locations": locations[:10],
            "start_date": parse_date(get_text(root, "start_date")),
            "completion_date": parse_date(get_text(root, "completion_date")),
            "enrollment_target": (
                int(t) if (t := get_text(root, "enrollment")) and t.isdigit() else None
            ),
        }

    except ET.ParseError:
        return None
    except Exception as e:
        return None


# -----------------------------------------------
# Database Setup
# -----------------------------------------------
async def setup_database(conn) -> None:
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS trials (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            nct_id VARCHAR(20) UNIQUE NOT NULL,
            title TEXT NOT NULL,
            brief_summary TEXT,
            detailed_description TEXT,
            status VARCHAR(50),
            phase VARCHAR(50),
            eligibility_criteria_raw TEXT,
            minimum_age INTEGER,
            maximum_age INTEGER,
            gender VARCHAR(20),
            accepts_healthy_volunteers BOOLEAN DEFAULT FALSE,
            conditions JSONB,
            interventions JSONB,
            sponsor VARCHAR(500),
            locations JSONB,
            start_date TIMESTAMP,
            completion_date TIMESTAMP,
            enrollment_target INTEGER,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )
    """)
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_trials_nct_id ON trials(nct_id)"
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_trials_status ON trials(status)"
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_trials_phase ON trials(phase)"
    )
    print("Database ready")


# -----------------------------------------------
# Batch Insert
# -----------------------------------------------
async def insert_batch(conn, batch: list[dict]) -> int:
    """Insert a batch of trials using UPSERT."""
    inserted = 0
    for trial in batch:
        try:
            await conn.execute("""
                INSERT INTO trials (
                    nct_id, title, brief_summary, detailed_description,
                    status, phase, eligibility_criteria_raw,
                    minimum_age, maximum_age, gender,
                    accepts_healthy_volunteers, conditions, interventions,
                    sponsor, locations, start_date, completion_date,
                    enrollment_target
                ) VALUES (
                    $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,
                    $11,$12,$13,$14,$15,$16,$17,$18
                )
                ON CONFLICT (nct_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    updated_at = NOW()
            """,
                trial["nct_id"],
                trial["title"],
                trial["brief_summary"],
                trial["detailed_description"],
                trial["status"],
                trial["phase"],
                trial["eligibility_criteria_raw"],
                trial["minimum_age"],
                trial["maximum_age"],
                trial["gender"],
                trial["accepts_healthy_volunteers"],
                json.dumps(trial["conditions"]),
                json.dumps(trial["interventions"]),
                trial["sponsor"],
                json.dumps(trial["locations"]),
                trial["start_date"],
                trial["completion_date"],
                trial["enrollment_target"],
            )
            inserted += 1
        except Exception as e:
            pass
    return inserted


# -----------------------------------------------
# Main Pipeline
# -----------------------------------------------
async def main():
    print("=" * 55)
    print("TrialBridge — Bulk XML Ingestion Pipeline")
    print("=" * 55)

    # Connect to database
    print("\nConnecting to database...")
    try:
        conn = await asyncpg.connect(DB_URL)
        print("Connected")
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Run: docker compose up postgres -d")
        return

    await setup_database(conn)

    # Walk all subdirectories and collect XML files
    print(f"\nScanning {DATA_DIR} for XML files...")
    xml_files = sorted(DATA_DIR.rglob("*.xml"))
    total_files = len(xml_files)
    print(f"Found {total_files:,} XML files")
    print(f"Loading first {MAX_TRIALS:,} trials\n")

    total_inserted = 0
    total_skipped = 0
    batch = []
    files_processed = 0

    for xml_file in xml_files:
        if total_inserted >= MAX_TRIALS:
            break

        trial = parse_xml_file(xml_file)
        files_processed += 1

        if trial:
            batch.append(trial)
        else:
            total_skipped += 1

        # Insert in batches
        if len(batch) >= BATCH_SIZE:
            inserted = await insert_batch(conn, batch)
            total_inserted += inserted
            batch = []
            print(
                f"  Progress: {total_inserted:,} trials loaded"
                f" | {files_processed:,} files processed"
                f" | {total_skipped} skipped"
            )

    # Insert remaining batch
    if batch:
        inserted = await insert_batch(conn, batch)
        total_inserted += inserted

    # Final stats
    total_in_db = await conn.fetchval("SELECT COUNT(*) FROM trials")

    print("\n" + "=" * 55)
    print(f"Ingestion complete!")
    print(f"Trials loaded this run : {total_inserted:,}")
    print(f"Total in database      : {total_in_db:,}")
    print(f"Files skipped          : {total_skipped:,}")
    print("=" * 55)

    # Show breakdown by status
    print("\nBreakdown by status:")
    rows = await conn.fetch("""
        SELECT status, COUNT(*) as count
        FROM trials
        GROUP BY status
        ORDER BY count DESC
    """)
    for row in rows:
        print(f"  {row['status']:30} {row['count']:,}")

    # Show breakdown by phase
    print("\nBreakdown by phase:")
    rows = await conn.fetch("""
        SELECT phase, COUNT(*) as count
        FROM trials
        GROUP BY phase
        ORDER BY count DESC
    """)
    for row in rows:
        print(f"  {row['phase']:20} {row['count']:,}")

    # Show 5 sample trials
    print("\n5 random trials from your database:")
    rows = await conn.fetch("""
        SELECT nct_id, status, phase, minimum_age, maximum_age,
               LEFT(title, 55) as short_title
        FROM trials
        ORDER BY RANDOM()
        LIMIT 5
    """)
    for row in rows:
        print(
            f"  {row['nct_id']} | "
            f"{row['status']:20} | "
            f"{row['phase']:10} | "
            f"age {row['minimum_age']}-{row['maximum_age']} | "
            f"{row['short_title']}"
        )

    await conn.close()
    print("\nDone. Real clinical trial data is in your database.")


if __name__ == "__main__":
    asyncio.run(main())