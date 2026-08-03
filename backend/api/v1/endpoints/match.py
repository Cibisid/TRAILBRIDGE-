"""
TrialBridge — Match Endpoint
POST /api/v1/match — takes a patient note, returns ranked trial matches.
This is the core product endpoint. Everything built in Days 1-9 leads here.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import get_db
from backend.matching.matcher import TrialMatchingEngine

router = APIRouter()

# Single shared engine instance — model loads once, stays in memory
_engine = TrialMatchingEngine()


# -----------------------------------------------
# Request / Response Models
# -----------------------------------------------
class MatchRequest(BaseModel):
    note: str
    n_results: int = 10

    class Config:
        json_schema_extra = {
            "example": {
                "note": "58-year-old female with Type 2 Diabetes. HbA1c 8.9%, eGFR 72. Currently on Metformin. No prior insulin therapy.",
                "n_results": 10,
            }
        }


class TrialMatchResult(BaseModel):
    nct_id: str
    title: str
    status: str
    phase: str
    sponsor: str | None
    conditions: list[str]
    scores: dict
    eligibility_status: str
    inclusion_met: list[str]
    exclusion_flags: list[str]
    needs_verification: list[str]


class MatchResponse(BaseModel):
    patient_profile: dict
    matches: list[TrialMatchResult]
    total_matches: int
    query_string: str


# -----------------------------------------------
# Endpoint
# -----------------------------------------------
@router.post(
    "/match",
    response_model=MatchResponse,
    summary="Match patient to clinical trials",
    description="""
    Takes a raw clinical note and returns ranked clinical trial matches.

    Pipeline:
    1. NLP extracts structured patient profile from the note
    2. Profile converts to semantic embedding vector
    3. pgvector searches 2,344 trials for closest semantic matches
    4. Eligibility rules engine scores each candidate
    5. Composite score (semantic 60% + eligibility 40%) ranks final results

    Returns up to n_results matches with full scoring breakdown.
    """,
)
async def match_patient(
    request: MatchRequest,
    db: AsyncSession = Depends(get_db),
) -> MatchResponse:
    if not request.note or len(request.note.strip()) < 10:
        raise HTTPException(
            status_code=400,
            detail="Note is too short. Please provide a complete clinical note.",
        )

    if request.n_results < 1 or request.n_results > 50:
        raise HTTPException(
            status_code=400,
            detail="n_results must be between 1 and 50.",
        )

    try:
        # Get raw database connection for pgvector queries
        raw_conn = await db.connection()
        raw_conn = await raw_conn.get_raw_connection()
        driver_conn = raw_conn.driver_connection

        profile, matches = await _engine.match_from_note(
            request.note,
            driver_conn,
            n_results=request.n_results,
        )

        return MatchResponse(
            patient_profile=profile.to_dict(),
            matches=[TrialMatchResult(**m.to_dict()) for m in matches],
            total_matches=len(matches),
            query_string=profile.to_query_string(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Matching failed: {str(e)}",
        )
