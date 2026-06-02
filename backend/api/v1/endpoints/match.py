"""TrialBridge — Match Endpoints (stub for Day 1)"""
from fastapi import APIRouter
router = APIRouter()

@router.post("/match")
async def match_patient():
    """Matching endpoint. Full implementation: Week 3."""
    return {"message": "Match endpoint — implementation coming Week 3"}
