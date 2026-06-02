"""TrialBridge — Trials Endpoints (stub for Day 1)"""
from fastapi import APIRouter
router = APIRouter()

@router.get("/trials")
async def list_trials():
    """Trial listing endpoint. Full implementation: Day 4."""
    return {"message": "Trials endpoint — implementation coming Day 4"}
