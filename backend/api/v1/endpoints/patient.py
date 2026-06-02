"""TrialBridge — Patient Endpoints (stub for Day 1)"""
from fastapi import APIRouter
router = APIRouter()

@router.post("/parse-patient")
async def parse_patient():
    """NLP extraction endpoint. Full implementation: Week 2."""
    return {"message": "Patient parsing endpoint — implementation coming Week 2"}
