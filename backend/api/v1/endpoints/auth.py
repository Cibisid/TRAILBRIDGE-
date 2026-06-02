"""TrialBridge — Auth Endpoints (stub for Day 1)"""
from fastapi import APIRouter
router = APIRouter()

@router.post("/auth/token")
async def login():
    """Returns JWT token pair. Full implementation: Day 5."""
    return {"message": "Auth endpoint — implementation coming Day 5"}
