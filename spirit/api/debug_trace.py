from fastapi import APIRouter
from langsmith import traceable

router = APIRouter(prefix="/debug", tags=["debug"])

@router.get("/trace")
@traceable(name="debug_trace_ping")
async def debug_trace():
    return {"ok": True}
