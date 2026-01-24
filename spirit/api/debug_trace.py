import os
from fastapi import APIRouter
from langsmith import traceable

router = APIRouter(prefix="/debug", tags=["debug"])

@router.get("/env")
async def env():
    return {
        "LANGSMITH_TRACING": os.getenv("LANGSMITH_TRACING"),
        "LANGSMITH_PROJECT": os.getenv("LANGSMITH_PROJECT"),
        "LANGSMITH_API_ENDPOINT": os.getenv("LANGSMITH_API_ENDPOINT"),
        "LANGSMITH_API_KEY_present": bool(os.getenv("LANGSMITH_API_KEY")),
        "LANGSMITH_API_KEY_prefix": (os.getenv("LANGSMITH_API_KEY") or "")[:6],
    }

@router.get("/trace")
@traceable(name="debug_trace_ping")
async def trace():
    return {"ok": True}
