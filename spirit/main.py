"""
Spirit Behavioral Research Agent - Main Application
Continuity ledger + Behavioral research engine + Causal inference
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from spirit.config import settings
from spirit.db import create_db_and_tables
from spirit.db.supabase_client import close_behavioral_store
from spirit.api import auth, goals, trajectory, strategic, anchors, calibrate, debug_trace
from spirit.api.ingestion import router as ingestion_router
from spirit.api.causal import router as causal_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - SQLite + Supabase behavioral store."""
    
    # Initialize SQLite (existing behavior)
    if settings.env != "prod":
        await create_db_and_tables()
        print(f"Spirit v{app.version} SQLite initialized ({settings.env} mode)")
    
    # Initialize Supabase behavioral store if configured
    if settings.supabase_url:
        from spirit.db.supabase_client import get_behavioral_store
        store = await get_behavioral_store()
        if store and store.client:
            print("Supabase behavioral store connected")
        else:
            print("Supabase URL configured but connection failed - ingestion disabled")
    
    yield
    
    # Cleanup
    print("Spirit shutting down...")
    await close_behavioral_store()


app = FastAPI(
    title="Spirit",
    description="Continuity ledger for human intention + Behavioral research engine + Causal inference",
    version="0.9.0",  # Bumped for causal inference
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "message": "Spirit continuity ledger is running",
        "docs": "/docs",
        "version": "0.9.0",
        "features": {
            "continuity_ledger": True,
            "behavioral_ingestion": bool(settings.supabase_url),
            "causal_inference": bool(settings.supabase_url)
        }
    }

@app.get("/continuity")
async def continuity():
    from datetime import date
    from sqlalchemy import select, func
    from spirit.db import async_session
    from spirit.models import Execution

    async with async_session() as session:
        oldest = await session.scalar(select(func.min(Execution.day)))
    return {"oldest_execution": oldest.isoformat() if oldest else None}

# Existing routers (preserved)
app.include_router(auth.router, prefix="/api", tags=["auth"])
app.include_router(goals.router, prefix="/api", tags=["goals"])
app.include_router(trajectory.router, prefix="/api", tags=["trajectory"])
app.include_router(strategic.router, prefix="/api", tags=["strategic"])
app.include_router(anchors.router, prefix="/api", tags=["anchors"])
app.include_router(calibrate.router, prefix="/api", tags=["calibrate"])
app.include_router(debug_trace.router, prefix="/api", tags=["debug"])

# NEW: Behavioral data ingestion router
app.include_router(ingestion_router)

# NEW: Causal inference router
app.include_router(causal_router)
