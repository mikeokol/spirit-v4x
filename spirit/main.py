from contextlib import asynccontextmanager
from fastapi import FastAPI
from spirit.config import settings
from spirit.db import create_db_and_tables
from spirit.api import auth, goals, trajectory, strategic, anchors, calibrate, debug_trace

@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.env != "prod":
        await create_db_and_tables()
    yield

app = FastAPI(
    title="Spirit",
    description="Continuity ledger for human intention",
    version="0.7.1",
    lifespan=lifespan,
)

@app.get("/")
def read_root():
    return {"message": "Spirit continuity ledger is running", "docs": "/docs"}

@app.get("/continuity")
async def continuity():
    from datetime import date
    from sqlalchemy import select, func
    from spirit.db import async_session
    from spirit.models import Execution

    async with async_session() as session:
        oldest = await session.scalar(select(func.min(Execution.day)))
    return {"oldest_execution": oldest.isoformat() if oldest else None}

# routers
app.include_router(auth.router, prefix="/api", tags=["auth"])
app.include_router(goals.router, prefix="/api", tags=["goals"])
app.include_router(trajectory.router, prefix="/api", tags=["trajectory"])
app.include_router(strategic.router, prefix="/api", tags=["strategic"])
app.include_router(anchors.router, prefix="/api", tags=["anchors"])
app.include_router(calibrate.router, prefix="/api", tags=["calibrate"])
app.include_router(debug_trace.router, prefix="/api", tags=["debug"])
