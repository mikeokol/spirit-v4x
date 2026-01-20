from contextlib import asynccontextmanager
from fastapi import FastAPI
from spirit.config import settings
from spirit.db import create_db_and_tables
from spirit.api import auth, goals, trajectory, strategic

@asynccontextmanager
async def lifespan(_: FastAPI):
    await create_db_and_tables()
    yield

app = FastAPI(
    title="Spirit",
    description="Continuity ledger for human intention",
    version="0.1.0",
    lifespan=lifespan,
)

# health
@app.get("/")
def read_root():
    return {"message": "Spirit continuity ledger is running", "docs": "/docs"}

# routers
app.include_router(auth.router, prefix="/api", tags=["auth"])
app.include_router(goals.router, prefix="/api", tags=["goals"])
app.include_router(trajectory.router, prefix="/api", tags=["trajectory"])
app.include_router(strategic.router, prefix="/api", tags=["strategic"])
