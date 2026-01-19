from datetime import date
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from spirit.db import async_session
from spirit.models import Execution, Goal, GoalState
from spirit.services.execution_service import log_execution
from spirit.api.auth import get_current_user as current_user

router = APIRouter(prefix="/trajectory", tags=["trajectory"])

async def get_db() -> AsyncSession:
    async with async_session() as session:
        yield session

@router.post("/execute")
async def log_day(
    objective_text: str,
    executed: bool,
    day: date = date.today(),
    user=Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    goal = await db.get(Goal, user.active_goal_id)  # simplified lookup
    if not goal or goal.state != GoalState.active:
        raise HTTPException(status_code=409, detail="No active goal")
    ex = await log_execution(db, goal.id, day, objective_text, executed)
    return ex

@router.get("/history")
async def history(limit: int = 30, user=Depends(current_user), db: AsyncSession = Depends(get_db)):
    goal = await db.get(Goal, user.active_goal_id)
    if not goal:
        return []
    res = await db.execute(
        select(Execution)
        .where(Execution.goal_id == goal.id)
        .order_by(Execution.day.desc())
        .limit(limit)
    )
    return res.scalars().all()
