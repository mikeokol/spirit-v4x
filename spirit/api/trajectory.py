from datetime import date
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from spirit.db import async_session
from spirit.models import Goal, GoalState
from spirit.api.auth import get_current_user
from spirit.graphs.daily_objective_graph import daily_objective_graph
from spirit.models import User

router = APIRouter(prefix="/trajectory", tags=["trajectory"])

async def get_db() -> AsyncSession:
    async with async_session() as session:
        yield session

@router.post("/execute")
async def log_day(
    objective_text: str,
    executed: bool,
    day: date = date.today(),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    goal = await db.get(Goal, user.active_goal_id)  # simplified lookup
    if not goal or goal.state != GoalState.active:
        raise HTTPException(status_code=409, detail="No active goal")
    from spirit.services.execution_service import log_execution
    ex = await log_execution(db, goal.id, day, objective_text, executed)
    return ex

@router.get("/history")
async def history(limit: int = 30, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    goal = await db.get(Goal, user.active_goal_id)
    if not goal:
        return []
    executions = await db.execute(
        select(Execution)
        .where(Execution.goal_id == goal.id)
        .order_by(Execution.day.desc())
        .limit(limit)
    )
    return executions.scalars().all()

# NEW: generate today's objective via graph
@router.post("/daily/generate")
async def generate_daily_objective(
    user: User = Depends(get_current_user),
):
    state = {"user_id": user.id}
    result = await daily_objective_graph.ainvoke(state)
    return result["objective"]
