from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select, func
from spirit.models import Execution, Goal, GoalState

STABILITY_DAYS = 30
MIN_EXECUTION_RATE = 0.7
MIN_STREAK = 5

async def check_strategic_unlock(db: AsyncSession, user_id: int) -> bool:
    # Must have an active goal
    goal = await db.get(Goal, user_id)  # simplified; real code looks up active goal
    if not goal or goal.state != GoalState.active:
        return False

    # Last 30 days of executions
    since = datetime.utcnow() - timedelta(days=STABILITY_DAYS)
    res = await db.execute(
        select(func.count(Execution.id), func.sum(Execution.executed.cast(Integer)))
        .where(Execution.goal_id == goal.id)
        .where(Execution.day >= since.date())
    )
    total, done = res.one()
    if not total:
        return False
    rate = done / total
    if rate < MIN_EXECUTION_RATE:
        return False

    # Last 5 days continuous
    streak_since = datetime.utcnow().date() - timedelta(days=MIN_STREAK)
    streak_res = await db.execute(
        select(Execution.day, Execution.executed)
        .where(Execution.goal_id == goal.id)
        .where(Execution.day >= streak_since)
        .order_by(Execution.day)
    )
    days = streak_res.all()
    if len(days) != MIN_STREAK:
        return False
    if any(not d.executed for d in days):
        return False

    return True
