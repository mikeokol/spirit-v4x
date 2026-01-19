from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from spirit.models import Goal, GoalState

async def create_goal(db: AsyncSession, user_id: int, text: str) -> Goal:
    goal = Goal(user_id=user_id, text=text.strip(), state=GoalState.active)
    db.add(goal)
    await db.commit()
    await db.refresh(goal)
    return goal

async def get_active_goal(db: AsyncSession, user_id: int) -> Goal | None:
    from sqlmodel import select
    res = await db.execute(
        select(Goal)
        .where(Goal.user_id == user_id)
        .where(Goal.state == GoalState.active)
    )
    return res.scalar_one_or_none()
