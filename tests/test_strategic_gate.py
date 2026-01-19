import pytest
from datetime import date, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from spirit.models import Goal, GoalState, Execution
from spirit.services.strategic_gate import check_strategic_unlock

pytestmark = pytest.mark.asyncio

async def test_gate_locked_on_fresh_goal(db: AsyncSession, user):
    goal = Goal(user_id=user.id, text="test", state=GoalState.active)
    db.add(goal)
    await db.commit()
    locked = await check_strategic_unlock(db, user.id)
    assert locked is False

async def test_gate_unlocks_after_30_days_70_percent(db: AsyncSession, user):
    goal = Goal(user_id=user.id, text="test", state=GoalState.active)
    db.add(goal)
    await db.flush()
    # 21 executed out of 30 days
    for i in range(30):
        day = date.today() - timedelta(days=29-i)
        ex = Execution(
            goal_id=goal.id,
            day=day.isoformat(),
            objective_text="x",
            executed=(i < 21),
        )
        db.add(ex)
    await db.commit()
    unlocked = await check_strategic_unlock(db, user.id)
    assert unlocked is True
