from datetime import date
from sqlalchemy.ext.asyncio import AsyncSession
from spirit.models import Execution

async def log_execution(
    db: AsyncSession,
    goal_id: int,
    day: date,
    objective_text: str,
    executed: bool,
) -> Execution:
    # Upsert: one row per day
    existing = await db.get(Execution, (goal_id, day))
    if existing:
        existing.objective_text = objective_text
        existing.executed = executed
    else:
        existing = Execution(
            goal_id=goal_id,
            day=day,
            objective_text=objective_text,
            executed=executed,
        )
        db.add(existing)
    await db.commit()
    await db.refresh(existing)
    return existing
