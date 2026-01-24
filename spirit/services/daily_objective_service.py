from datetime import date as dt_date
from typing import Dict
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from spirit.models import DailyObjective

async def get_or_create_daily_objective(
    db: AsyncSession,
    *,
    user_id: UUID,
    goal_id: UUID,
    today: dt_date,
    payload: Dict[str, any],
) -> DailyObjective:
    """
    Idempotent get-or-create for (goal_id, today) tuple.
    Race-condition safe under concurrent requests.
    """
    stmt = select(DailyObjective).where(
        DailyObjective.goal_id == goal_id,
        DailyObjective.day == today,
    )
    existing = await db.scalar(stmt)
    if existing:
        return existing

    obj = DailyObjective(
        goal_id=goal_id,
        day=today,
        primary_objective=payload["primary_objective"],
        micro_steps=payload.get("micro_steps", []),
        time_budget_minutes=payload["time_budget_minutes"],
        success_criteria=payload["success_criteria"],
        difficulty=payload["difficulty"],
        adjustment_reason=payload.get("adjustment_reason"),
    )
    db.add(obj)
    try:
        await db.commit()
        await db.refresh(obj)
        return obj
    except IntegrityError:
        await db.rollback()
        return await db.scalar(stmt)  # fetch the winner
