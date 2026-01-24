from datetime import date as dt_date
from typing import Optional, Dict
from uuid import UUID
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from spirit.models import Execution

async def upsert_execution(
    db: AsyncSession,
    *,
    user_id: UUID,
    goal_id: UUID,
    daily_objective_id: UUID,
    day: dt_date,
    status: str,  # 'done' | 'miss' | 'partial'
    proof: Optional[Dict] = None,
) -> Execution:
    """
    Postgres-native UPSERT on unique (daily_objective_id).
    Idempotent under retries.
    """
    stmt = (
        insert(Execution)
        .values(
            user_id=user_id,
            goal_id=goal_id,
            daily_objective_id=daily_objective_id,
            day=day,
            executed=(status == "done"),
            objective_text=proof.get("objective_text", "") if proof else "",
            logged_at=dt_date.today(),
        )
        .on_conflict_do_update(
            index_elements=[Execution.daily_objective_id],
            set_={
                "executed": (status == "done"),
                "objective_text": proof.get("objective_text", "") if proof else "",
                "logged_at": dt_date.today(),
            },
        )
        .returning(Execution)
    )
    res = await db.execute(stmt)
    await db.commit()
    return res.scalar_one()
