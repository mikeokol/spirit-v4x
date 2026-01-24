from datetime import date as dt_date
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import UUID
from spirit.db import async_session
from spirit.api.auth import get_current_user
from spirit.models import Goal, GoalState, ModeState, User
from spirit.strategies.library import DEFAULT_STRATEGY_BY_DOMAIN, STRATEGIES
from spirit.strategies.review import review_week

router = APIRouter(prefix="/strategic", tags=["strategic"])

async def get_db() -> AsyncSession:
    async with async_session() as session:
        yield session

@router.get("/review")
async def strategic_review(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    goal = await db.scalar(select(Goal).where(Goal.user_id == user.id, Goal.state == GoalState.active))
    if not goal:
        raise HTTPException(status_code=409, detail="No active goal")

    mode = await db.scalar(select(ModeState).where(ModeState.user_id == user.id))
    if not mode:
        mode = ModeState(user_id=user.id, constraint_level="observer", strategic_enabled=False, strategic_state="locked")
        db.add(mode)
        await db.commit()
        await db.refresh(mode)

    domain = mode.domain or "business"
    strategy_key = mode.strategy_key or DEFAULT_STRATEGY_BY_DOMAIN[domain]
    if strategy_key not in STRATEGIES:
        strategy_key = DEFAULT_STRATEGY_BY_DOMAIN[domain]

    today = dt_date.today()
    result = await review_week(db, goal_id=goal.id, domain=domain, strategy_key=strategy_key, today=today)

    mode.last_reviewed_at = today
    mode.execution_rate_30d = result.adherence_rate_7d  # placeholder until real 30d calc

    if result.decision == "pivot":
        mode.strategy_key = result.pivot_to or DEFAULT_STRATEGY_BY_DOMAIN[domain]
        mode.strategy_started_at = today
        mode.strategic_state = "testing"
    elif result.decision == "continue":
        mode.strategic_state = "testing"
    else:  # stabilizing
        mode.strategic_state = "stabilizing"

    await db.commit()

    return {
        "goal_id": goal.id,
        "domain": domain,
        "strategy_key": mode.strategy_key,
        "decision": result.decision,
        "adherence_rate_7d": result.adherence_rate_7d,
        "days": {"done": result.done_days, "miss": result.miss_days, "partial": result.partial_days},
        "signals_sum": result.signals_sum,
        "caps": {"difficulty_cap": result.difficulty_cap, "time_budget_cap": result.time_budget_cap},
        "explanation": result.explanation,
    }
