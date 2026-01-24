from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import UUID
from spirit.db import async_session
from spirit.models import Goal, GoalState, User, ModeState
from spirit.api.auth import get_current_user
from spirit.strategies.library import DEFAULT_STRATEGY_BY_DOMAIN

router = APIRouter(prefix="/goals", tags=["goals"])

async def get_db() -> AsyncSession:
    async with async_session() as session:
        yield session

@router.post("", response_model=dict)
async def declare_goal(
    text: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    active = await db.scalar(select(Goal).where(Goal.user_id == user.id, Goal.state == GoalState.active))
    if active:
        raise HTTPException(status_code=409, detail="Active goal already exists; abandon or complete it first")
    goal = Goal(user_id=user.id, text=text.strip())
    db.add(goal)
    await db.commit()
    await db.refresh(goal)
    return {"id": goal.id, "text": goal.text, "state": goal.state}

@router.patch("/{goal_id}/activate", response_model=dict)
async def activate_goal(goal_id: UUID, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """
    Guardrail: when user switches active goal, reset strategic mode.
    """
    goal = await db.get(Goal, goal_id)
    if not goal or goal.user_id != user.id:
        raise HTTPException(status_code=404, detail="Goal not found")

    # deactivate any other active goal
    await db.execute(
        select(Goal).where(Goal.user_id == user.id, Goal.state == GoalState.active)
        .execution_options(synchronize_session="fetch")
    )
    for g in await db.scalars(select(Goal).where(Goal.user_id == user.id, Goal.state == GoalState.active)):
        g.state = GoalState.abandoned

    goal.state = GoalState.active
    goal.domain = goal.domain or "business"  # default
    await db.commit()

    # reset strategic mode
    mode = await db.scalar(select(ModeState).where(ModeState.user_id == user.id))
    if mode:
        mode.strategic_enabled = False
        mode.strategic_state = ModeState.locked
        mode.strategy_key = None
        mode.strategy_started_at = None
        await db.commit()

    return {"detail": "Goal activated, strategic mode reset"}

@router.get("/active", response_model=dict | None)
async def active_goal(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    goal = await db.scalar(select(Goal).where(Goal.user_id == user.id, Goal.state == GoalState.active))
    return {"id": goal.id, "text": goal.text, "state": goal.state, "domain": goal.domain} if goal else None

@router.patch("/{goal_id}/complete", response_model=dict)
async def complete_goal(goal_id: UUID, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    goal = await db.get(Goal, goal_id)
    if not goal or goal.user_id != user.id:
        raise HTTPException(status_code=404, detail="Goal not found")
    goal.state = GoalState.completed
    await db.commit()
    return {"detail": "Goal marked completed"}

@router.patch("/{goal_id}/abandon", response_model=dict)
async def abandon_goal(goal_id: UUID, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    goal = await db.get(Goal, goal_id)
    if not goal or goal.user_id != user.id:
        raise HTTPException(status_code=404, detail="Goal not found")
    goal.state = GoalState.abandoned
    await db.commit()
    return {"detail": "Goal abandoned"}
