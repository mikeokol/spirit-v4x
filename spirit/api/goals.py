from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from spirit.db import async_session
from spirit.models import Goal, GoalState
from spirit.services.goal_service import create_goal, get_active_goal
from spirit.api.auth import get_current_user
from spirit.schema.response import GoalRead

router = APIRouter(prefix="/goals", tags=["goals"])

async def get_db() -> AsyncSession:
    async with async_session() as session:
        yield session

@router.post("", response_model=GoalRead)
async def declare_goal(
    text: str,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    active = await get_active_goal(db, user.id)
    if active:
        raise HTTPException(status_code=409, detail="Active goal already exists; abandon or complete it first")
    goal = await create_goal(db, user.id, text)
    return goal

@router.get("/active", response_model=GoalRead | None)
async def active_goal(user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    return await get_active_goal(db, user.id)

@router.patch("/{goal_id}/complete", response_model=dict)
async def complete_goal(goal_id: int, user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    goal = await db.get(Goal, goal_id)
    if not goal or goal.user_id != user.id:
        raise HTTPException(status_code=404, detail="Goal not found")
    goal.state = GoalState.completed
    await db.commit()
    return {"detail": "Goal marked completed"}

@router.patch("/{goal_id}/abandon", response_model=dict)
async def abandon_goal(goal_id: int, user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    goal = await db.get(Goal, goal_id)
    if not goal or goal.user_id != user.id:
        raise HTTPException(status_code=404, detail="Goal not found")
    goal.state = GoalState.abandoned
    await db.commit()
    return {"detail": "Goal abandoned"}
