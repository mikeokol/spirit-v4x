from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from spirit.db import async_session
from spirit.models import Goal, GoalState, User
from spirit.services.goal_service import create_goal, get_active_goal
from spirit.api.auth import create_access_token, verify_pw, hash_pw

router = APIRouter(prefix="/goals", tags=["goals"])

async def get_db() -> AsyncSession:
    async with async_session() as session:
        yield session

async def current_user(db: AsyncSession = Depends(get_db), token: str = Depends(oauth2_bearer)) -> User:
    from spirit.api.auth import decode_token
    payload = decode_token(token)
    user_id = int(payload.get("sub"))
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.post("", response_model=GoalRead)
async def declare_goal(text: str, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)):
    active = await get_active_goal(db, user.id)
    if active:
        raise HTTPException(status_code=409, detail="Active goal already exists; abandon or complete it first")
    goal = await create_goal(db, user.id, text)
    return goal

@router.get("/active", response_model=GoalRead | None)
async def active_goal(user: User = Depends(current_user), db: AsyncSession = Depends(get_db)):
    return await get_active_goal(db, user.id)

@router.patch("/{goal_id}/complete")
async def complete_goal(goal_id: int, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)):
    goal = await db.get(Goal, goal_id)
    if not goal or goal.user_id != user.id:
        raise HTTPException(status_code=404, detail="Goal not found")
    goal.state = GoalState.completed
    await db.commit()
    return {"detail": "Goal marked completed"}

@router.patch("/{goal_id}/abandon")
async def abandon_goal(goal_id: int, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)):
    goal = await db.get(Goal, goal_id)
    if not goal or goal.user_id != user.id:
        raise HTTPException(status_code=404, detail="Goal not found")
    goal.state = GoalState.abandoned
    await db.commit()
    return {"detail": "Goal abandoned"}
