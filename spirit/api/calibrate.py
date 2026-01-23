from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from spirit.db import async_session
from spirit.models import GoalProfile, Goal, GoalState, User
from spirit.api.auth import get_current_user
from spirit.schemas.goal_profile import GoalProfileCreate, GoalProfileRead
from spirit.services.calibrator import complexity_score

router = APIRouter(prefix="/calibrate", tags=["calibrate"])

async def get_db() -> AsyncSession:
    async with async_session() as session:
        yield session

@router.post("", response_model=GoalProfileRead)
async def create_profile(
    body: GoalProfileCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    goal = await db.scalar(select(Goal).where(Goal.user_id == user.id, Goal.state == GoalState.active))
    if not goal:
        raise HTTPException(status_code=409, detail="No active goal")

    existing = await db.scalar(select(GoalProfile).where(GoalProfile.goal_id == goal.id))
    if existing:
        raise HTTPException(status_code=409, detail="Profile already exists for this goal")

    comp = complexity_score(goal.text)
    goal.complexity = comp
    await db.commit()

    profile = GoalProfile(goal_id=goal.id, **body.model_dump())
    db.add(profile)
    await db.commit()
    await db.refresh(profile)
    return profile

@router.get("", response_model=GoalProfileRead | None)
async def get_profile(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    goal = await db.scalar(select(Goal).where(Goal.user_id == user.id, Goal.state == GoalState.active))
    if not goal:
        return None
    profile = await db.scalar(select(GoalProfile).where(GoalProfile.goal_id == goal.id))
    return profile
