from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from spirit.db import async_session
from spirit.models import RealityAnchor, Goal, GoalState
from spirit.api.auth import get_current_user
from spirit.schemas.reality_anchor import RealityAnchorSchema
from spirit.models import User

router = APIRouter(prefix="/anchors", tags=["anchors"])

async def get_db() -> AsyncSession:
    async with async_session() as session:
        yield session

@router.post("", response_model=RealityAnchorSchema)
async def create_anchor(
    body: RealityAnchorSchema,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    goal = await db.scalar(select(Goal).where(Goal.user_id == user.id, Goal.state == GoalState.active))
    if not goal:
        raise HTTPException(status_code=409, detail="No active goal")

    existing = await db.scalar(select(RealityAnchor).where(RealityAnchor.goal_id == goal.id))
    if existing:
        raise HTTPException(status_code=409, detail="Anchor already exists for this goal")

    anchor = RealityAnchor(goal_id=goal.id, **body.model_dump())
    db.add(anchor)
    await db.commit()
    await db.refresh(anchor)
    return anchor

@router.get("", response_model=RealityAnchorSchema | None)
async def get_anchor(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    goal = await db.scalar(select(Goal).where(Goal.user_id == user.id, Goal.state == GoalState.active))
    if not goal:
        return None
    anchor = await db.scalar(select(RealityAnchor).where(RealityAnchor.goal_id == goal.id))
    return anchor
