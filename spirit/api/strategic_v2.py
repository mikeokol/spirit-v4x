"""
Strategic API v2: Behaviorally-informed long-term planning.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.security import HTTPBearer

from spirit.strategic_v2 import StrategicUnlockEngine, StrategicPlanGenerator, StrategicMaturityLevel
from spirit.db import async_session


router = APIRouter(prefix="/api/strategic", tags=["strategic"])
security = HTTPBearer()


async def get_db():
    async with async_session() as session:
        yield session


async def get_current_user(credentials=Depends(security)) -> int:
    # Your existing user extraction
    import base64
    import json
    try:
        payload = credentials.credentials.split('.')[1]
        padding = 4 - len(payload) % 4
        if padding != 4:
            payload += '=' * padding
        decoded = base64.urlsafe_b64decode(payload)
        claims = json.loads(decoded)
        return int(claims['sub'])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


@router.get("/status")
async def get_strategic_status(
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user)
):
    """
    Get comprehensive strategic unlock status with behavioral criteria.
    """
    engine = StrategicUnlockEngine(user_id)
    status = await engine.check_strategic_unlock_v2(db)
    
    return status


@router.get("/plan")
async def get_12_week_plan(
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user)
):
    """
    Generate 12-week strategic plan based on behavioral patterns.
    """
    # First check maturity
    engine = StrategicUnlockEngine(user_id)
    status = await engine.check_strategic_unlock_v2(db)
    
    maturity = StrategicMaturityLevel(status["maturity_level"])
    
    # Generate plan
    planner = StrategicPlanGenerator(user_id, maturity)
    plan = await planner.generate_12_week_plan(db)
    
    return plan


@router.post("/force-check")
async def force_strategic_recheck(
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user)
):
    """
    Force recalculation of strategic status (after new behavioral data).
    """
    engine = StrategicUnlockEngine(user_id)
    status = await engine.check_strategic_unlock_v2(db)
    
    # If newly unlocked, trigger celebration/notification
    if status["can_unlock_strategic"] and status["maturity_level"] in ["strategic", "adaptive"]:
        # TODO: Trigger notification
        pass
    
    return status
