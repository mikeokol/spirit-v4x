from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from spirit.db import async_session
from spirit.services.strategic_gate import check_strategic_unlock
from spirit.api.auth import get_current_user

router = APIRouter(prefix="/strategic", tags=["strategic"])

async def get_db() -> AsyncSession:
    async with async_session() as session:
        yield session

@router.get("/status")
async def strategic_status(user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    unlocked = await check_strategic_unlock(db, user.id)
    return {"unlocked": unlocked}

@router.post("/enter")
async def enter_strategic(user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if not await check_strategic_unlock(db, user.id):
        raise HTTPException(status_code=403, detail="Stability threshold not met")
    return {"detail": "Strategic mode unlocked (analytical endpoints TBD)"}
