import pytest
from datetime import date
from sqlalchemy.ext.asyncio import AsyncSession
from spirit.models import Goal, GoalState

@pytest.fixture
async def goal(db: AsyncSession, user) -> Goal:
    g = Goal(user_id=user.id, text="Test goal", state=GoalState.active)
    db.add(g)
    await db.commit()
    await db.refresh(g)
    return g

@pytest.fixture
async def token(client, user):
    r = client.post("/api/auth/login", data={"username": user.email, "password": "str0ng!"})
    return r.json()["access_token"]
