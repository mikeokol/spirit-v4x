import pytest
import asyncio
from typing import AsyncGenerator
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlmodel import SQLModel
from spirit.main import app
from spirit.db import async_session
from spirit.models import User
from spirit.config import settings

# in-memory test db
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"
engine = create_async_engine(TEST_DB_URL, pool_pre_ping=True, echo=False)

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session", autouse=True)
async def create_test_db():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)

@pytest.fixture
async def db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session

@pytest.fixture
def client() -> TestClient:
    return TestClient(app)

@pytest.fixture
async def user(db: AsyncSession) -> User:
    u = User(email="test@spirit.local", hashed_password="$2b$12$fakehash")
    db.add(u)
    await db.commit()
    await db.refresh(u)
    return u
