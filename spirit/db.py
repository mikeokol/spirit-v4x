import aiosqlite
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel
from spirit.config import settings

engine = create_async_engine(
    settings.db_url,
    echo=False,
    # no SQLite-only args for Postgres
    connect_args={"check_same_thread": False} if "sqlite" in settings.db_url else {},
)

async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def create_db_and_tables():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
