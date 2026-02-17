"""
Spirit Database Layer
Manages dual-database architecture:
- SQLite: Core application data (goals, executions, users)
- Supabase: Behavioral time-series data (observations, memory, causal graphs)
"""

import aiosqlite
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator

from spirit.config import settings


# ==========================================
# SQLITE CORE DATABASE (Existing)
# ==========================================

def _get_connect_args() -> dict:
    """Get database-specific connection arguments."""
    if "sqlite" in settings.db_url:
        return {"check_same_thread": False}
    # PostgreSQL via asyncpg needs no special connect_args
    return {}

engine = create_async_engine(
    settings.db_url,
    echo=settings.env == "dev",  # Enable SQL logging in dev
    future=True,
    pool_pre_ping=True,  # Verify connections before use
    connect_args=_get_connect_args(),
)

async_session = sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

async def create_db_and_tables():
    """Initialize SQLite database schema."""
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    print(f"Database tables created: {settings.db_url}")


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for FastAPI endpoints.
    Provides transactional database session with automatic cleanup.
    """
    session = async_session()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


# ==========================================
# SUPABASE BEHAVIORAL STORE (New)
# ==========================================

class SupabaseConnectionError(Exception):
    """Raised when Supabase connection fails."""
    pass


class BehavioralStore:
    """
    Manages Supabase connection for behavioral data.
    Singleton pattern with lazy initialization.
    """
    
    _instance: Optional["BehavioralStore"] = None
    _client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def client(self):
        """Get or create Supabase client."""
        if self._client is None and settings.supabase_url:
            self._initialize()
        return self._client
    
    def _initialize(self):
        """Initialize Supabase client."""
        if not settings.supabase_url or not settings.supabase_anon_key:
            raise SupabaseConnectionError("Supabase credentials not configured")
        
        try:
            from supabase import create_client
            
            # Use service role key for backend operations if available
            key = settings.supabase_service_key or settings.supabase_anon_key
            
            self._client = create_client(settings.supabase_url, key)
            print(f"Supabase client initialized: {settings.supabase_url}")
        except Exception as e:
            raise SupabaseConnectionError(f"Failed to initialize Supabase: {e}")
    
    async def health_check(self) -> bool:
        """Verify Supabase connection is alive."""
        if not self.client:
            return False
        
        try:
            # Lightweight query to test connection
            result = self.client.table('health_check').select('*').limit(1).execute()
            return True
        except Exception:
            return False
    
    def close(self):
        """Close Supabase connection."""
        self._client = None
        BehavioralStore._instance = None


# Global accessor
def get_behavioral_store() -> Optional[BehavioralStore]:
    """Get Supabase behavioral store instance."""
    if not settings.supabase_url:
        return None
    
    try:
        return BehavioralStore()
    except SupabaseConnectionError:
        return None


async def close_behavioral_store():
    """Cleanup Supabase connection on shutdown."""
    store = get_behavioral_store()
    if store:
        store.close()
        print("Supabase connection closed")


# ==========================================
# UNIFIED DATABASE INTERFACE (New)
# ==========================================

class SpiritDatabase:
    """
    Unified interface for both SQLite and Supabase operations.
    Provides transactional integrity across both stores where needed.
    """
    
    def __init__(self):
        self.sqlite_session: Optional[AsyncSession] = None
        self.supabase_store: Optional[BehavioralStore] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.sqlite_session = async_session()
        self.supabase_store = get_behavioral_store()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        if self.sqlite_session:
            if exc_type:
                await self.sqlite_session.rollback()
            else:
                await self.sqlite_session.commit()
            await self.sqlite_session.close()
    
    # SQLite operations
    async def execute_sqlite(self, statement):
        """Execute SQLAlchemy statement on SQLite."""
        if not self.sqlite_session:
            raise RuntimeError("SQLite session not initialized")
        result = await self.sqlite_session.execute(statement)
        return result
    
    async def add_sqlite(self, obj):
        """Add object to SQLite session."""
        if not self.sqlite_session:
            raise RuntimeError("SQLite session not initialized")
        self.sqlite_session.add(obj)
    
    # Supabase operations
    async def query_supabase(self, table: str, query_fn):
        """
        Execute query on Supabase.
        query_fn: function that takes table reference and returns query result
        """
        if not self.supabase_store or not self.supabase_store.client:
            raise SupabaseConnectionError("Supabase not available")
        
        table_ref = self.supabase_store.client.table(table)
        return query_fn(table_ref)
    
    async def insert_supabase(self, table: str, data: dict):
        """Insert data into Supabase table."""
        return await self.query_supabase(
            table, 
            lambda t: t.insert(data).execute()
        )
    
    async def upsert_supabase(self, table: str, data: dict, on_conflict: str):
        """Upsert data into Supabase table."""
        return await self.query_supabase(
            table,
            lambda t: t.upsert(data, on_conflict=on_conflict).execute()
        )


@asynccontextmanager
async def get_spirit_db() -> AsyncGenerator[SpiritDatabase, None]:
    """
    Unified database context manager.
    Provides access to both SQLite and Supabase in single context.
    """
    db = SpiritDatabase()
    async with db:
        yield db


# ==========================================
# MIGRATION & SETUP UTILITIES (New)
# ==========================================

async def verify_database_connections() -> dict:
    """
    Health check for all database connections.
    Returns status of each connection.
    """
    status = {
        "sqlite": False,
        "supabase": False,
        "errors": []
    }
    
    # Check SQLite
    try:
        async with engine.connect() as conn:
            await conn.execute("SELECT 1")
        status["sqlite"] = True
    except Exception as e:
        status["errors"].append(f"SQLite: {str(e)}")
    
    # Check Supabase
    store = get_behavioral_store()
    if store:
        try:
            status["supabase"] = await store.health_check()
        except Exception as e:
            status["errors"].append(f"Supabase: {str(e)}")
    else:
        status["errors"].append("Supabase: not configured")
    
    return status


async def init_databases():
    """
    Initialize all databases on application startup.
    Idempotent - safe to run multiple times.
    """
    # SQLite
    await create_db_and_tables()
    
    # Verify Supabase if configured
    if settings.supabase_url
