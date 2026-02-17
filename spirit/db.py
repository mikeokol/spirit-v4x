"""
Spirit Database Layer
Manages dual-database architecture:
- SQLite: Core application data (goals, executions, users)
- Supabase: Behavioral time-series data (observations, memory, causal graphs)
v1.4: Added tables for MAO debates, belief networks, ethical blocks, and memory consolidation.
"""

import aiosqlite
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator, Dict, Any, List
from datetime import datetime, timedelta

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
    v1.4: Added methods for new component tables.
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
            
            # NEW: Ensure required tables exist
            self._ensure_tables()
            
        except Exception as e:
            raise SupabaseConnectionError(f"Failed to initialize Supabase: {e}")
    
    def _ensure_tables(self):
        """
        Ensure all required tables exist for new components.
        Note: In production, use proper migrations. This is for development.
        """
        # Tables are created via SQL migrations, but we verify access here
        required_tables = [
            'behavioral_observations',
            'intervention_recommendations',  # NEW: For MAO debate queue
            'proactive_interventions',
            'blocked_interventions',         # NEW: Ethical guardrails
            'adversary_objections',          # NEW: MAO learning
            'belief_networks',               # NEW: User belief models
            'cognitive_dissonance_logs',     # NEW: Belief-reality gaps
            'memory_consolidation_queue',    # NEW: 3 AM processing
            'user_stress_metrics'            # NEW: HRV/burnout data
        ]
        
        # Verify we can query each table
        for table in required_tables:
            try:
                self._client.table(table).select('*').limit(1).execute()
            except Exception as e:
                print(f"Warning: Table {table} may not exist: {e}")
    
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
    
    # NEW: MAO Debate Operations
    async def get_pending_recommendations(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get intervention recommendations pending MAO debate."""
        if not self.client:
            return []
        
        result = self.client.table('intervention_recommendations').select('*').eq(
            'user_id', user_id
        ).eq('status', 'pending_debate').order('created_at').limit(limit).execute()
        
        return result.data if result.data else []
    
    async def update_recommendation_status(
        self, 
        recommendation_id: str, 
        status: str, 
        debate_result: Dict = None
    ):
        """Update recommendation status after MAO debate."""
        if not self.client:
            return
        
        update_data = {
            'status': status,
            'debated_at': datetime.utcnow().isoformat()
        }
        if debate_result:
            update_data['debate_result'] = debate_result
        
        self.client.table('intervention_recommendations').update(
            update_data
        ).eq('recommendation_id', recommendation_id).execute()
    
    # NEW: Belief Network Operations
    async def get_user_beliefs(self, user_id: str) -> Optional[Dict]:
        """Get user's current belief model."""
        if not self.client:
            return None
        
        result = self.client.table('belief_networks').select('*').eq(
            'user_id', user_id
        ).order('updated_at', desc=True).limit(1).execute()
        
        return result.data[0] if result.data else None
    
    async def update_belief_model(self, user_id: str, belief_update: Dict):
        """Update user's belief model."""
        if not self.client:
            return
        
        # Upsert belief model
        data = {
            'user_id': user_id,
            'beliefs': belief_update,
            'updated_at': datetime.utcnow().isoformat()
        }
        
        self.client.table('belief_networks').upsert(data).execute()
    
    async def log_cognitive_dissonance(self, user_id: str, dissonance_data: Dict):
        """Log detected cognitive dissonance for analysis."""
        if not self.client:
            return
        
        data = {
            'user_id': user_id,
            'user_belief': dissonance_data.get('user_belief'),
            'data_reality': dissonance_data.get('data_reality'),
            'gap_score': dissonance_data.get('gap', 0),
            'hypothesis_id': dissonance_data.get('hypothesis_id'),
            'detected_at': datetime.utcnow().isoformat()
        }
        
        self.client.table('cognitive_dissonance_logs').insert(data).execute()
    
    # NEW: Ethical Guardrails Operations
    async def log_blocked_intervention(
        self, 
        user_id: str, 
        prediction_data: Dict, 
        reason: str
    ):
        """Log when ethical guardrails block an intervention."""
        if not self.client:
            return
        
        data = {
            'user_id': user_id,
            'predicted_state': prediction_data.get('state_type'),
            'blocked_reason': reason,
            'confidence': prediction_data.get('confidence'),
            'blocked_at': datetime.utcnow().isoformat()
        }
        
        self.client.table('blocked_interventions').insert(data).execute()
    
    async def get_recent_stress_metrics(self, user_id: str, hours: int = 24) -> Dict:
        """Get recent stress/HRV metrics for ethical checks."""
        if not self.client:
            return {}
        
        since = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        
        result = self.client.table('user_stress_metrics').select('*').eq(
            'user_id', user_id
        ).gte('recorded_at', since).order('recorded_at', desc=True).execute()
        
        if not result.data:
            return {}
        
        # Calculate average stress
        avg_stress = sum(r.get('stress_score', 0) for r in result.data) / len(result.data)
        avg_hrv = sum(r.get('hrv_score', 0) for r in result.data) / len(result.data)
        
        return {
            'average_stress': avg_stress,
            'average_hrv': avg_hrv,
            'data_points': len(result.data),
            'latest': result.data[0]
        }
    
    # NEW: Memory Consolidation Operations
    async def get_episodes_for_consolidation(
        self, 
        user_id: str, 
        before_date: datetime,
        limit: int = 1000
    ) -> List[Dict]:
        """Get episodic memories ready for consolidation."""
        if not self.client:
            return []
        
        result = self.client.table('episodic_memories').select('*').eq(
            'user_id', user_id
        ).lt('created_at', before_date.isoformat()).eq(
            'consolidated', False
        ).limit(limit).execute()
        
        return result.data if result.data else []
    
    async def mark_episodes_consolidated(self, memory_ids: List[str]):
        """Mark episodic memories as consolidated."""
        if not self.client or not memory_ids:
            return
        
        for memory_id in memory_ids:
            self.client.table('episodic_memories').update({
                'consolidated': True,
                'consolidated_at': datetime.utcnow().isoformat()
            }).eq('memory_id', memory_id).execute()
    
    async def store_semantic_memory(self, user_id: str, semantic_insight: Dict):
        """Store compressed semantic memory."""
        if not self.client:
            return
        
        data = {
            'user_id': user_id,
            'insight_type': semantic_insight.get('type'),
            'content': semantic_insight.get('content'),
            'confidence': semantic_insight.get('confidence'),
            'source_episodes': semantic_insight.get('source_episodes', []),
            'created_at': datetime.utcnow().isoformat()
        }
        
        self.client.table('semantic_memories').insert(data).execute()
    
    # NEW: Adversary Learning Operations
    async def log_adversary_objection(
        self,
        user_id: str,
        prediction_data: Dict,
        debate_result: Dict
    ):
        """Log adversary objections for strategic learning."""
        if not self.client:
            return
        
        data = {
            'user_id': user_id,
            'predicted_state': prediction_data.get('state_type'),
            'objection': debate_result.get('adversary_concerns', 'unknown'),
            'debate_rounds': debate_result.get('debate_rounds', 0),
            'intervention_type': prediction_data.get('optimal_intervention'),
            'logged_at': datetime.utcnow().isoformat()
        }
        
        self.client.table('adversary_objections').insert(data).execute()
    
    async def get_adversary_history(self, user_id: str, state_type: str = None) -> List[Dict]:
        """Get history of adversary objections for a user."""
        if not self.client:
            return []
        
        query = self.client.table('adversary_objections').select('*').eq(
            'user_id', user_id
        )
        
        if state_type:
            query = query.eq('predicted_state', state_type)
        
        result = query.order('logged_at', desc=True).limit(50).execute()
        return result.data if result.data else []
    
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
    v1.4: Added component-specific query methods.
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
    
    # NEW: Component-specific convenience methods
    async def get_user_belief_model(self, user_id: str) -> Optional[Dict]:
        """Get user's belief model from Supabase."""
        if not self.supabase_store:
            return None
        return await self.supabase_store.get_user_beliefs(user_id)
    
    async def get_pending_mao_recommendations(self, user_id: str) -> List[Dict]:
        """Get recommendations waiting for MAO debate."""
        if not self.supabase_store:
            return []
        return await self.supabase_store.get_pending_recommendations(user_id)
    
    async def get_user_stress_level(self, user_id: str) -> float:
        """Get current stress level for ethical checks."""
        if not self.supabase_store:
            return 0.0
        
        metrics = await self.supabase_store.get_recent_stress_metrics(user_id, hours=6)
        return metrics.get('average_stress', 0.0)


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
            
            # NEW: Check new component tables
            if status["supabase"]:
                component_tables = [
                    'intervention_recommendations',
                    'blocked_interventions',
                    'adversary_objections',
                    'belief_networks',
                    'cognitive_dissonance_logs'
                ]
                
                missing = []
                for table in component_tables:
                    try:
                        store.client.table(table).select('*').limit(1).execute()
                    except Exception:
                        missing.append(table)
                
                if missing:
                    status["warnings"] = f"Missing component tables: {', '.join(missing)}"
                    
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
    if settings.supabase_url:
        store = get_behavioral_store()
        if store:
            # Verify tables exist (they should be created via migrations)
            try:
                await store.health_check()
                print("Supabase behavioral store verified")
            except Exception as e:
                print(f"Supabase verification warning: {e}")
