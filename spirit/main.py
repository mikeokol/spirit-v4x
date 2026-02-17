"""
Spirit Behavioral Research Agent - Main Application
Continuity ledger + Behavioral research + Causal inference + Goal integration 
+ Intelligence + Memory + Proactive Agent Loop + Real-time Processing 
+ Advanced Causal Discovery + Multi-Agent Debate + Belief Network 
+ Ethical Guardrails + Memory Consolidation
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from spirit.config import settings
from spirit.db import create_db_and_tables, verify_database_connections
from spirit.db.supabase_client import close_behavioral_store, get_behavioral_store
from spirit.api import auth, goals, trajectory, strategic, anchors, calibrate, debug_trace
from spirit.api.ingestion import router as ingestion_router
from spirit.api.causal import router as causal_router
from spirit.api.behavioral_goals import router as behavioral_goals_router
from spirit.api.intelligence import router as intelligence_router
from spirit.api.memory import router as memory_router
from spirit.api.delivery import router as delivery_router
from spirit.api.proactive import router as proactive_router
from spirit.api.realtime_causal import router as realtime_causal_router
from spirit.api.belief import router as belief_router  # NEW
from spirit.api.ethical import router as ethical_router  # NEW
from spirit.agents.proactive_loop import get_orchestrator
from spirit.streaming.realtime_pipeline import get_stream_processor


# NEW: Memory consolidation scheduler
async def memory_consolidation_scheduler():
    """
    Background task that runs at 3 AM to compress episodic memories 
    into semantic insights. Runs daily.
    """
    from spirit.memory.consolidation import MemoryConsolidation
    
    consolidator = MemoryConsolidation()
    
    while True:
        now = datetime.utcnow()
        
        # Run at 3:00 AM
        if now.hour == 3 and now.minute == 0:
            print("Starting memory consolidation (3 AM)...")
            try:
                await consolidator.consolidate_all_users()
                print("Memory consolidation complete")
            except Exception as e:
                print(f"Memory consolidation error: {e}")
            
            # Sleep 61 minutes to prevent double-run
            await asyncio.sleep(3660)
        else:
            # Check every minute
            await asyncio.sleep(60)


# NEW: MAO debate processor
async def mao_debate_processor():
    """
    Background task that processes pending intervention recommendations
    through Multi-Agent Debate before they reach the proactive loop.
    """
    from spirit.agents.multi_agent_debate import MultiAgentDebate
    from spirit.db.supabase_client import get_behavioral_store
    
    debate_system = MultiAgentDebate()
    store = get_behavioral_store()
    
    if not store:
        print("MAO debate processor: No Supabase, skipping")
        return
    
    while True:
        try:
            # Get all users with pending recommendations
            pending = store.client.table('intervention_recommendations').select(
                'user_id'
            ).eq('status', 'pending_debate').execute()
            
            if pending.data:
                # Get unique user IDs
                user_ids = list(set([p['user_id'] for p in pending.data]))
                
                for user_id in user_ids:
                    # Get recommendations for this user
                    user_pending = await store.get_pending_recommendations(user_id, limit=5)
                    
                    for rec in user_pending:
                        # Build context for debate
                        context = {
                            'current_state': rec.get('recommended_action', 'unknown'),
                            'recent_pattern': rec.get('hypothesis', ''),
                            'goal_progress': 0.5,  # Would fetch from goals
                            'rejection_rate': 0.0,  # Would calculate from history
                            'interventions_today': len(user_pending)
                        }
                        
                        # Run debate
                        debate_result = await debate_system.debate_intervention(
                            user_context=context,
                            proposed_intervention=rec.get('recommended_action'),
                            predicted_outcome={
                                'expected_improvement': rec.get('confidence', 0.5),
                                'confidence': rec.get('confidence', 0.5)
                            }
                        )
                        
                        # Update recommendation status
                        if debate_result['proceed']:
                            new_status = 'approved_for_delivery'
                        else:
                            new_status = 'blocked_by_adversary'
                        
                        await store.update_recommendation_status(
                            recommendation_id=rec['recommendation_id'],
                            status=new_status,
                            debate_result=debate_result
                        )
                        
                        print(f"MAO debate for {user_id}: {rec['recommended_action']} -> {new_status}")
            
            # Process every 30 seconds
            await asyncio.sleep(30)
            
        except Exception as e:
            print(f"MAO debate processor error: {e}")
            await asyncio.sleep(60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle with new components."""
    
    print("=" * 60)
    print("SPIRIT INITIALIZING")
    print("=" * 60)
    
    # Verify database connections first
    db_status = await verify_database_connections()
    print(f"Database status: {db_status}")
    
    # Initialize SQLite (existing behavior)
    if settings.env != "prod":
        await create_db_and_tables()
        print(f"✓ SQLite initialized ({settings.env} mode)")
    
    # Initialize Supabase behavioral store if configured
    if settings.supabase_url:
        store = get_behavioral_store()
        if store and store.client:
            print("✓ Supabase behavioral store connected")
            
            # NEW: Check component tables
            try:
                store._ensure_tables()
                print("✓ Component tables verified")
            except Exception as e:
                print(f"⚠ Component tables warning: {e}")
        else:
            print("✗ Supabase URL configured but connection failed - behavioral features disabled")
    
    # Initialize global proactive orchestrator
    orchestrator = get_orchestrator()
    print("✓ Global proactive orchestrator initialized")
    
    # Start real-time stream processor
    processor = get_stream_processor()
    asyncio.create_task(processor.start())
    print("✓ Real-time stream processor started (sub-second anomaly detection)")
    
    # NEW: Start memory consolidation scheduler
    asyncio.create_task(memory_consolidation_scheduler())
    print("✓ Memory consolidation scheduler started (runs at 3 AM)")
    
    # NEW: Start MAO debate processor
    asyncio.create_task(mao_debate_processor())
    print("✓ Multi-Agent Debate processor started")
    
    print("=" * 60)
    print("SPIRIT v1.4.0 ONLINE")
    print("Features: MAO | Belief Network | Ethical Guardrails | Memory Consolidation")
    print("=" * 60)
    
    yield
    
    # SHUTDOWN SEQUENCE
    print("\n" + "=" * 60)
    print("SPIRIT SHUTTING DOWN")
    print("=" * 60)
    
    # Cleanup: stop stream processor
    print("Stopping stream processor...")
    processor = get_stream_processor()
    processor.stop()
    
    # Cleanup: stop all proactive loops
    print("Stopping proactive loops...")
    orchestrator = get_orchestrator()
    for user_id in list(orchestrator.user_loops.keys()):
        orchestrator.stop_user_loop(user_id)
    
    # Cleanup database connections
    print("Closing database connections...")
    await close_behavioral_store()
    
    print("=" * 60)
    print("SPIRIT OFFLINE")
    print("=" * 60)


app = FastAPI(
    title="Spirit",
    description="""
    Continuity ledger + Behavioral research engine + Causal inference + 
    Goal integration + Intelligence + Memory + Proactive Agent Loop + 
    Real-time Processing + Advanced Causal Discovery + Multi-Agent Debate +
    Belief Network + Ethical Guardrails + Memory Consolidation
    """,
    version="1.4.0",  # MAO + Belief Network + Ethical Guardrails + Memory Consolidation
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    """Health check with feature flags."""
    # NEW: Check component health
    components = {
        "continuity_ledger": True,
        "behavioral_ingestion": bool(settings.supabase_url),
        "causal_inference": bool(settings.supabase_url),
        "goal_integration": bool(settings.supabase_url),
        "intelligence_engine": bool(settings.openai_api_key),
        "memory_system": bool(settings.supabase_url),
        "delivery_system": bool(settings.supabase_url),
        "proactive_loop": bool(settings.supabase_url),
        "realtime_processing": bool(settings.supabase_url),
        "advanced_causal_discovery": bool(settings.openai_api_key) and bool(settings.supabase_url),
        # NEW components
        "multi_agent_debate": bool(settings.openai_api_key) and bool(settings.supabase_url),
        "belief_network": bool(settings.supabase_url),
        "ethical_guardrails": bool(settings.supabase_url),
        "memory_consolidation": bool(settings.supabase_url)
    }
    
    # Calculate system health
    enabled = sum(components.values())
    total = len(components)
    
    return {
        "message": "Spirit continuity ledger is running",
        "docs": "/docs",
        "version": "1.4.0",
        "system_health": f"{enabled}/{total} components enabled",
        "features": components,
        "status": "healthy" if enabled >= total * 0.7 else "degraded"
    }

@app.get("/health")
async def health_check():
    """Detailed health check with database connectivity."""
    db_status = await verify_database_connections()
    
    return {
        "status": "healthy" if db_status.get("sqlite") else "degraded",
        "version": "1.4.0",
        "databases": db_status,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/continuity")
async def continuity():
    from datetime import date
    from sqlalchemy import select, func
    from spirit.db import async_session
    from spirit.models import Execution

    async with async_session() as session:
        oldest = await session.scalar(select(func.min(Execution.day)))
    return {"oldest_execution": oldest.isoformat() if oldest else None}

# NEW: System metrics endpoint
@app.get("/metrics")
async def system_metrics():
    """Get current system metrics for monitoring."""
    from spirit.db.supabase_client import get_behavioral_store
    
    store = get_behavioral_store()
    metrics = {
        "pending_mao_debates": 0,
        "blocked_interventions_24h": 0,
        "consolidated_memories": 0,
        "active_proactive_loops": len(get_orchestrator().user_loops)
    }
    
    if store and store.client:
        try:
            # Pending MAO debates
            pending = store.client.table('intervention_recommendations').select(
                '*', count='exact'
            ).eq('status', 'pending_debate').execute()
            metrics["pending_mao_debates"] = pending.count if hasattr(pending, 'count') else 0
            
            # Blocked interventions (24h)
            day_ago = (datetime.utcnow() - timedelta(hours=24)).isoformat()
            blocked = store.client.table('blocked_interventions').select(
                '*', count='exact'
            ).gte('blocked_at', day_ago).execute()
            metrics["blocked_interventions_24h"] = blocked.count if hasattr(blocked, 'count') else 0
            
            # Consolidated memories
            consolidated = store.client.table('episodic_memories').select(
                '*', count='exact'
            ).eq('consolidated', True).execute()
            metrics["consolidated_memories"] = consolidated.count if hasattr(consolidated, 'count') else 0
            
        except Exception as e:
            metrics["error"] = str(e)
    
    return metrics

# Existing routers (preserved)
app.include_router(auth.router, prefix="/api", tags=["auth"])
app.include_router(goals.router, prefix="/api", tags=["goals"])
app.include_router(trajectory.router, prefix="/api", tags=["trajectory"])
app.include_router(strategic.router, prefix="/api", tags=["strategic"])
app.include_router(anchors.router, prefix="/api", tags=["anchors"])
app.include_router(calibrate.router, prefix="/api", tags=["calibrate"])
app.include_router(debug_trace.router, prefix="/api", tags=["debug"])

# NEW: Behavioral data ingestion
app.include_router(ingestion_router)

# NEW: Causal inference
app.include_router(causal_router)

# NEW: Goal-behavior integration
app.include_router(behavioral_goals_router)

# NEW: Intelligence engine (LangGraph agent)
app.include_router(intelligence_router)

# NEW: Memory system (episodic + collective)
app.include_router(memory_router)

# NEW: Delivery system (notifications)
app.include_router(delivery_router)

# NEW: Proactive agent loop (autonomous predictions & interventions)
app.include_router(proactive_router)

# NEW: Real-time processing + Advanced causal discovery
app.include_router(realtime_causal_router)

# NEW: Belief Network API
app.include_router(belief_router, prefix="/api/belief", tags=["belief"])

# NEW: Ethical Guardrails API
app.include_router(ethical_router, prefix="/api/ethical", tags=["ethical"])
