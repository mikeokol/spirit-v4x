"""
Spirit Behavioral Research Agent - Main Application v2.0
Continuity ledger + Behavioral research + Causal inference + Goal integration 
+ Intelligence + Memory + Proactive Agent Loop + Real-time Processing 
+ Advanced Causal Discovery + Multi-Agent Debate + Belief Network 
+ Ethical Guardrails + Memory Consolidation + Human-Centered Systems
+ Reality Filter Engine + Personal Evidence Ladder + Disproven Hypothesis Archive
+ Human Operating Model (Mechanistic hypothesis generation)
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from spirit.config import settings
from spirit.db import create_db_and_tables, verify_database_connections
from spirit.db.supabase_client import close_behavioral_store, get_behavioral_store

# Existing routers
from spirit.api import auth, goals, trajectory, strategic, anchors, calibrate, debug_trace
from spirit.api.ingestion import router as ingestion_router
from spirit.api.causal import router as causal_router
from spirit.api.behavioral_goals import router as behavioral_goals_router
from spirit.api.intelligence import router as intelligence_router
from spirit.api.memory import router as memory_router
from spirit.api.delivery import router as delivery_router
from spirit.api.proactive import router as proactive_router
from spirit.api.realtime_causal import router as realtime_causal_router
from spirit.api.belief import router as belief_router
from spirit.api.ethical import router as ethical_router
from spirit.api.onboarding import router as onboarding_router
from spirit.api.empathy import router as empathy_router

# NEW: RFE API endpoints
from spirit.api.rfe import router as rfe_router

# Core systems
from spirit.agents.proactive_loop import get_orchestrator
from spirit.streaming.realtime_pipeline import get_stream_processor


# NEW: RFE background processors
async def rfe_evidence_processor():
    """
    Background task that processes pending observations through RFE.
    Runs evidence grading, confound detection, and routes to appropriate handler.
    """
    from spirit.evidence.core import RealityFilterEngine
    
    rfe = RealityFilterEngine()
    store = get_behavioral_store()
    
    if not store:
        print("RFE processor: No Supabase, skipping")
        return
    
    while True:
        try:
            # Get unprocessed observations from last 5 minutes
            five_min_ago = (datetime.utcnow() - timedelta(minutes=5)).isoformat()
            
            pending = store.client.table('behavioral_observations').select('*').eq(
                'rfe_processed', False  # NEW: Flag for RFE processing
            ).gte('timestamp', five_min_ago).limit(100).execute()
            
            if pending.data:
                for obs in pending.data:
                    # Process through RFE
                    result = await rfe.process_observation(obs, obs['user_id'])
                    
                    # Mark as processed
                    store.client.table('behavioral_observations').update({
                        'rfe_processed': True,
                        'rfe_decision': result['action'],
                        'pel_level': result['evidence_grading']['level_value'],
                        'evidence_confidence': result['evidence_grading']['confidence'],
                        'confounds_detected': result['confound_assessment']['confounds_detected']
                    }).eq('observation_id', obs['observation_id']).execute()
                    
                    # If experiment proposed, queue it
                    if result.get('experiment_proposed'):
                        store.client.table('experiment_queue').insert({
                            'user_id': obs['user_id'],
                            'experiment_proposal': result['experiment_proposed'],
                            'proposed_at': datetime.utcnow().isoformat(),
                            'status': 'pending_user_consent'
                        }).execute()
                    
                    print(f"RFE processed {obs['observation_id']}: {result['action']}")
            
            # Process every 10 seconds
            await asyncio.sleep(10)
            
        except Exception as e:
            print(f"RFE processor error: {e}")
            await asyncio.sleep(30)


async def evidence_upgrade_monitor():
    """
    Background task that monitors evidence for potential upgrades.
    Runs daily to check if Level 2 evidence can be upgraded to Level 3+.
    """
    from spirit.evidence.core import RealityFilterEngine, EvidenceLevel
    
    rfe = RealityFilterEngine()
    store = get_behavioral_store()
    
    if not store:
        return
    
    while True:
        try:
            now = datetime.utcnow()
            
            # Run at 2 AM (before memory consolidation)
            if now.hour == 2 and now.minute < 5:
                print("Starting evidence upgrade check...")
                
                # Get Level 2 evidence older than 3 days
                three_days_ago = (now - timedelta(days=3)).isoformat()
                
                candidates = store.client.table('evidence_grading').select('*').eq(
                    'level', EvidenceLevel.CONTEXTUALIZED_PATTERN.value
                ).lt('graded_at', three_days_ago).limit(100).execute()
                
                upgraded_count = 0
                for evidence in candidates.data if candidates.data else []:
                    upgraded = await rfe.should_upgrade_evidence(
                        _dict_to_grading(evidence),
                        evidence['user_id']
                    )
                    
                    if upgraded and upgraded.level != EvidenceLevel.CONTEXTUALIZED_PATTERN:
                        upgraded_count += 1
                
                print(f"Evidence upgrade check complete: {upgraded_count} upgraded")
                
                # Sleep 2 hours to prevent double-run
                await asyncio.sleep(7200)
            else:
                await asyncio.sleep(60)
                
        except Exception as e:
            print(f"Evidence upgrade monitor error: {e}")
            await asyncio.sleep(300)


async def hypothesis_falsification_monitor():
    """
    Background task that monitors active hypotheses for falsification.
    Auto-archives falsified hypotheses to DHA.
    """
    from spirit.memory.disproven_hypothesis_archive import HypothesisFalsificationTracker
    
    store = get_behavioral_store()
    if not store:
        return
    
    while True:
        try:
            # Get active hypotheses with recent observations
            active = store.client.table('active_hypotheses').select('*').eq(
                'status', 'active'
            ).execute()
            
            for hyp in active.data if active.data else []:
                tracker = HypothesisFalsificationTracker(hyp['user_id'])
                
                # Get recent observations against this hypothesis
                recent = store.client.table('hypothesis_observations').select('*').eq(
                    'hypothesis_id', hyp['hypothesis_id']
                ).order('observed_at', desc=True).limit(5).execute()
                
                for obs in recent.data if recent.data else []:
                    result = await tracker.record_observation(
                        hyp['hypothesis_id'],
                        obs,
                        supports=obs.get('supports_hypothesis', False),
                        confidence=obs.get('confidence', 0.5)
                    )
                    
                    if result.get('status') == 'falsified':
                        print(f"Hypothesis {hyp['hypothesis_id']} falsified and archived")
                        break
            
            await asyncio.sleep(300)  # Check every 5 minutes
            
        except Exception as e:
            print(f"Hypothesis falsification monitor error: {e}")
            await asyncio.sleep(600)


# Existing schedulers (preserved)
async def memory_consolidation_scheduler():
    """Background task that runs at 3 AM to compress episodic memories."""
    from spirit.memory.consolidation import MemoryConsolidation
    
    consolidator = MemoryConsolidation()
    
    while True:
        now = datetime.utcnow()
        
        if now.hour == 3 and now.minute == 0:
            print("Starting memory consolidation (3 AM)...")
            try:
                # NEW: Apply Memory Admission Control before consolidation
                await _apply_memory_admission_control()
                
                await consolidator.consolidate_all_users()
                print("Memory consolidation complete")
            except Exception as e:
                print(f"Memory consolidation error: {e}")
            
            await asyncio.sleep(3660)
        else:
            await asyncio.sleep(60)


async def _apply_memory_admission_control():
    """NEW: Filter observations before consolidation using RFE criteria."""
    from spirit.evidence.core import MemoryAdmissionControl
    
    admission = MemoryAdmissionControl()
    store = get_behavioral_store()
    
    if not store:
        return
    
    # Get yesterday's unprocessed observations
    yesterday = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    observations = store.client.table('behavioral_observations').select('*').eq(
        'date', yesterday
    ).eq('memory_admission_processed', False).limit(10000).execute()
    
    retained = 0
    discarded = 0
    
    for obs in observations.data if observations.data else []:
        # Get evidence grading
        grading = store.client.table('evidence_grading').select('*').eq(
            'observation_id', obs['observation_id']
        ).execute()
        
        evidence_grade = grading.data[0] if grading.data else None
        
        retain, category, priority = await admission.evaluate_for_retention(
            obs, 
            evidence_grade,
            obs['user_id']
        )
        
        # Mark decision
        store.client.table('behavioral_observations').update({
            'memory_admission_processed': True,
            'retain_in_semantic_memory': retain,
            'retention_category': category,
            'retention_priority': priority
        }).eq('observation_id', obs['observation_id']).execute()
        
        if retain:
            retained += 1
        else:
            discarded += 1
    
    print(f"Memory Admission Control: {retained} retained, {discarded} discarded "
          f"({discarded/(retained+discarded)*100:.1f}% discard rate)")


async def mao_debate_processor():
    """Background task that processes pending interventions through MAO debate."""
    from spirit.agents.multi_agent_debate import MultiAgentDebate
    
    debate_system = MultiAgentDebate()
    store = get_behavioral_store()
    
    if not store:
        print("MAO debate processor: No Supabase, skipping")
        return
    
    while True:
        try:
            # NEW: Only process interventions that passed RFE
            pending = store.client.table('intervention_recommendations').select(
                'user_id'
            ).eq('status', 'pending_debate').eq(
                'rfe_approved', True  # NEW: Only RFE-approved interventions
            ).execute()
            
            if pending.data:
                user_ids = list(set([p['user_id'] for p in pending.data]))
                
                for user_id in user_ids:
                    user_pending = store.client.table('intervention_recommendations').select('*').eq(
                        'user_id', user_id
                    ).eq('status', 'pending_debate').eq('rfe_approved', True).limit(5).execute()
                    
                    for rec in user_pending.data:
                        context = {
                            'current_state': rec.get('recommended_action', 'unknown'),
                            'recent_pattern': rec.get('hypothesis', ''),
                            'goal_progress': rec.get('goal_progress', 0.5),
                            'rejection_rate': rec.get('rejection_rate', 0.0),
                            'interventions_today': len(user_pending.data),
                            # NEW: Include RFE metadata
                            'pel_level': rec.get('pel_level'),
                            'evidence_confidence': rec.get('evidence_confidence'),
                            'confounds_controlled': rec.get('confounds_controlled', [])
                        }
                        
                        debate_result = await debate_system.debate_intervention(
                            user_context=context,
                            proposed_intervention=rec.get('recommended_action'),
                            predicted_outcome={
                                'expected_improvement': rec.get('confidence', 0.5),
                                'confidence': rec.get('confidence', 0.5)
                            }
                        )
                        
                        new_status = 'approved_for_delivery' if debate_result['proceed'] else 'blocked_by_adversary'
                        
                        store.client.table('intervention_recommendations').update({
                            'status': new_status,
                            'debate_result': debate_result,
                            'mao_processed_at': datetime.utcnow().isoformat()
                        }).eq('recommendation_id', rec['recommendation_id']).execute()
                        
                        print(f"MAO debate for {user_id}: {rec['recommended_action']} -> {new_status}")
            
            await asyncio.sleep(30)
            
        except Exception as e:
            print(f"MAO debate processor error: {e}")
            await asyncio.sleep(60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle with RFE components."""
    
    print("=" * 60)
    print("SPIRIT INITIALIZING v2.0")
    print("=" * 60)
    
    # Verify database connections
    db_status = await verify_database_connections()
    print(f"Database status: {db_status}")
    
    # Initialize SQLite (existing)
    if settings.env != "prod":
        await create_db_and_tables()
        print(f"✓ SQLite initialized ({settings.env} mode)")
    
    # Initialize Supabase
    if settings.supabase_url:
        store = get_behavioral_store()
        if store and store.client:
            print("✓ Supabase behavioral store connected")
            
            try:
                store._ensure_tables()
                print("✓ Component tables verified")
            except Exception as e:
                print(f"⚠ Component tables warning: {e}")
        else:
            print("✗ Supabase connection failed - behavioral features disabled")
    
    # Initialize global proactive orchestrator
    orchestrator = get_orchestrator()
    print("✓ Global proactive orchestrator initialized")
    
    # Start real-time stream processor
    processor = get_stream_processor()
    asyncio.create_task(processor.start())
    print("✓ Real-time stream processor started")
    
    # NEW: Start RFE evidence processor (sub-second processing)
    asyncio.create_task(rfe_evidence_processor())
    print("✓ RFE evidence processor started")
    
    # NEW: Start evidence upgrade monitor
    asyncio.create_task(evidence_upgrade_monitor())
    print("✓ Evidence upgrade monitor started")
    
    # NEW: Start hypothesis falsification monitor
    asyncio.create_task(hypothesis_falsification_monitor())
    print("✓ Hypothesis falsification monitor started")
    
    # Existing processors
    asyncio.create_task(memory_consolidation_scheduler())
    print("✓ Memory consolidation scheduler started (3 AM)")
    
    asyncio.create_task(mao_debate_processor())
    print("✓ Multi-Agent Debate processor started")
    
    print("=" * 60)
    print("SPIRIT v2.0 ONLINE")
    print("Features: MAO | Belief Network | Ethical Guardrails | Memory Consolidation")
    print("NEW: RFE | PEL | DHA | HOM | Mechanistic Hypothesis Generation")
    print("=" * 60)
    
    yield
    
    # SHUTDOWN SEQUENCE
    print("\n" + "=" * 60)
    print("SPIRIT SHUTTING DOWN")
    print("=" * 60)
    
    print("Stopping stream processor...")
    processor = get_stream_processor()
    processor.stop()
    
    print("Stopping proactive loops...")
    orchestrator = get_orchestrator()
    for user_id in list(orchestrator.user_loops.keys()):
        orchestrator.stop_user_loop(user_id)
    
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
    Belief Network + Ethical Guardrails + Memory Consolidation +
    Rich Onboarding + Empathy Calibration + Agency Preservation +
    Reality Filter Engine + Personal Evidence Ladder + 
    Disproven Hypothesis Archive + Human Operating Model
    """,
    version="2.0.0",  # Major version bump for RFE integration
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
    components = {
        # Core systems
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
        "multi_agent_debate": bool(settings.openai_api_key) and bool(settings.supabase_url),
        "belief_network": bool(settings.supabase_url),
        "ethical_guardrails": bool(settings.supabase_url),
        "memory_consolidation": bool(settings.supabase_url),
        "rich_onboarding": bool(settings.supabase_url),
        "empathy_calibration": bool(settings.supabase_url),
        "agency_preservation": bool(settings.supabase_url),
        # NEW: RFE v2.0 systems
        "reality_filter_engine": bool(settings.supabase_url),
        "personal_evidence_ladder": bool(settings.supabase_url),
        "disproven_hypothesis_archive": bool(settings.supabase_url),
        "human_operating_model": True,  # Local, no external dependency
        "mechanistic_hypothesis_generation": bool(settings.openai_api_key)
    }
    
    enabled = sum(components.values())
    total = len(components)
    
    return {
        "message": "Spirit continuity ledger is running",
        "docs": "/docs",
        "version": "2.0.0",
        "system_health": f"{enabled}/{total} components enabled",
        "features": components,
        "status": "healthy" if enabled >= total * 0.7 else "degraded",
        "new_in_v2": {
            "rfe": "Reality Filter Engine - causal inference gateway",
            "pel": "Personal Evidence Ladder - structured evidence grading",
            "dha": "Disproven Hypothesis Archive - anti-pattern database",
            "hom": "Human Operating Model - mechanistic behavior generation"
        },
        "human_centered_systems": {
            "onboarding": "/v1/onboarding",
            "empathy_profile": "/v1/empathy/profile",
            "partnership_contract": "/v1/empathy/partnership-contract"
        },
        "rfe_endpoints": {
            "evidence_grading": "/v1/rfe/evidence/grade",
            "process_observation": "/v1/rfe/process",
            "experiment_design": "/v1/rfe/experiments/design",
            "mechanism_generation": "/v1/rfe/mechanisms/generate",
            "archive_check": "/v1/rfe/archive/check"
        }
    }


@app.get("/health")
async def health_check():
    """Detailed health check with database connectivity."""
    db_status = await verify_database_connections()
    
    # NEW: Check RFE system health
    rfe_status = await _check_rfe_health()
    
    return {
        "status": "healthy" if db_status.get("sqlite") and rfe_status["operational"] else "degraded",
        "version": "2.0.0",
        "databases": db_status,
        "rfe_system": rfe_status,
        "timestamp": datetime.utcnow().isoformat()
    }


async def _check_rfe_health() -> Dict[str, Any]:
    """Check RFE subsystem health."""
    store = get_behavioral_store()
    if not store:
        return {"operational": False, "reason": "no_database"}
    
    try:
        # Check recent processing
        five_min_ago = (datetime.utcnow() - timedelta(minutes=5)).isoformat()
        recent = store.client.table('rfe_decision_log').select('*', count='exact').gte(
            'processed_at', five_min_ago
        ).execute()
        
        pending = store.client.table('behavioral_observations').select('*', count='exact').eq(
            'rfe_processed', False
        ).execute()
        
        return {
            "operational": True,
            "recent_decisions_5m": recent.count if hasattr(recent, 'count') else 0,
            "pending_observations": pending.count if hasattr(pending, 'count') else 0,
            "queue_healthy": (pending.count if hasattr(pending, 'count') else 0) < 100
        }
    except Exception as e:
        return {"operational": False, "reason": str(e)}


@app.get("/continuity")
async def continuity():
    from datetime import date
    from sqlalchemy import select, func
    from spirit.db import async_session
    from spirit.models import Execution

    async with async_session() as session:
        oldest = await session.scalar(select(func.min(Execution.day)))
    return {"oldest_execution": oldest.isoformat() if oldest else None}


@app.get("/metrics")
async def system_metrics():
    """Get current system metrics for monitoring."""
    store = get_behavioral_store()
    metrics = {
        "pending_mao_debates": 0,
        "blocked_interventions_24h": 0,
        "consolidated_memories": 0,
        "active_proactive_loops": len(get_orchestrator().user_loops),
        "active_onboarding_sessions": 0,
        "completed_onboardings": 0,
        "empathy_feedback_received_24h": 0,
        # NEW: RFE metrics
        "rfe_pending_observations": 0,
        "rfe_avg_evidence_level": 0,
        "active_experiments": 0,
        "disproven_hypotheses_archived": 0,
        "mechanism_hypotheses_generated_24h": 0
    }
    
    if store and store.client:
        try:
            # Existing metrics
            pending = store.client.table('intervention_recommendations').select(
                '*', count='exact'
            ).eq('status', 'pending_debate').execute()
            metrics["pending_mao_debates"] = pending.count if hasattr(pending, 'count') else 0
            
            day_ago = (datetime.utcnow() - timedelta(hours=24)).isoformat()
            blocked = store.client.table('blocked_interventions').select(
                '*', count='exact'
            ).gte('blocked_at', day_ago).execute()
            metrics["blocked_interventions_24h"] = blocked.count if hasattr(blocked, 'count') else 0
            
            # NEW: RFE metrics
            pending_rfe = store.client.table('behavioral_observations').select(
                '*', count='exact'
            ).eq('rfe_processed', False).execute()
            metrics["rfe_pending_observations"] = pending_rfe.count if hasattr(pending_rfe, 'count') else 0
            
            avg_level = store.client.table('evidence_grading').select(
                'level'
            ).gte('graded_at', day_ago).execute()
            if avg_level.data:
                metrics["rfe_avg_evidence_level"] = round(
                    sum(r['level'] for r in avg_level.data) / len(avg_level.data), 2
                )
            
            active_exp = store.client.table('experiments').select(
                '*', count='exact'
            ).in_('status', ['running', 'scheduled']).execute()
            metrics["active_experiments"] = active_exp.count if hasattr(active_exp, 'count') else 0
            
            archived = store.client.table('disproven_hypotheses').select(
                '*', count='exact'
            ).execute()
            metrics["disproven_hypotheses_archived"] = archived.count if hasattr(archived, 'count') else 0
            
        except Exception as e:
            metrics["error"] = str(e)
    
    return metrics


# Include all routers
app.include_router(auth.router, prefix="/api", tags=["auth"])
app.include_router(goals.router, prefix="/api", tags=["goals"])
app.include_router(trajectory.router, prefix="/api", tags=["trajectory"])
app.include_router(strategic.router, prefix="/api", tags=["strategic"])
app.include_router(anchors.router, prefix="/api", tags=["anchors"])
app.include_router(calibrate.router, prefix="/api", tags=["calibrate"])
app.include_router(debug_trace.router, prefix="/api", tags=["debug"])

# Behavioral data ingestion
app.include_router(ingestion_router)

# Causal inference
app.include_router(causal_router)

# Goal-behavior integration
app.include_router(behavioral_goals_router)

# Intelligence engine (LangGraph agent)
app.include_router(intelligence_router)

# Memory system
app.include_router(memory_router)

# Delivery system
app.include_router(delivery_router)

# Proactive agent loop
app.include_router(proactive_router)

# Real-time processing
app.include_router(realtime_causal_router)

# Belief Network
app.include_router(belief_router, prefix="/api/belief", tags=["belief"])

# Ethical Guardrails
app.include_router(ethical_router, prefix="/api/ethical", tags=["ethical"])

# Human-centered systems
app.include_router(onboarding_router, prefix="/v1/onboarding", tags=["onboarding"])
app.include_router(empathy_router, prefix="/v1/empathy", tags=["empathy"])

# NEW: RFE endpoints
app.include_router(rfe_router)


# Helper for evidence grading dict conversion
def _dict_to_grading(data: Dict) -> Any:
    """Convert database dict to EvidenceGrading object."""
    from spirit.evidence.core import EvidenceLevel, EvidenceGrading
    from datetime import datetime
    
    return EvidenceGrading(
        observation_id=data['observation_id'],
        user_id=data['user_id'],
        level=EvidenceLevel(data['level']),
        confidence=data['confidence'],
        grading_reason=data['grading_reason'],
        graded_by=data['graded_by'],
        graded_at=datetime.fromisoformat(data['graded_at'].replace('Z', '+00:00')),
        level_metadata=data.get('level_metadata', {}),
        upgraded_from=EvidenceLevel(data['upgraded_from']) if data.get('upgraded_from') else None,
        upgrade_reason=data.get('upgrade_reason'),
        validation_checks_passed=data.get('validation_checks_passed', []),
        validation_checks_failed=data.get('validation_checks_failed', [])
    )
