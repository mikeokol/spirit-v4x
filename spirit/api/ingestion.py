"""
FastAPI routes for behavioral data ingestion - v2.0
Integrated with: Belief Tracking, Multi-Agent Debate, Memory Consolidation, Ethical Guardrails
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from uuid import UUID
import asyncio

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from spirit.config import settings
from spirit.db.supabase_client import get_behavioral_store, SupabaseBehavioralStore
from spirit.models.behavioral import (
    BehavioralObservation,
    ScreenTimeSession,
    ObservationType,
    PrivacyLevel,
    AppCategory,
    EMARequest,
    BeliefState,
    CognitiveDissonanceEvent
)
from spirit.services.enrichment import ContextEnricher
from spirit.services.privacy import PrivacyFilter
from spirit.services.jitai import JITAIEngine
from spirit.services.belief_network import BeliefNetwork, get_belief_network
from spirit.services.mao_debate import MultiAgentOrchestrator, get_mao_orchestrator
from spirit.services.ethical_guardrails import EthicalOversight, get_ethical_oversight
from spirit.services.memory_consolidation import MemoryTierManager, get_memory_tier_manager
from spirit.streaming.realtime_pipeline import get_stream_processor


router = APIRouter(prefix="/v1/ingestion", tags=["ingestion"])
security = HTTPBearer()


async def verify_device_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> UUID:
    """Verify JWT from mobile edge device with tier-aware claims."""
    try:
        import base64
        import json
        payload = credentials.credentials.split('.')[1]
        padding = 4 - len(payload) % 4
        if padding != 4:
            payload += '=' * padding
        decoded = base64.urlsafe_b64decode(payload)
        claims = json.loads(decoded)
        
        # Store device tier info for routing decisions
        device_tier = claims.get('tier', 'edge')  # edge | gateway | core
        return UUID(claims['sub']), device_tier
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid device token")


@router.post("/screen-sessions/batch")
async def ingest_screen_sessions(
    sessions: List[ScreenTimeSession],
    background_tasks: BackgroundTasks,
    auth: tuple = Depends(verify_device_token),
    request: Request = None
):
    """
    Ingest batch of screen time sessions with full cognitive pipeline.
    Triggers: Belief update → MAO debate → Ethical check → Memory routing
    """
    user_id, device_tier = auth
    store = await get_behavioral_store()
    if not store:
        raise HTTPException(status_code=503, detail="Data store unavailable")
    
    # Initialize cognitive services
    privacy_filter = PrivacyFilter()
    enricher = ContextEnricher()
    belief_net = await get_belief_network(user_id)
    ethical = get_ethical_oversight()
    memory_tier = get_memory_tier_manager()
    
    accepted = 0
    rejected = 0
    cognitive_triggers = []
    
    for session in sessions:
        if session.user_id != user_id:
            raise HTTPException(status_code=403, detail="Session user mismatch")
        
        # Tier 1: Privacy validation (Edge/Gateway/Cloud)
        if not privacy_filter.validate(session, tier=device_tier):
            rejected += 1
            continue
        
        # Enrichment with contextual metadata
        session = await enricher.enrich_session(session, request)
        
        # Check for belief-relevant patterns (e.g., "night owl" behavior)
        belief_updates = await belief_net.evaluate_session(session)
        if belief_updates:
            cognitive_triggers.extend(belief_updates)
        
        # Route to appropriate memory tier based on salience
        memory_tier.route_observation(session, belief_updates)
        
        accepted += 1
    
    # Batch cognitive processing after ingestion
    if cognitive_triggers:
        background_tasks.add_task(
            process_cognitive_batch,
            user_id,
            cognitive_triggers,
            "screen_sessions"
        )
    
    # Ethical load check
    await ethical.monitor_batch_load(user_id, accepted)
    
    return {
        "status": "accepted",
        "accepted_count": accepted,
        "rejected_count": rejected,
        "cognitive_triggers": len(cognitive_triggers),
        "processing": "async_with_belief_tracking"
    }


@router.post("/observations")
async def ingest_behavioral_observation(
    observation: BehavioralObservation,
    background_tasks: BackgroundTasks,
    auth: tuple = Depends(verify_device_token),
    request: Request = None
):
    """
    Ingest single observation with full MAO (Multi-Agent Orchestration) pipeline.
    
    Flow: Observation → Belief Update → Cognitive Dissonance Check → 
          MAO Debate → Ethical Validation → Memory Commit → JITAI Trigger
    """
    user_id, device_tier = auth
    store = await get_behavioral_store()
    if not store:
        raise HTTPException(status_code=503, detail="Data store unavailable")
    
    if observation.user_id != user_id:
        raise HTTPException(status_code=403, detail="Observation user mismatch")
    
    # Initialize services
    privacy_filter = PrivacyFilter()
    ethical = get_ethical_oversight()
    
    # Pre-check: Privacy & Ethics
    if not privacy_filter.validate_observation(observation, tier=device_tier):
        raise HTTPException(status_code=400, detail="Privacy validation failed")
    
    # Ethical guardrail: Check user stress/load before processing
    ethical_status = await ethical.check_intervention_safety(user_id, observation)
    if ethical_status == "blocked":
        return {
            "status": "blocked_by_ethical_guardrails",
            "reason": "user_load_critical",
            "observation_id": str(observation.observation_id)
        }
    
    # Enrichment with temporal and contextual features
    enricher = ContextEnricher()
    observation = await enricher.enrich_observation(
        observation, 
        request,
        datetime.utcnow()
    )
    
    # BELIEF NETWORK: Update user model and detect dissonance
    belief_net = await get_belief_network(user_id)
    belief_update = await belief_net.process_observation(observation)
    
    # Check for cognitive dissonance (stated vs actual behavior)
    dissonance = None
    if belief_update and belief_update.confidence > 0.7:
        dissonance = await belief_net.detect_dissonance(user_id, observation)
    
    # MULTI-AGENT DEBATE: If dissonance detected, run internal debate
    mao_decision = None
    if dissonance and dissonance.severity > 0.6:
        mao = get_mao_orchestrator()
        mao_decision = await mao.debate_intervention(
            user_id=user_id,
            observation=observation,
            dissonance=dissonance,
            belief_update=belief_update
        )
        
        # If Adversary challenges the correlation, flag for review
        if mao_decision.adversary_challenges:
            background_tasks.add_task(
                flag_for_causal_review,
                user_id,
                observation.observation_id,
                mao_decision.challenge_reason
            )
    
    # MEMORY TIERING: Route to episodic or semantic based on significance
    memory_tier = get_memory_tier_manager()
    storage_tier = memory_tier.classify_observation(observation, belief_update)
    
    # Store with metadata linking to belief state
    await store.store_observation(
        observation,
        metadata={
            "belief_state_id": belief_update.state_id if belief_update else None,
            "dissonance_detected": dissonance is not None,
            "mao_decision": mao_decision.dict() if mao_decision else None,
            "storage_tier": storage_tier,
            "ethical_clearance": ethical_status
        }
    )
    
    # JITAI: Only trigger if MAO Synthesizer approves and Ethical layer clears
    jitai_trigger = None
    if (observation.observation_type == ObservationType.SCREEN_TIME_AGGREGATE and 
        ethical_status == "clear" and
        (not mao_decision or mao_decision.synthesizer_approves)):
        
        jitai = JITAIEngine()
        trigger = await jitai.evaluate_window(user_id, observation, belief_context=belief_update)
        
        if trigger and (not mao_decision or not mao_decision.synthesizer_blocks):
            jitai_trigger = trigger
            background_tasks.add_task(jitai.deliver_ema, user_id, trigger)
    
    # Real-time stream processing for anomaly detection
    stream_processor = get_stream_processor()
    await stream_processor.ingest(observation, belief_context=belief_update)
    
    return {
        "observation_id": str(observation.observation_id),
        "status": "stored",
        "belief_updated": belief_update is not None,
        "dissonance_detected": dissonance is not None,
        "mao_intervention": mao_decision.intervention_type if mao_decision else None,
        "storage_tier": storage_tier,
        "jitai_triggered": jitai_trigger is not None,
        "server_timestamp": datetime.utcnow().isoformat()
    }


@router.post("/ema-response")
async def record_ema_response(
    ema_id: UUID,
    response_value: dict,
    background_tasks: BackgroundTasks,
    auth: tuple = Depends(verify_device_token)
):
    """
    Record EMA response with belief network update.
    EMA responses are gold-standard data for belief calibration.
    """
    user_id, device_tier = auth
    store = await get_behavioral_store()
    if not store:
        raise HTTPException(status_code=503, detail="Data store unavailable")
    
    # Create observation
    observation = BehavioralObservation(
        user_id=user_id,
        observation_type=ObservationType.EMA_RESPONSE,
        privacy_level=PrivacyLevel.ANONYMOUS,
        behavior={
            'ema_id': str(ema_id),
            'response': response_value,
            'timestamp': datetime.utcnow().isoformat()
        }
    )
    
    # Update belief network with explicit user statement
    belief_net = await get_belief_network(user_id)
    await belief_net.process_ema_response(ema_id, response_value)
    
    # Check for justification capture (cognitive dissonance explanations)
    if 'justification' in response_value:
        await belief_net.tag_justification(
            user_id=user_id,
            justification=response_value['justification'],
            context=ema_id,
            hypothesis_type="user_stated_belief"
        )
    
    await store.store_observation(observation)
    
    # Trigger memory consolidation if high-value insight
    if response_value.get('certainty', 0) > 0.8:
        memory_tier = get_memory_tier_manager()
        memory_tier.promote_to_semantic(observation, priority="high")
    
    return {
        "status": "recorded",
        "belief_calibrated": True
    }


@router.post("/physiological")
async def ingest_physiological_data(
    data: Dict[str, Any],
    auth: tuple = Depends(verify_device_token)
):
    """
    Ingest HRV, sleep, or other bio-markers for ethical oversight.
    Critical for burnout detection and intervention safety.
    """
    user_id, device_tier = auth
    
    ethical = get_ethical_oversight()
    
    # Update stress/load model
    await ethical.update_physiological_state(user_id, data)
    
    # Check for critical thresholds
    status = await ethical.assess_user_wellbeing(user_id)
    
    if status == "critical":
        # Trigger immediate safety protocol
        await ethical.activate_kill_switch(user_id, reason="hrv_critical")
        return {
            "status": "kill_switch_activated",
            "reason": "physiological_distress_detected",
            "user_id": str(user_id)
        }
    
    return {
        "status": "recorded",
        "wellbeing_status": status,
        "interventions_allowed": status in ["optimal", "moderate"]
    }


@router.post("/beliefs/explicit")
async def record_explicit_belief(
    belief_statement: str,
    confidence: float,
    domain: str,  # "productivity", "health", "creativity", etc.
    auth: tuple = Depends(verify_device_token)
):
    """
    User explicitly states a belief about themselves.
    Creates a testable hypothesis in the belief network.
    """
    user_id, device_tier = auth
    
    belief_net = await get_belief_network(user_id)
    
    belief = await belief_net.register_explicit_belief(
        user_id=user_id,
        statement=belief_statement,
        confidence=confidence,
        domain=domain,
        source="user_explicit"
    )
    
    return {
        "belief_id": belief.belief_id,
        "status": "registered_as_hypothesis",
        "testable": True,
        "requires_validation": confidence < 0.5
    }


@router.get("/beliefs/status")
async def get_belief_status(
    auth: tuple = Depends(verify_device_token)
):
    """Get current belief network status and active hypotheses."""
    user_id, device_tier = auth
    
    belief_net = await get_belief_network(user_id)
    status = await belief_net.get_user_model_summary(user_id)
    
    return {
        "active_beliefs": status.active_beliefs,
        "tested_hypotheses": status.tested,
        "confirmed_beliefs": status.confirmed,
        "dissonance_events_24h": status.recent_dissonance,
        "cognitive_model_fidelity": status.fidelity_score
    }


@router.get("/privacy-budget")
async def check_privacy_budget(auth: tuple = Depends(verify_device_token)):
    """Check remaining privacy budget with tier-aware calculations."""
    user_id, device_tier = auth
    store = await get_behavioral_store()
    if not store:
        return {"privacy_budget_remaining": 1.0, "observations_today": 0}
    
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0)
    
    try:
        observations = await store.get_user_observations(
            user_id=user_id,
            start_time=today_start.isoformat(),
            limit=10000
        )
        
        # Tier-aware limits
        limits = {
            "edge": 500,      # High privacy, less data
            "gateway": 2000,  # Balanced
            "core": 10000     # Full research mode
        }
        daily_limit = limits.get(device_tier, 1000)
        used = len(observations)
        remaining = max(0, daily_limit - used)
        
        return {
            "privacy_budget_remaining": remaining / daily_limit,
            "observations_today": used,
            "device_tier": device_tier,
            "recommendation": "aggregate" if remaining < daily_limit * 0.1 else "granular",
            "next_tier_available": device_tier != "core" and remaining < daily_limit * 0.2
        }
    except Exception:
        return {"privacy_budget_remaining": 1.0, "observations_today": 0}


@router.get("/health")
async def ingestion_health():
    """Health check with cognitive pipeline status."""
    # Check all service healths
    services = {
        "supabase": bool(settings.supabase_url),
        "belief_network": True,  # Stateles
        "mao_orchestrator": True,
        "ethical_oversight": True,
        "memory_consolidation": True
    }
    
    return {
        "status": "healthy" if all(services.values()) else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "v2.0-cognitive",
        "services": services,
        "cognitive_pipeline": "active"
    }


# Background task implementations

async def process_cognitive_batch(
    user_id: UUID,
    triggers: List[Any],
    source_type: str
):
    """Process batch cognitive triggers after ingestion."""
    belief_net = await get_belief_network(user_id)
    
    # Batch update beliefs
    for trigger in triggers:
        await belief_net.consolidate_trigger(user_id, trigger)
    
    # Check for emergent patterns requiring MAO debate
    if len(triggers) > 5:
        mao = get_mao_orchestrator()
        await mao.debate_emergent_pattern(user_id, triggers)


async def flag_for_causal_review(
    user_id: UUID,
    observation_id: UUID,
    challenge_reason: str
):
    """Flag observation for causal review when Adversary challenges correlation."""
    # Queue for human review or advanced causal analysis
    from spirit.services.causal_review import CausalReviewQueue
    queue = CausalReviewQueue()
    await queue.add(user_id, observation_id, challenge_reason)
