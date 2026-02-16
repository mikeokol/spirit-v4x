"""
FastAPI routes for behavioral data ingestion.
Updated to use privacy filter and enrichment services.
"""

from datetime import datetime, timedelta
from typing import Optional, List
from uuid import UUID

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
    EMARequest
)
from spirit.services.enrichment import ContextEnricher
from spirit.services.privacy import PrivacyFilter
from spirit.services.jitai import JITAIEngine


router = APIRouter(prefix="/v1/ingestion", tags=["ingestion"])
security = HTTPBearer()


async def verify_device_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> UUID:
    """Verify JWT from mobile edge device."""
    try:
        import base64
        import json
        payload = credentials.credentials.split('.')[1]
        padding = 4 - len(payload) % 4
        if padding != 4:
            payload += '=' * padding
        decoded = base64.urlsafe_b64decode(payload)
        claims = json.loads(decoded)
        return UUID(claims['sub'])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid device token")


@router.post("/screen-sessions/batch")
async def ingest_screen_sessions(
    sessions: List[ScreenTimeSession],
    background_tasks: BackgroundTasks,
    user_id: UUID = Depends(verify_device_token),
    request: Request = None
):
    """Ingest batch of screen time sessions from mobile edge device."""
    store = await get_behavioral_store()
    if not store:
        raise HTTPException(status_code=503, detail="Data store unavailable")
    
    privacy_filter = PrivacyFilter()
    enricher = ContextEnricher()
    
    accepted = 0
    rejected = 0
    
    for session in sessions:
        if session.user_id != user_id:
            raise HTTPException(status_code=403, detail="Session user mismatch")
        
        # Privacy validation
        if not privacy_filter.validate(session):
            rejected += 1
            continue
        
        # Enrichment
        session = await enricher.enrich_session(session, request)
        
        # Store async
        background_tasks.add_task(store.store_screen_session, session)
        accepted += 1
    
    return {
        "status": "accepted",
        "accepted_count": accepted,
        "rejected_count": rejected,
        "processing": "async"
    }


@router.post("/observations")
async def ingest_behavioral_observation(
    observation: BehavioralObservation,
    background_tasks: BackgroundTasks,
    user_id: UUID = Depends(verify_device_token),
    request: Request = None
):
    """Ingest single behavioral observation with JITAI evaluation."""
    store = await get_behavioral_store()
    if not store:
        raise HTTPException(status_code=503, detail="Data store unavailable")
    
    if observation.user_id != user_id:
        raise HTTPException(status_code=403, detail="Observation user mismatch")
    
    privacy_filter = PrivacyFilter()
    if not privacy_filter.validate_observation(observation):
        raise HTTPException(status_code=400, detail="Privacy validation failed")
    
    # Enrichment
    enricher = ContextEnricher()
    observation = await enricher.enrich_observation(
        observation, 
        request,
        datetime.utcnow()
    )
    
    # JITAI evaluation
    if observation.observation_type == ObservationType.SCREEN_TIME_AGGREGATE:
        jitai = JITAIEngine()
        trigger = await jitai.evaluate_window(user_id, observation)
        if trigger:
            background_tasks.add_task(jitai.deliver_ema, user_id, trigger)
    
    await store.store_observation(observation)
    
    return {
        "observation_id": str(observation.observation_id),
        "status": "stored",
        "server_timestamp": datetime.utcnow().isoformat()
    }


@router.post("/ema-response")
async def record_ema_response(
    ema_id: UUID,
    response_value: dict,
    user_id: UUID = Depends(verify_device_token)
):
    """Record user response to EMA."""
    store = await get_behavioral_store()
    if not store:
        raise HTTPException(status_code=503, detail="Data store unavailable")
    
    observation = BehavioralObservation(
        user_id=user_id,
        observation_type=ObservationType.EMA_RESPONSE,
        privacy_level=PrivacyLevel.ANONYMOUS,
        behavior={
            'ema_id': str(ema_id),
            'response': response_value
        }
    )
    
    await store.store_observation(observation)
    return {"status": "recorded"}


@router.get("/privacy-budget")
async def check_privacy_budget(user_id: UUID = Depends(verify_device_token)):
    """Check remaining privacy budget for edge device."""
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
        
        daily_limit = 1000
        used = len(observations)
        remaining = max(0, daily_limit - used)
        
        return {
            "privacy_budget_remaining": remaining / daily_limit,
            "observations_today": used,
            "recommendation": "aggregate" if remaining < 100 else "granular"
        }
    except Exception:
        return {"privacy_budget_remaining": 1.0, "observations_today": 0}


@router.get("/health")
async def ingestion_health():
    """Health check for the ingestion pipeline."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "v4x",
        "supabase_configured": bool(settings.supabase_url)
    }
