"""
FastAPI routes for behavioral data ingestion.
Mobile screen time data endpoint with validation.
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


router = APIRouter(prefix="/v1/ingestion", tags=["ingestion"])
security = HTTPBearer()


async def verify_device_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> UUID:
    """Verify JWT from mobile edge device."""
    # TODO: Implement proper JWT validation with jose
    # For now, extract user_id from sub claim
    try:
        import base64
        import json
        payload = credentials.credentials.split('.')[1]
        # Add padding if needed
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
    store: Optional[SupabaseBehavioralStore] = Depends(get_behavioral_store),
    request: Request = None
):
    """
    Ingest batch of screen time sessions from mobile edge device.
    Edge devices aggregate raw OS events before sending.
    """
    if not store:
        raise HTTPException(status_code=503, detail="Data store unavailable")
        
    accepted = 0
    rejected = 0
    
    for session in sessions:
        if session.user_id != user_id:
            raise HTTPException(status_code=403, detail="Session user mismatch")
        
        # Basic validation
        if not session.app_name_hash or not session.app_category:
            rejected += 1
            continue
            
        # Server-side enrichment
        session.collected_at = datetime.utcnow()
        
        # Store
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
    store: Optional[SupabaseBehavioralStore] = Depends(get_behavioral_store),
    request: Request = None
):
    """
    Ingest single behavioral observation with JITAI evaluation.
    """
    if not store:
        raise HTTPException(status_code=503, detail="Data store unavailable")
        
    if observation.user_id != user_id:
        raise HTTPException(status_code=403, detail="Observation user mismatch")
    
    # Server-side timestamp for latency calculation
    server_received_at = datetime.utcnow()
    observation.processing_metadata['server_received_at'] = server_received_at.isoformat()
    observation.processing_metadata['network_latency_ms'] = (
        server_received_at - observation.timestamp
    ).total_seconds() * 1000
    
    # Simple JITAI evaluation
    if observation.observation_type == ObservationType.SCREEN_TIME_AGGREGATE:
        # Check for vulnerability signals
        behavior = observation.behavior
        if behavior.get('app_switches_5min', 0) > 5:
            # High switching = potential distraction
            pass  # TODO: Trigger EMA via background task
    
    await store.store_observation(observation)
    
    return {
        "observation_id": str(observation.observation_id),
        "status": "stored",
        "server_timestamp": server_received_at.isoformat()
    }


@router.get("/privacy-budget")
async def check_privacy_budget(user_id: UUID = Depends(verify_device_token)):
    """
    Check remaining privacy budget for edge device.
    """
    # Simple implementation - count today's observations
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
