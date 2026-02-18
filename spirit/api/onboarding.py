"""
API endpoints for Spirit's rich onboarding dialogue system.
Bootstraps user beliefs, calibrates empathy, and establishes agency partnership.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.security import HTTPBearer

from spirit.services.onboarding import RichOnboardingDialogue, OnboardingState
from spirit.db.supabase_client import get_behavioral_store
from spirit.api.auth import get_current_user
from spirit.models import User


router = APIRouter(prefix="/v1/onboarding", tags=["onboarding"])
security = HTTPBearer()

# In-memory storage for active dialogues (use Redis in production)
active_dialogues: Dict[str, RichOnboardingDialogue] = {}


@router.post("/start")
async def start_onboarding(
    initial_goal: Optional[str] = None,
    user: User = Depends(get_current_user)
):
    """
    Begin rich onboarding dialogue.
    
    This is not a form to fill out—it's a conversation to understand you
    as a partner, not just a user. Takes 10-15 minutes.
    """
    user_id = str(user.id)
    
    # Check if already has active dialogue
    if user_id in active_dialogues:
        dialogue = active_dialogues[user_id]
        return {
            "status": "already_in_progress",
            "current_phase": dialogue.state.current_phase.value,
            "progress": dialogue._calculate_progress(),
            "last_message": dialogue.state.turns[-1].ai_message if dialogue.state.turns else None
        }
    
    # Check if already completed onboarding
    store = await get_behavioral_store()
    if store:
        existing_beliefs = await store.get_user_beliefs(user_id)
        if existing_beliefs and existing_beliefs.get("onboarded_at"):
            return {
                "status": "already_completed",
                "onboarded_at": existing_beliefs.get("onboarded_at"),
                "message": "Onboarding already completed. Start using Spirit!"
            }
    
    # Start new dialogue
    dialogue = RichOnboardingDialogue(user_id)
    opening_message = await dialogue.start_dialogue(initial_goal)
    
    # Store in active dialogues
    active_dialogues[user_id] = dialogue
    
    return {
        "status": "started",
        "opening_message": opening_message,
        "current_phase": "rapport_building",
        "progress": 0.0,
        "instructions": "Respond naturally. This is a conversation, not a form. Take your time.",
        "estimated_duration": "10-15 minutes",
        "can_pause": True,
        "next_endpoint": "/v1/onboarding/respond"
    }


@router.post("/respond")
async def continue_onboarding(
    response: str,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user)
):
    """
    Continue onboarding dialogue with your response.
    
    Spirit will adapt based on what you share. There are no wrong answers—
    only honest ones that help us build a genuine partnership.
    """
    user_id = str(user.id)
    
    # Get active dialogue
    dialogue = active_dialogues.get(user_id)
    if not dialogue:
        raise HTTPException(
            status_code=404, 
            detail="No active onboarding session. Start with /start first."
        )
    
    # Process response
    result = await dialogue.process_response(response)
    
    # If completed, finalize in background
    if result.get("completed"):
        background_tasks.add_task(_finalize_onboarding, user_id, dialogue.state)
        del active_dialogues[user_id]
    
    return {
        "ai_response": result["ai_response"],
        "current_phase": result["current_phase"],
        "progress": result["progress"],
        "insights_so_far": result["insights_so_far"],
        "completed": result["completed"],
        "next_action": "respond" if not result["completed"] else "onboarding_complete"
    }


@router.get("/status")
async def get_onboarding_status(
    user: User = Depends(get_current_user)
):
    """
    Check current onboarding progress and status.
    """
    user_id = str(user.id)
    
    # Check active dialogue
    dialogue = active_dialogues.get(user_id)
    if dialogue:
        return {
            "status": "in_progress",
            "current_phase": dialogue.state.current_phase.value,
            "progress": dialogue._calculate_progress(),
            "turns_completed": len(dialogue.state.turns),
            "phases_completed": list(set(t.phase.value for t in dialogue.state.turns)),
            "empathy_profile": {
                "validation_need": dialogue.state.empathy_calibration.get("validation_needed", 0.5),
                "agency_score": dialogue.state.agency_score
            },
            "can_resume": True,
            "last_activity": dialogue.state.turns[-1].timestamp if dialogue.state.turns else None
        }
    
    # Check if completed
    store = await get_behavioral_store()
    if store:
        existing_beliefs = await store.get_user_beliefs(user_id)
        if existing_beliefs and existing_beliefs.get("onboarded_at"):
            return {
                "status": "completed",
                "onboarded_at": existing_beliefs.get("onboarded_at"),
                "beliefs_established": list(existing_beliefs.get("beliefs", {}).keys()),
                "partnership_ready": True
            }
    
    return {
        "status": "not_started",
        "message": "Begin onboarding with POST /v1/onboarding/start"
    }


@router.post("/pause")
async def pause_onboarding(
    user: User = Depends(get_current_user)
):
    """
    Pause onboarding and save progress.
    Can be resumed later with /resume.
    """
    user_id = str(user.id)
    
    dialogue = active_dialogues.get(user_id)
    if not dialogue:
        raise HTTPException(status_code=404, detail="No active onboarding to pause")
    
    # Save state to database (in production, use proper serialization)
    store = await get_behavioral_store()
    if store:
        store.client.table('onboarding_sessions').upsert({
            'user_id': user_id,
            'phases_completed': [p.value for p in set(t.phase for t in dialogue.state.turns)],
            'accumulated_beliefs': dialogue.state.accumulated_beliefs,
            'agency_score': dialogue.state.agency_score,
            'empathy_calibration': dialogue.state.empathy_calibration,
            'paused_at': datetime.utcnow().isoformat(),
            'turns_count': len(dialogue.state.turns)
        }).execute()
    
    # Remove from active (but saved)
    del active_dialogues[user_id]
    
    return {
        "status": "paused",
        "message": "Onboarding paused. Resume anytime with /resume",
        "progress_preserved": True
    }


@router.post("/resume")
async def resume_onboarding(
    user: User = Depends(get_current_user)
):
    """
    Resume a paused onboarding session.
    """
    user_id = str(user.id)
    
    # Check if already active
    if user_id in active_dialogues:
        return {
            "status": "already_active",
            "message": "Onboarding already in progress. Use /respond to continue."
        }
    
    # Load paused state
    store = await get_behavioral_store()
    if not store:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    paused_session = store.client.table('onboarding_sessions').select('*').eq(
        'user_id', user_id
    ).order('paused_at', desc=True).limit(1).execute()
    
    if not paused_session.data:
        raise HTTPException(status_code=404, detail="No paused onboarding found. Start fresh with /start")
    
    # Restore dialogue (simplified—full implementation would restore full state)
    session = paused_session.data[0]
    dialogue = RichOnboardingDialogue(user_id)
    
    # Restore state
    dialogue.state.accumulated_beliefs = session.get('accumulated_beliefs', {})
    dialogue.state.agency_score = session.get('agency_score', 0.5)
    dialogue.state.empathy_calibration = session.get('empathy_calibration', {})
    
    active_dialogues[user_id] = dialogue
    
    return {
        "status": "resumed",
        "restored_from": session.get('paused_at'),
        "progress": dialogue._calculate_progress(),
        "message": "Welcome back! Let's continue where we left off.",
        "next_action": "POST /v1/onboarding/respond"
    }


@router.post("/skip")
async def skip_onboarding(
    user: User = Depends(get_current_user)
):
    """
    Skip rich onboarding and use defaults.
    
    Not recommended—Spirit will be less effective without understanding you.
    But the choice is yours.
    """
    user_id = str(user.id)
    
    # Remove active dialogue if exists
    if user_id in active_dialogues:
        del active_dialogues[user_id]
    
    # Create minimal belief profile
    store = await get_behavioral_store()
    if store:
        store.client.table('belief_networks').upsert({
            'user_id': user_id,
            'beliefs': {
                "self_efficacy": 0.5,
                "agency_score": 0.5,
                "optimal_time": "unknown",
                "work_style": "unknown",
                "skipped_onboarding": True
            },
            "confidence": 0.3,
            "onboarded_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }).execute()
    
    return {
        "status": "skipped",
        "warning": "Spirit will use generic defaults. Effectiveness reduced by ~40%.",
        "can_restart_later": True,
        "message": "You can always complete onboarding later via /start if you change your mind."
    }


@router.get("/insights")
async def get_onboarding_insights(
    user: User = Depends(get_current_user)
):
    """
    View insights gathered during onboarding.
    Only available after completion.
    """
    user_id = str(user.id)
    
    store = await get_behavioral_store()
    if not store:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    # Get onboarding memory
    memory = store.client.table('episodic_memories').select('*').eq(
        'user_id', user_id
    ).eq('episode_type', 'onboarding_dialogue').execute()
    
    if not memory.data:
        raise HTTPException(
            status_code=404, 
            detail="No onboarding insights found. Complete onboarding first."
        )
    
    onboarding_data = memory.data[0]
    content = onboarding_data.get('content', {})
    
    return {
        "onboarded_at": onboarding_data.get('created_at'),
        "conversation_turns": content.get('turns_count', 0),
        "phases_completed": content.get('phases_completed', []),
        "key_insights": content.get('key_insights', {}),
        "agency_established": content.get('agency_established', False),
        "partnership_ready": True,
        "message": "These insights power Spirit's personalized support for you."
    }


# Helper functions

async def _finalize_onboarding(user_id: str, state: OnboardingState):
    """
    Background task to finalize onboarding and save all data.
    """
    try:
        await state._finalize_onboarding()
        print(f"Onboarding finalized for user {user_id}")
    except Exception as e:
        print(f"Error finalizing onboarding for {user_id}: {e}")
        # Log error but don't fail—user experience already complete
