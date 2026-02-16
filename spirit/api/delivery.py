"""
API endpoints for notification delivery and feedback collection.
"""

from datetime import datetime
from typing import Optional, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.security import HTTPBearer

from spirit.services.notification_engine import NotificationEngine, NotificationPriority, ChannelType
from spirit.services.feedback_loop import FeedbackLoopEngine, ExplicitFeedbackCollector, InterventionOutcome


router = APIRouter(prefix="/v1/delivery", tags=["delivery"])
security = HTTPBearer()


async def get_current_user(credentials=Depends(security)) -> UUID:
    import base64
    import json
    try:
        payload = credentials.credentials.split('.')[1]
        padding = 4 - len(payload) % 4
        if padding != 4:
            payload += '=' * padding
        decoded = base64.urlsafe_b64decode(payload)
        claims = json.loads(decoded)
        return UUID(claims['sub'])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


@router.post("/notify")
async def send_notification(
    title: str,
    body: str,
    priority: NotificationPriority = NotificationPriority.NORMAL,
    notification_type: str = "intervention",
    channel: Optional[ChannelType] = None,
    data: Optional[dict] = None,
    user_id: UUID = Depends(get_current_user)
):
    """
    Send notification to user through optimal channel and timing.
    """
    engine = NotificationEngine(user_id)
    
    content = {
        "title": title,
        "body": body,
        "data": data or {}
    }
    
    result = await engine.send_notification(
        content=content,
        priority=priority,
        notification_type=notification_type,
        preferred_channel=channel
    )
    
    return result


@router.get("/digest/daily")
async def get_daily_digest(
    user_id: UUID = Depends(get_current_user)
):
    """
    Generate and retrieve daily digest content.
    """
    from spirit.services.notification_engine import SmartDigestGenerator
    
    generator = SmartDigestGenerator(user_id)
    digest = await generator.generate_daily_digest()
    
    return digest


@router.post("/feedback/intervention")
async def record_intervention_feedback(
    intervention_id: str,
    response: Literal['engaged', 'dismissed', 'ignored'],
    response_time_seconds: Optional[float] = None,
    behavior_after: Optional[dict] = None,
    rating: Optional[int] = None,
    feedback_text: Optional[str] = None,
    user_id: UUID = Depends(get_current_user)
):
    """
    Record user response to an intervention.
    Closes the feedback loop for model improvement.
    """
    # Record outcome
    outcome = InterventionOutcome(
        intervention_id=intervention_id,
        user_id=str(user_id),
        intervention_type="unknown",  # Look up from DB
        delivered_at=datetime.utcnow(),  # Look up from DB
        user_response=response,
        response_time_seconds=response_time_seconds,
        behavior_change_30min=behavior_after,
        ema_response=None,
        goal_progress_delta=None,
        explicit_rating=rating,
        qualitative_feedback=feedback_text
    )
    
    engine = FeedbackLoopEngine(user_id)
    await engine.record_outcome(outcome)
    
    # Handle explicit rating if provided
    if rating:
        collector = ExplicitFeedbackCollector(user_id)
        await collector.record_rating(intervention_id, rating, feedback_text)
    
    return {
        "recorded": True,
        "feedback_id": str(uuid4()),
        "will_improve_models": True
    }


@router.get("/feedback/effectiveness")
async def get_intervention_effectiveness(
    days: int = 30,
    user_id: UUID = Depends(get_current_user)
):
    """
    Get report on which interventions work best for this user.
    """
    engine = FeedbackLoopEngine(user_id)
    report = await engine.get_intervention_effectiveness_report(days)
    
    return report


@router.get("/in-app/pending")
async def get_pending_in_app_notifications(
    user_id: UUID = Depends(get_current_user)
):
    """
    Get in-app notifications waiting to be displayed.
    Called when app opens.
    """
    from spirit.db.supabase_client import get_behavioral_store
    
    store = await get_behavioral_store()
    if not store:
        return {"notifications": []}
    
    now = datetime.utcnow().isoformat()
    
    pending = store.client.table('in_app_notifications').select('*').eq(
        'user_id', str(user_id)
    ).eq('displayed', False).gte('expires_at', now).execute()
    
    # Mark as fetched (but not yet displayed)
    notifications = []
    for n in pending.data if pending.data else []:
        notifications.append({
            "notification_id": n.get('id'),
            "title": n['content'].get('title'),
            "body": n['content'].get('body'),
            "data": n['content'].get('data'),
            "created_at": n['created_at']
        })
    
    return {
        "notifications": notifications,
        "count": len(notifications)
    }


@router.post("/in-app/{notification_id}/displayed")
async def mark_in_app_displayed(
    notification_id: str,
    user_id: UUID = Depends(get_current_user)
):
    """Mark in-app notification as displayed."""
    from spirit.db.supabase_client import get_behavioral_store
    
    store = await get_behavioral_store()
    if store:
        store.client.table('in_app_notifications').update({
            'displayed': True,
            'displayed_at': datetime.utcnow().isoformat()
        }).eq('id', notification_id).eq('user_id', str(user_id)).execute()
    
    return {"marked": True}


@router.post("/in-app/{notification_id}/engaged")
async def mark_in_app_engaged(
    notification_id: str,
    user_id: UUID = Depends(get_current_user)
):
    """Mark in-app notification as engaged (user clicked/acted)."""
    from spirit.db.supabase_client import get_behavioral_store
    
    store = await get_behavioral_store()
    if store:
        store.client.table('in_app_notifications').update({
            'engaged': True,
            'engaged_at': datetime.utcnow().isoformat()
        }).eq('id', notification_id).eq('user_id', str(user_id)).execute()
    
    # Record in feedback loop
    engine = FeedbackLoopEngine(user_id)
    # Create minimal outcome
    outcome = InterventionOutcome(
        intervention_id=notification_id,
        user_id=str(user_id),
        intervention_type="in_app",
        delivered_at=datetime.utcnow(),
        user_response='engaged',
        response_time_seconds=None,
        behavior_change_30min=None,
        ema_response=None,
        goal_progress_delta=None,
        explicit_rating=None,
        qualitative_feedback=None
    )
    await engine.record_outcome(outcome)
    
    return {"marked": True}
