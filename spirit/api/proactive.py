"""
API endpoints for proactive agent loop management.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.security import HTTPBearer

from spirit.agents.proactive_loop import (
    get_orchestrator, 
    ProactiveScheduler,
    AutonomousExperimentRunner
)


router = APIRouter(prefix="/v1/proactive", tags=["proactive"])
security = HTTPBearer()


async def get_current_user(credentials=Depends(security)) -> int:
    import base64
    import json
    try:
        payload = credentials.credentials.split('.')[1]
        padding = 4 - len(payload) % 4
        if padding != 4:
            payload += '=' * padding
        decoded = base64.urlsafe_b64decode(payload)
        claims = json.loads(decoded)
        return int(claims['sub'])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


@router.post("/start")
async def start_proactive_loop(
    background_tasks: BackgroundTasks,
    user_id: int = Depends(get_current_user)
):
    """
    Start the autonomous proactive loop for this user.
    Spirit will now predict and intervene without waiting for input.
    """
    orchestrator = get_orchestrator()
    
    # Start in background so API returns immediately
    background_tasks.add_task(orchestrator.start_user_loop, user_id)
    
    return {
        "status": "starting",
        "user_id": user_id,
        "message": "Proactive loop initializing. First predictions in 60 seconds.",
        "features_enabled": [
            "imminent_vulnerability_detection",
            "opportunity_window_alerts",
            "predictive_risk_warnings",
            "autonomous_micro_experiments"
        ]
    }


@router.post("/stop")
async def stop_proactive_loop(
    user_id: int = Depends(get_current_user)
):
    """Stop the proactive loop for this user."""
    orchestrator = get_orchestrator()
    orchestrator.stop_user_loop(user_id)
    
    return {
        "status": "stopped",
        "user_id": user_id
    }


@router.get("/status")
async def get_proactive_status(
    user_id: int = Depends(get_current_user)
):
    """Get status of proactive loop and upcoming predictions."""
    orchestrator = get_orchestrator()
    
    is_active = user_id in orchestrator.user_loops
    
    if not is_active:
        return {
            "active": False,
            "message": "Proactive loop not running. Start to enable predictive interventions."
        }
    
    # Get scheduled checks
    scheduler = orchestrator.user_loops.get(user_id)
    scheduled = []
    if scheduler:
        for check_id, scheduled_time in scheduler.scheduled_checks.items():
            scheduled.append({
                "check_id": check_id,
                "scheduled_for": scheduled_time.isoformat(),
                "time_until": (scheduled_time - datetime.utcnow()).total_seconds() / 60  # minutes
            })
    
    return {
        "active": True,
        "user_id": user_id,
        "loop_running": scheduler.running if scheduler else False,
        "scheduled_interventions": len(scheduled),
        "upcoming": sorted(scheduled, key=lambda x: x['scheduled_for'])[:5],
        "experiment_runner_active": user_id in orchestrator.experiment_runners
    }


@router.get("/predictions")
async def get_current_predictions(
    horizon: str = "all",  # imminent, short, medium, long, all
    user_id: int = Depends(get_current_user)
):
    """
    Get current predictions for this user.
    Shows what Spirit sees coming.
    """
    orchestrator = get_orchestrator()
    
    if user_id not in orchestrator.user_loops:
        raise HTTPException(status_code=400, detail="Proactive loop not active")
    
    # Generate fresh predictions
    scheduler = orchestrator.user_loops[user_id]
    predictions = await scheduler._generate_predictions()
    
    # Filter by horizon if requested
    if horizon != "all":
        predictions = [p for p in predictions if p.horizon.value == horizon]
    
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "predictions": [
            {
                "horizon": p.horizon.value,
                "predicted_time": p.predicted_time.isoformat(),
                "state_type": p.state_type,
                "confidence": p.confidence,
                "predicted_behavior": p.predicted_behavior,
                "optimal_intervention": p.optimal_intervention,
                "expected_outcome_if_intervene": p.expected_outcome_if_intervene,
                "expected_outcome_if_ignore": p.expected_outcome_if_ignore
            }
            for p in predictions
        ],
        "overall_risk_score": max([p.confidence for p in predictions if p.state_type == "risk"], default=0),
        "opportunity_score": max([p.confidence for p in predictions if p.state_type == "opportunity"], default=0)
    }


@router.post("/feedback/prediction-accuracy")
async def report_prediction_accuracy(
    prediction_id: str,
    was_accurate: bool,
    what_actually_happened: str,
    user_id: int = Depends(get_current_user)
):
    """
    User feedback on whether a prediction was accurate.
    Improves future predictions.
    """
    store = await get_behavioral_store()
    if not store:
        raise HTTPException(status_code=503, detail="Store unavailable")
    
    # Log accuracy
    store.client.table('prediction_feedback').insert({
        'feedback_id': str(uuid4()),
        'user_id': str(user_id),
        'prediction_id': prediction_id,
        'was_accurate': was_accurate,
        'actual_outcome': what_actually_happened,
        'reported_at': datetime.utcnow().isoformat()
    }).execute()
    
    # Trigger model update if enough feedback
    feedback_count = len(store.client.table('prediction_feedback').select('*').eq(
        'user_id', str(user_id)
    ).execute().data or [])
    
    return {
        "recorded": True,
        "total_feedback_given": feedback_count,
        "message": "Thank you. Your feedback improves Spirit's predictions."
    }


@router.get("/experiments")
async def get_active_experiments(
    user_id: int = Depends(get_current_user)
):
    """
    Get experiments Spirit is currently running on your behalf.
    """
    store = await get_behavioral_store()
    if not store:
        return {"experiments": []}
    
    experiments = store.client.table('proactive_experiments').select('*').eq(
        'user_id', str(user_id)
    ).eq('status', 'running').execute()
    
    return {
        "experiments": experiments.data if experiments.data else [],
        "message": "Spirit continuously runs micro-experiments to find what works best for you."
    }
