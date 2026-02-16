"""
API routes connecting behavioral data to goals and trajectories.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import HTTPBearer

from spirit.services.goal_integration import BehavioralGoalBridge, TrajectoryBehavioralAnalyzer


router = APIRouter(prefix="/v1/behavioral-goals", tags=["behavioral_goals"])
security = HTTPBearer()


async def get_current_user(credentials=Depends(security)) -> UUID:
    """Extract user from JWT."""
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


@router.get("/goals/{goal_id}/progress")
async def get_goal_behavioral_progress(
    goal_id: UUID,
    date: Optional[datetime] = Query(None),
    user_id: UUID = Depends(get_current_user)
):
    """
    Get goal progress computed from behavioral data.
    """
    bridge = BehavioralGoalBridge(user_id)
    result = await bridge.compute_goal_progress(goal_id, date)
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result


@router.get("/goals/{goal_id}/suggestions")
async def get_goal_suggestions(
    goal_id: UUID,
    n: int = 3,
    user_id: UUID = Depends(get_current_user)
):
    """
    Get AI-generated intervention suggestions based on causal analysis.
    """
    bridge = BehavioralGoalBridge(user_id)
    suggestions = await bridge.suggest_interventions(goal_id, n)
    
    return {
        "goal_id": str(goal_id),
        "suggestions": suggestions,
        "based_on_causal_model": len(suggestions) > 0
    }


@router.get("/trajectories/{trajectory_id}/risk")
async def get_trajectory_risk_assessment(
    trajectory_id: UUID,
    horizon_days: int = 7,
    user_id: UUID = Depends(get_current_user)
):
    """
    Assess if current behavior threatens trajectory success.
    """
    analyzer = TrajectoryBehavioralAnalyzer(user_id)
    assessment = await analyzer.detect_trajectory_risk(trajectory_id, horizon_days)
    
    return assessment


@router.get("/dashboard")
async def get_behavioral_dashboard(
    user_id: UUID = Depends(get_current_user)
):
    """
    Unified dashboard: goals + behavioral metrics + risks.
    """
    from spirit.db import async_session
    from sqlalchemy import select
    from spirit.models import Goal
    
    # Get active goals
    async with async_session() as session:
        result = await session.execute(
            select(Goal).where(Goal.user_id == user_id).limit(5)
        )
        goals = result.scalars().all()
    
    # Compute progress for each
    bridge = BehavioralGoalBridge(user_id)
    goal_progress = []
    
    for goal in goals:
        progress = await bridge.compute_goal_progress(goal.id)
        goal_progress.append({
            "goal_id": str(goal.id),
            "title": goal.title,
            "progress": progress.get("progress_score"),
            "insights": progress.get("insights", [])
        })
    
    # Get trajectory risks (if any trajectory active)
    # TODO: Link to actual trajectory system
    
    return {
        "user_id": str(user_id),
        "date": datetime.utcnow().isoformat(),
        "goals": goal_progress,
        "behavioral_summary": {
            "data_available": any(g["progress"] is not None for g in goal_progress),
            "goals_at_risk": sum(1 for g in goal_progress if g["progress"] and g["progress"] < 0.3)
        }
    }
