"""
API routes for causal inference and experiment management.
"""

from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer

from spirit.services.causal_inference import CausalInferenceEngine, ExperimentDesigner
from spirit.models.behavioral import UserCausalHypothesis


router = APIRouter(prefix="/v1/causal", tags=["causal_inference"])
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


@router.post("/analyze")
async def analyze_causal_relationship(
    cause_variable: str,
    effect_variable: str,
    lag_hours: int = 8,
    user_id: UUID = Depends(get_current_user)
):
    """
    Trigger causal analysis between two variables.
    
    Example:
    - cause: "evening_social_media_minutes"
    - effect: "next_morning_focus_score"
    - lag: 10 (hours)
    """
    engine = CausalInferenceEngine(user_id)
    hypothesis = await engine.analyze_variable_pair(
        cause_var=cause_variable,
        effect_var=effect_variable,
        lag_hours=lag_hours
    )
    
    if not hypothesis:
        raise HTTPException(
            status_code=400,
            detail="Insufficient data for analysis (need 20+ observations)"
        )
    
    return {
        "hypothesis_id": str(hypothesis.hypothesis_id),
        "cause": hypothesis.cause_variable,
        "effect": hypothesis.effect_variable,
        "effect_size": hypothesis.effect_size,
        "confidence_interval": hypothesis.confidence_interval,
        "n_observations": hypothesis.n_observations,
        "falsified": hypothesis.falsified,
        "significant": hypothesis.last_validated_at is not None
    }


@router.get("/hypotheses")
async def get_user_hypotheses(
    user_id: UUID = Depends(get_current_user),
    include_falsified: bool = False
):
    """Get all causal hypotheses for this user."""
    from spirit.db.supabase_client import get_behavioral_store
    
    store = await get_behavioral_store()
    if not store:
        raise HTTPException(status_code=503, detail="Store unavailable")
    
    # Query from Supabase
    result = store.client.table('causal_graph').select('*').eq(
        'user_id', str(user_id)
    ).execute()
    
    hypotheses = result.data if result.data else []
    
    if not include_falsified:
        hypotheses = [h for h in hypotheses if not h.get('falsified')]
    
    return {
        "hypotheses": hypotheses,
        "count": len(hypotheses),
        "validated": len([h for h in hypotheses if h.get('last_validated_at')])
    }


@router.post("/intervention/evaluate")
async def evaluate_intervention(
    intervention_id: UUID,
    outcome_variable: str,
    window_days: int = 7,
    user_id: UUID = Depends(get_current_user)
):
    """
    Evaluate the causal effect of a specific intervention.
    """
    engine = CausalInferenceEngine(user_id)
    result = await engine.evaluate_intervention_effect(
        intervention_id=intervention_id,
        outcome_var=outcome_variable,
        window_days=window_days
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.post("/experiment/design")
async def design_experiment(
    hypothesis_id: UUID,
    user_id: UUID = Depends(get_current_user)
):
    """
    Design an experiment to test a hypothesis.
    """
    from spirit.db.supabase_client import get_behavioral_store
    
    store = await get_behavioral_store()
    if not store:
        raise HTTPException(status_code=503, detail="Store unavailable")
    
    # Get hypothesis
    result = store.client.table('causal_graph').select('*').eq(
        'hypothesis_id', str(hypothesis_id)
    ).single().execute()
    
    if not result.data:
        raise HTTPException(status_code=404, detail="Hypothesis not found")
    
    hypothesis = UserCausalHypothesis(**result.data)
    
    # Design experiment
    designer = ExperimentDesigner(user_id)
    experiment = await designer.design_experiment(
        hypothesis=hypothesis,
        intervention_options=[
            {"name": "notification_break", "type": "EMA"},
            {"name": "focus_mode_prompt", "type": "system_intervention"}
        ]
    )
    
    return experiment
