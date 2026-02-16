"""
API endpoints for Spirit's predictive intelligence and scientific reasoning.
Powered by LangGraph agents.
"""

from datetime import datetime
from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.security import HTTPBearer

from spirit.agents.behavioral_scientist import (
    BehavioralScientistAgent,
    PredictiveEngine,
    InterventionOptimizer
)
from spirit.db.supabase_client import get_behavioral_store


router = APIRouter(prefix="/v1/intelligence", tags=["intelligence"])
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
async def analyze_observation(
    observation: dict,
    background_tasks: BackgroundTasks,
    user_id: UUID = Depends(get_current_user)
):
    """
    Process a behavioral observation through the scientific pipeline.
    Returns hypothesis, prediction, and recommended intervention.
    
    This is called automatically when new data arrives from /v1/ingestion,
    or can be called manually for testing.
    """
    agent = BehavioralScientistAgent(user_id)
    
    # Run the LangGraph pipeline
    result = await agent.process_observation(observation)
    
    # If intervention recommended, queue it
    if result.get("intervention_queued"):
        background_tasks.add_task(
            _deliver_intervention,
            user_id,
            result["recommended_action"],
            result["hypothesis"]
        )
    
    return {
        "analysis_id": str(uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "scientific_reasoning": result["reasoning"],
        "hypothesis": result["hypothesis"],
        "confidence": result["confidence"],
        "recommended_intervention": result["recommended_action"],
        "intervention_queued": result["intervention_queued"]
    }


@router.get("/predict/goal/{goal_id}")
async def predict_goal_achievement(
    goal_id: UUID,
    horizon_days: int = 7,
    user_id: UUID = Depends(get_current_user)
):
    """
    Predict probability of achieving a goal based on current behavioral trajectory.
    """
    engine = PredictiveEngine(user_id)
    prediction = await engine.predict_goal_outcome(goal_id, horizon_days)
    
    if "error" in prediction:
        raise HTTPException(status_code=400, detail=prediction["error"])
    
    return prediction


@router.post("/optimize-intervention")
async def optimize_intervention_selection(
    context: dict,
    available_interventions: List[str],
    user_id: UUID = Depends(get_current_user)
):
    """
    Use multi-armed bandit to select optimal intervention.
    Balances trying new approaches vs using proven ones.
    """
    optimizer = InterventionOptimizer(user_id)
    result = await optimizer.optimize_intervention(context, available_interventions)
    
    return result


@router.get("/scientific-report")
async def get_scientific_report(
    days: int = 7,
    user_id: UUID = Depends(get_current_user)
):
    """
    Generate a scientific report on user's behavioral patterns and
    intervention effectiveness.
    """
    store = await get_behavioral_store()
    if not store:
        raise HTTPException(status_code=503, detail="Data store unavailable")
    
    # Get recent observations
    start = (datetime.utcnow() - timedelta(days=days)).isoformat()
    observations = await store.get_user_observations(
        user_id=user_id,
        start_time=start,
        limit=10000
    )
    
    # Get causal hypotheses
    hypotheses_result = store.client.table('causal_graph').select('*').eq(
        'user_id', str(user_id)
    ).execute()
    hypotheses = hypotheses_result.data if hypotheses_result.data else []
    
    # Generate insights with LLM
    from spirit.agents.behavioral_scientist import ChatOpenAI
    from spirit.config import settings
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key)
    
    messages = [
        SystemMessage(content="""
        You are a behavioral scientist writing a progress report.
        Be rigorous but accessible. Highlight key findings and actionable insights.
        """),
        HumanMessage(content=f"""
        Observations (n={len(observations)}): {observations[-20:]}
        Validated hypotheses: {[h for h in hypotheses if h.get('last_validated_at')]}
        Falsified hypotheses: {[h for h in hypotheses if h.get('falsified')]}
        
        Write a scientific report covering:
        1. Behavioral patterns observed
        2. Causal relationships discovered
        3. Intervention effectiveness
        4. Recommendations for next period
        """)
    ]
    
    response = llm.invoke(messages)
    
    return {
        "report_period_days": days,
        "observations_analyzed": len(observations),
        "active_hypotheses": len([h for h in hypotheses if not h.get('falsified')]),
        "report": response.content,
        "generated_at": datetime.utcnow().isoformat()
    }


async def _deliver_intervention(user_id: UUID, action: str, hypothesis: str):
    """Background task to deliver intervention to user."""
    # TODO: Integrate with push notification service
    # TODO: Log intervention for causal analysis
    print(f"Delivering {action} to {user_id} based on: {hypothesis[:50]}...")
