"""
API endpoints for Spirit's predictive intelligence and scientific reasoning.
Powered by LangGraph agents.
v1.4: Integrated with MAO debate queue and Belief Network insights.
"""

from datetime import datetime, timedelta
from typing import Optional, List
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.security import HTTPBearer

from spirit.agents.behavioral_scientist import (
    BehavioralScientistAgent,
    PredictiveEngine,
    InterventionOptimizer
)
from spirit.db.supabase_client import get_behavioral_store
from spirit.memory.episodic_memory import EpisodicMemorySystem
from langchain_core.messages import SystemMessage, HumanMessage  # NEW: Missing import


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
    
    v1.4: Now queues for MAO debate instead of direct delivery.
    This ensures all interventions are adversary-checked before reaching user.
    """
    agent = BehavioralScientistAgent(user_id)
    
    # Run the LangGraph pipeline
    result = await agent.process_observation(observation)
    
    # NEW: If intervention recommended, queue for MAO debate (not direct delivery)
    if result.get("intervention_queued"):
        # Store in recommendation queue for MAO processor to pick up
        store = await get_behavioral_store()
        if store:
            store.client.table('intervention_recommendations').insert({
                'recommendation_id': str(uuid4()),
                'user_id': str(user_id),
                'hypothesis': result["hypothesis"],
                'confidence': result["confidence"],
                'recommended_action': result["recommended_action"],
                'dissonance_detected': result.get("dissonance_detected", False),
                'belief_alignment': result.get("belief_alignment"),
                'reasoning': result["reasoning"],
                'source_observation': observation,
                'created_at': datetime.utcnow().isoformat(),
                'status': 'pending_debate'
            }).execute()
        
        # Background task to check if MAO approves quickly (for urgent cases)
        if result.get("dissonance_detected"):
            # High priority: check debate result faster
            background_tasks.add_task(
                _check_debate_urgent,
                user_id,
                result
            )
    
    return {
        "analysis_id": str(uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "scientific_reasoning": result["reasoning"],
        "hypothesis": result["hypothesis"],
        "confidence": result["confidence"],
        "recommended_intervention": result["recommended_action"],
        "intervention_queued": result["intervention_queued"],
        # NEW: Include belief network status
        "dissonance_detected": result.get("dissonance_detected", False),
        "belief_alignment": result.get("belief_alignment"),
        "mao_status": "pending_debate" if result.get("intervention_queued") else "not_required"
    }


@router.get("/predict/goal/{goal_id}")
async def predict_goal_achievement(
    goal_id: UUID,
    horizon_days: int = 7,
    user_id: UUID = Depends(get_current_user)
):
    """
    Predict probability of achieving a goal based on current behavioral trajectory.
    v1.4: Now includes belief-aware predictions.
    """
    engine = PredictiveEngine(user_id)
    prediction = await engine.predict_goal_outcome(goal_id, horizon_days)
    
    if "error" in prediction:
        raise HTTPException(status_code=400, detail=prediction["error"])
    
    # NEW: Add recommendation based on belief alignment
    if prediction.get("belief_alignment"):
        belief_gap = prediction["belief_alignment"].get("self_efficacy", 0.5) - prediction.get("current_progress", 0)
        
        if belief_gap > 0.3:
            prediction["recommendation"] = "address_overconfidence"
            prediction["warning"] = "User's confidence exceeds current trajectory. Risk of discouragement."
        elif belief_gap < -0.2:
            prediction["recommendation"] = "build_self_efficacy"
            prediction["opportunity"] = "User underestimates capability. Hidden potential detected."
    
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
    v1.4: Now filters by belief compatibility before optimization.
    """
    optimizer = InterventionOptimizer(user_id)
    result = await optimizer.optimize_intervention(context, available_interventions)
    
    # NEW: If interventions were filtered due to beliefs, explain why
    if result.get("filtered_out"):
        result["explanation"] = (
            f"Removed {len(result['filtered_out'])} interventions that conflict "
            f"with current user beliefs to prevent rebound."
        )
    
    return result


@router.get("/scientific-report")
async def get_scientific_report(
    days: int = 7,
    include_beliefs: bool = True,  # NEW: Option to include belief analysis
    user_id: UUID = Depends(get_current_user)
):
    """
    Generate a scientific report on user's behavioral patterns and
    intervention effectiveness.
    v1.4: Now includes belief network analysis and MAO debate statistics.
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
    
    # NEW: Get belief network data
    belief_data = None
    if include_beliefs:
        belief_result = store.client.table('belief_networks').select('*').eq(
            'user_id', str(user_id)
        ).order('updated_at', desc=True).limit(1).execute()
        belief_data = belief_result.data[0] if belief_result.data else None
    
    # NEW: Get MAO debate statistics
    debate_stats = store.client.table('intervention_recommendations').select('*').eq(
        'user_id', str(user_id)
    ).gte('created_at', start).execute()
    
    approved = sum(1 for d in debate_stats.data if d.get('status') == 'approved_for_delivery') if debate_stats.data else 0
    blocked = sum(1 for d in debate_stats.data if d.get('status') == 'blocked_by_adversary') if debate_stats.data else 0
    
    # Generate insights with LLM
    from spirit.agents.behavioral_scientist import ChatOpenAI
    from spirit.config import settings
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key)
    
    # NEW: Enhanced prompt with belief and MAO data
    belief_context = ""
    if belief_data:
        belief_context = f"""
        User's current beliefs: {belief_data.get('beliefs', {})}
        Recent cognitive dissonance events: {
            store.client.table('cognitive_dissonance_logs').select('*').eq(
                'user_id', str(user_id)
            ).gte('detected_at', start).execute().data or []
        }
        """
    
    messages = [
        SystemMessage(content="""
        You are a behavioral scientist writing a progress report.
        Be rigorous but accessible. Highlight key findings and actionable insights.
        
        If belief-reality gaps exist, suggest gentle intervention strategies
        that respect user identity while nudging toward data-driven insights.
        """),
        HumanMessage(content=f"""
        Observations (n={len(observations)}): {observations[-20:]}
        Validated hypotheses: {[h for h in hypotheses if h.get('last_validated_at')]}
        Falsified hypotheses: {[h for h in hypotheses if h.get('falsified')]}
        
        MAO Debate Statistics:
        - Interventions proposed: {len(debate_stats.data) if debate_stats.data else 0}
        - Approved by adversary: {approved}
        - Blocked (safety): {blocked}
        - Approval rate: {approved / (approved + blocked) if (approved + blocked) > 0 else 0:.1%}
        
        {belief_context}
        
        Write a scientific report covering:
        1. Behavioral patterns observed
        2. Causal relationships discovered
        3. Intervention effectiveness (including MAO validation rates)
        4. Belief-reality alignment (if applicable)
        5. Recommendations for next period
        """)
    ]
    
    response = llm.invoke(messages)
    
    return {
        "report_period_days": days,
        "observations_analyzed": len(observations),
        "active_hypotheses": len([h for h in hypotheses if not h.get('falsified')]),
        # NEW: Include MAO and belief metrics
        "mao_statistics": {
            "proposed": len(debate_stats.data) if debate_stats.data else 0,
            "approved": approved,
            "blocked": blocked,
            "approval_rate": approved / (approved + blocked) if (approved + blocked) > 0 else 0
        },
        "belief_analysis": {
            "current_beliefs": belief_data.get('beliefs') if belief_data else None,
            "dissonance_events": len(store.client.table('cognitive_dissonance_logs').select('*').eq(
                'user_id', str(user_id)
            ).gte('detected_at', start).execute().data or [])
        } if include_beliefs else None,
        "report": response.content,
        "generated_at": datetime.utcnow().isoformat()
    }


@router.get("/beliefs")
async def get_user_belief_model(
    user_id: UUID = Depends(get_current_user)
):
    """
    NEW: Get user's current belief model.
    Shows what Spirit understands about the user's self-model.
    """
    store = await get_behavioral_store()
    if not store:
        raise HTTPException(status_code=503, detail="Data store unavailable")
    
    belief_model = await store.get_user_beliefs(str(user_id))
    
    if not belief_model:
        return {
            "status": "no_data",
            "message": "Insufficient data to build belief model. Continue using Spirit."
        }
    
    # Get recent dissonance events
    dissonance_logs = store.client.table('cognitive_dissonance_logs').select('*').eq(
        'user_id', str(user_id)
    ).order('detected_at', desc=True).limit(5).execute()
    
    return {
        "belief_model": belief_model.get('beliefs', {}),
        "last_updated": belief_model.get('updated_at'),
        "confidence": belief_model.get('confidence', 0.5),
        "recent_dissonance": dissonance_logs.data if dissonance_logs.data else [],
        "active_hypotheses_being_tested": belief_model.get('active_hypotheses', [])
    }


@router.post("/test-belief")
async def test_user_belief(
    belief_statement: str,
    user_id: UUID = Depends(get_current_user)
):
    """
    NEW: Manually test a user belief against recent behavioral data.
    Useful for debugging belief-reality gaps.
    """
    from spirit.agents.belief_network import CognitiveDissonanceDetector
    
    detector = CognitiveDissonanceDetector(user_id)
    
    # Get recent observations
    store = await get_behavioral_store()
    if not store:
        raise HTTPException(status_code=503, detail="Data store unavailable")
    
    week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
    observations = await store.get_user_observations(
        user_id=user_id,
        start_time=week_ago,
        limit=1000
    )
    
    # Test the belief
    test_result = await detector.test_specific_belief(
        belief_statement=belief_statement,
        observations=observations
    )
    
    return {
        "belief_tested": belief_statement,
        "data_supports_belief": test_result.get('supported', False),
        "confidence": test_result.get('confidence', 0),
        "contradictory_evidence": test_result.get('contradictions', []),
        "recommendation": (
            "Belief aligns with data" if test_result.get('supported') 
            else "Consider belief adjustment intervention"
        )
    }


# NEW: Background task for urgent dissonance cases
async def _check_debate_urgent(user_id: UUID, analysis_result: dict):
    """
    For high-priority dissonance detections, check if MAO approves quickly.
    If adversary blocks, escalate to human review.
    """
    import asyncio
    
    # Wait 60 seconds for MAO processor to handle
    await asyncio.sleep(60)
    
    store = await get_behavioral_store()
    if not store:
        return
    
    # Check if this recommendation was blocked
    rec_id = analysis_result.get("recommendation_id")
    if not rec_id:
        return
    
    result = store.client.table('intervention_recommendations').select('*').eq(
        'recommendation_id', rec_id
    ).execute()
    
    if result.data and result.data[0].get('status') == 'blocked_by_adversary':
        # Log for strategic review - belief challenge was too risky
        store.client.table('escalated_belief_challenges').insert({
            'user_id': str(user_id),
            'belief': analysis_result.get('belief_alignment', {}).get('user_belief'),
            'adversary_objection': result.data[0].get('debate_result', {}).get('adversary_concerns'),
            'escalated_at': datetime.utcnow().isoformat(),
            'requires_human_review': True
        }).execute()
        
        print(f"URGENT: Belief challenge blocked for {user_id}, escalated to human review")


async def _deliver_intervention(user_id: UUID, action: str, hypothesis: str):
    """Background task to deliver intervention to user."""
    # DEPRECATED: Now handled by MAO debate processor
    # Kept for backward compatibility
    print(f"Intervention {action} queued for {user_id} (MAO will process)")
