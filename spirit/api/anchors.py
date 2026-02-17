"""
Reality Anchors API: Stabilizing mechanisms for goal pursuit.
v2.0: Integrated with belief networks, behavioral triggers, and environmental design.
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from spirit.db import async_session, get_behavioral_store
from spirit.models import RealityAnchor, Goal, GoalState, User
from spirit.api.auth import get_current_user
from spirit.schemas.reality_anchor import RealityAnchorSchema, AnchorTriggerType, AnchorEffectiveness
from spirit.agents.behavioral_scientist import PredictiveEngine
from spirit.memory.episodic_memory import EpisodicMemorySystem


router = APIRouter(prefix="/anchors", tags=["anchors"])


async def get_db() -> AsyncSession:
    async with async_session() as session:
        yield session


@router.post("", response_model=RealityAnchorSchema)
async def create_anchor(
    body: RealityAnchorSchema,
    environmental_context: Optional[Dict[str, Any]] = None,  # NEW: Where/when anchor applies
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a reality anchor with behavioral trigger optimization.
    v2.0: Anchors are now context-aware and belief-aligned.
    """
    goal = await db.scalar(
        select(Goal).where(Goal.user_id == user.id, Goal.state == GoalState.active)
    )
    if not goal:
        raise HTTPException(status_code=409, detail="No active goal")

    # Check for existing anchor
    existing = await db.scalar(
        select(RealityAnchor).where(RealityAnchor.goal_id == goal.id)
    )
    if existing:
        raise HTTPException(status_code=409, detail="Anchor already exists for this goal")

    # NEW: Validate anchor against user's belief model
    store = await get_behavioral_store()
    belief_alignment = {}
    if store:
        belief_model = await store.get_user_beliefs(str(user.id))
        if belief_model:
            belief_alignment = _check_anchor_belief_compatibility(
                body.anchor_type, 
                body.trigger_condition,
                belief_model.get('beliefs', {})
            )
    
    # NEW: Optimize trigger based on behavioral patterns
    optimized_trigger = await _optimize_trigger(
        user.id, 
        body.trigger_condition,
        body.trigger_type
    )

    # Create anchor with enhanced metadata
    anchor = RealityAnchor(
        goal_id=goal.id,
        anchor_type=body.anchor_type,
        trigger_type=body.trigger_type,
        trigger_condition=optimized_trigger or body.trigger_condition,
        action_cue=body.action_cue,
        reward_design=body.reward_design,
        belief_alignment_score=belief_alignment.get('score', 0.5),
        environmental_context=environmental_context or {},
        created_at=datetime.utcnow(),
        metadata={
            "belief_compatibility": belief_alignment,
            "optimization_reason": optimized_trigger.get('reason') if optimized_trigger else None,
            "expected_effectiveness": _predict_anchor_effectiveness(body, belief_alignment)
        }
    )
    
    db.add(anchor)
    await db.commit()
    await db.refresh(anchor)
    
    # NEW: Store anchor in episodic memory for pattern learning
    memory = EpisodicMemorySystem(user.id)
    await memory.store_episode(
        episode_type="anchor_created",
        content={
            "anchor_id": str(anchor.id),
            "anchor_type": body.anchor_type,
            "trigger": body.trigger_condition,
            "belief_alignment": belief_alignment.get('score')
        },
        significance=0.6
    )
    
    return anchor


@router.get("", response_model=RealityAnchorSchema | None)
async def get_anchor(
    include_effectiveness: bool = True,  # NEW: Include effectiveness prediction
    user: User = Depends(get_current_user), 
    db: AsyncSession = Depends(get_db)
):
    """
    Get active reality anchor with behavioral context and effectiveness forecast.
    v2.0: Returns anchor with predicted effectiveness based on current state.
    """
    goal = await db.scalar(
        select(Goal).where(Goal.user_id == user.id, Goal.state == GoalState.active)
    )
    if not goal:
        return None
    
    anchor = await db.scalar(
        select(RealityAnchor).where(RealityAnchor.goal_id == goal.id)
    )
    
    if not anchor:
        return None
    
    # NEW: Calculate real-time effectiveness prediction
    if include_effectiveness:
        effectiveness = await _calculate_effectiveness(anchor, user.id)
        anchor.effectiveness_forecast = effectiveness
    
    return anchor


@router.patch("/{anchor_id}/trigger")
async def record_anchor_trigger(
    anchor_id: int,
    trigger_context: Dict[str, Any],  # NEW: Context of trigger (time, location, state)
    user_response: str,  # NEW: Did user follow through? ("followed", "ignored", "delayed")
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Record when anchor was triggered and user's response.
    v2.0: Critical for learning which anchors work in which contexts.
    """
    anchor = await db.get(RealityAnchor, anchor_id)
    if not anchor or anchor.goal_id != (await _get_active_goal_id(user.id, db)):
        raise HTTPException(status_code=404, detail="Anchor not found")
    
    # Log trigger event
    store = await get_behavioral_store()
    if store:
        trigger_record = {
            'trigger_id': str(uuid4()),
            'anchor_id': anchor_id,
            'user_id': str(user.id),
            'triggered_at': datetime.utcnow().isoformat(),
            'context': trigger_context,
            'user_response': user_response,
            'day_of_week': datetime.utcnow().weekday(),
            'hour_of_day': datetime.utcnow().hour
        }
        
        store.client.table('anchor_triggers').insert(trigger_record).execute()
        
        # Update anchor statistics
        await _update_anchor_stats(anchor_id, user_response)
        
        # If user ignored, analyze why
        if user_response == "ignored":
            await _analyze_anchor_failure(anchor_id, trigger_context, user.id)
    
    # Update anchor last triggered
    anchor.last_triggered_at = datetime.utcnow()
    await db.commit()
    
    return {
        "status": "recorded",
        "response": user_response,
        "anchor_strength_updated": True
    }


@router.post("/{anchor_id}/strengthen")
async def strengthen_anchor(
    anchor_id: int,
    reinforcement_method: str,  # "stacking", "temptation_bundle", "environmental"
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    NEW: Strengthen an existing anchor using behavioral science techniques.
    """
    anchor = await db.get(RealityAnchor, anchor_id)
    if not anchor or anchor.goal_id != (await _get_active_goal_id(user.id, db)):
        raise HTTPException(status_code=404, detail="Anchor not found")
    
    # Apply reinforcement
    if reinforcement_method == "stacking":
        # Add another trigger condition (habit stacking)
        anchor.metadata = anchor.metadata or {}
        anchor.metadata["stacked_triggers"] = anchor.metadata.get("stacked_triggers", []) + [
            {"added_at": datetime.utcnow().isoformat(), "method": "stacking"}
        ]
        message = "Anchor strengthened with habit stacking"
        
    elif reinforcement_method == "temptation_bundle":
        # Pair with enjoyable activity
        anchor.metadata = anchor.metadata or {}
        anchor.metadata["temptation_bundle"] = True
        message = "Anchor paired with temptation bundling"
        
    elif reinforcement_method == "environmental":
        # Add physical environmental cue
        anchor.metadata = anchor.metadata or {}
        anchor.metadata["environmental_cue"] = True
        message = "Physical environmental cue added"
    
    anchor.strength_level = min(1.0, (anchor.strength_level or 0.5) + 0.1)
    await db.commit()
    
    return {
        "detail": message,
        "new_strength": anchor.strength_level,
        "reinforcement_method": reinforcement_method
    }


@router.get("/effectiveness-report")
async def get_anchor_effectiveness_report(
    days: int = 30,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    NEW: Comprehensive report on anchor effectiveness with behavioral insights.
    """
    goal = await db.scalar(
        select(Goal).where(Goal.user_id == user.id, Goal.state == GoalState.active)
    )
    if not goal:
        raise HTTPException(status_code=404, detail="No active goal")
    
    store = await get_behavioral_store()
    if not store:
        return {"error": "Behavioral store unavailable"}
    
    # Get trigger history
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()
    triggers = store.client.table('anchor_triggers').select('*').eq(
        'user_id', str(user.id)
    ).gte('triggered_at', since).execute()
    
    if not triggers.data:
        return {
            "period_days": days,
            "total_triggers": 0,
            "message": "No anchor triggers recorded yet. Keep using Spirit!"
        }
    
    # Calculate metrics
    total = len(triggers.data)
    followed = sum(1 for t in triggers.data if t.get('user_response') == 'followed')
    ignored = sum(1 for t in triggers.data if t.get('user_response') == 'ignored')
    delayed = sum(1 for t in triggers.data if t.get('user_response') == 'delayed')
    
    # Find optimal conditions
    optimal_conditions = _find_optimal_conditions(triggers.data)
    
    # Compare to days without anchors
    execution_rate_with = followed / total if total > 0 else 0
    execution_rate_without = await _get_baseline_execution_rate(user.id, goal.id, days, db)
    
    # Generate insight
    from spirit.agents.behavioral_scientist import ChatOpenAI
    from spirit.config import settings
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key)
    
    messages = [
        SystemMessage(content="""
        You are a behavioral designer analyzing reality anchor effectiveness.
        Provide specific, actionable insights about when and why anchors work.
        """),
        HumanMessage(content=f"""
        Anchor trigger data (last {days} days):
        - Total triggers: {total}
        - Followed: {followed} ({followed/total*100:.1f}%)
        - Ignored: {ignored} ({ignored/total*100:.1f}%)
        - Delayed: {delayed} ({delayed/total*100:.1f}%)
        
        Optimal conditions: {optimal_conditions}
        
        Comparison: {execution_rate_with:.1%} with anchors vs {execution_rate_without:.1%} without
        
        Provide 2-3 insights about what's working and how to improve.
        """)
    ]
    
    response = llm.invoke(messages)
    
    return {
        "period_days": days,
        "total_triggers": total,
        "effectiveness": {
            "follow_rate": followed / total if total > 0 else 0,
            "ignore_rate": ignored / total if total > 0 else 0,
            "delay_rate": delayed / total if total > 0 else 0
        },
        "optimal_conditions": optimal_conditions,
        "comparison": {
            "with_anchors": execution_rate_with,
            "without_anchors": execution_rate_without,
            "lift": execution_rate_with - execution_rate_without
        },
        "insights": response.content,
        "recommendations": _generate_anchor_recommendations(
            followed/total if total > 0 else 0, 
            optimal_conditions
        )
    }


@router.post("/suggest-optimal")
async def suggest_optimal_anchor(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    NEW: AI-suggested optimal anchor based on user's behavioral patterns.
    """
    goal = await db.scalar(
        select(Goal).where(Goal.user_id == user.id, Goal.state == GoalState.active)
    )
    if not goal:
        raise HTTPException(status_code=404, detail="No active goal")
    
    # Analyze user's behavioral patterns
    store = await get_behavioral_store()
    if not store:
        raise HTTPException(status_code=503, detail="Behavioral data unavailable")
    
    # Get recent behavioral data
    week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
    observations = await store.get_user_observations(
        user_id=user.id,
        start_time=week_ago,
        limit=1000
    )
    
    # Find existing routines (high-frequency behaviors)
    routines = _identify_existing_routines(observations)
    
    # Get belief model
    belief_model = await store.get_user_beliefs(str(user.id))
    
    # Generate suggestion
    from spirit.agents.behavioral_scientist import ChatOpenAI
    from spirit.config import settings
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key)
    
    messages = [
        SystemMessage(content="""
        You are a behavioral designer creating reality anchors.
        Suggest specific, actionable anchors based on user's existing routines.
        Use habit stacking: attach new behavior to existing routine.
        Consider user's beliefs about their optimal working style.
        """),
        HumanMessage(content=f"""
        Goal: {goal.text}
        Existing routines: {routines}
        User beliefs: {belief_model.get('beliefs', {}) if belief_model else 'Unknown'}
        
        Suggest 2-3 optimal reality anchors using habit stacking principles.
        Format: "After [EXISTING ROUTINE], I will [GOAL ACTION] because [REASON]"
        """)
    ]
    
    response = llm.invoke(messages)
    
    return {
        "suggested_anchors": response.content,
        "based_on_routines": routines,
        "confidence": "high" if len(routines) >= 3 else "medium"
    }


# Helper functions

def _check_anchor_belief_compatibility(anchor_type: str, trigger: str, beliefs: Dict) -> Dict:
    """Check if anchor aligns with user's beliefs."""
    score = 0.5
    concerns = []
    
    # Check if trigger conflicts with optimal time belief
    optimal_time = beliefs.get('optimal_time', 'unknown')
    if 'morning' in trigger.lower() and optimal_time == 'evening':
        score -= 0.2
        concerns.append("Trigger conflicts with user's evening preference")
    if 'evening' in trigger.lower() and optimal_time == 'morning':
        score -= 0.2
        concerns.append("Trigger conflicts with user's morning preference")
    
    # Check self-efficacy
    if beliefs.get('self_efficacy', 0.5) < 0.4:
        score -= 0.1
        concerns.append("Low self-efficacy may reduce anchor effectiveness")
    
    return {
        "score": max(0, score),
        "concerns": concerns,
        "recommendation": "Consider adjusting trigger timing" if concerns else "Anchor aligns with beliefs"
    }


async def _optimize_trigger(user_id: int, original_trigger: str, trigger_type: str) -> Optional[Dict]:
    """Optimize trigger timing based on behavioral patterns."""
    store = await get_behavioral_store()
    if not store:
        return None
    
    # Get user's most consistent routine times
    week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
    observations = await store.get_user_observations(
        user_id=user_id,
        start_time=week_ago,
        limit=1000
    )
    
    # Find most consistent time for similar triggers
    trigger_hours = {}
    for obs in observations:
        hour = datetime.fromisoformat(obs.timestamp.replace('Z', '+00:00')).hour
        trigger_hours[hour] = trigger_hours.get(hour, 0) + 1
    
    if trigger_hours:
        optimal_hour = max(trigger_hours.items(), key=lambda x: x[1])[0]
        if abs(optimal_hour - _extract_hour(original_trigger)) > 2:
            return {
                "original": original_trigger,
                "optimized": f"{original_trigger} at {optimal_hour}:00",
                "reason": f"User most active at {optimal_hour}:00 based on past 7 days",
                "confidence": trigger_hours[optimal_hour] / len(observations)
            }
    
    return None


def _extract_hour(trigger: str) -> int:
    """Extract hour from trigger string (simplified)."""
    import re
    match = re.search(r'(\d+):00', trigger)
    return int(match.group(1)) if match else 9


def _predict_anchor_effectiveness(anchor_data: RealityAnchorSchema, belief_alignment: Dict) -> float:
    """Predict anchor effectiveness based on design and beliefs."""
    base = 0.5
    # Bonus for belief alignment
    base += belief_alignment.get('score', 0) * 0.3
    # Bonus for specific trigger types
    if anchor_data.trigger_type == "time_based":
        base += 0.1
    if anchor_data.trigger_type == "location_based":
        base += 0.15
    return min(0.95, base)


async def _calculate_effectiveness(anchor: RealityAnchor, user_id: int) -> Dict:
    """Calculate real-time effectiveness forecast."""
    store = await get_behavioral_store()
    if not store:
        return {"score": 0.5, "confidence": "low"}
    
    # Get recent triggers
    week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
    triggers = store.client.table('anchor_triggers').select('*').eq(
        'anchor_id', anchor.id
    ).gte('triggered_at', week_ago).execute()
    
    if not triggers.data:
        return {"score": anchor.metadata.get('expected_effectiveness', 0.5), "confidence": "predicted"}
    
    followed = sum(1 for t in triggers.data if t.get('user_response') == 'followed')
    total = len(triggers.data)
    
    return {
        "score": followed / total,
        "confidence": "observed",
        "sample_size": total,
        "trend": "improving" if followed/total > 0.6 else "stable" if followed/total > 0.4 else "declining"
    }


async def _update_anchor_stats(anchor_id: int, response: str):
    """Update anchor performance statistics."""
    store = await get_behavioral_store()
    if not store:
        return
    
    # Get current stats
    stats = store.client.table('anchor_stats').select('*').eq('anchor_id', anchor_id).execute()
    
    if stats.data:
        current = stats.data[0]
        updates = {
            'total_triggers': current.get('total_triggers', 0) + 1,
            'followed_count': current.get('followed_count', 0) + (1 if response == 'followed' else 0),
            'ignored_count': current.get('ignored_count', 0) + (1 if response == 'ignored' else 0),
            'last_updated': datetime.utcnow().isoformat()
        }
        store.client.table('anchor_stats').update(updates).eq('anchor_id', anchor_id).execute()
    else:
        store.client.table('anchor_stats').insert({
            'anchor_id': anchor_id,
            'total_triggers': 1,
            'followed_count': 1 if response == 'followed' else 0,
            'ignored_count': 1 if response == 'ignored' else 0,
            'created_at': datetime.utcnow().isoformat()
        }).execute()


async def _analyze_anchor_failure(anchor_id: int, context: Dict, user_id: int):
    """Analyze why anchor was ignored."""
    # Log for pattern analysis
    store = await get_behavioral_store()
    if store:
        store.client.table('anchor_failures').insert({
            'failure_id': str(uuid4()),
            'anchor_id': anchor_id,
            'user_id': str(user_id),
            'context': context,
            'analyzed_at': datetime.utcnow().isoformat()
        }).execute()


def _find_optimal_conditions(triggers: List[Dict]) -> Dict:
    """Find conditions under which anchors work best."""
    if not triggers:
        return {}
    
    followed = [t for t in triggers if t.get('user_response') == 'followed']
    if not followed:
        return {"message": "No successful triggers yet"}
    
    # Find common patterns
    hours = [t.get('hour_of_day') for t in followed if t.get('hour_of_day') is not None]
    days = [t.get('day_of_week') for t in followed if t.get('day_of_week') is not None]
    
    from collections import Counter
    best_hour = Counter(hours).most_common(1)[0] if hours else None
    best_day = Counter(days).most_common(1)[0] if days else None
    
    return {
        "best_time": f"{best_hour[0]}:00" if best_hour else "unknown",
        "best_day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][best_day[0]] if best_day else "unknown",
        "confidence": best_hour[1]/len(followed) if best_hour else 0
    }


async def _get_baseline_execution_rate(user_id: int, goal_id: int, days: int, db: AsyncSession) -> float:
    """Get execution rate before anchors were used."""
    from spirit.models import Execution
    
    since = date.today() - timedelta(days=days*2)  # Look at period before anchors
    
    executions = await db.execute(
        select(Execution).where(
            Execution.goal_id == goal_id,
            Execution.day >= since,
            Execution.day < date.today() - timedelta(days=days)
        )
    )
    exec_list = executions.scalars().all()
    
    if not exec_list:
        return 0.5  # Default assumption
    
    completed = sum(1 for e in exec_list if e.executed)
    return completed / len(exec_list)


def _generate_anchor_recommendations(follow_rate: float, optimal_conditions: Dict) -> List[str]:
    """Generate recommendations for improving anchor effectiveness."""
    recs = []
    if follow_rate < 0.5:
        recs.append("Consider strengthening anchor with habit stacking")
        recs.append("Try temptation bundling to increase motivation")
    if optimal_conditions.get('best_time'):
        recs.append(f"Schedule triggers for {optimal_conditions['best_time']} when possible")
    if not recs:
        recs.append("Anchor is performing well. Maintain current design.")
    return recs


def _identify_existing_routines(observations: List[Any]) -> List[str]:
    """Identify high-frequency behaviors for habit stacking."""
    if not observations:
        return []
    
    # Group by hour and find consistent patterns
    hourly_patterns = {}
    for obs in observations:
        hour = datetime.fromisoformat(obs.timestamp.replace('Z', '+00:00')).hour
        app = obs.behavior.get('app_category', 'unknown') if hasattr(obs, 'behavior') else 'unknown'
        
        key = f"{hour:02d}:00 - {app}"
        hourly_patterns[key] = hourly_patterns.get(key, 0) + 1
    
    # Return top 3 most frequent
    sorted_patterns = sorted(hourly_patterns.items(), key=lambda x: x[1], reverse=True)
    return [p[0] for p in sorted_patterns[:3]]


async def _get_active_goal_id(user_id: int, db: AsyncSession) -> Optional[int]:
    """Helper to get active goal ID."""
    goal = await db.scalar(
        select(Goal).where(Goal.user_id == user_id, Goal.state == GoalState.active)
    )
    return goal.id if goal else None
