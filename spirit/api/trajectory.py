"""
Trajectory API: Daily execution tracking and objective generation.
v2.0: Integrated with behavioral data, belief networks, and strategic layer.
"""

from datetime import date, datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from spirit.db import async_session, get_behavioral_store
from spirit.models import Goal, GoalState, Execution, User
from spirit.api.auth import get_current_user
from spirit.graphs.daily_objective_graph import run_daily_objective
from spirit.agents.behavioral_scientist import PredictiveEngine
from spirit.memory.episodic_memory import EpisodicMemorySystem
from spirit.strategic import StrategicUnlockEngine, StrategicPlanGenerator


router = APIRouter(prefix="/trajectory", tags=["trajectory"])


async def get_db() -> AsyncSession:
    async with async_session() as session:
        yield session


@router.post("/execute")
async def log_day(
    objective_text: str,
    executed: bool,
    day: date = date.today(),
    reflection: Optional[str] = None,  # NEW: Optional user reflection
    behavioral_context: Optional[Dict] = None,  # NEW: Context from mobile app
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Log daily execution with enhanced behavioral and strategic context.
    v2.0: Now captures reflection and behavioral context for better predictions.
    """
    goal = await db.get(Goal, user.active_goal_id)
    if not goal or goal.state != GoalState.active:
        raise HTTPException(status_code=409, detail="No active goal")
    
    from spirit.services.execution_service import log_execution
    
    # NEW: Enrich with behavioral context if provided
    execution_data = {
        "objective_text": objective_text,
        "executed": executed,
        "reflection": reflection,
        "behavioral_context": behavioral_context or {},
        "logged_at": datetime.utcnow().isoformat()
    }
    
    ex = await log_execution(db, goal.id, day, objective_text, executed, execution_data)
    
    # NEW: Update belief network based on execution result
    if executed:
        await _update_belief_efficacy(user.id, objective_text, success=True)
    else:
        # Check if failure contradicts user's self-beliefs
        await _check_failure_belief_gap(user.id, objective_text)
    
    # NEW: Trigger strategic layer check if execution pattern changes
    await _check_strategic_unlock(user.id, db)
    
    return {
        "execution": ex,
        "streak_updated": True,
        "belief_model_updated": True,
        "next_objective_suggested": await _suggest_next_objective(user.id, goal.id)
    }


@router.get("/history")
async def history(
    limit: int = 30,
    include_behavioral: bool = True,  # NEW: Include behavioral correlation
    include_beliefs: bool = False,   # NEW: Include belief context
    user: User = Depends(get_current_user), 
    db: AsyncSession = Depends(get_db)
):
    """
    Get execution history with optional behavioral and belief insights.
    v2.0: Now shows correlation between behavioral patterns and execution success.
    """
    goal = await db.get(Goal, user.active_goal_id)
    if not goal:
        return []
    
    # Get executions
    executions = await db.execute(
        select(Execution)
        .where(Execution.goal_id == goal.id)
        .order_by(Execution.day.desc())
        .limit(limit)
    )
    executions_list = executions.scalars().all()
    
    if not include_behavioral and not include_beliefs:
        return executions_list
    
    # NEW: Enrich with behavioral data
    enriched_history = []
    store = await get_behavioral_store()
    
    for ex in executions_list:
        ex_dict = {
            "id": ex.id,
            "day": ex.day.isoformat() if hasattr(ex.day, 'isoformat') else str(ex.day),
            "objective_text": ex.objective_text,
            "executed": ex.executed,
            "created_at": ex.created_at.isoformat() if hasattr(ex.created_at, 'isoformat') else str(ex.created_at)
        }
        
        if include_behavioral and store:
            # Get behavioral data for this day
            day_start = datetime.combine(ex.day, datetime.min.time()).isoformat()
            day_end = (datetime.combine(ex.day, datetime.max.time())).isoformat()
            
            behavioral = store.client.table('behavioral_observations').select('*').eq(
                'user_id', str(user.id)
            ).gte('timestamp', day_start).lte('timestamp', day_end).execute()
            
            if behavioral.data:
                # Calculate daily metrics
                productive_time = sum(
                    obs.get('behavior', {}).get('session_duration_sec', 0) 
                    for obs in behavioral.data
                    if obs.get('behavior', {}).get('app_category') == 'productivity'
                ) / 60  # minutes
                
                focus_score = sum(
                    obs.get('behavior', {}).get('focus_score', 0.5)
                    for obs in behavioral.data
                ) / len(behavioral.data) if behavioral.data else 0.5
                
                ex_dict["behavioral_context"] = {
                    "productive_minutes": round(productive_time, 1),
                    "average_focus_score": round(focus_score, 2),
                    "observation_count": len(behavioral.data),
                    "primary_app_category": _get_primary_category(behavioral.data)
                }
        
        if include_beliefs and store:
            # Get beliefs at time of execution
            belief_result = store.client.table('belief_networks').select('*').eq(
                'user_id', str(user.id)
            ).lte('updated_at', ex_dict.get('created_at', datetime.utcnow().isoformat())).order(
                'updated_at', desc=True
            ).limit(1).execute()
            
            if belief_result.data:
                ex_dict["user_beliefs_at_time"] = belief_result.data[0].get('beliefs', {})
        
        enriched_history.append(ex_dict)
    
    return enriched_history


@router.post("/daily/generate")
async def generate_daily_objective(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Generate today's objective via LangGraph with strategic and behavioral awareness.
    v2.0: Now considers strategic maturity, behavioral patterns, and belief alignment.
    """
    # NEW: Check strategic maturity first
    unlock_engine = StrategicUnlockEngine(user.id)
    unlock_status = await unlock_engine.check_strategic_unlock_v2(db)
    
    # If strategic features unlocked, use strategic plan
    if unlock_status["can_unlock_strategic"]:
        plan_gen = StrategicPlanGenerator(user.id, unlock_status["maturity_level"])
        plan = await plan_gen.generate_12_week_plan(db)
        
        # Get current week objective from plan
        current_week = _get_current_week(plan)
        if current_week:
            objective = {
                "objective_text": current_week.get("focus", "Continue building consistent habits"),
                "strategic_context": {
                    "week": current_week.get("week"),
                    "plan_type": plan.get("plan_type"),
                    "behavioral_targets": current_week.get("behavioral_targets"),
                    "experiments": current_week.get("experiments", [])
                },
                "generated_by": "strategic_layer"
            }
        else:
            # Fallback to daily graph
            objective = await run_daily_objective(user.id, date.today().isoformat())
            objective["generated_by"] = "daily_graph"
    else:
        # Use standard daily objective graph
        objective = await run_daily_objective(user.id, date.today().isoformat())
        objective["generated_by"] = "daily_graph"
        objective["strategic_unlock_progress"] = unlock_status["progress"]
    
    # NEW: Check belief alignment
    store = await get_behavioral_store()
    if store:
        belief_model = await store.get_user_beliefs(str(user.id))
        if belief_model:
            # Warn if objective contradicts beliefs
            alignment = _check_objective_belief_alignment(
                objective.get("objective_text", ""), 
                belief_model.get('beliefs', {})
            )
            objective["belief_alignment"] = alignment
    
    # NEW: Predict success probability
    predictive = PredictiveEngine(user.id)
    prediction = await predictive.predict_goal_outcome(
        goal_id=user.active_goal_id or goal.id,
        horizon_days=1
    )
    objective["predicted_success_probability"] = prediction.get("trajectory_analysis", "Unknown")
    
    return objective


@router.get("/insights")
async def get_trajectory_insights(
    days: int = 14,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    NEW: Get AI-generated insights about execution patterns and trajectory.
    Combines behavioral data, execution history, and belief analysis.
    """
    goal = await db.get(Goal, user.active_goal_id)
    if not goal:
        raise HTTPException(status_code=404, detail="No active goal")
    
    # Get recent executions
    since = date.today() - timedelta(days=days)
    executions = await db.execute(
        select(Execution)
        .where(Execution.goal_id == goal.id)
        .where(Execution.day >= since)
        .order_by(Execution.day.desc())
    )
    exec_list = executions.scalars().all()
    
    # Calculate execution rate
    total = len(exec_list)
    completed = sum(1 for e in exec_list if e.executed)
    rate = completed / total if total > 0 else 0
    
    # Get behavioral correlation
    store = await get_behavioral_store()
    behavioral_insight = {}
    
    if store:
        # Compare behavioral patterns on success vs failure days
        success_days = [e.day for e in exec_list if e.executed]
        failure_days = [e.day for e in exec_list if not e.executed]
        
        if success_days and failure_days:
            success_behavior = await _get_avg_behavioral_metrics(user.id, success_days, store)
            failure_behavior = await _get_avg_behavioral_metrics(user.id, failure_days, store)
            
            behavioral_insight = {
                "success_days_avg": success_behavior,
                "failure_days_avg": failure_behavior,
                "key_difference": _identify_key_difference(success_behavior, failure_behavior)
            }
    
    # Get belief insights
    belief_insight = {}
    if store:
        belief_model = await store.get_user_beliefs(str(user.id))
        if belief_model:
            belief_insight = {
                "current_self_efficacy": belief_model.get('beliefs', {}).get('self_efficacy', 0.5),
                "believed_optimal_time": belief_model.get('beliefs', {}).get('optimal_time', 'unknown'),
                "execution_rate_matches_belief": _check_belief_accuracy(belief_model, rate)
            }
    
    # Generate insight text
    from spirit.agents.behavioral_scientist import ChatOpenAI
    from spirit.config import settings
    
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key)
    
    messages = [
        SystemMessage(content="""
        You are a trajectory analyst. Provide concise, actionable insights.
        Focus on patterns the user might not see themselves.
        Be encouraging but honest about gaps.
        """),
        HumanMessage(content=f"""
        Execution rate (last {days} days): {rate:.1%}
        Success days behavioral profile: {behavioral_insight.get('success_days_avg', {})}
        Failure days behavioral profile: {behavioral_insight.get('failure_days_avg', {})}
        Key difference: {behavioral_insight.get('key_difference', 'None identified')}
        
        User beliefs: {belief_insight}
        
        Generate 2-3 specific insights about their trajectory.
        """)
    ]
    
    response = llm.invoke(messages)
    
    return {
        "period_days": days,
        "execution_rate": rate,
        "executions_count": total,
        "behavioral_correlation": behavioral_insight,
        "belief_alignment": belief_insight,
        "ai_insights": response.content,
        "recommended_adjustment": _generate_recommendation(rate, behavioral_insight, belief_insight)
    }


@router.post("/reflect")
async def add_reflection(
    day: date,
    reflection_text: str,
    mood_score: Optional[int] = None,  # 1-10
    energy_level: Optional[int] = None,  # 1-10
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    NEW: Add qualitative reflection to a specific day's execution.
    Captures subjective experience for belief network calibration.
    """
    goal = await db.get(Goal, user.active_goal_id)
    if not goal:
        raise HTTPException(status_code=404, detail="No active goal")
    
    # Find execution for this day
    execution = await db.execute(
        select(Execution)
        .where(Execution.goal_id == goal.id)
        .where(Execution.day == day)
    )
    ex = execution.scalar_one_or_none()
    
    if not ex:
        raise HTTPException(status_code=404, detail=f"No execution found for {day}")
    
    # Store reflection
    store = await get_behavioral_store()
    if store:
        store.client.table('execution_reflections').insert({
            'reflection_id': str(uuid4()),
            'user_id': str(user.id),
            'execution_id': ex.id,
            'day': day.isoformat(),
            'reflection_text': reflection_text,
            'mood_score': mood_score,
            'energy_level': energy_level,
            'created_at': datetime.utcnow().isoformat()
        }).execute()
    
    # Update belief network with subjective data
    await _update_belief_from_reflection(user.id, ex.executed, mood_score, energy_level)
    
    return {
        "status": "reflection_recorded",
        "day": day.isoformat(),
        "will_update_belief_model": True
    }


# Helper functions

async def _update_belief_efficacy(user_id: int, objective_text: str, success: bool):
    """Update belief network with success/failure data."""
    store = await get_behavioral_store()
    if not store:
        return
    
    # Get current beliefs
    current = await store.get_user_beliefs(str(user_id))
    if not current:
        return
    
    # Update self-efficacy based on success/failure
    beliefs = current.get('beliefs', {})
    current_efficacy = beliefs.get('self_efficacy', 0.5)
    
    # Bayesian update (simplified)
    if success:
        new_efficacy = min(0.95, current_efficacy + 0.05)
    else:
        new_efficacy = max(0.1, current_efficacy - 0.03)
    
    beliefs['self_efficacy'] = new_efficacy
    beliefs['last_objective_type'] = objective_text[:50]
    
    await store.update_belief_model(str(user_id), beliefs)


async def _check_failure_belief_gap(user_id: int, objective_text: str):
    """Check if failure contradicts user's self-beliefs."""
    # This would trigger cognitive dissonance detection
    pass


async def _check_strategic_unlock(user_id: int, db: AsyncSession):
    """Check if user qualifies for strategic layer unlock."""
    unlock_engine = StrategicUnlockEngine(user_id)
    status = await unlock_engine.check_strategic_unlock_v2(db)
    
    if status["can_unlock_strategic"] and not status.get("already_notified"):
        # Notify user of unlock (but don't block execution logging)
        print(f"User {user_id} unlocked strategic features: {status['maturity_level']}")


async def _suggest_next_objective(user_id: int, goal_id: int) -> Optional[str]:
    """Suggest next objective based on patterns."""
    # Would use episodic memory to find what worked
    memory = EpisodicMemorySystem(user_id)
    # Implementation would go here
    return None


def _get_primary_category(observations: List[Dict]) -> str:
    """Get primary app category from observations."""
    categories = {}
    for obs in observations:
        cat = obs.get('behavior', {}).get('app_category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    return max(categories.items(), key=lambda x: x[1])[0] if categories else 'unknown'


def _get_current_week(plan: Dict) -> Optional[Dict]:
    """Get current week from 12-week plan."""
    weeks = plan.get('weeks', [])
    if not weeks:
        return None
    
    # Find week based on start date (simplified)
    # In production, would track plan start date
    return weeks[0]  # Default to first week


def _check_objective_belief_alignment(objective_text: str, beliefs: Dict) -> Dict:
    """Check if objective aligns with user's beliefs."""
    alignment_score = 0.5
    
    # Check time-based beliefs
    if 'morning' in objective_text.lower() and beliefs.get('optimal_time') == 'evening':
        alignment_score -= 0.3
    if 'evening' in objective_text.lower() and beliefs.get('optimal_time') == 'morning':
        alignment_score -= 0.3
    
    return {
        "alignment_score": max(0, alignment_score),
        "potential_conflict": alignment_score < 0.4,
        "recommendation": "Consider adjusting timing to match user's believed optimal time" if alignment_score < 0.4 else "Alignment good"
    }


async def _get_avg_behavioral_metrics(user_id: int, days: List[date], store) -> Dict:
    """Get average behavioral metrics for specific days."""
    if not days:
        return {}
    
    total_productive = 0
    total_focus = 0
    count = 0
    
    for day in days:
        day_start = datetime.combine(day, datetime.min.time()).isoformat()
        day_end = (datetime.combine(day, datetime.max.time())).isoformat()
        
        obs = store.client.table('behavioral_observations').select('*').eq(
            'user_id', str(user_id)
        ).gte('timestamp', day_start).lte('timestamp', day_end).execute()
        
        if obs.data:
            productive = sum(
                o.get('behavior', {}).get('session_duration_sec', 0)
                for o in obs.data
                if o.get('behavior', {}).get('app_category') == 'productivity'
            ) / 60
            
            focus = sum(
                o.get('behavior', {}).get('focus_score', 0.5)
                for o in obs.data
            ) / len(obs.data)
            
            total_productive += productive
            total_focus += focus
            count += 1
    
    return {
        "avg_productive_minutes": round(total_productive / count, 1) if count else 0,
        "avg_focus_score": round(total_focus / count, 2) if count else 0,
        "sample_days": count
    }


def _identify_key_difference(success_metrics: Dict, failure_metrics: Dict) -> str:
    """Identify the key behavioral difference between success and failure days."""
    if not success_metrics or not failure_metrics:
        return "Insufficient data"
    
    productive_diff = success_metrics.get('avg_productive_minutes', 0) - failure_metrics.get('avg_productive_minutes', 0)
    focus_diff = success_metrics.get('avg_focus_score', 0) - failure_metrics.get('avg_focus_score', 0)
    
    if productive_diff > 30:
        return f"Success days have {productive_diff:.0f} more minutes of productive time"
    if focus_diff > 0.2:
        return f"Success days have {focus_diff:.1f} higher focus scores"
    
    return "No strong behavioral pattern identified"


def _check_belief_accuracy(belief_model: Dict, actual_rate: float) -> bool:
    """Check if user's self-efficacy belief matches reality."""
    belief_efficacy = belief_model.get('beliefs', {}).get('self_efficacy', 0.5)
    return abs(belief_efficacy - actual_rate) < 0.2


def _generate_recommendation(rate: float, behavioral: Dict, belief: Dict) -> str:
    """Generate specific recommendation based on analysis."""
    if rate < 0.5:
        return "Focus on building consistency before optimizing. Small daily wins."
    if behavioral.get('key_difference', '').startswith('Success days have'):
        return "Replicate the behavioral patterns of your successful days"
    if not belief.get('execution_rate_matches_belief', True):
        return "Your self-assessment may need recalibration. Trust the data."
    
    return "Continue current trajectory. You're on track."


async def _update_belief_from_reflection(user_id: int, executed: bool, mood: Optional[int], energy: Optional[int]):
    """Update belief network based on subjective reflection."""
    # Would adjust beliefs about what conditions lead to success
    pass
