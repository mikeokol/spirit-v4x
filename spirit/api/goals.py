"""
Goals API: Goal declaration, lifecycle management, and behavioral integration.
v2.0: Integrated with belief networks, strategic layer, and behavioral validation.
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from uuid import UUID, uuid4

from spirit.db import async_session, get_behavioral_store
from spirit.models import Goal, GoalState, User, ModeState
from spirit.api.auth import get_current_user
from spirit.strategies.library import DEFAULT_STRATEGY_BY_DOMAIN
from spirit.agents.behavioral_scientist import PredictiveEngine
from spirit.strategic import StrategicUnlockEngine, StrategicPlanGenerator
from spirit.memory.episodic_memory import EpisodicMemorySystem


router = APIRouter(prefix="/goals", tags=["goals"])


async def get_db() -> AsyncSession:
    async with async_session() as session:
        yield session


@router.post("", response_model=dict)
async def declare_goal(
    text: str,
    domain: Optional[str] = None,
    target_date: Optional[datetime] = None,  # NEW: Optional target completion date
    success_criteria: Optional[str] = None,   # NEW: How will user know they've succeeded
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Declare a new goal with behavioral science integration.
    v2.0: Now captures domain, target date, and success criteria for better prediction.
    """
    # Check for existing active goal
    active = await db.scalar(
        select(Goal).where(Goal.user_id == user.id, Goal.state == GoalState.active)
    )
    if active:
        raise HTTPException(
            status_code=409, 
            detail="Active goal already exists; abandon or complete it first"
        )
    
    # NEW: Infer domain from text if not provided
    inferred_domain = domain or _infer_domain(text)
    
    # NEW: Validate goal against user's belief model
    store = await get_behavioral_store()
    belief_alignment = {}
    if store:
        belief_model = await store.get_user_beliefs(str(user.id))
        if belief_model:
            belief_alignment = _check_goal_belief_alignment(text, belief_model.get('beliefs', {}))
    
    # Create goal with enhanced metadata
    goal = Goal(
        user_id=user.id,
        text=text.strip(),
        domain=inferred_domain,
        target_date=target_date,
        success_criteria=success_criteria,
        created_at=datetime.utcnow(),
        metadata={
            "belief_alignment": belief_alignment,
            "inferred_domain": inferred_domain,
            "declaration_context": {
                "time_of_day": datetime.utcnow().hour,
                "day_of_week": datetime.utcnow().weekday()
            }
        }
    )
    
    db.add(goal)
    await db.commit()
    await db.refresh(goal)
    
    # NEW: Initialize goal-specific belief tracking
    if store:
        store.client.table('goal_beliefs').insert({
            'goal_belief_id': str(uuid4()),
            'user_id': str(user.id),
            'goal_id': str(goal.id),
            'initial_confidence': belief_alignment.get('confidence', 0.5),
            'alignment_score': belief_alignment.get('alignment_score', 0.5),
            'created_at': datetime.utcnow().isoformat()
        }).execute()
    
    # NEW: Generate initial trajectory prediction
    predictive = PredictiveEngine(user.id)
    initial_prediction = await predictive.predict_goal_outcome(goal.id, horizon_days=30)
    
    return {
        "id": goal.id,
        "text": goal.text,
        "state": goal.state,
        "domain": inferred_domain,
        "target_date": target_date.isoformat() if target_date else None,
        "belief_alignment": belief_alignment,
        "initial_prediction": {
            "success_probability": initial_prediction.get("trajectory_analysis", "Unknown"),
            "key_risks": initial_prediction.get("key_risks", []),
            "critical_intervention_points": initial_prediction.get("critical_intervention_points", [])
        },
        "recommendation": _generate_declaration_recommendation(belief_alignment, initial_prediction)
    }


@router.patch("/{goal_id}/activate", response_model=dict)
async def activate_goal(
    goal_id: UUID, 
    user: User = Depends(get_current_user), 
    db: AsyncSession = Depends(get_db)
):
    """
    Activate a goal with strategic layer initialization.
    v2.0: Now checks behavioral stability before enabling strategic mode.
    """
    goal = await db.get(Goal, goal_id)
    if not goal or goal.user_id != user.id:
        raise HTTPException(status_code=404, detail="Goal not found")

    # Deactivate any other active goal
    active_goals = await db.scalars(
        select(Goal).where(Goal.user_id == user.id, Goal.state == GoalState.active)
    )
    for g in active_goals:
        g.state = GoalState.abandoned
        # NEW: Log goal abandonment for causal analysis
        await _log_goal_transition(user.id, g.id, "abandoned", "new_goal_activated")

    goal.state = GoalState.active
    goal.domain = goal.domain or "business"
    goal.activated_at = datetime.utcnow()
    await db.commit()

    # NEW: Check if user qualifies for strategic mode (behavioral stability required)
    unlock_engine = StrategicUnlockEngine(user.id)
    unlock_status = await unlock_engine.check_strategic_unlock_v2(db)
    
    # Initialize or update mode state
    mode = await db.scalar(select(ModeState).where(ModeState.user_id == user.id))
    if not mode:
        mode = ModeState(user_id=user.id)
        db.add(mode)
    
    if unlock_status["can_unlock_strategic"]:
        # User has proven behavioral stability - enable strategic
        mode.strategic_enabled = True
        mode.strategic_state = ModeState.unlocked
        mode.strategy_key = DEFAULT_STRATEGY_BY_DOMAIN.get(goal.domain, "default")
        mode.strategy_started_at = datetime.utcnow()
        
        # NEW: Generate initial strategic plan
        plan_gen = StrategicPlanGenerator(user.id, unlock_status["maturity_level"])
        strategic_plan = await plan_gen.generate_12_week_plan(db)
        
        # Store plan reference
        goal.metadata = goal.metadata or {}
        goal.metadata["strategic_plan"] = {
            "plan_id": str(uuid4()),
            "maturity_level": unlock_status["maturity_level"].value,
            "generated_at": datetime.utcnow().isoformat(),
            "weeks": strategic_plan.get("weeks", [])[:4]  # Store first 4 weeks
        }
        await db.commit()
        
        strategic_message = f"Strategic mode unlocked ({unlock_status['maturity_level'].value})"
    else:
        # Reset to locked - user needs to prove behavioral stability
        mode.strategic_enabled = False
        mode.strategic_state = ModeState.locked
        mode.strategy_key = None
        mode.strategy_started_at = None
        await db.commit()
        
        strategic_message = (
            f"Strategic mode locked. Requirements: "
            f"{unlock_status['next_requirements']['requirements']}"
        )

    return {
        "detail": "Goal activated",
        "strategic_status": strategic_message,
        "maturity_level": unlock_status["maturity_level"].value,
        "unlocked_features": unlock_status.get("unlocked_features", []),
        "estimated_days_to_strategic": unlock_status.get("estimated_days_to_strategic"),
        "behavioral_requirements": unlock_status["next_requirements"]
    }


@router.get("/active", response_model=dict)
async def active_goal(
    include_progress: bool = True,      # NEW: Include execution progress
    include_predictions: bool = True,   # NEW: Include AI predictions
    user: User = Depends(get_current_user), 
    db: AsyncSession = Depends(get_db)
):
    """
    Get active goal with full behavioral and predictive context.
    v2.0: Now includes execution progress, predictions, and belief alignment.
    """
    goal = await db.scalar(
        select(Goal).where(Goal.user_id == user.id, Goal.state == GoalState.active)
    )
    
    if not goal:
        return None
    
    response = {
        "id": goal.id,
        "text": goal.text,
        "state": goal.state,
        "domain": goal.domain,
        "target_date": goal.target_date.isoformat() if goal.target_date else None,
        "success_criteria": goal.success_criteria,
        "activated_at": goal.activated_at.isoformat() if goal.activated_at else None,
        "days_active": (datetime.utcnow() - goal.activated_at).days if goal.activated_at else 0
    }
    
    if include_progress:
        # Get execution statistics
        from spirit.models import Execution
        executions = await db.execute(
            select(Execution).where(Execution.goal_id == goal.id)
        )
        exec_list = executions.scalars().all()
        
        total = len(exec_list)
        completed = sum(1 for e in exec_list if e.executed)
        
        # Calculate streak
        streak = 0
        for e in sorted(exec_list, key=lambda x: x.day, reverse=True):
            if e.executed:
                streak += 1
            else:
                break
        
        response["progress"] = {
            "total_days": total,
            "completed_days": completed,
            "completion_rate": completed / total if total > 0 else 0,
            "current_streak": streak,
            "longest_streak": _calculate_longest_streak(exec_list)
        }
    
    if include_predictions:
        # Get updated prediction
        predictive = PredictiveEngine(user.id)
        prediction = await predictive.predict_goal_outcome(goal.id, horizon_days=7)
        
        response["predictions"] = {
            "trajectory": prediction.get("trajectory_analysis"),
            "confidence": prediction.get("confidence"),
            "belief_alignment": prediction.get("belief_alignment")
        }
    
    # NEW: Get strategic plan if available
    if goal.metadata and goal.metadata.get("strategic_plan"):
        response["strategic_plan"] = goal.metadata["strategic_plan"]
    
    return response


@router.patch("/{goal_id}/complete", response_model=dict)
async def complete_goal(
    goal_id: UUID, 
    completion_reflection: Optional[str] = None,  # NEW: What did user learn
    would_recommend_strategy: Optional[bool] = None,  # NEW: Feedback
    user: User = Depends(get_current_user), 
    db: AsyncSession = Depends(get_db)
):
    """
    Complete a goal with post-hoc analysis and belief network update.
    v2.0: Captures completion insights for future goal recommendations.
    """
    goal = await db.get(Goal, goal_id)
    if not goal or goal.user_id != user.id:
        raise HTTPException(status_code=404, detail="Goal not found")
    
    goal.state = GoalState.completed
    goal.completed_at = datetime.utcnow()
    
    # NEW: Store completion metadata
    goal.metadata = goal.metadata or {}
    goal.metadata["completion"] = {
        "completed_at": datetime.utcnow().isoformat(),
        "reflection": completion_reflection,
        "would_recommend_strategy": would_recommend_strategy,
        "days_active": (datetime.utcnow() - goal.activated_at).days if goal.activated_at else 0
    }
    
    await db.commit()
    
    # NEW: Update belief network with success
    store = await get_behavioral_store()
    if store:
        # Mark goal beliefs as validated
        store.client.table('goal_beliefs').update({
            'completed': True,
            'success': True,
            'completed_at': datetime.utcnow().isoformat()
        }).eq('goal_id', str(goal.id)).execute()
        
        # Update general self-efficacy
        await _update_efficacy_from_completion(user.id, goal.id, success=True)
    
    # NEW: Generate completion insights
    insights = await _generate_completion_insights(user.id, goal.id)
    
    # Reset strategic mode
    mode = await db.scalar(select(ModeState).where(ModeState.user_id == user.id))
    if mode:
        mode.strategic_enabled = False
        mode.strategic_state = ModeState.locked
        mode.strategy_key = None
        await db.commit()
    
    return {
        "detail": "Goal marked completed",
        "days_active": goal.metadata["completion"]["days_active"],
        "completion_insights": insights,
        "belief_updated": True,
        "next_goal_recommendation": await _recommend_next_goal_type(user.id)
    }


@router.patch("/{goal_id}/abandon", response_model=dict)
async def abandon_goal(
    goal_id: UUID, 
    abandonment_reason: Optional[str] = None,  # NEW: Why are they giving up
    learned_anything: Optional[bool] = None,   # NEW: Was there value anyway
    user: User = Depends(get_current_user), 
    db: AsyncSession = Depends(get_db)
):
    """
    Abandon a goal with causal analysis for future predictions.
    v2.0: Captures abandonment reasons to improve future goal recommendations.
    """
    goal = await db.get(Goal, goal_id)
    if not goal or goal.user_id != user.id:
        raise HTTPException(status_code=404, detail="Goal not found")
    
    goal.state = GoalState.abandoned
    goal.abandoned_at = datetime.utcnow()
    
    # NEW: Store abandonment analysis
    goal.metadata = goal.metadata or {}
    goal.metadata["abandonment"] = {
        "abandoned_at": datetime.utcnow().isoformat(),
        "reason": abandonment_reason,
        "learned_anything": learned_anything,
        "days_active": (datetime.utcnow() - goal.activated_at).days if goal.activated_at else 0
    }
    
    await db.commit()
    
    # NEW: Update belief network (failure affects self-efficacy)
    store = await get_behavioral_store()
    if store:
        store.client.table('goal_beliefs').update({
            'completed': True,
            'success': False,
            'abandoned': True,
            'abandoned_at': datetime.utcnow().isoformat()
        }).eq('goal_id', str(goal.id)).execute()
        
        # Adjust self-efficacy (but not as harshly as raw failure)
        adjustment = -0.1 if learned_anything else -0.2
        await _update_efficacy_from_completion(user.id, goal.id, success=False, adjustment=adjustment)
    
    # NEW: Analyze abandonment for strategic insights
    await _analyze_abandonment(user.id, goal.id, abandonment_reason)
    
    return {
        "detail": "Goal abandoned",
        "days_active": goal.metadata["abandonment"]["days_active"],
        "belief_updated": True,
        "alternative_suggested": await _suggest_alternative_goal(user.id, goal.domain, abandonment_reason)
    }


@router.get("/history", response_model=List[dict])
async def goal_history(
    include_abandoned: bool = True,
    include_analysis: bool = False,  # NEW: Include AI analysis of patterns
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get history of all goals with optional behavioral pattern analysis.
    v2.0: Shows patterns across goals to identify user archetype.
    """
    query = select(Goal).where(Goal.user_id == user.id)
    if not include_abandoned:
        query = query.where(Goal.state != GoalState.abandoned)
    
    query = query.order_by(Goal.created_at.desc())
    goals = await db.scalars(query)
    goal_list = goals.all()
    
    if not include_analysis:
        return [
            {
                "id": g.id,
                "text": g.text,
                "state": g.state.value,
                "domain": g.domain,
                "created_at": g.created_at.isoformat() if g.created_at else None,
                "completed_at": g.completed_at.isoformat() if g.completed_at else None
            }
            for g in goal_list
        ]
    
    # NEW: Generate pattern analysis
    analysis = await _analyze_goal_patterns(user.id, goal_list)
    
    return {
        "goals": [
            {
                "id": g.id,
                "text": g.text,
                "state": g.state.value,
                "domain": g.domain,
                "created_at": g.created_at.isoformat() if g.created_at else None,
                "completed_at": g.completed_at.isoformat() if g.completed_at else None
            }
            for g in goal_list
        ],
        "pattern_analysis": analysis,
        "user_archetype": analysis.get("archetype"),
        "success_factors": analysis.get("success_factors", []),
        "failure_patterns": analysis.get("failure_patterns", [])
    }


@router.post("/{goal_id}/adjust", response_model=dict)
async def adjust_goal(
    goal_id: UUID,
    new_text: Optional[str] = None,
    new_target_date: Optional[datetime] = None,
    reason: Optional[str] = None,  # NEW: Why is this adjustment happening
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    NEW: Adjust goal parameters mid-flight based on learning.
    Captures goal evolution for belief network calibration.
    """
    goal = await db.get(Goal, goal_id)
    if not goal or goal.user_id != user.id:
        raise HTTPException(status_code=404, detail="Goal not found")
    
    if goal.state != GoalState.active:
        raise HTTPException(status_code=409, detail="Can only adjust active goals")
    
    # Track adjustment
    adjustment_record = {
        "adjusted_at": datetime.utcnow().isoformat(),
        "reason": reason,
        "old_text": goal.text if new_text else None,
        "new_text": new_text,
        "old_target": goal.target_date.isoformat() if goal.target_date and new_target_date else None,
        "new_target": new_target_date.isoformat() if new_target_date else None
    }
    
    if new_text:
        goal.text = new_text
    if new_target_date:
        goal.target_date = new_target_date
    
    goal.metadata = goal.metadata or {}
    goal.metadata.setdefault("adjustments", []).append(adjustment_record)
    
    await db.commit()
    
    # NEW: Check if adjustment indicates belief shift
    if reason and "realized" in reason.lower():
        await _detect_belief_shift_from_adjustment(user.id, adjustment_record)
    
    return {
        "detail": "Goal adjusted",
        "adjustment_logged": True,
        "total_adjustments": len(goal.metadata.get("adjustments", []))
    }


# Helper functions

def _infer_domain(text: str) -> str:
    """Infer goal domain from text."""
    text_lower = text.lower()
    domains = {
        "fitness": ["weight", "gym", "run", "exercise", "workout", "health", "diet"],
        "business": ["revenue", "business", "startup", "company", "client", "sales"],
        "creative": ["book", "write", "art", "music", "design", "create", "album"],
        "learning": ["learn", "study", "course", "degree", "certification", "language"],
        "finance": ["save", "invest", "debt", "money", "financial", "budget"]
    }
    
    for domain, keywords in domains.items():
        if any(kw in text_lower for kw in keywords):
            return domain
    
    return "personal"


def _check_goal_belief_alignment(goal_text: str, beliefs: Dict) -> Dict:
    """Check if goal aligns with user's self-beliefs."""
    alignment_score = 0.5
    concerns = []
    
    # Check self-efficacy
    efficacy = beliefs.get('self_efficacy', 0.5)
    if efficacy < 0.4:
        concerns.append("Low self-efficacy may impede goal pursuit")
        alignment_score -= 0.2
    
    # Check goal size vs past performance
    past_completion_rate = beliefs.get('goal_completion_rate', 0.5)
    if "100k" in goal_text.lower() or "million" in goal_text.lower():
        if past_completion_rate < 0.3:
            concerns.append("Ambitious goal with low historical completion rate")
            alignment_score -= 0.3
    
    return {
        "alignment_score": max(0, alignment_score),
        "confidence": efficacy,
        "concerns": concerns,
        "recommendation": "Consider smaller milestone first" if concerns else "Goal aligns with self-model"
    }


def _generate_declaration_recommendation(alignment: Dict, prediction: Dict) -> str:
    """Generate recommendation based on goal declaration analysis."""
    if alignment.get("alignment_score", 0.5) < 0.3:
        return "Consider starting with a smaller version of this goal to build confidence"
    if "key_risks" in prediction and len(prediction["key_risks"]) > 2:
        return "High-risk goal detected. Ensure you have support systems in place"
    return "Goal declaration looks solid. Focus on consistent daily execution"


async def _log_goal_transition(user_id: int, goal_id: UUID, transition: str, reason: str):
    """Log goal transition for causal analysis."""
    store = await get_behavioral_store()
    if store:
        store.client.table('goal_transitions').insert({
            'transition_id': str(uuid4()),
            'user_id': str(user_id),
            'goal_id': str(goal_id),
            'transition': transition,
            'reason': reason,
            'logged_at': datetime.utcnow().isoformat()
        }).execute()


def _calculate_longest_streak(executions: List[Any]) -> int:
    """Calculate longest streak from execution list."""
    if not executions:
        return 0
    
    sorted_execs = sorted(executions, key=lambda x: x.day)
    longest = 0
    current = 0
    
    for e in sorted_execs:
        if e.executed:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    
    return longest


async def _update_efficacy_from_completion(user_id: int, goal_id: UUID, success: bool, adjustment: float = None):
    """Update self-efficacy belief based on goal completion."""
    store = await get_behavioral_store()
    if not store:
        return
    
    current = await store.get_user_beliefs(str(user_id))
    if not current:
        return
    
    beliefs = current.get('beliefs', {})
    current_efficacy = beliefs.get('self_efficacy', 0.5)
    
    if adjustment:
        new_efficacy = max(0.1, min(0.95, current_efficacy + adjustment))
    elif success:
        new_efficacy = min(0.95, current_efficacy + 0.1)
    else:
        new_efficacy = max(0.1, current_efficacy - 0.15)
    
    beliefs['self_efficacy'] = new_efficacy
    
    # Update completion rate
    completed = beliefs.get('goals_completed', 0)
    attempted = beliefs.get('goals_attempted', 0) + 1
    if success:
        completed += 1
    
    beliefs['goals_completed'] = completed
    beliefs['goals_attempted'] = attempted
    beliefs['goal_completion_rate'] = completed / attempted if attempted > 0 else 0
    
    await store.update_belief_model(str(user_id), beliefs)


async def _generate_completion_insights(user_id: int, goal_id: UUID) -> Dict:
    """Generate insights from goal completion."""
    # Would use episodic memory and causal analysis
    return {
        "what_worked": ["Consistent daily execution", "Strategic interventions"],
        "key_learnings": "User responds well to micro-interventions",
        "transferable_skills": ["Habit stacking", "Focus maintenance"]
    }


async def _recommend_next_goal_type(user_id: int) -> Dict:
    """Recommend next goal type based on success patterns."""
    store = await get_behavioral_store()
    if not store:
        return {"recommendation": "Continue with similar goals"}
    
    # Analyze past successes
    successes = store.client.table('goal_beliefs').select('*').eq(
        'user_id', str(user_id)
    ).eq('success', True).execute()
    
    if successes.data:
        domains = [s.get('domain', 'unknown') for s in successes.data]
        from collections import Counter
        most_common = Counter(domains).most_common(1)
        
        if most_common:
            return {
                "recommendation": f"Consider another {most_common[0][0]} goal",
                "confidence": most_common[0][1] / len(successes.data),
                "reasoning": f"You've shown strong performance in {most_common[0][0]} domains"
            }
    
    return {"recommendation": "Start with a small, achievable goal to build momentum"}


async def _analyze_abandonment(user_id: int, goal_id: UUID, reason: str):
    """Analyze abandonment for strategic insights."""
    # Would update collective intelligence about failure modes
    pass


async def _suggest_alternative_goal(user_id: int, domain: str, reason: str) -> Optional[str]:
    """Suggest alternative goal if abandonment was due to poor fit."""
    if not reason:
        return None
    
    if "too hard" in reason.lower() or "overwhelming" in reason.lower():
        return f"Consider a smaller {domain} goal with shorter timeline"
    if "not interested" in reason.lower():
        return "Consider exploring a different domain entirely"
    
    return None


async def _analyze_goal_patterns(user_id: int, goals: List[Goal]) -> Dict:
    """Analyze patterns across user's goal history."""
    if not goals:
        return {"archetype": "new_user"}
    
    completed = [g for g in goals if g.state == GoalState.completed]
    abandoned = [g for g in goals if g.state == GoalState.abandoned]
    
    completion_rate = len(completed) / len(goals) if goals else 0
    
    # Identify archetype
    if completion_rate > 0.7:
        archetype = "finisher"
    elif completion_rate < 0.3:
        archetype = "explorer"  # Tries many things, completes few
    elif len(abandoned) > len(completed):
        archetype = "pivoter"  # Abandons strategically
    else:
        archetype = "developing"
    
    # Identify success factors
    success_factors = []
    if completed:
        # Check if specific domains work better
        domains = [g.domain for g in completed if g.domain]
        from collections import Counter
        if domains:
            best_domain = Counter(domains).most_common(1)[0]
            success_factors.append(f"Strong performance in {best_domain[0]} domain")
    
    # Identify failure patterns
    failure_patterns = []
    if abandoned:
        reasons = [g.metadata.get("abandonment", {}).get("reason", "") for g in abandoned if g.metadata]
        if any("time" in r.lower() for r in reasons if r):
            failure_patterns.append("Time management challenges")
        if any("hard" in r.lower() for r in reasons if r):
            failure_patterns.append("Goal difficulty calibration issues")
    
    return {
        "archetype": archetype,
        "completion_rate": completion_rate,
        "total_goals": len(goals),
        "completed": len(completed),
        "abandoned": len(abandoned),
        "success_factors": success_factors,
        "failure_patterns": failure_patterns
    }


async def _detect_belief_shift_from_adjustment(user_id: int, adjustment: Dict):
    """Detect if goal adjustment indicates a shift in user's beliefs."""
    # Would log for belief network analysis
    pass
