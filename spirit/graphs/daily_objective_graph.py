"""
Daily-objective generation state-machine with Reality-Anchor policy + LangSmith traces.
INTEGRATED: Behavioral intelligence, belief tracking, multi-agent debate, ethical oversight.
"""

from datetime import date, timedelta
from typing import Any, Dict, Optional, List
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from sqlalchemy import select, func
from langsmith import traceable

from spirit.db import async_session
from spirit.models import Goal, GoalState, Execution, DailyObjective, RealityAnchor, GoalProfile, ModeState
from spirit.schemas.daily_objective import DailyObjectiveSchema
from spirit.schemas.reality_anchor import RealityAnchorSchema
from spirit.services.openai_client import plan_daily_objective
from spirit.services.decomposer import driver_math, bottleneck_pick
from spirit.services.calibrator import build_prompt
from spirit.strategies.library import STRATEGIES, DEFAULT_STRATEGY_BY_DOMAIN

# NEW: Behavioral intelligence imports
from spirit.belief.belief_network import BayesianBeliefNetwork, BeliefType
from spirit.agents.multi_agent_debate import MultiAgentDebate
from spirit.ethics.oversight import get_ethical_oversight, ethical_check
from spirit.memory.episodic_memory import EpisodicMemorySystem

from uuid import UUID
import logging

logger = logging.getLogger("spirit")


class GraphState(TypedDict):
    user_id: UUID
    today: date
    goal: Optional[Dict[str, Any]]
    profile: Optional[Dict[str, Any]]
    anchor: Optional[RealityAnchorSchema]
    drivers: Optional[Dict[str, float]]
    bottleneck: Optional[str]
    strategy_key: Optional[str]
    last7: List[Dict[str, Any]]
    constraints: Optional[Dict[str, Any]]
    
    # NEW: Behavioral intelligence fields
    user_beliefs: Optional[List[Dict]]  # Active beliefs that might affect planning
    behavioral_patterns: Optional[List[str]]  # Semantic memory patterns
    recent_dissonance: Optional[Dict]  # Any cognitive dissonance detected
    ethical_clearance: Optional[bool]  # Can we intervene today?
    
    ai_objective: Optional[Dict[str, Any]]
    stored_objective: Optional[Dict[str, Any]]
    debate_result: Optional[Dict]  # NEW: Multi-agent debate output


@traceable(run_type="chain", name="apply_rules")
def apply_rules(state: GraphState) -> GraphState:
    last7 = state.get("last7", [])
    misses = sum(1 for e in last7 if e.get("status") == "miss")
    stabilization = misses >= 3
    drivers = state["drivers"]
    bottleneck = state["bottleneck"]

    # NEW: Factor in behavioral patterns from semantic memory
    patterns = state.get("behavioral_patterns", [])
    
    # Adjust constraints based on learned patterns
    if "sugar_crash_afternoon" in patterns and not stabilization:
        # User crashes after lunch—don't schedule hard tasks then
        difficulty_cap, time_cap = 3, 40
    elif "morning_deep_work_best" in patterns and not stabilization:
        # User does best deep work in morning—capitalize on it
        difficulty_cap, time_cap = 5, 90
    elif stabilization:
        difficulty_cap, time_cap = 2, 30
    elif bottleneck == "lead_gen":
        difficulty_cap, time_cap = 3, 45
    elif bottleneck == "conversation":
        difficulty_cap, time_cap = 4, 60
    else:  # close
        difficulty_cap, time_cap = 5, 75

    # NEW: Check for belief-based constraints
    beliefs = state.get("user_beliefs", [])
    for belief in beliefs:
        if belief.get("type") == BeliefType.CONSTRAINT.value:
            # User believes they can't do something—respect it or challenge gently
            if "night" in belief.get("statement", "").lower():
                time_cap = min(time_cap, 60)  # Don't push late work

    state["constraints"] = {
        "stabilization": stabilization,
        "difficulty_cap": difficulty_cap,
        "time_budget_cap": time_cap,
        "max_micro_steps": 3,
        "bottleneck": bottleneck,
        "behavioral_patterns": patterns,  # Pass to LLM
    }
    return state


@traceable(run_type="chain", name="load_state")
async def load_state(state: GraphState) -> GraphState:
    user_id = state["user_id"]
    today = state["today"]
    start = today - timedelta(days=7)

    async with async_session() as session:
        goal_row = await session.scalar(select(Goal).where(Goal.user_id == user_id, Goal.state == GoalState.active))
        if not goal_row:
            raise ValueError("No active goal")

        profile_row = await session.scalar(select(GoalProfile).where(GoalProfile.goal_id == goal_row.id))
        profile = RealityAnchorSchema.from_orm(profile_row).dict() if profile_row else None

        anchor_row = await session.scalar(select(RealityAnchor).where(RealityAnchor.goal_id == goal_row.id))
        if anchor_row:
            anchor = RealityAnchorSchema(
                offer=anchor_row.offer,
                target_customer=anchor_row.target_customer,
                channel=anchor_row.channel,
                price=anchor_row.price,
                weekly_lead_target=anchor_row.weekly_lead_target,
                weekly_conversation_target=anchor_row.weekly_conversation_target,
                weekly_close_target=anchor_row.weekly_close_target,
            )
            drivers = driver_math(anchor)
            bottleneck = bottleneck_pick(last7=[], drivers=drivers)
        else:
            anchor = None
            drivers = None
            bottleneck = "anchor_creation"

        # Load strategy from ModeState
        mode_row = await session.scalar(select(ModeState).where(ModeState.user_id == user_id))
        if mode_row and mode_row.strategy_key:
            strategy_key = mode_row.strategy_key
        else:
            domain = mode_row.domain if mode_row and mode_row.domain else "business"
            strategy_key = DEFAULT_STRATEGY_BY_DOMAIN[domain]

        rows = await session.scalars(
            select(Execution)
            .where(Execution.goal_id == goal_row.id)
            .where(Execution.day >= start)
            .order_by(Execution.day.desc())
        )
        last7 = [{"date": r.day, "status": "done" if r.executed else "miss"} for r in rows.all()]

    # NEW: Load behavioral intelligence
    # 1. User beliefs
    belief_network = BayesianBeliefNetwork(int(user_id))
    beliefs_raw = await belief_network.get_beliefs_for_testing()
    beliefs = [
        {
            "id": b.belief_id,
            "statement": b.statement,
            "type": b.belief_type.value,
            "confidence": b.posterior_probability,
            "tested": b.times_tested
        }
        for b in beliefs_raw
    ]

    # 2. Semantic memory patterns (from consolidation)
    from spirit.memory.episodic_memory import EpisodicMemorySystem
    memory = EpisodicMemorySystem(int(user_id))
    # Get patterns from last 14 days
    recent_memories = await memory.retrieve_relevant_memories(
        current_context={"goal_id": str(goal_row.id)},
        time_horizon=timedelta(days=14),
        n_results=50
    )
    # Extract patterns (simplified—would query semantic_memories table)
    patterns = []
    for mem in recent_memories:
        if mem.episode_type in ["breakthrough", "insight"]:
            patterns.append(f"{mem.episode_type}_{mem.tags[0]}" if mem.tags else "general_pattern")

    # 3. Recent dissonance (from belief network)
    # Would check for recent dissonance events in database
    recent_dissonance = None  # Placeholder—would query from DB

    # 4. Ethical clearance
    ethical_ok = await ethical_check(int(user_id), "daily_objective_generation")

    return {
        **state,
        "goal": {"id": goal_row.id, "title": goal_row.text, "complexity": goal_row.complexity},
        "profile": profile,
        "anchor": anchor,
        "drivers": drivers,
        "bottleneck": bottleneck,
        "strategy_key": strategy_key,
        "last7": last7,
        # NEW: Behavioral intelligence
        "user_beliefs": beliefs,
        "behavioral_patterns": list(set(patterns)),  # Deduplicate
        "recent_dissonance": recent_dissonance,
        "ethical_clearance": ethical_ok,
    }


@traceable(run_type="llm", name="plan_with_llm")
async def plan_with_llm(state: GraphState) -> GraphState:
    goal = state["goal"]
    constraints = state["constraints"]
    profile = state["profile"]
    bottleneck = constraints["bottleneck"]
    strategy_key = state["strategy_key"]
    card = STRATEGIES[strategy_key]

    # NEW: Include behavioral context
    beliefs = state.get("user_beliefs", [])
    patterns = constraints.get("behavioral_patterns", [])
    dissonance = state.get("recent_dissonance")

    recent = "\n".join(f"- {e['date']}: {e['status']}" for e in state["last7"][:3])

    # Build enhanced prompt with behavioral intelligence
    behavioral_context = ""
    if patterns:
        behavioral_context += f"\nLearned patterns: {', '.join(patterns[:3])}"
    if beliefs:
        top_belief = max(beliefs, key=lambda b: b["confidence"])
        behavioral_context += f"\nUser belief: '{top_belief['statement']}' (confidence: {top_belief['confidence']:.2f})"
    if dissonance:
        behavioral_context += f"\nRecent tension detected: {dissonance.get('belief_statement', 'unknown')}"

    if profile is None:
        prompt = (
            "The user has not yet completed calibration (3–5 questions). "
            "Create ONE objective that helps the user finish calibration (≤2 min)."
        )
    else:
        prompt = (
            f"Goal: {goal['title']}\n"
            f"Strategy: {card.name}\n"
            f"Hypothesis: {card.hypothesis}\n"
            f"Signals to log today (in proof.signals): {', '.join(card.signals)}\n"
            f"Recent 3 days:\n{recent}\n"
            f"{behavioral_context}\n"
            f"Constraints: difficulty ≤ {constraints['difficulty_cap']}, time ≤ {constraints['time_budget_cap']} min, "
            f"stabilization={constraints['stabilization']}, max_micro_steps={constraints['max_micro_steps']}.\n"
            "Produce exactly one concrete daily objective that obeys these limits.\n"
            "It must include a success criteria that can be verified today.\n"
            "IMPORTANT: If a user belief contradicts the plan, either respect it or gently challenge it with evidence."
        )

    ai_output = await plan_daily_objective(prompt)
    state["ai_objective"] = ai_output
    return state


# NEW: Multi-agent debate node
@traceable(run_type="chain", name="multi_agent_debate")
async def debate_objective(state: GraphState) -> GraphState:
    """
    Run Observer-Adversary-Synthesizer debate on the proposed objective.
    Ensures quality and prevents rebound effects.
    """
    debate = MultiAgentDebate()
    
    # Prepare context for debate
    user_context = {
        "current_state": state["goal"]["title"],
        "recent_pattern": state["constraints"]["bottleneck"],
        "goal_progress": 50,  # Would calculate actual
        "rejection_rate": 0.1,  # Would calculate from history
        "interventions_today": 0,
        "beliefs": state.get("user_beliefs", [])
    }
    
    result = await debate.debate_intervention(
        user_context=user_context,
        proposed_intervention=state["ai_objective"].get("primary_objective", "unknown"),
        predicted_outcome={
            "expected_improvement": 0.7,
            "confidence": 0.8
        }
    )
    
    state["debate_result"] = result
    
    # If debate rejected, modify objective or mark for review
    if not result.get("proceed", True):
        # Use adversary's concerns to improve objective
        state["ai_objective"]["adjustment_reason"] = (
            state["ai_objective"].get("adjustment_reason", "") + 
            f" [DEBATE: {result.get('adversary_concerns', 'quality concerns')}]"
        )
        # Reduce difficulty
        state["ai_objective"]["difficulty"] = max(1, state["ai_objective"].get("difficulty", 3) - 1)
    
    return state


@traceable(run_type="chain", name="validate_and_store")
async def validate_and_store(state: GraphState) -> Dict[str, Any]:
    raw = state.get("ai_objective", {})
    goal = state["goal"]
    today = state["today"]
    user_id = state["user_id"]

    # NEW: Check ethical clearance one more time
    if not state.get("ethical_clearance", True):
        # Override with gentle wellbeing objective
        raw = {
            "primary_objective": "Take a break and check in with yourself. No work objectives today.",
            "micro_steps": ["Step outside", "Breathe deeply for 2 minutes", "Drink water"],
            "time_budget_minutes": 10,
            "success_criteria": "You've taken a moment for yourself.",
            "difficulty": 1,
            "adjustment_reason": "Ethical pause: system detected elevated stress. Wellbeing first.",
        }

    try:
        obj = DailyObjectiveSchema.model_validate(raw)
    except Exception:
        obj = DailyObjectiveSchema(
            primary_objective="Complete or refine your calibration (3–5 questions, 2 min).",
            micro_steps=["Open calibration", "Answer 3 questions", "Save"],
            time_budget_minutes=5,
            success_criteria="You have saved your calibration.",
            difficulty=1,
            adjustment_reason="Fallback for missing calibration",
        )

    async with async_session() as session:
        # Guardrail 9: idempotent get-or-create
        existing = await session.scalar(
            select(DailyObjective).where(
                DailyObjective.goal_id == goal["id"], 
                DailyObjective.day == today
            )
        )
        if existing:
            state["stored_objective"] = {
                "id": existing.id,
                "goal_id": existing.goal_id,
                "day": existing.day.isoformat(),
                "primary_objective": existing.primary_objective,
                "micro_steps": existing.micro_steps,
                "time_budget_minutes": existing.time_budget_minutes,
                "success_criteria": existing.success_criteria,
                "difficulty": existing.difficulty,
                "adjustment_reason": existing.adjustment_reason,
                "debate_validated": existing.ai_objective_json.get("debate_validated") if existing.ai_objective_json else False,
            }
            return state

        # NEW: Include debate validation in stored JSON
        ai_json = raw.copy()
        ai_json["debate_validated"] = state.get("debate_result", {}).get("proceed", False)
        ai_json["behavioral_patterns_used"] = state.get("behavioral_patterns", [])
        ai_json["beliefs_considered"] = [b["statement"] for b in state.get("user_beliefs", [])]

        stored = DailyObjective(
            goal_id=goal["id"],
            day=today,
            primary_objective=obj.primary_objective,
            micro_steps=obj.micro_steps,
            time_budget_minutes=obj.time_budget_minutes,
            success_criteria=obj.success_criteria,
            difficulty=obj.difficulty,
            adjustment_reason=obj.adjustment_reason,
            ai_objective_json=ai_json,  # Guardrail 22 + NEW fields
        )
        session.add(stored)
        await session.commit()
        await session.refresh(stored)

        # NEW: Create episodic memory of this planning session
        memory = EpisodicMemorySystem(int(user_id))
        await memory.record_episode(
            episode_type="insight",
            what_happened=f"Daily objective planned: {obj.primary_objective[:50]}...",
            behavioral_context={
                "difficulty": obj.difficulty,
                "time_budget": obj.time_budget_minutes,
                "bottleneck": state["constraints"]["bottleneck"],
                "debate_validated": ai_json["debate_validated"]
            },
            emotional_valence=0.3 if obj.difficulty <= 3 else 0.0,  # Neutral to slightly positive
            importance_hint=0.6
        )

    state["stored_objective"] = {
        "id": stored.id,
        "goal_id": stored.goal_id,
        "day": stored.day.isoformat(),
        "primary_objective": stored.primary_objective,
        "micro_steps": stored.micro_steps,
        "time_budget_minutes": stored.time_budget_minutes,
        "success_criteria": stored.success_criteria,
        "difficulty": stored.difficulty,
        "adjustment_reason": stored.adjustment_reason,
        "debate_validated": ai_json["debate_validated"],
        "behavioral_intelligence_used": True,
    }
    return state


def build_daily_objective_graph() -> StateGraph:
    g = StateGraph(GraphState)
    g.add_node("load_state", load_state)
    g.add_node("apply_rules", apply_rules)
    g.add_node("plan_with_llm", plan_with_llm)
    g.add_node("debate_objective", debate_objective)  # NEW
    g.add_node("validate_and_store", validate_and_store)
    
    g.set_entry_point("load_state")
    g.add_edge("load_state", "apply_rules")
    g.add_edge("apply_rules", "plan_with_llm")
    g.add_edge("plan_with_llm", "debate_objective")  # NEW: Debate before storing
    g.add_edge("debate_objective", "validate_and_store")
    g.add_edge("validate_and_store", END)
    return g.compile()


daily_objective_app = build_daily_objective_graph()


@traceable(name="run_daily_objective")
async def run_daily_objective(user_id: UUID, today: date) -> Dict[str, Any]:
    # NEW: Pre-flight ethical check
    ethical_ok = await ethical_check(int(user_id), "daily_objective_generation")
    if not ethical_ok:
        logger.warning(f"Ethical pause active for user {user_id}, returning wellbeing objective")
    
    final = await daily_objective_app.ainvoke(
        {
            "user_id": user_id,
            "today": today,
            # Initialize new fields
            "user_beliefs": None,
            "behavioral_patterns": None,
            "recent_dissonance": None,
            "ethical_clearance": ethical_ok,
        },
        config={
            "tags": ["spirit", "daily_objective", "behavioral_intelligence"],
            "metadata": {
                "user_id": str(user_id),
                "today": str(today),
                "version": "2.0_with_debate"
            }
        },
    )
    return final["stored_objective"]
