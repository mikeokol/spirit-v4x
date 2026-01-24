"""
Daily-objective generation state-machine with Reality-Anchor policy + LangSmith traces.
"""
from datetime import date, timedelta
from typing import Any, Dict, Optional
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
from uuid import UUID
import logging

logger = logging.getLogger("spirit")

class GraphState(TypedDict):
    user_id: UUID
    today: date
    goal: Optional[Dict[str, Any]]
    profile: Optional[Dict[str, Any]]  # goal_profile
    anchor: Optional[RealityAnchorSchema]
    drivers: Optional[Dict[str, float]]
    bottleneck: Optional[str]
    strategy_key: Optional[str]  # <-- NEW
    last7: list[Dict[str, Any]]
    constraints: Optional[Dict[str, Any]]
    ai_objective: Optional[Dict[str, Any]]
    stored_objective: Optional[Dict[str, Any]]


@traceable(run_type="chain", name="apply_rules")
def apply_rules(state: GraphState) -> GraphState:
    last7 = state.get("last7", [])
    misses = sum(1 for e in last7 if e.get("status") == "miss")
    stabilization = misses >= 3
    drivers = state["drivers"]
    bottleneck = state["bottleneck"]

    if stabilization:
        difficulty_cap, time_cap = 2, 30
    elif bottleneck == "lead_gen":
        difficulty_cap, time_cap = 3, 45
    elif bottleneck == "conversation":
        difficulty_cap, time_cap = 4, 60
    else:  # close
        difficulty_cap, time_cap = 5, 75

    state["constraints"] = {
        "stabilization": stabilization,
        "difficulty_cap": difficulty_cap,
        "time_budget_cap": time_cap,
        "max_micro_steps": 3,
        "bottleneck": bottleneck,
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

        # NEW: load strategy from ModeState
        mode_row = await session.scalar(select(ModeState).where(ModeState.user_id == user_id))
        if mode_row and mode_row.strategy_key:
            strategy_key = mode_row.strategy_key
        else:
            # infer from goal text or default
            domain = mode_row.domain if mode_row and mode_row.domain else "business"
            strategy_key = DEFAULT_STRATEGY_BY_DOMAIN[domain]

        rows = await session.scalars(
            select(Execution)
            .where(Execution.goal_id == goal_row.id)
            .where(Execution.day >= start)
            .order_by(Execution.day.desc())
        )
        last7 = [{"date": r.day, "status": "done" if r.executed else "miss"} for r in rows.all()]

    return {
        **state,
        "goal": {"id": goal_row.id, "title": goal_row.text, "complexity": goal_row.complexity},
        "profile": profile,
        "anchor": anchor,
        "drivers": drivers,
        "bottleneck": bottleneck,
        "strategy_key": strategy_key,
        "last7": last7,
    }


@traceable(run_type="llm", name="plan_with_llm")
async def plan_with_llm(state: GraphState) -> GraphState:
    goal = state["goal"]
    constraints = state["constraints"]
    profile = state["profile"]
    bottleneck = constraints["bottleneck"]
    strategy_key = state["strategy_key"]
    card = STRATEGIES[strategy_key]

    recent = "\n".join(f"- {e['date']}: {e['status']}" for e in state["last7"][:3])

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
            f"Constraints: difficulty ≤ {constraints['difficulty_cap']}, time ≤ {constraints['time_budget_cap']} min, "
            f"stabilization={constraints['stabilization']}, max_micro_steps={constraints['max_micro_steps']}.\n"
            "Produce exactly one concrete daily objective that obeys these limits.\n"
            "It must include a success criteria that can be verified today."
        )

    ai_output = await plan_daily_objective(prompt)
    state["ai_objective"] = ai_output
    return state


@traceable(run_type="chain", name="validate_and_store")
async def validate_and_store(state: GraphState) -> GraphState:
    raw = state.get("ai_objective", {})
    goal = state["goal"]
    today = state["today"]

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
        existing = await session.scalar(select(DailyObjective).where(DailyObjective.goal_id == goal["id"], DailyObjective.day == today))
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
            }
            return state

        stored = DailyObjective(
            goal_id=goal["id"],
            day=today,
            primary_objective=obj.primary_objective,
            micro_steps=obj.micro_steps,
            time_budget_minutes=obj.time_budget_minutes,
            success_criteria=obj.success_criteria,
            difficulty=obj.difficulty,
            adjustment_reason=obj.adjustment_reason,
            ai_objective_json=raw,  # Guardrail 22
        )
        session.add(stored)
        await session.commit()
        await db.refresh(stored)

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
    }
    return state


def build_daily_objective_graph() -> StateGraph:
    g = StateGraph(GraphState)
    g.add_node("load_state", load_state)
    g.add_node("apply_rules", apply_rules)
    g.add_node("plan_with_llm", plan_with_llm)
    g.add_node("validate_and_store", validate_and_store)
    g.set_entry_point("load_state")
    g.add_edge("load_state", "apply_rules")
    g.add_edge("apply_rules", "plan_with_llm")
    g.add_edge("plan_with_llm", "validate_and_store")
    g.add_edge("validate_and_store", END)
    return g.compile()

daily_objective_app = build_daily_objective_graph()

async def run_daily_objective(user_id: UUID, today: date) -> Dict[str, Any]:
    final = await daily_objective_app.ainvoke({"user_id": user_id, "today": today})
    return final["stored_objective"]
