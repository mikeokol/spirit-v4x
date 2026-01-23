"""
Daily-objective generation state-machine with Reality-Anchor policy.
"""
from datetime import date, timedelta
from typing import Any, Dict, Optional
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from sqlalchemy import select
from spirit.db import async_session
from spirit.models import Goal, GoalState, Execution, DailyObjective, RealityAnchor
from spirit.schemas.daily_objective import DailyObjectiveSchema
from spirit.schemas.reality_anchor import RealityAnchorSchema
from spirit.services.openai_client import plan_daily_objective
from spirit.services.decomposer import driver_math, bottleneck_pick


class GraphState(TypedDict):
    user_id: int
    today: date
    goal: Optional[Dict[str, Any]]
    anchor: Optional[RealityAnchorSchema]
    drivers: Optional[Dict[str, float]]
    bottleneck: Optional[str]
    last7: list[Dict[str, Any]]
    constraints: Optional[Dict[str, Any]]
    ai_objective: Optional[Dict[str, Any]]
    stored_objective: Optional[Dict[str, Any]]


def apply_rules(state: GraphState) -> GraphState:
    last7 = state.get("last7", [])
    misses = sum(1 for e in last7 if e.get("status") == "miss")
    stabilization = misses >= 3
    drivers = state["drivers"]
    bottleneck = state["bottleneck"]

    # difficulty/time cap based on bottleneck + stabilization
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


async def load_state(state: GraphState) -> GraphState:
    user_id = state["user_id"]
    today = state["today"]
    start = today - timedelta(days=7)

    async with async_session() as session:
        goal_row = await session.scalar(
            select(Goal).where(Goal.user_id == user_id, Goal.state == GoalState.active)
        )
        if not goal_row:
            raise ValueError("No active goal")

        anchor_row = await session.scalar(
            select(RealityAnchor).where(RealityAnchor.goal_id == goal_row.id)
        )
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
            bottleneck = bottleneck_pick(last7=[], drivers=drivers)  # no history yet
        else:
            # Days 1-3: create anchors
            anchor = None
            drivers = None
            bottleneck = "anchor_creation"

        rows = await session.scalars(
            select(Execution)
            .where(Execution.goal_id == goal_row.id)
            .where(Execution.day >= start)
            .order_by(Execution.day.desc())
        )
        last7 = [{"date": r.day, "status": "done" if r.executed else "miss"} for r in rows.all()]

    return {
        **state,
        "goal": {"id": goal_row.id, "title": goal_row.text},
        "anchor": anchor,
        "drivers": drivers,
        "bottleneck": bottleneck,
        "last7": last7,
    }


async def plan_with_llm(state: GraphState) -> GraphState:
    goal = state["goal"]
    constraints = state["constraints"]
    anchor = state["anchor"]
    bottleneck = constraints["bottleneck"]

    if anchor is None:
        prompt = (
            "The user has not yet defined business anchors (offer, customer, channel, price, weekly targets). "
            "Create ONE concrete objective that helps the user write these anchors."
        )
    else:
        drivers = state["drivers"]
        prompt = (
            f"Business anchors: {anchor.offer} to {anchor.target_customer} via {anchor.channel} at ${anchor.price/100}.\n"
            f"Driver targets: {drivers['daily_leads']:.1f} leads, {drivers['daily_conversations']:.1f} convos, {drivers['daily_closes']:.1f} closes per day.\n"
            f"Current bottleneck: {bottleneck}. "
            f"Constraints: difficulty ≤ {constraints['difficulty_cap']}, time ≤ {constraints['time_budget_cap']} min, ≤3 micro-steps.\n"
            "Produce exactly one concrete daily objective that reduces this bottleneck."
        )

    ai_output = await plan_daily_objective(prompt)
    state["ai_objective"] = ai_output
    return state


async def validate_and_store(state: GraphState) -> GraphState:
    raw = state.get("ai_objective", {})
    goal = state["goal"]
    today = state["today"]

    try:
        obj = DailyObjectiveSchema.model_validate(raw)
    except Exception:
        obj = DailyObjectiveSchema(
            primary_objective="Write or refine your business anchors (offer, customer, channel, price, weekly targets).",
            micro_steps=["Draft offer sentence", "Define target customer", "Set weekly lead target"],
            time_budget_minutes=30,
            success_criteria="You have written all 7 anchor fields.",
            difficulty=1,
            adjustment_reason="Fallback for anchor-creation phase",
        )

    async with async_session() as session:
        stored = DailyObjective(
            goal_id=goal["id"],
            day=today,
            primary_objective=obj.primary_objective,
            micro_steps=obj.micro_steps,
            time_budget_minutes=obj.time_budget_minutes,
            success_criteria=obj.success_criteria,
            difficulty=obj.difficulty,
            adjustment_reason=obj.adjustment_reason,
        )
        session.add(stored)
        await session.commit()
        await session.refresh(stored)

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

async def run_daily_objective(user_id: int, today: date) -> Dict[str, Any]:
    final = await daily_objective_app.ainvoke({"user_id": user_id, "today": today})
    return final["stored_objective"]
