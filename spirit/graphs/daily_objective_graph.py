from spirit.graphs.daily_objective_graph import daily_objective_graph
""
Daily-objective generation state-machine.
Nodes are pure async functions; edges are defined in compile_graph().
"""
from datetime import date, timedelta
from typing import Any, Dict, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from spirit.db import async_session
from spirit.models import Goal, GoalState, Execution, DailyObjective
from spirit.schemas.daily_objective import DailyObjectiveSchema
from spirit.services.openai_client import plan_daily_objective


class GraphState(TypedDict):
    user_id: int
    today: str  # ISO date
    goal: Optional[Dict[str, Any]]
    mode: Optional[Dict[str, str]]
    last7: list[Dict[str, Any]]
    constraints: Optional[Dict[str, Any]]  # <- added
    ai_objective: Optional[Dict[str, Any]]
    stored_objective: Optional[Dict[str, Any]]


# ---------- 1. load_state ----------
async def load_state(state: GraphState) -> GraphState:
    user_id = state["user_id"]
    today = date.fromisoformat(state["today"])
    start = today - timedelta(days=7)

    async with async_session() as session:
        goal_row = await session.scalar(
            select(Goal).where(Goal.user_id == user_id, Goal.state == GoalState.active)
        )
        if not goal_row:
            raise ValueError("No active goal")

        mode = {"constraint_level": "observer", "strategic_state": "locked"}

        rows = await session.scalars(
            select(Execution)
            .where(Execution.goal_id == goal_row.id)
            .where(Execution.day >= start.isoformat())
            .order_by(Execution.day.desc())
        )
        last7 = [{"date": r.day, "status": "done" if r.executed else "miss"} for r in rows.all()]

    return {
        **state,
        "goal": {"id": goal_row.id, "title": goal_row.text},
        "mode": mode,
        "last7": last7,
    }


# ---------- 2. apply_rules ----------
async def apply_rules(state: GraphState) -> GraphState:
    last7 = state.get("last7", [])
    mode = state.get("mode", {})
    misses = sum(1 for e in last7 if e.get("status") == "miss")
    stabilization = misses >= 3
    constraints = {
        "constraint_level": mode.get("constraint_level", "observer"),
        "stabilization": stabilization,
        "difficulty_cap": 4 if stabilization else 8,
        "max_micro_steps": 3,
        "time_budget_cap": 60 if stabilization else 120,
    }
    state["constraints"] = constraints
    return state


# ---------- 3. plan_with_llm ----------
async def plan_with_llm(state: GraphState) -> GraphState:
    goal = state["goal"]
    constraints = state["constraints"]
    last7 = state.get("last7", [])
    recent = "\n".join(f"- {e['date']}: {e['status']}" for e in last7[:3])
    prompt = (
        f"Goal: {goal['title']}\nRecent 3 days:\n{recent}\n"
        f"Constraints: difficulty ≤ {constraints['difficulty_cap']}, "
        f"time ≤ {constraints['time_budget_cap']} min, "
        f"stabilization={constraints['stabilization']}, "
        f"max_micro_steps={constraints['max_micro_steps']}.\n"
        "Produce exactly one concrete daily objective that obeys these limits."
    )
    ai_output = await plan_daily_objective(prompt)
    state["ai_objective"] = ai_output
    return state


# ---------- 4. validate_and_store ----------
async def validate_and_store(state: GraphState) -> GraphState:
    raw = state.get("ai_objective", {})
    goal = state["goal"]
    today = date.fromisoformat(state["today"])

    try:
        obj = DailyObjectiveSchema.model_validate(raw)
    except Exception:
        obj = DailyObjectiveSchema(
            primary_objective="Do the smallest possible action that moves your goal forward (10 minutes).",
            micro_steps=["Start a 10-minute timer", "Do one concrete step", "Stop when timer ends"],
            time_budget_minutes=10,
            success_criteria="You can name the one step you completed.",
            difficulty=1,
            adjustment_reason="Fallback due to invalid model output",
        )

    async with async_session() as session:
        stored = DailyObjective(
            goal_id=goal["id"],
            day=today.isoformat(),
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

    state["stored_objective"] = stored
    return state


# ---------- wire & run ----------
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

async def run_daily_objective(user_id: int, today_iso: str) -> Dict[str, Any]:
    final = await daily_objective_app.ainvoke({"user_id": user_id, "today": today_iso})
    return final["stored_objective"]

"""
