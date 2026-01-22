"""
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
from spirit.models import Goal, GoalState, Execution
from spirit.schemas.daily_objective import DailyObjectiveSchema
from spirit.rules.constraint_engine import adjust_for_misses
from spirit.services.openai_client import plan_daily_objective


class GraphState(TypedDict):
    user_id: int
    messages: list[str]  # LangGraph convention for streaming/debug
    goal: Optional[Dict[str, Any]]
    mode: Optional[Dict[str, str]]
    last7: list[Dict[str, Any]]
    raw_plan: Optional[Dict[str, Any]]
    validated_plan: Optional[DailyObjectiveSchema]


# ---------- 1. load_state ----------
async def load_state(state: GraphState) -> GraphState:
    user_id = state["user_id"]
    today = date.today()
    start = today - timedelta(days=7)

    async with async_session() as session:
        # ----- active goal -----
        goal_row = await session.scalar(
            select(Goal).where(Goal.user_id == user_id, Goal.state == GoalState.active)
        )
        if not goal_row:
            raise ValueError("No active goal")

        # ----- mode state (placeholder until you have a Mode table) -----
        mode = {
            "constraint_level": "observer",  # or query real table later
            "strategic_state": "locked",
        }

        # ----- last 7 executions -----
        rows = await session.scalars(
            select(Execution)
            .where(Execution.goal_id == goal_row.id)
            .where(Execution.day >= start.isoformat())
            .order_by(Execution.day.desc())
        )
        last7 = [
            {"date": r.day, "status": "done" if r.executed else "miss"}
            for r in rows.all()
        ]

    return {
        **state,
        "goal": {
            "id": goal_row.id,
            "title": goal_row.text,
            "success_metric": "execution_rate",  # placeholder
            "target_value": 0.7,
            "deadline": None,
        },
        "mode": mode,
        "last7": last7,
    }


# ---------- 2. apply_rules (deterministic) ----------
async def apply_rules(state: GraphState) -> GraphState:
    last7 = state.get("last7", [])
    mode = state.get("mode", {})

    misses = sum(1 for e in last7 if e.get("status") in ("missed", "partial"))
    did_yesterday = any(e.get("status") == "done" for e in last7[-1:])

    constraint_level = mode.get("constraint_level", "observer")

    stabilization = misses >= 3

    constraints = {
        "constraint_level": constraint_level,
        "stabilization": stabilization,
        "difficulty_cap": 4 if stabilization else (6 if not did_yesterday else 8),
        "max_micro_steps": 3,
        "time_budget_cap": 60 if stabilization else 120,
    }

    state["constraints"] = constraints
    return state


# ---------- 3. plan with LLM ----------
async def plan_with_llm(state: GraphState) -> GraphState:
    raw = await plan_daily_objective(prompt=state["messages"][0])
    return {**state, "raw_plan": raw}


# ---------- 4. validate & store ----------
async def validate_and_store(state: GraphState) -> GraphState:
    schema = DailyObjectiveSchema.model_validate(state["raw_plan"])
    async with async_session() as session:
        from spirit.models import DailyObjective
        row = DailyObjective(
            goal_id=state["goal"]["id"],
            day=date.today(),
            primary_objective=schema.primary_objective,
            micro_steps=schema.micro_steps,
            time_budget_minutes=schema.time_budget_minutes,
            success_criteria=schema.success_criteria,
            difficulty=schema.difficulty,
            adjustment_reason=schema.adjustment_reason,
        )
        session.add(row)
        await session.commit()
    return {**state, "validated_plan": schema}


# ---------- 5. return payload ----------
async def return_payload(state: GraphState) -> GraphState:
    return {"objective": state["validated_plan"].model_dump()}


# ---------- compile graph ----------
def compile_graph() -> StateGraph:
    workflow = StateGraph(GraphState)
    workflow.add_node("load_state", load_state)
    workflow.add_node("apply_rules", apply_rules)
    workflow.add_node("plan_with_llm", plan_with_llm)
    workflow.add_node("validate_and_store", validate_and_store)
    workflow.add_node("return_payload", return_payload)

    workflow.add_edge("load_state", "apply_rules")
    workflow.add_edge("apply_rules", "plan_with_llm")
    workflow.add_edge("plan_with_llm", "validate_and_store")
    workflow.add_edge("validate_and_store", "return_payload")
    workflow.add_edge("return_payload", END)

    workflow.set_entry_point("load_state")
    return workflow.compile()


daily_objective_graph = compile_graph()
