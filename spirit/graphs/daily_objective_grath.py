"""
Daily-objective generation state-machine.
Nodes are pure async functions; edges are defined in compile_graph().
"""
from datetime import date, timedelta
from typing import Any, Dict, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from sqlalchemy.ext.asyncio import AsyncSession
from spirit.db import async_session
from spirit.models import Goal, GoalState, Execution
from spirit.schemas.daily_objective import DailyObjectiveSchema
from spirit.rules.constraint_engine import adjust_for_misses
from spirit.services.openai_client import plan_daily_objective


class GraphState(TypedDict):
    user_id: int
    messages: list[str]  # LangGraph convention for streaming/debug
    active_goal: Optional[Goal]
    recent_executions: list[Execution]
    mode: str  # "trajectory" | "strategic"
    raw_plan: Optional[Dict[str, Any]]
    validated_plan: Optional[DailyObjectiveSchema]


# ---------- 1. load state ----------
async def load_state(state: GraphState) -> GraphState:
    async with async_session() as db:
        # active goal
        goal = await db.get(Goal, state["user_id"])  # simplified lookup
        if not goal or goal.state != GoalState.active:
            raise ValueError("No active goal")

        # last 7 executions
        since = date.today() - timedelta(days=7)
        rows = await db.execute(
            select(Execution)
            .where(Execution.goal_id == goal.id)
            .where(Execution.day >= since)
            .order_by(Execution.day.desc())
        )
        executions = rows.scalars().all()

        # mode gate (placeholder)
        mode = "strategic" if goal.execution_rate > 0.7 else "trajectory"

    return {
        **state,
        "active_goal": goal,
        "recent_executions": executions,
        "mode": mode,
    }


# ---------- 2. apply rules ----------
async def apply_rules(state: GraphState) -> GraphState:
    plan_prompt = adjust_for_misses(
        goal=state["active_goal"],
        executions=state["recent_executions"],
        mode=state["mode"],
    )
    return {**state, "messages": [plan_prompt]}


# ---------- 3. plan with LLM ----------
async def plan_with_llm(state: GraphState) -> GraphState:
    raw = await plan_daily_objective(prompt=state["messages"][0])
    return {**state, "raw_plan": raw}


# ---------- 4. validate & store ----------
async def validate_and_store(state: GraphState) -> GraphState:
    schema = DailyObjectiveSchema.model_validate(state["raw_plan"])
    async with async_session() as db:
        # insert into daily_objectives table (create model if needed)
        from spirit.models import DailyObjective
        row = DailyObjective(
            goal_id=state["active_goal"].id,
            day=date.today(),
            primary_objective=schema.primary_objective,
            micro_steps=schema.micro_steps,
            time_budget_minutes=schema.time_budget_minutes,
            success_criteria=schema.success_criteria,
            difficulty=schema.difficulty,
            adjustment_reason=schema.adjustment_reason,
        )
        db.add(row)
        await db.commit()
    return {**state, "validated_plan": schema}


# ---------- 5. return payload ----------
async def return_payload(state: GraphState) -> GraphState:
    # Graph must return serialisable dict
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
