from datetime import date
from typing import List
from spirit.models import Goal, Execution

def adjust_for_misses(*, goal: Goal, executions: List[Execution], mode: str) -> str:
    """
    Deterministic rule layer: simplify or scale objective based on recent reality.
    Returns a single prompt string for the LLM.
    """
    last_3 = [e for e in executions if e.day >= (date.today() - timedelta(days=3))]
    misses = sum(1 for e in last_3 if not e.executed)

    if misses >= 2:
        difficulty_hint = "difficulty=2"
        time_hint = "time_budget_minutes=15"
        reason = "Recent misses detected—simplify."
    elif mode == "strategic" and goal.execution_rate > 0.8:
        difficulty_hint = "difficulty=7"
        time_hint = "time_budget_minutes=90"
        reason = "Stable trajectory—push leverage."
    else:
        difficulty_hint = "difficulty=5"
        time_hint = "time_budget_minutes=45"
        reason = "Maintain momentum."

    return (
        f"Primary goal: {goal.text}\n"
        f"Today is {date.today().isoformat()}.\n"
        f"{reason}\n"
        f"Constraints: {difficulty_hint}, {time_hint}, micro_steps ≤ 3."
    )
