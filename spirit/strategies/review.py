# spirit/strategies/review.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Literal, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from spirit.models import Execution  # your SQLModel
from spirit.strategies.library import STRATEGIES, StrategyCard, Domain, Decision


@dataclass
class ReviewResult:
    decision: Decision
    adherence_rate_7d: float
    done_days: int
    miss_days: int
    partial_days: int
    signals_sum: Dict[str, float]
    explanation: str
    # suggested next caps (optional)
    difficulty_cap: int
    time_budget_cap: int
    pivot_to: Optional[str] = None


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _sum_signals(executions: List[Execution], keys: List[str]) -> Dict[str, float]:
    out = {k: 0.0 for k in keys}
    for ex in executions:
        proof = ex.proof or {}
        signals = proof.get("signals") or {}
        for k in keys:
            out[k] += _safe_float(signals.get(k, 0))
    return out


def _decision_thresholds(domain: Domain, strategy: StrategyCard, adherence: float, done: int, miss: int, sums: Dict[str, float]) -> Tuple[Decision, Optional[str], str]:
    # 1) Stabilize if unstable
    if miss >= 3 or adherence < 0.5:
        return ("stabilize", None, "Too many misses this week. We stabilize first: smaller objective, lower difficulty, same direction.")

    # 2) Domain-specific pivot checks (7d)
    if domain == "business":
        outreach = sums.get("outreach_sent", 0)
        replies = sums.get("replies", 0)
        calls = sums.get("calls_booked", 0)
        # If consistent effort but no response → pivot inside strategy first; if still dead → pivot strategy
        if outreach >= 50 and replies < 3:
            return ("pivot", "biz_service_first", "High outreach with low replies suggests offer/persona/message mismatch. We pivot the approach.")
        if done >= 5 and calls < 1 and replies < 3:
            return ("pivot", "biz_service_first", "No calls booked after consistent effort. We change the strategy to increase conversations.")
        return ("continue", None, "Signals are compatible with progress. Keep the strategy and slightly raise challenge if it feels easy.")

    if domain == "career":
        apps = sums.get("applications_sent", 0)
        replies = sums.get("replies", 0)
        interviews = sums.get("interviews", 0)
        if apps >= 40 and replies < 2:
            return ("pivot", "car_portfolio_first", "High application volume with low replies suggests targeting/resume signal mismatch. We pivot to proof + targeting fixes.")
        if done >= 5 and interviews < 1 and replies < 2:
            return ("pivot", "car_outreach_first", "Low conversion to interviews. We pivot to warm outreach to increase response rate.")
        return ("continue", None, "Pipeline is producing (or trending). Keep the strategy.")

    if domain == "fitness":
        workouts = sums.get("workouts_completed", 0)
        nutrition = sums.get("nutrition_days_on_plan", 0)
        if workouts >= 4 and nutrition >= 5:
            return ("continue", None, "Strong adherence signals. Continue. If outcomes stall later, we adjust the plan—not the habit.")
        # If adherence ok but low signals, continue but simplify
        return ("continue", None, "We keep direction and focus on consistency. We’ll only pivot after stable adherence.")

    if domain == "creator":
        posts = sums.get("posts_published", 0)
        dist = sums.get("distribution_actions", 0)
        subs = sums.get("subs_gained", 0)
        if posts >= 5 and dist >= 20 and subs <= 0:
            return ("pivot", "cre_quality_first", "High output/distribution with flat growth suggests the content format/hook needs change. We pivot.")
        return ("continue", None, "Output/distribution is trending. Continue and iterate on format.")

    return ("continue", None, "Continue.")


async def review_week(
    db: AsyncSession,
    *,
    goal_id: str,
    domain: Domain,
    strategy_key: str,
    today: date,
) -> ReviewResult:
    start = today - timedelta(days=7)

    rows = await db.scalars(
        select(Execution)
        .where(Execution.goal_id == goal_id)
        .where(Execution.date >= start)
        .order_by(Execution.date.desc())
    )
    executions = rows.all()

    done = sum(1 for e in executions if e.status == "done")
    miss = sum(1 for e in executions if e.status == "miss")
    partial = sum(1 for e in executions if e.status == "partial")
    denom = max(done + miss + partial, 1)
    adherence = done / denom

    strategy = STRATEGIES[strategy_key]
    sums = _sum_signals(executions, strategy.signals)

    decision, pivot_to, explanation = _decision_thresholds(domain, strategy, adherence, done, miss, sums)

    # caps
    if decision == "stabilize":
        difficulty_cap = 4
        time_budget_cap = 60
    else:
        difficulty_cap = 8
        time_budget_cap = 120

    return ReviewResult(
        decision=decision,
        adherence_rate_7d=round(adherence, 3),
        done_days=done,
        miss_days=miss,
        partial_days=partial,
        signals_sum=sums,
        explanation=explanation,
        difficulty_cap=difficulty_cap,
        time_budget_cap=time_budget_cap,
        pivot_to=pivot_to,
    )
