from datetime import date as dt_date
from pydantic import BaseModel
from typing import List, Optional, Literal

Status = Literal["done", "miss", "partial"]

class ExecutionSummary(BaseModel):
    day: dt_date
    status: Status
    signal: Optional[int] = None  # 0-10 business signal strength


class ReviewResult(BaseModel):
    decision: Literal["continue", "stabilize", "pivot"]
    adherence_rate_7d: float
    done_days: int
    miss_days: int
    partial_days: int
    signals_sum: int
    difficulty_cap: int
    time_budget_cap: int
    explanation: str
    pivot_to: Optional[str] = None  # new strategy key if decision == pivot
