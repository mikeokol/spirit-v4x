from pydantic import BaseModel, Field
from typing import List, Optional

class DailyObjectiveSchema(BaseModel):
    primary_objective: str = Field(..., min_length=5, description="One concrete action for today.")
    micro_steps: List[str] = Field(default_factory=list, max_length=3, description="Up to three small steps.")
    time_budget_minutes: int = Field(..., ge=5, le=180, description="Estimated time in minutes.")
    success_criteria: str = Field(..., min_length=5, description="How success is verified.")
    difficulty: int = Field(..., ge=1, le=10, description="Perceived difficulty from 1–10.")
    adjustment_reason: Optional[str] = Field(None, description="Why this objective was scaled up/down.")
