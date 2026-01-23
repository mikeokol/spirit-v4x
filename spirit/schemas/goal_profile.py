from pydantic import BaseModel, Field, conlist
from typing import List, Optional, Literal
from uuid import UUID

Constraint = Literal["time", "energy-health", "skill-gap", "money", "environment", "confidence", "consistency"]

class GoalProfileCreate(BaseModel):
    time_budget_weekly: int = Field(..., ge=1, le=168, description="Hours per week user can commit")
    money_budget_monthly: Optional[int] = Field(None, ge=0, description="Monthly budget in cents")
    constraints: conlist(Constraint, max_length=3) = Field(..., description="Biggest limitations")
    starting_point: str = Field(..., min_length=3, max_length=60, description="Where user is today")
    success_definition: str = Field(..., min_length=5, max_length=140, description="1-sentence done-state")
    confidence: Optional[int] = Field(None, ge=1, le=10, description="Self-rated confidence 1–10")

class GoalProfileRead(GoalProfileCreate):
    id: UUID
    goal_id: UUID
    created_at: str
