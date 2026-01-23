from datetime import datetime, date
from enum import Enum
from typing import Optional, List
from uuid import UUID
from sqlmodel import Field, SQLModel
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import Column

class User(SQLModel, table=True):
    id: Optional[UUID] = Field(default=None, primary_key=True)
    email: str = Field(unique=True, index=True)
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class GoalState(str, Enum):
    active = "active"
    completed = "completed"
    abandoned = "abandoned"

class Complexity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

class Goal(SQLModel, table=True):
    id: Optional[UUID] = Field(default=None, primary_key=True)
    user_id: UUID = Field(index=True)
    text: str
    complexity: Complexity = Complexity.medium
    created_at: datetime = Field(default_factory=datetime.utcnow)
    state: GoalState = GoalState.active
    execution_rate: float = 0.0
    days_active: int = 0
    last_metric_update: datetime = Field(default_factory=datetime.utcnow)

class GoalProfile(SQLModel, table=True):
    id: Optional[UUID] = Field(default=None, primary_key=True)
    goal_id: UUID = Field(foreign_key="goal.id", unique=True)
    time_budget_weekly: int
    money_budget_monthly: Optional[int] = None
    constraints: List[str] = Field(sa_column=Column(JSONB))
    starting_point: str
    success_definition: str
    confidence: Optional[int] = Field(None, ge=1, le=10)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class RealityAnchor(SQLModel, table=True):
    id: Optional[UUID] = Field(default=None, primary_key=True)
    goal_id: UUID = Field(foreign_key="goal.id", unique=True)
    offer: str
    target_customer: str
    channel: str
    price: int
    weekly_lead_target: int
    weekly_conversation_target: int
    weekly_close_target: int
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Execution(SQLModel, table=True):
    id: Optional[UUID] = Field(default=None, primary_key=True)
    goal_id: UUID = Field(foreign_key="goal.id")
    day: date = Field(index=True)
    objective_text: str
    executed: bool
    logged_at: datetime = Field(default_factory=datetime.utcnow)

class DailyObjective(SQLModel, table=True):
    id: Optional[UUID] = Field(default=None, primary_key=True)
    goal_id: UUID = Field(foreign_key="goal.id")
    day: date = Field(unique=True, index=True)
    primary_objective: str
    micro_steps: List[str] = Field(sa_column=Column(JSONB))
    time_budget_minutes: int
    success_criteria: str
    difficulty: int
    adjustment_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
