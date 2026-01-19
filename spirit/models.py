from datetime import datetime
from enum import Enum
from typing import Optional

from sqlmodel import Field, SQLModel

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(unique=True, index=True)
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class GoalState(str, Enum):
    active = "active"
    completed = "completed"
    abandoned = "abandoned"

class Goal(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    text: str  # immutable once set
    created_at: datetime = Field(default_factory=datetime.utcnow)
    state: GoalState = GoalState.active
    # stability metrics for Strategic Mode gate
    execution_rate: float = 0.0  # 0-1 over last 30 days
    days_active: int = 0
    last_metric_update: datetime = Field(default_factory=datetime.utcnow)

class Execution(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    goal_id: int = Field(foreign_key="goal.id")
    day: str = Field(index=True)  # ISO date yyyy-mm-dd
    objective_text: str
    executed: bool
    logged_at: datetime = Field(default_factory=datetime.utcnow)
