from datetime import datetime
from pydantic import BaseModel
from spirit.models import GoalState

class GoalRead(BaseModel):
    id: int
    user_id: int
    text: str
    created_at: datetime
    state: GoalState

    class Config:
        from_attributes = True


class ExecutionRead(BaseModel):
    id: int
    goal_id: int
    day: str
    objective_text: str
    executed: bool
    logged_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
