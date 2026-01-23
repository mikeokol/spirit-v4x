from pydantic import BaseModel, Field, PositiveInt

class RealityAnchorSchema(BaseModel):
    offer: str = Field(..., min_length=5, description="What you are selling")
    target_customer: str = Field(..., min_length=5, description="Who you are selling to")
    channel: str = Field(..., description="Primary outreach channel")
    price: PositiveInt = Field(..., description="Price in cents")
    weekly_lead_target: PositiveInt = Field(..., description="New leads per week")
    weekly_conversation_target: PositiveInt = Field(..., description="Qualified conversations per week")
    weekly_close_target: PositiveInt = Field(..., description="Closed deals per week")
