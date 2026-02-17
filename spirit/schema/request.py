"""
Request schemas for Spirit API.
Includes authentication + behavioral intelligence + proactive systems.
"""

from datetime import datetime, date
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, EmailStr, Field
from uuid import UUID


# ==========================================
# AUTHENTICATION SCHEMAS (Existing - Preserved)
# ==========================================

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


# ==========================================
# BEHAVIORAL DATA INGESTION SCHEMAS (New)
# ==========================================

class ScreenTimeSessionRequest(BaseModel):
    """Mobile device sends this for screen time tracking."""
    session_id: UUID
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    app_package: Optional[str] = None  # Hashed on device
    app_category: str  # productivity, social_media, entertainment, etc.
    app_name_hash: str
    entry_point: Optional[str] = None  # notification, search, habit
    exit_reason: Optional[str] = None  # interrupted, completed, switched
    engagement_score: Optional[float] = Field(None, ge=0, le=1)
    distraction_flag: bool = False
    device_id_hash: str


class BehavioralObservationRequest(BaseModel):
    """Real-time behavioral observation from mobile."""
    observation_id: UUID = Field(default_factory=UUID)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    observation_type: Literal[
        "screen_time_aggregate",
        "app_usage_session",
        "device_interaction",
        "ema_response",
        "ema_dismissed",
        "intervention_delivered",
        "intervention_engaged",
        "intervention_ignored"
    ]
    privacy_level: Literal["public", "anonymous", "restricted", "sensitive"]
    context: Dict[str, Any] = Field(default_factory=dict)  # location_type, time_of_day, etc.
    behavior: Dict[str, Any] = Field(default_factory=dict)  # app_switches, focus_score, etc.
    intervention_id: Optional[UUID] = None
    experiment_arm: Optional[str] = None  # control, treatment_A, etc.
    outcome: Optional[Dict[str, Any]] = None


class EMAResponseRequest(BaseModel):
    """User responds to Ecological Momentary Assessment."""
    ema_id: UUID
    response_value: Dict[str, Any]  # {mood: 6, energy: 5} or {answer: "yes"}
    responded_at: Optional[datetime] = None


# ==========================================
# CAUSAL DISCOVERY REQUEST SCHEMAS (New)
# ==========================================

class CausalDiscoveryRequest(BaseModel):
    """Request causal effect discovery."""
    cause_variable: str  # e.g., "evening_social_media"
    effect_variable: str  # e.g., "next_morning_focus"
    method: Optional[Literal[
        "backdoor_adjustment",
        "diff_in_diff",
        "instrumental_variable",
        "synthetic_control",
        "causal_forest",
        "regression_discontinuity"
    ]] = None  # Auto-selected if not specified
    lag_hours: int = 8


class BatchCausalDiscoveryRequest(BaseModel):
    """Discover all causal relationships among variables."""
    variables: List[str]
    min_confidence: float = Field(0.6, ge=0, le=1)


class ValidateHypothesisRequest(BaseModel):
    """Re-validate existing hypothesis with multiple methods."""
    hypothesis_id: str


# ==========================================
# MEMORY & BELIEF REQUEST SCHEMAS (New)
# ==========================================

class RecordMemoryEpisodeRequest(BaseModel):
    """Record a significant moment in user's journey."""
    episode_type: Literal["breakthrough", "struggle", "insight", "milestone", "routine"]
    what_happened: str
    emotional_valence: float = Field(..., ge=-1, le=1)  # -1 negative to +1 positive
    context: Dict[str, Any] = Field(default_factory=dict)
    importance_hint: float = Field(0.5, ge=0, le=1)


class RetrieveMemoriesRequest(BaseModel):
    """Query for relevant memories."""
    query: Optional[str] = None  # Semantic search query
    context: Dict[str, Any] = Field(default_factory=dict)
    n_results: int = Field(3, ge=1, le=20)
    time_horizon_days: Optional[int] = None


class AddBeliefRequest(BaseModel):
    """Add a new belief stated by user."""
    statement: str  # "I work best at night"
    belief_type: Literal["causal", "identity", "strategy", "constraint"]
    initial_confidence: float = Field(0.7, ge=0, le=1)


class UpdateBeliefEvidenceRequest(BaseModel):
    """Update belief with new evidence."""
    belief_statement: str
    evidence: Dict[str, Any]
    supports: bool  # True if evidence supports belief, False if contradicts


# ==========================================
# PROACTIVE & REAL-TIME REQUEST SCHEMAS (New)
# ==========================================

class StartProactiveLoopRequest(BaseModel):
    """Start autonomous proactive monitoring."""
    intensity: Literal["gentle", "balanced", "intensive"] = "balanced"


class ProcessRealtimeRequest(BaseModel):
    """Process observation through real-time pipeline."""
    observation: Dict[str, Any]  # Raw behavioral observation


class ScheduleInterventionRequest(BaseModel):
    """Manually schedule an intervention."""
    content: Dict[str, str]  # {title, body, action_url?}
    priority: Literal["critical", "high", "normal", "low", "digest"] = "normal"
    notification_type: str
    deliver_at: Optional[datetime] = None  # If None, uses optimal timing
    channel: Optional[Literal["push", "in_app", "sms", "email"]] = None


class PredictionFeedbackRequest(BaseModel):
    """User feedback on prediction accuracy."""
    prediction_id: str
    was_accurate: bool
    what_actually_happened: str


# ==========================================
# INTELLIGENCE & ANALYSIS REQUEST SCHEMAS (New)
# ==========================================

class AnalyzeObservationRequest(BaseModel):
    """Request LangGraph agent analysis."""
    observation: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None


class PredictGoalOutcomeRequest(BaseModel):
    """Predict probability of goal achievement."""
    goal_id: UUID
    horizon_days: int = Field(7, ge=1, le=90)


class OptimizeInterventionRequest(BaseModel):
    """Use multi-armed bandit to select optimal intervention."""
    context: Dict[str, Any]
    available_interventions: List[str]


class GenerateReportRequest(BaseModel):
    """Request scientific report generation."""
    days: int = Field(7, ge=1, le=30)


# ==========================================
# ETHICAL & DELIVERY REQUEST SCHEMAS (New)
# ==========================================

class ManualKillSwitchRequest(BaseModel):
    """User or researcher triggers ethical pause."""
    reason: str
    duration_hours: int = Field(24, ge=1, le=168)


class RecordInterventionFeedbackRequest(BaseModel):
    """Record user response to intervention."""
    intervention_id: str
    response: Literal["engaged", "dismissed", "ignored"]
    response_time_seconds: Optional[float] = None
    behavior_after: Optional[Dict[str, Any]] = None
    rating: Optional[int] = Field(None, ge=1, le=5)
    feedback_text: Optional[str] = None


class NotificationPreferenceRequest(BaseModel):
    """Update notification preferences."""
    preferred_channels: List[Literal["push", "in_app", "sms", "email"]]
    quiet_hours_start: Optional[int] = Field(None, ge=0, le=23)  # 0-23
    quiet_hours_end: Optional[int] = Field(None, ge=0, le=23)
    max_daily_interventions: int = Field(5, ge=1, le=20)


# ==========================================
# STRATEGIC PLANNING REQUEST SCHEMAS (New)
# ==========================================

class Generate12WeekPlanRequest(BaseModel):
    """Request 12-week strategic plan."""
    goal_id: UUID


class ForceStrategicRecheckRequest(BaseModel):
    """Force recalculation of strategic status."""
    pass


# ==========================================
# DAILY OBJECTIVE REQUEST SCHEMAS (New)
# ==========================================

class GenerateDailyObjectiveRequest(BaseModel):
    """Request daily objective generation."""
    target_date: Optional[date] = None  # Defaults to today


class CompleteObjectiveRequest(BaseModel):
    """Mark daily objective as complete."""
    objective_id: int
    notes: Optional[str] = None
    actual_time_minutes: Optional[int] = None
