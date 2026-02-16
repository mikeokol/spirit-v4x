"""
Behavioral data models for Spirit's ingestion layer.
Defines the schema for longitudinal behavioral observations.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, validator


class ObservationType(str, Enum):
    """Taxonomy of behavioral observations."""
    # Passive sensing
    SCREEN_TIME_AGGREGATE = "screen_time_aggregate"
    APP_USAGE_SESSION = "app_usage_session"
    DEVICE_INTERACTION = "device_interaction"
    
    # Active EMA
    EMA_RESPONSE = "ema_response"
    EMA_DISMISSED = "ema_dismissed"
    
    # Interventions
    INTERVENTION_DELIVERED = "intervention_delivered"
    INTERVENTION_ENGAGED = "intervention_engaged"
    INTERVENTION_IGNORED = "intervention_ignored"


class PrivacyLevel(str, Enum):
    """Data sensitivity classification for edge filtering."""
    PUBLIC = "public"           # Can be shared, aggregated
    ANONYMOUS = "anonymous"     # De-identified but granular
    RESTRICTED = "restricted"   # Only local processing
    SENSITIVE = "sensitive"     # Never leaves device


class AppCategory(str, Enum):
    """Categorization for screen time analysis."""
    PRODUCTIVITY = "productivity"
    COMMUNICATION = "communication"
    SOCIAL_MEDIA = "social_media"
    ENTERTAINMENT = "entertainment"
    HEALTH = "health"
    SHOPPING = "shopping"
    NEWS = "news"
    GAMING = "gaming"
    UTILITY = "utility"
    OTHER = "other"


class ScreenTimeSession(BaseModel):
    """
    A single app usage session from mobile OS APIs.
    This is the atomic unit of screen time data.
    """
    session_id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    
    # Temporal
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    
    # App context (privacy-filtered)
    app_package: Optional[str] = None  # Hashed/anon on edge device
    app_category: AppCategory
    app_name_hash: str  # One-way hash for privacy
    
    # Behavioral context
    entry_point: Optional[str] = None  # Notification, search, habit
    exit_reason: Optional[str] = None  # Interrupted, completed, switched
    
    # Quality metrics
    engagement_score: Optional[float] = Field(None, ge=0, le=1)
    distraction_flag: bool = False  # True if interrupted focus session
    
    # Metadata
    collected_at: datetime = Field(default_factory=datetime.utcnow)
    device_id_hash: str  # Anonymized device identifier
    data_quality_score: float = Field(0.9, ge=0, le=1)
    
    @validator('duration_seconds', always=True)
    def calculate_duration(cls, v, values):
        if v is None and values.get('ended_at') and values.get('started_at'):
            return int((values['ended_at'] - values['started_at']).total_seconds())
        return v


class BehavioralObservation(BaseModel):
    """
    The core scientific unit: what happened, when, and in what context.
    """
    observation_id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Classification
    observation_type: ObservationType
    privacy_level: PrivacyLevel
    
    # Context vector (structured for querying)
    context: Dict[str, Any] = Field(default_factory=dict)
    
    # Behavior vector (what they did)
    behavior: Dict[str, Any] = Field(default_factory=dict)
    
    # Intervention tracking (for causal inference)
    intervention_id: Optional[UUID] = None
    experiment_arm: Optional[str] = None  # 'control', 'treatment_A', etc.
    
    # Proximal outcome (immediate effect)
    outcome: Optional[Dict[str, Any]] = None
    
    # Scientific metadata
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Validation timestamps
    received_at: datetime = Field(default_factory=datetime.utcnow)
    validated_at: Optional[datetime] = None


class EMARequest(BaseModel):
    """Ecological Momentary Assessment trigger."""
    ema_id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    triggered_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Trigger context
    trigger_type: str  # 'vulnerability_detected', 'routine_deviation', 'user_request'
    trigger_confidence: float = Field(..., ge=0, le=1)
    
    # Content
    question_text: str
    response_type: str  # 'likert_5', 'likert_7', 'boolean', 'text', 'numeric'
    response_options: Optional[List[str]] = None
    
    # Timing
    expiry_at: datetime
    responded_at: Optional[datetime] = None
    response_value: Optional[Any] = None
    
    # Outcome
    delivered_successfully: bool = False
    dismissed: bool = False


class UserCausalHypothesis(BaseModel):
    """Scientific model of user behavior."""
    hypothesis_id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # The causal claim
    cause_variable: str
    effect_variable: str
    
    # Statistical evidence
    effect_size: float
    confidence_interval: tuple[float, float]
    p_value: Optional[float] = None
    n_observations: int
    
    # Model metadata
    model_type: str = "linear_regression"
    last_validated_at: Optional[datetime] = None
    falsified: bool = False
    
    # Intervention history
    tested_interventions: List[UUID] = Field(default_factory=list)
