"""
Behavioral data models for Spirit's ingestion layer.
Defines the schema for longitudinal behavioral observations stored in Supabase.
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
    engagement_score: Optional[float] = Field(None, ge=0, le=1)  # Derived from interactions
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
    Stored in Supabase 'behavioral_observations' table.
    """
    observation_id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Classification
    observation_type: ObservationType
    privacy_level: PrivacyLevel
    
    # Context vector (structured for querying)
    context: Dict[str, Any] = Field(default_factory=dict)
    """
    Example context:
    {
        "location_type": "home",  # home/work/transit/social
        "time_of_day": "morning",  # night/morning/afternoon/evening
        "day_of_week": 1,  # Monday
        "social_context": "alone",  # alone/with_others/unknown
        "device_state": "unlocked",  # locked/unlocked/dozing
        "battery_level": 0.45,
        "last_interaction_minutes_ago": 5
    }
    """
    
    # Behavior vector (what they did)
    behavior: Dict[str, Any] = Field(default_factory=dict)
    """
    Example behavior:
    {
        "app_category": "social_media",
        "session_duration_sec": 320,
        "scroll_velocity": 45.2,  # pixels/sec
        "tap_frequency": 12,  # taps per minute
        "typing_present": false,
        "notification_interactions": 3
    }
    """
    
    # Intervention tracking (for causal inference)
    intervention_id: Optional[UUID] = None
    experiment_arm: Optional[str] = None  # 'control', 'treatment_A', etc.
    
    # Proximal outcome (immediate effect)
    outcome: Optional[Dict[str, Any]] = None
    """
    Example outcome:
    {
        "next_session_app_category": "productivity",
        "time_to_next_unlock_minutes": 45,
        "ema_response": {"mood": 6, "energy": 5}
    }
    """
    
    # Scientific metadata
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)
    """
    {
        "edge_processing_latency_ms": 120,
        "inference_confidence": 0.85,
        "sensor_fusion_sources": ["screen", "accel"],
        "privacy_noise_added": 0.01
    }
    """
    
    # Validation timestamps
    received_at: datetime = Field(default_factory=datetime.utcnow)
    validated_at: Optional[datetime] = None


class EMARequest(BaseModel):
    """
    Ecological Momentary Assessment trigger.
    Generated by edge device when vulnerability window detected.
    """
    ema_id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    triggered_at: datetime
    
    # Trigger context
    trigger_type: str  # 'vulnerability_detected', 'routine_deviation', 'user_request'
    trigger_confidence: float = Field(..., ge=0, le=1)
    
    # Content
    question_text: str
    response_type: str  # 'likert_5', 'likert_7', 'boolean', 'text', 'numeric'
    response_options: Optional[List[str]] = None
    
    # Timing
    expiry_at: datetime  # When the EMA becomes invalid
    responded_at: Optional[datetime] = None
    response_value: Optional[Any] = None
    
    # Outcome
    delivered_successfully: bool = False
    dismissed: bool = False


class UserCausalHypothesis(BaseModel):
    """
    Scientific model of user behavior.
    Stored in Supabase 'causal_graph' table.
    """
    hypothesis_id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # The causal claim
    cause_variable: str  # e.g., 'late_night_social_media'
    effect_variable: str  # e.g., 'next_morning_focus_score'
    
    # Statistical evidence
    effect_size: float  # Estimated causal impact
    confidence_interval: tuple[float, float]  # [lower, upper]
    p_value: Optional[float] = None
    n_observations: int
    
    # Model metadata
    model_type: str = "linear_regression"  # Or 'causal_forest', 'bayesian_structural'
    last_validated_at: Optional[datetime] = None
    falsified: bool = False  # If evidence contradicts hypothesis
    
    # Intervention history
    tested_interventions: List[UUID] = Field(default_factory=list)
