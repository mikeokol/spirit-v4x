"""
Response schemas for Spirit API.
Includes core models + behavioral intelligence + proactive systems.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from spirit.models import GoalState


# ==========================================
# CORE SCHEMAS (Existing - Preserved)
# ==========================================

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


# ==========================================
# BEHAVIORAL INTELLIGENCE SCHEMAS (New)
# ==========================================

class BehavioralObservationRead(BaseModel):
    """Response for behavioral observation ingestion."""
    observation_id: str
    timestamp: datetime
    observation_type: str
    privacy_level: str
    context: Dict[str, Any]
    behavior: Dict[str, Any]
    stored: bool


class ScreenTimeSessionRead(BaseModel):
    """Response for screen time session data."""
    session_id: str
    started_at: datetime
    duration_seconds: Optional[int]
    app_category: str
    engagement_score: Optional[float]
    distraction_flag: bool


class CausalHypothesisRead(BaseModel):
    """Response for causal discovery results."""
    hypothesis_id: str
    cause_variable: str
    effect_variable: str
    effect_size: float
    confidence_interval: tuple[float, float]
    robustness_score: float
    method_used: str
    n_observations: int
    falsified: bool


class InterventionOutcomeRead(BaseModel):
    """Response for intervention effectiveness."""
    intervention_id: str
    intervention_type: str
    delivered_at: datetime
    user_response: str
    response_time_seconds: Optional[float]
    goal_progress_delta: Optional[float]
    explicit_rating: Optional[int]


# ==========================================
# MEMORY & BELIEF SCHEMAS (New)
# ==========================================

class EpisodicMemoryRead(BaseModel):
    """Response for episodic memory retrieval."""
    episode_id: str
    timestamp: datetime
    episode_type: str  # breakthrough, struggle, insight, milestone, routine
    what_happened: str
    emotional_valence: float
    importance_score: float
    tags: List[str]
    lesson_learned: Optional[str]


class UserBeliefRead(BaseModel):
    """Response for belief network queries."""
    belief_id: str
    belief_type: str  # causal, identity, strategy, constraint
    statement: str
    posterior_probability: float
    evidence_for_count: int
    evidence_against_count: int
    times_tested: int
    currently_held: bool
    contradictions_detected: int


class BehavioralArchetypeRead(BaseModel):
    """Response for archetype discovery."""
    archetype_id: str
    name: str  # e.g., "Morning Warrior", "Night Owl"
    description: str
    population_size: int
    avg_goal_achievement_rate: float
    common_struggles: List[str]
    effective_interventions: List[Dict[str, Any]]


class PeerInsightRead(BaseModel):
    """Response for collective intelligence insights."""
    insight_available: bool
    your_archetype: Optional[str]
    peers_like_you: Optional[int]
    their_success_rate: Optional[float]
    what_worked_for_them: Optional[List[Dict]]
    trend: Optional[str]


# ==========================================
# PROACTIVE & REAL-TIME SCHEMAS (New)
# ==========================================

class RealTimeEventRead(BaseModel):
    """Response for real-time anomaly detection."""
    event_id: str
    timestamp: datetime
    severity: str  # info, warning, critical
    event_type: str  # anomaly, pattern_match, threshold_cross
    trigger_feature: str
    trigger_value: float
    deviation_score: float
    recommended_action: str
    action_taken: bool


class PredictedStateRead(BaseModel):
    """Response for proactive predictions."""
    horizon: str  # imminent, short, medium, long
    predicted_time: datetime
    state_type: str  # vulnerability, opportunity, maintenance, risk
    predicted_behavior: Dict[str, Any]
    confidence: float
    optimal_intervention: Optional[str]
    expected_outcome_if_intervene: float
    expected_outcome_if_ignore: float


class ProactiveInterventionRead(BaseModel):
    """Response for proactive intervention status."""
    intervention_id: str
    predicted_state: str
    predicted_time: datetime
    confidence: float
    intervention_type: str
    executed_at: datetime
    expected_outcome: float
    user_responded: Optional[bool]


class DebateResultRead(BaseModel):
    """Response for multi-agent debate outcomes."""
    proceed: bool
    message: Optional[str]
    debate_rounds: int
    consensus_reached: bool
    adversary_concerns_addressed: bool
    reason: Optional[str]


# ==========================================
# ETHICAL & DELIVERY SCHEMAS (New)
# ==========================================

class EthicalStatusRead(BaseModel):
    """Response for ethical oversight status."""
    current_risk_level: str  # green, yellow, orange, red
    intervention_permitted: bool
    pause_active: Optional[bool]
    pause_until: Optional[datetime]
    hrv_status: Optional[str]
    intervention_burden_24h: int


class NotificationDeliveryRead(BaseModel):
    """Response for notification delivery attempts."""
    sent: bool
    message_id: Optional[str]
    channel: Optional[str]
    scheduled: bool
    scheduled_for: Optional[datetime]
    reason: Optional[str]


class DailyDigestRead(BaseModel):
    """Response for generated daily digests."""
    title: str
    body: str
    streak: Dict[str, Any]
    narrative: str
    tomorrow_preview: str


# ==========================================
# STRATEGIC & INTELLIGENCE SCHEMAS (New)
# ==========================================

class StrategicStatusRead(BaseModel):
    """Response for strategic unlock status."""
    can_unlock_strategic: bool
    maturity_level: str  # novice, emerging, stable, strategic, adaptive
    progress: Dict[str, Any]
    unlocked_features: List[str]
    next_requirements: Dict[str, Any]
    estimated_days_to_strategic: Optional[int]


class IntelligenceAnalysisRead(BaseModel):
    """Response for LangGraph agent analysis."""
    analysis_id: str
    timestamp: datetime
    hypothesis: Optional[str]
    confidence: float
    recommended_intervention: Optional[str]
    intervention_queued: bool
    scientific_reasoning: List[str]


class ScientificReportRead(BaseModel):
    """Response for automated scientific reports."""
    report_period_days: int
    observations_analyzed: int
    active_hypotheses: int
    report: str
    generated_at: datetime


# ==========================================
# UNIFIED DASHBOARD SCHEMA (New)
# ==========================================

class SpiritDashboardRead(BaseModel):
    """
    Unified dashboard combining all Spirit systems.
    """
    user_id: int
    generated_at: datetime
    
    # Core
    active_goal: Optional[GoalRead]
    today_execution: Optional[ExecutionRead]
    
    # Behavioral
    behavioral_data_available: bool
    today_screen_time_minutes: Optional[int]
    focus_score: Optional[float]
    current_streak_days: int
    momentum: str
    
    # Intelligence
    current_predictions: List[PredictedStateRead]
    active_interventions: List[ProactiveInterventionRead]
    risk_level: str
    
    # Memory & Archetype
    archetype: Optional[str]
    recent_insights: List[str]
    
    # Strategic
    strategic_status: StrategicStatusRead
    
    # Recommended action
    recommended_next_action: Optional[str]
