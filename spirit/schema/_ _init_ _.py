"""
Spirit Schemas Package
Exports all request and response schemas for type safety and validation.
"""

# ==========================================
# REQUEST SCHEMAS
# ==========================================

from spirit.schemas.request import (
    # Authentication
    RegisterRequest,
    LoginRequest,
    
    # Behavioral Ingestion
    ScreenTimeSessionRequest,
    BehavioralObservationRequest,
    EMAResponseRequest,
    
    # Causal Discovery
    CausalDiscoveryRequest,
    BatchCausalDiscoveryRequest,
    ValidateHypothesisRequest,
    
    # Memory & Belief
    RecordMemoryEpisodeRequest,
    RetrieveMemoriesRequest,
    AddBeliefRequest,
    UpdateBeliefEvidenceRequest,
    
    # Proactive & Real-time
    StartProactiveLoopRequest,
    ProcessRealtimeRequest,
    ScheduleInterventionRequest,
    PredictionFeedbackRequest,
    
    # Intelligence & Analysis
    AnalyzeObservationRequest,
    PredictGoalOutcomeRequest,
    OptimizeInterventionRequest,
    GenerateReportRequest,
    
    # Ethical & Delivery
    ManualKillSwitchRequest,
    RecordInterventionFeedbackRequest,
    NotificationPreferenceRequest,
    
    # Strategic Planning
    Generate12WeekPlanRequest,
    ForceStrategicRecheckRequest,
    
    # Daily Objective
    GenerateDailyObjectiveRequest,
    CompleteObjectiveRequest,
)


# ==========================================
# RESPONSE SCHEMAS
# ==========================================

from spirit.schemas.response import (
    # Core
    GoalRead,
    ExecutionRead,
    TokenResponse,
    
    # Behavioral
    BehavioralObservationRead,
    ScreenTimeSessionRead,
    CausalHypothesisRead,
    InterventionOutcomeRead,
    
    # Memory & Belief
    EpisodicMemoryRead,
    UserBeliefRead,
    BehavioralArchetypeRead,
    PeerInsightRead,
    
    # Proactive & Real-time
    RealTimeEventRead,
    PredictedStateRead,
    ProactiveInterventionRead,
    DebateResultRead,
    
    # Ethical & Delivery
    EthicalStatusRead,
    NotificationDeliveryRead,
    DailyDigestRead,
    
    # Strategic & Intelligence
    StrategicStatusRead,
    IntelligenceAnalysisRead,
    ScientificReportRead,
    
    # Unified
    SpiritDashboardRead,
)


# ==========================================
# EXPORTS
# ==========================================

__all__ = [
    # Requests
    "RegisterRequest",
    "LoginRequest",
    "ScreenTimeSessionRequest",
    "BehavioralObservationRequest",
    "EMAResponseRequest",
    "CausalDiscoveryRequest",
    "BatchCausalDiscoveryRequest",
    "ValidateHypothesisRequest",
    "RecordMemoryEpisodeRequest",
    "RetrieveMemoriesRequest",
    "AddBeliefRequest",
    "UpdateBeliefEvidenceRequest",
    "StartProactiveLoopRequest",
    "ProcessRealtimeRequest",
    "ScheduleInterventionRequest",
    "PredictionFeedbackRequest",
    "AnalyzeObservationRequest",
    "PredictGoalOutcomeRequest",
    "OptimizeInterventionRequest",
    "GenerateReportRequest",
    "ManualKillSwitchRequest",
    "RecordInterventionFeedbackRequest",
    "NotificationPreferenceRequest",
    "Generate12WeekPlanRequest",
    "ForceStrategicRecheckRequest",
    "GenerateDailyObjectiveRequest",
    "CompleteObjectiveObjectiveRequest",
    
    # Responses
    "GoalRead",
    "ExecutionRead",
    "TokenResponse",
    "BehavioralObservationRead",
    "ScreenTimeSessionRead",
    "CausalHypothesisRead",
    "InterventionOutcomeRead",
    "EpisodicMemoryRead",
    "UserBeliefRead",
    "BehavioralArchetypeRead",
    "PeerInsightRead",
    "RealTimeEventRead",
    "PredictedStateRead",
    "ProactiveInterventionRead",
    "DebateResultRead",
    "EthicalStatusRead",
    "NotificationDeliveryRead",
    "DailyDigestRead",
    "StrategicStatusRead",
    "IntelligenceAnalysisRead",
    "ScientificReportRead",
    "SpiritDashboardRead",
]
