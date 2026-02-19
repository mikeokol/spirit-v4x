
# Continue creating the VIM API router

vim_router_code = '''"""
API Router for Values Inference Module (VIM).
Endpoints for value inference, experiment design guidance, and intervention framing.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime

from spirit.cognition.values_inference_module import (
    get_values_inference_module,
    ValueCategory,
    TradeoffType
)
from spirit.db.supabase_client import get_behavioral_store

router = APIRouter(prefix="/v1/vim", tags=["values_inference"])


class ObservationProcessRequest(BaseModel):
    user_id: str
    observation_id: Optional[str] = None
    observation_data: Optional[Dict[str, Any]] = None


class ObservationProcessResponse(BaseModel):
    user_id: str
    observation_processed: Optional[str]
    new_sacrifices_detected: int
    emotional_gradients_updated: int
    current_value_profile: Dict[str, Any]
    dominant_values: List[Dict[str, Any]]
    value_conflicts_detected: List[Dict[str, Any]]


class RecoveryPatternRequest(BaseModel):
    user_id: str
    disruption_event: Dict[str, Any]
    subsequent_observation_ids: List[str]


class RecoveryPatternResponse(BaseModel):
    return_vector_recorded: bool
    inferred_value: Optional[str]
    automatic_restart: Optional[bool]
    confidence: float
    resumption_delay_hours: Optional[float]


class ValueProfileResponse(BaseModel):
    user_id: str
    profile_generated_at: str
    n_values_inferred: int
    values: Dict[str, Any]
    dominant_values: List[Dict[str, Any]]
    value_conflicts: List[Dict[str, Any]]
    inference_reliability: Dict[str, Any]


class ExperimentDesignRequest(BaseModel):
    user_id: str
    proposed_goal: Optional[str] = None


class ExperimentDesignResponse(BaseModel):
    user_id: str
    design_principles: List[str]
    framing_suggestions: List[str]
    tradeoffs_to_test: List[str]
    avoid: List[str]


class InterventionFramingRequest(BaseModel):
    user_id: str
    intervention_type: str


class InterventionFramingResponse(BaseModel):
    user_id: str
    primary_value: str
    framing: str
    suggested_language: List[str]
    language_to_avoid: List[str]
    secondary_values: List[str]
    rationale: str


class ValueConflictCheckRequest(BaseModel):
    user_id: str
    proposed_action: str
    context: Optional[Dict[str, Any]] = None


class ValueConflictCheckResponse(BaseModel):
    user_id: str
    conflict_detected: bool
    conflicting_values: List[Dict[str, Any]]
    severity: str
    recommendation: str


@router.post("/process", response_model=ObservationProcessResponse)
async def process_observation(request: ObservationProcessRequest):
    """
    Process behavioral observation through VIM to detect values.
    
    This is the primary endpoint for updating value inferences from new data.
    """
    vim = get_values_inference_module()
    
    # Get observation data
    observation = request.observation_data or {}
    if request.observation_id and not observation:
        store = await get_behavioral_store()
        if store:
            result = store.client.table('behavioral_observations').select('*').eq(
                'observation_id', request.observation_id
            ).execute()
            if result.data:
                observation = result.data[0]
    
    if not observation:
        raise HTTPException(status_code=400, detail="No observation data provided")
    
    # Process through VIM
    result = await vim.process_observation(observation, request.user_id)
    
    return ObservationProcessResponse(**result)


@router.post("/recovery", response_model=RecoveryPatternResponse)
async def process_recovery_pattern(request: RecoveryPatternRequest):
    """
    Process post-disruption recovery pattern to detect automatic restarts.
    
    Values restart automatically after disruption; goals require effort.
    This endpoint analyzes what behavior resumes and how.
    """
    vim = get_values_inference_module()
    
    # Get subsequent observations
    store = await get_behavioral_store()
    subsequent = []
    
    if store:
        for obs_id in request.subsequent_observation_ids:
            result = store.client.table('behavioral_observations').select('*').eq(
                'observation_id', obs_id
            ).execute()
            if result.data:
                subsequent.append(result.data[0])
    
    # Process recovery
    result = await vim.process_recovery_pattern(
        request.user_id,
        request.disruption_event,
        subsequent
    )
    
    return RecoveryPatternResponse(**result)


@router.get("/profile/{user_id}", response_model=ValueProfileResponse)
async def get_value_profile(user_id: str):
    """
    Get complete inferred value profile for a user.
    
    Shows all detected values with confidence levels, evidence counts,
    and detected value conflicts.
    """
    vim = get_values_inference_module()
    profile = vim.get_value_profile(user_id)
    
    return ValueProfileResponse(**profile)


@router.post("/experiment-design", response_model=ExperimentDesignResponse)
async def get_experiment_design_guidance(request: ExperimentDesignRequest):
    """
    Get value-aligned guidance for experiment design.
    
    VIM influences the system through design guidance, not direct recommendations.
    This ensures experiments respect inferred values and test appropriate tradeoffs.
    """
    vim = get_values_inference_module()
    
    guidance = vim.get_experiment_design_guidance(
        request.user_id,
        request.proposed_goal
    )
    
    return ExperimentDesignResponse(**guidance)


@router.post("/intervention-framing", response_model=InterventionFramingResponse)
async def get_intervention_framing(request: InterventionFramingRequest):
    """
    Get value-aligned framing for an intervention.
    
    VIM shapes how interventions are communicated, not what is recommended.
    This ensures language resonates with user's enforced preferences.
    """
    vim = get_values_inference_module()
    
    framing = vim.get_intervention_framing(
        request.user_id,
        request.intervention_type
    )
    
    return InterventionFramingResponse(**framing)


@router.post("/conflict-check", response_model=ValueConflictCheckResponse)
async def check_value_conflicts(request: ValueConflictCheckRequest):
    """
    Check if proposed action conflicts with inferred values.
    
    This is the key "sabotage detection" endpoint - it identifies when
    a goal conflicts with deeper values that the user will defend at cost.
    """
    vim = get_values_inference_module()
    
    # Get user profile
    profile = vim.get_value_profile(request.user_id)
    conflicts = profile.get("value_conflicts", [])
    values = profile.get("values", {})
    
    # Analyze proposed action against values
    action_lower = request.proposed_action.lower()
    conflicting = []
    
    # Check for value-threatening language or implications
    threat_indicators = {
        "autonomy": ["must", "required", "forced", "mandatory", "no choice"],
        "responsibility": ["optional", "skip", "ignore", "not your problem"],
        "meaning": ["just do it", "get it over with", "doesn\'t matter"],
        "dignity": ["fix your problem", "what\'s wrong with you", "failure"],
        "mastery": ["easy way", "shortcut", "quick fix", "good enough"],
        "connection": ["alone", "independent", "don\'t need anyone"],
        "security": ["risk everything", "uncertain", "unpredictable"]
    }
    
    for value_name, value_data in values.items():
        if value_data.get("strength", 0) < 0.4:
            continue  # Skip weak inferences
        
        threats = threat_indicators.get(value_name, [])
        for threat in threats:
            if threat in action_lower:
                conflicting.append({
                    "value": value_name,
                    "threat_detected": threat,
                    "value_strength": value_data.get("strength"),
                    "value_confidence": value_data.get("confidence")
                })
                break
    
    # Also check structural conflicts
    for conflict in conflicts:
        # If action forces one side of a known conflict
        if conflict["value_a"] in action_lower or conflict["value_b"] in action_lower:
            conflicting.append({
                "value": f"{conflict['value_a']} vs {conflict['value_b']}",
                "threat_detected": "structural_conflict_activation",
                "value_strength": conflict["both_strength"],
                "value_confidence": conflict["both_strength"]
            })
    
    severity = "none"
    if conflicting:
        avg_strength = sum(c["value_strength"] for c in conflicting) / len(conflicting)
        if avg_strength > 0.7:
            severity = "high"
        elif avg_strength > 0.4:
            severity = "moderate"
        else:
            severity = "low"
    
    recommendation = "proceed"
    if severity == "high":
        recommendation = "redesign_to_align_with_values - goal will likely be sabotaged"
    elif severity == "moderate":
        recommendation = "adjust_framing_and_test_alignment"
    
    return ValueConflictCheckResponse(
        user_id=request.user_id,
        conflict_detected=len(conflicting) > 0,
        conflicting_values=conflicting,
        severity=severity,
        recommendation=recommendation
    )


@router.get("/sacrifices/{user_id}")
async def get_sacrifice_history(user_id: str, limit: int = 20):
    """
    Get recorded sacrifice events for a user.
    
    Shows what the user has consistently protected at cost - the raw
    evidence for inferred values.
    """
    vim = get_values_inference_module()
    
    # Access sacrifice history
    history = vim.sacrifice_detector.sacrifice_history.get(user_id, [])[-limit:]
    
    return {
        "user_id": user_id,
        "total_sacrifices_recorded": len(vim.sacrifice_detector.sacrifice_history.get(user_id, [])),
        "recent_sacrifices": [
            {
                "timestamp": s.timestamp.isoformat(),
                "tradeoff_type": s.tradeoff_type.value,
                "sacrificed": s.sacrificed_resource,
                "protected_value": s.protected_value.value,
                "repeated_pattern": s.repeated_pattern,
                "emotional_intensity": s.emotional_intensity,
                "confidence": s.inference_confidence
            }
            for s in reversed(history)
        ]
    }


@router.get("/return-vectors/{user_id}")
async def get_return_vectors(user_id: str, limit: int = 10):
    """
    Get return vectors showing post-disruption automatic restarts.
    
    The cleanest signal for values: what resumes without planning after failure.
    """
    vim = get_values_inference_module()
    
    # Access return vector history
    history = vim.return_tracker.disruption_history.get(user_id, [])[-limit:]
    
    return {
        "user_id": user_id,
        "total_disruptions_recorded": len(vim.return_tracker.disruption_history.get(user_id, [])),
        "recent_return_vectors": [
            {
                "disruption_type": h["vector"].disruption_type,
                "disrupted_goal": h["vector"].disrupted_goal,
                "actual_resumption": h["vector"].actual_resumption,
                "automatic_restart": h["vector"].automatic_restart,
                "resumption_delay_hours": h["vector"].resumption_delay_hours,
                "inferred_value": h["vector"].inferred_value.value if h["vector"].inferred_value else None,
                "confidence": h["vector"].confidence,
                "observed_at": h["timestamp"].isoformat()
            }
            for h in reversed(history)
        ],
        "automatic_restart_rate": sum(
            1 for h in history if h["vector"].automatic_restart
        ) / max(1, len(history))
    }


@router.get("/tradeoffs/{user_id}")
async def get_tradeoff_analysis(user_id: str):
    """
    Analyze tradeoff patterns to understand value hierarchy.
    
    Shows which tradeoffs the user consistently makes in one direction.
    """
    vim = get_values_inference_module()
    
    # Get sacrifice history and analyze patterns
    sacrifices = vim.sacrifice_detector.sacrifice_history.get(user_id, [])
    
    tradeoff_counts = {}
    for s in sacrifices:
        tt = s.tradeoff_type.value
        if tt not in tradeoff_counts:
            tradeoff_counts[tt] = {"count": 0, "protected": s.protected_value.value}
        tradeoff_counts[tt]["count"] += 1
    
    # Sort by frequency
    sorted_tradeoffs = sorted(
        tradeoff_counts.items(),
        key=lambda x: x[1]["count"],
        reverse=True
    )
    
    return {
        "user_id": user_id,
        "total_tradeoffs_observed": len(sacrifices),
        "consistent_tradeoffs": [
            {
                "tradeoff": tt,
                "frequency": data["count"],
                "protected_value": data["protected"],
                "inference": f"User consistently protects {data['protected']} over alternative"
            }
            for tt, data in sorted_tradeoffs[:5]
        ],
        "value_hierarchy_suggestion": [
            data["protected"] for _, data in sorted_tradeoffs[:3]
        ]
    }
'''

print(f"VIM Router created: {len(vim_router_code)} bytes")
