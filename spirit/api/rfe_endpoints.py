"""
RFE API Endpoints: Expose Reality Filter Engine functionality to Spirit.
Evidence grading, experiment management, disproven hypothesis queries.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.security import HTTPBearer

from spirit.evidence.personal_evidence_ladder import (
    EvidenceGradingEngine,
    EvidenceLevel,
    EvidenceGrading
)
from spirit.evidence.reality_filter_engine import RealityFilterEngine
from spirit.memory.disproven_hypothesis_archive import (
    DisprovenHypothesisArchive,
    HypothesisFalsificationTracker
)
from spirit.cognition.human_operating_model import (
    get_human_operating_model,
    Subsystem,
    CognitiveMechanism
)
from spirit.db.supabase_client import get_behavioral_store


router = APIRouter(prefix="/v1/rfe", tags=["reality_filter_engine"])
security = HTTPBearer()


async def get_current_user(credentials=Depends(security)) -> UUID:
    """Extract user from JWT."""
    import base64
    import json
    try:
        payload = credentials.credentials.split('.')[1]
        padding = 4 - len(payload) % 4
        if padding != 4:
            payload += '=' * padding
        decoded = base64.urlsafe_b64decode(payload)
        claims = json.loads(decoded)
        return UUID(claims['sub'])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


# ============================================================
# EVIDENCE GRADING ENDPOINTS
# ============================================================

@router.post("/evidence/grade")
async def grade_observation(
    observation: Dict[str, Any],
    background_tasks: BackgroundTasks,
    user_id: UUID = Depends(get_current_user)
):
    """
    Grade a new observation through the Personal Evidence Ladder.
    Returns PEL level and confidence.
    """
    engine = EvidenceGradingEngine()
    
    grading = await engine.grade_observation(
        observation=observation,
        user_id=str(user_id)
    )
    
    # Persist in background
    background_tasks.add_task(engine.persist_grading, grading)
    
    return {
        "observation_id": grading.observation_id,
        "pel_level": {
            "value": grading.level.value,
            "name": grading.level.name,
            "description": _get_level_description(grading.level)
        },
        "confidence": grading.confidence,
        "grading_reason": grading.grading_reason,
        "validation_passed": grading.validation_checks_passed,
        "can_form_hypothesis": grading.level.value >= EvidenceLevel.INTERVENTION_RESPONSE.value,
        "can_modify_beliefs": grading.level.value >= EvidenceLevel.COUNTERFACTUAL_STABILITY.value
    }


@router.post("/evidence/{observation_id}/upgrade")
async def attempt_evidence_upgrade(
    observation_id: str,
    upgrade_context: Dict[str, Any],
    user_id: UUID = Depends(get_current_user)
):
    """
    Attempt to upgrade evidence to higher PEL level based on new data.
    Most upgrades will fail - strict criteria.
    """
    # Get current grading
    store = await get_behavioral_store()
    if not store:
        raise HTTPException(status_code=503, detail="Data store unavailable")
    
    result = store.client.table('evidence_grading').select('*').eq(
        'observation_id', observation_id
    ).eq('user_id', str(user_id)).execute()
    
    if not result.data:
        raise HTTPException(status_code=404, detail="Observation not found")
    
    current = _dict_to_grading(result.data[0])
    
    # Attempt upgrade
    engine = EvidenceGradingEngine()
    upgraded = await engine.upgrade_evidence(current, upgrade_context)
    
    if upgraded.level != current.level:
        await engine.persist_grading(upgraded)
        return {
            "upgrade_successful": True,
            "new_level": {
                "value": upgraded.level.value,
                "name": upgraded.level.name
            },
            "upgrade_reason": upgraded.upgrade_reason,
            "can_now_form_hypothesis": upgraded.level.value >= EvidenceLevel.INTERVENTION_RESPONSE.value,
            "can_now_modify_beliefs": upgraded.level.value >= EvidenceLevel.COUNTERFACTUAL_STABILITY.value
        }
    
    return {
        "upgrade_successful": False,
        "current_level": {
            "value": current.level.value,
            "name": current.level.name
        },
        "reason": "Validation criteria not met",
        "suggestion": "Continue accumulating evidence or design targeted experiment"
    }


@router.get("/evidence/strong-enough-for-beliefs")
async def get_evidence_for_belief_modification(
    min_level: int = Query(4, ge=0, le=5),
    limit: int = Query(50, le=100),
    user_id: UUID = Depends(get_current_user)
):
    """
    Retrieve only evidence strong enough to modify beliefs.
    Default: Level 4+ (Counterfactual Stability and above).
    """
    engine = EvidenceGradingEngine()
    
    evidence = await engine.get_evidence_for_belief_modification(
        user_id=str(user_id),
        min_level=EvidenceLevel(min_level)
    )
    
    return {
        "min_level_required": min_level,
        "evidence_count": len(evidence),
        "evidence": [
            {
                "observation_id": e.observation_id,
                "level": e.level.name,
                "confidence": e.confidence,
                "graded_at": e.graded_at.isoformat(),
                "metadata": e.level_metadata
            }
            for e in evidence[:limit]
        ],
        "belief_modification_threshold": "Level 4+ recommended for belief network updates"
    }


# ============================================================
# RFE PROCESSING ENDPOINTS
# ============================================================

@router.post("/process")
async def process_observation_through_rfe(
    observation: Dict[str, Any],
    background_tasks: BackgroundTasks,
    user_id: UUID = Depends(get_current_user)
):
    """
    Main RFE entry point: Process observation through full pipeline.
    Returns decision: form_hypothesis, schedule_experiment, accumulate_data, or discard.
    """
    rfe = RealityFilterEngine()
    
    result = await rfe.process_observation(observation, str(user_id))
    
    # Log decision for learning
    background_tasks.add_task(_log_rfe_decision, result, user_id)
    
    return {
        "observation_id": result["observation_id"],
        "processed_at": result["processed_at"],
        
        # Evidence assessment
        "evidence": {
            "pel_level": result["evidence_grading"]["level"],
            "level_value": result["evidence_grading"]["level_value"],
            "confidence": result["evidence_grading"]["confidence"]
        },
        
        # Confound assessment
        "confounds": {
            "detected": result["confound_assessment"]["confounds_detected"],
            "summary": result["confound_assessment"]["confound_summary"],
            "severe": result["confound_assessment"]["severe_confounds"]
        },
        
        # Evidence score (if applicable)
        "evidence_score": result.get("evidence_score", {}).get("composite"),
        
        # RFE Decision
        "decision": {
            "action": result["action"],
            "reason": result["action_reason"],
            "hypothesis_confidence": result.get("hypothesis_confidence", "standard")
        },
        
        # Experiment (if scheduled)
        "experiment": result.get("experiment_proposed"),
        
        # Memory decision
        "memory": {
            "retain": result["memory_decision"]["retain"],
            "category": result["memory_decision"]["category"],
            "priority": result["memory_decision"]["priority"]
        }
    }


@router.post("/process-batch")
async def process_batch_through_rfe(
    observations: List[Dict[str, Any]],
    background_tasks: BackgroundTasks,
    user_id: UUID = Depends(get_current_user)
):
    """
    Process multiple observations through RFE.
    Returns summary statistics and individual results.
    """
    rfe = RealityFilterEngine()
    
    results = []
    decisions = {"form_hypothesis": 0, "schedule_experiment": 0, "accumulate_data": 0, "discard": 0}
    
    for obs in observations:
        result = await rfe.process_observation(obs, str(user_id))
        results.append(result)
        decisions[result["action"]] += 1
    
    # Log batch
    background_tasks.add_task(_log_batch_rfe, results, user_id)
    
    return {
        "batch_size": len(observations),
        "decision_summary": decisions,
        "retention_rate": sum(1 for r in results if r["memory_decision"]["retain"]) / len(results),
        "avg_evidence_level": sum(r["evidence_grading"]["level_value"] for r in results) / len(results),
        "severe_confound_rate": sum(1 for r in results if r["confound_assessment"]["severe_confounds"]) / len(results),
        "individual_results": results[:10]  # First 10 for inspection
    }


# ============================================================
# EXPERIMENT MANAGEMENT ENDPOINTS
# ============================================================

@router.post("/experiments/design")
async def design_experiment(
    ambiguity: str,
    current_hypothesis: Optional[str] = None,
    user_constraints: Optional[Dict[str, Any]] = None,
    user_id: UUID = Depends(get_current_user)
):
    """
    Design an experiment to resolve specific ambiguity.
    Returns experiment design for user approval.
    """
    from spirit.evidence.reality_filter_engine import ExperimentScheduler
    
    scheduler = ExperimentScheduler()
    
    experiment = await scheduler.design_experiment(
        user_id=str(user_id),
        ambiguity=ambiguity,
        current_hypothesis=current_hypothesis,
        user_constraints=user_constraints
    )
    
    if not experiment:
        return {
            "design_possible": False,
            "reason": "Cannot schedule experiment at this time (cooldown active or max concurrent reached)"
        }
    
    # Convert to user-facing proposal
    proposal = await scheduler.propose_experiment_to_user(experiment)
    
    return {
        "design_possible": True,
        "experiment_id": experiment.experiment_id,
        "proposal": proposal,
        "raw_design": {
            "conditions": experiment.conditions,
            "duration_days": experiment.duration_days,
            "n_trials": experiment.n_trials_per_condition,
            "randomization": experiment.randomization_method
        }
    }


@router.post("/experiments/{experiment_id}/opt-in")
async def opt_in_to_experiment(
    experiment_id: str,
    opt_in: bool,
    modifications: Optional[Dict[str, Any]] = None,
    user_id: UUID = Depends(get_current_user)
):
    """
    User opts in or out of proposed experiment.
    Preserves agency - experiments require explicit consent.
    """
    store = await get_behavioral_store()
    if not store:
        raise HTTPException(status_code=503, detail="Data store unavailable")
    
    # Update experiment status
    update_data = {
        "user_opted_in": opt_in,
        "opted_in_at": datetime.utcnow().isoformat(),
        "status": "scheduled" if opt_in else "user_declined"
    }
    
    if modifications:
        update_data["user_modifications"] = modifications
    
    result = store.client.table('experiments').update(update_data).eq(
        'experiment_id', experiment_id
    ).eq('user_id', str(user_id)).execute()
    
    if not result.data:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    return {
        "experiment_id": experiment_id,
        "opted_in": opt_in,
        "status": update_data["status"],
        "message": (
            "Experiment scheduled. You'll receive prompts according to the design."
            if opt_in else
            "Experiment declined. Spirit will not schedule these trials."
        ),
        "modifications_accepted": bool(modifications)
    }


@router.get("/experiments/active")
async def get_active_experiments(
    user_id: UUID = Depends(get_current_user)
):
    """
    Get all active experiments for user.
    """
    store = await get_behavioral_store()
    if not store:
        raise HTTPException(status_code=503, detail="Data store unavailable")
    
    experiments = store.client.table('experiments').select('*').eq(
        'user_id', str(user_id)
    ).in_('status', ['running', 'scheduled']).execute()
    
    return {
        "active_experiments": [
            {
                "experiment_id": e['experiment_id'],
                "ambiguity": e['ambiguity_to_resolve'],
                "status": e['status'],
                "started_at": e.get('started_at'),
                "progress": await _calculate_experiment_progress(e['experiment_id'])
            }
            for e in (experiments.data or [])
        ]
    }


@router.post("/experiments/{experiment_id}/report-outcome")
async def report_experiment_outcome(
    experiment_id: str,
    trial_number: int,
    outcome: Dict[str, Any],
    user_id: UUID = Depends(get_current_user)
):
    """
    Report outcome for specific experiment trial.
    Used for causal inference.
    """
    store = await get_behavioral_store()
    if not store:
        raise HTTPException(status_code=503, detail="Data store unavailable")
    
    # Update trial
    store.client.table('experiment_trials').update({
        "executed_at": datetime.utcnow().isoformat(),
        "outcome_measures": outcome,
        "compliance_score": outcome.get("compliance", 1.0)
    }).eq('experiment_id', experiment_id).eq(
        'trial_number', trial_number
    ).execute()
    
    # Check if experiment complete
    trials = store.client.table('experiment_trials').select('*').eq(
        'experiment_id', experiment_id
    ).execute()
    
    completed = sum(1 for t in trials.data if t.get('executed_at'))
    total = len(trials.data) if trials.data else 0
    
    if completed >= total:
        # Mark complete and trigger analysis
        store.client.table('experiments').update({
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat()
        }).eq('experiment_id', experiment_id).execute()
        
        return {
            "recorded": True,
            "experiment_complete": True,
            "message": "Experiment complete! Results will be analyzed within 24 hours."
        }
    
    return {
        "recorded": True,
        "experiment_complete": False,
        "progress": f"{completed}/{total} trials completed"
    }


# ============================================================
# DISPROVEN HYPOTHESIS ARCHIVE ENDPOINTS
# ============================================================

@router.get("/archive/check")
async def check_similar_falsifications(
    proposed_hypothesis: str,
    user_id: UUID = Depends(get_current_user)
):
    """
    Check if similar hypothesis was already falsified.
    Prevents re-testing failed hypotheses.
    """
    archive = DisprovenHypothesisArchive()
    
    # Detect current life phase for relevance scoring
    current_phase = await archive._detect_life_phase_context(str(user_id))
    
    matches = await archive.check_similar_falsifications(
        user_id=str(user_id),
        proposed_hypothesis=proposed_hypothesis,
        current_life_phase=current_phase
    )
    
    return {
        "current_life_phase": current_phase,
        "similar_falsifications_found": len(matches),
        "matches": matches,
        "recommendation": (
            "Consider formulating more nuanced variant" 
            if matches and matches[0]['combined_relevance'] > 0.8 
            else "Proceed with hypothesis formation"
        )
    }


@router.get("/archive/summary")
async def get_falsification_learning_summary(
    days: int = Query(90, ge=7, le=365),
    user_id: UUID = Depends(get_current_user)
):
    """
    Get summary of what Spirit has learned from falsifications.
    Shows evolution of understanding.
    """
    archive = DisprovenHypothesisArchive()
    
    summary = await archive.get_learning_summary(str(user_id), days)
    
    return summary


@router.post("/archive/attempt-revival")
async def attempt_hypothesis_revival(
    hypothesis_id: str,
    new_evidence: Dict[str, Any],
    revival_reason: str,
    user_id: UUID = Depends(get_current_user)
):
    """
    Attempt to revive a falsified hypothesis with new evidence.
    Strict criteria - most revivals fail.
    """
    archive = DisprovenHypothesisArchive()
    
    result = await archive.attempt_revival(hypothesis_id, new_evidence, revival_reason)
    
    return result


@router.get("/archive/collective-patterns")
async def get_collective_falsification_patterns(
    archetype_name: Optional[str] = None,
    user_id: UUID = Depends(get_current_user)
):
    """
    Get common falsification patterns for archetypes.
    Powers: "People like you often mistakenly believe..."
    """
    archive = DisprovenHypothesisArchive()
    
    if archetype_name:
        patterns = await archive.get_falsification_patterns_for_archetype(archetype_name)
    else:
        # Get user's archetype first
        from spirit.memory.collective_intelligence import CollectiveIntelligenceEngine
        collective = CollectiveIntelligenceEngine()
        archetype = await collective.get_user_archetype(user_id)
        
        if archetype:
            patterns = await archive.get_falsification_patterns_for_archetype(archetype.name)
        else:
            patterns = []
    
    return {
        "archetype": archetype_name or (archetype.name if 'archetype' in locals() else "unknown"),
        "patterns": patterns,
        "insight": "These are common misconceptions people similar to you have held"
    }


# ============================================================
# HUMAN OPERATING MODEL ENDPOINTS
# ============================================================

@router.post("/mechanisms/generate")
async def generate_mechanism_hypotheses(
    observation: Dict[str, Any],
    top_n: int = Query(5, ge=1, le=10),
    user_id: UUID = Depends(get_current_user)
):
    """
    Generate candidate mechanisms from Human Operating Model.
    Returns ranked mechanisms that could explain observed behavior.
    """
    hom = get_human_operating_model()
    
    candidates = hom.generate_candidate_mechanisms(observation, top_n)
    
    return {
        "observation_type": observation.get("observation_type"),
        "candidate_mechanisms": [
            {
                "mechanism_id": m.mechanism_id,
                "name": m.name,
                "subsystem": m.subsystem.value,
                "confidence_score": round(score, 3),
                "description": m.description,
                "framing_template": m.user_framing_templates[0] if m.user_framing_templates else None,
                "intervention_suggestion": m.intervention_levers[0] if m.intervention_levers else None
            }
            for m, score in candidates
        ],
        "top_mechanism": candidates[0][0].mechanism_id if candidates else None
    }


@router.get("/mechanisms/{mechanism_id}")
async def get_mechanism_details(
    mechanism_id: str,
    include_related: bool = Query(True),
    user_id: UUID = Depends(get_current_user)
):
    """
    Get full details of a specific cognitive mechanism.
    """
    hom = get_human_operating_model()
    
    mechanism = hom.get_mechanism(mechanism_id)
    if not mechanism:
        raise HTTPException(status_code=404, detail="Mechanism not found")
    
    result = {
        "mechanism_id": mechanism.mechanism_id,
        "name": mechanism.name,
        "subsystem": mechanism.subsystem.value,
        "description": mechanism.description,
        "empirical_basis": mechanism.empirical_basis,
        "confidence": mechanism.confidence.name,
        "observable_signatures": mechanism.observable_signatures,
        "predictions": mechanism.predictions,
        "intervention_levers": mechanism.intervention_levers,
        "user_framing_templates": mechanism.user_framing_templates,
        "mimicking_confounds": mechanism.mimicking_confounds
    }
    
    if include_related:
        related = hom.get_related_mechanisms(mechanism_id)
        result["related_mechanisms"] = [
            {"id": r.mechanism_id, "name": r.name, "subsystem": r.subsystem.value}
            for r in related
        ]
    
    return result


@router.post("/mechanisms/design-experiment")
async def design_mechanism_disambiguation_experiment(
    candidate_mechanisms: List[str],
    ambiguity: str,
    user_id: UUID = Depends(get_current_user)
):
    """
    Design experiment to test between candidate mechanisms.
    """
    hom = get_human_operating_model()
    
    # Get mechanism objects
    mechs_with_scores = []
    for mech_id in candidate_mechanisms[:3]:  # Top 3
        mech = hom.get_mechanism(mech_id)
        if mech:
            # Score based on current context (simplified)
            mechs_with_scores.append((mech, 0.7))
    
    design = hom.generate_experimental_design(mechs_with_scores, ambiguity)
    
    return design


# ============================================================
# INTEGRATION ENDPOINTS (for existing Spirit systems)
# ============================================================

@router.post("/behavioral-scientist/enhanced-analyze")
async def enhanced_behavioral_analysis(
    observation: Dict[str, Any],
    background_tasks: BackgroundTasks,
    user_id: UUID = Depends(get_current_user)
):
    """
    Enhanced analysis that integrates RFE + HOM into behavioral scientist pipeline.
    Replacement for standard /analyze endpoint.
    """
    # Step 1: RFE processing
    rfe = RealityFilterEngine()
    rfe_result = await rfe.process_observation(observation, str(user_id))
    
    # Step 2: If allowed to form hypothesis, generate mechanistic explanation
    if rfe_result["action"] == "form_hypothesis":
        hom = get_human_operating_model()
        mechanisms = hom.generate_candidate_mechanisms(observation, top_n=3)
        
        # Select top mechanism for framing
        top_mechanism = mechanisms[0][0] if mechanisms else None
        
        # Check archive for similar falsifications
        archive = DisprovenHypothesisArchive()
        similar_falsified = await archive.check_similar_falsifications(
            str(user_id),
            f"{top_mechanism.name if top_mechanism else 'unknown'}: {observation.get('behavior', {})}"
        ) if top_mechanism else []
        
        return {
            "rfe_decision": rfe_result["action"],
            "evidence_level": rfe_result["evidence_grading"]["level"],
            "confounds_detected": rfe_result["confound_assessment"]["confounds_detected"],
            
            "mechanistic_hypothesis": {
                "primary_mechanism": {
                    "id": top_mechanism.mechanism_id if top_mechanism else None,
                    "name": top_mechanism.name if top_mechanism else None,
                    "framing": top_mechanism.user_framing_templates[0] if top_mechanism else None
                } if top_mechanism else None,
                "confidence": mechanisms[0][1] if mechanisms else 0,
                "alternative_mechanisms": [
                    {"id": m.mechanism_id, "name": m.name, "confidence": s}
                    for m, s in mechanisms[1:]
                ] if len(mechanisms) > 1 else []
            },
            
            "archive_warnings": [
                {
                    "similar_hypothesis": m["original_hypothesis"],
                    "falsified_at": m["falsified_at"],
                    "warning": m["warning"]
                }
                for m in similar_falsified[:2]
            ],
            
            "recommended_intervention": (
                hom.get_intervention_for_mechanism(
                    top_mechanism.mechanism_id,
                    {"time_of_day": observation.get("context", {}).get("hour")}
                ) if top_mechanism else None
            ),
            
            "queued_for_mao": rfe_result["action"] == "form_hypothesis",
            "experiment_proposed": rfe_result.get("experiment_proposed")
        }
    
    # Not enough evidence - return RFE guidance
    return {
        "rfe_decision": rfe_result["action"],
        "reason": rfe_result["action_reason"],
        "evidence_level": rfe_result["evidence_grading"]["level"],
        "next_steps": (
            "Experiment scheduled to gather causal evidence" 
            if rfe_result.get("experiment_proposed") 
            else "Continue observation to accumulate evidence"
        ),
        "experiment": rfe_result.get("experiment_proposed")
    }


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _get_level_description(level: EvidenceLevel) -> str:
    """Get human-readable description of PEL level."""
    descriptions = {
        EvidenceLevel.RAW_OBSERVATION: "Raw sensor data, no behavioral meaning",
        EvidenceLevel.BEHAVIORAL_METRIC: "Derived metric, single-point measurement",
        EvidenceLevel.CONTEXTUALIZED_PATTERN: "Repeated across contexts, predictive",
        EvidenceLevel.INTERVENTION_RESPONSE: "Intentional change produced response",
        EvidenceLevel.COUNTERFACTUAL_STABILITY: "Holds across varying conditions",
        EvidenceLevel.IDENTITY_LAW: "Cross-domain invariant, core trait"
    }
    return descriptions.get(level, "Unknown level")


def _dict_to_grading(data: Dict) -> EvidenceGrading:
    """Convert database dict to EvidenceGrading object."""
    from spirit.evidence.personal_evidence_ladder import EvidenceLevel
    
    return EvidenceGrading(
        observation_id=data['observation_id'],
        user_id=data['user_id'],
        level=EvidenceLevel(data['level']),
        confidence=data['confidence'],
        grading_reason=data['grading_reason'],
        graded_by=data['graded_by'],
        graded_at=datetime.fromisoformat(data['graded_at'].replace('Z', '+00:00')),
        level_metadata=data.get('level_metadata', {}),
        upgraded_from=EvidenceLevel(data['upgraded_from']) if data.get('upgraded_from') else None,
        upgrade_reason=data.get('upgrade_reason'),
        validation_checks_passed=data.get('validation_checks_passed', []),
        validation_checks_failed=data.get('validation_checks_failed', [])
    )


async def _calculate_experiment_progress(experiment_id: str) -> Dict:
    """Calculate completion progress for experiment."""
    store = await get_behavioral_store()
    if not store:
        return {"unknown": True}
    
    trials = store.client.table('experiment_trials').select('*').eq(
        'experiment_id', experiment_id
    ).execute()
    
    if not trials.data:
        return {"unknown": True}
    
    total = len(trials.data)
    completed = sum(1 for t in trials.data if t.get('executed_at'))
    
    return {
        "total_trials": total,
        "completed_trials": completed,
        "percentage": round(completed / total * 100, 1) if total > 0 else 0
    }


async def _log_rfe_decision(result: Dict, user_id: UUID):
    """Log RFE decision for learning."""
    store = await get_behavioral_store()
    if not store:
        return
    
    store.client.table('rfe_decision_log').insert({
        'observation_id': result['observation_id'],
        'user_id': str(user_id),
        'evidence_level': result['evidence_grading']['level_value'],
        'evidence_confidence': result['evidence_grading']['confidence'],
        'confounds_detected': result['confound_assessment']['confounds_detected'],
        'severe_confounds': result['confound_assessment']['severe_confounds'],
        'evidence_score_composite': result.get('evidence_score', {}).get('composite'),
        'decision_action': result['action'],
        'decision_reason': result['action_reason'],
        'experiment_id': result.get('experiment_proposed', {}).get('experiment_id') if result.get('experiment_proposed') else None,
        'memory_retain': result['memory_decision']['retain'],
        'memory_category': result['memory_decision']['category'],
        'processed_at': result['processed_at']
    }).execute()


async def _log_batch_rfe(results: List[Dict], user_id: UUID):
    """Log batch RFE processing."""
    for result in results:
        await _log_rfe_decision(result, user_id)
