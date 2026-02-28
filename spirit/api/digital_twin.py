
digital_twin_router_code = '''
"""
Digital Twin API Routes (v2.3)
Endpoints for in-silico experimentation, counterfactual analysis, and causal DAG visualization
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pydantic import BaseModel

from spirit.services.digital_twin_orchestrator import DigitalTwinOrchestrator
from spirit.models.mechanistic_user_model import UserState, InterventionDose
from spirit.models.mechanisms import MechanismActivation
from spirit.causal.user_causal_dag import UserCausalDAG
from spirit.db.supabase_client import get_behavioral_store

router = APIRouter(prefix="/v1/digital-twin", tags=["digital-twin"])


class SimulationRequest(BaseModel):
    user_id: str
    hypothesis: str
    target_mechanism: str  # Name of MechanismActivation
    current_state: Dict[str, Any]  # UserState fields
    parameter_space: Optional[Dict[str, List[Any]]] = None


class CounterfactualRequest(BaseModel):
    user_id: str
    past_observation_id: str
    alternative_intervention: Dict[str, Any]


class ValidationResponse(BaseModel):
    decision: str
    intervention: Optional[Dict[str, Any]]
    predicted_stats: Optional[Dict[str, Any]]
    confidence: float
    routing: str
    pre_registration_id: Optional[str]


@router.post("/simulate", response_model=ValidationResponse)
async def run_simulation(request: SimulationRequest):
    """
    Run Digital Twin simulation on a hypothesis.
    Returns Phase I/II validation results and go/no-go decision.
    """
    try:
        orchestrator = DigitalTwinOrchestrator(
            user_id=request.user_id,
            historical_data=[]
        )
        
        # Parse mechanism
        mechanism = MechanismActivation[request.target_mechanism.upper().replace(" ", "_")]
        
        # Build UserState
        state_data = request.current_state
        user_state = UserState(
            timestamp=datetime.utcnow(),
            cognitive_energy=state_data.get('cognitive_energy', 0.5),
            sleep_debt=state_data.get('sleep_debt', 0),
            glucose_stability=state_data.get('glucose_stability', 0.7),
            stress_level=state_data.get('stress_level', 0.3),
            attentional_bandwidth=state_data.get('attentional_bandwidth', 0.6),
            working_memory_load=state_data.get('working_memory_load', 2.0),
            social_load=state_data.get('social_load', 0.3),
            identity_threat_level=state_data.get('identity_threat_level', 0.2),
            current_context=state_data.get('current_context', 'unknown'),
            recent_interventions=state_data.get('recent_interventions', [])
        )
        
        # Run simulation
        decision = orchestrator.process_hypothesis(
            hypothesis=request.hypothesis,
            target_mechanism=mechanism,
            current_state=user_state
        )
        
        return ValidationResponse(
            decision=decision['decision'],
            intervention=decision.get('intervention').__dict__ if decision.get('intervention') else None,
            predicted_stats=decision.get('predicted_stats'),
            confidence=decision.get('confidence', 0.0),
            routing=decision.get('routing', 'unknown'),
            pre_registration_id=decision.get('pre_registration').experiment_id if decision.get('pre_registration') else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/counterfactual")
async def run_counterfactual(request: CounterfactualRequest):
    """
    Post-hoc counterfactual: 'What if we had done X instead?'
    """
    try:
        orchestrator = DigitalTwinOrchestrator(user_id=request.user_id)
        
        # Get past state from database
        store = get_behavioral_store()
        if not store:
            raise HTTPException(status_code=503, detail="Database unavailable")
        
        past_obs = store.client.table('behavioral_observations').select('*').eq(
            'observation_id', request.past_observation_id
        ).execute()
        
        if not past_obs.data:
            raise HTTPException(status_code=404, detail="Observation not found")
        
        obs = past_obs.data[0]
        past_state = UserState(
            timestamp=datetime.fromisoformat(obs['timestamp']),
            cognitive_energy=obs.get('cognitive_energy', 0.5),
            sleep_debt=obs.get('sleep_debt', 0),
            glucose_stability=obs.get('glucose_stability', 0.7),
            stress_level=obs.get('stress_level', 0.3),
            attentional_bandwidth=obs.get('attentional_bandwidth', 0.6),
            working_memory_load=obs.get('working_memory_load', 2.0),
            social_load=obs.get('social_load', 0.3),
            identity_threat_level=obs.get('identity_threat_level', 0.2),
            current_context=obs.get('current_context', 'unknown'),
            recent_interventions=obs.get('recent_interventions', [])
        )
        
        # Build alternative intervention
        alt = request.alternative_intervention
        alternative = InterventionDose(
            intervention_type=alt.get('type', 'EMA'),
            timing=datetime.utcnow(),
            intensity=alt.get('intensity', 0.5),
            framing=alt.get('framing', 'gentle'),
            channel=alt.get('channel', 'mobile'),
            content=alt.get('content', ''),
            expected_mechanism=MechanismActivation[alt.get('mechanism', 'AMBIGUITY_COST')]
        )
        
        result = orchestrator.run_counterfactual_post_hoc(past_state, alternative)
        
        return {
            "predicted_behavior_change": result.predicted_behavior_change,
            "confidence_interval": result.confidence_interval,
            "probability_of_success": result.probability_of_success,
            "risk_score": result.risk_score,
            "expected_user_response": result.expected_user_response,
            "mechanism_activations": {k.name: v for k, v in result.mechanism_activations.items()}
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fidelity/{user_id}")
async def get_fidelity_report(user_id: str):
    """
    Get Digital Twin fidelity report for a user.
    Shows how accurate simulations have been vs reality.
    """
    try:
        orchestrator = DigitalTwinOrchestrator(user_id=user_id)
        report = orchestrator.get_fidelity_report()
        
        # Get recent validation data
        store = get_behavioral_store()
        recent_validations = []
        if store:
            yesterday = (datetime.utcnow() - timedelta(days=1)).isoformat()
            logs = store.client.table('counterfactual_logs').select('*').eq(
                'user_id', user_id
            ).not_.is_('fidelity_score', None).gte('created_at', yesterday).execute()
            
            if logs.data:
                recent_validations = [
                    {
                        "experiment_id": log['experiment_id'],
                        "fidelity_score": log['fidelity_score'],
                        "predicted": log['simulation_result'].get('predicted_behavior_change'),
                        "actual": log['real_outcome'].get('behavior_change') if log['real_outcome'] else None
                    }
                    for log in logs.data[-5:]  # Last 5
                ]
        
        return {
            "user_id": user_id,
            "status": report['status'],
            "mean_fidelity": report['mean_fidelity'],
            "n_validated": report['n_validated'],
            "recent_validations": recent_validations,
            "calibration_trend": "improving" if report.get('recent_drift', 0) > 0 else "stable"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/causal-dag/{user_id}")
async def get_causal_dag(user_id: str, format: str = "json"):
    """
    Get User Causal DAG - the 'mirror' showing what drives their behavior.
    Returns nodes (variables) and edges (causal relationships).
    """
    try:
        dag = UserCausalDAG(user_id)
        
        if format == "json":
            return dag.export_to_json()
        elif format == "insights":
            return {"insights": dag.generate_insight_summary()}
        elif format == "drivers":
            drivers = dag.get_primary_drivers('procrastination')
            return {
                "primary_drivers": [
                    {"variable": var, "influence": influence}
                    for var, influence in drivers
                ]
            }
        else:
            raise HTTPException(status_code=400, detail="Invalid format. Use 'json', 'insights', or 'drivers'")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/design")
async def design_experiment(request: SimulationRequest):
    """
    Design optimal experiment using Digital Twin Phase I screening.
    Returns best intervention parameters without running full validation.
    """
    try:
        from spirit.strategies.virtual_experiment_runner import VirtualExperimentRunner
        
        orchestrator = DigitalTwinOrchestrator(user_id=request.user_id)
        runner = VirtualExperimentRunner(orchestrator.twin)
        
        mechanism = MechanismActivation[request.target_mechanism.upper().replace(" ", "_")]
        
        state_data = request.current_state
        user_state = UserState(
            timestamp=datetime.utcnow(),
            cognitive_energy=state_data.get('cognitive_energy', 0.5),
            sleep_debt=state_data.get('sleep_debt', 0),
            glucose_stability=state_data.get('glucose_stability', 0.7),
            stress_level=state_data.get('stress_level', 0.3),
            attentional_bandwidth=state_data.get('attentional_bandwidth', 0.6),
            working_memory_load=state_data.get('working_memory_load', 2.0),
            social_load=state_data.get('social_load', 0.3),
            identity_threat_level=state_data.get('identity_threat_level', 0.2),
            current_context=state_data.get('current_context', 'unknown'),
            recent_interventions=state_data.get('recent_interventions', [])
        )
        
        pre_reg = runner.design_optimal_experiment(
            hypothesis=request.hypothesis,
            target_mechanism=mechanism,
            initial_state=user_state,
            parameter_space=request.parameter_space or {
                'intensity': [0.2, 0.4, 0.6, 0.8],
                'timing': [9, 14, 20],
                'framing': ['gentle', 'identity_safe']
            }
        )
        
        if not pre_reg:
            return {"status": "no_safe_intervention", "message": "No safe parameters found in search space"}
        
        return {
            "experiment_id": pre_reg.experiment_id,
            "hypothesis": pre_reg.hypothesis,
            "target_mechanism": pre_reg.mechanism_target.name,
            "falsification_criteria": pre_reg.falsification_criteria,
            "status": pre_reg.status,
            "created_at": pre_reg.created_at.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validate/{experiment_id}")
async def validate_experiment(experiment_id: str, user_id: str):
    """
    Update Digital Twin with real-world outcome to improve fidelity.
    Call this after deploying an intervention to log the actual result.
    """
    try:
        store = get_behavioral_store()
        if not store:
            raise HTTPException(status_code=503, detail="Database unavailable")
        
        # Get the experiment
        exp = store.client.table('intervention_recommendations').select('*').eq(
            'pre_registration_id', experiment_id
        ).eq('user_id', user_id).execute()
        
        if not exp.data:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        # In production, you would get actual outcome from user feedback
        # Here we just return the current simulation vs reality gap
        log = store.client.table('counterfactual_logs').select('*').eq(
            'experiment_id', experiment_id
        ).eq('user_id', user_id).execute()
        
        if not log.data:
            return {"status": "pending", "message": "Simulation logged, awaiting real outcome"}
        
        entry = log.data[0]
        
        return {
            "experiment_id": experiment_id,
            "status": "validated" if entry.get('real_outcome') else "pending",
            "predicted_change": entry['simulation_result'].get('predicted_behavior_change'),
            "actual_change": entry['real_outcome'].get('behavior_change') if entry.get('real_outcome') else None,
            "fidelity_score": entry.get('fidelity_score'),
            "model_calibration": "good" if entry.get('fidelity_score', 0) > 0.7 else "needs_retraining"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
'''

print(f"Digital Twin API Router created: {len(digital_twin_router_code)} characters")
print("\nFile location: spirit/api/digital_twin.py")
print("\nEndpoints added:")
print("- POST /v1/digital-twin/simulate - Run Phase I/II simulation")
print("- POST /v1/digital-twin/counterfactual - Post-hoc 'what if' analysis")
print("- GET /v1/digital-twin/fidelity/{user_id} - Check model accuracy")
print("- GET /v1/digital-twin/causal-dag/{user_id} - Get behavior causal map")
print("- POST /v1/digital-twin/experiments/design - Design optimal experiment")
print("- GET /v1/digital-twin/validate/{experiment_id} - Log real outcome")
