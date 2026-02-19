
# Create the LAE API router
code = '''"""
API Router for Layer Arbitration Engine (LAE).
Endpoints for layer detection, arbitration, and diagnostic experiments.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime

from spirit.cognition.layer_arbitration_engine import (
    get_layer_arbitration_engine,
    ControlLayer,
    LayerConfidence
)
from spirit.db.supabase_client import get_behavioral_store

router = APIRouter(prefix="/v1/lae", tags=["layer_arbitration"])


class LayerArbitrationRequest(BaseModel):
    user_id: str
    observation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class LayerArbitrationResponse(BaseModel):
    primary_layer: str
    primary_confidence: float
    confidence_tier: str
    runner_up_layer: Optional[str]
    confidence_gap: float
    scores: Dict[str, float]
    action: str
    reasoning: str
    diagnostic_experiment: Optional[Dict[str, Any]]
    arbitrated_at: str


class DiagnosticExperimentRequest(BaseModel):
    user_id: str
    competing_layers: List[str]
    observation_id: Optional[str] = None


class DiagnosticExperimentResponse(BaseModel):
    experiment_id: str
    target_layers: List[str]
    user_description: str
    burden_level: str
    duration_days: int
    conditions: List[Dict[str, Any]]
    predictions: Dict[str, str]


class InterventionMatchRequest(BaseModel):
    user_id: str
    available_interventions: List[Dict[str, Any]]


class InterventionMatchResponse(BaseModel):
    selected_intervention: Optional[Dict[str, Any]]
    layer_match: Dict[str, Any]
    alternatives: List[Dict[str, Any]]


@router.post("/arbitrate", response_model=LayerArbitrationResponse)
async def arbitrate_layer(request: LayerArbitrationRequest):
    """
    Determine which layer (HOM/HSM/PNM) controls current behavior.
    
    This is the core LAE endpoint that decides:
    - Is the person UNABLE? (HOM)
    - Is the person UNWILLING? (HSM)
    - Is the person PROTECTING something? (PNM)
    """
    
    lae = get_layer_arbitration_engine()
    
    # Get observation if ID provided
    observation = {}
    if request.observation_id:
        store = await get_behavioral_store()
        if store:
            result = store.client.table('behavioral_observations').select('*').eq(
                'observation_id', request.observation_id
            ).execute()
            if result.data:
                observation = result.data[0]
    
    # Run arbitration
    result = await lae.arbitrate(
        observation=observation,
        user_id=request.user_id,
        context=request.context or {}
    )
    
    return LayerArbitrationResponse(
        primary_layer=result.primary_layer.value,
        primary_confidence=result.primary_confidence,
        confidence_tier=result.confidence_tier.name,
        runner_up_layer=result.runner_up_layer.value if result.runner_up_layer else None,
        confidence_gap=result.confidence_gap,
        scores={
            "hom": result.hom_score,
            "hsm": result.hsm_score,
            "pnm": result.pnm_score
        },
        action=result.action,
        reasoning=result.reasoning,
        diagnostic_experiment=result.diagnostic_experiment,
        arbitrated_at=result.arbitrated_at.isoformat()
    )


@router.post("/diagnostic-experiment", response_model=DiagnosticExperimentResponse)
async def design_diagnostic_experiment(request: DiagnosticExperimentRequest):
    """
    Design an experiment to disambiguate between competing layers.
    """
    
    if len(request.competing_layers) != 2:
        raise HTTPException(
            status_code=400,
            detail="Exactly 2 competing layers required"
        )
    
    # Parse layers
    layer_map = {
        "hom": ControlLayer.HOM,
        "hsm": ControlLayer.HSM,
        "pnm": ControlLayer.PNM
    }
    
    layers = []
    for l in request.competing_layers:
        if l.lower() not in layer_map:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid layer: {l}. Must be hom, hsm, or pnm"
            )
        layers.append(layer_map[l.lower()])
    
    lae = get_layer_arbitration_engine()
    
    # Get observation
    observation = {}
    if request.observation_id:
        store = await get_behavioral_store()
        if store:
            result = store.client.table('behavioral_observations').select('*').eq(
                'observation_id', request.observation_id
            ).execute()
            if result.data:
                observation = result.data[0]
    
    # Design experiment
    experiment = await lae.experiment_designer.design_experiment(
        competing_layers=tuple(layers),
        observation=observation,
        user_id=request.user_id
    )
    
    if not experiment:
        raise HTTPException(
            status_code=500,
            detail="Failed to design experiment"
        )
    
    return DiagnosticExperimentResponse(
        experiment_id=experiment.experiment_id,
        target_layers=[l.value for l in experiment.target_layers],
        user_description=experiment.user_description,
        burden_level=experiment.burden_level,
        duration_days=experiment.duration_days,
        conditions=[experiment.condition_a, experiment.condition_b],
        predictions={
            "if_hom": experiment.if_hom_predicts,
            "if_hsm": experiment.if_hsm_predicts,
            "if_pnm": experiment.if_pnm_predicts
        }
    )


@router.post("/match-intervention", response_model=InterventionMatchResponse)
async def match_intervention_to_layer(request: InterventionMatchRequest):
    """
    Select intervention matched to user's current layer.
    
    HOM interventions: Reduce load, optimize timing, remove friction
    HSM interventions: Change incentives, reduce risk, alter payoff
    PNM interventions: Narrative reframing, identity bridge, meaning
    """
    
    # First arbitrate to determine layer
    lae = get_layer_arbitration_engine()
    
    # Get recent observation for context
    store = await get_behavioral_store()
    observation = {}
    if store:
        result = store.client.table('behavioral_observations').select('*').eq(
            'user_id', request.user_id
        ).order('timestamp', desc=True).limit(1).execute()
        if result.data:
            observation = result.data[0]
    
    arbitration = await lae.arbitrate(
        observation=observation,
        user_id=request.user_id
    )
    
    # Match intervention
    matched = lae.get_intervention_match(
        arbitration,
        request.available_interventions
    )
    
    # Get alternatives for other layers
    alternatives = []
    for layer in [ControlLayer.HOM, ControlLayer.HSM, ControlLayer.PNM]:
        if layer != arbitration.primary_layer:
            alt_arbitration = type('obj', (object,), {
                'primary_layer': layer,
                'primary_confidence': 0.6,
                'confidence_tier': LayerConfidence.MODERATE
            })()
            
            alt = lae.get_intervention_match(alt_arbitration, request.available_interventions)
            if alt:
                alternatives.append({
                    'layer': layer.value,
                    'intervention': alt
                })
    
    return InterventionMatchResponse(
        selected_intervention=matched,
        layer_match={
            'primary_layer': arbitration.primary_layer.value,
            'confidence': arbitration.primary_confidence,
            'confidence_tier': arbitration.confidence_tier.name
        },
        alternatives=alternatives
    )


@router.get("/signatures/{user_id}")
async def get_detected_signatures(user_id: str, lookback_days: int = 7):
    """
    Get all detected behavioral signatures for a user.
    Shows evidence for each layer.
    """
    
    lae = get_layer_arbitration_engine()
    
    # Get recent observation
    store = await get_behavioral_store()
    observation = {}
    if store:
        result = store.client.table('behavioral_observations').select('*').eq(
            'user_id', user_id
        ).order('timestamp', desc=True).limit(1).execute()
        if result.data:
            observation = result.data[0]
    
    # Detect signatures
    hom_sigs = await lae.hom_detector.detect_signatures(observation, user_id, lookback_days)
    hsm_sigs = await lae.hsm_detector.detect_signatures(observation, user_id, lookback_days)
    pnm_sigs = await lae.pnm_detector.detect_signatures(observation, user_id, lookback_days)
    
    def sig_to_dict(s):
        return {
            'type': s.signature_type,
            'confidence': s.confidence,
            'diagnostic_value': s.diagnostic_value,
            'evidence': s.evidence
        }
    
    return {
        'user_id': user_id,
        'lookback_days': lookback_days,
        'hom_signatures': [sig_to_dict(s) for s in hom_sigs],
        'hsm_signatures': [sig_to_dict(s) for s in hsm_sigs],
        'pnm_signatures': [sig_to_dict(s) for s in pnm_sigs],
        'summary': {
            'hom_score': lae._calculate_layer_score(hom_sigs),
            'hsm_score': lae._calculate_layer_score(hsm_sigs),
            'pnm_score': lae._calculate_layer_score(pnm_sigs)
        }
    }


@router.get("/layer-history/{user_id}")
async def get_layer_history(user_id: str, days: int = 30):
    """
    Get history of layer attributions for a user.
    Shows how layer dominance has shifted over time.
    """
    
    store = await get_behavioral_store()
    if not store:
        return {'error': 'no_data_store'}
    
    since = (datetime.utcnow() - __import__('datetime').timedelta(days=days)).isoformat()
    
    # Get arbitration history
    history = store.client.table('layer_arbitrations').select('*').eq(
        'user_id', user_id
    ).gte('arbitrated_at', since).order('arbitrated_at').execute()
    
    if not history.data:
        return {
            'user_id': user_id,
            'days': days,
            'arbitrations': [],
            'layer_distribution': {'hom': 0, 'hsm': 0, 'pnm': 0}
        }
    
    # Calculate distribution
    distribution = {'hom': 0, 'hsm': 0, 'pnm': 0}
    for h in history.data:
        layer = h.get('primary_layer')
        if layer in distribution:
            distribution[layer] += 1
    
    # Normalize
    total = sum(distribution.values())
    if total > 0:
        distribution = {k: v/total for k, v in distribution.items()}
    
    return {
        'user_id': user_id,
        'days': days,
        'n_arbitrations': len(history.data),
        'arbitrations': [
            {
                'timestamp': h['arbitrated_at'],
                'primary_layer': h['primary_layer'],
                'confidence': h['primary_confidence'],
                'action': h['action']
            }
            for h in history.data[-20:]  # Last 20
        ],
        'layer_distribution': distribution,
        'dominant_layer': max(distribution, key=distribution.get) if total > 0 else None
    }
'''

with open('/mnt/kimi/output/lae_router.py', 'w') as f:
    f.write(code)

print("Created: lae_router.py")
print(f"Size: {len(code)} bytes")
