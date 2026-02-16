"""
API endpoints for real-time processing and advanced causal discovery.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.security import HTTPBearer

from spirit.streaming.realtime_pipeline import get_stream_processor, RealTimeEvent
from spirit.causal.advanced_discovery import AdvancedCausalEngine, CausalDiscoveryPipeline, CausalMethod


router = APIRouter(prefix="/v1/realtime-causal", tags=["realtime_causal"])
security = HTTPBearer()


async def get_current_user(credentials=Depends(security)) -> int:
    import base64
    import json
    try:
        payload = credentials.credentials.split('.')[1]
        padding = 4 - len(payload) % 4
        if padding != 4:
            payload += '=' * padding
        decoded = base64.urlsafe_b64decode(payload)
        claims = json.loads(decoded)
        return int(claims['sub'])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


@router.post("/process")
async def process_realtime(
    observation: dict,
    background_tasks: BackgroundTasks,
    user_id: int = Depends(get_current_user)
):
    """
    Process observation through real-time pipeline.
    Returns immediately, processing happens in background.
    """
    processor = get_stream_processor()
    
    # Queue for processing
    background_tasks.add_task(processor.ingest, user_id, observation)
    
    return {
        "queued": True,
        "timestamp": datetime.utcnow().isoformat(),
        "will_process": "sub-second anomaly detection active"
    }


@router.get("/baselines")
async def get_user_baselines(
    user_id: int = Depends(get_current_user)
):
    """
    Get learned behavioral baselines for this user.
    Shows what Spirit knows about your normal patterns.
    """
    processor = get_stream_processor()
    
    baselines = processor.user_baselines.get(user_id, {})
    
    return {
        "user_id": user_id,
        "baselines_learned": len(baselines),
        "features": {
            feature: {
                "mean": stats['mean'],
                "std": round(stats['std'], 3),
                "samples": stats['n']
            }
            for feature, stats in baselines.items()
        },
        "last_updated": datetime.utcnow().isoformat()
    }


@router.post("/causal/discover")
async def discover_causal_effect(
    cause: str,
    effect: str,
    method: Optional[CausalMethod] = None,
    user_id: int = Depends(get_current_user)
):
    """
    Discover causal effect using advanced methods.
    Auto-selects best method if none specified.
    """
    engine = AdvancedCausalEngine(user_id)
    
    if method:
        estimate = await engine.discover_causal_effect(cause, effect, method)
        used_method = method
    else:
        used_method, estimate = await engine.auto_discover_best_method(cause, effect)
    
    if not estimate:
        return {
            "discovered": False,
            "reason": "insufficient_data_or_no_valid_method",
            "message": "Need more observations or different data structure for causal inference"
        }
    
    return {
        "discovered": True,
        "method_used": used_method.value,
        "cause": cause,
        "effect": effect,
        "estimate": {
            "ate": estimate.ate,
            "confidence_interval": estimate.ate_ci,
            "robustness_score": estimate.robustness_score,
            "heterogeneous_effects": estimate.cate
        },
        "diagnostics": estimate.method_diagnostics,
        "robustness_checks_passed": estimate.placebo_tests_passed,
        "recommendation": "strong_evidence" if estimate.robustness_score > 0.8 else "moderate_evidence" if estimate.robustness_score > 0.5 else "weak_evidence"
    }


@router.post("/causal/discover-all")
async def discover_all_relationships(
    variables: List[str],
    min_confidence: float = 0.6,
    user_id: int = Depends(get_current_user)
):
    """
    Automated discovery of all causal relationships among variables.
    """
    pipeline = CausalDiscoveryPipeline(user_id)
    
    results = await pipeline.discover_all_relationships(variables, min_confidence)
    
    return {
        "variables_tested": len(variables),
        "relationships_found": len(results),
        "discoveries": [
            {
                "cause": r.cause,
                "effect": r.effect,
                "method": r.method.value,
                "effect_size": r.ate,
                "confidence": r.robustness_score,
                "heterogeneity": r.cate is not None
            }
            for r in results
        ],
        "strongest_findings": [
            {"cause": r.cause, "effect": r.effect, "strength": r.ate}
            for r in sorted(results, key=lambda x: abs(x.ate), reverse=True)[:5]
        ]
    }


@router.post("/causal/validate/{hypothesis_id}")
async def validate_hypothesis(
    hypothesis_id: str,
    user_id: int = Depends(get_current_user)
):
    """
    Re-validate existing hypothesis with multiple causal methods.
    """
    pipeline = CausalDiscoveryPipeline(user_id)
    result = await pipeline.validate_existing_hypothesis(hypothesis_id)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.get("/events/recent")
async def get_recent_events(
    n: int = 10,
    user_id: int = Depends(get_current_user)
):
    """
    Get recent real-time events for this user.
    Shows what Spirit detected and responded to.
    """
    # Query from database
    from spirit.db.supabase_client import get_behavioral_store
    store = await get_behavioral_store()
    
    if not store:
        return {"events": []}
    
    events = store.client.table('proactive_interventions').select('*').eq(
        'user_id', str(user_id)
    ).order('executed_at', desc=True).limit(n).execute()
    
    return {
        "events": events.data if events.data else [],
        "count": len(events.data) if events.data else 0
    }
