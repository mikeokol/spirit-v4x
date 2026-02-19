"""
Proactive Agent Loop: Spirit's autonomous operation system.
Predicts, schedules, and executes interventions without waiting for user input.
The goal: intervene before the user fails, not after.
v2.0: Full cognition stack integration - HOM/HSM/PNM/LAE/VIM all inform intervention design.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
import random

from spirit.db.supabase_client import get_behavioral_store
from spirit.services.notification_engine import NotificationEngine, NotificationPriority
from spirit.agents.behavioral_scientist import BehavioralScientistAgent, PredictiveEngine
from spirit.memory.episodic_memory import EpisodicMemorySystem
from spirit.services.causal_inference import CausalInferenceEngine

# Full cognition stack imports
from spirit.agents.multi_agent_debate import MultiAgentDebate
from spirit.services.empathy_agency import (
    EmpatheticInterventionWrapper,
    AgencyInterventionType,
    EmpathyCalibrationEngine
)

# NEW: Full cognition stack
from spirit.cognition.human_strategy_model import (
    get_human_strategy_model,
    OptimizationTarget,
    GameType
)
from spirit.cognition.personal_narrative_model import (
    get_personal_narrative_model,
    NarrativeAxis
)
from spirit.cognition.layer_arbitration_engine import (
    get_layer_arbitration_engine,
    ControlLayer
)
from spirit.cognition.values_inference_module import (
    get_values_inference_module,
    ValueCategory
)


class PredictionHorizon(Enum):
    IMMINENT = "imminent"      # 0-30 minutes
    SHORT = "short"            # 30 min - 4 hours
    MEDIUM = "medium"          # 4-24 hours
    LONG = "long"              # 1-7 days


@dataclass
class PredictedState:
    """
    A forecast of user's future state with confidence and intervention opportunity.
    v2.0: Enriched with cognition layer information.
    """
    horizon: PredictionHorizon
    predicted_time: datetime
    state_type: str  # 'vulnerability', 'opportunity', 'maintenance', 'risk'
    
    # What we predict
    predicted_behavior: Dict[str, Any]
    confidence: float  # 0-1
    
    # Why we predict this
    trigger_features: Dict[str, float]
    historical_pattern_match: Optional[str]
    
    # What to do about it
    optimal_intervention: Optional[str]
    intervention_window: Optional[tuple]
    expected_outcome_if_intervene: float
    expected_outcome_if_ignore: float
    
    # NEW: Cognition layer information (from LAE)
    primary_control_layer: Optional[ControlLayer] = None
    layer_confidence: float = 0.0
    
    # NEW: Strategic pressure information (from HSM)
    active_strategic_pressures: List[str] = field(default_factory=list)
    
    # NEW: Narrative context (from PNM)
    narrative_axis_alignment: Optional[str] = None
    
    # NEW: Value conflict warning (from VIM)
    value_conflict_risk: Optional[str] = None
    value_alignment_score: float = 0.5


@dataclass
class CognitionInformedContext:
    """
    Rich context from all cognition layers for intervention design.
    This is how the full stack informs proactive decisions.
    """
    # LAE: Which layer controls behavior
    primary_layer: ControlLayer
    layer_scores: Dict[str, float]
    
    # HSM: What strategic pressures are active
    strategic_pressures: List[Dict[str, Any]]
    optimization_targets: List[OptimizationTarget]
    
    # PNM: Narrative structure
    dominant_narrative_axes: List[Dict[str, Any]]
    identity_threat_assessment: Optional[Dict[str, Any]]
    
    # VIM: What values are enforced
    inferred_values: List[Dict[str, Any]]
    value_conflicts: List[Dict[str, Any]]
    sacrifice_patterns: List[str]
    
    # Integration: What this means for intervention
    recommended_intervention_type: str
    framing_strategy: str
    agency_preservation_level: str


class ProactiveScheduler:
    """
    Schedules autonomous check-ins and interventions.
    v2.0: Full cognition stack integration for human-centered intervention design.
    """
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.scheduled_checks: Dict[str, datetime] = {}
        self.running = False
        self.check_interval_seconds = 60
        
        # Callbacks for different prediction types
        self.intervention_handlers: Dict[str, Callable] = {}
        
        # Core systems
        self.debate_system = MultiAgentDebate()
        
        # Human-centered systems
        self.empathy_wrapper = EmpatheticInterventionWrapper(str(user_id))
        self.onboarding_complete = False
        self.empathy_engine = EmpathyCalibrationEngine(str(user_id))
        
        # NEW: Cognition stack access
        self.hsm = get_human_strategy_model()
        self.pnm = get_personal_narrative_model(str(user_id))
        self.lae = get_layer_arbitration_engine()
        self.vim = get_values_inference_module()
    
    def register_handler(self, state_type: str, handler: Callable):
        """Register a function to handle predicted states."""
        self.intervention_handlers[state_type] = handler
    
    async def start(self):
        """Start the autonomous loop."""
        self.running = True
        print(f"Proactive loop v2.0 started for user {self.user_id}")
        
        while self.running:
            try:
                await self._run_prediction_cycle()
            except Exception as e:
                print(f"Error in proactive loop: {e}")
            
            await asyncio.sleep(self.check_interval_seconds)
    
    def stop(self):
        """Stop the autonomous loop."""
        self.running = False
    
    async def _run_prediction_cycle(self):
        """One cycle: predict, enrich with cognition, schedule, execute if due."""
        now = datetime.utcnow()
        
        # Check if onboarding complete
        if not await self._check_onboarding_status():
            print(f"User {self.user_id} onboarding not complete, skipping prediction cycle")
            await asyncio.sleep(300)
            return
        
        # Generate base predictions
        predictions = await self._generate_predictions()
        
        # NEW: Enrich predictions with full cognition stack
        enriched_predictions = []
        for pred in predictions:
            enriched = await self._enrich_with_cognition(pred)
            if enriched:  # May filter out predictions with high value conflict
                enriched_predictions.append(enriched)
        
        # For each enriched prediction, schedule or execute
        for pred in enriched_predictions:
            check_id = f"{pred.state_type}_{pred.predicted_time.isoformat()}"
            
            # Skip if already handled
            if check_id in self.scheduled_checks:
                if self.scheduled_checks[check_id] < now:
                    await self._execute_intervention(pred)
                    del self.scheduled_checks[check_id]
                continue
            
            # NEW: Check value conflict before scheduling
            if pred.value_alignment_score < 0.3:
                print(f"Skipping prediction {check_id} - high value conflict risk: {pred.value_conflict_risk}")
                await self._log_value_conflict_skip(pred)
                continue
            
            # Schedule if in planning window
            if pred.predicted_time > now and pred.predicted_time < now + timedelta(hours=4):
                self.scheduled_checks[check_id] = pred.predicted_time
                print(f"Scheduled {pred.state_type} check for {pred.predicted_time} "
                      f"[Layer: {pred.primary_control_layer.value if pred.primary_control_layer else 'unknown'}, "
                      f"Values: {pred.value_alignment_score:.2f}]")
                
                delay = (pred.predicted_time - now).total_seconds()
                asyncio.create_task(self._delayed_execution(check_id, delay, pred))
    
    async def _check_onboarding_status(self) -> bool:
        """Check if user has completed rich onboarding."""
        if self.onboarding_complete:
            return True
        
        store = await get_behavioral_store()
        if not store:
            return True
        
        belief = await store.get_user_beliefs(str(self.user_id))
        if belief and belief.get("onboarded_at"):
            self.onboarding_complete = True
            return True
        
        return False
    
    async def _generate_predictions(self) -> List[PredictedState]:
        """Generate multi-horizon predictions for this user."""
        predictions = []
        context = await self._get_current_context()
        
        # IMMINENT: Next 30 minutes
        imminent = await self._predict_imminent(context)
        if imminent:
            predictions.append(imminent)
        
        # SHORT: Next 4 hours
        short = await self._predict_short_term(context)
        if short:
            predictions.extend(short)
        
        # MEDIUM: Next 24 hours
        medium = await self._predict_medium_term()
        if medium:
            predictions.extend(medium)
        
        # LONG: Next 7 days
        long_term = await self._predict_long_term()
        if long_term:
            predictions.extend(long_term)
        
        return predictions
    
    async def _enrich_with_cognition(self, prediction: PredictedState) -> Optional[PredictedState]:
        """
        NEW: Enrich prediction with full cognition stack analysis.
        This is the core integration point - all models inform the intervention.
        """
        store = await get_behavioral_store()
        if not store:
            return prediction
        
        # 1. LAE: Determine which layer controls this behavior
        recent_obs = store.client.table('behavioral_observations').select('*').eq(
            'user_id', str(self.user_id)
        ).order('timestamp', desc=True).limit(1).execute()
        
        if recent_obs.data:
            lae_result = await self.lae.arbitrate(
                observation=recent_obs.data[0],
                user_id=str(self.user_id),
                context={'prediction_type': prediction.state_type}
            )
            
            prediction.primary_control_layer = lae_result.primary_layer
            prediction.layer_confidence = lae_result.primary_confidence
            
            # If PNM dominates, intervention must be narrative-based
            if lae_result.primary_layer == ControlLayer.PNM and lae_result.primary_confidence > 0.7:
                prediction.optimal_intervention = "narrative_reframe"
            
            # If HSM dominates, intervention must address strategic optimization
            elif lae_result.primary_layer == ControlLayer.HSM and lae_result.primary_confidence > 0.7:
                prediction.optimal_intervention = "incentive_restructure"
        
        # 2. HSM: Detect active strategic pressures
        if prediction.primary_control_layer == ControlLayer.HSM or prediction.state_type == "vulnerability":
            # Check for strategic pressures that might be causing the predicted state
            pressures = self.hsm.detect_active_pressures(
                observation=recent_obs.data[0] if recent_obs.data else {},
                user_history=[],  # Would load from store
                top_n=3
            )
            
            prediction.active_strategic_pressures = [p[0].pressure_id for p in pressures]
            
            # Adjust intervention based on strategic pressure
            for pressure, confidence in pressures:
                if pressure.game_type == GameType.EFFORT_ENERGY and confidence > 0.6:
                    # Energy optimization pressure - make intervention effortless
                    prediction.optimal_intervention = "friction_reduction"
                elif pressure.game_type == GameType.IDENTITY_COHERENCE and confidence > 0.6:
                    # Identity threat - need narrative bridge
                    prediction.optimal_intervention = "identity_bridge"
        
        # 3. PNM: Check narrative alignment
        pnm_profile = self.pnm.get_dominant_axes(min_confidence=0.4)
        if pnm_profile:
            # Check if predicted intervention aligns with narrative
            alignment = self._check_narrative_alignment(prediction, pnm_profile)
            prediction.narrative_axis_alignment = alignment["alignment"]
            
            # Check for identity threat
            if prediction.state_type == "vulnerability":
                threat = self.pnm.get_identity_threat_assessment(prediction.predicted_behavior.get('action', ''))
                if threat['threat_level'] == 'high':
                    # High identity threat - intervention will be resisted
                    prediction.value_alignment_score = 0.2
                    prediction.value_conflict_risk = "identity_threat"
        
        # 4. VIM: Check value alignment (CRITICAL - prevents sabotage)
        vim_profile = self.vim.get_value_profile(str(self.user_id))
        
        # Check if intervention conflicts with inferred values
        if prediction.optimal_intervention:
            conflict_check = await self._check_value_conflict(prediction.optimal_intervention, vim_profile)
            prediction.value_alignment_score = conflict_check["alignment_score"]
            prediction.value_conflict_risk = conflict_check["conflict_reason"]
            
            # If high conflict, suggest alternative intervention
            if prediction.value_alignment_score < 0.3:
                alternative = self._suggest_value_aligned_intervention(prediction, vim_profile)
                if alternative:
                    prediction.optimal_intervention = alternative
                    prediction.value_alignment_score = 0.6  # Improved alignment
        
        # 5. Integrate: Build framing strategy
        framing = self._build_framing_strategy(prediction, vim_profile, pnm_profile)
        prediction.framing_strategy = framing
        
        return prediction
    
    def _check_narrative_alignment(
        self,
        prediction: PredictedState,
        narrative_axes: List[Any]
    ) -> Dict[str, Any]:
        """Check if intervention aligns with user's narrative structure."""
        # Simplified alignment check
        alignment_score = 0.5
        
        for axis in narrative_axes:
            if axis.axis == NarrativeAxis.CONTROL_DISCOVERY:
                if "structure" in prediction.optimal_intervention or "plan" in prediction.optimal_intervention:
                    alignment_score += 0.2 if axis.position.value < 0 else -0.1
            
            elif axis.axis == NarrativeAxis.BUILDER_PROVER:
                if "process" in prediction.optimal_intervention:
                    alignment_score += 0.2 if axis.position.value < 0 else -0.1
        
        return {
            "alignment": "aligned" if alignment_score > 0.6 else "neutral" if alignment_score > 0.4 else "misaligned",
            "score": alignment_score
        }
    
    async def _check_value_conflict(
        self,
        intervention: str,
        vim_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if intervention conflicts with inferred values."""
        dominant_values = [v["value"] for v in vim_profile.get("dominant_values", [])]
        
        # Check for threatening language
        threat_indicators = {
            "autonomy": ["must", "required", "forced", "mandatory"],
            "responsibility": ["optional", "skip", "ignore"],
            "meaning": ["just do it", "get it over with"],
            "dignity": ["fix your problem", "what's wrong"],
            "mastery": ["easy way", "shortcut", "quick fix"],
        }
        
        intervention_lower = intervention.lower()
        
        for value in dominant_values:
            threats = threat_indicators.get(value, [])
            for threat in threats:
                if threat in intervention_lower:
                    return {
                        "alignment_score": 0.2,
                        "conflict_reason": f"intervention_threatens_{value}"
                    }
        
        return {
            "alignment_score": 0.7,
            "conflict_reason": None
        }
    
    def _suggest_value_aligned_intervention(
        self,
        prediction: PredictedState,
        vim_profile: Dict[str, Any]
    ) -> Optional[str]:
        """Suggest alternative intervention that aligns with values."""
        dominant = [v["value"] for v in vim_profile.get("dominant_values", [])]
        
        if not dominant:
            return None
        
        primary = dominant[0]
        
        # Value-aligned alternatives
        alternatives = {
            "autonomy": "choice_architecture",
            "responsibility": "commitment_reinforcement",
            "meaning": "purpose_connection",
            "dignity": "growth_framing",
            "mastery": "challenge_invitation",
            "connection": "social_accountability",
            "security": "stability_preservation"
        }
        
        return alternatives.get(primary)
    
    def _build_framing_strategy(
        self,
        prediction: PredictedState,
        vim_profile: Dict[str, Any],
        pnm_axes: List[Any]
    ) -> str:
        """Build integrated framing strategy from all cognition layers."""
        strategies = []
        
        # Layer-based framing
        if prediction.primary_control_layer == ControlLayer.HOM:
            strategies.append("energy_compassion")
        elif prediction.primary_control_layer == ControlLayer.HSM:
            strategies.append("strategic_reframe")
        elif prediction.primary_control_layer == ControlLayer.PNM:
            strategies.append("narrative_bridge")
        
        # Value-based framing
        dominant_values = [v["value"] for v in vim_profile.get("dominant_values", [])]
        if dominant_values:
            strategies.append(f"value_aligned_{dominant_values[0]}")
        
        # Narrative-based framing
        for axis in pnm_axes[:2]:
            if axis.confidence > 0.5:
                strategies.append(f"narrative_{axis.axis.value}")
        
        return " + ".join(strategies) if strategies else "neutral"
    
    async def _predict_imminent(self, context: Dict) -> Optional[PredictedState]:
        """Predict immediate next state (0-30 min) based on current momentum."""
        recent = context.get("recent_observations", [])
        if len(recent) < 3:
            return None
        
        last_3 = recent[-3:]
        focus_trend = sum(
            o.get("behavior", {}).get("focus_score", 0.5) 
            for o in last_3
        ) / 3
        
        if focus_trend < 0.4 and recent[-1].get("behavior", {}).get("app_category") != "productivity":
            return PredictedState(
                horizon=PredictionHorizon.IMMINENT,
                predicted_time=datetime.utcnow() + timedelta(minutes=15),
                state_type="vulnerability",
                predicted_behavior={"focus_score": 0.2, "likely_distraction": True},
                confidence=0.7,
                trigger_features={"focus_decline_rate": 0.3, "context_switches": 5},
                historical_pattern_match=None,
                optimal_intervention="micro_focus_prompt",
                intervention_window=(
                    datetime.utcnow() + timedelta(minutes=10),
                    datetime.utcnow() + timedelta(minutes=20)
                ),
                expected_outcome_if_intervene=0.6,
                expected_outcome_if_ignore=0.2
            )
        
        if focus_trend > 0.7:
            return PredictedState(
                horizon=PredictionHorizon.IMMINENT,
                predicted_time=datetime.utcnow() + timedelta(minutes=20),
                state_type="opportunity",
                predicted_behavior={"focus_score": 0.8, "deep_work_continuation": True},
                confidence=0.6,
                trigger_features={"sustained_attention": 25, "no_interruptions": 5},
                historical_pattern_match=None,
                optimal_intervention="offer_focus_extension",
                intervention_window=(
                    datetime.utcnow() + timedelta(minutes=15),
                    datetime.utcnow() + timedelta(minutes=30)
                ),
                expected_outcome_if_intervene=0.9,
                expected_outcome_if_ignore=0.5
            )
        
        return None
    
    async def _predict_short_term(self, context: Dict) -> List[PredictedState]:
        """Predict next 4 hours based on daily patterns."""
        predictions = []
        now = datetime.utcnow()
        hour = now.hour
        
        store = await get_behavioral_store()
        if not store:
            return predictions
        
        # Predict lunch slump (12-14h)
        if 11 <= hour <= 13:
            predictions.append(PredictedState(
                horizon=PredictionHorizon.SHORT,
                predicted_time=now + timedelta(hours=1),
                state_type="risk",
                predicted_behavior={"energy_low": True, "productivity_drop": True},
                confidence=0.6,
                trigger_features={"time_of_day": hour, "pre_lunch": True},
                historical_pattern_match="post_lunch_slump",
                optimal_intervention="suggest_walk_or_nap",
                intervention_window=(now + timedelta(minutes=30), now + timedelta(hours=2)),
                expected_outcome_if_intervene=0.7,
                expected_outcome_if_ignore=0.3
            ))
        
        # Predict evening wind-down need (20-22h)
        if 19 <= hour <= 21:
            predictions.append(PredictedState(
                horizon=PredictionHorizon.SHORT,
                predicted_time=now + timedelta(hours=2),
                state_type="opportunity",
                predicted_behavior={"sleep_prep": True, "screen_time_reduction": True},
                confidence=0.65,
                trigger_features={"evening_time": True, "high_screen_time_today": True},
                historical_pattern_match="evening_ritual",
                optimal_intervention="wind_down_prompt",
                intervention_window=(now + timedelta(hours=1), now + timedelta(hours=3)),
                expected_outcome_if_intervene=0.8,
                expected_outcome_if_ignore=0.4
            ))
        
        return predictions
    
    async def _predict_medium_term(self) -> List[PredictedState]:
        """Predict next 24 hours based on weekly patterns and upcoming events."""
        predictions = []
        return predictions
    
    async def _predict_long_term(self) -> List[PredictedState]:
        """Predict next 7 days based on trend analysis."""
        predictions = []
        return predictions
    
    async def _get_current_context(self) -> Dict:
        """Get user's current behavioral context."""
        store = await get_behavioral_store()
        if not store:
            return {}
        
        one_hour_ago = (datetime.utcnow() - timedelta(hours=1)).isoformat()
        recent = await store.get_user_observations(
            user_id=self.user_id,
            start_time=one_hour_ago,
            limit=100
        )
        
        return {
            "recent_observations": [o.dict() for o in recent],
            "current_hour": datetime.utcnow().hour,
            "day_of_week": datetime.utcnow().weekday()
        }
    
    async def _execute_intervention(self, prediction: PredictedState):
        """
        Execute the optimal intervention with full cognition stack awareness.
        v2.0: All cognition layers inform execution.
        """
        if not await self._check_onboarding_status():
            print(f"Onboarding not complete for {self.user_id}, skipping intervention")
            return
        
        # Check value alignment one more time before execution
        if prediction.value_alignment_score < 0.3:
            print(f"Value conflict detected, aborting intervention: {prediction.value_conflict_risk}")
            await self._log_value_conflict_abort(prediction)
            return
        
        emotional_state = await self._assess_user_emotional_state()
        
        # Build rich context for debate with cognition information
        user_context = await self._build_cognition_aware_context(prediction)
        
        # MAO debate with cognition-aware context
        debate_result = await self.debate_system.debate_intervention(
            user_context=user_context,
            proposed_intervention=prediction.optimal_intervention or "default_intervention",
            predicted_outcome={
                'expected_improvement': prediction.expected_outcome_if_intervene,
                'confidence': prediction.confidence
            }
        )
        
        if not debate_result['proceed']:
            print(f"Adversary blocked intervention for {self.user_id}")
            await self._log_adversary_objection(prediction, debate_result)
            return
        
        # Empathy calibration with cognition context
        intervention_type = self._map_to_agency_type(debate_result, prediction)
        
        # Get value-aligned framing from VIM
        vim_framing = self.vim.get_intervention_framing(
            str(self.user_id),
            prediction.optimal_intervention or "default"
        )
        
        # Merge debate message with value-aligned framing
        final_message = self._merge_framing(
            debate_result.get('message', ''),
            vim_framing,
            prediction.framing_strategy
        )
        
        empathetic_delivery = await self.empathy_wrapper.deliver_intervention(
            raw_intervention=final_message,
            context=prediction.state_type,
            user_emotional_state=emotional_state,
            intervention_type=intervention_type
        )
        
        if not empathetic_delivery['delivered']:
            print(f"Agency preservation blocked intervention: {empathetic_delivery['reason']}")
            await self._log_agency_preservation_block(prediction, empathetic_delivery)
            return
        
        # Execute with fully informed message
        handler = self.intervention_handlers.get(prediction.state_type)
        
        if handler:
            await handler(prediction, self.user_id, {
                **debate_result,
                'message': empathetic_delivery['message'],
                'empathy_mode': empathetic_delivery['empathy_mode'],
                'agency_preserved': True,
                # NEW: Cognition metadata
                'primary_layer': prediction.primary_control_layer.value if prediction.primary_control_layer else None,
                'strategic_pressures': prediction.active_strategic_pressures,
                'value_alignment': prediction.value_alignment_score,
                'framing_strategy': prediction.framing_strategy
            })
        else:
            await self._default_intervention(prediction, {
                **debate_result,
                'message': empathetic_delivery['message'],
                'empathy_mode': empathetic_delivery['empathy_mode'],
                'agency_preserved': True,
                'primary_layer': prediction.primary_control_layer.value if prediction.primary_control_layer else None,
                'strategic_pressures': prediction.active_strategic_pressures,
                'value_alignment': prediction.value_alignment_score,
                'framing_strategy': prediction.framing_strategy
            })
    
    def _merge_framing(
        self,
        debate_message: str,
        vim_framing: Dict[str, Any],
        framing_strategy: str
    ) -> str:
        """Merge multiple framing sources into final message."""
        # Start with debate message
        message = debate_message
        
        # Apply value-aligned language suggestions
        suggested_language = vim_framing.get('suggested_language', [])
        if suggested_language:
            # Enhance message with value-aligned terms
            message = f"{message} ({suggested_language[0]})"
        
        return message
    
    async def _build_cognition_aware_context(self, prediction: PredictedState) -> Dict:
        """Build rich context for debate including all cognition layers."""
        base_context = await self._build_debate_context(prediction)
        
        # Add cognition layer information
        return {
            **base_context,
            'primary_control_layer': prediction.primary_control_layer.value if prediction.primary_control_layer else None,
            'layer_confidence': prediction.layer_confidence,
            'active_strategic_pressures': prediction.active_strategic_pressures,
            'narrative_alignment': prediction.narrative_axis_alignment,
            'value_alignment_score': prediction.value_alignment_score,
            'value_conflict_risk': prediction.value_conflict_risk,
            'framing_strategy': prediction.framing_strategy,
            # Integration insight
            'intervention_design_principle': self._get_design_principle(prediction)
        }
    
    def _get_design_principle(self, prediction: PredictedState) -> str:
        """Get design principle based on cognition stack analysis."""
        if prediction.primary_control_layer == ControlLayer.HOM:
            return "reduce_load_remove_friction"
        elif prediction.primary_control_layer == ControlLayer.HSM:
            return "change_incentives_alter_payoff"
        elif prediction.primary_control_layer == ControlLayer.PNM:
            return "narrative_reframe_identity_bridge"
        return "standard"
    
    async def _assess_user_emotional_state(self) -> str:
        """Assess current emotional state from recent data."""
        store = await get_behavioral_store()
        if not store:
            return "neutral"
        
        recent_reflections = store.client.table('execution_reflections').select('*').eq(
            'user_id', str(self.user_id)
        ).order('created_at', desc=True).limit(3).execute()
        
        if recent_reflections.data:
            moods = [r.get('mood_score') for r in recent_reflections.data if r.get('mood_score')]
            if moods:
                avg_mood = sum(moods) / len(moods)
                if avg_mood < 4:
                    return "struggling"
                elif avg_mood > 7:
                    return "positive"
        
        return "neutral"
    
    def _map_to_agency_type(self, debate_result: Dict, prediction: PredictedState) -> AgencyInterventionType:
        """Map MAO result to agency-preserving intervention type."""
        # Consider value alignment
        if prediction.value_alignment_score < 0.5:
            # Low value alignment - use question format to preserve agency
            return AgencyInterventionType.QUESTION
        
        if prediction.confidence > 0.8 and prediction.state_type == "vulnerability":
            return AgencyInterventionType.DIRECTIVE
        
        if debate_result.get('consensus_reached'):
            return AgencyInterventionType.COLLABORATION
        
        if "suggest" in debate_result.get('message', '').lower():
            return AgencyInterventionType.SUGGESTION
        
        return AgencyInterventionType.SUGGESTION
    
    async def _build_debate_context(self, prediction: PredictedState) -> Dict:
        """Build base context for MAO debate."""
        store = await get_behavioral_store()
        rejection_rate = 0.0
        interventions_today = 0
        
        if store:
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0).isoformat()
            recent = store.client.table('proactive_interventions').select('*').eq(
                'user_id', str(self.user_id)
            ).gte('executed_at', today_start).execute()
            
            if recent.data:
                interventions_today = len(recent.data)
                last_20 = store.client.table('proactive_interventions').select('*').eq(
                    'user_id', str(self.user_id)
                ).order('executed_at', desc=True).limit(20).execute()
                
                if last_20.data:
                    rejected = sum(1 for i in last_20.data if i.get('user_response') == 'dismissed')
                    rejection_rate = rejected / len(last_20.data)
        
        return {
            'current_state': prediction.state_type,
            'recent_pattern': prediction.historical_pattern_match,
            'goal_progress': 0.5,
            'rejection_rate': rejection_rate,
            'interventions_today': interventions_today,
            'predicted_vulnerability': prediction.state_type == 'vulnerability',
            'confidence': prediction.confidence
        }
    
    async def _default_intervention(self, prediction: PredictedState, debate_result: Optional[Dict] = None):
        """Default intervention with full cognition awareness."""
        engine = NotificationEngine(self.user_id)
        
        if debate_result and debate_result.get('message'):
            title = "Spirit"
            body = debate_result['message']
        else:
            title = self._get_intervention_title(prediction)
            body = self._get_intervention_body(prediction)
        
        content = {
            "title": title,
            "body": body,
            "data": {
                "prediction_type": prediction.state_type,
                "confidence": prediction.confidence,
                "expected_outcome": prediction.expected_outcome_if_intervene,
                "debate_validated": debate_result is not None,
                "debate_rounds": debate_result.get('debate_rounds', 0) if debate_result else 0,
                "empathy_mode": debate_result.get('empathy_mode', 'balanced') if debate_result else 'balanced',
                "agency_preserved": debate_result.get('agency_preserved', True) if debate_result else True,
                # NEW: Cognition metadata
                "primary_layer": debate_result.get('primary_layer') if debate_result else None,
                "value_alignment": debate_result.get('value_alignment', 0.5) if debate_result else 0.5,
                "framing_strategy": debate_result.get('framing_strategy', 'neutral') if debate_result else 'neutral'
            }
        }
        
        priority = NotificationPriority.HIGH if prediction.confidence > 0.7 else NotificationPriority.NORMAL
        
        await engine.send_notification(
            content=content,
            priority=priority,
            notification_type=f"proactive_{prediction.state_type}",
            context={"predicted_vulnerability": prediction.state_type == "vulnerability"}
        )
        
        await self._log_proactive_intervention(prediction, debate_result)
    
    def _get_intervention_title(self, prediction: PredictedState) -> str:
        """Generate contextual title based on prediction and cognition."""
        # Layer-aware titles
        if prediction.primary_control_layer == ControlLayer.PNM:
            titles = {
                "vulnerability": "A moment for reflection",
                "opportunity": "This fits who you're becoming",
                "risk": "Your path forward"
            }
        elif prediction.primary_control_layer == ControlLayer.HSM:
            titles = {
                "vulnerability": "Strategic pause?",
                "opportunity": "Optimize this moment",
                "risk": "Cost-benefit check"
            }
        else:
            titles = {
                "vulnerability": "Focus fading? Try this",
                "opportunity": "You're in flow—extend it?",
                "risk": "Heads up: energy dip predicted",
                "maintenance": "Quick check-in"
            }
        
        return titles.get(prediction.state_type, "Spirit here")
    
    def _get_intervention_body(self, prediction: PredictedState) -> str:
        """Generate contextual body text with value awareness."""
        if prediction.state_type == "vulnerability":
            if prediction.primary_control_layer == ControlLayer.PNM:
                return "This moment connects to what matters to you. A small step is still forward."
            elif prediction.primary_control_layer == ControlLayer.HSM:
                return "Current path has low ROI. Alternative available with better payoff."
            return "Your focus score is dropping. 2-minute reset available."
        
        elif prediction.state_type == "opportunity":
            if prediction.primary_control_layer == ControlLayer.PNM:
                return "This aligns with your deeper purpose. Worth extending?"
            return "You've been deep in work for 25 min. Want to lock in for another 25?"
        
        elif prediction.state_type == "risk":
            return "You usually struggle after lunch. Pre-positioned support ready."
        
        return "How are you doing?"
    
    async def _delayed_execution(self, check_id: str, delay_seconds: float, prediction: PredictedState):
        """Execute after delay, if not cancelled."""
        await asyncio.sleep(delay_seconds)
        
        if check_id in self.scheduled_checks:
            await self._execute_intervention(prediction)
            del self.scheduled_checks[check_id]
    
    async def _log_proactive_intervention(self, prediction: PredictedState, debate_result: Optional[Dict] = None):
        """Log for learning with full cognition context."""
        store = await get_behavioral_store()
        if store:
            store.client.table('proactive_interventions').insert({
                'intervention_id': str(uuid4()),
                'user_id': str(self.user_id),
                'predicted_state': prediction.state_type,
                'predicted_time': prediction.predicted_time.isoformat(),
                'confidence': prediction.confidence,
                'intervention_type': prediction.optimal_intervention,
                'executed_at': datetime.utcnow().isoformat(),
                'expected_outcome': prediction.expected_outcome_if_intervene,
                'debate_validated': debate_result is not None,
                'debate_rounds': debate_result.get('debate_rounds', 0) if debate_result else 0,
                'consensus_reached': debate_result.get('consensus_reached', False) if debate_result else False,
                'empathy_mode': debate_result.get('empathy_mode') if debate_result else None,
                'agency_preserved': debate_result.get('agency_preserved', True) if debate_result else True,
                # NEW: Cognition logging
                'primary_control_layer': prediction.primary_control_layer.value if prediction.primary_control_layer else None,
                'active_strategic_pressures': prediction.active_strategic_pressures,
                'value_alignment_score': prediction.value_alignment_score,
                'value_conflict_risk': prediction.value_conflict_risk,
                'framing_strategy': prediction.framing_strategy
            }).execute()
    
    async def _log_value_conflict_skip(self, prediction: PredictedState):
        """Log when value conflict prevents scheduling."""
        store = await get_behavioral_store()
        if store:
            store.client.table('proactive_intervention_skips').insert({
                'skip_id': str(uuid4()),
                'user_id': str(self.user_id),
                'predicted_state': prediction.state_type,
                'reason': 'value_conflict',
                'value_conflict_risk': prediction.value_conflict_risk,
                'value_alignment_score': prediction.value_alignment_score,
                'logged_at': datetime.utcnow().isoformat()
            }).execute()
    
    async def _log_value_conflict_abort(self, prediction: PredictedState):
        """Log when value conflict aborts execution."""
        store = await get_behavioral_store()
        if store:
            store.client.table('proactive_intervention_aborts').insert({
                'abort_id': str(uuid4()),
                'user_id': str(self.user_id),
                'predicted_state': prediction.state_type,
                'reason': 'value_conflict_pre_execution',
                'value_conflict_risk': prediction.value_conflict_risk,
                'value_alignment_score': prediction.value_alignment_score,
                'logged_at': datetime.utcnow().isoformat()
            }).execute()
    
    async def _log_adversary_objection(self, prediction: PredictedState, debate_result: Dict):
        """Log adversary objections."""
        store = await get_behavioral_store()
        if store:
            store.client.table('adversary_objections').insert({
                'user_id': str(self.user_id),
                'predicted_state': prediction.state_type,
                'objection': debate_result.get('adversary_concerns', 'unknown'),
                'debate_rounds': debate_result.get('debate_rounds', 0),
                'intervention_type': prediction.optimal_intervention,
                'logged_at': datetime.utcnow().isoformat()
            }).execute()
    
    async def _log_agency_preservation_block(self, prediction: PredictedState, delivery_result: Dict):
        """Log when agency preservation blocks an intervention."""
        store = await get_behavioral_store()
        if store:
            store.client.table('agency_preservation_logs').insert({
                'log_id': str(uuid4()),
                'user_id': str(self.user_id),
                'predicted_state': prediction.state_type,
                'block_reason': delivery_result.get('reason'),
                'agency_impact_score': delivery_result.get('agency_impact'),
                'logged_at': datetime.utcnow().isoformat()
            }).execute()


class AutonomousExperimentRunner:
    """
    Automatically runs micro-experiments to improve predictions.
    v2.0: Cognition-aware experiment design.
    """
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.experiment_queue: List[Dict] = []
        self.min_hours_between_experiments = 4
        
        # NEW: Cognition access for experiment design
        self.vim = get_values_inference_module()
        self.lae = get_layer_arbitration_engine()
    
    async def design_and_queue_experiment(self, context: Dict):
        """Design a micro-experiment based on current uncertainty and cognition."""
        if not await self._can_run_experiment():
            return
        
        # NEW: Check value alignment before designing experiment
        vim_profile = self.vim.get_value_profile(str(self.user_id))
        
        uncertain_prediction = await self._find_uncertainty(context)
        
        if uncertain_prediction:
            # Design value-aligned experiment
            experiment = self._create_cognition_aware_experiment(
                uncertain_prediction,
                vim_profile
            )
            
            if experiment:
                self.experiment_queue.append(experiment)
                
                if experiment.get("priority") == "high":
                    await self._run_experiment(experiment)
    
    async def _can_run_experiment(self) -> bool:
        """Check if enough time has passed since last experiment."""
        store = await get_behavioral_store()
        if not store:
            return False
        
        last = store.client.table('proactive_experiments').select('*').eq(
            'user_id', str(self.user_id)
        ).order('started_at', desc=True).limit(1).execute()
        
        if not last.data:
            return True
        
        last_time = datetime.fromisoformat(last.data[0]['started_at'])
        hours_since = (datetime.utcnow() - last_time).total_seconds() / 3600
        
        return hours_since > self.min_hours_between_experiments
    
    async def _find_uncertainty(self, context: Dict) -> Optional[Dict]:
        """Find what we're most uncertain about predicting."""
        store = await get_behavioral_store()
        if not store:
            return None
        
        past = store.client.table('proactive_interventions').select('*').eq(
            'user_id', str(self.user_id)
        ).execute()
        
        accuracy_by_type = {}
        for p in past.data if past.data else []:
            state = p['predicted_state']
            if state not in accuracy_by_type:
                accuracy_by_type[state] = {'correct': 0, 'total': 0}
            accuracy_by_type[state]['total'] += 1
        
        most_uncertain = None
        lowest_acc = 1.0
        for state, stats in accuracy_by_type.items():
            if stats['total'] >= 3:
                acc = stats.get('correct', 0) / stats['total']
                if acc < lowest_acc:
                    lowest_acc = acc
                    most_uncertain = state
        
        if most_uncertain and lowest_acc < 0.6:
            return {"state_type": most_uncertain, "current_accuracy": lowest_acc}
        
        return None
    
    def _create_cognition_aware_experiment(
        self,
        uncertainty: Dict,
        vim_profile: Dict[str, Any]
    ) -> Optional[Dict]:
        """Create experiment that respects inferred values."""
        # Get dominant values to design value-aligned experiment
        dominant_values = [v["value"] for v in vim_profile.get("dominant_values", [])]
        
        if not dominant_values:
            return None
        
        primary_value = dominant_values[0]
        
        # Design experiment that tests intervention while respecting value
        experiment_designs = {
            "autonomy": {
                "arms": ["choice_prompt", "directive_prompt", "control_prompt"],
                "hypothesis": "Choice architecture improves outcomes for autonomy-valued users"
            },
            "responsibility": {
                "arms": ["commitment_prompt", "flexible_prompt", "social_prompt"],
                "hypothesis": "Commitment framing improves outcomes for responsibility-valued users"
            },
            "meaning": {
                "arms": ["purpose_prompt", "efficiency_prompt", "process_prompt"],
                "hypothesis": "Purpose connection improves outcomes for meaning-valued users"
            },
            "mastery": {
                "arms": ["challenge_prompt", "easy_prompt", "support_prompt"],
                "hypothesis": "Challenge framing improves outcomes for mastery-valued users"
            }
        }
        
        design = experiment_designs.get(primary_value)
        
        if not design:
            return None
        
        return {
            "experiment_id": str(uuid4()),
            "hypothesis": design["hypothesis"],
            "state_type": uncertainty['state_type'],
            "arms": design["arms"],
            "sample_size": 21,
            "priority": "medium",
            "design": "micro_randomized",
            "value_alignment": primary_value
        }
    
    async def _run_experiment(self, experiment: Dict):
        """Execute the experiment."""
        arm = random.choice(experiment['arms'])
        
        store = await get_behavioral_store()
        if store:
            store.client.table('proactive_experiments').insert({
                'experiment_id': experiment['experiment_id'],
                'user_id': str(self.user_id),
                'hypothesis': experiment['hypothesis'],
                'arm': arm,
                'started_at': datetime.utcnow().isoformat(),
                'status': 'running',
                'value_alignment': experiment.get('value_alignment')
            }).execute()
        
        print(f"Started experiment {experiment['experiment_id']} for user {self.user_id} "
              f"in arm {arm} (aligned with {experiment.get('value_alignment', 'unknown')})")


class GlobalProactiveOrchestrator:
    """
    Manages proactive loops for all active users.
    v2.0: Full cognition stack orchestration.
    """
    
    def __init__(self):
        self.user_loops: Dict[int, ProactiveScheduler] = {}
        self.experiment_runners: Dict[int, AutonomousExperimentRunner] = {}
        self.running = False
    
    async def start_user_loop(self, user_id: int):
        """Start proactive loop for a user with full cognition stack."""
        if user_id in self.user_loops:
            return
        
        scheduler = ProactiveScheduler(user_id)
        
        # Register intervention handlers
        scheduler.register_handler("vulnerability", self._handle_vulnerability)
        scheduler.register_handler("opportunity", self._handle_opportunity)
        scheduler.register_handler("risk", self._handle_risk)
        
        self.user_loops[user_id] = scheduler
        
        asyncio.create_task(scheduler.start())
        
        runner = AutonomousExperimentRunner(user_id)
        self.experiment_runners[user_id] = runner
        
        print(f"Started proactive loop v2.0 for user {user_id} with cognition stack")
    
    def stop_user_loop(self, user_id: int):
        """Stop loop for a user."""
        if user_id in self.user_loops:
            self.user_loops[user_id].stop()
            del self.user_loops[user_id]
        
        if user_id in self.experiment_runners:
            del self.experiment_runners[user_id]
    
    async def _handle_vulnerability(self, prediction: PredictedState, user_id: int, debate_result: Optional[Dict] = None):
        """Handle predicted vulnerability with full cognition awareness."""
        engine = NotificationEngine(user_id)
        
        # Layer-aware message selection
        if prediction.primary_control_layer == ControlLayer.PNM:
            body = debate_result['message'] if debate_result else "This moment matters to your journey. A small step is still forward."
        elif prediction.primary_control_layer == ControlLayer.HSM:
            body = debate_result['message'] if debate_result else "Current path has low ROI. Better alternative available."
        else:
            body = debate_result['message'] if debate_result else "Your attention seems scattered. 30-second reset?"
        
        await engine.send_notification(
            content={
                "title": "Focus check",
                "body": body,
                "action": "focus_reset",
                "data": {
                    "debate_validated": debate_result is not None,
                    "debate_rounds": debate_result.get('debate_rounds', 0) if debate_result else 0,
                    "empathy_mode": debate_result.get('empathy_mode', 'balanced') if debate_result else 'balanced',
                    "agency_preserved": debate_result.get('agency_preserved', True) if debate_result else True,
                    "primary_layer": debate_result.get('primary_layer') if debate_result else None,
                    "value_alignment": debate_result.get('value_alignment', 0.5) if debate_result else 0.5,
                    "framing_strategy": debate_result.get('framing_strategy', 'neutral') if debate_result else 'neutral'
                }
            },
            priority=NotificationPriority.HIGH,
            notification_type="proactive_vulnerability"
        )
    
    async def _handle_opportunity(self, prediction: PredictedState, user_id: int, debate_result: Optional[Dict] = None):
        """Handle predicted opportunity with full cognition awareness."""
        engine = NotificationEngine(user_id)
        
        if debate_result and debate_result.get('message'):
            body = debate_result['message']
        else:
            gain = int(prediction.expected_outcome_if_intervene * 100)
            body = f"You're in flow. Extend this session? Predicted productivity gain: +{gain}%"
        
        await engine.send_notification(
            content={
                "title": "You're in flow",
                "body": body,
                "action": "extend_focus",
                "data": {
                    "debate_validated": debate_result is not None,
                    "debate_rounds": debate_result.get('debate_rounds', 0) if debate_result else 0,
                    "empathy_mode": debate_result.get('empathy_mode', 'balanced') if debate_result else 'balanced',
                    "agency_preserved": debate_result.get('agency_preserved', True) if debate_result else True,
                    "primary_layer": debate_result.get('primary_layer') if debate_result else None,
                    "value_alignment": debate_result.get('value_alignment', 0.5) if debate_result else 0.5
                }
            },
            priority=NotificationPriority.NORMAL,
            notification_type="proactive_opportunity"
        )
    
    async def _handle_risk(self, prediction: PredictedState, user_id: int, debate_result: Optional[Dict] = None):
        """Handle predicted risk with full cognition awareness."""
        engine = NotificationEngine(user_id)
        
        body = debate_result['message'] if debate_result else "Your energy typically drops now. Pre-positioned support ready."
        
        await engine.send_notification(
            content={
                "title": "Heads up",
                "body": body,
                "action": "preventive_break",
                "data": {
                    "debate_validated": debate_result is not None,
                    "debate_rounds": debate_result.get('debate_rounds', 0) if debate_result else 0,
                    "empathy_mode": debate_result.get('empathy_mode', 'balanced') if debate_result else 'balanced',
                    "agency_preserved": debate_result.get('agency_preserved', True) if debate_result else True,
                    "primary_layer": debate_result.get('primary_layer') if debate_result else None
                }
            },
            priority=NotificationPriority.NORMAL,
            notification_type="proactive_risk"
        )


# Global singleton
_orchestrator: Optional[GlobalProactiveOrchestrator] = None


def get_orchestrator() -> GlobalProactiveOrchestrator:
    """Get or create global orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = GlobalProactiveOrchestrator()
    return _orchestrator
'''

print(f"Updated proactive_loop.py created: {len(updated_proactive_loop)} bytes")
