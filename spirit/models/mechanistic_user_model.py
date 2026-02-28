import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from spirit.models.mechanisms import MechanismActivation

@dataclass
class UserState:
    """Current latent state of the user for Digital Twin simulation"""
    timestamp: datetime
    cognitive_energy: float  # 0-1, front-loaded daily
    sleep_debt: float  # hours deficit
    glucose_stability: float  # 0-1
    stress_level: float  # 0-1
    attentional_bandwidth: float  # 0-1
    working_memory_load: float  # 0-1, 4 chunks max
    social_load: float  # 0-1, recent interaction burden
    identity_threat_level: float  # 0-1, defensive reasoning trigger
    current_context: str  # location, social setting
    recent_interventions: List[str] = field(default_factory=list)
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for simulation"""
        return np.array([
            self.cognitive_energy,
            self.sleep_debt,
            self.glucose_stability,
            self.stress_level,
            self.attentional_bandwidth,
            self.working_memory_load,
            self.social_load,
            self.identity_threat_level
        ])

@dataclass
class InterventionDose:
    """Intervention parameters with pharmacological precision"""
    intervention_type: str  # 'EMA', 'nudge', 'belief_challenge', 'environmental_mod'
    timing: datetime
    intensity: float  # 0-1, dosage
    framing: str  # 'gentle', 'direct', 'identity_safe'
    channel: str  # 'mobile', 'desktop', 'wearable_haptic'
    content: str
    expected_mechanism: MechanismActivation  # which of the 47 mechanisms it targets

@dataclass
class SimulationResult:
    """Outcome of a counterfactual run"""
    simulation_id: str
    hypothesis: str
    predicted_behavior_change: float  # -1 to 1
    confidence_interval: Tuple[float, float]
    mechanism_activations: Dict[MechanismActivation, float]
    probability_of_success: float  # 0-1
    expected_user_response: str
    risk_score: float  # 0-1, ethical/safety risk
    calibration_score: float  # how similar this sim is to past real data

class MechanisticUserModel:
    """
    Core generative model combining:
    1. Hard-coded mechanistic regularities (the 39 principles)
    2. Learned user-specific residuals (from PEL Level 4-5)
    """
    
    def __init__(self, user_id: str, historical_data: List[Dict] = None):
        self.user_id = user_id
        self.mechanism_weights = self._initialize_mechanisms()
        self.context_patterns = self._learn_context_patterns(historical_data or [])
        self.prediction_error_history = []
        
    def _initialize_mechanisms(self) -> Dict[MechanismActivation, float]:
        """Base rates for mechanisms, tuned by user-specific data"""
        return {mech: 0.5 for mech in MechanismActivation}
    
    def _learn_context_patterns(self, data: List[Dict]) -> Dict:
        """Extract Level 4-5 PEL data: stable traits, not artifacts"""
        patterns = {
            'chronotype': 'evening',
            'cognitive_latency': 'high',
            'ambiguity_sensitivity': 0.7,
            'social_recharge_need': 0.5,
            'intervention_saturation_curve': lambda x: 1 / (1 + np.exp(5*(x-0.5)))
        }
        return patterns
    
    def simulate_response(
        self, 
        initial_state: UserState, 
        intervention: InterventionDose,
        time_horizon: timedelta = timedelta(hours=1)
    ) -> SimulationResult:
        """
        Run counterfactual: What happens if we apply intervention at this state?
        """
        # Step 1: Calculate mechanism activations (the 39 regularities)
        activations = self._calculate_mechanism_load(initial_state, intervention)
        
        # Step 2: Compute behavioral outcome based on target mechanism
        target_mech = intervention.expected_mechanism
        base_response = self._mechanism_to_behavior(target_mech, activations.get(target_mech, 0.5))
        
        # Step 3: Apply interaction effects (mechanisms compound non-linearly)
        interaction_mod = self._calculate_interactions(activations, initial_state)
        
        # Step 4: Add user-specific residual (learned from PEL)
        user_residual = self._get_user_residual(initial_state, intervention)
        
        # Final prediction with uncertainty
        predicted_change = np.clip(base_response * interaction_mod + user_residual, -1, 1)
        uncertainty = self._estimate_uncertainty(initial_state, intervention)
        
        # Risk assessment (Constitutional check)
        risk = self._assess_intervention_risk(intervention, activations, initial_state)
        
        return SimulationResult(
            simulation_id=f"sim_{datetime.now().timestamp()}",
            hypothesis=f"{intervention.intervention_type} targeting {target_mech.name}",
            predicted_behavior_change=predicted_change,
            confidence_interval=(predicted_change - uncertainty, predicted_change + uncertainty),
            mechanism_activations=activations,
            probability_of_success=0.5 + (predicted_change * 0.5) if predicted_change > 0 else 0.5 - (abs(predicted_change) * 0.3),
            expected_user_response=self._generate_expected_response(intervention, activations),
            risk_score=risk,
            calibration_score=self._calculate_calibration(initial_state)
        )
    
    def _calculate_mechanism_load(
        self, 
        state: UserState, 
        intervention: InterventionDose
    ) -> Dict[MechanismActivation, float]:
        """Calculate activation strength for all 39 mechanisms"""
        activations = {}
        
        # ENERGY & NEUROBIOLOGY (1-7)
        activations[MechanismActivation.COGNITIVE_ENERGY_DEPLETION] = 1 - state.cognitive_energy
        activations[MechanismActivation.SLEEP_DEBT_IMPAIRMENT] = min(state.sleep_debt / 3, 1.0)
        activations[MechanismActivation.DECISION_FATIGUE] = 1 - state.cognitive_energy * 0.8
        activations[MechanismActivation.GLUCASE_STABILITY] = 1 - state.glucose_stability
        activations[MechanismActivation.STRESS_NARROWING] = state.stress_level
        activations[MechanismActivation.HIGH_AROUSAL_REACTIVE] = state.stress_level * 0.8
        activations[MechanismActivation.FATIGUE_FAMILIARITY_BIAS] = 1 - state.cognitive_energy
        
        # ATTENTION & INFORMATION (8-15)
        activations[MechanismActivation.WORKING_MEMORY_OVERLOAD] = state.working_memory_load / 4.0
        activations[MechanismActivation.AMBIGUITY_COST] = 0.5 + (state.stress_level * 0.3)
        activations[MechanismActivation.TASK_SWITCHING_MOMENTUM] = 0.3 if len(state.recent_interventions) > 2 else 0.1
        activations[MechanismActivation.INTERRUPTION_RESET] = 0.4 if len(state.recent_interventions) > 0 else 0.1
        activations[MechanismActivation.COGNITIVE_FLUENCY_OPT] = state.cognitive_energy * 0.7
        activations[MechanismActivation.STORY_COMPRESSION] = 0.6
        activations[MechanismActivation.ATTENTION_RESIDUE] = 0.2 if state.working_memory_load > 2 else 0.1
        activations[MechanismActivation.COMPLEXITY_PERCEPTION] = 0.4 + (state.stress_level * 0.4)
        
        # MOTIVATION & REWARD (16-23)
        activations[MechanismActivation.PREDICTED_REWARD_FOLLOWING] = 0.5
        activations[MechanismActivation.PROGRESS_SIGNALS] = 0.6 if state.cognitive_energy > 0.4 else 0.3
        activations[MechanismActivation.IMMEDIATE_CERTAINTY_BIAS] = 0.5 + (state.stress_level * 0.3)
        activations[MechanismActivation.EFFORT_ANTICIPATION] = 1 - state.cognitive_energy
        activations[MechanismActivation.THREAT_PREDICTION_ERROR] = state.identity_threat_level * 0.7
        activations[MechanismActivation.NOVELTY_INITIATION] = 0.5
        activations[MechanismActivation.SMALL_WINS_PREFERENCE] = 0.6
        activations[MechanismActivation.COMPLETION_SATISFACTION_DECAY] = 0.4
        
        # AVOIDANCE & THREAT (24-32)
        activations[MechanismActivation.AVOIDANCE_REINFORCEMENT] = (
            state.stress_level * 0.5 + state.identity_threat_level * 0.5
        )
        activations[MechanismActivation.UNCERTAINTY_AS_THREAT] = state.stress_level * 0.8
        activations[MechanismActivation.IDENTITY_DEFENSIVE_REASONING] = (
            state.identity_threat_level * 0.8 if 'identity' in intervention.framing else 
            state.identity_threat_level * 0.3
        )
        activations[MechanismActivation.SELF_EVALUATION_AMBIGUITY] = state.identity_threat_level * 0.6
        activations[MechanismActivation.SOCIAL_JUDGMENT_FEAR] = state.social_load * 0.7
        activations[MechanismActivation.UNCERTAINTY_NEUROLOGICAL_DANGER] = state.stress_level
        activations[MechanismActivation.LOSS_AVERSION_2X] = 0.6
        activations[MechanismActivation.SELF_COHERENCE_PROTECTION] = state.identity_threat_level
        
        # LEARNING & BELIEF (33-39)
        activations[MechanismActivation.BELIEF_UPDATE_FROM_EXPERIENCE] = 0.5
        activations[MechanismActivation.CONTRADICTORY_EVIDENCE_REINTERPRET] = 0.4 + state.identity_threat_level * 0.4
        activations[MechanismActivation.SMALL_CONSISTENT_SIGNALS] = 0.5
        activations[MechanismActivation.ACTION_FASTER_THAN_THOUGHT] = 0.5
        activations[MechanismActivation.TRAIT_INFERENCE_FROM_STATES] = 0.4
        activations[MechanismActivation.HABIT_CONTEXT_STABILITY] = 0.3
        activations[MechanismActivation.SOCIAL_OBSERVATION_EFFECT] = 0.2 + state.social_load * 0.3
        activations[MechanismActivation.IDENTITY_CONSISTENCY_OVERRIDE] = state.identity_threat_level * 0.6
        
        return activations
    
    def _mechanism_to_behavior(self, mechanism: MechanismActivation, activation: float) -> float:
        """Convert mechanism activation to behavioral change magnitude"""
        effect_sizes = {
            MechanismActivation.COGNITIVE_ENERGY_DEPLETION: -0.8,
            MechanismActivation.PROGRESS_SIGNALS: 0.6,
            MechanismActivation.AVOIDANCE_REINFORCEMENT: -0.7,
            MechanismActivation.SMALL_WINS_PREFERENCE: 0.4,
            MechanismActivation.IDENTITY_DEFENSIVE_REASONING: -0.9,
            MechanismActivation.AMBIGUITY_COST: -0.6,
        }
        default_effect = 0.3
        effect = effect_sizes.get(mechanism, default_effect)
        return effect * activation
    
    def _calculate_interactions(
        self, 
        activations: Dict[MechanismActivation, float], 
        state: UserState
    ) -> float:
        """Non-linear interactions between mechanisms"""
        sleep_stress = (activations.get(MechanismActivation.SLEEP_DEBT_IMPAIRMENT, 0) * 
                       activations.get(MechanismActivation.STRESS_NARROWING, 0))
        
        if sleep_stress > 0.5:
            return 0.7
        return 1.0
    
    def _get_user_residual(self, state: UserState, intervention: InterventionDose) -> float:
        """Learned user-specific deviation from population model"""
        return np.random.normal(0, 0.1)
    
    def _estimate_uncertainty(self, state: UserState, intervention: InterventionDose) -> float:
        """Uncertainty quantification"""
        base_uncertainty = 0.2
        if state.stress_level > 0.8:
            base_uncertainty += 0.15
        return min(base_uncertainty, 0.5)
    
    def _assess_intervention_risk(
        self, 
        intervention: InterventionDose, 
        activations: Dict,
        state: UserState
    ) -> float:
        """Constitutional risk assessment"""
        risk = 0.0
        if (activations.get(MechanismActivation.IDENTITY_DEFENSIVE_REASONING, 0) > 0.6 and
            intervention.expected_mechanism == MechanismActivation.BELIEF_UPDATE_FROM_EXPERIENCE):
            risk += 0.4
            
        if state.cognitive_energy < 0.2 and intervention.intensity > 0.7:
            risk += 0.5
            
        return min(risk, 1.0)
    
    def _generate_expected_response(
        self, 
        intervention: InterventionDose, 
        activations: Dict
    ) -> str:
        """Generate natural language prediction of user reaction"""
        if activations.get(MechanismActivation.IDENTITY_DEFENSIVE_REASONING, 0) > 0.5:
            return "Likely deflection or justification; may feel judged"
        elif activations.get(MechanismActivation.AVOIDANCE_REINFORCEMENT, 0) > 0.5:
            return "Postponement or distraction-seeking"
        elif activations.get(MechanismActivation.PROGRESS_SIGNALS, 0) > 0.6:
            return "Mood lift, continuation of task"
        return "Neutral acknowledgment"
    
    def _calculate_calibration(self, state: UserState) -> float:
        """How similar is this state to previously validated simulations?"""
        return 0.75
