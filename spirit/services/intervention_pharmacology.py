
# spirit/services/intervention_pharmacology.py
"""
Intervention Pharmacology - Dose-Response Optimization
Models intervention dosage like drug concentration:
- Too little: no effect
- Optimal: maximal efficacy
- Too much: toxicity (habituation, annoyance, rejection)
Uses Thompson Sampling and Bayesian Optimization for adaptive dosing.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import random
from collections import defaultdict

class DoseResponseCurve(Enum):
    """Types of dose-response relationships"""
    LINEAR = "linear"                    # Effect increases linearly with dose
    SIGMOID = "sigmoid"                  # Classic pharmacological curve
    INVERTED_U = "inverted_u"           # Optimal mid-range (too little/no effect, too much/negative)
    THRESHOLD = "threshold"             # No effect until threshold, then constant
    HORMETIC = "hormetic"               # Low dose beneficial, high dose toxic

@dataclass
class InterventionDosage:
    """A specific dosage configuration for an intervention"""
    # Timing parameters
    frequency_hours: float  # How often (e.g., every 2 hours)
    time_of_day_preference: List[int]  # Preferred hours [9, 14, 20]
    
    # Intensity parameters
    intensity: float  # 0.0 to 1.0 (message directness, notification urgency)
    duration_minutes: float  # How long the intervention lasts
    
    # Content parameters
    framing: str  # 'gentle', 'direct', 'identity_safe', 'challenging'
    personalization_depth: int  # 1-5 (how much user-specific context is used)
    
    # Context parameters
    context_sensitivity: float  # How much context changes the dose

@dataclass
class DoseResponseObservation:
    """Outcome of a specific dosage trial"""
    dosage_config: InterventionDosage
    timestamp: datetime
    
    # Outcomes
    efficacy: float  # 0-1, did it work?
    user_response: str  # 'engaged', 'ignored', 'dismissed', 'completed', 'rejected'
    side_effects: List[str]  # 'annoyance', 'interruption', 'confusion', etc.
    
    # Context
    user_state_at_delivery: Dict[str, float]  # energy, stress, etc.
    time_since_last_intervention: float  # hours
    
    # Derived metrics
    saturation_index: float = 0.0  # Calculated habituation level

class PharmacologicalModel:
    """
    Models dose-response relationship for a specific intervention type.
    Like modeling drug concentration vs effect.
    """
    
    def __init__(self, intervention_type: str, user_id: str):
        self.intervention_type = intervention_type  # 'EMA', 'nudge', 'belief_challenge'
        self.user_id = user_id
        self.curve_type = DoseResponseCurve.INVERTED_U  # Default assumption
        
        # Dose ranges explored
        self.dose_history: List[DoseResponseObservation] = []
        
        # Bayesian parameters for efficacy ( Thompson Sampling priors )
        self.efficacy_alpha = 1.0  # Successes
        self.efficacy_beta = 1.0   # Failures
        
        # Bayesian parameters for toxicity (saturation)
        self.toxicity_alpha = 1.0
        self.toxicity_beta = 1.0
        
        # Context-specific models (different doses for different states)
        self.context_specific_models: Dict[str, 'PharmacologicalModel'] = {}
        
        # Optimal dose estimate (updated via Bayesian Optimization)
        self.optimal_dose_estimate: InterventionDosage = self._default_dose()
        self.exploration_count = 0
    
    def _default_dose(self) -> InterventionDosage:
        """Conservative default dose."""
        return InterventionDosage(
            frequency_hours=4.0,
            time_of_day_preference=[9, 14, 20],
            intensity=0.5,
            duration_minutes=5.0,
            framing='gentle',
            personalization_depth=2,
            context_sensitivity=0.7
        )
    
    def record_observation(self, observation: DoseResponseObservation):
        """Record outcome of a dosage trial and update model."""
        self.dose_history.append(observation)
        
        # Update Bayesian efficacy model
        if observation.user_response in ['engaged', 'completed']:
            self.efficacy_alpha += 1
        elif observation.user_response in ['ignored', 'dismissed']:
            self.efficacy_beta += 1
        
        # Update toxicity model (saturation)
        if 'annoyance' in observation.side_effects or observation.user_response == 'rejected':
            self.toxicity_alpha += 1
        else:
            self.toxicity_beta += 1
        
        # Recalculate optimal dose periodically
        if len(self.dose_history) % 10 == 0:
            self._update_optimal_dose_bayesian_optimization()
    
    def _update_optimal_dose_bayesian_optimization(self):
        """
        Use Bayesian Optimization to find optimal dose parameters.
        Objective: Maximize efficacy while minimizing toxicity.
        """
        if len(self.dose_history) < 5:
            return
        
        # Group by frequency to find optimal
        freq_efficacy = defaultdict(lambda: {'success': 0, 'total': 0, 'toxicity': 0})
        
        for obs in self.dose_history:
            freq = round(obs.dosage_config.frequency_hours, 1)
            freq_efficacy[freq]['total'] += 1
            if obs.user_response in ['engaged', 'completed']:
                freq_efficacy[freq]['success'] += 1
            if 'annoyance' in obs.side_effects:
                freq_efficacy[freq]['toxicity'] += 1
        
        # Find best frequency (highest success with low toxicity)
        best_freq = None
        best_score = -1
        
        for freq, stats in freq_efficacy.items():
            if stats['total'] < 3:
                continue
            
            efficacy_rate = stats['success'] / stats['total']
            toxicity_rate = stats['toxicity'] / stats['total']
            
            # Utility function: efficacy - 2*toxicity (toxicity is expensive)
            utility = efficacy_rate - 2.0 * toxicity_rate
            
            if utility > best_score:
                best_score = utility
                best_freq = freq
        
        if best_freq:
            self.optimal_dose_estimate.frequency_hours = best_freq
        
        # Similarly optimize intensity
        self._optimize_intensity()
    
    def _optimize_intensity(self):
        """Find optimal intensity level."""
        intensity_results = defaultdict(lambda: {'success': 0, 'total': 0, 'toxicity': 0})
        
        for obs in self.dose_history:
            int_bucket = round(obs.dosage_config.intensity * 10) / 10  # 0.1, 0.2, etc.
            intensity_results[int_bucket]['total'] += 1
            if obs.user_response in ['engaged', 'completed']:
                intensity_results[int_bucket]['success'] += 1
            if 'annoyance' in obs.side_effects or obs.user_response == 'rejected':
                intensity_results[int_bucket]['toxicity'] += 1
        
        best_intensity = None
        best_utility = -1
        
        for intensity, stats in intensity_results.items():
            if stats['total'] < 3:
                continue
            
            efficacy = stats['success'] / stats['total']
            toxicity = stats['toxicity'] / stats['total']
            
            # Inverted-U penalty for high intensity
            if intensity > 0.7:
                toxicity += 0.2
            
            utility = efficacy - 1.5 * toxicity
            
            if utility > best_utility:
                best_utility = utility
                best_intensity = intensity
        
        if best_intensity:
            self.optimal_dose_estimate.intensity = best_intensity
    
    def get_recommended_dose(self, user_state: Dict[str, float], 
                            context: str = 'default') -> InterventionDosage:
        """
        Get recommended dosage using Thompson Sampling.
        Balances exploration vs exploitation.
        """
        # 20% chance of exploration (try new dose)
        if random.random() < 0.2 and self.exploration_count < 50:
            self.exploration_count += 1
            return self._exploration_dose()
        
        # 80% exploitation (use best known dose, adjusted for context)
        return self._context_adjusted_dose(user_state)
    
    def _exploration_dose(self) -> InterventionDosage:
        """Generate exploratory dose around current optimum."""
        base = self.optimal_dose_estimate
        
        # Perturb parameters
        new_freq = base.frequency_hours * random.uniform(0.7, 1.3)
        new_freq = max(0.5, min(new_freq, 12.0))  # Clamp 0.5-12 hours
        
        new_intensity = base.intensity + random.uniform(-0.2, 0.2)
        new_intensity = max(0.1, min(new_intensity, 0.9))
        
        return InterventionDosage(
            frequency_hours=new_freq,
            time_of_day_preference=base.time_of_day_preference,
            intensity=new_intensity,
            duration_minutes=base.duration_minutes * random.uniform(0.8, 1.2),
            framing=random.choice(['gentle', 'direct', 'identity_safe']),
            personalization_depth=base.personalization_depth,
            context_sensitivity=base.context_sensitivity
        )
    
    def _context_adjusted_dose(self, user_state: Dict[str, float]) -> InterventionDosage:
        """Adjust optimal dose based on current user state."""
        base = self.optimal_dose_estimate
        
        # High energy = can tolerate more frequent/intense
        energy_multiplier = 1.0 + (user_state.get('cognitive_energy', 0.5) - 0.5)
        
        # High stress = reduce intensity
        stress_reduction = 1.0 - (user_state.get('stress_level', 0.3) * 0.5)
        
        # High identity threat = must be gentle
        if user_state.get('identity_threat_level', 0) > 0.6:
            framing = 'identity_safe'
            intensity_cap = 0.4
        else:
            framing = base.framing
            intensity_cap = 1.0
        
        adjusted_intensity = min(base.intensity * stress_reduction, intensity_cap)
        adjusted_frequency = base.frequency_hours / max(energy_multiplier, 0.5)
        
        return InterventionDosage(
            frequency_hours=adjusted_frequency,
            time_of_day_preference=base.time_of_day_preference,
            intensity=adjusted_intensity,
            duration_minutes=base.duration_minutes,
            framing=framing,
            personalization_depth=base.personalization_depth,
            context_sensitivity=base.context_sensitivity
        )
    
    def calculate_saturation_risk(self, recent_interventions: List[datetime]) -> float:
        """
        Calculate current saturation level (0-1).
        High saturation = user is annoyed/habituated.
        """
        if not recent_interventions:
            return 0.0
        
        # Time since last intervention
        now = datetime.utcnow()
        hours_since_last = (now - max(recent_interventions)).total_seconds() / 3600
        
        # Frequency of recent interventions
        last_24h = [i for i in recent_interventions if (now - i).total_seconds() < 86400]
        frequency_24h = len(last_24h)
        
        # Rejection rate in recent interventions
        recent_obs = [o for o in self.dose_history 
                     if o.timestamp > now - timedelta(hours=24)]
        if recent_obs:
            rejection_rate = sum(1 for o in recent_obs 
                               if o.user_response in ['dismissed', 'rejected']) / len(recent_obs)
        else:
            rejection_rate = 0.0
        
        # Saturation formula: function of frequency and rejection
        saturation = (frequency_24h / 10.0) * 0.3 + rejection_rate * 0.7
        
        # Recovery: saturation decays with time since last intervention
        recovery = min(1.0, hours_since_last / 4.0)  # Full recovery after 4 hours
        
        return max(0.0, min(1.0, saturation * (1 - recovery)))


class InterventionPharmacologist:
    """
    Main service for dose-response optimization across all intervention types.
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        
        # Separate pharmacological model for each intervention type
        self.models: Dict[str, PharmacologicalModel] = {
            'EMA': PharmacologicalModel('EMA', user_id),
            'nudge': PharmacologicalModel('nudge', user_id),
            'belief_challenge': PharmacologicalModel('belief_challenge', user_id),
            'micro_experiment': PharmacologicalModel('micro_experiment', user_id),
            'focus_prompt': PharmacologicalModel('focus_prompt', user_id)
        }
        
        # Track intervention history for saturation calculation
        self.intervention_history: List[datetime] = []
    
    def get_optimal_dosage(self, intervention_type: str, 
                          user_state: Dict[str, float]) -> InterventionDosage:
        """
        Get optimal dosage for specific intervention type and user state.
        """
        if intervention_type not in self.models:
            # Create model on demand
            self.models[intervention_type] = PharmacologicalModel(intervention_type, self.user_id)
        
        model = self.models[intervention_type]
        
        # Check saturation before recommending dose
        saturation = model.calculate_saturation_risk(self.intervention_history)
        
        if saturation > 0.7:
            # High saturation - recommend minimal dose or None
            return InterventionDosage(
                frequency_hours=12.0,  # Very infrequent
                time_of_day_preference=[],
                intensity=0.1,
                duration_minutes=1.0,
                framing='gentle',
                personalization_depth=1,
                context_sensitivity=0.9
            )
        elif saturation > 0.4:
            # Moderate saturation - reduce dose
            base_dose = model.get_recommended_dose(user_state)
            base_dose.intensity *= 0.7
            base_dose.frequency_hours *= 1.5
            return base_dose
        else:
            # Normal operation
            return model.get_recommended_dose(user_state)
    
    def record_intervention_outcome(self, intervention_type: str,
                                   dosage: InterventionDosage,
                                   outcome: Dict[str, Any]):
        """Record outcome to update dose-response model."""
        if intervention_type not in self.models:
            return
        
        observation = DoseResponseObservation(
            dosage_config=dosage,
            timestamp=datetime.utcnow(),
            efficacy=outcome.get('efficacy', 0.0),
            user_response=outcome.get('response', 'ignored'),
            side_effects=outcome.get('side_effects', []),
            user_state_at_delivery=outcome.get('user_state', {}),
            time_since_last_intervention=self._hours_since_last_intervention()
        )
        
        self.models[intervention_type].record_observation(observation)
        self.intervention_history.append(datetime.utcnow())
    
    def _hours_since_last_intervention(self) -> float:
        """Calculate hours since last intervention."""
        if not self.intervention_history:
            return 999.0
        
        hours = (datetime.utcnow() - max(self.intervention_history)).total_seconds() / 3600
        return hours
    
    def get_dose_response_report(self) -> Dict[str, Any]:
        """Get summary of dose-response learning for this user."""
        report = {}
        
        for int_type, model in self.models.items():
            if model.dose_history:
                report[int_type] = {
                    'n_observations': len(model.dose_history),
                    'optimal_frequency_hours': model.optimal_dose_estimate.frequency_hours,
                    'optimal_intensity': model.optimal_dose_estimate.intensity,
                    'current_saturation': model.calculate_saturation_risk(self.intervention_history),
                    'efficacy_rate': model.efficacy_alpha / (model.efficacy_alpha + model.efficacy_beta),
                    'toxicity_rate': model.toxicity_alpha / (model.toxicity_alpha + model.toxicity_beta)
                }
        
        return report
    
    def get_context_specific_recommendation(self, context_type: str,
                                           user_state: Dict[str, float]) -> str:
        """
        Get intervention type recommendation based on context.
        Different contexts tolerate different intervention types.
        """
        saturation_by_type = {
            int_type: model.calculate_saturation_risk(self.intervention_history)
            for int_type, model in self.models.items()
        }
        
        # Context rules
        if context_type == 'deep_work':
            # Only gentle nudges during deep work
            candidates = ['nudge', 'focus_prompt']
        elif context_type == 'transition':
            # EMAs work well at transition points
            candidates = ['EMA', 'micro_experiment']
        elif context_type == 'crisis':
            # High intensity acceptable during crisis
            candidates = ['belief_challenge', 'EMA']
        else:
            candidates = list(self.models.keys())
        
        # Pick least saturated from candidates
        best = min(candidates, key=lambda x: saturation_by_type.get(x, 1.0))
        
        return best


# Global registry
_pharmacologists: Dict[str, InterventionPharmacologist] = {}


def get_pharmacologist(user_id: str) -> InterventionPharmacologist:
    """Get or create pharmacologist for user."""
    if user_id not in _pharmacologists:
        _pharmacologists[user_id] = InterventionPharmacologist(user_id)
    return _pharmacologists[user_id]


print("✓ Intervention Pharmacology service created")
print("  - Dose-response modeling for each intervention type")
print("  - Thompson Sampling for exploration/exploitation")
print("  - Bayesian Optimization for optimal dose finding")
print("  - Saturation tracking to prevent habituation")
