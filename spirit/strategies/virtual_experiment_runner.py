import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import product
import uuid
from scipy import stats

from spirit.models.mechanisms import MechanismActivation
from spirit.models.mechanistic_user_model import (
    MechanisticUserModel, UserState, InterventionDose, SimulationResult
)

class ExperimentStatus:
    PRE_REGISTERED = "pre_registered"
    READY_FOR_DEPLOYMENT = "ready"
    REJECTED_BY_SIMULATION = "rejected"

@dataclass
class VirtualCohort:
    """Simulated user instances for A/B testing"""
    cohort_id: str
    size: int
    base_state: UserState
    interventions: List[InterventionDose]
    outcomes: List[SimulationResult] = field(default_factory=list)

@dataclass
class PreRegistration:
    """Immutable hypothesis registration"""
    experiment_id: str
    hypothesis: str
    primary_outcome: str
    mechanism_target: MechanismActivation
    falsification_criteria: str
    n_simulations: int
    created_at: datetime
    status: str = ExperimentStatus.PRE_REGISTERED

class VirtualExperimentRunner:
    """
    Runs parallel universe simulations to optimize interventions
    Equivalent to clinical trial Phase I/II before human testing
    """
    
    def __init__(self, twin_model: MechanisticUserModel):
        self.model = twin_model
        self.experiment_log = []
        self.disproven_hypotheses_archive = []
        
    def design_optimal_experiment(
        self,
        hypothesis: str,
        target_mechanism: MechanismActivation,
        initial_state: UserState,
        parameter_space: Dict[str, List[Any]]
    ) -> Optional[PreRegistration]:
        """
        Phase I: Design experiment using simulations to find optimal parameters
        """
        experiment_id = str(uuid.uuid4())[:8]
        
        candidates = self._generate_intervention_grid(target_mechanism, parameter_space)
        
        best_candidate = None
        best_utility = -1
        
        print(f"[Virtual Experiment {experiment_id}] Phase I screening...")
        print(f"  Testing {len(candidates)} variants on virtual cohort (n=100)")
        
        for candidate in candidates:
            cohort = self._run_virtual_cohort(
                initial_state, candidate, n=100, time_horizon=timedelta(hours=2)
            )
            
            effect_size = np.mean([o.predicted_behavior_change for o in cohort.outcomes])
            success_rate = np.mean([o.probability_of_success for o in cohort.outcomes])
            avg_risk = np.mean([o.risk_score for o in cohort.outcomes])
            
            utility = (effect_size * 0.6) + (success_rate * 0.3) - (avg_risk * 0.4)
            
            if utility > best_utility and avg_risk < 0.3:
                best_utility = utility
                best_candidate = candidate
        
        if not best_candidate:
            return None
            
        print(f"  Optimal: Intensity={best_candidate.intensity:.2f}, "
              f"Risk={avg_risk:.2f}")
        
        return PreRegistration(
            experiment_id=experiment_id,
            hypothesis=hypothesis,
            primary_outcome="behavioral_stability_change",
            mechanism_target=target_mechanism,
            falsification_criteria="Effect < 0.15 or risk > 0.3",
            n_simulations=1000,
            created_at=datetime.now()
        )
    
    def run_phase_ii_validation(
        self, 
        pre_reg: PreRegistration,
        initial_state: UserState,
        intervention: InterventionDose
    ) -> Tuple[bool, Dict]:
        """
        Phase II: Large-scale simulation (n=1000) with statistical validation
        """
        print(f"[Virtual Experiment {pre_reg.experiment_id}] Phase II validation...")
        
        # Treatment group
        treatment_cohort = self._run_virtual_cohort(
            initial_state, intervention, n=500, time_horizon=timedelta(days=1)
        )
        
        # Control group (null intervention)
        control = InterventionDose(
            intervention_type="null",
            timing=intervention.timing,
            intensity=0,
            framing="neutral",
            channel="none",
            content="",
            expected_mechanism=MechanismActivation.COGNITIVE_ENERGY_DEPLETION
        )
        control_cohort = self._run_virtual_cohort(
            initial_state, control, n=500, time_horizon=timedelta(days=1)
        )
        
        treatment_effects = [o.predicted_behavior_change for o in treatment_cohort.outcomes]
        control_effects = [o.predicted_behavior_change for o in control_cohort.outcomes]
        
        effect_size = np.mean(treatment_effects) - np.mean(control_effects)
        t_stat, p_value = stats.ttest_ind(treatment_effects, control_effects)
        max_risk = max([o.risk_score for o in treatment_cohort.outcomes])
        
        stats_result = {
            'effect_size': effect_size,
            'p_value': p_value,
            'treatment_mean': np.mean(treatment_effects),
            'control_mean': np.mean(control_effects),
            'max_risk': max_risk
        }
        
        approved = (
            effect_size > 0.15 and 
            p_value < 0.05 and 
            max_risk < 0.25
        )
        
        if approved:
            pre_reg.status = ExperimentStatus.READY_FOR_DEPLOYMENT
            print(f"  APPROVED: Effect={effect_size:.3f}, p={p_value:.3f}")
        else:
            pre_reg.status = ExperimentStatus.REJECTED_BY_SIMULATION
            self.disproven_hypotheses_archive.append({
                'hypothesis': pre_reg.hypothesis,
                'stats': stats_result
            })
            print(f"  REJECTED: Effect={effect_size:.3f}, Risk={max_risk:.2f}")
        
        return approved, stats_result
    
    def _generate_intervention_grid(
        self, 
        mechanism: MechanismActivation, 
        params: Dict
    ) -> List[InterventionDose]:
        """Generate combinatorial intervention space"""
        intensities = params.get('intensity', [0.3, 0.5, 0.7])
        timings = params.get('timing', [9, 14, 20])
        framings = params.get('framing', ['gentle', 'identity_safe'])
        
        candidates = []
        for inten, hour, frame in product(intensities, timings, framings):
            if isinstance(hour, int):
                time_val = datetime.now().replace(hour=hour % 24, minute=0)
            else:
                time_val = hour
                
            candidates.append(InterventionDose(
                intervention_type='micro_experiment',
                timing=time_val,
                intensity=inten,
                framing=frame,
                channel='mobile',
                content=f"Test: {mechanism.name}",
                expected_mechanism=mechanism
            ))
        return candidates
    
    def _run_virtual_cohort(
        self, 
        base_state: UserState, 
        intervention: InterventionDose,
        n: int,
        time_horizon: timedelta
    ) -> VirtualCohort:
        """Run n parallel simulations with biological variability"""
        cohort = VirtualCohort(
            cohort_id=str(uuid.uuid4())[:6],
            size=n,
            base_state=base_state,
            interventions=[intervention]
        )
        
        for i in range(n):
            perturbed = self._perturb_state(base_state, seed=i)
            result = self.model.simulate_response(perturbed, intervention, time_horizon)
            cohort.outcomes.append(result)
            
        return cohort
    
    def _perturb_state(self, state: UserState, seed: int) -> UserState:
        """Add realistic biological noise"""
        np.random.seed(seed)
        noise = lambda x: np.clip(x + np.random.normal(0, 0.05), 0, 1)
        
        return UserState(
            timestamp=state.timestamp,
            cognitive_energy=noise(state.cognitive_energy),
            sleep_debt=max(0, state.sleep_debt + np.random.normal(0, 0.5)),
            glucose_stability=noise(state.glucose_stability),
            stress_level=noise(state.stress_level),
            attentional_bandwidth=noise(state.attentional_bandwidth),
            working_memory_load=max(0, min(4, state.working_memory_load + np.random.normal(0, 0.2))),
            social_load=noise(state.social_load),
            identity_threat_level=noise(state.identity_threat_level),
            current_context=state.current_context,
            recent_interventions=state.recent_interventions.copy()
        )
