from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from spirit.models.mechanisms import MechanismActivation
from spirit.models.mechanistic_user_model import (
    MechanisticUserModel, UserState, InterventionDose, SimulationResult
)
from spirit.strategies.virtual_experiment_runner import (
    VirtualExperimentRunner, PreRegistration
)

@dataclass
class CounterfactualMemory:
    """Stores simulations alongside reality for fidelity tracking"""
    memory_id: str
    timestamp: datetime
    simulation_result: SimulationResult
    context_hash: str
    real_outcome: Optional[Dict] = None
    fidelity_score: Optional[float] = None

class DigitalTwinOrchestrator:
    """
    Main controller for Layer 3.5
    Sits between RFE (Reality Filter) and MAO (Multi-Agent Debate)
    """
    
    def __init__(self, user_id: str, historical_data: List[Dict] = None):
        self.user_id = user_id
        self.twin = MechanisticUserModel(user_id, historical_data or [])
        self.experimenter = VirtualExperimentRunner(self.twin)
        self.counterfactual_memory = []
        
    def process_hypothesis(
        self,
        hypothesis: str,
        target_mechanism: MechanismActivation,
        current_state: UserState,
        proposed_interventions: List[InterventionDose] = None
    ) -> Dict[str, Any]:
        """
        Main entry point from LangGraph Layer 3
        Returns decision on whether to proceed to real world
        """
        print(f"\n[Digital Twin Layer 3.5] Processing: {hypothesis[:60]}...")
        
        # Step 1: Reality Filter Check
        if not self._passes_reality_filter(hypothesis, current_state):
            return {
                'decision': 'REJECT',
                'reason': 'Fails confound audit',
                'layer': 'RFE'
            }
        
        # Step 2: Phase I - Parameter Optimization
        optimal_params = self.experimenter.design_optimal_experiment(
            hypothesis=hypothesis,
            target_mechanism=target_mechanism,
            initial_state=current_state,
            parameter_space={
                'intensity': [0.2, 0.4, 0.6, 0.8],
                'timing': [
                    current_state.timestamp.hour - 1, 
                    current_state.timestamp.hour,
                    current_state.timestamp.hour + 2
                ],
                'framing': ['gentle', 'identity_safe']
            }
        )
        
        if not optimal_params:
            return {
                'decision': 'REJECT',
                'reason': 'No safe intervention found',
                'layer': 'Digital Twin'
            }
        
        # Step 3: Phase II - Validation
        winning_intervention = self._get_optimal_intervention(optimal_params)
        approved, stats = self.experimenter.run_phase_ii_validation(
            optimal_params, current_state, winning_intervention
        )
        
        if not approved:
            return {
                'decision': 'REJECT',
                'reason': f"Simulation failed: Effect={stats['effect_size']:.3f}",
                'statistics': stats,
                'layer': 'Digital Twin'
            }
        
        # Step 4: Uncertainty Quantification & Routing
        fidelity = self.get_fidelity_report()
        model_confidence = fidelity.get('mean_fidelity', 0.5)
        
        requires_human_oversight = (
            stats['max_risk'] > 0.15 or 
            model_confidence < 0.6 or
            target_mechanism == MechanismActivation.IDENTITY_DEFENSIVE_REASONING
        )
        
        # Store counterfactual
        sim_result = self.twin.simulate_response(current_state, winning_intervention)
        self.counterfactual_memory.append(CounterfactualMemory(
            memory_id=optimal_params.experiment_id,
            timestamp=datetime.now(),
            simulation_result=sim_result,
            context_hash=self._hash_state(current_state)
        ))
        
        return {
            'decision': 'APPROVE_WITH_SIMULATION' if not requires_human_oversight else 'APPROVE_WITH_HUMAN_CHECK',
            'intervention': winning_intervention,
            'pre_registration': optimal_params,
            'predicted_stats': stats,
            'confidence': model_confidence,
            'routing': 'TO_MAO_DEBATE' if not requires_human_oversight else 'TO_HUMAN_REVIEW'
        }
    
    def update_from_real_outcome(self, experiment_id: str, real_outcome: Dict):
        """Post-deployment validation to improve model fidelity"""
        for mem in self.counterfactual_memory:
            if mem.memory_id == experiment_id:
                mem.real_outcome = real_outcome
                predicted = mem.simulation_result.predicted_behavior_change
                actual = real_outcome.get('behavior_change', 0)
                mem.fidelity_score = 1 - abs(predicted - actual)
                
                if mem.fidelity_score < 0.5:
                    print(f"[Digital Twin] High error detected ({mem.fidelity_score:.2f}), retraining...")
                break
    
    def run_counterfactual_post_hoc(
        self, 
        past_state: UserState, 
        alternative: InterventionDose
    ) -> SimulationResult:
        """What if we had done X instead?"""
        return self.twin.simulate_response(past_state, alternative)
    
    def _passes_reality_filter(self, hypothesis: str, state: UserState) -> bool:
        """Basic confound checks"""
        return state.sleep_debt < 6  # Don't experiment on sleep deprivation
    
    def _get_optimal_intervention(self, pre_reg: PreRegistration) -> InterventionDose:
        """Retrieve winning intervention from Phase I"""
        return InterventionDose(
            intervention_type='EMA',
            timing=datetime.now(),
            intensity=0.5,
            framing='identity_safe',
            channel='mobile',
            content=f"Test: {pre_reg.hypothesis}",
            expected_mechanism=pre_reg.mechanism_target
        )
    
    def _hash_state(self, state: UserState) -> str:
        return str(hash(f"{state.cognitive_energy:.2f}_{state.stress_level:.2f}"))
    
    def get_fidelity_report(self) -> Dict:
        """How accurate is this user's Digital Twin?"""
        valid = [m for m in self.counterfactual_memory if m.fidelity_score is not None]
        if not valid:
            return {'status': 'uncalibrated', 'mean_fidelity': 0.5}
        
        scores = [m.fidelity_score for m in valid]
        return {
            'status': 'calibrated' if np.mean(scores) > 0.7 else 'drifting',
            'mean_fidelity': np.mean(scores),
            'n_validated': len(valid)
        }
