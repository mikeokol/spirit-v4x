
# spirit/cognition/ontology_expansion.py
"""
Dynamic Ontology Expansion - The Discovery Layer
Discovers new mechanistic regularities beyond the 39 base HOM mechanisms.
Monitors prediction failures to hypothesize missing variables.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from collections import defaultdict

class MechanismHypothesisStatus(Enum):
    PROPOSED = "proposed"           # Initial hypothesis from pattern detection
    TESTING = "testing"             # Under empirical validation
    VALIDATED = "validated"         # Meets criteria for new mechanism
    REJECTED = "rejected"           # Falsified
    INTEGRATED = "integrated"       # Added to User-Specific Mechanism Library

@dataclass
class MechanismHypothesis:
    """A hypothesized new mechanism discovered from pattern analysis"""
    hypothesis_id: str
    user_id: str
    name: str  # e.g., "Post-Call Cognitive Residue"
    description: str  # Mechanistic explanation
    trigger_pattern: Dict[str, Any]  # What conditions activate this?
    predicted_effect: str  # How does this affect behavior?
    status: MechanismHypothesisStatus
    
    # Evidence accumulation
    supporting_observations: List[str] = field(default_factory=list)
    contradicting_observations: List[str] = field(default_factory=list)
    prediction_tests: List[Dict] = field(default_factory=list)
    
    # Validation metrics
    base_rate_in_user: float = 0.0  # How often does this trigger?
    effect_size_when_active: float = 0.0
    confidence: float = 0.0
    
    # Tracking
    proposed_at: datetime = field(default_factory=datetime.utcnow)
    validated_at: Optional[datetime] = None
    integrated_at: Optional[datetime] = None

@dataclass
class PredictionFailurePattern:
    """A cluster of prediction failures suggesting missing mechanism"""
    pattern_id: str
    user_id: str
    
    # What failed?
    base_mechanism: str  # Which of 39 mechanisms was used?
    predicted_outcome: str
    actual_outcome: str
    
    # Context where failure occurred
    context_signature: Dict[str, Any]  # Time, location, preceding events, etc.
    temporal_pattern: Optional[str]  # "After video calls", "Before meals", etc.
    
    # Clustering info
    n_occurrences: int = 1
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    
    # Potential mechanism hint
    suggested_variable: Optional[str] = None  # "Social exhaustion", "Decision debt", etc.

class OntologyExpansionAgent:
    """
    Discovers new mechanisms by analyzing gaps in current HOM predictions.
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.mechanism_id_counter = 40  # Start after existing 39
        self.active_hypotheses: Dict[str, MechanismHypothesis] = {}
        self.detected_patterns: Dict[str, PredictionFailurePattern] = {}
        
    async def analyze_disproven_hypotheses(self, disproven_archive: List[Dict]) -> List[MechanismHypothesis]:
        """
        Main entry point: Scan archive for systematic prediction failures.
        Returns new mechanism hypotheses to test.
        """
        # Group failures by context similarity
        failure_clusters = self._cluster_failures_by_context(disproven_archive)
        
        new_hypotheses = []
        for cluster in failure_clusters:
            if len(cluster) >= 3:  # Minimum pattern threshold
                # Check if existing 39 mechanisms explain this
                if not self._explained_by_existing_mechanisms(cluster):
                    hypothesis = self._generate_mechanism_hypothesis(cluster)
                    if hypothesis:
                        self.active_hypotheses[hypothesis.hypothesis_id] = hypothesis
                        new_hypotheses.append(hypothesis)
        
        return new_hypotheses
    
    def _cluster_failures_by_context(self, failures: List[Dict]) -> List[List[Dict]]:
        """Group prediction failures by shared context signatures."""
        clusters = defaultdict(list)
        
        for failure in failures:
            # Extract context fingerprint
            context = failure.get('context', {})
            
            # Create signature based on key features
            signature_parts = []
            
            # Time-based patterns
            timestamp = failure.get('timestamp', '')
            if timestamp:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                hour = dt.hour
                if 9 <= hour <= 11:
                    signature_parts.append("morning")
                elif 14 <= hour <= 16:
                    signature_parts.append("afternoon")
                elif 20 <= hour <= 22:
                    signature_parts.append("evening")
            
            # Activity patterns
            preceding_activity = context.get('preceding_activity', '')
            if preceding_activity:
                signature_parts.append(f"after_{preceding_activity}")
            
            # Social context
            if context.get('had_video_call', False):
                signature_parts.append("post_call")
            if context.get('social_interaction_intensity', 0) > 0.7:
                signature_parts.append("high_social_load")
            
            # Cognitive context
            if context.get('task_switches', 0) > 5:
                signature_parts.append("high_switching")
            if context.get('decision_count', 0) > 10:
                signature_parts.append("decision_fatigue")
            
            signature = "_".join(sorted(signature_parts)) if signature_parts else "general"
            clusters[signature].append(failure)
        
        return list(clusters.values())
    
    def _explained_by_existing_mechanisms(self, failure_cluster: List[Dict]) -> bool:
        """
        Check if existing 39 mechanisms can explain this failure pattern.
        Uses confound detection from RFE.
        """
        # Check for obvious confounds first
        contexts = [f.get('context', {}) for f in failure_cluster]
        
        # High sleep debt explains many failures
        sleep_debts = [c.get('sleep_debt', 0) for c in contexts]
        if np.mean(sleep_debts) > 2.0:
            return True  # Explained by Mechanism #2
        
        # High stress explains many failures
        stress_levels = [c.get('stress_level', 0) for c in contexts]
        if np.mean(stress_levels) > 0.7:
            return True  # Explained by Mechanism #5
        
        # Identity threat
        identity_threats = [c.get('identity_threat_level', 0) for c in contexts]
        if np.mean(identity_threats) > 0.6:
            return True  # Explained by Mechanism #26
        
        # If none of the above, potential new mechanism
        return False
    
    def _generate_mechanism_hypothesis(self, failure_cluster: List[Dict]) -> Optional[MechanismHypothesis]:
        """Generate a new mechanism hypothesis from failure pattern."""
        if not failure_cluster:
            return None
        
        # Analyze common features
        contexts = [f.get('context', {}) for f in failure_cluster]
        
        # Detect specific pattern types
        
        # Pattern: Post-video-call cognitive impairment
        video_calls = [c for c in contexts if c.get('had_video_call', False)]
        if len(video_calls) / len(contexts) > 0.7:
            return MechanismHypothesis(
                hypothesis_id=f"mech_{self.user_id}_{self.mechanism_id_counter}",
                user_id=self.user_id,
                name="Post-Call Cognitive Residue",
                description="After video calls, users experience 15-30 min of lingering attention fragmentation and social processing load that impairs deep work initiation, independent of general energy levels.",
                trigger_pattern={
                    "preceding_event": "video_call_end",
                    "time_window_minutes": 30,
                    "required_context": "knowledge_work_context"
                },
                predicted_effect="reduces_deep_work_initiation_by_40_percent",
                status=MechanismHypothesisStatus.PROPOSED,
                supporting_observations=[f.get('observation_id') for f in failure_cluster],
                base_rate_in_user=len(failure_cluster) / 30  # Approximate daily rate
            )
        
        # Pattern: Decision debt accumulation
        high_decisions = [c for c in contexts if c.get('decision_count', 0) > 8]
        if len(high_decisions) / len(contexts) > 0.6:
            return MechanismHypothesis(
                hypothesis_id=f"mech_{self.user_id}_{self.mechanism_id_counter}",
                user_id=self.user_id,
                name="Decision Debt Accumulation",
                description="Each decision made depletes a 'decision budget' that recovers slowly. After 8+ decisions in an hour, subsequent decisions show 2x slower response times and 30% more avoidance, independent of general cognitive energy.",
                trigger_pattern={
                    "decision_count_last_hour": ">8",
                    "recovery_required": "30_min_no_decisions"
                },
                predicted_effect="increases_decision_avoidance_reduces_quality",
                status=MechanismHypothesisStatus.PROPOSED,
                supporting_observations=[f.get('observation_id') for f in failure_cluster],
                base_rate_in_user=len(failure_cluster) / 20
            )
        
        # Pattern: Social context switch cost
        social_transitions = [c for c in contexts if c.get('social_transition', False)]
        if len(social_transitions) / len(contexts) > 0.6:
            return MechanismHypothesis(
                hypothesis_id=f"mech_{self.user_id}_{self.mechanism_id_counter}",
                user_id=self.user_id,
                name="Social Context Transition Cost",
                description="Switching between social modes (alone→meeting, meeting→deep work) incurs a 10-15 min cognitive realignment cost not captured by general task-switching costs.",
                trigger_pattern={
                    "social_mode_change": True,
                    "time_since_transition_minutes": "<15"
                },
                predicted_effect="reduces_cognitive_performance_during_realignment",
                status=MechanismHypothesisStatus.PROPOSED,
                supporting_observations=[f.get('observation_id') for f in failure_cluster],
                base_rate_in_user=len(failure_cluster) / 25
            )
        
        return None
    
    async def test_hypothesis(self, hypothesis: MechanismHypothesis, 
                             new_observations: List[Dict]) -> MechanismHypothesis:
        """
        Validate hypothesis against new observations.
        Updates hypothesis status based on evidence.
        """
        supporting = 0
        contradicting = 0
        
        for obs in new_observations:
            # Check if observation matches trigger pattern
            if self._matches_trigger_pattern(obs, hypothesis.trigger_pattern):
                # Check if predicted effect occurred
                if self._effect_observed(obs, hypothesis.predicted_effect):
                    supporting += 1
                    hypothesis.supporting_observations.append(obs.get('observation_id'))
                else:
                    contradicting += 1
                    hypothesis.contradicting_observations.append(obs.get('observation_id'))
        
        # Update metrics
        total_tests = supporting + contradicting
        if total_tests > 0:
            hypothesis.effect_size_when_active = supporting / total_tests
            
            # Bayesian confidence update
            prior_confidence = hypothesis.confidence
            likelihood = supporting / total_tests
            hypothesis.confidence = (prior_confidence + likelihood * 0.5) / 1.5
        
        # Check validation criteria
        if hypothesis.confidence > 0.7 and len(hypothesis.supporting_observations) >= 5:
            if len(hypothesis.contradicting_observations) / len(hypothesis.supporting_observations) < 0.3:
                hypothesis.status = MechanismHypothesisStatus.VALIDATED
                hypothesis.validated_at = datetime.utcnow()
        elif hypothesis.confidence < 0.3 and total_tests >= 5:
            hypothesis.status = MechanismHypothesisStatus.REJECTED
        
        return hypothesis
    
    def _matches_trigger_pattern(self, observation: Dict, pattern: Dict) -> bool:
        """Check if observation matches trigger conditions."""
        context = observation.get('context', {})
        
        for key, value in pattern.items():
            if key == "preceding_event":
                if context.get('preceding_activity') != value:
                    return False
            elif key == "time_window_minutes":
                # Check recency
                time_since = context.get('minutes_since_event', 999)
                if time_since > value:
                    return False
            elif key == "decision_count_last_hour":
                count = context.get('decision_count', 0)
                threshold = int(value.replace(">", ""))
                if count <= threshold:
                    return False
        
        return True
    
    def _effect_observed(self, observation: Dict, predicted_effect: str) -> bool:
        """Check if predicted behavioral effect was observed."""
        behavior = observation.get('behavior', {})
        
        if "reduces_deep_work" in predicted_effect:
            return behavior.get('deep_work_initiated', True) == False or behavior.get('focus_score', 1.0) < 0.4
        elif "increases_decision_avoidance" in predicted_effect:
            return behavior.get('decision_deferred', False) or behavior.get('avoidance_behavior', False)
        elif "reduces_cognitive_performance" in predicted_effect:
            return behavior.get('error_rate', 0) > 0.3 or behavior.get('task_completion_time', 0) > 1.5
        
        return False
    
    async def integrate_validated_mechanism(self, hypothesis: MechanismHypothesis) -> bool:
        """
        Add validated mechanism to User-Specific Mechanism Library.
        Returns True if integration successful.
        """
        if hypothesis.status != MechanismHypothesisStatus.VALIDATED:
            return False
        
        # Assign permanent mechanism ID
        mechanism_id = self.mechanism_id_counter
        self.mechanism_id_counter += 1
        
        # Create mechanism definition
        mechanism_def = {
            "mechanism_id": mechanism_id,
            "name": hypothesis.name,
            "description": hypothesis.description,
            "trigger_pattern": hypothesis.trigger_pattern,
            "predicted_effect": hypothesis.predicted_effect,
            "base_rate": hypothesis.base_rate_in_user,
            "effect_size": hypothesis.effect_size_when_active,
            "confidence": hypothesis.confidence,
            "discovered_from_hypothesis": hypothesis.hypothesis_id,
            "integrated_at": datetime.utcnow().isoformat()
        }
        
        # Store in database (would be called here)
        # await self._store_mechanism(mechanism_def)
        
        hypothesis.status = MechanismHypothesisStatus.INTEGRATED
        hypothesis.integrated_at = datetime.utcnow()
        
        print(f"✓ New mechanism integrated for user {self.user_id}: {hypothesis.name} (ID: {mechanism_id})")
        
        return True
    
    def get_user_specific_mechanisms(self) -> List[Dict]:
        """Get all validated and integrated mechanisms for this user."""
        return [
            {
                "id": h.hypothesis_id,
                "name": h.name,
                "status": h.status.value,
                "confidence": h.confidence,
                "effect_size": h.effect_size_when_active
            }
            for h in self.active_hypotheses.values()
            if h.status in [MechanismHypothesisStatus.VALIDATED, MechanismHypothesisStatus.INTEGRATED]
        ]


# Global registry
_ontology_agents: Dict[str, OntologyExpansionAgent] = {}


def get_ontology_agent(user_id: str) -> OntologyExpansionAgent:
    """Get or create ontology agent for user."""
    if user_id not in _ontology_agents:
        _ontology_agents[user_id] = OntologyExpansionAgent(user_id)
    return _ontology_agents[user_id]


print("✓ Ontology Expansion Agent created")
print("  - Analyzes disproven hypotheses for missing mechanisms")
print("  - Generates hypotheses for mechanisms 40, 41, 42...")
print("  - Validates with empirical testing")
print("  - Integrates validated mechanisms into user library")
