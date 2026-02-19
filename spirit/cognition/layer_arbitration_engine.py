
# Create the Layer Arbitration Engine file
code = '''"""
Layer Arbitration Engine (LAE): Decides which layer controls behavior.

Core principle: Every repeated behavior answers one question:
- Is the person UNABLE? (HOM - capacity failure)
- Is the person UNWILLING? (HSM - strategic avoidance)  
- Is the person PROTECTING something? (PNM - identity threat)

The LAE's job is to infer which category the evidence best fits.
Winner-take-all selection. No blending. Wrong layer = failed intervention.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum, auto
import asyncio
import math

from spirit.db.supabase_client import get_behavioral_store
from spirit.cognition.human_operating_model import (
    HumanOperatingModel, 
    Subsystem,
    get_human_operating_model
)


class ControlLayer(Enum):
    """The three layers that can dominate behavior."""
    HOM = "hom"  # Human Operating Model - biological/energy constraints
    HSM = "hsm"  # Human Strategy Model - rational avoidance under hidden costs
    PNM = "pnm"  # Personal Narrative Model - identity protection


class LayerConfidence(Enum):
    """Confidence in layer attribution."""
    HIGH = auto()      # > 0.85 - Clear signature, intervene confidently
    MODERATE = auto()  # 0.70-0.85 - Likely correct, but verify with experiment
    LOW = auto()       # 0.50-0.70 - Ambiguous, run diagnostic experiment
    AMBIGUOUS = auto() # < 0.50 - Cannot determine, defer to MAO


@dataclass
class LayerSignature:
    """Behavioral signature indicating a specific layer is active."""
    layer: ControlLayer
    signature_type: str  # e.g., 'variance_pattern', 'substitution_pattern'
    confidence: float  # 0-1
    evidence: Dict[str, Any]
    diagnostic_value: float  # How well this distinguishes from other layers


@dataclass
class LayerArbitrationResult:
    """Result of layer arbitration."""
    primary_layer: ControlLayer
    primary_confidence: float
    confidence_tier: LayerConfidence
    
    # Runner-up for comparison
    runner_up_layer: Optional[ControlLayer]
    runner_up_confidence: float
    confidence_gap: float
    
    # Evidence summary
    hom_score: float
    hsm_score: float
    pnm_score: float
    
    # Decision
    action: str  # 'intervene', 'diagnostic_experiment', 'defer_to_mao'
    reasoning: str
    
    # If diagnostic experiment needed
    diagnostic_experiment: Optional[Dict[str, Any]]
    
    # Timestamp
    arbitrated_at: datetime


@dataclass
class DiagnosticExperiment:
    """Experiment designed to disambiguate between layers."""
    experiment_id: str
    target_layers: Tuple[ControlLayer, ControlLayer]
    
    # Test conditions
    condition_a: Dict[str, Any]  # Tests layer A
    condition_b: Dict[str, Any]  # Tests layer B
    
    # Predictions
    if_hom_predicts: str
    if_hsm_predicts: str
    if_pnm_predicts: str
    
    # Execution
    duration_days: int
    measurements: List[str]
    
    # User-facing
    user_description: str
    burden_level: str  # 'minimal', 'light', 'moderate'


class HOMSignatureDetector:
    """Detects signatures of biological/energy constraints."""
    
    def __init__(self):
        self.hom = get_human_operating_model()
    
    async def detect_signatures(
        self,
        observation: Dict[str, Any],
        user_id: str,
        lookback_days: int = 7
    ) -> List[LayerSignature]:
        """Detect HOM-specific signatures in recent behavior."""
        signatures = []
        
        # Signature 1: Performance varies with physiological state
        variance_sig = await self._check_physiological_variance(user_id, lookback_days)
        if variance_sig:
            signatures.append(variance_sig)
        
        # Signature 2: Rapid reversibility
        reversal_sig = await self._check_rapid_reversibility(observation, user_id)
        if reversal_sig:
            signatures.append(reversal_sig)
        
        # Signature 3: Time-of-day correlation
        circadian_sig = await self._check_circadian_pattern(user_id, lookback_days)
        if circadian_sig:
            signatures.append(circadian_sig)
        
        # Signature 4: User surprise at failure
        surprise_sig = await self._check_user_surprise(user_id)
        if surprise_sig:
            signatures.append(surprise_sig)
        
        return signatures
    
    async def _check_physiological_variance(
        self, 
        user_id: str, 
        lookback_days: int
    ) -> Optional[LayerSignature]:
        """Check if performance correlates with sleep, HRV, etc."""
        store = await get_behavioral_store()
        if not store:
            return None
        
        since = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()
        
        # Get observations with performance metrics
        observations = store.client.table('behavioral_observations').select('*').eq(
            'user_id', user_id
        ).gte('timestamp', since).execute()
        
        if not observations.data or len(observations.data) < 5:
            return None
        
        # Check for performance variance
        performances = []
        for obs in observations.data:
            perf = obs.get('behavior', {}).get('performance_score')
            if perf is not None:
                performances.append(perf)
        
        if len(performances) < 5:
            return None
        
        # High variance suggests HOM
        variance = self._calculate_variance(performances)
        
        if variance > 0.3:  # High variance threshold
            return LayerSignature(
                layer=ControlLayer.HOM,
                signature_type="high_performance_variance",
                confidence=min(0.9, 0.6 + variance),
                evidence={
                    "variance": variance,
                    "n_observations": len(performances),
                    "range": (min(performances), max(performances))
                },
                diagnostic_value=0.8  # Strongly distinguishes from HSM/PNM
            )
        
        return None
    
    async def _check_rapid_reversibility(
        self,
        observation: Dict,
        user_id: str
    ) -> Optional[LayerSignature]:
        """Check if condition improves rapidly when energy/sleep improves."""
        # Check if recent intervention produced rapid improvement
        store = await get_behavioral_store()
        if not store:
            return None
        
        # Look for recent rapid improvements
        since = (datetime.utcnow() - timedelta(days=2)).isoformat()
        
        improvements = store.client.table('behavioral_observations').select('*').eq(
            'user_id', user_id
        ).gte('timestamp', since).execute()
        
        if not improvements.data:
            return None
        
        # Check for rapid swings (suggests energy, not strategy)
        performances = [
            o.get('behavior', {}).get('performance_score', 0.5) 
            for o in improvements.data 
            if o.get('behavior', {}).get('performance_score') is not None
        ]
        
        if len(performances) >= 3:
            # Check for rapid improvement
            if performances[-1] - performances[0] > 0.4:
                return LayerSignature(
                    layer=ControlLayer.HOM,
                    signature_type="rapid_reversibility",
                    confidence=0.75,
                    evidence={
                        "performance_change": performances[-1] - performances[0],
                        "time_hours": 48
                    },
                    diagnostic_value=0.7
                )
        
        return None
    
    async def _check_circadian_pattern(
        self,
        user_id: str,
        lookback_days: int
    ) -> Optional[LayerSignature]:
        """Check for time-of-day performance patterns."""
        store = await get_behavioral_store()
        if not store:
            return None
        
        since = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()
        
        observations = store.client.table('behavioral_observations').select('*').eq(
            'user_id', user_id
        ).gte('timestamp', since).execute()
        
        if not observations.data:
            return None
        
        # Group by hour
        hourly_performance = {}
        for obs in observations.data:
            ts = datetime.fromisoformat(obs['timestamp'].replace('Z', '+00:00'))
            hour = ts.hour
            perf = obs.get('behavior', {}).get('performance_score')
            if perf is not None:
                if hour not in hourly_performance:
                    hourly_performance[hour] = []
                hourly_performance[hour].append(perf)
        
        if len(hourly_performance) < 3:
            return None
        
        # Check for clear time-of-day pattern
        avg_by_hour = {
            h: sum(p)/len(p) for h, p in hourly_performance.items() 
            if len(p) >= 2
        }
        
        if len(avg_by_hour) >= 3:
            values = list(avg_by_hour.values())
            variance = self._calculate_variance(values)
            
            if variance > 0.15:  # Clear time-of-day effect
                peak_hour = max(avg_by_hour, key=avg_by_hour.get)
                trough_hour = min(avg_by_hour, key=avg_by_hour.get)
                
                return LayerSignature(
                    layer=ControlLayer.HOM,
                    signature_type="circadian_performance_pattern",
                    confidence=min(0.85, 0.6 + variance * 2),
                    evidence={
                        "peak_hour": peak_hour,
                        "trough_hour": trough_hour,
                        "performance_range": max(values) - min(values),
                        "variance": variance
                    },
                    diagnostic_value=0.85  # Very distinctive
                )
        
        return None
    
    async def _check_user_surprise(self, user_id: str) -> Optional[LayerSignature]:
        """Check if user expresses surprise at their own failure."""
        store = await get_behavioral_store()
        if not store:
            return None
        
        # Check recent reflections for surprise language
        since = (datetime.utcnow() - timedelta(days=3)).isoformat()
        
        reflections = store.client.table('execution_reflections').select('*').eq(
            'user_id', user_id
        ).gte('created_at', since).execute()
        
        if not reflections.data:
            return None
        
        surprise_indicators = [
            "don't know why", "surprised", "unexpected", "thought I could",
            "usually can", "normally", "strange that", "weird that"
        ]
        
        for ref in reflections.data:
            text = (ref.get('what_worked', '') + ' ' + ref.get('what_blocked', '')).lower()
            if any(ind in text for ind in surprise_indicators):
                return LayerSignature(
                    layer=ControlLayer.HOM,
                    signature_type="user_surprise_at_failure",
                    confidence=0.7,
                    evidence={
                        "reflection_snippet": text[:100],
                        "matched_indicators": [i for i in surprise_indicators if i in text]
                    },
                    diagnostic_value=0.6
                )
        
        return None
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate normalized variance."""
        if len(values) < 2:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)  # Return standard deviation


class HSMSignatureDetector:
    """Detects signatures of strategic/rational avoidance."""
    
    async def detect_signatures(
        self,
        observation: Dict[str, Any],
        user_id: str,
        lookback_days: int = 14
    ) -> List[LayerSignature]:
        """Detect HSM-specific signatures."""
        signatures = []
        
        # Signature 1: Selective application (works in some domains, not this one)
        selective_sig = await self._check_selective_application(user_id, lookback_days)
        if selective_sig:
            signatures.append(selective_sig)
        
        # Signature 2: Productive substitution
        substitution_sig = await self._check_productive_substitution(user_id, lookback_days)
        if substitution_sig:
            signatures.append(substitution_sig)
        
        # Signature 3: Information gathering instead of action
        info_sig = await self._check_information_gathering(user_id)
        if info_sig:
            signatures.append(info_sig)
        
        # Signature 4: Delay near commitment
        delay_sig = await self._check_commitment_delay(user_id)
        if delay_sig:
            signatures.append(delay_sig)
        
        # Signature 5: Post-hoc rationalization
        rationalization_sig = await self._check_post_hoc_rationalization(user_id)
        if rationalization_sig:
            signatures.append(rationalization_sig)
        
        return signatures
    
    async def _check_selective_application(
        self,
        user_id: str,
        lookback_days: int
    ) -> Optional[LayerSignature]:
        """Check if user works hard in some domains but not target domain."""
        store = await get_behavioral_store()
        if not store:
            return None
        
        since = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()
        
        # Get goal progress across domains
        from spirit.db import async_session
        from spirit.models import Execution, Goal
        from sqlalchemy import select
        
        async with async_session() as session:
            goals = await session.execute(
                select(Goal).where(Goal.user_id == int(user_id))
            )
            goal_list = goals.scalars().all()
            
            if len(goal_list) < 2:
                return None
            
            # Check for differential success
            domain_success = {}
            for goal in goal_list:
                executions = await session.execute(
                    select(Execution).where(
                        Execution.goal_id == goal.id
                    ).order_by(Execution.day.desc()).limit(7)
                )
                execs = executions.scalars().all()
                
                if execs:
                    success_rate = sum(1 for e in execs if e.executed) / len(execs)
                    domain = goal.domain if hasattr(goal, 'domain') else 'general'
                    domain_success[domain] = success_rate
            
            if len(domain_success) >= 2:
                rates = list(domain_success.values())
                if max(rates) - min(rates) > 0.5:  # Large differential
                    best_domain = max(domain_success, key=domain_success.get)
                    worst_domain = min(domain_success, key=domain_success.get)
                    
                    return LayerSignature(
                        layer=ControlLayer.HSM,
                        signature_type="selective_domain_success",
                        confidence=0.8,
                        evidence={
                            "best_domain": best_domain,
                            "best_rate": domain_success[best_domain],
                            "worst_domain": worst_domain,
                            "worst_rate": domain_success[worst_domain],
                            "differential": max(rates) - min(rates)
                        },
                        diagnostic_value=0.85  # Very distinctive
                    )
        
        return None
    
    async def _check_productive_substitution(
        self,
        user_id: str,
        lookback_days: int
    ) -> Optional[LayerSignature]:
        """Check if user substitutes other productive tasks."""
        store = await get_behavioral_store()
        if not store:
            return None
        
        since = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()
        
        # Check for "productive procrastination"
        observations = store.client.table('behavioral_observations').select('*').eq(
            'user_id', user_id
        ).gte('timestamp', since).execute()
        
        if not observations.data:
            return None
        
        # Look for pattern: avoiding target task but doing other productive things
        target_avoidance = 0
        other_productivity = 0
        
        for obs in observations.data:
            behavior = obs.get('behavior', {})
            app_cat = behavior.get('app_category')
            task_type = behavior.get('task_type')
            
            # This is simplified - would need to know target task
            if app_cat == 'productivity' and task_type == 'secondary':
                other_productivity += 1
            elif app_cat == 'entertainment':
                target_avoidance += 1
        
        if other_productivity > target_avoidance and other_productivity > 5:
            return LayerSignature(
                layer=ControlLayer.HSM,
                signature_type="productive_substitution",
                confidence=0.75,
                evidence={
                    "secondary_productivity_count": other_productivity,
                    "avoidance_count": target_avoidance,
                    "substitution_ratio": other_productivity / max(1, target_avoidance)
                },
                diagnostic_value=0.8
            )
        
        return None
    
    async def _check_information_gathering(self, user_id: str) -> Optional[LayerSignature]:
        """Check if user gathers information instead of acting."""
        store = await get_behavioral_store()
        if not store:
            return None
        
        since = (datetime.utcnow() - timedelta(days=7)).isoformat()
        
        # Check for research patterns
        observations = store.client.table('behavioral_observations').select('*').eq(
            'user_id', user_id
        ).gte('timestamp', since).execute()
        
        if not observations.data:
            return None
        
        research_time = 0
        action_time = 0
        
        for obs in observations.data:
            behavior = obs.get('behavior', {})
            app_cat = behavior.get('app_category')
            
            if app_cat in ['research', 'reading', 'information']:
                research_time += behavior.get('duration_minutes', 0)
            elif app_cat == 'productivity' and behavior.get('task_type') == 'execution':
                action_time += behavior.get('duration_minutes', 0)
        
        if research_time > action_time * 2 and research_time > 60:
            return LayerSignature(
                layer=ControlLayer.HSM,
                signature_type="excessive_information_gathering",
                confidence=0.7,
                evidence={
                    "research_minutes": research_time,
                    "action_minutes": action_time,
                    "research_to_action_ratio": research_time / max(1, action_time)
                },
                diagnostic_value=0.75
            )
        
        return None
    
    async def _check_commitment_delay(self, user_id: str) -> Optional[LayerSignature]:
        """Check if user delays near irreversible steps."""
        store = await get_behavioral_store()
        if not store:
            return None
        
        # Check for patterns of delay before commitment points
        since = (datetime.utcnow() - timedelta(days=14)).isoformat()
        
        # This would need to track specific commitment points
        # Simplified: check for "almost started" patterns
        
        return None  # Placeholder - requires more sophisticated tracking
    
    async def _check_post_hoc_rationalization(self, user_id: str) -> Optional[LayerSignature]:
        """Check for post-hoc explanations that don't match timing."""
        store = await get_behavioral_store()
        if not store:
            return None
        
        # Check if explanations come after the fact with sophisticated reasoning
        since = (datetime.utcnow() - timedelta(days=7)).isoformat()
        
        reflections = store.client.table('execution_reflections').select('*').eq(
            'user_id', user_id
        ).gte('created_at', since).execute()
        
        if not reflections.data:
            return None
        
        # Look for sophisticated explanations that suggest post-hoc reasoning
        sophisticated_terms = [
            "realized that", "understood that", "came to see",
            "recognized", "acknowledged", "admitted to myself"
        ]
        
        for ref in reflections.data:
            text = (ref.get('what_blocked', '') + ' ' + ref.get('what_worked', '')).lower()
            if any(term in text for term in sophisticated_terms):
                return LayerSignature(
                    layer=ControlLayer.HSM,
                    signature_type="post_hoc_rationalization",
                    confidence=0.65,
                    evidence={
                        "reflection_snippet": text[:150],
                        "sophistication_markers": [t for t in sophisticated_terms if t in text]
                    },
                    diagnostic_value=0.6
                )
        
        return None


class PNMSignatureDetector:
    """Detects signatures of identity/narrative protection."""
    
    async def detect_signatures(
        self,
        observation: Dict[str, Any],
        user_id: str,
        lookback_days: int = 30
    ) -> List[LayerSignature]:
        """Detect PNM-specific signatures."""
        signatures = []
        
        # Signature 1: Persistence despite incentives
        persistence_sig = await self._check_incentive_resistance(user_id, lookback_days)
        if persistence_sig:
            signatures.append(persistence_sig)
        
        # Signature 2: Rejection of help
        help_rejection_sig = await self._check_help_rejection(user_id)
        if help_rejection_sig:
            signatures.append(help_rejection_sig)
        
        # Signature 3: Success creates discomfort
        success_discomfort_sig = await self._check_success_discomfort(user_id)
        if success_discomfort_sig:
            signatures.append(success_discomfort_sig)
        
        # Signature 4: Emotional defense
        emotional_defense_sig = await self._check_emotional_defense(user_id)
        if emotional_defense_sig:
            signatures.append(emotional_defense_sig)
        
        # Signature 5: Stability across contexts
        stability_sig = await self._check_cross_context_stability(user_id, lookback_days)
        if stability_sig:
            signatures.append(stability_sig)
        
        return signatures
    
    async def _check_incentive_resistance(
        self,
        user_id: str,
        lookback_days: int
    ) -> Optional[LayerSignature]:
        """Check if behavior persists despite strong incentives to change."""
        store = await get_behavioral_store()
        if not store:
            return None
        
        # Check for failed interventions with high expected value
        since = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()
        
        interventions = store.client.table('intervention_outcomes').select('*').eq(
            'user_id', user_id
        ).gte('timestamp', since).execute()
        
        if not interventions.data:
            return None
        
        # Count high-confidence interventions that failed
        high_conf_failures = 0
        for intv in interventions.data:
            if (intv.get('expected_success_rate', 0) > 0.7 and 
                intv.get('actual_success', False) == False):
                high_conf_failures += 1
        
        if high_conf_failures >= 3:
            return LayerSignature(
                layer=ControlLayer.PNM,
                signature_type="incentive_resistant_behavior",
                confidence=0.75,
                evidence={
                    "high_confidence_interventions_attempted": len(interventions.data),
                    "high_confidence_failures": high_conf_failures,
                    "failure_rate": high_conf_failures / len(interventions.data)
                },
                diagnostic_value=0.8
            )
        
        return None
    
    async def _check_help_rejection(self, user_id: str) -> Optional[LayerSignature]:
        """Check if user rejects help or support."""
        store = await get_behavioral_store()
        if not store:
            return None
        
        # Check for rejection of offered support
        since = (datetime.utcnow() - timedelta(days=14)).isoformat()
        
        support_offers = store.client.table('support_interactions').select('*').eq(
            'user_id', user_id
        ).gte('offered_at', since).execute()
        
        if not support_offers.data:
            return None
        
        rejections = sum(1 for s in support_offers.data if s.get('user_response') == 'declined')
        
        if rejections >= 2 and rejections / len(support_offers.data) > 0.5:
            return LayerSignature(
                layer=ControlLayer.PNM,
                signature_type="help_rejection",
                confidence=0.7,
                evidence={
                    "support_offers": len(support_offers.data),
                    "rejections": rejections,
                    "rejection_rate": rejections / len(support_offers.data)
                },
                diagnostic_value=0.7
            )
        
        return None
    
    async def _check_success_discomfort(self, user_id: str) -> Optional[LayerSignature]:
        """Check if success creates discomfort or self-sabotage."""
        store = await get_behavioral_store()
        if not store:
            return None
        
        # Check for patterns of success followed by regression
        since = (datetime.utcnow() - timedelta(days=30)).isoformat()
        
        executions = store.client.table('executions').select('*').eq(
            'user_id', user_id
        ).gte('day', since).order('day').execute()
        
        if not executions.data or len(executions.data) < 5:
            return None
        
        # Look for success followed by failure pattern
        success_failures = 0
        for i in range(len(executions.data) - 1):
            if executions.data[i].get('executed') and not executions.data[i+1].get('executed'):
                # Check if there was a reason
                if not executions.data[i+1].get('blocker_reason'):
                    success_failures += 1
        
        if success_failures >= 2:
            return LayerSignature(
                layer=ControlLayer.PNM,
                signature_type="success_discomfort",
                confidence=0.65,
                evidence={
                    "success_followed_by_failure_count": success_failures,
                    "pattern_rate": success_failures / len(executions.data)
                },
                diagnostic_value=0.75
            )
        
        return None
    
    async def _check_emotional_defense(self, user_id: str) -> Optional[LayerSignature]:
        """Check for emotional rather than practical explanations."""
        store = await get_behavioral_store()
        if not store:
            return None
        
        since = (datetime.utcnow() - timedelta(days=14)).isoformat()
        
        reflections = store.client.table('execution_reflections').select('*').eq(
            'user_id', user_id
        ).gte('created_at', since).execute()
        
        if not reflections.data:
            return None
        
        # Look for emotional language
        emotional_terms = [
            "felt like", "didn't feel", "wasn't feeling", "emotional",
            "overwhelmed", "anxious", "scared", "afraid", "terrified"
        ]
        
        emotional_count = 0
        for ref in reflections.data:
            text = (ref.get('what_blocked', '') + ' ' + ref.get('notes', '')).lower()
            if any(term in text for term in emotional_terms):
                emotional_count += 1
        
        if emotional_count >= 3:
            return LayerSignature(
                layer=ControlLayer.PNM,
                signature_type="emotional_explanation_pattern",
                confidence=0.7,
                evidence={
                    "emotional_explanations": emotional_count,
                    "total_reflections": len(reflections.data),
                    "emotional_language_examples": emotional_terms[:3]
                },
                diagnostic_value=0.65
            )
        
        return None
    
    async def _check_cross_context_stability(
        self,
        user_id: str,
        lookback_days: int
    ) -> Optional[LayerSignature]:
        """Check if behavior is stable across different contexts."""
        store = await get_behavioral_store()
        if not store:
            return None
        
        since = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()
        
        observations = store.client.table('behavioral_observations').select('*').eq(
            'user_id', user_id
        ).gte('timestamp', since).execute()
        
        if not observations.data:
            return None
        
        # Check behavior across contexts
        contexts = {}
        for obs in observations.data:
            ctx = obs.get('context', {})
            location = ctx.get('location_type', 'unknown')
            time_of_day = ctx.get('time_of_day', 'unknown')
            
            key = f"{location}_{time_of_day}"
            if key not in contexts:
                contexts[key] = []
            
            behavior = obs.get('behavior', {})
            contexts[key].append(behavior.get('target_behavior_present', False))
        
        # Check if behavior present in most contexts
        if len(contexts) >= 3:
            presence_rates = [
                sum(ctx) / len(ctx) for ctx in contexts.values() if len(ctx) >= 2
            ]
            
            if presence_rates:
                variance = max(presence_rates) - min(presence_rates)
                
                if variance < 0.3:  # Stable across contexts
                    return LayerSignature(
                        layer=ControlLayer.PNM,
                        signature_type="cross_context_stability",
                        confidence=0.75,
                        evidence={
                            "n_contexts": len(contexts),
                            "presence_rates": presence_rates,
                            "variance_across_contexts": variance
                        },
                        diagnostic_value=0.8
                    )
        
        return None


class DiagnosticExperimentDesigner:
    """Designs experiments to disambiguate between layers."""
    
    def __init__(self):
        self.experiment_templates = {
            (ControlLayer.HOM, ControlLayer.HSM): self._design_hom_hsm_experiment,
            (ControlLayer.HOM, ControlLayer.PNM): self._design_hom_pnm_experiment,
            (ControlLayer.HSM, ControlLayer.PNM): self._design_hsm_pnm_experiment,
        }
    
    async def design_experiment(
        self,
        competing_layers: Tuple[ControlLayer, ControlLayer],
        observation: Dict,
        user_id: str
    ) -> Optional[DiagnosticExperiment]:
        """Design experiment to disambiguate between two layers."""
        # Sort to ensure consistent key
        key = tuple(sorted(competing_layers, key=lambda x: x.value))
        
        designer = self.experiment_templates.get(key)
        if designer:
            return await designer(observation, user_id)
        
        return None
    
    async def _design_hom_hsm_experiment(
        self,
        observation: Dict,
        user_id: str
    ) -> DiagnosticExperiment:
        """
        Disambiguate: Is it energy depletion (HOM) or risk avoidance (HSM)?
        
        Test: Move task to peak energy time but increase stakes
        - HOM predicts: Performance improves (energy was constraint)
        - HSM predicts: Performance unchanged or worse (risk now dominates)
        """
        return DiagnosticExperiment(
            experiment_id=f"diag_hom_hsm_{user_id}_{datetime.utcnow().timestamp()}",
            target_layers=(ControlLayer.HOM, ControlLayer.HSM),
            condition_a={
                "name": "peak_energy_high_stakes",
                "description": "Do task at your best energy time, but announce publicly",
                "modifications": ["schedule_at_peak_energy", "add_social_accountability"]
            },
            condition_b={
                "name": "off_peak_low_stakes",
                "description": "Do task at lower energy time, keep private",
                "modifications": ["schedule_at_off_peak", "keep_private"]
            },
            if_hom_predicts="Performance best in Condition A (peak energy)",
            if_hsm_predicts="Performance similar or better in Condition B (lower stakes)",
            if_pnm_predicts="N/A - not testing PNM",
            duration_days=4,
            measurements=["task_completion", "quality_rating", "subjective_difficulty", "stress_level"],
            user_description=(
                "I'd like to understand if your challenge is energy timing or something else. "
                "We'll try the same task at two different times with different stakes."
            ),
            burden_level="light"
        )
    
    async def _design_hom_pnm_experiment(
        self,
        observation: Dict,
        user_id: str
    ) -> DiagnosticExperiment:
        """
        Disambiguate: Is it fatigue (HOM) or identity threat (PNM)?
        
        Test: Reduce all friction but frame as threat to identity
        - HOM predicts: Performance improves (friction removed)
        - PNM predicts: Performance worsens (identity threatened)
        """
        return DiagnosticExperiment(
            experiment_id=f"diag_hom_pnm_{user_id}_{datetime.utcnow().timestamp()}",
            target_layers=(ControlLayer.HOM, ControlLayer.PNM),
            condition_a={
                "name": "low_friction_identity_safe",
                "description": "Ultra-easy version framed as 'just experimenting'",
                "modifications": ["reduce_scope_90", "frame_as_experiment", "no_identity_implications"]
            },
            condition_b={
                "name": "low_friction_identity_relevant",
                "description": "Same easy version framed as 'this is who you are becoming'",
                "modifications": ["reduce_scope_90", "frame_as_identity_becoming"]
            },
            if_hom_predicts="Both conditions improve (friction was issue)",
            if_hsm_predicts="N/A - not testing HSM",
            if_pnm_predicts="Condition A >> Condition B (identity framing blocks)",
            duration_days=4,
            measurements=["initiation_latency", "completion_rate", "self_reported_comfort", "identity_salience"],
            user_description=(
                "I'd like to test whether the challenge is energy or how the task connects to your sense of self. "
                "We'll try the same easy version with two different framings."
            ),
            burden_level="minimal"
        )
    
    async def _design_hsm_pnm_experiment(
        self,
        observation: Dict,
        user_id: str
    ) -> DiagnosticExperiment:
        """
        Disambiguate: Is it strategic avoidance (HSM) or identity protection (PNM)?
        
        Test: Offer high reward but frame as betrayal of past self
        - HSM predicts: Performance improves (incentive dominates)
        - PNM predicts: Performance worsens (narrative coherence dominates)
        """
        return DiagnosticExperiment(
            experiment_id=f"diag_hsm_pnm_{user_id}_{datetime.utcnow().timestamp()}",
            target_layers=(ControlLayer.HSM, ControlLayer.PNM),
            condition_a={
                "name": "high_reward_continuity",
                "description": "Big reward framed as 'building on your past'",
                "modifications": ["add_external_reward", "frame_as_continuity"]
            },
            condition_b={
                "name": "high_reward_discontinuity",
                "description": "Same reward framed as 'leaving old self behind'",
                "modifications": ["add_external_reward", "frame_as_transformation"]
            },
            if_hom_predicts="N/A - not testing HOM",
            if_hsm_predicts="Both conditions improve (reward dominates)",
            if_pnm_predicts="Condition A >> Condition B (continuity matters)",
            duration_days=6,
            measurements=["effort_expended", "sustained_attention", "self_reported_authenticity", "reward_salience"],
            user_description=(
                "I'd like to understand if you're avoiding this due to risk or identity concerns. "
                "We'll test the same reward with two different stories about what it means."
            ),
            burden_level="moderate"
        )


class LayerArbitrationEngine:
    """
    Main orchestrator: Detects signatures, scores layers, decides action.
    Inserts between RFE and intervention design.
    """
    
    def __init__(self):
        self.hom_detector = HOMSignatureDetector()
        self.hsm_detector = HSMSignatureDetector()
        self.pnm_detector = PNMSignatureDetector()
        self.experiment_designer = DiagnosticExperimentDesigner()
        
        # Thresholds
        self.min_confidence_for_intervention = 0.70
        self.min_confidence_gap = 0.15
        self.max_ambiguity_threshold = 0.50
    
    async def arbitrate(
        self,
        observation: Dict[str, Any],
        user_id: str,
        context: Optional[Dict] = None
    ) -> LayerArbitrationResult:
        """
        Main entry point: Determine which layer controls current behavior.
        
        Returns:
            - primary_layer: The winning layer
            - action: 'intervene', 'diagnostic_experiment', or 'defer_to_mao'
            - diagnostic_experiment: If action is 'diagnostic_experiment'
        """
        context = context or {}
        
        # STEP 1: Detect signatures for all three layers
        hom_signatures = await self.hom_detector.detect_signatures(
            observation, user_id
        )
        hsm_signatures = await self.hsm_detector.detect_signatures(
            observation, user_id
        )
        pnm_signatures = await self.pnm_detector.detect_signatures(
            observation, user_id
        )
        
        # STEP 2: Calculate layer scores
        hom_score = self._calculate_layer_score(hom_signatures)
        hsm_score = self._calculate_layer_score(hsm_signatures)
        pnm_score = self._calculate_layer_score(pnm_signatures)
        
        # STEP 3: Determine winner and confidence tier
        scores = {
            ControlLayer.HOM: hom_score,
            ControlLayer.HSM: hsm_score,
            ControlLayer.PNM: pnm_score
        }
        
        primary_layer = max(scores, key=scores.get)
        primary_confidence = scores[primary_layer]
        
        # Find runner-up
        remaining = {k: v for k, v in scores.items() if k != primary_layer}
        runner_up_layer = max(remaining, key=remaining.get) if remaining else None
        runner_up_confidence = remaining[runner_up_layer] if runner_up_layer else 0
        confidence_gap = primary_confidence - runner_up_confidence
        
        # Determine confidence tier
        confidence_tier = self._determine_confidence_tier(
            primary_confidence, confidence_gap
        )
        
        # STEP 4: Determine action
        action, reasoning, diagnostic_experiment = await self._determine_action(
            primary_layer, primary_confidence, runner_up_layer, runner_up_confidence,
            confidence_gap, confidence_tier, observation, user_id
        )
        
        return LayerArbitrationResult(
            primary_layer=primary_layer,
            primary_confidence=primary_confidence,
            confidence_tier=confidence_tier,
            runner_up_layer=runner_up_layer,
            runner_up_confidence=runner_up_confidence,
            confidence_gap=confidence_gap,
            hom_score=hom_score,
            hsm_score=hsm_score,
            pnm_score=pnm_score,
            action=action,
            reasoning=reasoning,
            diagnostic_experiment=diagnostic_experiment,
            arbitrated_at=datetime.utcnow()
        )
    
    def _calculate_layer_score(self, signatures: List[LayerSignature]) -> float:
        """Calculate composite score from detected signatures."""
        if not signatures:
            return 0.3  # Base rate
        
        # Weight by diagnostic value and confidence
        weighted_scores = [
            sig.confidence * sig.diagnostic_value 
            for sig in signatures
        ]
        
        # Take top 2 signatures
        weighted_scores.sort(reverse=True)
        top_scores = weighted_scores[:2]
        
        # Combine scores (boost for multiple independent signatures)
        if len(top_scores) == 2:
            # Two strong signatures = very confident
            return min(0.95, top_scores[0] * 0.7 + top_scores[1] * 0.4)
        else:
            return min(0.85, top_scores[0])
    
    def _determine_confidence_tier(
        self,
        primary_confidence: float,
        confidence_gap: float
    ) -> LayerConfidence:
        """Determine confidence tier based on score and separation."""
        if primary_confidence < self.max_ambiguity_threshold:
            return LayerConfidence.AMBIGUOUS
        elif primary_confidence < self.min_confidence_for_intervention:
            return LayerConfidence.LOW
        elif confidence_gap < self.min_confidence_gap:
            return LayerConfidence.MODERATE
        else:
            return LayerConfidence.HIGH
    
    async def _determine_action(
        self,
        primary_layer: ControlLayer,
        primary_confidence: float,
        runner_up_layer: Optional[ControlLayer],
        runner_up_confidence: float,
        confidence_gap: float,
        confidence_tier: LayerConfidence,
        observation: Dict,
        user_id: str
    ) -> Tuple[str, str, Optional[Dict]]:
        """Determine what action to take based on arbitration results."""
        
        if confidence_tier == LayerConfidence.HIGH:
            return (
                "intervene",
                f"High confidence ({primary_confidence:.2f}) in {primary_layer.value.upper()} "
                f"with clear separation from runner-up ({confidence_gap:.2f})",
                None
            )
        
        elif confidence_tier == LayerConfidence.MODERATE:
            # Good confidence but close runner-up - run diagnostic experiment
            if runner_up_layer:
                experiment = await self.experiment_designer.design_experiment(
                    (primary_layer, runner_up_layer),
                    observation,
                    user_id
                )
                
                if experiment:
                    return (
                        "diagnostic_experiment",
                        f"Moderate confidence in {primary_layer.value.upper()} but {runner_up_layer.value.upper()} "
                        f"is close runner-up. Running diagnostic experiment.",
                        {
                            "experiment_id": experiment.experiment_id,
                            "target_layers": [l.value for l in experiment.target_layers],
                            "user_description": experiment.user_description,
                            "burden_level": experiment.burden_level,
                            "duration_days": experiment.duration_days
                        }
                    )
            
            # Fallback to intervention with warning
            return (
                "intervene",
                f"Moderate confidence in {primary_layer.value.upper()}. Proceeding with caution.",
                None
            )
        
        elif confidence_tier == LayerConfidence.LOW:
            # Low confidence - need more data or defer
            return (
                "defer_to_mao",
                f"Low confidence ({primary_confidence:.2f}). Insufficient evidence to determine layer.",
                None
            )
        
        else:  # AMBIGUOUS
            return (
                "defer_to_mao",
                f"Ambiguous - all layers below threshold. Cannot determine primary driver.",
                None
            )
    
    def get_intervention_match(
        self,
        arbitration_result: LayerArbitrationResult,
        available_interventions: List[Dict]
    ) -> Optional[Dict]:
        """
        Select intervention matched to winning layer.
        
        HOM interventions: Reduce load, optimize timing, remove friction
        HSM interventions: Change incentives, reduce risk, alter payoff
        PNM interventions: Narrative reframing, identity bridge, meaning
        """
        layer = arbitration_result.primary_layer
        
        # Filter interventions by layer match
        layer_matched = [
            i for i in available_interventions 
            if i.get('target_layer') == layer.value
        ]
        
        if not layer_matched:
            # Fall back to generic interventions
            return None
        
        # Select highest confidence intervention
        best = max(layer_matched, key=lambda x: x.get('confidence', 0))
        
        return {
            **best,
            "layer_arbitration": {
                "primary_layer": layer.value,
                "confidence": arbitration_result.primary_confidence,
                "confidence_tier": arbitration_result.confidence_tier.name
            }
        }


# Global singleton
_lae_instance: Optional[LayerArbitrationEngine] = None

def get_layer_arbitration_engine() -> LayerArbitrationEngine:
    """Get or create global LAE instance."""
    global _lae_instance
    if _lae_instance is None:
        _lae_instance = LayerArbitrationEngine()
    return _lae_instance
'''

with open('/mnt/kimi/output/layer_arbitration_engine.py', 'w') as f:
    f.write(code)

print("Created: layer_arbitration_engine.py")
print(f"Size: {len(code)} bytes")
