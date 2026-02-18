"""
Reality Filter Engine (RFE): The causal inference gateway.
Inserts between sensing and hypothesis formation.

Four components:
1. Confound Detection - Hidden variable audit before hypothesis formation
2. Evidence Scoring - confidence = repeatability × cross-context stability × intervention response
3. Experiment Scheduling - Engineer disambiguation moments, don't wait for data
4. Memory Admission Control - 90-98% of data discarded, store only high-value signals
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass
from enum import Enum, auto
import asyncio
import hashlib

from spirit.db.supabase_client import get_behavioral_store
from spirit.evidence.personal_evidence_ladder import (
    EvidenceGradingEngine, 
    EvidenceLevel, 
    EvidenceGrading
)
from spirit.memory.collective_intelligence import CollectiveIntelligenceEngine


class ConfoundType(Enum):
    """Types of hidden variables that threaten causal inference."""
    SLEEP_DEBT = "sleep_debt"
    SOCIAL_LOAD = "social_load"
    NOVELTY = "novelty"
    TIME_PRESSURE = "time_pressure"
    EMOTIONAL_AROUSAL = "emotional_arousal"
    CIRCADIAN_PHASE = "circadian_phase"
    WEATHER = "weather"
    MENSTRUAL_CYCLE = "menstrual_cycle"
    CAFFEINE_STATUS = "caffeine_status"
    EXERCISE_DEBT = "exercise_debt"


@dataclass
class ConfoundAssessment:
    """Assessment of a potential confounding variable."""
    confound_type: ConfoundType
    present: bool
    severity: float  # 0-1
    evidence_strength: float  # 0-1
    inferred_from: List[str]  # What signals indicated this confound
    data_source: str  # 'explicit_report', 'behavioral_proxy', 'inferred_pattern'


@dataclass
class ExperimentDesign:
    """A designed experiment to disambiguate causality."""
    experiment_id: str
    user_id: str
    
    # What we're testing
    target_hypothesis: str
    ambiguity_to_resolve: str
    
    # Design
    conditions: List[Dict]  # A/B or A/B/C conditions
    n_trials_per_condition: int
    duration_days: int
    
    # Scheduling
    proposed_schedule: List[datetime]
    randomization_method: str  # 'simple', 'block', 'adaptive'
    
    # Constraints
    user_constraints: Dict[str, Any]  # "Can't do mornings", etc.
    ethical_checks: List[str]
    
    # Status
    status: str  # 'designed', 'scheduled', 'running', 'completed', 'abandoned'
    created_at: datetime


class ConfoundDetector:
    """
    Detects hidden variables that threaten causal inference.
    Hypothesis cannot form without confound audit.
    """
    
    def __init__(self):
        self.min_confidence_for_detection = 0.6
        self.behavioral_proxy_weights = {
            "sleep_debt": {
                "late_night_usage": 0.3,
                "morning_delay": 0.4,
                "caffeine_app_usage": 0.2,
                "typing_speed_variance": 0.3
            },
            "social_load": {
                "messaging_frequency": 0.4,
                "calendar_density": 0.5,
                "social_app_context_switches": 0.3,
                "response_latency_variance": 0.2
            },
            "emotional_arousal": {
                "typing_dynamics_variance": 0.3,
                "app_switching_chaos": 0.4,
                "session_abandonment_rate": 0.3,
                "notification_reactivity": 0.2
            }
        }
    
    async def audit_observation(
        self,
        observation: Dict[str, Any],
        user_id: str,
        lookback_days: int = 7
    ) -> List[ConfoundAssessment]:
        """
        Full confound audit before allowing hypothesis formation.
        Returns list of detected confounds with severity.
        """
        confounds = []
        
        # Check each confound type
        for confound_type in ConfoundType:
            assessment = await self._assess_confound(
                confound_type, observation, user_id, lookback_days
            )
            if assessment.present and assessment.evidence_strength >= self.min_confidence_for_detection:
                confounds.append(assessment)
        
        # Sort by severity
        confounds.sort(key=lambda x: x.severity, reverse=True)
        
        return confounds
    
    async def _assess_confound(
        self,
        confound_type: ConfoundType,
        observation: Dict[str, Any],
        user_id: str,
        lookback_days: int
    ) -> ConfoundAssessment:
        """Assess specific confound type."""
        store = await get_behavioral_store()
        
        # Try explicit data first
        explicit = await self._check_explicit_confound_data(confound_type, user_id, lookback_days)
        if explicit:
            return explicit
        
        # Fall back to behavioral proxies
        return await self._infer_from_behavioral_proxies(
            confound_type, observation, user_id, lookback_days
        )
    
    async def _check_explicit_confound_data(
        self,
        confound_type: ConfoundType,
        user_id: str,
        lookback_days: int
    ) -> Optional[ConfoundAssessment]:
        """Check if user explicitly reported this confound."""
        store = await get_behavioral_store()
        if not store:
            return None
        
        # Query EMA responses for explicit reports
        since = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()
        
        if confound_type == ConfoundType.SLEEP_DEBT:
            # Check for sleep quality reports
            reports = store.client.table('ema_responses').select('*').eq(
                'user_id', user_id
            ).eq('question_type', 'sleep_quality').gte('responded_at', since).execute()
            
            if reports.data:
                avg_quality = sum(r.get('response_value', 5) for r in reports.data) / len(reports.data)
                if avg_quality < 4:  # Poor sleep
                    return ConfoundAssessment(
                        confound_type=confound_type,
                        present=True,
                        severity=(5 - avg_quality) / 5,
                        evidence_strength=0.8,
                        inferred_from=["explicit_sleep_report"],
                        data_source="explicit_report"
                    )
        
        elif confound_type == ConfoundType.SOCIAL_LOAD:
            # Check for social load EMAs
            reports = store.client.table('ema_responses').select('*').eq(
                'user_id', user_id
            ).eq('question_type', 'social_drain').gte('responded_at', since).execute()
            
            if reports.data:
                avg_drain = sum(r.get('response_value', 1) for r in reports.data) / len(reports.data)
                if avg_drain > 3:  # High social load
                    return ConfoundAssessment(
                        confound_type=confound_type,
                        present=True,
                        severity=avg_drain / 5,
                        evidence_strength=0.75,
                        inferred_from=["explicit_social_report"],
                        data_source="explicit_report"
                    )
        
        return None
    
    async def _infer_from_behavioral_proxies(
        self,
        confound_type: ConfoundType,
        observation: Dict[str, Any],
        user_id: str,
        lookback_days: int
    ) -> ConfoundAssessment:
        """Infer confound from behavioral patterns."""
        store = await get_behavioral_store()
        
        # Get recent observations for pattern analysis
        since = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()
        recent = []
        
        if store:
            result = store.client.table('behavioral_observations').select('*').eq(
                'user_id', user_id
            ).gte('timestamp', since).execute()
            recent = result.data if result.data else []
        
        # Calculate proxy scores
        proxies = self.behavioral_proxy_weights.get(confound_type.value, {})
        scores = []
        indicators = []
        
        for proxy, weight in proxies.items():
            score = self._calculate_proxy_score(proxy, recent, observation)
            if score > 0.5:
                scores.append(score * weight)
                indicators.append(proxy)
        
        if not scores:
            return ConfoundAssessment(
                confound_type=confound_type,
                present=False,
                severity=0,
                evidence_strength=0,
                inferred_from=[],
                data_source="behavioral_proxy"
            )
        
        avg_score = sum(scores) / len(scores)
        
        return ConfoundAssessment(
            confound_type=confound_type,
            present=avg_score > 0.6,
            severity=min(1.0, avg_score),
            evidence_strength=min(0.7, avg_score),  # Cap at 0.7 for inferred
            inferred_from=indicators,
            data_source="behavioral_proxy"
        )
    
    def _calculate_proxy_score(
        self,
        proxy: str,
        recent_observations: List[Dict],
        current_observation: Dict
    ) -> float:
        """Calculate score for specific behavioral proxy."""
        
        if proxy == "late_night_usage":
            # Check for usage after 11 PM
            late_night = [
                o for o in recent_observations 
                if datetime.fromisoformat(o['timestamp'].replace('Z', '+00:00')).hour >= 23
            ]
            return min(1.0, len(late_night) / 3) if late_night else 0
        
        elif proxy == "morning_delay":
            # Check for delayed first productive session
            if not recent_observations:
                return 0
            
            # Find first productivity session each day
            daily_starts = {}
            for obs in recent_observations:
                ts = datetime.fromisoformat(obs['timestamp'].replace('Z', '+00:00'))
                day = ts.date()
                behavior = obs.get('behavior', {})
                
                if behavior.get('app_category') == 'productivity':
                    if day not in daily_starts or ts.hour < daily_starts[day]:
                        daily_starts[day] = ts.hour
            
            if len(daily_starts) >= 2:
                avg_start = sum(daily_starts.values()) / len(daily_starts)
                # Delay if starting after 10 AM
                return max(0, (avg_start - 8) / 4) if avg_start > 8 else 0
            
            return 0
        
        elif proxy == "app_switching_chaos":
            # High context switching indicates emotional arousal
            if not recent_observations:
                return 0
            
            recent_switches = [
                o.get('behavior', {}).get('app_switches_5min', 0) 
                for o in recent_observations[-10:]
            ]
            if recent_switches:
                avg_switches = sum(recent_switches) / len(recent_switches)
                return min(1.0, avg_switches / 10)
            
            return 0
        
        # Default for unimplemented proxies
        return 0
    
    def get_confound_summary(self, confounds: List[ConfoundAssessment]) -> str:
        """Generate human-readable summary of detected confounds."""
        if not confounds:
            return "No significant confounds detected."
        
        parts = []
        for c in confounds[:3]:  # Top 3
            source_emoji = "📝" if c.data_source == "explicit_report" else "🔍"
            parts.append(
                f"{source_emoji} {c.confound_type.value.replace('_', ' ').title()}: "
                f"{c.severity:.0%} severity ({c.evidence_strength:.0%} confidence)"
            )
        
        return " | ".join(parts)


class EvidenceScorer:
    """
    Scores evidence using: 
    confidence = repeatability × cross-context stability × intervention response
    """
    
    def __init__(self):
        self.min_repeatability = 0.5
        self.min_contexts = 2
    
    def calculate_evidence_score(
        self,
        observation: Dict,
        historical_matches: List[Dict],
        intervention_responses: List[Dict],
        contexts_seen: Set[str]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate composite evidence score with component breakdown.
        """
        # Component 1: Repeatability
        repeatability = self._calculate_repeatability(observation, historical_matches)
        
        # Component 2: Cross-context stability
        context_stability = self._calculate_context_stability(
            observation, historical_matches, contexts_seen
        )
        
        # Component 3: Intervention response (if applicable)
        intervention_score = self._calculate_intervention_response(
            observation, intervention_responses
        )
        
        # Composite: Multiplicative (all must be decent for high score)
        # Use geometric mean to penalize weak components
        components = [repeatability, context_stability, max(0.5, intervention_score)]
        composite = (components[0] * components[1] * components[2]) ** (1/3)
        
        breakdown = {
            "repeatability": repeatability,
            "cross_context_stability": context_stability,
            "intervention_response": intervention_score,
            "composite": composite,
            "n_historical_matches": len(historical_matches),
            "n_contexts": len(contexts_seen),
            "n_interventions": len(intervention_responses)
        }
        
        return composite, breakdown
    
    def _calculate_repeatability(
        self,
        observation: Dict,
        historical_matches: List[Dict]
    ) -> float:
        """How often has this pattern occurred?"""
        if not historical_matches:
            return 0.3  # Novel observation, low repeatability
        
        # More matches = higher repeatability, with diminishing returns
        n_matches = len(historical_matches)
        return min(0.95, 0.4 + (n_matches * 0.1))
    
    def _calculate_context_stability(
        self,
        observation: Dict,
        historical_matches: List[Dict],
        contexts_seen: Set[str]
    ) -> float:
        """Does pattern hold across different contexts?"""
        if len(contexts_seen) < self.min_contexts:
            return 0.4  # Insufficient context diversity
        
        # Score based on context diversity
        # More diverse contexts = higher stability
        context_diversity = min(1.0, len(contexts_seen) / 5)  # Cap at 5 contexts
        
        # Check if pattern holds in each context
        context_consistency = 0.7  # Base assumption
        
        return (context_diversity * 0.4) + (context_consistency * 0.6)
    
    def _calculate_intervention_response(
        self,
        observation: Dict,
        intervention_responses: List[Dict]
    ) -> float:
        """Has this pattern responded to interventions?"""
        if not intervention_responses:
            return 0.5  # Neutral - no intervention data
        
        # Calculate average response magnitude
        responses = [
            r.get('response_magnitude', 0) 
            for r in intervention_responses 
            if r.get('causal_confidence', 0) > 0.5
        ]
        
        if not responses:
            return 0.5
        
        avg_response = sum(responses) / len(responses)
        # Normalize to 0-1 scale (responses can be negative)
        return 0.5 + (avg_response * 0.5)


class ExperimentScheduler:
    """
    Engineers disambiguation moments.
    Spirit becomes an experimental designer, not just a passive observer.
    """
    
    def __init__(self):
        self.min_hours_between_experiments = 4
        self.max_concurrent_experiments = 2
        self.collective_engine = CollectiveIntelligenceEngine()
    
    async def design_experiment(
        self,
        user_id: str,
        ambiguity: str,
        current_hypothesis: Optional[str] = None,
        user_constraints: Optional[Dict] = None
    ) -> Optional[ExperimentDesign]:
        """
        Design an experiment to resolve specific ambiguity.
        """
        # Check if we can run experiments now
        if not await self._can_schedule_experiment(user_id):
            return None
        
        # Get user archetype for design constraints
        archetype = await self.collective_engine.get_user_archetype(user_id)
        
        # Design based on ambiguity type
        if "energy_vs_time" in ambiguity:
            return await self._design_energy_timing_experiment(
                user_id, current_hypothesis, user_constraints, archetype
            )
        elif "avoidance_vs_capacity" in ambiguity:
            return await self._design_avoidance_experiment(
                user_id, current_hypothesis, user_constraints, archetype
            )
        elif "habit_vs_intention" in ambiguity:
            return await self._design_habit_intention_experiment(
                user_id, current_hypothesis, user_constraints, archetype
            )
        
        # Default: simple A/B test
        return await self._design_default_ab_test(
            user_id, ambiguity, user_constraints, archetype
        )
    
    async def _can_schedule_experiment(self, user_id: str) -> bool:
        """Check if user is eligible for new experiment."""
        store = await get_behavioral_store()
        if not store:
            return False
        
        # Check concurrent experiments
        running = store.client.table('experiments').select('*').eq(
            'user_id', user_id
        ).eq('status', 'running').execute()
        
        if running.data and len(running.data) >= self.max_concurrent_experiments:
            return False
        
        # Check last experiment time
        last = store.client.table('experiments').select('*').eq(
            'user_id', user_id
        ).order('created_at', desc=True).limit(1).execute()
        
        if last.data:
            last_time = datetime.fromisoformat(last.data[0]['created_at'])
            hours_since = (datetime.utcnow() - last_time).total_seconds() / 3600
            if hours_since < self.min_hours_between_experiments:
                return False
        
        return True
    
    async def _design_energy_timing_experiment(
        self,
        user_id: str,
        hypothesis: Optional[str],
        constraints: Optional[Dict],
        archetype: Optional[Any]
    ) -> ExperimentDesign:
        """
        Design experiment to disambiguate energy vs timing.
        Example: "Is it that you're tired, or that afternoon meetings drain you?"
        """
        conditions = [
            {
                "name": "control",
                "description": "Normal schedule, observe baseline",
                "modifications": []
            },
            {
                "name": "morning_deep_work",
                "description": "Move cognitively demanding task to 9 AM",
                "modifications": ["schedule_deep_work_9am", "protect_morning"]
            },
            {
                "name": "afternoon_with_breaks",
                "description": "Keep afternoon timing but add structured breaks",
                "modifications": ["add_breaks_2pm_4pm", "pomodoro_afternoon"]
            }
        ]
        
        return ExperimentDesign(
            experiment_id=f"exp_{user_id}_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            target_hypothesis=hypothesis or "Energy decline is circadian, not task-related",
            ambiguity_to_resolve="energy_vs_time",
            conditions=conditions,
            n_trials_per_condition=3,
            duration_days=9,  # 3 days per condition
            proposed_schedule=self._generate_schedule(conditions, 9, constraints),
            randomization_method="block",  # Block by day
            user_constraints=constraints or {},
            ethical_checks=["no_sleep_disruption", "work_obligation_compatible"],
            status="designed",
            created_at=datetime.utcnow()
        )
    
    async def _design_avoidance_experiment(
        self,
        user_id: str,
        hypothesis: Optional[str],
        constraints: Optional[Dict],
        archetype: Optional[Any]
    ) -> ExperimentDesign:
        """
        Design experiment to disambiguate avoidance vs capacity.
        Example: "Are you procrastinating, or is the task actually too hard?"
        """
        conditions = [
            {
                "name": "baseline",
                "description": "Normal approach to task",
                "modifications": []
            },
            {
                "name": "reduced_scope",
                "description": "Same task, 50% scope reduction",
                "modifications": ["reduce_scope_50", "maintain_quality_standard"]
            },
            {
                "name": "external_accountability",
                "description": "Full scope, but with check-in partner",
                "modifications": ["add_accountability", "schedule_checkin"]
            }
        ]
        
        return ExperimentDesign(
            experiment_id=f"exp_{user_id}_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            target_hypothesis=hypothesis or "Procrastination is scope-based, not capacity-based",
            ambiguity_to_resolve="avoidance_vs_capacity",
            conditions=conditions,
            n_trials_per_condition=2,
            duration_days=6,
            proposed_schedule=self._generate_schedule(conditions, 6, constraints),
            randomization_method="adaptive",  # Adjust based on early results
            user_constraints=constraints or {},
            ethical_checks=["no_shame_induction", "voluntary_participation"],
            status="designed",
            created_at=datetime.utcnow()
        )
    
    def _generate_schedule(
        self,
        conditions: List[Dict],
        duration_days: int,
        constraints: Optional[Dict]
    ) -> List[datetime]:
        """Generate experiment schedule respecting constraints."""
        schedule = []
        start = datetime.utcnow() + timedelta(days=1)  # Start tomorrow
        
        # Simple round-robin through conditions
        for day in range(duration_days):
            condition_idx = day % len(conditions)
            schedule.append(start + timedelta(days=day))
        
        return schedule
    
    async def propose_experiment_to_user(
        self,
        experiment: ExperimentDesign,
        empathy_wrapper: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Convert experiment design to user-facing proposal.
        Preserves agency - user must opt in.
        """
        # Generate human-readable description
        description = self._generate_user_description(experiment)
        
        # Calculate burden score
        burden = self._calculate_burden_score(experiment)
        
        return {
            "experiment_id": experiment.experiment_id,
            "title": f"Help me understand: {experiment.ambiguity_to_resolve.replace('_', ' ').title()}",
            "description": description,
            "duration": f"{experiment.duration_days} days",
            "time_commitment": f"~{burden['minutes_per_day']} minutes/day",
            "burden_level": burden["level"],
            "opt_in_required": True,
            "can_modify": True,
            "what_youll_learn": self._generate_learning_promise(experiment),
            "conditions_preview": [
                c["description"] for c in experiment.conditions
            ]
        }
    
    def _generate_user_description(self, experiment: ExperimentDesign) -> str:
        """Generate accessible description of experiment."""
        templates = {
            "energy_vs_time": (
                "I noticed your energy seems to dip at certain times, but I'm not sure "
                "if it's *when* you're working or *what* you're working on. "
                "Over {duration} days, let's try a few different schedules to find out."
            ),
            "avoidance_vs_capacity": (
                "Sometimes we avoid tasks because they're hard, sometimes because they "
                "feel too big. Let's figure out which is which for you."
            ),
            "habit_vs_intention": (
                "I'd like to understand whether your current patterns are choices "
                "or just momentum. We'll test a small intentional change."
            )
        }
        
        template = templates.get(
            experiment.ambiguity_to_resolve,
            "I'd like to run a small experiment to better understand your patterns."
        )
        
        return template.format(duration=experiment.duration_days)
    
    def _calculate_burden_score(self, experiment: ExperimentDesign) -> Dict[str, Any]:
        """Calculate how burdensome this experiment is."""
        # Base burden on number of modifications
        total_mods = sum(len(c.get("modifications", [])) for c in experiment.conditions)
        
        minutes_per_day = 5 + (total_mods * 2)  # Base 5 min + 2 per modification
        
        if minutes_per_day < 10:
            level = "minimal"
        elif minutes_per_day < 20:
            level = "light"
        else:
            level = "moderate"
        
        return {
            "minutes_per_day": minutes_per_day,
            "level": level,
            "total_modifications": total_mods
        }
    
    def _generate_learning_promise(self, experiment: ExperimentDesign) -> str:
        """What will user learn from this experiment?"""
        promises = {
            "energy_vs_time": "Your optimal work schedule based on data, not guesswork",
            "avoidance_vs_capacity": "Whether to break tasks down or push through",
            "habit_vs_intention": "Which of your routines are actually serving you"
        }
        
        return promises.get(
            experiment.ambiguity_to_resolve,
            "A personalized insight about your behavioral patterns"
        )


class MemoryAdmissionControl:
    """
    Filters what enters long-term memory.
    90-98% of behavioral data should be discarded.
    """
    
    def __init__(self):
        self.retention_categories = {
            "prediction_failure": 1.0,      # Always keep - high learning value
            "intervention_reversal": 1.0,    # Always keep - causal gold
            "high_confidence_causal": 0.9,   # Keep if confidence > 0.8
            "belief_challenge_event": 0.9,   # Keep belief-reality gaps
            "experiment_result": 0.8,        # Keep experimental data
            "archetype_transition": 0.8,     # Keep major behavioral shifts
            "routine_pattern": 0.1,          # Rarely keep - low information
            "noise": 0.0                     # Never keep
        }
    
    async def evaluate_for_retention(
        self,
        observation: Dict,
        grading: EvidenceGrading,
        user_id: str
    ) -> Tuple[bool, str, float]:
        """
        Decide whether to retain this observation in semantic memory.
        Returns: (retain, category, priority_score)
        """
        # Check for high-value categories first
        
        # 1. Prediction failure?
        if await self._is_prediction_failure(observation, user_id):
            return True, "prediction_failure", 1.0
        
        # 2. Intervention reversal?
        if await self._is_intervention_reversal(observation, user_id):
            return True, "intervention_reversal", 1.0
        
        # 3. High-confidence causal link?
        if grading.level.value >= EvidenceLevel.INTERVENTION_RESPONSE.value:
            if grading.confidence > 0.8:
                return True, "high_confidence_causal", 0.9
        
        # 4. Belief challenge event?
        if await self._is_belief_challenge_event(observation, user_id):
            return True, "belief_challenge_event", 0.9
        
        # 5. Part of running experiment?
        if observation.get("experiment_id"):
            return True, "experiment_result", 0.8
        
        # 6. Archetype transition?
        if await self._is_archetype_transition(observation, user_id):
            return True, "archetype_transition", 0.8
        
        # 7. Routine pattern - rarely keep
        if grading.level.value <= EvidenceLevel.BEHAVIORAL_METRIC.value:
            # Only keep 2% of routine data (stochastic sparsity)
            import random
            if random.random() < 0.02:
                return True, "routine_pattern", 0.1
        
        # Default: discard
        return False, "noise", 0.0
    
    async def _is_prediction_failure(self, observation: Dict, user_id: str) -> bool:
        """Check if this observation contradicts a previous prediction."""
        store = await get_behavioral_store()
        if not store:
            return False
        
        # Look for recent predictions about this time period
        recent_predictions = store.client.table('predictions').select('*').eq(
            'user_id', user_id
        ).eq('status', 'resolved').order('predicted_for', desc=True).limit(5).execute()
        
        if not recent_predictions.data:
            return False
        
        # Check if observation contradicts any prediction
        for pred in recent_predictions.data:
            predicted_state = pred.get('predicted_state')
            actual_behavior = observation.get('behavior', {})
            
            # Simple contradiction check
            if predicted_state == 'high_focus' and actual_behavior.get('focus_score', 0) < 0.3:
                return True
            if predicted_state == 'vulnerability' and actual_behavior.get('focus_score', 0) > 0.7:
                return True
        
        return False
    
    async def _is_intervention_reversal(self, observation: Dict, user_id: str) -> bool:
        """Check if this shows an intervention having opposite effect."""
        if not observation.get('intervention_id'):
            return False
        
        store = await get_behavioral_store()
        if not store:
            return False
        
        # Get intervention details
        intervention = store.client.table('interventions').select('*').eq(
            'intervention_id', observation['intervention_id']
        ).execute()
        
        if not intervention.data:
            return False
        
        expected_outcome = intervention.data[0].get('expected_outcome')
        actual_outcome = observation.get('outcome', {})
        
        # Check for reversal
        if expected_outcome == 'increased_focus' and actual_outcome.get('focus_change', 0) < -0.2:
            return True
        
        return False
    
    async def _is_belief_challenge_event(self, observation: Dict, user_id: str) -> bool:
        """Check if this triggered a belief challenge."""
        return observation.get('processing_metadata', {}).get('dissonance_triggered', False)
    
    async def _is_archetype_transition(self, observation: Dict, user_id: str) -> bool:
        """Check if user is transitioning between archetypes."""
        # Would check archetype stability over time
        return False  # Placeholder


class RealityFilterEngine:
    """
    Main orchestrator: Confound Detection → Evidence Scoring → Experiment Scheduling → Memory Admission.
    Inserts between sensing and hypothesis formation.
    """
    
    def __init__(self):
        self.confound_detector = ConfoundDetector()
        self.evidence_scorer = EvidenceScorer()
        self.experiment_scheduler = ExperimentScheduler()
        self.memory_admission = MemoryAdmissionControl()
        self.grading_engine = EvidenceGradingEngine()
        
        # Thresholds
        self.min_evidence_score_for_hypothesis = 0.5
        self.max_confound_severity_for_hypothesis = 0.7
    
    async def process_observation(
        self,
        observation: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """
        Main entry point: Filter reality before it becomes belief.
        
        Returns:
            - action: 'form_hypothesis', 'schedule_experiment', 'accumulate_data', 'discard'
            - evidence_grading: PEL level assigned
            - confound_assessment: Detected confounds
            - experiment_proposed: If experiment scheduled
            - memory_decision: Whether to retain
        """
        result = {
            "observation_id": observation.get('observation_id'),
            "user_id": user_id,
            "processed_at": datetime.utcnow().isoformat()
        }
        
        # STEP 1: Grade the evidence (PEL)
        grading = await self.grading_engine.grade_observation(observation, user_id)
        result["evidence_grading"] = {
            "level": grading.level.name,
            "level_value": grading.level.value,
            "confidence": grading.confidence,
            "reason": grading.grading_reason
        }
        
        # STEP 2: Confound Detection
        confounds = await self.confound_detector.audit_observation(
            observation, user_id
        )
        result["confound_assessment"] = {
            "confounds_detected": len(confounds),
            "confound_summary": self.confound_detector.get_confound_summary(confounds),
            "severe_confounds": [
                c.confound_type.value for c in confounds 
                if c.severity > self.max_confound_severity_for_hypothesis
            ]
        }
        
        # STEP 3: Evidence Scoring (if not Level 0)
        evidence_score = 0
        score_breakdown = {}
        
        if grading.level.value >= EvidenceLevel.BEHAVIORAL_METRIC.value:
            # Get historical context
            store = await get_behavioral_store()
            historical = []
            contexts = set()
            interventions = []
            
            if store:
                since = (datetime.utcnow() - timedelta(days=14)).isoformat()
                hist_result = store.client.table('behavioral_observations').select('*').eq(
                    'user_id', user_id
                ).gte('timestamp', since).execute()
                historical = hist_result.data if hist_result.data else []
                
                # Extract contexts
                contexts = set(
                    o.get('context', {}).get('location_type', 'unknown') 
                    for o in historical
                )
                
                # Get intervention responses
                int_result = store.client.table('intervention_outcomes').select('*').eq(
                    'user_id', user_id
                ).gte('timestamp', since).execute()
                interventions = int_result.data if int_result.data else []
            
            evidence_score, score_breakdown = self.evidence_scorer.calculate_evidence_score(
                observation, historical, interventions, contexts
            )
            
            result["evidence_score"] = {
                "composite": evidence_score,
                "breakdown": score_breakdown
            }
        
        # STEP 4: Decide action based on grade, confounds, score
        
        # High confounds → Schedule experiment to disambiguate
        severe_confounds = [
            c for c in confounds 
            if c.severity > self.max_confound_severity_for_hypothesis
        ]
        
        if severe_confounds and grading.level.value < EvidenceLevel.INTERVENTION_RESPONSE.value:
            # Can't form hypothesis with severe confounds at low evidence level
            # Design experiment instead
            experiment = await self.experiment_scheduler.design_experiment(
                user_id=user_id,
                ambiguity=f"{severe_confounds[0].confound_type.value}_confound",
                current_hypothesis=None
            )
            
            result["action"] = "schedule_experiment"
            result["action_reason"] = f"Severe confound detected: {severe_confounds[0].confound_type.value}"
            result["experiment_proposed"] = (
                await self.experiment_scheduler.propose_experiment_to_user(experiment)
                if experiment else None
            )
            
            # Still evaluate for memory (prediction failures in experiments are valuable)
            retain, category, priority = await self.memory_admission.evaluate_for_retention(
                observation, grading, user_id
            )
            result["memory_decision"] = {
                "retain": retain,
                "category": category,
                "priority": priority
            }
            
            await self.grading_engine.persist_grading(grading)
            return result
        
        # Low evidence score at low level → Accumulate more data
        if (grading.level.value <= EvidenceLevel.CONTEXTUALIZED_PATTERN.value and 
            evidence_score < self.min_evidence_score_for_hypothesis):
            
            result["action"] = "accumulate_data"
            result["action_reason"] = f"Evidence score {evidence_score:.2f} below threshold {self.min_evidence_score_for_hypothesis}"
            
            # Minimal memory retention
            result["memory_decision"] = {"retain": False, "category": "noise", "priority": 0}
            
            await self.grading_engine.persist_grading(grading)
            return result
        
        # Level 3+ with acceptable confounds → Can form hypothesis
        if grading.level.value >= EvidenceLevel.INTERVENTION_RESPONSE.value:
            result["action"] = "form_hypothesis"
            result["action_reason"] = f"Evidence level {grading.level.name} sufficient for causal inference"
        
        # Level 2 with high score → Can form tentative hypothesis
        elif (grading.level.value == EvidenceLevel.CONTEXTUALIZED_PATTERN.value and 
              evidence_score >= self.min_evidence_score_for_hypothesis):
            result["action"] = "form_hypothesis"
            result["action_reason"] = "Strong contextualized pattern with good evidence score"
            result["hypothesis_confidence"] = "tentative"
        
        else:
            result["action"] = "accumulate_data"
            result["action_reason"] = "Insufficient evidence for hypothesis formation"
        
        # STEP 5: Memory Admission Control
        retain, category, priority = await self.memory_admission.evaluate_for_retention(
            observation, grading, user_id
        )
        result["memory_decision"] = {
            "retain": retain,
            "category": category,
            "priority": priority
        }
        
        # Persist grading
        await self.grading_engine.persist_grading(grading)
        
        return result
    
    async def should_upgrade_evidence(
        self,
        current_grading: EvidenceGrading,
        user_id: str
    ) -> Optional[EvidenceGrading]:
        """
        Check if evidence should be upgraded based on new data.
        Called periodically by consolidation or when new data arrives.
        """
        # Gather upgrade context
        store = await get_behavioral_store()
        context = {}
        
        if store:
            # Get observations since this grading
            since = current_grading.graded_at.isoformat()
            recent = store.client.table('behavioral_observations').select('*').eq(
                'user_id', user_id
            ).gte('timestamp', since).execute()
            
            context["n_observations"] = len(recent.data) if recent.data else 0
            
            # Get unique contexts
            contexts = set()
            for obs in recent.data if recent.data else []:
                ctx = obs.get('context', {})
                contexts.add(ctx.get('location_type', 'unknown'))
                contexts.add(ctx.get('time_of_day', 'unknown'))
            context["n_contexts"] = len(contexts)
            
            # Calculate timespan
            if recent.data and len(recent.data) > 1:
                first = datetime.fromisoformat(recent.data[0]['timestamp'].replace('Z', '+00:00'))
                last = datetime.fromisoformat(recent.data[-1]['timestamp'].replace('Z', '+00:00'))
                context["timespan_days"] = (last - first).days
            
            # Get intervention responses if applicable
            if current_grading.level.value == EvidenceLevel.CONTEXTUALIZED_PATTERN.value:
                interventions = store.client.table('intervention_outcomes').select('*').eq(
                    'user_id', user_id
                ).gte('timestamp', since).execute()
                
                if interventions.data:
                    context["intervention_id"] = interventions.data[0].get('intervention_id')
                    context["intervention_type"] = interventions.data[0].get('intervention_type')
                    context["response_magnitude"] = interventions.data[0].get('outcome_magnitude')
                    context["causal_confidence"] = interventions.data[0].get('causal_confidence', 0.5)
                    context["confounds_controlled"] = interventions.data[0].get('confounds_controlled', [])
        
        # Attempt upgrade
        upgraded = await self.grading_engine.upgrade_evidence(current_grading, context)
        
        if upgraded.level != current_grading.level:
            await self.grading_engine.persist_grading(upgraded)
            return upgraded
        
        return None
