
# Create the Values Inference Module (VIM) - Core Implementation

vim_core_code = '''"""
Values Inference Module (VIM): Inferring enforced preferences from behavioral tradeoffs.

Core Principle:
- Goals are planned preferences (negotiable)
- Values are enforced preferences (defended at cost)

VIM does not read statements. It watches tradeoffs under pressure.
Values only appear when two desirable things cannot coexist.

Key Insight:
A value only exists when cost is willingly paid multiple times.
No cost → no value.

Integration:
- Informs experiment design (what tradeoffs to test)
- Shapes intervention framing (value-aligned language)
- Never generates recommendations directly (preserves user agency)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from enum import Enum, auto
from datetime import datetime, timedelta
from collections import defaultdict
import json
import math
from abc import ABC, abstractmethod


class ValueCategory(Enum):
    """Core value dimensions that organize human behavior."""
    AUTONOMY = "autonomy"              # Self-direction, freedom from control
    RESPONSIBILITY = "responsibility"  # Duty, obligation to others
    MEANING = "meaning"                # Purpose, significance, depth
    DIGNITY = "dignity"                # Self-respect, avoidance of shame
    SECURITY = "security"              # Safety, predictability, order
    CONNECTION = "connection"          # Belonging, intimacy, community
    MASTERY = "mastery"                # Competence, growth, achievement
    PLEASURE = "pleasure"              # Enjoyment, comfort, satisfaction
    INTEGRITY = "integrity"            # Coherence between values and action
    LEGACY = "legacy"                  # Impact beyond self, transmission


class TradeoffType(Enum):
    """Types of tradeoffs that reveal values."""
    PROGRESS_VS_FLEXIBILITY = "progress_vs_flexibility"
    REST_VS_COMPLETION = "rest_vs_completion"
    MONEY_VS_MEANING = "money_vs_meaning"
    OPPORTUNITY_VS_EXPOSURE = "opportunity_vs_exposure"
    SAFETY_VS_GROWTH = "safety_vs_growth"
    INDEPENDENCE_VS_BELONGING = "independence_vs_belonging"
    DEPTH_VS_BREADTH = "depth_vs_breadth"
    COMFORT_VS_PURPOSE = "comfort_vs_purpose"
    PRESENT_VS_FUTURE = "present_vs_future"
    SELF_VS_OTHER = "self_vs_other"


@dataclass
class SacrificeEvent:
    """
    A recorded instance where user paid cost to preserve something.
    Core evidence for value inference.
    """
    event_id: str
    user_id: str
    timestamp: datetime
    
    # What was sacrificed (the cost)
    sacrificed_resource: str  # 'time', 'money', 'opportunity', 'rest', 'progress'
    sacrificed_amount: float  # Quantified cost
    sacrificed_description: str
    
    # What was protected (the value)
    protected_value: ValueCategory
    protected_description: str
    
    # Context
    tradeoff_type: TradeoffType
    decision_context: Dict[str, Any]  # Situational factors
    
    # Verification
    repeated_pattern: bool  # Has this sacrifice pattern occurred before?
    emotional_intensity: float  # 0-1, measured or inferred
    recovery_time_hours: Optional[float] = None  # How long to recover from cost
    
    # Source
    source_observation_id: Optional[str] = None
    inference_confidence: float = 0.5


@dataclass
class EmotionalGradient:
    """
    Tracks emotional energy expenditure patterns.
    Values generate disproportionate emotional investment.
    """
    user_id: str
    value_category: ValueCategory
    
    # Intensity signals
    frustration_spikes: List[Dict[str, Any]]  # When this value threatened
    pride_moments: List[Dict[str, Any]]       # When this value affirmed
    rumination_episodes: List[Dict[str, Any]]  # Persistent thought patterns
    
    # Energy patterns
    fatigue_recovery_rate: float  # Fast recovery = value-aligned activity
    sustained_attention_minutes: float  # Long focus without effort sensation
    
    # Computed
    total_emotional_energy_invested: float
    significance_score: float  # 0-1, derived from intensity patterns
    
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ReturnVector:
    """
    Post-failure behavior drift.
    The cleanest signal: values restart automatically, goals require effort.
    """
    user_id: str
    disruption_event_id: str
    disruption_type: str  # 'failure', 'interruption', 'external_block'
    
    # What was disrupted
    disrupted_goal: Optional[str]
    disrupted_behavior: str
    
    # Recovery pattern
    planned_resumption: Optional[str]  # What they intended to resume
    actual_resumption: str  # What they actually resumed
    resumption_delay_hours: float
    
    # Automatic vs Effortful
    automatic_restart: bool  # No conscious decision to resume
    required_effort_to_restart: float  # 0-1, self-reported or inferred
    
    # Value inference
    inferred_value: Optional[ValueCategory]
    confidence: float
    
    observed_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class InferredValue:
    """
    A value inferred from multiple behavioral signals.
    Confidence builds through convergent evidence.
    """
    user_id: str
    value_category: ValueCategory
    
    # Evidence base
    sacrifice_events: List[SacrificeEvent] = field(default_factory=list)
    emotional_gradients: List[EmotionalGradient] = field(default_factory=list)
    return_vectors: List[ReturnVector] = field(default_factory=list)
    
    # Confidence metrics
    n_observations: int = 0
    consistency_score: float = 0.0  # Across contexts
    cost_willingness_score: float = 0.0  # Repeated cost payment
    
    # Inferred properties
    strength: float = 0.0  # 0-1, how non-negotiable
    hierarchy_position: Optional[int] = None  # Rank among user's values
    conflicts_with: List[ValueCategory] = field(default_factory=list)
    
    # Temporal dynamics
    first_inferred: datetime = field(default_factory=datetime.utcnow)
    last_confirmed: datetime = field(default_factory=datetime.utcnow)
    stability_score: float = 0.5  # How consistent over time
    
    # Status
    status: str = "hypothetical"  # hypothetical → emerging → confirmed → core
    
    def calculate_confidence(self) -> float:
        """Calculate overall confidence from convergent evidence."""
        if self.n_observations < 3:
            return 0.3
        
        # Weight different signals
        sacrifice_weight = min(0.4, len(self.sacrifice_events) * 0.1)
        emotional_weight = min(0.3, len(self.emotional_gradients) * 0.15)
        return_weight = min(0.3, len(self.return_vectors) * 0.15)
        
        # Consistency bonus
        consistency_bonus = self.consistency_score * 0.2 if self.consistency_score > 0.7 else 0
        
        # Cost willingness is strongest signal
        cost_bonus = self.cost_willingness_score * 0.3 if self.cost_willingness_score > 0.6 else 0
        
        base = sacrifice_weight + emotional_weight + return_weight
        return min(0.95, base + consistency_bonus + cost_bonus)


class SacrificeDetector:
    """
    Detects sacrifice patterns from behavioral observations.
    Core rule: A value only exists when cost is willingly paid multiple times.
    """
    
    TRADEOFF_PATTERNS = {
        TradeoffType.PROGRESS_VS_FLEXIBILITY: {
            "sacrificed": "progress",
            "protected": ValueCategory.AUTONOMY,
            "markers": ["delayed_deadline", "changed_plan", "kept_options_open"],
            "emotional_signal": "relief_after_delay"
        },
        TradeoffType.REST_VS_COMPLETION: {
            "sacrificed": "rest",
            "protected": ValueCategory.RESPONSIBILITY,
            "markers": ["worked_while_tired", "finished_despite_exhaustion", "sacrificed_sleep"],
            "emotional_signal": "satisfaction_after_completion"
        },
        TradeoffType.MONEY_VS_MEANING: {
            "sacrificed": "money",
            "protected": ValueCategory.MEANING,
            "markers": ["took_lower_pay", "paid_for_learning", "invested_in_passion"],
            "emotional_signal": "fulfillment_despite_cost"
        },
        TradeoffType.OPPORTUNITY_VS_EXPOSURE: {
            "sacrificed": "opportunity",
            "protected": ValueCategory.DIGNITY,
            "markers": ["avoided_visibility", "declined_spotlight", "skipped_networking"],
            "emotional_signal": "relief_from_avoidance"
        },
        TradeoffType.SAFETY_VS_GROWTH: {
            "sacrificed": "security",
            "protected": ValueCategory.MASTERY,
            "markers": ["took_risk", "left_stable_situation", "embraced_uncertainty"],
            "emotional_signal": "excitement_despite_anxiety"
        },
        TradeoffType.INDEPENDENCE_VS_BELONGING: {
            "sacrificed": "connection",
            "protected": ValueCategory.AUTONOMY,
            "markers": ["worked_alone", "declined_collaboration", "maintained_distance"],
            "emotional_signal": "contentment_in_solitude"
        },
        TradeoffType.DEPTH_VS_BREADTH: {
            "sacrificed": "opportunities",
            "protected": ValueCategory.MASTERY,
            "markers": ["specialized", "declined_side_projects", "focused_narrowly"],
            "emotional_signal": "satisfaction_in_expertise"
        },
        TradeoffType.COMFORT_VS_PURPOSE: {
            "sacrificed": "comfort",
            "protected": ValueCategory.MEANING,
            "markers": ["endured_hardship", "persisted_through_difficulty", "accepted_discomfort"],
            "emotional_signal": "purpose_sustains_effort"
        },
        TradeoffType.PRESENT_VS_FUTURE: {
            "sacrificed": "present_ease",
            "protected": ValueCategory.LEGACY,
            "markers": ["delayed_gratification", "invested_long_term", "sacrificed_now"],
            "emotional_signal": "meaning_from_deferral"
        },
        TradeoffType.SELF_VS_OTHER: {
            "sacrificed": "self_interest",
            "protected": ValueCategory.CONNECTION,
            "markers": ["helped_others", "put_someone_first", "compromised_for_group"],
            "emotional_signal": "warmth_from_giving"
        }
    }
    
    def __init__(self):
        self.sacrifice_history: Dict[str, List[SacrificeEvent]] = defaultdict(list)
    
    async def detect_from_observation(
        self,
        observation: Dict[str, Any],
        user_id: str
    ) -> List[SacrificeEvent]:
        """
        Analyze observation for sacrifice patterns.
        Returns detected sacrifice events.
        """
        detected = []
        text = json.dumps(observation).lower()
        behavior = observation.get('behavior', {})
        ema = observation.get('ema_response', {})
        
        for tradeoff_type, pattern in self.TRADEOFF_PATTERNS.items():
            # Check for markers
            markers_found = [m for m in pattern["markers"] if m in text]
            
            if len(markers_found) >= 2:  # Multiple markers = stronger signal
                # Check for emotional signal
                emotional_match = self._check_emotional_signal(
                    pattern["emotional_signal"], 
                    observation
                )
                
                # Check if cost was actually paid
                cost_verified = self._verify_cost_payment(observation, pattern["sacrificed"])
                
                if cost_verified:
                    # Check if this is a repeated pattern
                    repeated = self._is_repeated_pattern(user_id, tradeoff_type)
                    
                    sacrifice = SacrificeEvent(
                        event_id=f"sac_{user_id}_{datetime.utcnow().timestamp()}",
                        user_id=user_id,
                        timestamp=datetime.utcnow(),
                        sacrificed_resource=pattern["sacrificed"],
                        sacrificed_amount=self._quantify_cost(observation, pattern["sacrificed"]),
                        sacrificed_description=f"Sacrificed {pattern['sacrificed']} to preserve {pattern['protected'].value}",
                        protected_value=pattern["protected"],
                        protected_description=f"Protected {pattern['protected'].value}",
                        tradeoff_type=tradeoff_type,
                        decision_context=observation.get('context', {}),
                        repeated_pattern=repeated,
                        emotional_intensity=emotional_match,
                        source_observation_id=observation.get('observation_id'),
                        inference_confidence=0.6 + (0.1 * len(markers_found)) + (0.1 if repeated else 0)
                    )
                    
                    detected.append(sacrifice)
                    self.sacrifice_history[user_id].append(sacrifice)
        
        return detected
    
    def _check_emotional_signal(self, signal_type: str, observation: Dict) -> float:
        """Check if expected emotional signal is present."""
        ema = observation.get('ema_response', {})
        reflection = observation.get('reflection', {})
        
        text = json.dumps({**ema, **reflection}).lower()
        
        signal_markers = {
            "relief_after_delay": ["relieved", "less_stress", "better now", "glad i waited"],
            "satisfaction_after_completion": ["satisfied", "worth it", "glad i finished", "proud"],
            "fulfillment_despite_cost": ["fulfilled", "meaningful", "worth the cost", "right choice"],
            "relief_from_avoidance": ["relieved", "avoided", "dodged", "glad i didn't"],
            "excitement_despite_anxiety": ["excited", "thrilled", "scary but", "worth the risk"],
            "contentment_in_solitude": ["peaceful", "focused", "clear", "my own pace"],
            "satisfaction_in_expertise": ["deeply satisfying", "mastery", "expert", "depth"],
            "purpose_sustains_effort": ["purpose", "meaning", "why i'm doing this", "matters"],
            "meaning_from_deferral": ["future", "later", "investment", "building"],
            "warmth_from_giving": ["warm", "connected", "helped", "worthwhile"]
        }
        
        markers = signal_markers.get(signal_type, [])
        matches = sum(1 for m in markers if m in text)
        
        return min(1.0, matches * 0.3)
    
    def _verify_cost_payment(self, observation: Dict, resource: str) -> bool:
        """Verify that actual cost was paid (not just intended)."""
        behavior = observation.get('behavior', {})
        
        cost_indicators = {
            "progress": ['deadline_missed', 'goal_delayed', 'opportunity_lost'],
            "rest": ['sleep_lost', 'fatigue_reported', 'recovery_needed'],
            "money": ['expense_incurred', 'income_reduced', 'investment_made'],
            "opportunity": ['opportunity_declined', 'option_closed', 'path_not_taken'],
            "security": ['risk_taken', 'stability_left', 'unknown_embraced'],
            "connection": ['isolation_chosen', 'collaboration_declined', 'distance_kept'],
            "comfort": ['discomfort_accepted', 'hardship_endured', 'difficulty_persisted'],
            "present_ease": ['effort_expended', 'difficulty_accepted', 'delay_accepted'],
            "self_interest": ['others_prioritized', 'own_needs_delayed', 'help_given']
        }
        
        indicators = cost_indicators.get(resource, [])
        text = json.dumps(behavior).lower()
        
        return any(ind in text for ind in indicators)
    
    def _quantify_cost(self, observation: Dict, resource: str) -> float:
        """Attempt to quantify the cost paid."""
        # Simplified quantification - would be more sophisticated in production
        behavior = observation.get('behavior', {})
        
        if resource == "rest":
            return behavior.get('sleep_hours_lost', 0) or behavior.get('fatigue_score', 0) * 2
        elif resource == "money":
            return behavior.get('amount', 0) or behavior.get('opportunity_cost', 0)
        elif resource == "progress":
            return behavior.get('days_delayed', 0) * 10 or behavior.get('milestone_missed', 0) * 20
        
        return 1.0  # Default minimal cost
    
    def _is_repeated_pattern(self, user_id: str, tradeoff_type: TradeoffType) -> bool:
        """Check if this sacrifice pattern has occurred before."""
        history = self.sacrifice_history.get(user_id, [])
        similar = [s for s in history if s.tradeoff_type == tradeoff_type]
        return len(similar) >= 2  # At least 2 previous instances


class EmotionalEnergyTracker:
    """
    Tracks emotional energy gradients to identify value-aligned activities.
    Values generate disproportionate emotional investment.
    """
    
    def __init__(self):
        self.user_gradients: Dict[str, Dict[ValueCategory, EmotionalGradient]] = {}
    
    async def update_from_observation(
        self,
        observation: Dict[str, Any],
        user_id: str
    ) -> List[EmotionalGradient]:
        """Update emotional gradients from observation."""
        updated = []
        
        ema = observation.get('ema_response', {})
        behavior = observation.get('behavior', {})
        
        # Extract emotional signals
        frustration = self._detect_frustration(ema, behavior)
        pride = self._detect_pride(ema, behavior)
        rumination = self._detect_rumination(ema, behavior)
        
        # Map to potential values
        value_signals = self._map_emotions_to_values(frustration, pride, rumination)
        
        for value_cat, intensity in value_signals.items():
            if value_cat not in self.user_gradients.get(user_id, {}):
                self.user_gradients[user_id] = {}
            
            if value_cat not in self.user_gradients[user_id]:
                self.user_gradients[user_id][value_cat] = EmotionalGradient(
                    user_id=user_id,
                    value_category=value_cat,
                    frustration_spikes=[],
                    pride_moments=[],
                    rumination_episodes=[],
                    fatigue_recovery_rate=0.5,
                    sustained_attention_minutes=0,
                    total_emotional_energy_invested=0,
                    significance_score=0
                )
            
            gradient = self.user_gradients[user_id][value_cat]
            
            # Update with new signals
            if frustration:
                gradient.frustration_spikes.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "intensity": frustration * intensity,
                    "trigger": observation.get('context', {}).get('situation')
                })
            
            if pride:
                gradient.pride_moments.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "intensity": pride * intensity,
                    "achievement": observation.get('behavior', {}).get('action')
                })
            
            if rumination:
                gradient.rumination_episodes.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "duration_minutes": rumination,
                    "topic": observation.get('ema_response', {}).get('dominant_thought')
                })
            
            # Recalculate significance
            gradient.significance_score = self._calculate_significance(gradient)
            gradient.total_emotional_energy_invested += intensity
            gradient.last_updated = datetime.utcnow()
            
            updated.append(gradient)
        
        return updated
    
    def _detect_frustration(self, ema: Dict, behavior: Dict) -> float:
        """Detect frustration intensity from signals."""
        frustration_markers = [
            'frustrated', 'annoyed', 'angry', 'blocked', 'prevented',
            'couldn\'t', 'failed', 'interrupted', 'disrupted'
        ]
        text = json.dumps(ema).lower()
        matches = sum(1 for m in frustration_markers if m in text)
        return min(1.0, matches * 0.2)
    
    def _detect_pride(self, ema: Dict, behavior: Dict) -> float:
        """Detect pride/satisfaction intensity."""
        pride_markers = [
            'proud', 'satisfied', 'accomplished', 'completed', 'achieved',
            'did it', 'success', 'nailed it', 'fulfilled'
        ]
        text = json.dumps(ema).lower()
        matches = sum(1 for m in pride_markers if m in text)
        return min(1.0, matches * 0.2)
    
    def _detect_rumination(self, ema: Dict, behavior: Dict) -> float:
        """Detect rumination (persistent thought) duration."""
        # Check for repeated mentions or extended focus
        thoughts = ema.get('dominant_thoughts', [])
        if isinstance(thoughts, list) and len(thoughts) > 0:
            # Check for repetition
            unique_thoughts = set(thoughts)
            if len(thoughts) > len(unique_thoughts) * 1.5:  # Repetition detected
                return len(thoughts) * 5  # Minutes
        return 0
    
    def _map_emotions_to_values(
        self,
        frustration: float,
        pride: float,
        rumination: float
    ) -> Dict[ValueCategory, float]:
        """Map emotional signals to value categories."""
        # This would be more sophisticated with context
        # Simplified mapping based on common patterns
        mappings = {}
        
        if frustration > 0.3:
            # Frustration often signals threatened values
            mappings[ValueCategory.AUTONOMY] = frustration * 0.8
            mappings[ValueCategory.MASTERY] = frustration * 0.6
        
        if pride > 0.3:
            # Pride signals affirmed values
            mappings[ValueCategory.RESPONSIBILITY] = pride * 0.7
            mappings[ValueCategory.MASTERY] = pride * 0.8
            mappings[ValueCategory.MEANING] = pride * 0.6
        
        if rumination > 0:
            # Rumination indicates significance
            mappings[ValueCategory.MEANING] = min(1.0, rumination / 60) * 0.9
            mappings[ValueCategory.CONNECTION] = min(1.0, rumination / 60) * 0.7
        
        return mappings
    
    def _calculate_significance(self, gradient: EmotionalGradient) -> float:
        """Calculate overall significance score from patterns."""
        n_frustrations = len(gradient.frustration_spikes)
        n_pride = len(gradient.pride_moments)
        n_rumination = len(gradient.rumination_episodes)
        
        # More signals = higher significance
        signal_count = min(10, n_frustrations + n_pride + n_rumination)
        
        # Average intensity
        avg_frustration = sum(f['intensity'] for f in gradient.frustration_spikes[-5:]) / max(1, min(5, n_frustrations))
        avg_pride = sum(p['intensity'] for p in gradient.pride_moments[-5:]) / max(1, min(5, n_pride))
        
        # Recency-weighted significance
        base = signal_count * 0.05
        intensity_component = (avg_frustration + avg_pride) * 0.3
        
        return min(1.0, base + intensity_component)


class ReturnVectorTracker:
    """
    Tracks post-disruption behavior to identify automatic restarts (values)
    vs effortful restarts (goals).
    
    Cleanest signal: values restart automatically without planning.
    """
    
    def __init__(self):
        self.disruption_history: Dict[str, List[Dict]] = defaultdict(list)
    
    async def track_recovery(
        self,
        user_id: str,
        disruption_event: Dict[str, Any],
        subsequent_observations: List[Dict[str, Any]]
    ) -> Optional[ReturnVector]:
        """
        Track what behavior resumes after disruption.
        """
        disruption_type = disruption_event.get('type', 'unknown')
        disrupted_goal = disruption_event.get('goal_id')
        disrupted_behavior = disruption_event.get('behavior_pattern', '')
        
        # Analyze subsequent observations for recovery pattern
        if not subsequent_observations:
            return None
        
        # Find first significant behavior
        first_behavior = subsequent_observations[0].get('behavior', {})
        actual_resumption = first_behavior.get('primary_action', '')
        resumption_delay = self._calculate_delay(
            disruption_event.get('timestamp'),
            subsequent_observations[0].get('timestamp')
        )
        
        # Determine if restart was automatic
        automatic = self._assess_automatic_restart(
            first_behavior,
            subsequent_observations[0].get('ema_response', {})
        )
        
        # Infer value from resumption pattern
        inferred_value, confidence = self._infer_value_from_resumption(
            actual_resumption,
            disrupted_behavior,
            automatic
        )
        
        vector = ReturnVector(
            user_id=user_id,
            disruption_event_id=disruption_event.get('event_id', 'unknown'),
            disruption_type=disruption_type,
            disrupted_goal=disrupted_goal,
            disrupted_behavior=disrupted_behavior,
            planned_resumption=disruption_event.get('intended_resumption'),
            actual_resumption=actual_resumption,
            resumption_delay_hours=resumption_delay,
            automatic_restart=automatic,
            required_effort_to_restart=0.0 if automatic else 0.7,
            inferred_value=inferred_value,
            confidence=confidence
        )
        
        self.disruption_history[user_id].append({
            'vector': vector,
            'timestamp': datetime.utcnow()
        })
        
        return vector
    
    def _calculate_delay(self, disruption_ts: Any, recovery_ts: Any) -> float:
        """Calculate delay in hours between disruption and recovery."""
        try:
            if isinstance(disruption_ts, str):
                disruption = datetime.fromisoformat(disruption_ts.replace('Z', '+00:00'))
            else:
                disruption = disruption_ts or datetime.utcnow()
            
            if isinstance(recovery_ts, str):
                recovery = datetime.fromisoformat(recovery_ts.replace('Z', '+00:00'))
            else:
                recovery = recovery_ts or datetime.utcnow()
            
            diff = (recovery - disruption).total_seconds() / 3600
            return max(0, diff)
        except:
            return 0
    
    def _assess_automatic_restart(self, behavior: Dict, ema: Dict) -> bool:
        """
        Assess if behavior restart was automatic (value) or effortful (goal).
        """
        # Automatic restarts show:
        # - No conscious planning language
        # - Immediate action without deliberation
        # - Emotional ease (not dread)
        
        text = json.dumps({**behavior, **ema}).lower()
        
        automatic_markers = [
            'just did', 'found myself', 'automatically', 'naturally',
            'without thinking', 'habit', 'routine'
        ]
        
        effortful_markers = [
            'forced myself', 'made myself', 'finally', 'after procrastinating',
            'struggled to', 'hard to start', 'dreaded'
        ]
        
        auto_score = sum(1 for m in automatic_markers if m in text)
        effort_score = sum(1 for m in effortful_markers if m in text)
        
        # Also check timing - fast restart suggests automatic
        if behavior.get('time_since_disruption_hours', 24) < 1:
            auto_score += 1
        
        return auto_score > effort_score
    
    def _infer_value_from_resumption(
        self,
        actual_resumption: str,
        disrupted_behavior: str,
        automatic: bool
    ) -> Tuple[Optional[ValueCategory], float]:
        """Infer which value drives automatic resumption."""
        if not automatic:
            return None, 0.3
        
        # Map resumption patterns to values
        resumption_value_map = {
            'work': ValueCategory.RESPONSIBILITY,
            'exercise': ValueCategory.MASTERY,
            'create': ValueCategory.MEANING,
            'socialize': ValueCategory.CONNECTION,
            'learn': ValueCategory.MASTERY,
            'organize': ValueCategory.SECURITY,
            'help': ValueCategory.CONNECTION,
            'reflect': ValueCategory.MEANING,
            'independent': ValueCategory.AUTONOMY,
            'rest': ValueCategory.PLEASURE,
        }
        
        text = (actual_resumption + ' ' + disrupted_behavior).lower()
        
        for pattern, value in resumption_value_map.items():
            if pattern in text:
                return value, 0.7
        
        return None, 0.2


class ValuesInferenceModule:
    """
    Main orchestrator for values inference.
    Integrates sacrifice detection, emotional tracking, and return vectors.
    """
    
    def __init__(self):
        self.sacrifice_detector = SacrificeDetector()
        self.emotional_tracker = EmotionalEnergyTracker()
        self.return_tracker = ReturnVectorTracker()
        
        # User value profiles
        self.inferred_values: Dict[str, Dict[ValueCategory, InferredValue]] = {}
    
    async def process_observation(
        self,
        observation: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """
        Process observation through all VIM detectors.
        Returns updated value inferences.
        """
        # Run all detectors
        sacrifices = await self.sacrifice_detector.detect_from_observation(observation, user_id)
        emotional_updates = await self.emotional_tracker.update_from_observation(observation, user_id)
        
        # Update value inferences
        for sacrifice in sacrifices:
            await self._integrate_sacrifice(sacrifice, user_id)
        
        for gradient in emotional_updates:
            await self._integrate_emotional_gradient(gradient, user_id)
        
        # Generate summary
        user_values = self.inferred_values.get(user_id, {})
        
        return {
            "user_id": user_id,
            "observation_processed": observation.get('observation_id'),
            "new_sacrifices_detected": len(sacrifices),
            "emotional_gradients_updated": len(emotional_updates),
            "current_value_profile": {
                v.value_category.value: {
                    "strength": v.strength,
                    "confidence": v.calculate_confidence(),
                    "status": v.status,
                    "n_observations": v.n_observations
                }
                for v in user_values.values()
            },
            "dominant_values": self._get_dominant_values(user_id, top_n=3),
            "value_conflicts_detected": self._detect_value_conflicts(user_id)
        }
    
    async def process_recovery_pattern(
        self,
        user_id: str,
        disruption_event: Dict[str, Any],
        subsequent_observations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process post-disruption recovery for return vector analysis."""
        vector = await self.return_tracker.track_recovery(
            user_id, disruption_event, subsequent_observations
        )
        
        if vector and vector.inferred_value:
            await self._integrate_return_vector(vector, user_id)
        
        return {
            "return_vector_recorded": vector is not None,
            "inferred_value": vector.inferred_value.value if vector and vector.inferred_value else None,
            "automatic_restart": vector.automatic_restart if vector else None,
            "confidence": vector.confidence if vector else 0
        }
    
    async def _integrate_sacrifice(self, sacrifice: SacrificeEvent, user_id: str):
        """Integrate sacrifice event into value inference."""
        value_cat = sacrifice.protected_value
        
        if user_id not in self.inferred_values:
            self.inferred_values[user_id] = {}
        
        if value_cat not in self.inferred_values[user_id]:
            self.inferred_values[user_id][value_cat] = InferredValue(
                user_id=user_id,
                value_category=value_cat
            )
        
        inferred = self.inferred_values[user_id][value_cat]
        inferred.sacrifice_events.append(sacrifice)
        inferred.n_observations += 1
        inferred.last_confirmed = datetime.utcnow()
        
        # Update cost willingness
        if sacrifice.repeated_pattern:
            inferred.cost_willingness_score = min(1.0, inferred.cost_willingness_score + 0.15)
        
        # Update status based on evidence accumulation
        if inferred.n_observations >= 5 and inferred.calculate_confidence() > 0.7:
            inferred.status = "confirmed"
        elif inferred.n_observations >= 3 and inferred.calculate_confidence() > 0.5:
            inferred.status = "emerging"
        
        # Recalculate strength
        inferred.strength = inferred.calculate_confidence()
    
    async def _integrate_emotional_gradient(
        self,
        gradient: EmotionalGradient,
        user_id: str
    ):
        """Integrate emotional gradient into value inference."""
        value_cat = gradient.value_category
        
        if user_id not in self.inferred_values:
            self.inferred_values[user_id] = {}
        
        if value_cat not in self.inferred_values[user_id]:
            self.inferred_values[user_id][value_cat] = InferredValue(
                user_id=user_id,
                value_category=value_cat
            )
        
        inferred = self.inferred_values[user_id][value_cat]
        inferred.emotional_gradients.append(gradient)
        inferred.n_observations += 1
        
        # Significant emotional investment increases confidence
        if gradient.significance_score > 0.6:
            inferred.stability_score = min(1.0, inferred.stability_score + 0.1)
    
    async def _integrate_return_vector(self, vector: ReturnVector, user_id: str):
        """Integrate return vector into value inference."""
        if not vector.inferred_value:
            return
        
        value_cat = vector.inferred_value
        
        if user_id not in self.inferred_values:
            self.inferred_values[user_id] = {}
        
        if value_cat not in self.inferred_values[user_id]:
            self.inferred_values[user_id][value_cat] = InferredValue(
                user_id=user_id,
                value_category=value_cat
            )
        
        inferred = self.inferred_values[user_id][value_cat]
        inferred.return_vectors.append(vector)
        inferred.n_observations += 1
        
        # Automatic restart is strong evidence for value
        if vector.automatic_restart and vector.confidence > 0.6:
            inferred.strength = min(1.0, inferred.strength + 0.2)
    
    def _get_dominant_values(self, user_id: str, top_n: int = 3) -> List[Dict]:
        """Get top N dominant values for user."""
        user_values = self.inferred_values.get(user_id, {})
        
        sorted_values = sorted(
            user_values.values(),
            key=lambda v: (v.calculate_confidence(), v.strength),
            reverse=True
        )
        
        return [
            {
                "value": v.value_category.value,
                "strength": v.strength,
                "confidence": v.calculate_confidence(),
                "status": v.status,
                "evidence_count": v.n_observations
            }
            for v in sorted_values[:top_n]
        ]
    
    def _detect_value_conflicts(self, user_id: str) -> List[Dict]:
        """Detect potential conflicts between user's values."""
        user_values = self.inferred_values.get(user_id, {})
        conflicts = []
        
        # Known conflict pairs
        conflict_pairs = [
            (ValueCategory.AUTONOMY, ValueCategory.CONNECTION),
            (ValueCategory.SECURITY, ValueCategory.MASTERY),
            (ValueCategory.PLEASURE, ValueCategory.RESPONSIBILITY),
            (ValueCategory.PRESENT_EASE, ValueCategory.LEGACY),
        ]
        
        for v1, v2 in conflict_pairs:
            if v1 in user_values and v2 in user_values:
                conf1 = user_values[v1].calculate_confidence()
                conf2 = user_values[v2].calculate_confidence()
                
                # Both strong = potential conflict
                if conf1 > 0.6 and conf2 > 0.6:
                    conflicts.append({
                        "value_a": v1.value,
                        "value_b": v2.value,
                        "conflict_type": "structural_tension",
                        "both_strength": (conf1 + conf2) / 2,
                        "resolution_strategy": "context_dependent_prioritization"
                    })
        
        return conflicts
    
    def get_value_profile(self, user_id: str) -> Dict[str, Any]:
        """Get complete value profile for user."""
        user_values = self.inferred_values.get(user_id, {})
        
        return {
            "user_id": user_id,
            "profile_generated_at": datetime.utcnow().isoformat(),
            "n_values_inferred": len(user_values),
            "values": {
                v.value_category.value: {
                    "category": v.value_category.value,
                    "strength": v.strength,
                    "confidence": v.calculate_confidence(),
                    "status": v.status,
                    "first_inferred": v.first_inferred.isoformat(),
                    "last_confirmed": v.last_confirmed.isoformat(),
                    "n_sacrifice_events": len(v.sacrifice_events),
                    "n_emotional_signals": len(v.emotional_gradients),
                    "n_return_vectors": len(v.return_vectors),
                    "cost_willingness": v.cost_willingness_score,
                    "stability": v.stability_score
                }
                for v in user_values.values()
            },
            "dominant_values": self._get_dominant_values(user_id, top_n=5),
            "value_conflicts": self._detect_value_conflicts(user_id),
            "inference_reliability": self._assess_reliability(user_id)
        }
    
    def _assess_reliability(self, user_id: str) -> Dict[str, Any]:
        """Assess reliability of value inferences for user."""
        user_values = self.inferred_values.get(user_id, {})
        
        total_observations = sum(v.n_observations for v in user_values.values())
        avg_confidence = sum(v.calculate_confidence() for v in user_values.values()) / max(1, len(user_values))
        
        if total_observations < 10:
            reliability = "low"
        elif total_observations < 30:
            reliability = "moderate"
        else:
            reliability = "high"
        
        return {
            "level": reliability,
            "total_observations": total_observations,
            "avg_confidence": avg_confidence,
            "n_distinct_values": len(user_values),
            "recommendation": "continue_observation" if reliability != "high" else "sufficient_for_design"
        }
    
    def get_experiment_design_guidance(
        self,
        user_id: str,
        proposed_goal: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate guidance for experiment design based on inferred values.
        This is how VIM influences the system - through design, not recommendations.
        """
        profile = self.get_value_profile(user_id)
        dominant = profile.get("dominant_values", [])
        conflicts = profile.get("value_conflicts", [])
        
        guidance = {
            "user_id": user_id,
            "design_principles": [],
            "framing_suggestions": [],
            "tradeoffs_to_test": [],
            "avoid": []
        }
        
        # Design principles based on dominant values
        for val in dominant[:3]:
            v = val["value"]
            
            if v == "autonomy":
                guidance["design_principles"].append("Preserve user agency - offer choices, not directives")
                guidance["framing_suggestions"].append("Frame as 'your choice' rather than 'should do'")
                guidance["tradeoffs_to_test"].append("autonomy_vs_efficiency")
            
            elif v == "responsibility":
                guidance["design_principles"].append("Connect to duty and commitment")
                guidance["framing_suggestions"].append("Emphasize impact on others who depend on user")
                guidance["tradeoffs_to_test"].append("responsibility_vs_rest")
            
            elif v == "meaning":
                guidance["design_principles"].append("Connect to purpose and significance")
                guidance["framing_suggestions"].append("Show how action serves deeper purpose")
                guidance["tradeoffs_to_test"].append("meaning_vs_comfort")
            
            elif v == "dignity":
                guidance["design_principles"].append("Protect from shame exposure")
                guidance["framing_suggestions"].append("Normalize difficulty, emphasize growth")
                guidance["avoid"].append("public_accountability_mechanisms")
            
            elif v == "mastery":
                guidance["design_principles"].append("Emphasize growth and skill development")
                guidance["framing_suggestions"].append("Frame as challenge to overcome")
                guidance["tradeoffs_to_test"].append("mastery_vs_completion_speed")
        
        # Warn about conflicts
        if conflicts:
            guidance["design_principles"].append("Be aware of value conflicts - design for context-dependent expression")
            for conflict in conflicts:
                guidance["avoid"].append(f"forcing_choice_between_{conflict['value_a']}_and_{conflict['value_b']}")
        
        return guidance
    
    def get_intervention_framing(
        self,
        user_id: str,
        intervention_type: str
    ) -> Dict[str, Any]:
        """
        Generate value-aligned framing for intervention.
        VIM shapes how things are said, not what is recommended.
        """
        profile = self.get_value_profile(user_id)
        dominant = [v["value"] for v in profile.get("dominant_values", [])]
        
        if not dominant:
            return {
                "user_id": user_id,
                "framing": "neutral",
                "rationale": "insufficient_value_data"
            }
        
        primary = dominant[0]
        
        # Value-aligned framings
        framings = {
            "autonomy": {
                "frame": "self_direction",
                "language": ["you choose", "your decision", "what works for you", "your path"],
                "avoid": ["should", "must", "required", "obligation"]
            },
            "responsibility": {
                "frame": "commitment",
                "language": ["your commitment", "others counting on you", "follow through", "reliability"],
                "avoid": ["optional", "if you feel like it", "when convenient"]
            },
            "meaning": {
                "frame": "purpose",
                "language": ["what matters to you", "deeper purpose", "significance", "why this counts"],
                "avoid": ["just do it", "get it done", "check the box"]
            },
            "dignity": {
                "frame": "growth",
                "language": ["developing", "becoming", "growth", "learning"],
                "avoid": ["fix", "problem", "failure", "inadequate"]
            },
            "mastery": {
                "frame": "challenge",
                "language": ["level up", "skill", "mastery", "challenge", "expertise"],
                "avoid": ["easy", "simple", "quick", "shortcut"]
            },
            "connection": {
                "frame": "belonging",
                "language": ["together", "shared", "community", "support"],
                "avoid": ["alone", "independent", "self-sufficient"]
            },
            "security": {
                "frame": "stability",
                "language": ["solid foundation", "reliable", "steady", "secure"],
                "avoid": ["risk", "uncertain", "unknown", "unpredictable"]
            }
        }
        
        framing = framings.get(primary, {
            "frame": "neutral",
            "language": [],
            "avoid": []
        })
        
        return {
            "user_id": user_id,
            "primary_value": primary,
            "framing": framing["frame"],
            "suggested_language": framing["language"][:3],
            "language_to_avoid": framing["avoid"][:3],
            "secondary_values": dominant[1:3] if len(dominant) > 1 else [],
            "rationale": f"Aligned with inferred primary value: {primary}"
        }


# Global singleton
_vim_instance: Optional[ValuesInferenceModule] = None

def get_values_inference_module() -> ValuesInferenceModule:
    """Get or create global VIM instance."""
    global _vim_instance
    if _vim_instance is None:
        _vim_instance = ValuesInferenceModule()
    return _vim_instance
'''

print(f"VIM Core created: {len(vim_core_code)} bytes")
