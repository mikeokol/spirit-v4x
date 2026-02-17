"""
Belief Network: Explicit model of user's internal justifications.
Bayesian belief tracking with cognitive dissonance detection.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import math

from spirit.db.supabase_client import get_behavioral_store
from spirit.streaming.realtime_pipeline import RealTimeEvent, get_stream_processor


class BeliefType(Enum):
    CAUSAL = "causal"           # "X causes Y"
    IDENTITY = "identity"       # "I am a night person"
    STRATEGY = "strategy"       # "This approach works for me"
    CONSTRAINT = "constraint"   # "I can't do X because Y"


@dataclass
class UserBelief:
    """
    A belief held by the user, with confidence and evidence tracking.
    """
    belief_id: str
    user_id: int
    belief_type: BeliefType
    statement: str  # Natural language: "I work best at night"
    
    # Bayesian tracking
    prior_probability: float  # Initial confidence (0-1)
    posterior_probability: float  # Updated based on evidence
    evidence_for: List[Dict]  # Observations supporting
    evidence_against: List[Dict]  # Observations contradicting
    
    # Metadata
    first_stated: datetime
    last_referenced: datetime
    times_tested: int
    currently_held: bool  # User may have abandoned it
    
    # Linked to data
    related_hypothesis_ids: List[str]
    contradictions_detected: int


class CognitiveDissonanceDetector:
    """
    Detects gaps between user beliefs and observed behavior.
    Triggers justification queries when dissonance found.
    """
    
    def __init__(self):
        self.dissonance_threshold = 0.3  # 30% gap triggers query
        self.recent_checks: Dict[int, datetime] = {}  # Rate limiting
    
    async def check_dissonance(
        self,
        user_id: int,
        observation: Dict,
        user_beliefs: List[UserBelief]
    ) -> Optional[Dict]:
        """
        Check if observation contradicts any user beliefs.
        Returns dissonance report if found.
        """
        # Rate limit: max 1 check per minute per user
        last_check = self.recent_checks.get(user_id)
        if last_check and (datetime.utcnow() - last_check).seconds < 60:
            return None
        
        self.recent_checks[user_id] = datetime.utcnow()
        
        dissonances = []
        
        for belief in user_beliefs:
            if not belief.currently_held:
                continue
            
            # Check if observation contradicts this belief
            contradiction = self._evaluate_contradiction(belief, observation)
            
            if contradiction['strength'] > self.dissonance_threshold:
                dissonances.append({
                    'belief': belief,
                    'contradiction': contradiction,
                    'severity': self._calculate_severity(belief, contradiction)
                })
        
        if not dissonances:
            return None
        
        # Return strongest dissonance
        strongest = max(dissonances, key=lambda d: d['severity'])
        
        return {
            'detected': True,
            'belief_statement': strongest['belief'].statement,
            'belief_confidence': strongest['belief'].posterior_probability,
            'observed_behavior': self._extract_behavior_summary(observation),
            'contradiction_strength': strongest['contradiction']['strength'],
            'severity': strongest['severity'],
            'query_triggered': strongest['severity'] > 0.7,
            'suggested_query': self._generate_justification_query(strongest)
        }
    
    def _evaluate_contradiction(self, belief: UserBelief, observation: Dict) -> Dict:
        """
        Evaluate how much observation contradicts belief.
        """
        behavior = observation.get('behavior', {})
        context = observation.get('context', {})
        
        # Parse belief statement (simplified NLP)
        belief_text = belief.statement.lower()
        
        # Belief: "I work best at night"
        if 'night' in belief_text and 'work' in belief_text:
            hour = observation.get('timestamp', datetime.utcnow())
            if isinstance(hour, str):
                hour = datetime.fromisoformat(hour.replace('Z', '+00:00')).hour
            
            productivity = behavior.get('productive_time_minutes', 0)
            
            if hour >= 21 and productivity < 30:  # Low productivity at night
                return {
                    'strength': 0.8,
                    'expected': 'high_productivity',
                    'observed': f'low_productivity ({productivity}min)',
                    'feature': 'night_productivity'
                }
        
        # Belief: "I need coffee to focus"
        if 'coffee' in belief_text:
            # Check if focus exists without coffee
            had_coffee = context.get('consumed_coffee', False)
            focus_score = behavior.get('focus_score', 0)
            
            if not had_coffee and focus_score > 0.7:
                return {
                    'strength': 0.6,
                    'expected': 'low_focus_without_coffee',
                    'observed': f'high_focus ({focus_score}) without coffee',
                    'feature': 'coffee_independence'
                }
        
        # Belief: "Social media helps me relax"
        if 'social' in belief_text and 'relax' in belief_text:
            app = behavior.get('app_category')
            post_usage_stress = behavior.get('stress_indicator', 0)
            
            if app == 'social_media' and post_usage_stress > 0.6:
                return {
                    'strength': 0.7,
                    'expected': 'reduced_stress',
                    'observed': f'increased_stress ({post_usage_stress})',
                    'feature': 'social_media_stress'
                }
        
        return {'strength': 0.0}
    
    def _calculate_severity(self, belief: UserBelief, contradiction: Dict) -> float:
        """Calculate severity of dissonance."""
        # Higher severity if:
        # - Belief is strongly held (high posterior)
        # - Contradiction is strong
        # - Belief is central to identity
        
        base = contradiction['strength'] * belief.posterior_probability
        
        # Identity beliefs are more severe
        if belief.belief_type == BeliefType.IDENTITY:
            base *= 1.3
        
        # Beliefs tested many times and still held are more entrenched
        if belief.times_tested > 3:
            base *= 1.2
        
        return min(1.0, base)
    
    def _generate_justification_query(self, dissonance: Dict) -> str:
        """Generate query to capture user's justification."""
        belief = dissonance['belief']
        
        queries = {
            BeliefType.IDENTITY: f"I noticed you said '{belief.statement}', but today's data shows something different. Help me understand—what's going on?",
            BeliefType.CAUSAL: f"You mentioned that {belief.statement}. I observed {dissonance['contradiction']['observed']}. Was this an exception, or has something changed?",
            BeliefType.STRATEGY: f"Your usual approach is '{belief.statement}', but today you did something else. Was that intentional?",
            BeliefType.CONSTRAINT: f"I thought you couldn't because '{belief.statement}', but you did. What enabled that?"
        }
        
        return queries.get(belief.belief_type, "I noticed something interesting. Can you help me understand?")
    
    def _extract_behavior_summary(self, observation: Dict) -> str:
        """Extract human-readable behavior summary."""
        behavior = observation.get('behavior', {})
        parts = []
        
        if 'app_category' in behavior:
            parts.append(f"using {behavior['app_category']}")
        if 'session_duration_sec' in behavior:
            mins = behavior['session_duration_sec'] / 60
            parts.append(f"for {mins:.0f} minutes")
        if 'focus_score' in behavior:
            parts.append(f"with focus {behavior['focus_score']:.1f}")
        
        return " ".join(parts) if parts else "unknown activity"


class BayesianBeliefNetwork:
    """
    Maintains and updates beliefs using Bayesian inference.
    """
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.beliefs: Dict[str, UserBelief] = {}
    
    async def add_belief(
        self,
        statement: str,
        belief_type: BeliefType,
        initial_confidence: float = 0.7
    ) -> UserBelief:
        """Add a new belief stated by user."""
        belief = UserBelief(
            belief_id=f"belief_{datetime.utcnow().timestamp()}",
            user_id=self.user_id,
            belief_type=belief_type,
            statement=statement,
            prior_probability=initial_confidence,
            posterior_probability=initial_confidence,
            evidence_for=[],
            evidence_against=[],
            first_stated=datetime.utcnow(),
            last_referenced=datetime.utcnow(),
            times_tested=0,
            currently_held=True,
            related_hypothesis_ids=[],
            contradictions_detected=0
        )
        
        self.beliefs[belief.statement] = belief
        await self._persist_belief(belief)
        
        return belief
    
    async def update_from_evidence(
        self,
        belief_statement: str,
        evidence: Dict,
        supports: bool
    ):
        """
        Update belief using Bayes' rule.
        P(Belief|Evidence) ∝ P(Evidence|Belief) * P(Belief)
        """
        belief = self.beliefs.get(belief_statement)
        if not belief:
            return
        
        # Likelihood ratio (simplified)
        # If supports: likelihood = 0.8 (strong evidence)
        # If contradicts: likelihood = 0.2 (weak evidence for belief)
        likelihood = 0.8 if supports else 0.2
        
        # Bayes update
        prior = belief.posterior_probability
        posterior = (likelihood * prior) / (
            (likelihood * prior) + ((1 - likelihood) * (1 - prior))
        )
        
        belief.posterior_probability = posterior
        belief.last_referenced = datetime.utcnow()
        
        if supports:
            belief.evidence_for.append({
                'timestamp': datetime.utcnow().isoformat(),
                'evidence': evidence
            })
        else:
            belief.evidence_against.append({
                'timestamp': datetime.utcnow().isoformat(),
                'evidence': evidence
            })
            belief.contradictions_detected += 1
            
            # If too many contradictions, flag for review
            if belief.contradictions_detected > 5 and posterior < 0.3:
                belief.currently_held = False
        
        await self._persist_belief(belief)
    
    async def get_beliefs_for_testing(self) -> List[UserBelief]:
        """
        Get beliefs that should be actively tested.
        """
        testable = []
        
        for belief in self.beliefs.values():
            if not belief.currently_held:
                continue
            
            # Test if:
            # - High confidence but few tests (overconfidence)
            # - Recently contradicted
            # - Central to identity but untested
            
            should_test = (
                (belief.posterior_probability > 0.8 and belief.times_tested < 3) or
                (belief.contradictions_detected > 0 and belief.posterior_probability > 0.5) or
                (belief.belief_type == BeliefType.IDENTITY and belief.times_tested < 5)
            )
            
            if should_test:
                testable.append(belief)
        
        return sorted(testable, key=lambda b: b.posterior_probability, reverse=True)
    
    async def _persist_belief(self, belief: UserBelief):
        """Save belief to database."""
        store = await get_behavioral_store()
        if not store:
            return
        
        store.client.table('user_beliefs').upsert({
            'belief_id': belief.belief_id,
            'user_id': str(self.user_id),
            'belief_type': belief.belief_type.value,
            'statement': belief.statement,
            'posterior_probability': belief.posterior_probability,
            'evidence_for': belief.evidence_for,
            'evidence_against': belief.evidence_against,
            'currently_held': belief.currently_held,
            'contradictions_detected': belief.contradictions_detected,
            'updated_at': datetime.utcnow().isoformat()
        }, on_conflict='belief_id').execute()


# Hook into real-time pipeline
async def setup_belief_detection():
    """Register belief detection with stream processor."""
    processor = get_stream_processor()
    detector = CognitiveDissonanceDetector()
    
    async def belief_handler(event: RealTimeEvent):
        """Handle real-time events for belief checking."""
        # Load user beliefs
        network = BayesianBeliefNetwork(event.user_id)
        # Would load existing beliefs from DB
        
        # Check for dissonance
        dissonance = await detector.check_dissonance(
            event.user_id,
            event.raw_observation,
            list(network.beliefs.values())
        )
        
        if dissonance and dissonance.get('query_triggered'):
            # Trigger immediate EMA to capture justification
            from spirit.services.notification_engine import NotificationEngine
            
            engine = NotificationEngine(event.user_id)
            await engine.send_notification(
                content={
                    "title": "Quick check-in",
                    "body": dissonance['suggested_query'],
                    "data": {
                        "type": "belief_justification",
                        "belief_statement": dissonance['belief_statement'],
                        "dissonance_strength": dissonance['contradiction_strength']
                    }
                },
                priority="high",
                notification_type="cognitive_dissonance_query"
            )
    
    processor.register_handler(belief_handler)
