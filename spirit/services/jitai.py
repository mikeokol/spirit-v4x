"""
Just-In-Time Adaptive Intervention (JITAI) Engine.
Evaluates behavioral observations and decides when to trigger EMAs or interventions.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from uuid import UUID, uuid4

from spirit.db.supabase_client import get_behavioral_store
from spirit.models.behavioral import (
    BehavioralObservation,
    EMARequest
)


class JITAIEngine:
    """
    Implements HeartSteps-style micro-randomization.
    Decides: WHEN to intervene, WHAT to deliver, and to WHOM.
    """
    
    def __init__(self):
        self.cooldown_tracker: Dict[UUID, datetime] = {}
        self.min_cooldown_minutes = 30
    
    async def evaluate_window(
        self,
        user_id: UUID,
        observation: BehavioralObservation
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate if current moment is a vulnerability window or opportunity.
        """
        # Check cooldown
        last_trigger = self.cooldown_tracker.get(user_id)
        if last_trigger:
            time_since = datetime.utcnow() - last_trigger
            if time_since < timedelta(minutes=self.min_cooldown_minutes):
                return None
        
        # Get recent history
        store = await get_behavioral_store()
        if not store:
            return None
            
        recent_obs = await store.get_user_observations(
            user_id=user_id,
            start_time=(datetime.utcnow() - timedelta(hours=4)).isoformat(),
            limit=50
        )
        
        # State detection
        state = self._detect_state(observation, recent_obs)
        
        if state == "vulnerability_detected":
            return self._create_vulnerability_trigger(observation, recent_obs)
        
        if state == "routine_deviation":
            return self._create_deviation_trigger(observation, recent_obs)
        
        if state == "opportunity_window":
            return self._create_opportunity_trigger(observation, recent_obs)
        
        return None
    
    def _detect_state(
        self,
        current: BehavioralObservation,
        history: list
    ) -> Optional[str]:
        """Detect user's current psychological/behavioral state."""
        behavior = current.behavior
        
        # Vulnerability: High app switching + low engagement
        if behavior.get('app_switches_5min', 0) > 5:
            if behavior.get('avg_session_duration_sec', 0) < 30:
                return "vulnerability_detected"
        
        # Routine deviation
        current_hour = current.timestamp.hour
        historical_apps = [
            obs.behavior.get('app_category') 
            for obs in history 
            if abs((obs.timestamp.hour - current_hour)) < 2
        ]
        current_app = behavior.get('app_category')
        
        if historical_apps and current_app not in historical_apps:
            return "routine_deviation"
        
        # Opportunity
        if behavior.get('session_type') == 'deep_work' and behavior.get('duration_minutes', 0) > 25:
            return "opportunity_window"
        
        return None
    
    def _create_vulnerability_trigger(
        self,
        observation: BehavioralObservation,
        history: list
    ) -> Dict[str, Any]:
        """User appears distracted/overwhelmed."""
        return {
            "trigger_type": "vulnerability_detected",
            "confidence": 0.75,
            "ema_content": {
                "question": "You seem to be switching apps a lot. What's your current focus level?",
                "response_type": "likert_5",
                "options": ["1 - Scattered", "2", "3 - Neutral", "4", "5 - Focused"]
            },
            "timing": "immediate",
            "expiry_minutes": 10
        }
    
    def _create_deviation_trigger(
        self,
        observation: BehavioralObservation,
        history: list
    ) -> Dict[str, Any]:
        """User broke routine."""
        return {
            "trigger_type": "routine_deviation",
            "confidence": 0.6,
            "ema_content": {
                "question": "You're doing something different than usual. Is this intentional?",
                "response_type": "boolean",
                "follow_up": "What's your goal right now?"
            },
            "timing": "delayed_5min",
            "expiry_minutes": 15
        }
    
    def _create_opportunity_trigger(
        self,
        observation: BehavioralObservation,
        history: list
    ) -> Dict[str, Any]:
        """User in good state for intervention."""
        return {
            "trigger_type": "opportunity_window",
            "confidence": 0.8,
            "ema_content": {
                "question": "Great focus session! Ready for a quick micro-break or shall we continue?",
                "response_type": "choice",
                "options": ["5-min walk", "Breathing exercise", "Keep working", "Done for now"]
            },
            "timing": "immediate",
            "expiry_minutes": 5
        }
    
    async def deliver_ema(self, user_id: UUID, trigger: Dict[str, Any]):
        """Deliver EMA to user."""
        self.cooldown_tracker[user_id] = datetime.utcnow()
        
        ema = EMARequest(
            ema_id=uuid4(),
            user_id=user_id,
            triggered_at=datetime.utcnow(),
            trigger_type=trigger['trigger_type'],
            trigger_confidence=trigger['confidence'],
            question_text=trigger['ema_content']['question'],
            response_type=trigger['ema_content']['response_type'],
            response_options=trigger['ema_content'].get('options'),
            expiry_at=datetime.utcnow() + timedelta(minutes=trigger['expiry_minutes'])
        )
        
        # TODO: Store EMA and send push notification
        print(f"JITAI: EMA for {user_id}: {trigger['ema_content']['question'][:50]}...")
