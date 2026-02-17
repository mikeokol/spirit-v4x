"""
Ethical Oversight: HRV monitoring, burnout detection, kill switch.
Protects the human subject from optimization harm.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional
from enum import Enum

from spirit.db.supabase_client import get_behavioral_store
from spirit.services.notification_engine import NotificationEngine, NotificationPriority


class RiskLevel(Enum):
    GREEN = "green"      # Normal
    YELLOW = "yellow"    # Elevated stress, monitor
    ORANGE = "orange"    # High risk, reduce interventions
    RED = "red"          # Critical, pause all experiments


class EthicalOversight:
    """
    Continuous monitoring of user's wellbeing.
    Implements "First, do no harm" principle.
    """
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.current_risk_level = RiskLevel.GREEN
        self.intervention_pause_until: Optional[datetime] = None
        
        # Thresholds
        self.hrv_low_threshold = 25  # ms RMSSD (very simplified)
        self.stress_streak_hours = 4
        self.max_daily_interventions = 8
    
    async def assess_wellbeing(self) -> RiskLevel:
        """
        Assess current wellbeing from available data.
        """
        # Collect indicators
        indicators = {
            'hrv': await self._get_hrv_data(),
            'sleep': await self._get_sleep_quality(),
            'intervention_burden': await self._calculate_intervention_burden(),
            'user_responsiveness': await self._get_responsiveness_trend(),
            'explicit_feedback': await self._get_recent_feedback()
        }
        
        # Calculate risk
        risk_score = 0
        
        # HRV low for extended period
        if indicators['hrv'] and indicators['hrv'] < self.hrv_low_threshold:
            risk_score += 2
        
        # Too many interventions today
        if indicators['intervention_burden'] > self.max_daily_interventions:
            risk_score += 2
        
        # User ignoring/not responding (rejection)
        if indicators['user_responsiveness'] < 0.3:
            risk_score += 1
        
        # Explicit negative feedback
        if indicators['explicit_feedback'] and indicators['explicit_feedback'] < 3:
            risk_score += 2
        
        # Map to risk level
        if risk_score >= 4:
            new_level = RiskLevel.RED
        elif risk_score >= 2:
            new_level = RiskLevel.ORANGE
        elif risk_score >= 1:
            new_level = RiskLevel.YELLOW
        else:
            new_level = RiskLevel.GREEN
        
        # Handle state transitions
        if new_level != self.current_risk_level:
            await self._handle_risk_transition(new_level, indicators)
        
        self.current_risk_level = new_level
        return new_level
    
    async def check_intervention_permitted(self, intervention_type: str) -> bool:
        """
        Check if intervention is ethically permissible now.
        """
        # Check pause
        if self.intervention_pause_until and datetime.utcnow() < self.intervention_pause_until:
            return False
        
        # Check risk level
        if self.current_risk_level == RiskLevel.RED:
            return intervention_type == 'wellbeing_check'  # Only wellness allowed
        
        if self.current_risk_level == RiskLevel.ORANGE:
            # Reduce to essential only
            return intervention_type in ['wellbeing_check', 'critical_alert']
        
        return True
    
    async def _handle_risk_transition(self, new_level: RiskLevel, indicators: Dict):
        """Handle change in risk level."""
        store = await get_behavioral_store()
        
        # Log transition
        if store:
            store.client.table('ethical_events').insert({
                'user_id': str(self.user_id),
                'event_type': 'risk_transition',
                'from_level': self.current_risk_level.value,
                'to_level': new_level.value,
                'indicators': indicators,
                'timestamp': datetime.utcnow().isoformat()
            }).execute()
        
        # Take action
        if new_level == RiskLevel.RED:
            # PAUSE EVERYTHING
            self.intervention_pause_until = datetime.utcnow() + timedelta(hours=24)
            
            # Notify user (gently)
            engine = NotificationEngine(self.user_id)
            await engine.send_notification(
                content={
                    "title": "Taking a break",
                    "body": "I noticed you might be feeling overwhelmed. I'll pause check-ins for 24 hours. Take care of yourself.",
                    "data": {"type": "ethical_pause", "duration_hours": 24}
                },
                priority=NotificationPriority.HIGH,
                notification_type="ethical_pause_activated"
            )
            
            # Alert researcher/admin if available
            await self._alert_researcher("CRITICAL: User wellbeing risk detected")
            
        elif new_level == RiskLevel.ORANGE:
            # Reduce interventions
            self.intervention_pause_until = datetime.utcnow() + timedelta(hours=4)
            
            # Gentle notification
            engine = NotificationEngine(self.user_id)
            await engine.send_notification(
                content={
                    "title": "Adjusting pace",
                    "body": "I'll reduce check-ins for a few hours to give you space.",
                    "data": {"type": "pace_reduction"}
                },
                priority=NotificationPriority.NORMAL,
                notification_type="pace_reduction"
            )
    
    async def _get_hrv_data(self) -> Optional[float]:
        """Get HRV from health data integration."""
        # Would query Apple Health, Whoop, etc.
        # For now, placeholder
        store = await get_behavioral_store()
        if not store:
            return None
        
        # Check for recent HRV
        recent = store.client.table('health_data').select('*').eq(
            'user_id', str(self.user_id)
        ).eq('metric_type', 'hrv').order('timestamp', desc=True).limit(1).execute()
        
        if recent.data:
            return recent.data[0].get('value')
        return None
    
    async def _get_sleep_quality(self) -> Optional[float]:
        """Get recent sleep quality."""
        # Placeholder
        return None
    
    async def _calculate_intervention_burden(self) -> int:
        """Count interventions in last 24 hours."""
        store = await get_behavioral_store()
        if not store:
            return 0
        
        since = (datetime.utcnow() - timedelta(hours=24)).isoformat()
        
        count = store.client.table('proactive_interventions').select('*', count='exact').eq(
            'user_id', str(self.user_id)
        ).gte('executed_at', since).execute()
        
        return count.count if hasattr(count, 'count') else 0
    
    async def _get_responsiveness_trend(self) -> float:
        """Calculate recent response rate to interventions."""
        store = await get_behavioral_store()
        if not store:
            return 1.0
        
        since = (datetime.utcnow() - timedelta(days=3)).isoformat()
        
        interventions = store.client.table('proactive_interventions').select('*').eq(
            'user_id', str(self.user_id)
        ).gte('executed_at', since).execute()
        
        if not interventions.data:
            return 1.0
        
        responded = sum(1 for i in interventions.data if i.get('user_responded'))
        return responded / len(interventions.data)
    
    async def _get_recent_feedback(self) -> Optional[float]:
        """Get recent explicit feedback rating."""
        store = await get_behavioral_store()
        if not store:
            return None
        
        recent = store.client.table('intervention_outcomes').select('*').eq(
            'user_id', str(self.user_id)
        ).not_.is_('explicit_rating', None).order('responded_at', desc=True).limit(1).execute()
        
        if recent.data:
            return recent.data[0].get('explicit_rating')
        return None
    
    async def _alert_researcher(self, message: str):
        """Alert human researcher of ethical concern."""
        # Would send email/Slack to study PI
        print(f"ETHICAL ALERT for user {self.user_id}: {message}")
    
    async def manual_kill_switch(self, reason: str):
        """
        Manual kill switch - user or researcher can trigger.
        """
        self.current_risk_level = RiskLevel.RED
        self.intervention_pause_until = datetime.utcnow() + timedelta(days=7)
        
        store = await get_behavioral_store()
        if store:
            store.client.table('ethical_events').insert({
                'user_id': str(self.user_id),
                'event_type': 'manual_kill_switch',
                'reason': reason,
                'paused_until': self.intervention_pause_until.isoformat(),
                'timestamp': datetime.utcnow().isoformat()
            }).execute()
        
        return {
            "activated": True,
            "paused_until": self.intervention_pause_until.isoformat(),
            "message": "All interventions paused. Your wellbeing comes first."
        }


# Global oversight registry
_oversight_registry: Dict[int, EthicalOversight] = {}


def get_ethical_oversight(user_id: int) -> EthicalOversight:
    """Get or create oversight for user."""
    if user_id not in _oversight_registry:
        _oversight_registry[user_id] = EthicalOversight(user_id)
    return _oversight_registry[user_id]


# Hook into all intervention paths
async def ethical_check(user_id: int, intervention_type: str) -> bool:
    """
    Check if intervention is ethically permissible.
    Call this before ANY intervention.
    """
    oversight = get_ethical_oversight(user_id)
    
    # Update assessment
    await oversight.assess_wellbeing()
    
    # Check permission
    permitted = await oversight.check_intervention_permitted(intervention_type)
    
    if not permitted:
        print(f"BLOCKED: {intervention_type} for user {user_id} due to ethical pause")
    
    return permitted
