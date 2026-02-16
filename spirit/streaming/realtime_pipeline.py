"""
Real-time Processing Pipeline: Sub-second behavioral analysis.
Processes observations as they arrive, not in batches.
Enables immediate intervention on detected anomalies.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass
from enum import Enum
import json

from spirit.db.supabase_client import get_behavioral_store
from spirit.config import settings


class EventSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class RealTimeEvent:
    """
    A processed event ready for immediate action.
    """
    event_id: str
    user_id: int
    timestamp: datetime
    severity: EventSeverity
    event_type: str  # 'anomaly', 'pattern_match', 'threshold_cross', 'prediction_trigger'
    
    # What triggered it
    raw_observation: Dict
    trigger_feature: str
    trigger_value: float
    baseline_value: float
    deviation_score: float
    
    # What to do
    recommended_action: str
    action_priority: int  # 1-10
    action_window_seconds: int  # How long we have to act
    
    # Context
    recent_context: List[Dict]  # Last 5 minutes for context


class StreamProcessor:
    """
    Processes individual observations in real-time.
    Maintains per-user state windows for immediate anomaly detection.
    """
    
    def __init__(self):
        # Per-user state windows (circular buffers)
        self.user_windows: Dict[int, List[Dict]] = {}
        self.window_size = 20  # Last 20 observations
        
        # Per-user baselines (learned normals)
        self.user_baselines: Dict[int, Dict[str, Dict]] = {}  # feature -> {mean, std, last_updated}
        
        # Registered handlers
        self.handlers: List[Callable[[RealTimeEvent], None]] = []
        
        # Running state
        self.running = False
        self.processing_queue: asyncio.Queue = asyncio.Queue()
    
    def register_handler(self, handler: Callable[[RealTimeEvent], None]):
        """Register a function to handle real-time events."""
        self.handlers.append(handler)
    
    async def start(self):
        """Start the processing loop."""
        self.running = True
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._processing_loop()),
            asyncio.create_task(self._baseline_refresh_loop())
        ]
        
        await asyncio.gather(*tasks)
    
    def stop(self):
        """Stop processing."""
        self.running = False
    
    async def ingest(self, user_id: int, observation: Dict):
        """
        Ingest a single observation for real-time processing.
        Called immediately when mobile sends data.
        """
        await self.processing_queue.put((user_id, observation))
    
    async def _processing_loop(self):
        """Main processing loop."""
        while self.running:
            try:
                # Get with timeout to allow graceful shutdown
                user_id, observation = await asyncio.wait_for(
                    self.processing_queue.get(), 
                    timeout=1.0
                )
                
                # Process immediately
                await self._process_single(user_id, observation)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error in processing loop: {e}")
    
    async def _process_single(self, user_id: int, observation: Dict):
        """
        Process one observation through the real-time pipeline.
        """
        # 1. Update window
        if user_id not in self.user_windows:
            self.user_windows[user_id] = []
        
        window = self.user_windows[user_id]
        window.append(observation)
        if len(window) > self.window_size:
            window.pop(0)
        
        # 2. Check for immediate anomalies (sub-100ms)
        anomaly = self._detect_anomaly(user_id, observation, window)
        if anomaly:
            event = self._create_event(user_id, observation, anomaly, window)
            await self._dispatch_event(event)
            return  # Anomaly detected, handled
        
        # 3. Check pattern matches (known risky sequences)
        pattern = self._detect_pattern(user_id, window)
        if pattern:
            event = self._create_pattern_event(user_id, observation, pattern, window)
            await self._dispatch_event(event)
            return
        
        # 4. Update baselines (background learning)
        await self._update_baseline(user_id, observation)
    
    def _detect_anomaly(
        self, 
        user_id: int, 
        observation: Dict, 
        window: List[Dict]
    ) -> Optional[Dict]:
        """
        Detect if this observation is anomalous for this user.
        Uses statistical z-scores on key features.
        """
        if user_id not in self.user_baselines:
            return None  # No baseline yet
        
        baselines = self.user_baselines[user_id]
        behavior = observation.get('behavior', {})
        
        anomalies = []
        
        # Check key features
        features_to_check = [
            ('session_duration_sec', 300),  # 5 min default
            ('app_switches_5min', 3),
            ('scroll_velocity', 50),
            ('typing_interval_ms', 200)
        ]
        
        for feature, default_val in features_to_check:
            if feature not in behavior:
                continue
            
            if feature not in baselines:
                continue  # No baseline for this feature
            
            baseline = baselines[feature]
            value = behavior[feature]
            
            # Z-score calculation
            mean = baseline.get('mean', default_val)
            std = baseline.get('std', mean * 0.3)  # 30% of mean as default std
            
            if std == 0:
                continue
            
            z_score = abs(value - mean) / std
            
            # Critical: z > 3 (99.7% confidence)
            # Warning: z > 2 (95% confidence)
            if z_score > 3:
                anomalies.append({
                    'feature': feature,
                    'value': value,
                    'expected': mean,
                    'z_score': z_score,
                    'severity': EventSeverity.CRITICAL
                })
            elif z_score > 2.5:
                anomalies.append({
                    'feature': feature,
                    'value': value,
                    'expected': mean,
                    'z_score': z_score,
                    'severity': EventSeverity.WARNING
                })
        
        if anomalies:
            # Return most severe
            return max(anomalies, key=lambda x: x['z_score'])
        
        return None
    
    def _detect_pattern(self, user_id: int, window: List[Dict]) -> Optional[Dict]:
        """
        Detect known risky or opportunity patterns in recent window.
        """
        if len(window) < 5:
            return None
        
        recent = window[-5:]
        
        # Pattern 1: Rapid app switching (distraction cascade)
        switches = [r.get('behavior', {}).get('app_switches_5min', 0) for r in recent]
        if all(s > 5 for s in switches[-3:]):
            return {
                'pattern_name': 'distraction_cascade',
                'confidence': 0.8,
                'description': 'Rapid context switching detected',
                'severity': EventSeverity.WARNING
            }
        
        # Pattern 2: Extended deep work (opportunity)
        durations = [r.get('behavior', {}).get('session_duration_sec', 0) for r in recent]
        types = [r.get('behavior', {}).get('session_type', '') for r in recent]
        
        if all(d > 1500 for d in durations[-3:]) and all(t == 'deep_work' for t in types[-3:]):
            return {
                'pattern_name': 'sustained_flow',
                'confidence': 0.9,
                'description': 'User in sustained deep work state',
                'severity': EventSeverity.INFO,
                'is_opportunity': True
            }
        
        # Pattern 3: Late night usage (sleep risk)
        hour = recent[-1].get('timestamp', datetime.utcnow().isoformat())
        if isinstance(hour, str):
            hour = datetime.fromisoformat(hour.replace('Z', '+00:00')).hour
        
        if hour >= 23:
            duration = recent[-1].get('behavior', {}).get('session_duration_sec', 0)
            if duration > 300:  # 5+ min late night usage
                return {
                    'pattern_name': 'late_night_usage',
                    'confidence': 0.7,
                    'description': 'Screen usage detected after 11pm',
                    'severity': EventSeverity.WARNING
                }
        
        return None
    
    def _create_event(
        self,
        user_id: int,
        observation: Dict,
        anomaly: Dict,
        window: List[Dict]
    ) -> RealTimeEvent:
        """Create a real-time event from anomaly detection."""
        
        # Determine action based on anomaly type
        action_map = {
            'session_duration_sec': ('suggest_break', 8, 60),
            'app_switches_5min': ('focus_mode_prompt', 9, 30),
            'scroll_velocity': ('content_quality_check', 6, 120),
            'typing_interval_ms': ('fatigue_check', 7, 90)
        }
        
        feature = anomaly['feature']
        action, priority, window_sec = action_map.get(feature, ('general_check', 5, 300))
        
        return RealTimeEvent(
            event_id=f"evt_{datetime.utcnow().timestamp()}_{user_id}",
            user_id=user_id,
            timestamp=datetime.utcnow(),
            severity=anomaly['severity'],
            event_type='anomaly',
            raw_observation=observation,
            trigger_feature=feature,
            trigger_value=anomaly['value'],
            baseline_value=anomaly['expected'],
            deviation_score=anomaly['z_score'],
            recommended_action=action,
            action_priority=priority,
            action_window_seconds=window_sec,
            recent_context=window[-5:]
        )
    
    def _create_pattern_event(
        self,
        user_id: int,
        observation: Dict,
        pattern: Dict,
        window: List[Dict]
    ) -> RealTimeEvent:
        """Create event from pattern detection."""
        
        is_opportunity = pattern.get('is_opportunity', False)
        
        if is_opportunity:
            action = 'offer_extension'
            priority = 6
            window_sec = 300
        else:
            action = 'interrupt_pattern'
            priority = 7 if pattern['severity'] == EventSeverity.WARNING else 5
            window_sec = 60
        
        return RealTimeEvent(
            event_id=f"evt_{datetime.utcnow().timestamp()}_{user_id}",
            user_id=user_id,
            timestamp=datetime.utcnow(),
            severity=pattern['severity'],
            event_type='pattern_match',
            raw_observation=observation,
            trigger_feature=pattern['pattern_name'],
            trigger_value=pattern['confidence'],
            baseline_value=0.5,  # Expected baseline
            deviation_score=pattern['confidence'],
            recommended_action=action,
            action_priority=priority,
            action_window_seconds=window_sec,
            recent_context=window[-5:]
        )
    
    async def _dispatch_event(self, event: RealTimeEvent):
        """Dispatch event to all registered handlers."""
        for handler in self.handlers:
            try:
                # Run handlers concurrently
                asyncio.create_task(handler(event))
            except Exception as e:
                print(f"Handler error: {e}")
    
    async def _update_baseline(self, user_id: int, observation: Dict):
        """Update running baseline for user."""
        if user_id not in self.user_baselines:
            self.user_baselines[user_id] = {}
        
        baselines = self.user_baselines[user_id]
        behavior = observation.get('behavior', {})
        
        # Update each feature with exponential moving average
        for feature, value in behavior.items():
            if not isinstance(value, (int, float)):
                continue
            
            if feature not in baselines:
                baselines[feature] = {'mean': value, 'std': value * 0.1, 'n': 1}
            else:
                b = baselines[feature]
                # Exponential moving average (alpha = 0.1)
                alpha = 0.1
                old_mean = b['mean']
                b['mean'] = old_mean + alpha * (value - old_mean)
                
                # Update variance
                variance = (b['std'] ** 2) + alpha * ((value - old_mean) ** 2 - (b['std'] ** 2))
                b['std'] = max(variance ** 0.5, b['mean'] * 0.05)  # Min 5% coefficient of variation
                b['n'] += 1
    
    async def _baseline_refresh_loop(self):
        """Periodically persist baselines to database."""
        while self.running:
            await asyncio.sleep(300)  # Every 5 minutes
            
            try:
                await self._persist_baselines()
            except Exception as e:
                print(f"Baseline persistence error: {e}")
    
    async def _persist_baselines(self):
        """Save baselines to Supabase for recovery."""
        store = await get_behavioral_store()
        if not store:
            return
        
        for user_id, baselines in self.user_baselines.items():
            store.client.table('user_baselines').upsert({
                'user_id': str(user_id),
                'baselines': baselines,
                'updated_at': datetime.utcnow().isoformat(),
                'window_size': self.window_size
            }, on_conflict='user_id').execute()


class RealTimeInterventionHandler:
    """
    Handles real-time events by triggering immediate interventions.
    """
    
    def __init__(self):
        self.cooldowns: Dict[str, datetime] = {}  # action_type -> last_triggered
    
    async def handle(self, event: RealTimeEvent):
        """
        Handle a real-time event.
        """
        # Check cooldown
        cooldown_key = f"{event.user_id}_{event.recommended_action}"
        last_triggered = self.cooldowns.get(cooldown_key)
        
        if last_triggered:
            elapsed = (datetime.utcnow() - last_triggered).total_seconds()
            if elapsed < 300:  # 5 min cooldown
                return  # Skip, in cooldown
        
        # Check action window
        if event.action_window_seconds < 10:
            # Immediate action required
            await self._trigger_immediate(event)
        else:
            # Can schedule
            await self._schedule_action(event)
        
        # Update cooldown
        self.cooldowns[cooldown_key] = datetime.utcnow()
    
    async def _trigger_immediate(self, event: RealTimeEvent):
        """Trigger immediate intervention."""
        from spirit.services.notification_engine import NotificationEngine
        
        engine = NotificationEngine(event.user_id)
        
        # High priority for immediate
        content = {
            "title": self._get_title(event),
            "body": self._get_body(event),
            "data": {
                "event_type": event.event_type,
                "severity": event.severity.value,
                "action": event.recommended_action,
                "deviation": event.deviation_score
            }
        }
        
        await engine.send_notification(
            content=content,
            priority="critical" if event.severity == EventSeverity.CRITICAL else "high",
            notification_type=f"realtime_{event.event_type}"
        )
        
        # Log
        print(f"IMMEDIATE: {event.recommended_action} for user {event.user_id} "
              f"(z={event.deviation_score:.2f})")
    
    async def _schedule_action(self, event: RealTimeEvent):
        """Schedule action for optimal delivery."""
        # Add to proactive scheduler
        from spirit.agents.proactive_loop import get_orchestrator
        
        # Store for later delivery
        store = await get_behavioral_store()
        if store:
            store.client.table('scheduled_interventions').insert({
                'user_id': str(event.user_id),
                'trigger_event': event.event_id,
                'recommended_action': event.recommended_action,
                'priority': event.action_priority,
                'schedule_by': (datetime.utcnow() + timedelta(seconds=event.action_window_seconds)).isoformat(),
                'status': 'pending'
            }).execute()
    
    def _get_title(self, event: RealTimeEvent) -> str:
        """Generate notification title."""
        if event.severity == EventSeverity.CRITICAL:
            return "⚠️ Pattern detected"
        elif event.event_type == 'pattern_match':
            return "📊 Insight available"
        return "Spirit check-in"
    
    def _get_body(self, event: RealTimeEvent) -> str:
        """Generate notification body."""
        bodies = {
            'suggest_break': "You've been going for a while. Quick reset?",
            'focus_mode_prompt': "Distraction pattern detected. Focus mode available.",
            'offer_extension': "You're in flow. Extend this session?",
            'interrupt_pattern': f"{event.trigger_feature.replace('_', ' ').title()} detected. Intervention ready.",
            'content_quality_check': "Scrolling fast. Content engaging or escape?",
            'fatigue_check': "Typing slowing. Energy check?"
        }
        return bodies.get(event.recommended_action, "Quick check-in available")


# Global singleton processor
_processor: Optional[StreamProcessor] = None


def get_stream_processor() -> StreamProcessor:
    """Get or create global stream processor."""
    global _processor
    if _processor is None:
        _processor = StreamProcessor()
        
        # Register default handler
        handler = RealTimeInterventionHandler()
        _processor.register_handler(handler.handle)
    
    return _processor
