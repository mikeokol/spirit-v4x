"""
Notification/Delivery System: How Spirit actually reaches users.
Manages timing, channel selection, and delivery orchestration across
push notifications, in-app, email, and future channels.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Literal
from enum import Enum
from uuid import UUID, uuid4
import asyncio

from spirit.config import settings
from spirit.db.supabase_client import get_behavioral_store
from spirit.memory.episodic_memory import EpisodicMemorySystem


class ChannelType(str, Enum):
    PUSH = "push"           # Mobile push notification
    IN_APP = "in_app"       # In-app modal/banner
    SMS = "sms"             # Text message
    EMAIL = "email"         # Email digest
    WHATSAPP = "whatsapp"   # WhatsApp Business
    SLACK = "slack"         # Slack DM
    SMARTWATCH = "smartwatch"  # Wearable haptic/notification


class NotificationPriority(str, Enum):
    CRITICAL = "critical"   # Immediate delivery, all channels
    HIGH = "high"           # Within 5 minutes, primary channel
    NORMAL = "normal"       # Within 15 minutes, best channel
    LOW = "low"             # Batch with next interaction, in-app only
    DIGEST = "digest"       # Daily summary, email/push digest


class DeliveryWindow:
    """
    When a user is receptive to interventions.
    Learned from behavioral patterns.
    """
    
    def __init__(self):
        self.weekday_hours: List[int] = []      # Hours when user is active
        self.weekend_hours: List[int] = []
        self.receptivity_scores: Dict[int, float] = {}  # Hour -> 0-1 score
        self.last_updated: Optional[datetime] = None
    
    def is_optimal_now(self) -> bool:
        """Check if current time is in optimal delivery window."""
        now = datetime.utcnow()
        hour = now.hour
        is_weekend = now.weekday() >= 5
        
        hours = self.weekend_hours if is_weekend else self.weekday_hours
        return hour in hours
    
    def get_next_window(self) -> Optional[datetime]:
        """Calculate next optimal delivery time."""
        now = datetime.utcnow()
        
        # Check next 48 hours
        for hours_ahead in range(48):
            check_time = now + timedelta(hours=hours_ahead)
            hour = check_time.hour
            is_weekend = check_time.weekday() >= 5
            
            hours = self.weekend_hours if is_weekend else self.weekday_hours
            
            if hour in hours and check_time > now + timedelta(minutes=5):
                return check_time
        
        return None


class NotificationEngine:
    """
    Intelligent notification delivery system.
    Learns when and how each user prefers to be reached.
    """
    
    def __init__(self, user_id: UUID):
        self.user_id = user_id
        self.delivery_windows: Dict[str, DeliveryWindow] = {}  # By notification type
        self.channel_preferences: Dict[ChannelType, float] = {}  # Channel -> success rate
        self.rate_limit_tracker: Dict[str, datetime] = {}  # Last notification by type
        self.cooldown_minutes = {
            NotificationPriority.CRITICAL: 0,
            NotificationPriority.HIGH: 15,
            NotificationPriority.NORMAL: 60,
            NotificationPriority.LOW: 240,
            NotificationPriority.DIGEST: 1440
        }
    
    async def send_notification(
        self,
        content: Dict[str, Any],  # {title, body, action_url, data}
        priority: NotificationPriority,
        notification_type: str,  # 'intervention', 'insight', 'reminder', 'celebration'
        preferred_channel: Optional[ChannelType] = None,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Send notification through optimal channel at optimal time.
        """
        # Check rate limiting
        if not await self._check_rate_limit(notification_type, priority):
            return {
                "sent": False,
                "reason": "rate_limited",
                "retry_after": self._get_retry_time(notification_type)
            }
        
        # Determine channel
        channel = preferred_channel or await self._select_optimal_channel(
            notification_type, priority, context
        )
        
        # Check delivery window
        window = await self._get_delivery_window(notification_type)
        
        if not window.is_optimal_now() and priority != NotificationPriority.CRITICAL:
            # Schedule for later
            next_window = window.get_next_window()
            if next_window:
                await self._schedule_notification(
                    content, channel, priority, notification_type, next_window
                )
                return {
                    "sent": False,
                    "scheduled": True,
                    "deliver_at": next_window.isoformat(),
                    "channel": channel.value,
                    "reason": "outside_optimal_window"
                }
        
        # Deliver immediately
        result = await self._deliver(content, channel, notification_type)
        
        # Log for learning
        await self._log_delivery_attempt(notification_type, channel, result)
        
        return {
            "sent": result["success"],
            "message_id": result.get("message_id"),
            "channel": channel.value,
            "delivered_at": datetime.utcnow().isoformat(),
            "scheduled": False
        }
    
    async def _select_optimal_channel(
        self,
        notification_type: str,
        priority: NotificationPriority,
        context: Optional[Dict]
    ) -> ChannelType:
        """
        Select best channel based on historical effectiveness.
        """
        # Get historical success rates
        store = await get_behavioral_store()
        if store:
            # Query past delivery outcomes
            history = store.client.table('notification_history').select('*').eq(
                'user_id', str(self.user_id)
            ).eq('notification_type', notification_type).execute()
            
            if history.data:
                # Calculate success by channel
                channel_success = {}
                for record in history.data:
                    ch = record['channel']
                    success = record.get('engaged', False)
                    if ch not in channel_success:
                        channel_success[ch] = {'success': 0, 'total': 0}
                    channel_success[ch]['total'] += 1
                    if success:
                        channel_success[ch]['success'] += 1
                
                # Pick best channel with min 3 attempts
                best_channel = None
                best_rate = 0
                for ch, stats in channel_success.items():
                    if stats['total'] >= 3:
                        rate = stats['success'] / stats['total']
                        if rate > best_rate:
                            best_rate = rate
                            best_channel = ch
                
                if best_channel:
                    return ChannelType(best_channel)
        
        # Defaults by priority and context
        if priority == NotificationPriority.CRITICAL:
            return ChannelType.PUSH  # Most reliable
        
        if context and context.get('user_active_in_app'):
            return ChannelType.IN_APP
        
        if priority == NotificationPriority.DIGEST:
            return ChannelType.EMAIL
        
        return ChannelType.PUSH
    
    async def _deliver(
        self,
        content: Dict[str, Any],
        channel: ChannelType,
        notification_type: str
    ) -> Dict[str, Any]:
        """
        Actually deliver through selected channel.
        """
        # Route to appropriate delivery service
        if channel == ChannelType.PUSH:
            return await self._send_push(content)
        elif channel == ChannelType.IN_APP:
            return await self._send_in_app(content)
        elif channel == ChannelType.EMAIL:
            return await self._send_email(content)
        elif channel == ChannelType.SMS:
            return await self._send_sms(content)
        else:
            return {"success": False, "error": "unsupported_channel"}
    
    async def _send_push(self, content: Dict) -> Dict[str, Any]:
        """Send mobile push via FCM/APNs."""
        # TODO: Integrate with Firebase Cloud Messaging
        # For now, log and simulate
        print(f"PUSH to {self.user_id}: {content.get('title', 'No title')}")
        return {
            "success": True,
            "message_id": str(uuid4()),
            "channel": "push"
        }
    
    async def _send_in_app(self, content: Dict) -> Dict[str, Any]:
        """Queue in-app notification for next app open."""
        store = await get_behavioral_store()
        if store:
            store.client.table('in_app_notifications').insert({
                'user_id': str(self.user_id),
                'content': content,
                'created_at': datetime.utcnow().isoformat(),
                'expires_at': (datetime.utcnow() + timedelta(hours=24)).isoformat(),
                'displayed': False
            }).execute()
        
        return {
            "success": True,
            "message_id": str(uuid4()),
            "channel": "in_app"
        }
    
    async def _send_email(self, content: Dict) -> Dict[str, Any]:
        """Send email via SendGrid/AWS SES."""
        # TODO: Integrate with email service
        print(f"EMAIL to {self.user_id}: {content.get('subject', 'No subject')}")
        return {
            "success": True,
            "message_id": str(uuid4()),
            "channel": "email"
        }
    
    async def _send_sms(self, content: Dict) -> Dict[str, Any]:
        """Send SMS via Twilio."""
        # TODO: Integrate with Twilio
        print(f"SMS to {self.user_id}: {content.get('body', 'No body')[:50]}...")
        return {
            "success": True,
            "message_id": str(uuid4()),
            "channel": "sms"
        }
    
    async def _check_rate_limit(
        self,
        notification_type: str,
        priority: NotificationPriority
    ) -> bool:
        """Check if we're within rate limits for this user."""
        last_sent = self.rate_limit_tracker.get(notification_type)
        if not last_sent:
            return True
        
        cooldown = timedelta(minutes=self.cooldown_minutes[priority])
        return datetime.utcnow() - last_sent > cooldown
    
    def _get_retry_time(self, notification_type: str) -> str:
        """Calculate when this notification type can be sent again."""
        last_sent = self.rate_limit_tracker.get(notification_type)
        if not last_sent:
            return datetime.utcnow().isoformat()
        
        # Assume normal priority for retry
        cooldown = timedelta(minutes=self.cooldown_minutes[NotificationPriority.NORMAL])
        return (last_sent + cooldown).isoformat()
    
    async def _get_delivery_window(self, notification_type: str) -> DeliveryWindow:
        """Get or learn optimal delivery window for this notification type."""
        if notification_type in self.delivery_windows:
            return self.delivery_windows[notification_type]
        
        # Learn from historical engagement
        window = DeliveryWindow()
        
        store = await get_behavioral_store()
        if store:
            # Query when user typically engages with this notification type
            history = store.client.table('notification_history').select('*').eq(
                'user_id', str(self.user_id)
            ).eq('notification_type', notification_type).eq('engaged', True).execute()
            
            if history.data:
                hours = [datetime.fromisoformat(r['sent_at']).hour for r in history.data]
                window.weekday_hours = list(set([h for h in hours if h < 23 and h > 6]))
                window.receptivity_scores = {h: hours.count(h)/len(hours) for h in set(hours)}
        
        # Default if no history
        if not window.weekday_hours:
            window.weekday_hours = [9, 12, 18, 21]  # Safe defaults
        
        self.delivery_windows[notification_type] = window
        return window
    
    async def _schedule_notification(
        self,
        content: Dict,
        channel: ChannelType,
        priority: NotificationPriority,
        notification_type: str,
        deliver_at: datetime
    ):
        """Schedule notification for future delivery."""
        store = await get_behavioral_store()
        if store:
            store.client.table('scheduled_notifications').insert({
                'notification_id': str(uuid4()),
                'user_id': str(self.user_id),
                'content': content,
                'channel': channel.value,
                'priority': priority.value,
                'notification_type': notification_type,
                'scheduled_for': deliver_at.isoformat(),
                'status': 'pending'
            }).execute()
    
    async def _log_delivery_attempt(
        self,
        notification_type: str,
        channel: ChannelType,
        result: Dict
    ):
        """Log for learning and rate limiting."""
        self.rate_limit_tracker[notification_type] = datetime.utcnow()
        
        store = await get_behavioral_store()
        if store:
            store.client.table('notification_history').insert({
                'user_id': str(self.user_id),
                'notification_type': notification_type,
                'channel': channel.value,
                'sent_at': datetime.utcnow().isoformat(),
                'message_id': result.get('message_id'),
                'delivered': result.get('success', False)
            }).execute()
    
    async def process_scheduled_notifications(self):
        """
        Background task: Process pending scheduled notifications.
        Call this periodically (e.g., every minute).
        """
        store = await get_behavioral_store()
        if not store:
            return
        
        now = datetime.utcnow().isoformat()
        
        # Get pending notifications due now
        pending = store.client.table('scheduled_notifications').select('*').eq(
            'status', 'pending'
        ).lte('scheduled_for', now).execute()
        
        for notif in pending.data if pending.data else []:
            # Send
            await self.send_notification(
                content=notif['content'],
                priority=NotificationPriority(notif['priority']),
                notification_type=notif['notification_type'],
                preferred_channel=ChannelType(notif['channel'])
            )
            
            # Mark sent
            store.client.table('scheduled_notifications').update({
                'status': 'sent',
                'sent_at': now
            }).eq('notification_id', notif['notification_id']).execute()


class SmartDigestGenerator:
    """
    Generates personalized daily/weekly digests.
    Combines episodic memories, progress updates, and peer insights.
    """
    
    def __init__(self, user_id: UUID):
        self.user_id = user_id
    
    async def generate_daily_digest(self) -> Dict[str, Any]:
        """Generate personalized daily summary."""
        # Get episodic memories from yesterday
        memory = EpisodicMemorySystem(self.user_id)
        yesterday = datetime.utcnow() - timedelta(days=1)
        
        narrative = await memory.generate_narrative_summary(timedelta(days=1))
        streak = await memory.get_streak_and_momentum()
        
        # Get goal progress
        from spirit.services.goal_integration import BehavioralGoalBridge
        bridge = BehavioralGoalBridge(self.user_id)
        
        # TODO: Get active goals and compute progress
        
        content = {
            "subject": f"Your Spirit Daily • Day {streak['current_streak_days']}",
            "title": f"Day {streak['current_streak_days']} of your journey",
            "body": narrative,
            "streak": streak,
            "milestones": [],
            "tomorrow_preview": "Based on your patterns, tomorrow morning is optimal for deep work."
        }
        
        return content
