"""
Integration layer: Connects behavioral data (from v1/ingestion) to your existing goal system.
This reads FROM Supabase (behavioral data) and writes TO SQLite (goal progress).
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from uuid import UUID

from sqlalchemy import select
from spirit.db import async_session
from spirit.models import Goal, Execution
from spirit.db.supabase_client import get_behavioral_store


class BehavioralGoalBridge:
    """
    Translates behavioral observations into goal-relevant metrics.
    
    Usage:
        bridge = BehavioralGoalBridge(user_id)
        progress = await bridge.compute_goal_progress(goal_id)
    """
    
    def __init__(self, user_id: UUID):
        self.user_id = user_id
    
    async def compute_goal_progress(
        self,
        goal_id: UUID,
        date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Compute progress for a goal using behavioral data that was 
        previously ingested via /v1/ingestion endpoints.
        """
        if date is None:
            date = datetime.utcnow()
        
        # Get goal from your existing SQLite
        async with async_session() as session:
            result = await session.execute(
                select(Goal).where(Goal.id == goal_id)
            )
            goal = result.scalar_one_or_none()
            
            if not goal:
                return {"error": "goal_not_found"}
        
        # Query behavioral data from Supabase (ingested earlier)
        store = await get_behavioral_store()
        if not store:
            return {
                "goal_id": str(goal_id),
                "behavioral_data_available": False,
                "progress": None
            }
        
        day_start = date.replace(hour=0, minute=0, second=0)
        day_end = day_start + timedelta(days=1)
        
        # This queries data that mobile sent to /v1/ingestion/observations
        observations = await store.get_user_observations(
            user_id=self.user_id,
            start_time=day_start.isoformat(),
            end_time=day_end.isoformat(),
            limit=1000
        )
        
        metrics = self._extract_metrics(goal, observations)
        
        return {
            "goal_id": str(goal_id),
            "date": date.isoformat(),
            "behavioral_data_available": True,
            "metrics": metrics,
            "progress_score": self._calculate_progress(goal, metrics),
            "insights": self._generate_insights(goal, metrics)
        }
    
    def _extract_metrics(self, goal: Goal, observations: List[Any]) -> Dict:
        """Extract relevant metrics from behavioral observations."""
        metrics = {
            "total_screen_time_minutes": 0,
            "productive_time_minutes": 0,
            "focus_sessions": 0,
            "context_switches": 0
        }
        
        for obs in observations:
            behavior = obs.behavior
            
            if 'session_duration_sec' in behavior:
                minutes = behavior['session_duration_sec'] / 60
                metrics["total_screen_time_minutes"] += minutes
                
                category = behavior.get('app_category', 'other')
                if category in ['productivity', 'health']:
                    metrics["productive_time_minutes"] += minutes
            
            if behavior.get('session_type') == 'deep_work':
                metrics["focus_sessions"] += 1
            
            metrics["context_switches"] += behavior.get('app_switches_5min', 0)
        
        return metrics
    
    def _calculate_progress(self, goal: Goal, metrics: Dict) -> float:
        """Calculate 0-1 progress score."""
        # Simple example: productivity goals
        if "focus" in goal.title.lower() or "productiv" in goal.title.lower():
            target = 180  # 3 hours
            actual = metrics["productive_time_minutes"]
            return min(1.0, actual / target)
        
        # Default
        return 0.5
    
    def _generate_insights(self, goal: Goal, metrics: Dict) -> List[str]:
        """Generate insights from behavioral patterns."""
        insights = []
        
        if metrics["context_switches"] > 30:
            insights.append(f"High app switching ({metrics['context_switches']}) detected")
        
        if metrics["productive_time_minutes"] < 60:
            insights.append("Low productive time today")
        
        return insights
