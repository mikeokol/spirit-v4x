"""
Context enrichment service for behavioral observations.
Adds server-side context that edge devices don't have access to.
"""

from datetime import datetime
from typing import Optional

from fastapi import Request

from spirit.models.behavioral import BehavioralObservation, ScreenTimeSession


class ContextEnricher:
    """
    Enriches behavioral data with server-side context.
    All enrichment must respect privacy—no de-anonymization.
    """
    
    async def enrich_session(
        self, 
        session: ScreenTimeSession,
        request: Optional[Request] = None
    ) -> ScreenTimeSession:
        """Add server-side temporal context."""
        server_time = datetime.utcnow()
        session.collected_at = server_time
        
        # Time-of-day classification
        hour = session.started_at.hour if session.started_at else server_time.hour
        
        return session
    
    async def enrich_observation(
        self,
        observation: BehavioralObservation,
        request: Optional[Request] = None,
        server_received_at: Optional[datetime] = None
    ) -> BehavioralObservation:
        """Enrich with server context and calculate latency."""
        if server_received_at is None:
            server_received_at = datetime.utcnow()
        
        # Calculate processing latency
        edge_latency = observation.processing_metadata.get('edge_processing_latency_ms', 0)
        network_latency = (
            server_received_at - observation.timestamp
        ).total_seconds() * 1000
        
        observation.processing_metadata.update({
            'server_received_at': server_received_at.isoformat(),
            'network_latency_ms': max(0, network_latency),
            'total_latency_ms': edge_latency + max(0, network_latency),
            'enriched_at': server_received_at.isoformat()
        })
        
        # Add time-based context if not present
        if 'time_of_day' not in observation.context:
            hour = observation.timestamp.hour
            observation.context['time_of_day'] = self._classify_time_of_day(hour)
            observation.context['day_of_week'] = observation.timestamp.weekday()
            observation.context['is_weekend'] = observation.timestamp.weekday() >= 5
        
        # Add seasonal context
        month = observation.timestamp.month
        observation.context['season'] = self._classify_season(month)
        observation.context['quarter'] = (month - 1) // 3 + 1
        
        return observation
    
    def _classify_time_of_day(self, hour: int) -> str:
        """Classify hour into behavioral time periods."""
        if 5 <= hour < 9:
            return "early_morning"
        elif 9 <= hour < 12:
            return "morning"
        elif 12 <= hour < 14:
            return "midday"
        elif 14 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 22:
            return "evening"
        else:
            return "night"
    
    def _classify_season(self, month: int) -> str:
        """Northern hemisphere seasons."""
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "fall"
