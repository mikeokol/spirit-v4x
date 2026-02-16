"""
Supabase client for Spirit's behavioral data layer.
Handles batch inserts and high-throughput ingestion from mobile devices.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from uuid import UUID

from supabase import create_client, Client
from postgrest.exceptions import APIError

from spirit.config import settings
from spirit.models.behavioral import (
    BehavioralObservation, 
    ScreenTimeSession, 
    UserCausalHypothesis,
    EMARequest
)


class SupabaseBehavioralStore:
    """
    High-performance behavioral data storage with batching.
    """
    
    def __init__(self):
        self.client: Optional[Client] = None
        self._batch_queue: List[Dict] = []
        self._batch_size = settings.screen_time_batch_size
        self._lock = asyncio.Lock()
        
    async def connect(self):
        """Initialize Supabase client."""
        if not self.client and settings.supabase_url:
            key = settings.supabase_service_key or settings.supabase_anon_key
            if key:
                self.client = create_client(settings.supabase_url, key)
        return self
    
    async def disconnect(self):
        """Cleanup resources."""
        await self._flush_batch()
        self.client = None
    
    async def store_observation(self, observation: BehavioralObservation) -> bool:
        """Store single observation with automatic batching."""
        if not self.client:
            await self.connect()
            
        data = observation.dict()
        data['observation_id'] = str(data['observation_id'])
        data['user_id'] = str(data['user_id'])
        if data.get('intervention_id'):
            data['intervention_id'] = str(data['intervention_id'])
        
        async with self._lock:
            self._batch_queue.append({
                'table': 'behavioral_observations',
                'data': data
            })
            
            if len(self._batch_queue) >= self._batch_size:
                await self._flush_batch()
        
        return True
    
    async def store_screen_session(self, session: ScreenTimeSession) -> bool:
        """Store screen time session with upsert for idempotency."""
        if not self.client:
            await self.connect()
            
        if not self.client:
            return False  # No Supabase configured
            
        data = session.dict()
        data['session_id'] = str(data['session_id'])
        data['user_id'] = str(data['user_id'])
        
        try:
            result = self.client.table('screen_time_sessions').upsert(
                data,
                on_conflict='session_id'
            ).execute()
            return len(result.data) > 0
        except APIError as e:
            print(f"Failed to store screen session: {e}")
            return False
    
    async def get_user_observations(
        self, 
        user_id: UUID, 
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        observation_type: Optional[str] = None,
        limit: int = 1000
    ) -> List[BehavioralObservation]:
        """Retrieve observations for causal analysis."""
        if not self.client:
            await self.connect()
            
        if not self.client:
            return []
        
        query = self.client.table('behavioral_observations').select('*')
        query = query.eq('user_id', str(user_id))
        
        if start_time:
            query = query.gte('timestamp', start_time)
        if end_time:
            query = query.lte('timestamp', end_time)
        if observation_type:
            query = query.eq('observation_type', observation_type)
            
        query = query.order('timestamp', desc=True).limit(limit)
        
        result = query.execute()
        return [BehavioralObservation(**row) for row in result.data]
    
    async def _flush_batch(self):
        """Internal: Flush batched observations."""
        if not self._batch_queue or not self.client:
            return
            
        async with self._lock:
            batch = self._batch_queue[:]
            self._batch_queue = []
        
        by_table: Dict[str, List[Dict]] = {}
        for item in batch:
            table = item['table']
            by_table.setdefault(table, []).append(item['data'])
        
        for table, rows in by_table.items():
            await self._insert_batch(table, rows)
    
    async def _insert_batch(self, table: str, rows: List[Dict]):
        """Insert batch with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.client.table(table).insert(rows).execute()
                return
            except APIError as e:
                if attempt == max_retries - 1:
                    print(f"Failed to insert batch to {table}: {e}")
                await asyncio.sleep(0.1 * (2 ** attempt))


# Singleton instance
_behavioral_store: Optional[SupabaseBehavioralStore] = None


async def get_behavioral_store() -> Optional[SupabaseBehavioralStore]:
    """Get or create singleton store instance."""
    global _behavioral_store
    if _behavioral_store is None:
        _behavioral_store = SupabaseBehavioralStore()
        await _behavioral_store.connect()
    return _behavioral_store


async def close_behavioral_store():
    """Cleanup on shutdown."""
    global _behavioral_store
    if _behavioral_store:
        await _behavioral_store.disconnect()
        _behavioral_store = None
