"""
Memory Consolidation: Nightly compression of episodic → semantic memory.
Runs as Render cron job at 3 AM.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import asyncio

from spirit.db.supabase_client import get_behavioral_store
from spirit.memory.episodic_memory import EpisodicMemorySystem
from spirit.config import settings


class MemoryConsolidationEngine:
    """
    Converts daily episodic memories into long-term semantic knowledge.
    Implements forgetting curve and importance-based retention.
    """
    
    def __init__(self):
        self.consolidation_hour = 3  # 3 AM
        self.retention_threshold = 0.6  # Keep top 60% by importance
    
    async def run_consolidation(self, user_id: Optional[int] = None):
        """
        Run consolidation for one or all users.
        Called by Render cron at 3 AM.
        """
        store = await get_behavioral_store()
        if not store:
            return
        
        if user_id:
            users = [user_id]
        else:
            # Get all active users
            result = store.client.table('user_activity').select('user_id').gte(
                'last_active', (datetime.utcnow() - timedelta(days=7)).isoformat()
            ).execute()
            users = [r['user_id'] for r in result.data] if result.data else []
        
        for uid in users:
            await self._consolidate_user(uid)
    
    async def _consolidate_user(self, user_id: int):
        """Consolidate memories for single user."""
        print(f"Consolidating memories for user {user_id}...")
        
        # 1. Retrieve yesterday's episodic memories
        memory = EpisodicMemorySystem(user_id)
        
        yesterday_start = (datetime.utcnow() - timedelta(days=1)).replace(
            hour=0, minute=0, second=0
        )
        yesterday_end = yesterday_start + timedelta(days=1)
        
        episodes = await memory.retrieve_relevant_memories(
            current_context={},
            time_horizon=timedelta(hours=24),
            n_results=1000  # Get all
        )
        
        if not episodes:
            return
        
        # 2. Extract semantic patterns
        patterns = self._extract_patterns(episodes)
        
        # 3. Compress episodes (forgetting curve)
        retained_episodes = self._apply_forgetting_curve(episodes)
        
        # 4. Store semantic memories
        await self._store_semantic_memories(user_id, patterns)
        
        # 5. Mark consolidated episodes (don't delete, just mark)
        await self._mark_consolidated(episodes)
        
        # 6. Clean up low-importance noise
        await self._decay_noise(user_id, episodes)
        
        print(f"Consolidated {len(episodes)} episodes into {len(patterns)} patterns for user {user_id}")
    
    def _extract_patterns(self, episodes: List) -> List[Dict]:
        """
        Extract semantic patterns from episodic memories.
        Uses LLM for pattern extraction.
        """
        if not episodes:
            return []
        
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key)
        
        # Batch episodes for efficiency
        batch_size = 20
        patterns = []
        
        for i in range(0, len(episodes), batch_size):
            batch = episodes[i:i+batch_size]
            
            episode_texts = [
                f"- {e.timestamp.strftime('%H:%M')}: {e.what_happened} (importance: {e.importance_score})"
                for e in batch
            ]
            
            messages = [
                SystemMessage(content="""
                Extract 1-2 behavioral patterns from these episodes.
                Format: "When [situation], user tends to [behavior], leading to [outcome]"
                Be specific but generalizable. Focus on causal mechanisms.
                """),
                HumanMessage(content="\n".join(episode_texts))
            ]
            
            response = llm.invoke(messages)
            
            # Parse patterns (simplified)
            for line in response.content.split('\n'):
                if line.strip() and 'user' in line.lower():
                    patterns.append({
                        'pattern': line.strip(),
                        'source_episodes': [e.episode_id for e in batch],
                        'confidence': sum(e.importance_score for e in batch) / len(batch),
                        'extracted_at': datetime.utcnow().isoformat()
                    })
        
        return patterns
    
    def _apply_forgetting_curve(self, episodes: List) -> List:
        """
        Apply forgetting curve: keep high-importance, recent, or emotional.
        """
        now = datetime.utcnow()
        
        scored = []
        for ep in episodes:
            # Forgetting score: higher = more likely to remember
            age_hours = (now - ep.timestamp).total_seconds() / 3600
            
            # Ebbinghaus forgetting curve approximation
            retention = ep.importance_score * (0.9 ** (age_hours / 24))
            
            # Emotional boost
            if abs(ep.emotional_valence) > 0.5:
                retention *= 1.3
            
            scored.append((retention, ep))
        
        # Keep top 60%
        scored.sort(reverse=True, key=lambda x: x[0])
        cutoff = int(len(scored) * self.retention_threshold)
        
        return [ep for _, ep in scored[:cutoff]]
    
    async def _store_semantic_memories(self, user_id: int, patterns: List[Dict]):
        """Store extracted patterns as semantic memory."""
        store = await get_behavioral_store()
        if not store:
            return
        
        for pattern in patterns:
            store.client.table('semantic_memories').upsert({
                'user_id': str(user_id),
                'pattern': pattern['pattern'],
                'confidence': pattern['confidence'],
                'source_episodes': pattern['source_episodes'],
                'extracted_at': pattern['extracted_at'],
                'activation_count': 0,
                'last_activated': None
            }, on_conflict='pattern').execute()
    
    async def _mark_consolidated(self, episodes: List):
        """Mark episodes as consolidated (don't re-process)."""
        store = await get_behavioral_store()
        if not store:
            return
        
        for ep in episodes:
            store.client.table('episodic_memories').update({
                'consolidated': True,
                'consolidated_at': datetime.utcnow().isoformat()
            }).eq('episode_id', ep.episode_id).execute()
    
    async def _decay_noise(self, user_id: int, episodes: List):
        """Delete or archive low-importance, unconsolidated noise."""
        store = await get_behavioral_store()
        if not store:
            return
        
        # Find low-importance, old, unconsolidated episodes
        old_threshold = (datetime.utcnow() - timedelta(days=2)).isoformat()
        
        noise = store.client.table('episodic_memories').select('*').eq(
            'user_id', str(user_id)
        ).eq('consolidated', False).lt('importance_score', 0.3).lt(
            'timestamp', old_threshold
        ).execute()
        
        if noise.data:
            # Archive to cold storage (or delete if confident)
            # For now, mark as decayed
            for n in noise.data:
                store.client.table('episodic_memories').update({
                    'decayed': True,
                    'decayed_at': datetime.utcnow().isoformat()
                }).eq('episode_id', n['episode_id']).execute()
            
            print(f"Decayed {len(noise.data)} low-importance memories for user {user_id}")


# Render cron endpoint
async def run_nightly_consolidation():
    """Entry point for Render cron job."""
    engine = MemoryConsolidationEngine()
    await engine.run_consolidation()
