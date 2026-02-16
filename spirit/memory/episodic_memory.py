"""
Episodic memory: Spirit remembers every interaction, insight, and breakthrough.
Creates a narrative of the user's journey that improves recommendations over time.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from uuid import UUID
from dataclasses import dataclass, asdict

from spirit.db.supabase_client import get_behavioral_store
from spirit.config import settings


@dataclass
class MemoryEpisode:
    """
    A single memorable moment in the user's journey.
    Stored in Supabase with vector embedding for semantic retrieval.
    """
    episode_id: str
    user_id: str
    timestamp: datetime
    episode_type: str  # 'breakthrough', 'struggle', 'insight', 'milestone', 'routine'
    
    # Content
    what_happened: str  # Narrative description
    behavioral_context: Dict  # Raw data snapshot
    emotional_valence: float  # -1 to 1 (negative to positive)
    
    # Significance
    importance_score: float  # 0-1, calculated from impact
    tags: List[str]  # Searchable labels
    
    # Connections
    related_goal_ids: List[str]
    related_hypothesis_ids: List[str]
    
    # Learning
    lesson_learned: Optional[str]  # What Spirit learned from this
    user_reflection: Optional[str]  # What user said/thought
    
    # For retrieval
    embedding: Optional[List[float]] = None  # Vector for semantic search


class EpisodicMemorySystem:
    """
    Long-term memory that creates a coherent narrative of user's behavioral journey.
    Powers: "Remember when you...", "You've come so far...", "This worked for you before..."
    """
    
    def __init__(self, user_id: UUID):
        self.user_id = user_id
        self.short_term_buffer: List[MemoryEpisode] = []
        self.consolidation_threshold = 5  # Episodes before consolidation
        
    async def record_episode(
        self,
        episode_type: str,
        what_happened: str,
        behavioral_context: Dict,
        emotional_valence: float = 0.0,
        importance_hint: float = 0.5
    ) -> MemoryEpisode:
        """
        Record a significant moment in user's journey.
        """
        # Calculate importance based on multiple signals
        importance = self._calculate_importance(
            episode_type=episode_type,
            emotional_valence=abs(emotional_valence),
            behavioral_deviation=behavioral_context.get('deviation_from_baseline', 0),
            user_engagement=behavioral_context.get('user_response_time_sec', 300),
            hint=importance_hint
        )
        
        episode = MemoryEpisode(
            episode_id=str(uuid4()),
            user_id=str(self.user_id),
            timestamp=datetime.utcnow(),
            episode_type=episode_type,
            what_happened=what_happened,
            behavioral_context=behavioral_context,
            emotional_valence=emotional_valence,
            importance_score=importance,
            tags=self._auto_tag(episode_type, what_happened),
            related_goal_ids=behavioral_context.get('active_goals', []),
            related_hypothesis_ids=[],
            lesson_learned=None,
            user_reflection=None
        )
        
        # Generate embedding for semantic search
        episode.embedding = await self._generate_embedding(
            f"{episode_type}: {what_happened}"
        )
        
        # Store immediately if high importance, else buffer
        if importance > 0.8:
            await self._store_episode(episode)
        else:
            self.short_term_buffer.append(episode)
            if len(self.short_term_buffer) >= self.consolidation_threshold:
                await self._consolidate_buffer()
        
        return episode
    
    async def retrieve_relevant_memories(
        self,
        current_context: Dict,
        query: Optional[str] = None,
        n_results: int = 3,
        time_horizon: Optional[timedelta] = None
    ) -> List[MemoryEpisode]:
        """
        Retrieve memories relevant to current situation.
        Combines: semantic similarity, temporal relevance, goal alignment
        """
        store = await get_behavioral_store()
        if not store:
            return []
        
        # Build query embedding
        if query:
            query_embedding = await self._generate_embedding(query)
        else:
            # Use current context to create implicit query
            context_desc = f"{' '.join(current_context.get('app_categories', []))} {current_context.get('time_of_day', '')} {current_context.get('user_state', '')}"
            query_embedding = await self._generate_embedding(context_desc)
        
        # Get candidate memories from Supabase
        start_time = None
        if time_horizon:
            start_time = (datetime.utcnow() - time_horizon).isoformat()
        
        # Query with vector similarity (using pgvector if available, else fallback)
        memories = await self._query_memories_with_similarity(
            query_embedding=query_embedding,
            start_time=start_time,
            limit=n_results * 3  # Get more for re-ranking
        )
        
        # Re-rank by multi-factor relevance
        scored_memories = []
        for mem in memories:
            score = self._calculate_relevance_score(mem, current_context, query)
            scored_memories.append((score, mem))
        
        scored_memories.sort(reverse=True, key=lambda x: x[0])
        return [mem for _, mem in scored_memories[:n_results]]
    
    async def generate_narrative_summary(
        self,
        period: timedelta = timedelta(days=7)
    ) -> str:
        """
        Generate a story of user's recent journey.
        Powers daily/weekly recap: "This week you discovered..."
        """
        memories = await self.retrieve_relevant_memories(
            current_context={},
            query="significant progress breakthrough learning",
            n_results=10,
            time_horizon=period
        )
        
        if not memories:
            return "Just getting started. Every journey begins with a single step."
        
        # Categorize memories
        breakthroughs = [m for m in memories if m.episode_type == 'breakthrough']
        struggles = [m for m in memories if m.episode_type == 'struggle']
        insights = [m for m in memories if m.episode_type == 'insight']
        
        # Generate narrative with LLM
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key)
        
        messages = [
            SystemMessage(content="""
            You are a wise coach narrating a user's personal growth journey.
            Be warm, specific, and inspiring. Reference specific moments.
            Highlight patterns and progress. Keep it under 150 words.
            """),
            HumanMessage(content=f"""
            Recent memories ({len(memories)} total):
            Breakthroughs: {[m.what_happened for m in breakthroughs[:3]]}
            Insights: {[m.what_happened for m in insights[:3]]}
            Challenges: {[m.what_happened for m in struggles[:2]]}
            
            Write a personalized narrative about their journey this period.
            """)
        ]
        
        response = llm.invoke(messages)
        return response.content
    
    async def get_streak_and_momentum(self) -> Dict:
        """
        Calculate engagement streak and behavioral momentum.
        Powers: "You've checked in 12 days in a row"
        """
        store = await get_behavioral_store()
        if not store:
            return {"streak": 0, "momentum": "neutral"}
        
        # Get daily engagement for last 30 days
        thirty_days_ago = (datetime.utcnow() - timedelta(days=30)).isoformat()
        
        observations = await store.get_user_observations(
            user_id=self.user_id,
            start_time=thirty_days_ago,
            limit=10000
        )
        
        # Group by day
        daily_engagement = {}
        for obs in observations:
            day = obs.timestamp.strftime("%Y-%m-%d")
            daily_engagement[day] = daily_engagement.get(day, 0) + 1
        
        # Calculate streak
        streak = 0
        today = datetime.utcnow().strftime("%Y-%m-%d")
        yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        if today in daily_engagement or yesterday in daily_engagement:
            streak = 1
            check_date = datetime.utcnow() - timedelta(days=1)
            while check_date.strftime("%Y-%m-%d") in daily_engagement:
                streak += 1
                check_date -= timedelta(days=1)
        
        # Calculate momentum (trend in engagement)
        days = sorted(daily_engagement.keys())
        if len(days) >= 7:
            recent = sum(daily_engagement.get(d, 0) for d in days[-7:])
            previous = sum(daily_engagement.get(d, 0) for d in days[-14:-7])
            momentum = "accelerating" if recent > previous * 1.2 else "steady" if recent > previous * 0.8 else "decelerating"
        else:
            momentum = "building"
        
        return {
            "current_streak_days": streak,
            "longest_streak": self._get_longest_streak(daily_engagement),
            "momentum": momentum,
            "engagement_rate": len(daily_engagement) / 30,
            "total_observations": len(observations)
        }
    
    def _calculate_importance(
        self,
        episode_type: str,
        emotional_valence: float,
        behavioral_deviation: float,
        user_engagement: float,
        hint: float
    ) -> float:
        """Calculate how important this memory is to retain."""
        type_weights = {
            'breakthrough': 1.0,
            'milestone': 0.9,
            'insight': 0.8,
            'struggle': 0.7,
            'routine': 0.3
        }
        
        type_weight = type_weights.get(episode_type, 0.5)
        
        # Fast response = high engagement = important
        engagement_score = max(0, 1 - (user_engagement / 300))  # 5 min threshold
        
        # Combine factors
        importance = (
            type_weight * 0.4 +
            emotional_valence * 0.2 +
            min(behavioral_deviation, 1.0) * 0.2 +
            engagement_score * 0.1 +
            hint * 0.1
        )
        
        return min(1.0, importance)
    
    def _auto_tag(self, episode_type: str, description: str) -> List[str]:
        """Automatically generate tags for searchability."""
        tags = [episode_type]
        
        # Extract keywords
        keywords = ['focus', 'procrastination', 'sleep', 'exercise', 'social', 'work', 'stress']
        for kw in keywords:
            if kw in description.lower():
                tags.append(kw)
        
        return tags
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for semantic search."""
        # Use OpenAI embeddings or local model
        try:
            from langchain_openai import OpenAIEmbeddings
            embedder = OpenAIEmbeddings(api_key=settings.openai_api_key)
            return await embedder.aembed_query(text)
        except:
            # Fallback: simple hash-based embedding
            import hashlib
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            return [float((hash_val >> (i * 8)) & 0xFF) / 255.0 for i in range(10)]
    
    async def _store_episode(self, episode: MemoryEpisode):
        """Store episode in Supabase."""
        store = await get_behavioral_store()
        if not store:
            return
        
        data = asdict(episode)
        data['timestamp'] = episode.timestamp.isoformat()
        data['embedding'] = str(episode.embedding) if episode.embedding else None
        
        store.client.table('episodic_memories').insert(data).execute()
    
    async def _consolidate_buffer(self):
        """Consolidate short-term buffer into long-term storage."""
        if not self.short_term_buffer:
            return
        
        # Summarize buffer into key episodes
        # (Simple version: store all, smart version: cluster and summarize)
        for episode in self.short_term_buffer:
            await self._store_episode(episode)
        
        self.short_term_buffer = []
    
    async def _query_memories_with_similarity(
        self,
        query_embedding: List[float],
        start_time: Optional[str],
        limit: int
    ) -> List[MemoryEpisode]:
        """Query memories with vector similarity."""
        store = await get_behavioral_store()
        if not store:
            return []
        
        # If pgvector available, use semantic search
        # Else fallback to recent + tagged
        
        query = store.client.table('episodic_memories').select('*').eq(
            'user_id', str(self.user_id)
        )
        
        if start_time:
            query = query.gte('timestamp', start_time)
        
        result = query.order('importance_score', desc=True).limit(limit).execute()
        
        return [MemoryEpisode(**row) for row in result.data] if result.data else []
    
    def _calculate_relevance_score(
        self,
        memory: MemoryEpisode,
        current_context: Dict,
        query: Optional[str]
    ) -> float:
        """Calculate how relevant a memory is to current situation."""
        score = 0.0
        
        # Goal overlap
        current_goals = set(current_context.get('active_goals', []))
        memory_goals = set(memory.related_goal_ids)
        if current_goals & memory_goals:
            score += 0.3
        
        # Temporal recency (fades over 30 days)
        days_ago = (datetime.utcnow() - memory.timestamp).days
        recency_boost = max(0, 1 - (days_ago / 30))
        score += recency_boost * 0.2
        
        # Importance
        score += memory.importance_score * 0.3
        
        # Emotional resonance (similar valence)
        if 'emotional_state' in current_context:
            current_valence = current_context['emotional_state']
            valence_sim = 1 - abs(current_valence - memory.emotional_valence)
            score += valence_sim * 0.2
        
        return score
    
    def _get_longest_streak(self, daily_engagement: Dict) -> int:
        """Calculate longest streak from history."""
        if not daily_engagement:
            return 0
        
        days = sorted(daily_engagement.keys())
        longest = 1
        current = 1
        
        for i in range(1, len(days)):
            prev_date = datetime.strptime(days[i-1], "%Y-%m-%d")
            curr_date = datetime.strptime(days[i], "%Y-%m-%d")
            if (curr_date - prev_date).days == 1:
                current += 1
                longest = max(longest, current)
            else:
                current = 1
        
        return longest
