"""
API endpoints for episodic memory and collective intelligence.
Powers daily engagement, streaks, and social proof.
"""

from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer

from spirit.memory.episodic_memory import EpisodicMemorySystem
from spirit.memory.collective_intelligence import CollectiveIntelligenceEngine


router = APIRouter(prefix="/v1/memory", tags=["memory"])
security = HTTPBearer()


async def get_current_user(credentials=Depends(security)) -> UUID:
    """Extract user from JWT."""
    import base64
    import json
    try:
        payload = credentials.credentials.split('.')[1]
        padding = 4 - len(payload) % 4
        if padding != 4:
            payload += '=' * padding
        decoded = base64.urlsafe_b64decode(payload)
        claims = json.loads(decoded)
        return UUID(claims['sub'])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


@router.post("/record")
async def record_memory_episode(
    episode_type: str,  # 'breakthrough', 'struggle', 'insight', 'milestone'
    what_happened: str,
    emotional_valence: float = 0.0,  # -1 to 1
    context: dict = {},
    user_id: UUID = Depends(get_current_user)
):
    """
    Record a significant moment in user's journey.
    Called automatically by intelligence layer, or manually for milestones.
    """
    memory = EpisodicMemorySystem(user_id)
    
    episode = await memory.record_episode(
        episode_type=episode_type,
        what_happened=what_happened,
        behavioral_context=context,
        emotional_valence=emotional_valence
    )
    
    return {
        "episode_id": episode.episode_id,
        "importance_score": episode.importance_score,
        "tags": episode.tags,
        "stored": episode.importance_score > 0.8 or len(memory.short_term_buffer) > 0
    }


@router.get("/narrative")
async def get_personal_narrative(
    period_days: int = 7,
    user_id: UUID = Depends(get_current_user)
):
    """
    Get user's journey narrative for the period.
    Powers daily/weekly recap: "This week you..."
    """
    memory = EpisodicMemorySystem(user_id)
    narrative = await memory.generate_narrative_summary(timedelta(days=period_days))
    
    return {
        "period_days": period_days,
        "narrative": narrative,
        "generated_at": datetime.utcnow().isoformat()
    }


@router.get("/streak")
async def get_engagement_streak(
    user_id: UUID = Depends(get_current_user)
):
    """
    Get current streak and momentum metrics.
    Powers: "12 days in a row", "You're on fire!"
    """
    memory = EpisodicMemorySystem(user_id)
    streak_data = await memory.get_streak_and_momentum()
    
    return {
        **streak_data,
        "user_id": str(user_id),
        "next_milestone": self._next_streak_milestone(streak_data["current_streak_days"])
    }


def _next_streak_milestone(current: int) -> dict:
    """Calculate next streak milestone."""
    milestones = [3, 7, 14, 30, 60, 100]
    for m in milestones:
        if current < m:
            return {"days": m, "progress": current / m}
    return {"days": current + 30, "progress": 1.0}


@router.get("/remember")
async def retrieve_relevant_memories(
    query: Optional[str] = None,
    context: dict = {},
    n_results: int = 3,
    user_id: UUID = Depends(get_current_user)
):
    """
    Retrieve memories relevant to current situation.
    Powers contextual coaching: "This reminds me of when you..."
    """
    memory = EpisodicMemorySystem(user_id)
    
    memories = await memory.retrieve_relevant_memories(
        current_context=context,
        query=query,
        n_results=n_results
    )
    
    return {
        "memories": [
            {
                "when": m.timestamp.isoformat(),
                "what": m.what_happened,
                "type": m.episode_type,
                "importance": m.importance_score,
                "lesson": m.lesson_learned
            }
            for m in memories
        ],
        "query": query,
        "context_match": "semantic" if query else "implicit"
    }


@router.get("/archetype")
async def get_my_archetype(
    user_id: UUID = Depends(get_current_user)
):
    """
    Discover which behavioral archetype you belong to.
    Powers: "You're a Morning Warrior" identity formation.
    """
    collective = CollectiveIntelligenceEngine()
    archetype = await collective.get_user_archetype(user_id)
    
    if not archetype:
        return {
            "archetype_known": False,
            "message": "Keep using Spirit to discover your behavioral type",
            "observations_needed": 50
        }
    
    return {
        "archetype_known": True,
        "your_archetype": archetype.name,
        "description": archetype.description,
        "population": archetype.population_size,
        "success_rate": archetype.avg_goal_achievement_rate,
        "typical_struggles": archetype.common_struggles[:3],
        "evolution_path": archetype.typical_progression[:2]
    }


@router.get("/peer-insights")
async def get_peer_insights(
    context: str,  # 'focus', 'sleep', 'exercise', etc.
    user_id: UUID = Depends(get_current_user)
):
    """
    Get anonymized insights from users similar to you.
    Powers social proof: "People like you often..."
    """
    collective = CollectiveIntelligenceEngine()
    insights = await collective.get_peer_insights(user_id, context)
    
    return insights


@router.post("/predict-outcome")
async def predict_with_peers(
    proposed_action: str,
    user_id: UUID = Depends(get_current_user)
):
    """
    Predict success of an action based on similar users' results.
    Powers: "83% of people like you succeed with this approach"
    """
    collective = CollectiveIntelligenceEngine()
    prediction = await collective.predict_with_peer_data(user_id, proposed_action)
    
    return prediction


@router.get("/evolution")
async def get_evolution_path(
    user_id: UUID = Depends(get_current_user)
):
    """
    Show likely progression paths based on archetype transitions.
    Powers: "Your next stage is Deep Diver - here's how to get there"
    """
    collective = CollectiveIntelligenceEngine()
    paths = await collective.get_evolution_path(user_id)
    
    return paths
