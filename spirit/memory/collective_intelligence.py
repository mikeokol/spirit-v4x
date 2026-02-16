"""
Collective intelligence: Anonymized patterns across users.
Creates 'people like you' insights without compromising privacy.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from uuid import UUID
from dataclasses import dataclass
import hashlib

from spirit.db.supabase_client import get_behavioral_store
from spirit.config import settings


@dataclass
class BehavioralArchetype:
    """
    A cluster of users with similar behavioral patterns.
    Users flow between archetypes as they evolve.
    """
    archetype_id: str
    name: str  # e.g., "Morning Warrior", "Night Owl", "Chaotic Creative"
    description: str
    
    # Defining characteristics
    key_patterns: Dict[str, Tuple[float, float]]  # feature: (mean, std)
    
    # Success patterns
    effective_interventions: List[Dict]  # What works for this type
    common_struggles: List[str]
    typical_progression: List[str]  # Archetypes they tend to evolve into
    
    # Size
    population_size: int
    avg_goal_achievement_rate: float


class CollectiveIntelligenceEngine:
    """
    Learns from anonymized patterns across all Spirit users.
    Powers: "People like you often...", "The most successful approach is..."
    """
    
    def __init__(self):
        self.min_users_for_insight = 10  # Privacy threshold
        self.similarity_threshold = 0.7
        
    async def get_user_archetype(self, user_id: UUID) -> Optional[BehavioralArchetype]:
        """
        Determine which behavioral archetype a user belongs to.
        Archetypes are dynamic and evolve as users change.
        """
        # Get user's behavioral fingerprint
        fingerprint = await self._compute_behavioral_fingerprint(user_id)
        
        if not fingerprint:
            return None
        
        # Match to existing archetype or create new
        archetypes = await self._load_archetypes()
        
        best_match = None
        best_score = 0
        
        for archetype in archetypes:
            score = self._calculate_similarity(fingerprint, archetype.key_patterns)
            if score > best_score and score > self.similarity_threshold:
                best_score = score
                best_match = archetype
        
        if best_match:
            # Update archetype with this user's data (anonymized aggregation)
            await self._update_archetype_stats(best_match, fingerprint)
            return best_match
        
        # Create new archetype if no good match
        return await self._create_archetype_from_user(user_id, fingerprint)
    
    async def get_peer_insights(
        self,
        user_id: UUID,
        context: str  # 'focus', 'sleep', 'exercise', etc.
    ) -> Dict[str, Any]:
        """
        Get anonymized insights from similar users.
        """
        archetype = await self.get_user_archetype(user_id)
        
        if not archetype or archetype.population_size < self.min_users_for_insight:
            return {
                "insight_available": False,
                "reason": "insufficient_peer_data",
                "message": "Keep using Spirit to unlock community insights"
            }
        
        # Find what worked for this archetype
        effective = [
            i for i in archetype.effective_interventions 
            if i.get('success_rate', 0) > 0.6
        ]
        
        # Get trend data
        trend = await self._get_archetype_trend(archetype.archetype_id, context)
        
        return {
            "insight_available": True,
            "your_archetype": archetype.name,
            "archetype_description": archetype.description,
            "peers_like_you": archetype.population_size,
            "their_success_rate": archetype.avg_goal_achievement_rate,
            "what_worked_for_them": effective[:3],
            "common_challenges": archetype.common_struggles[:3],
            "trend": trend,
            "privacy_note": "All insights are anonymized and aggregated"
        }
    
    async def predict_with_peer_data(
        self,
        user_id: UUID,
        proposed_action: str
    ) -> Dict[str, Any]:
        """
        Predict outcome of an action based on similar users' experiences.
        """
        archetype = await self.get_user_archetype(user_id)
        
        if not archetype or archetype.population_size < self.min_users_for_insight:
            return {
                "prediction_available": False,
                "confidence": "low",
                "estimated_success": 0.5,
                "reason": "no_peer_data"
            }
        
        # Find similar interventions in archetype history
        similar = [
            i for i in archetype.effective_interventions
            if self._intervention_similarity(i.get('type', ''), proposed_action) > 0.8
        ]
        
        if not similar:
            return {
                "prediction_available": False,
                "confidence": "low",
                "estimated_success": 0.5,
                "reason": "novel_intervention"
            }
        
        # Calculate expected outcome
        success_rate = sum(s.get('success', 0) for s in similar) / len(similar)
        avg_improvement = sum(s.get('improvement', 0) for s in similar) / len(similar)
        
        # Compare to baseline (what happens without intervention)
        baseline = 0.3  # Assume 30% natural success
        lift = success_rate - baseline
        
        return {
            "prediction_available": True,
            "confidence": "medium" if len(similar) > 20 else "low",
            "estimated_success_rate": success_rate,
            "expected_improvement": avg_improvement,
            "comparison_to_baseline": f"{lift*100:+.0f}%",
            "based_on_n_peers": len(similar),
            "key_success_factors": self._extract_success_factors(similar)
        }
    
    async def get_evolution_path(self, user_id: UUID) -> Dict[str, Any]:
        """
        Show likely progression paths based on archetype transitions.
        """
        archetype = await self.get_user_archetype(user_id)
        
        if not archetype:
            return {"paths_available": False}
        
        paths = []
        for next_archetype_name in archetype.typical_progression[:3]:
            next_arch = await self._get_archetype_by_name(next_archetype_name)
            if next_arch:
                paths.append({
                    "stage": next_archetype_name,
                    "description": next_arch.description,
                    "achievement_rate": next_arch.avg_goal_achievement_rate,
                    "how_to_get_there": self._generate_transition_advice(
                        archetype, next_arch
                    )
                })
        
        return {
            "paths_available": True,
            "current_stage": archetype.name,
            "possible_next_stages": paths,
            "typical_journey_duration": "30-60 days"
        }
    
    async def _compute_behavioral_fingerprint(
        self,
        user_id: UUID
    ) -> Optional[Dict[str, float]]:
        """
        Create anonymized behavioral signature.
        Only uses patterns, never content.
        """
        store = await get_behavioral_store()
        if not store:
            return None
        
        # Get last 14 days of observations
        two_weeks_ago = (datetime.utcnow() - timedelta(days=14)).isoformat()
        observations = await store.get_user_observations(
            user_id=user_id,
            start_time=two_weeks_ago,
            limit=5000
        )
        
        if len(observations) < 50:
            return None  # Not enough data
        
        # Extract pattern features
        features = {
            'morning_productivity_ratio': 0.0,
            'evening_usage_ratio': 0.0,
            'context_switch_frequency': 0.0,
            'deep_work_consistency': 0.0,
            'social_media_dependency': 0.0,
            'notification_reactivity': 0.0,
            'weekend_weekday_similarity': 0.0
        }
        
        # Calculate each feature
        morning_sessions = [o for o in observations if 6 <= o.timestamp.hour <= 12]
        evening_sessions = [o for o in observations if 18 <= o.timestamp.hour <= 23]
        
        if morning_sessions:
            productive_morning = sum(
                1 for o in morning_sessions 
                if o.behavior.get('app_category') in ['productivity', 'health']
            )
            features['morning_productivity_ratio'] = productive_morning / len(morning_sessions)
        
        if evening_sessions:
            features['evening_usage_ratio'] = len(evening_sessions) / len(observations)
        
        # Context switches
        total_switches = sum(o.behavior.get('app_switches_5min', 0) for o in observations)
        features['context_switch_frequency'] = total_switches / len(observations)
        
        # Deep work consistency
        deep_sessions = [o for o in observations if o.behavior.get('session_type') == 'deep_work']
        features['deep_work_consistency'] = len(deep_sessions) / len(observations)
        
        # Social media
        social_sessions = [o for o in observations if o.behavior.get('app_category') == 'social_media']
        features['social_media_dependency'] = len(social_sessions) / len(observations)
        
        return features
    
    def _calculate_similarity(
        self,
        fingerprint: Dict[str, float],
        archetype_patterns: Dict[str, Tuple[float, float]]
    ) -> float:
        """Calculate similarity between user and archetype."""
        if not fingerprint or not archetype_patterns:
            return 0.0
        
        similarities = []
        for feature, user_val in fingerprint.items():
            if feature in archetype_patterns:
                arch_mean, arch_std = archetype_patterns[feature]
                # Z-score similarity
                if arch_std > 0:
                    z_score = abs(user_val - arch_mean) / arch_std
                    sim = max(0, 1 - (z_score / 2))  # 2 std devs = 0 similarity
                    similarities.append(sim)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    async def _load_archetypes(self) -> List[BehavioralArchetype]:
        """Load archetypes from database."""
        store = await get_behavioral_store()
        if not store:
            return []
        
        result = store.client.table('behavioral_archetypes').select('*').execute()
        
        archetypes = []
        for row in result.data if result.data else []:
            archetypes.append(BehavioralArchetype(
                archetype_id=row['archetype_id'],
                name=row['name'],
                description=row['description'],
                key_patterns=row.get('key_patterns', {}),
                effective_interventions=row.get('effective_interventions', []),
                common_struggles=row.get('common_struggles', []),
                typical_progression=row.get('typical_progression', []),
                population_size=row.get('population_size', 0),
                avg_goal_achievement_rate=row.get('avg_goal_achievement_rate', 0.0)
            ))
        
        return archetypes
    
    async def _update_archetype_stats(
        self,
        archetype: BehavioralArchetype,
        fingerprint: Dict[str, float]
    ):
        """Anonymously update archetype statistics with new user data."""
        # Running average update
        n = archetype.population_size
        
        for feature, value in fingerprint.items():
            if feature in archetype.key_patterns:
                old_mean, old_std = archetype.key_patterns[feature]
                # Update mean
                new_mean = (old_mean * n + value) / (n + 1)
                # Update std (simplified)
                new_std = ((old_std**2 * n + (value - new_mean)**2) / (n + 1)) ** 0.5
                archetype.key_patterns[feature] = (new_mean, new_std)
        
        archetype.population_size += 1
        
        # Persist update
        store = await get_behavioral_store()
        if store:
            store.client.table('behavioral_archetypes').update({
                'key_patterns': archetype.key_patterns,
                'population_size': archetype.population_size
            }).eq('archetype_id', archetype.archetype_id).execute()
    
    async def _create_archetype_from_user(
        self,
        user_id: UUID,
        fingerprint: Dict[str, float]
    ) -> BehavioralArchetype:
        """Create new archetype from user's unique pattern."""
        # Generate name based on dominant traits
        name = self._generate_archetype_name(fingerprint)
        
        archetype = BehavioralArchetype(
            archetype_id=str(uuid4()),
            name=name,
            description=f"Users with pattern: {self._describe_fingerprint(fingerprint)}",
            key_patterns={k: (v, 0.1) for k, v in fingerprint.items()},
            effective_interventions=[],
            common_struggles=[],
            typical_progression=[],
            population_size=1,
            avg_goal_achievement_rate=0.5
        )
        
        # Store
        store = await get_behavioral_store()
        if store:
            store.client.table('behavioral_archetypes').insert({
                'archetype_id': archetype.archetype_id,
                'name': archetype.name,
                'description': archetype.description,
                'key_patterns': archetype.key_patterns,
                'population_size': 1
            }).execute()
        
        return archetype
    
    def _generate_archetype_name(self, fingerprint: Dict[str, float]) -> str:
        """Generate descriptive name from fingerprint."""
        if fingerprint.get('morning_productivity_ratio', 0) > 0.7:
            return "Morning Warrior"
        elif fingerprint.get('evening_usage_ratio', 0) > 0.4:
            return "Night Owl"
        elif fingerprint.get('context_switch_frequency', 0) > 5:
            return "Chaotic Creative"
        elif fingerprint.get('deep_work_consistency', 0) > 0.5:
            return "Deep Diver"
        else:
            return "Steady Seeker"
    
    def _describe_fingerprint(self, fingerprint: Dict[str, float]) -> str:
        """Generate human-readable description of pattern."""
        traits = []
        if fingerprint.get('morning_productivity_ratio', 0) > 0.6:
            traits.append("morning-focused")
        if fingerprint.get('deep_work_consistency', 0) > 0.4:
            traits.append("sustained attention")
        if fingerprint.get('context_switch_frequency', 0) > 3:
            traits.append("high variety")
        return ", ".join(traits) if traits else "balanced pattern"
    
    async def _get_archetype_trend(self, archetype_id: str, context: str) -> Dict:
        """Get trend data for this archetype in specific context."""
        # Placeholder - would query time-series data
        return {"direction": "stable", "magnitude": "small"}
    
    async def _get_archetype_by_name(self, name: str) -> Optional[BehavioralArchetype]:
        """Load specific archetype by name."""
        archetypes = await self._load_archetypes()
        for a in archetypes:
            if a.name == name:
                return a
        return None
    
    def _intervention_similarity(self, a: str, b: str) -> float:
        """Calculate similarity between intervention types."""
        # Simple string similarity
        if a == b:
            return 1.0
        if a in b or b in a:
            return 0.8
        return 0.0
    
    def _extract_success_factors(self, interventions: List[Dict]) -> List[str]:
        """Extract common success factors from intervention history."""
        factors = []
        timing_counts = {}
        
        for i in interventions:
            timing = i.get('timing', 'unknown')
            timing_counts[timing] = timing_counts.get(timing, 0) + 1
        
        if timing_counts:
            best_timing = max(timing_counts, key=timing_counts.get)
            factors.append(f"Best timing: {best_timing}")
        
        return factors
    
    def _generate_transition_advice(
        self,
        current: BehavioralArchetype,
        target: BehavioralArchetype
    ) -> List[str]:
        """Generate advice for moving between archetypes."""
        advice = []
        
        # Compare key differences
        for feature, (curr_mean, _) in current.key_patterns.items():
            if feature in target.key_patterns:
                target_mean, _ = target.key_patterns[feature]
                diff = target_mean - curr_mean
                
                if abs(diff) > 0.2:
                    direction = "increase" if diff > 0 else "decrease"
                    advice.append(f"{direction} your {feature.replace('_', ' ')}")
        
        return advice[:3]
