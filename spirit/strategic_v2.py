"""
Strategic Layer v2: Behaviorally-informed long-term planning.
Unlocks and evolves based on behavioral stability, not just execution checkboxes.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Literal, Tuple
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select, func
from sqlalchemy import Integer

from spirit.models import Execution, Goal, GoalState
from spirit.db.supabase_client import get_behavioral_store
from spirit.memory.episodic_memory import EpisodicMemorySystem
from spirit.services.causal_inference import CausalInferenceEngine


class StrategicMaturityLevel(str, Enum):
    """Stages of strategic capability based on behavioral evidence."""
    NOVICE = "novice"           # Just starting, no behavioral data
    EMERGING = "emerging"       # Some data, patterns unclear
    STABLE = "stable"           # Consistent behavioral patterns
    STRATEGIC = "strategic"     # Full strategic features unlocked
    ADAPTIVE = "adaptive"       # Self-modifying strategies based on causal models


@dataclass
class BehavioralStabilityMetrics:
    """Quantified behavioral stability for strategic decisions."""
    consistency_score: float  # 0-1, temporal consistency
    predictability: float     # 0-1, how well behavior follows patterns
    resilience: float         # 0-1, recovery from disruptions
    intentionality: float     # 0-1, ratio of deliberate vs reactive behavior
    data_quality: float       # 0-1, completeness and recency of data


class StrategicUnlockEngine:
    """
    v2: Unlocks strategic features based on behavioral stability,
    not just execution counts. Uses Supabase behavioral data + SQLite executions.
    """
    
    # Thresholds
    MIN_BEHAVIORAL_DAYS = 14  # Need 2 weeks of behavioral data
    MIN_CONSISTENCY = 0.6     # Behavioral consistency score
    MIN_INTENTIONALITY = 0.5  # Must be mostly deliberate, not reactive
    
    def __init__(self, user_id: int):
        self.user_id = user_id
    
    async def check_strategic_unlock_v2(self, db: AsyncSession) -> Dict[str, any]:
        """
        Comprehensive unlock check using behavioral + execution data.
        Returns detailed reasoning, not just boolean.
        """
        # 1. Check basic execution (original criteria, softened)
        execution_ok, execution_details = await self._check_execution_base(db)
        
        # 2. Check behavioral stability (new)
        behavioral_ok, behavioral_details = await self._check_behavioral_stability()
        
        # 3. Check episodic memory (journey continuity)
        memory_ok, memory_details = await self._check_journey_continuity()
        
        # 4. Calculate maturity level
        maturity = self._calculate_maturity(
            execution_ok, behavioral_ok, memory_ok,
            execution_details, behavioral_details
        )
        
        # Determine what's unlocked
        unlocked_features = self._get_unlocked_features(maturity)
        
        return {
            "can_unlock_strategic": maturity.value in [StrategicMaturityLevel.STRATEGIC, StrategicMaturityLevel.ADAPTIVE],
            "maturity_level": maturity.value,
            "progress": {
                "execution": execution_details,
                "behavioral": behavioral_details,
                "memory": memory_details
            },
            "unlocked_features": unlocked_features,
            "next_requirements": self._get_next_requirements(maturity, execution_details, behavioral_details),
            "estimated_days_to_strategic": self._estimate_timeline(maturity, behavioral_details)
        }
    
    async def _check_execution_base(self, db: AsyncSession) -> Tuple[bool, Dict]:
        """Original execution check, but with detailed reporting."""
        goal = await self._get_active_goal(db)
        if not goal:
            return False, {"error": "no_active_goal", "rate": 0, "streak": 0}
        
        # 30-day rate
        since = datetime.utcnow() - timedelta(days=30)
        res = await db.execute(
            select(func.count(Execution.id), func.sum(Execution.executed.cast(Integer)))
            .where(Execution.goal_id == goal.id)
            .where(Execution.day >= since.date())
        )
        total, done = res.one()
        rate = (done / total) if total else 0
        
        # 5-day streak
        streak_ok, streak_details = await self._check_streak(db, goal.id)
        
        # Softened: allow 60% rate if behavioral data is strong
        base_ok = rate >= 0.6 or (rate >= 0.5 and streak_ok)
        
        return base_ok, {
            "goal_id": goal.id,
            "execution_rate_30d": rate,
            "executions_count": total,
            "current_streak": streak_details.get("current_streak", 0),
            "streak_requirement_met": streak_ok,
            "soft_threshold_used": rate < 0.7
        }
    
    async def _check_behavioral_stability(self) -> Tuple[bool, Dict]:
        """
        Check if user has stable, predictable behavioral patterns.
        This is the key new criterion for strategic unlock.
        """
        store = await get_behavioral_store()
        if not store:
            return False, {"error": "no_behavioral_store", "available": False}
        
        # Get 14 days of behavioral data
        since = (datetime.utcnow() - timedelta(days=self.MIN_BEHAVIORAL_DAYS)).isoformat()
        observations = await store.get_user_observations(
            user_id=self.user_id,
            start_time=since,
            limit=10000
        )
        
        if len(observations) < 50:  # Need meaningful volume
            return False, {
                "available": True,
                "days_data": self.MIN_BEHAVIORAL_DAYS,
                "observations_count": len(observations),
                "sufficient": False,
                "needed": 50
            }
        
        # Calculate stability metrics
        metrics = self._calculate_stability_metrics(observations)
        
        # Check thresholds
        stability_ok = (
            metrics.consistency_score >= self.MIN_CONSISTENCY and
            metrics.intentionality >= self.MIN_INTENTIONALITY and
            metrics.data_quality > 0.7
        )
        
        return stability_ok, {
            "available": True,
            "days_data": self.MIN_BEHAVIORAL_DAYS,
            "observations_count": len(observations),
            "sufficient": True,
            "metrics": {
                "consistency": metrics.consistency_score,
                "predictability": metrics.predictability,
                "resilience": metrics.resilience,
                "intentionality": metrics.intentionality,
                "data_quality": metrics.data_quality
            },
            "thresholds_met": {
                "consistency": metrics.consistency_score >= self.MIN_CONSISTENCY,
                "intentionality": metrics.intentionality >= self.MIN_INTENTIONALITY
            }
        }
    
    async def _check_journey_continuity(self) -> Tuple[bool, Dict]:
        """Check if user has established episodic memory continuity."""
        memory = EpisodicMemorySystem(self.user_id)
        
        # Get streak and engagement
        streak_data = await memory.get_streak_and_momentum()
        
        # Need at least some meaningful episodes
        recent = await memory.retrieve_relevant_memories(
            current_context={},
            time_horizon=timedelta(days=14),
            n_results=10
        )
        
        continuity_ok = (
            streak_data["current_streak_days"] >= 3 or
            len(recent) >= 3
        )
        
        return continuity_ok, {
            "engagement_streak": streak_data["current_streak_days"],
            "momentum": streak_data["momentum"],
            "significant_episodes_14d": len(recent),
            "continuity_established": continuity_ok
        }
    
    def _calculate_stability_metrics(
        self,
        observations: List
    ) -> BehavioralStabilityMetrics:
        """Calculate behavioral stability from observations."""
        # Group by day
        daily_patterns = {}
        for obs in observations:
            day = obs.timestamp.strftime("%Y-%m-%d")
            if day not in daily_patterns:
                daily_patterns[day] = []
            daily_patterns[day].append(obs)
        
        # Consistency: same patterns day-to-day
        if len(daily_patterns) < 7:
            consistency = 0.3
        else:
            # Compare consecutive days
            days = sorted(daily_patterns.keys())
            similarities = []
            for i in range(len(days)-1):
                sim = self._compare_days(daily_patterns[days[i]], daily_patterns[days[i+1]])
                similarities.append(sim)
            consistency = sum(similarities) / len(similarities) if similarities else 0.5
        
        # Predictability: variance in key metrics
        all_productive = []
        for day_obs in daily_patterns.values():
            productive = sum(
                o.behavior.get('session_duration_sec', 0) 
                for o in day_obs 
                if o.behavior.get('app_category') == 'productivity'
            ) / 60  # minutes
            all_productive.append(productive)
        
        if len(all_productive) > 1:
            mean_prod = sum(all_productive) / len(all_productive)
            variance = sum((p - mean_prod)**2 for p in all_productive) / len(all_productive)
            predictability = max(0, 1 - (variance / (mean_prod**2 + 1)))
        else:
            predictability = 0.5
        
        # Resilience: recovery after "bad" days
        # Simplified: check if productive time returns to mean after low days
        resilience = 0.6  # Default moderate
        
        # Intentionality: ratio of deep work to reactive usage
        total_deep = sum(
            1 for o in observations 
            if o.behavior.get('session_type') == 'deep_work'
        )
        total_reactive = sum(
            1 for o in observations
            if o.behavior.get('app_category') in ['social_media', 'entertainment']
            and o.behavior.get('entry_point') == 'notification'
        )
        
        total_intentional = total_deep
        total_reactive = max(total_reactive, 1)  # Avoid div by zero
        intentionality = total_intentional / (total_intentional + total_reactive * 0.5)
        
        # Data quality
        data_quality = min(1.0, len(observations) / 200)  # 200+ observations = full quality
        
        return BehavioralStabilityMetrics(
            consistency_score=consistency,
            predictability=predictability,
            resilience=resilience,
            intentionality=intentionality,
            data_quality=data_quality
        )
    
    def _compare_days(self, day1: List, day2: List) -> float:
        """Calculate similarity between two days of observations."""
        # Simple: compare app category distributions
        cats1 = {}
        cats2 = {}
        
        for o in day1:
            cat = o.behavior.get('app_category', 'unknown')
            cats1[cat] = cats1.get(cat, 0) + 1
        
        for o in day2:
            cat = o.behavior.get('app_category', 'unknown')
            cats2[cat] = cats2.get(cat, 0) + 1
        
        # Cosine similarity of category vectors
        all_cats = set(cats1.keys()) | set(cats2.keys())
        if not all_cats:
            return 0.5
        
        dot = sum(cats1.get(c, 0) * cats2.get(c, 0) for c in all_cats)
        norm1 = sum(cats1.get(c, 0)**2 for c in all_cats) ** 0.5
        norm2 = sum(cats2.get(c, 0)**2 for c in all_cats) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.5
        
        return dot / (norm1 * norm2)
    
    async def _check_streak(self, db: AsyncSession, goal_id: int) -> Tuple[bool, Dict]:
        """Check 5-day execution streak."""
        streak_since = datetime.utcnow().date() - timedelta(days=5)
        streak_res = await db.execute(
            select(Execution.day, Execution.executed)
            .where(Execution.goal_id == goal_id)
            .where(Execution.day >= streak_since)
            .order_by(Execution.day)
        )
        days = streak_res.all()
        
        if len(days) != 5:
            return False, {"current_streak": len(days), "complete": False}
        
        all_executed = all(d.executed for d in days)
        
        # Also calculate current streak length
        current_streak = 0
        for d in reversed(days):
            if d.executed:
                current_streak += 1
            else:
                break
        
        return all_executed, {
            "current_streak": current_streak,
            "complete": True,
            "all_executed": all_executed
        }
    
    async def _get_active_goal(self, db: AsyncSession) -> Optional[Goal]:
        """Get user's active goal."""
        # Simplified - adjust to your actual goal lookup
        result = await db.execute(
            select(Goal).where(Goal.user_id == self.user_id).where(Goal.state == GoalState.active)
        )
        return result.scalar_one_or_none()
    
    def _calculate_maturity(
        self,
        execution_ok: bool,
        behavioral_ok: bool,
        memory_ok: bool,
        exec_details: Dict,
        behav_details: Dict
    ) -> StrategicMaturityLevel:
        """Determine maturity level from all criteria."""
        # Count successes
        score = sum([execution_ok, behavioral_ok, memory_ok])
        
        if score == 0:
            return StrategicMaturityLevel.NOVICE
        if score == 1:
            return StrategicMaturityLevel.EMERGING
        if score == 2:
            # Check if which one is missing
            if not execution_ok and behavioral_ok:
                # Strong behavioral, weak execution = stable but not strategic
                return StrategicMaturityLevel.STABLE
            return StrategicMaturityLevel.STABLE
        if score == 3:
            # All criteria met
            # Check if we have causal models for adaptive
            if behav_details.get("metrics", {}).get("predictability", 0) > 0.8:
                return StrategicMaturityLevel.ADAPTIVE
            return StrategicMaturityLevel.STRATEGIC
        
        return StrategicMaturityLevel.EMERGING
    
    def _get_unlocked_features(self, maturity: StrategicMaturityLevel) -> List[str]:
        """Features available at each maturity level."""
        features = {
            StrategicMaturityLevel.NOVICE: ["daily_check_in", "basic_tracking"],
            StrategicMaturityLevel.EMERGING: ["daily_check_in", "basic_tracking", "weekly_review"],
            StrategicMaturityLevel.STABLE: ["daily_check_in", "basic_tracking", "weekly_review", "pattern_insights"],
            StrategicMaturityLevel.STRATEGIC: [
                "daily_check_in", "basic_tracking", "weekly_review", 
                "pattern_insights", "long_term_goals", "scenario_planning",
                "behavioral_predictions"
            ],
            StrategicMaturityLevel.ADAPTIVE: [
                "daily_check_in", "basic_tracking", "weekly_review",
                "pattern_insights", "long_term_goals", "scenario_planning",
                "behavioral_predictions", "auto_strategy_adjustment",
                "causal_interventions"
            ]
        }
        return features.get(maturity, features[StrategicMaturityLevel.NOVICE])
    
    def _get_next_requirements(
        self,
        current: StrategicMaturityLevel,
        exec_details: Dict,
        behav_details: Dict
    ) -> Dict[str, str]:
        """What user needs to reach next level."""
        next_level = {
            StrategicMaturityLevel.NOVICE: StrategicMaturityLevel.EMERGING,
            StrategicMaturityLevel.EMERGING: StrategicMaturityLevel.STABLE,
            StrategicMaturityLevel.STABLE: StrategicMaturityLevel.STRATEGIC,
            StrategicMaturityLevel.STRATEGIC: StrategicMaturityLevel.ADAPTIVE,
            StrategicMaturityLevel.ADAPTIVE: None
        }[current]
        
        if not next_level:
            return {"message": "Maximum maturity reached. Focus on maintenance."}
        
        requirements = []
        
        if not exec_details.get("streak_requirement_met"):
            requirements.append(f"Complete {5 - exec_details.get('current_streak', 0)} more daily check-ins")
        
        if behav_details.get("sufficient") and not behav_details.get("thresholds_met", {}).get("consistency"):
            requirements.append("Establish more consistent daily routines")
        
        if behav_details.get("sufficient") and not behav_details.get("thresholds_met", {}).get("intentionality"):
            requirements.append("Reduce reactive phone usage, increase intentional sessions")
        
        if not behav_details.get("sufficient"):
            requirements.append(f"Continue using Spirit for {14 - behav_details.get('days_data', 0)} more days")
        
        return {
            "target_level": next_level.value,
            "requirements": requirements,
            "estimated_effort": "1-2 weeks" if len(requirements) <= 2 else "2-4 weeks"
        }
    
    def _estimate_timeline(
        self,
        current: StrategicMaturityLevel,
        behav_details: Dict
    ) -> Optional[int]:
        """Estimate days until strategic unlock."""
        if current.value in [StrategicMaturityLevel.STRATEGIC, StrategicMaturityLevel.ADAPTIVE]:
            return 0
        
        if not behav_details.get("sufficient"):
            # Need more data
            return max(0, 14 - behav_details.get("days_data", 0))
        
        # Have data, need to improve metrics
        if current == StrategicMaturityLevel.EMERGING:
            return 7  # 1 week to establish consistency
        
        if current == StrategicMaturityLevel.STABLE:
            return 14  # 2 weeks to prove strategic capability
        
        return None


class StrategicPlanGenerator:
    """
    Generates long-term strategic plans based on behavioral patterns and goals.
    Only available at STRATEGIC or ADAPTIVE maturity levels.
    """
    
    def __init__(self, user_id: int, maturity: StrategicMaturityLevel):
        self.user_id = user_id
        self.maturity = maturity
    
    async def generate_12_week_plan(self, db: AsyncSession) -> Dict[str, any]:
        """
        Generate 12-week strategic plan based on behavioral data + goals.
        """
        if self.maturity.value not in [StrategicMaturityLevel.STRATEGIC, StrategicMaturityLevel.ADAPTIVE]:
            return {
                "error": "insufficient_maturity",
                "required": "strategic",
                "current": self.maturity.value,
                "message": "Continue building behavioral consistency to unlock strategic planning"
            }
        
        # Get goal
        goal = await self._get_active_goal(db)
        if not goal:
            return {"error": "no_active_goal"}
        
        # Get behavioral insights
        engine = CausalInferenceEngine(self.user_id)
        
        # Find what drives success for this user
        hypotheses = []  # Would query from causal_graph
        
        # Get archetype-based recommendations
        from spirit.memory.collective_intelligence import CollectiveIntelligenceEngine
        collective = CollectiveIntelligenceEngine()
        archetype = await collective.get_user_archetype(self.user_id)
        
        # Build plan
        weeks = []
        for week in range(1, 13):
            week_plan = {
                "week": week,
                "focus": self._determine_week_focus(week, goal, archetype),
                "behavioral_targets": self._set_weekly_targets(week, archetype),
                "experiments": self._design_weekly_experiments(week) if self.maturity == StrategicMaturityLevel.ADAPTIVE else [],
                "review_criteria": self._set_review_criteria(week)
            }
            weeks.append(week_plan)
        
        return {
            "plan_type": "12_week_behavioral_transformation",
            "based_on": {
                "goal": goal.title,
                "archetype": archetype.name if archetype else "unknown",
                "causal_insights": len(hypotheses)
            },
            "weeks": weeks,
            "success_probability": self._estimate_success_probability(archetype, weeks)
        }
    
    def _determine_week_focus(self, week: int, goal: Goal, archetype) -> str:
        """Determine focus area for each week."""
        phases = [
            "Baseline & Pattern Discovery",
            "Foundation Building",
            "Habit Stabilization",
            "First Milestone Push"
        ]
        phase = phases[min((week - 1) // 3, len(phases) - 1)]
        
        if week <= 2:
            return f"{phase}: Establish consistent tracking"
        elif week <= 4:
            return f"{phase}: Build core behavioral routines"
        elif week <= 8:
            return f"{phase}: Optimize and refine patterns"
        else:
            return f"{phase}: Sustain and lock in gains"
    
    def _set_weekly_targets(self, week: int, archetype) -> Dict[str, float]:
        """Set achievable behavioral targets based on archetype."""
        # Would customize based on archetype patterns
        base = {
            "productive_hours_per_day": 3 + (week * 0.2),
            "deep_work_sessions_per_week": 4 + week,
            "evening_screen_time_max": 60 - (week * 3)
        }
        return base
    
    def _design_weekly_experiments(self, week: int) -> List[Dict]:
        """Design A/B tests for adaptive users."""
        experiments = [
            {
                "week": week,
                "hypothesis": "Morning interventions are more effective than evening",
                "test": "randomize_intervention_timing",
                "metric": "engagement_rate",
                "sample_size": 7
            }
        ]
        return experiments
    
    def _set_review_criteria(self, week: int) -> Dict[str, any]:
        """Define success criteria for weekly review."""
        return {
            "minimum_execution_rate": 0.6 + (week * 0.02),
            "behavioral_consistency_threshold": 0.5 + (week * 0.03),
            "checkpoint": week in [4, 8, 12]
        }
    
    def _estimate_success_probability(self, archetype, weeks: List) -> float:
        """Estimate probability of plan success."""
        if not archetype:
            return 0.5
        
        base_rate = archetype.avg_goal_achievement_rate if archetype else 0.5
        
        # Adjust for plan quality
        has_experiments = any(w.get("experiments") for w in weeks)
        if has_experiments:
            base_rate += 0.1
        
        return min(0.95, base_rate)
    
    async def _get_active_goal(self, db: AsyncSession) -> Optional[Goal]:
        """Get user's active goal."""
        result = await db.execute(
            select(Goal).where(Goal.user_id == self.user_id).where(Goal.state == GoalState.active)
        )
        return result.scalar_one_or_none()
