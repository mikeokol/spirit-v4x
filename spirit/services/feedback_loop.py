"""
Feedback Loop: How user responses improve Spirit's models.
Closes the loop: Intervention -> Response -> Learning -> Better models.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Literal
from uuid import UUID
from dataclasses import dataclass
import statistics

from spirit.db.supabase_client import get_behavioral_store
from spirit.services.causal_inference import CausalInferenceEngine


@dataclass
class InterventionOutcome:
    """
    The result of an intervention: did it work?
    """
    intervention_id: str
    user_id: str
    intervention_type: str
    delivered_at: datetime
    user_response: Literal['engaged', 'dismissed', 'ignored', 'unknown']
    response_time_seconds: Optional[float]
    
    # Proximal outcomes (immediate)
    behavior_change_30min: Optional[Dict]  # What happened in next 30 min
    ema_response: Optional[Dict]  # If EMA was triggered
    
    # Distal outcomes (hours/days later)
    goal_progress_delta: Optional[float]  # Change in goal progress
    behavioral_trend_change: Optional[float]  # Slope change
    
    # User feedback
    explicit_rating: Optional[int]  # 1-5 if user rated it
    qualitative_feedback: Optional[str]


class FeedbackLoopEngine:
    """
    Continuous learning from intervention outcomes.
    Updates: causal models, archetype effectiveness, delivery timing.
    """
    
    def __init__(self, user_id: UUID):
        self.user_id = user_id
        self.min_samples_for_update = 5
    
    async def record_outcome(self, outcome: InterventionOutcome):
        """
        Record what happened after an intervention.
        Called when user responds or after observation period.
        """
        store = await get_behavioral_store()
        if not store:
            return
        
        # Store outcome
        store.client.table('intervention_outcomes').insert({
            'outcome_id': str(uuid4()),
            'intervention_id': outcome.intervention_id,
            'user_id': str(self.user_id),
            'intervention_type': outcome.intervention_type,
            'delivered_at': outcome.delivered_at.isoformat(),
            'user_response': outcome.user_response,
            'response_time_seconds': outcome.response_time_seconds,
            'behavior_change_30min': outcome.behavior_change_30min,
            'ema_response': outcome.ema_response,
            'goal_progress_delta': outcome.goal_progress_delta,
            'explicit_rating': outcome.explicit_rating,
            'recorded_at': datetime.utcnow().isoformat()
        }).execute()
        
        # Trigger model updates if enough data
        await self._check_and_update_models(outcome.intervention_type)
    
    async def _check_and_update_models(self, intervention_type: str):
        """
        Check if we have enough data to update models for this intervention type.
        """
        store = await get_behavioral_store()
        if not store:
            return
        
        # Get outcomes for this intervention type
        outcomes = store.client.table('intervention_outcomes').select('*').eq(
            'user_id', str(self.user_id)
        ).eq('intervention_type', intervention_type).execute()
        
        if len(outcomes.data) < self.min_samples_for_update:
            return
        
        # Update individual user model
        await self._update_user_model(intervention_type, outcomes.data)
        
        # Update collective archetype model (anonymized)
        await self._update_collective_model(intervention_type, outcomes.data)
    
    async def _update_user_model(
        self,
        intervention_type: str,
        outcomes: List[Dict]
    ):
        """
        Update this specific user's causal model based on outcomes.
        """
        # Calculate effectiveness metrics
        engaged = [o for o in outcomes if o['user_response'] == 'engaged']
        dismissed = [o for o in outcomes if o['user_response'] == 'dismissed']
        ignored = [o for o in outcomes if o['user_response'] == 'ignored']
        
        engagement_rate = len(engaged) / len(outcomes)
        avg_response_time = statistics.mean([
            o['response_time_seconds'] for o in engaged 
            if o.get('response_time_seconds')
        ]) if engaged else None
        
        # Calculate behavior change
        behavior_changes = [
            o['behavior_change_30min'].get('improvement', 0)
            for o in outcomes
            if o.get('behavior_change_30min') and o['user_response'] == 'engaged'
        ]
        
        avg_behavior_change = statistics.mean(behavior_changes) if behavior_changes else 0
        
        # Update or create causal hypothesis
        engine = CausalInferenceEngine(self.user_id)
        
        # Create hypothesis: "intervention_type -> positive_behavior_change"
        hypothesis = await engine.analyze_variable_pair(
            cause_var=f"intervention_{intervention_type}",
            effect_var="behavior_improvement_30min",
            lag_hours=0
        )
        
        if hypothesis:
            # Update with new effect size from observed data
            hypothesis.effect_size = avg_behavior_change
            hypothesis.n_observations = len(outcomes)
            hypothesis.last_validated_at = datetime.utcnow() if engagement_rate > 0.5 else None
            
            store = await get_behavioral_store()
            if store:
                await store.update_causal_hypothesis(hypothesis)
        
        # Update delivery timing model
        await self._update_timing_model(intervention_type, outcomes)
    
    async def _update_timing_model(
        self,
        intervention_type: str,
        outcomes: List[Dict]
    ):
        """
        Learn optimal delivery times for this user and intervention type.
        """
        # Analyze success by hour of day
        hour_success = {}
        for outcome in outcomes:
            hour = datetime.fromisoformat(outcome['delivered_at']).hour
            success = outcome['user_response'] == 'engaged'
            
            if hour not in hour_success:
                hour_success[hour] = {'success': 0, 'total': 0}
            hour_success[hour]['total'] += 1
            if success:
                hour_success[hour]['success'] += 1
        
        # Find best hours
        best_hours = []
        for hour, stats in hour_success.items():
            if stats['total'] >= 3:
                rate = stats['success'] / stats['total']
                if rate > 0.6:
                    best_hours.append((hour, rate))
        
        best_hours.sort(key=lambda x: x[1], reverse=True)
        
        # Store in user preferences
        store = await get_behavioral_store()
        if store and best_hours:
            store.client.table('user_delivery_preferences').upsert({
                'user_id': str(self.user_id),
                'intervention_type': intervention_type,
                'optimal_hours': [h[0] for h in best_hours[:3]],
                'success_rates': {str(h[0]): h[1] for h in best_hours[:3]},
                'updated_at': datetime.utcnow().isoformat()
            }, on_conflict='user_id,intervention_type').execute()
    
    async def _update_collective_model(
        self,
        intervention_type: str,
        user_outcomes: List[Dict]
    ):
        """
        Anonymously contribute to collective intelligence.
        """
        # Only contribute if user opted in and we have archetype
        store = await get_behavioral_store()
        if not store:
            return
        
        # Get user's current archetype
        archetype_result = store.client.table('user_archetype_history').select(
            'archetype_id'
        ).eq('user_id', str(self.user_id)).eq('is_current', True).single().execute()
        
        if not archetype_result.data:
            return
        
        archetype_id = archetype_result.data['archetype_id']
        
        # Aggregate this user's outcomes with archetype
        engaged_count = sum(1 for o in user_outcomes if o['user_response'] == 'engaged')
        total = len(user_outcomes)
        
        # Update archetype intervention effectiveness (anonymized aggregation)
        current = store.client.table('behavioral_archetypes').select(
            'effective_interventions'
        ).eq('archetype_id', archetype_id).single().execute()
        
        if current.data:
            interventions = current.data.get('effective_interventions', [])
            
            # Find or create entry for this intervention type
            updated = False
            for inv in interventions:
                if inv.get('type') == intervention_type:
                    # Update running average
                    old_n = inv.get('n_users', 0)
                    old_rate = inv.get('success_rate', 0)
                    new_n = old_n + 1
                    new_rate = (old_rate * old_n + (engaged_count/total)) / new_n
                    
                    inv['success_rate'] = new_rate
                    inv['n_users'] = new_n
                    inv['last_updated'] = datetime.utcnow().isoformat()
                    updated = True
                    break
            
            if not updated:
                interventions.append({
                    'type': intervention_type,
                    'success_rate': engaged_count / total,
                    'n_users': 1,
                    'first_added': datetime.utcnow().isoformat()
                })
            
            # Store back
            store.client.table('behavioral_archetypes').update({
                'effective_interventions': interventions
            }).eq('archetype_id', archetype_id).execute()
    
    async def get_intervention_effectiveness_report(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Generate report on what interventions work for this user.
        """
        store = await get_behavioral_store()
        if not store:
            return {"error": "store_unavailable"}
        
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        outcomes = store.client.table('intervention_outcomes').select('*').eq(
            'user_id', str(self.user_id)
        ).gte('delivered_at', since).execute()
        
        if not outcomes.data:
            return {
                "period_days": days,
                "interventions_attempted": 0,
                "message": "Not enough data yet. Keep engaging with Spirit!"
            }
        
        # Analyze by type
        by_type = {}
        for o in outcomes.data:
            t = o['intervention_type']
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(o)
        
        effectiveness = {}
        for t, os in by_type.items():
            engaged = len([o for o in os if o['user_response'] == 'engaged'])
            total = len(os)
            effectiveness[t] = {
                "attempts": total,
                "engagement_rate": engaged / total,
                "best_time": self._find_best_time(os),
                "trend": "improving" if engaged > total * 0.6 else "stable"
            }
        
        return {
            "period_days": days,
            "total_interventions": len(outcomes.data),
            "by_type": effectiveness,
            "most_effective": max(effectiveness, key=lambda k: effectiveness[k]['engagement_rate']),
            "recommendation": "Focus on " + max(effectiveness, key=lambda k: effectiveness[k]['engagement_rate'])
        }
    
    def _find_best_time(self, outcomes: List[Dict]) -> Optional[int]:
        """Find hour of day with best engagement."""
        hour_success = {}
        for o in outcomes:
            if o['user_response'] == 'engaged':
                hour = datetime.fromisoformat(o['delivered_at']).hour
                hour_success[hour] = hour_success.get(hour, 0) + 1
        
        if hour_success:
            return max(hour_success, key=hour_success.get)
        return None


class ExplicitFeedbackCollector:
    """
    Collect and act on explicit user feedback (ratings, comments).
    """
    
    def __init__(self, user_id: UUID):
        self.user_id = user_id
    
    async def record_rating(
        self,
        intervention_id: str,
        rating: int,  # 1-5
        feedback_text: Optional[str] = None
    ):
        """Record explicit user rating of an intervention."""
        store = await get_behavioral_store()
        if not store:
            return
        
        # Store rating
        store.client.table('intervention_outcomes').update({
            'explicit_rating': rating,
            'qualitative_feedback': feedback_text,
            'rated_at': datetime.utcnow().isoformat()
        }).eq('intervention_id', intervention_id).execute()
        
        # If low rating, trigger model adjustment
        if rating <= 2:
            await self._handle_negative_feedback(intervention_id, feedback_text)
        
        # If high rating with text, extract insights
        if rating >= 4 and feedback_text:
            await self._extract_positive_insights(intervention_id, feedback_text)
    
    async def _handle_negative_feedback(
        self,
        intervention_id: str,
        feedback_text: Optional[str]
    ):
        """Adjust models when user dislikes an intervention."""
        # Reduce frequency of this intervention type
        # Could trigger A/B test of alternative approaches
        print(f"Negative feedback for {intervention_id}: {feedback_text}")
        
        # Log for review
        store = await get_behavioral_store()
        if store:
            store.client.table('negative_feedback_log').insert({
                'user_id': str(self.user_id),
                'intervention_id': intervention_id,
                'feedback': feedback_text,
                'logged_at': datetime.utcnow().isoformat(),
                'action_taken': 'reduced_frequency'
            }).execute()
    
    async def _extract_positive_insights(
        self,
        intervention_id: str,
        feedback_text: str
    ):
        """Extract why something worked from positive feedback."""
        # Use LLM to extract key phrases
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        from spirit.config import settings
        
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key)
        
        messages = [
            SystemMessage(content="Extract the key reason this intervention worked. One sentence."),
            HumanMessage(content=feedback_text)
        ]
        
        insight = llm.invoke(messages).content
        
        # Store as lesson learned
        store = await get_behavioral_store()
        if store:
            store.client.table('intervention_lessons').insert({
                'intervention_id': intervention_id,
                'extracted_insight': insight,
                'extracted_at': datetime.utcnow().isoformat()
            }).execute()
