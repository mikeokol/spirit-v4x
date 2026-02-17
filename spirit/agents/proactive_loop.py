"""
Proactive Agent Loop: Spirit's autonomous operation system.
Predicts, schedules, and executes interventions without waiting for user input.
The goal: intervene before the user fails, not after.
v1.4: Integrated Multi-Agent Debate (MAO) and Ethical Guardrails
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import random

from spirit.db.supabase_client import get_behavioral_store
from spirit.services.notification_engine import NotificationEngine, NotificationPriority
from spirit.agents.behavioral_scientist import BehavioralScientistAgent, PredictiveEngine
from spirit.memory.episodic_memory import EpisodicMemorySystem
from spirit.services.causal_inference import CausalInferenceEngine
# NEW: Import MAO and Ethical Guardrails
from spirit.agents.multi_agent_debate import MultiAgentDebate
from spirit.services.ethical_guardrails import EthicalGuardrails


class PredictionHorizon(Enum):
    IMMINENT = "imminent"      # 0-30 minutes
    SHORT = "short"            # 30 min - 4 hours
    MEDIUM = "medium"          # 4-24 hours
    LONG = "long"              # 1-7 days


@dataclass
class PredictedState:
    """
    A forecast of user's future state with confidence and intervention opportunity.
    """
    horizon: PredictionHorizon
    predicted_time: datetime
    state_type: str  # 'vulnerability', 'opportunity', 'maintenance', 'risk'
    
    # What we predict
    predicted_behavior: Dict[str, Any]  # e.g., {"focus_score": 0.3, "app_category": "social_media"}
    confidence: float  # 0-1
    
    # Why we predict this
    trigger_features: Dict[str, float]  # What signals led to this prediction
    historical_pattern_match: Optional[str]  # Which past episode this resembles
    
    # What to do about it
    optimal_intervention: Optional[str]
    intervention_window: Optional[tuple]  # (start, end) when to deliver
    expected_outcome_if_intervene: float  # Predicted improvement
    expected_outcome_if_ignore: float     # Predicted decline


class ProactiveScheduler:
    """
    Schedules autonomous check-ins and interventions.
    Operates on predicted vulnerability windows, not fixed times.
    v1.4: All interventions routed through MAO debate and ethical checks.
    """
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.scheduled_checks: Dict[str, datetime] = {}  # check_id -> scheduled_time
        self.running = False
        self.check_interval_seconds = 60  # How often to run prediction loop
        
        # Callbacks for different prediction types
        self.intervention_handlers: Dict[str, Callable] = {}
        
        # NEW: Initialize MAO and Ethical Guardrails
        self.debate_system = MultiAgentDebate()
        self.ethical_guardrails = EthicalGuardrails()
    
    def register_handler(self, state_type: str, handler: Callable):
        """Register a function to handle predicted states."""
        self.intervention_handlers[state_type] = handler
    
    async def start(self):
        """Start the autonomous loop."""
        self.running = True
        print(f"Proactive loop started for user {self.user_id}")
        
        while self.running:
            try:
                await self._run_prediction_cycle()
            except Exception as e:
                print(f"Error in proactive loop: {e}")
            
            await asyncio.sleep(self.check_interval_seconds)
    
    def stop(self):
        """Stop the autonomous loop."""
        self.running = False
    
    async def _run_prediction_cycle(self):
        """
        One cycle: predict, schedule, execute if due.
        """
        now = datetime.utcnow()
        
        # 1. Generate predictions for all horizons
        predictions = await self._generate_predictions()
        
        # 2. For each prediction, schedule or execute
        for pred in predictions:
            check_id = f"{pred.state_type}_{pred.predicted_time.isoformat()}"
            
            # Skip if already handled
            if check_id in self.scheduled_checks:
                if self.scheduled_checks[check_id] < now:
                    # Time to execute
                    await self._execute_intervention(pred)
                    del self.scheduled_checks[check_id]
                continue
            
            # Schedule if in planning window
            if pred.predicted_time > now and pred.predicted_time < now + timedelta(hours=4):
                self.scheduled_checks[check_id] = pred.predicted_time
                print(f"Scheduled {pred.state_type} check for {pred.predicted_time}")
                
                # Set up async task for exact time
                delay = (pred.predicted_time - now).total_seconds()
                asyncio.create_task(self._delayed_execution(check_id, delay, pred))
    
    async def _generate_predictions(self) -> List[PredictedState]:
        """
        Generate multi-horizon predictions for this user.
        """
        predictions = []
        
        # Get current context
        context = await self._get_current_context()
        
        # IMMINENT: Next 30 minutes (based on current trajectory)
        imminent = await self._predict_imminent(context)
        if imminent:
            predictions.append(imminent)
        
        # SHORT: Next 4 hours (based on daily patterns)
        short = await self._predict_short_term(context)
        if short:
            predictions.extend(short)
        
        # MEDIUM: Next 24 hours (based on weekly patterns + calendar)
        medium = await self._predict_medium_term()
        if medium:
            predictions.extend(medium)
        
        # LONG: Next 7 days (based on trend analysis)
        long_term = await self._predict_long_term()
        if long_term:
            predictions.extend(long_term)
        
        return predictions
    
    async def _predict_imminent(self, context: Dict) -> Optional[PredictedState]:
        """
        Predict immediate next state (0-30 min) based on current momentum.
        """
        # Check current trajectory
        recent = context.get("recent_observations", [])
        if len(recent) < 3:
            return None
        
        # Simple momentum-based prediction
        last_3 = recent[-3:]
        focus_trend = sum(
            o.get("behavior", {}).get("focus_score", 0.5) 
            for o in last_3
        ) / 3
        
        # If declining focus, predict vulnerability
        if focus_trend < 0.4 and recent[-1].get("behavior", {}).get("app_category") != "productivity":
            return PredictedState(
                horizon=PredictionHorizon.IMMINENT,
                predicted_time=datetime.utcnow() + timedelta(minutes=15),
                state_type="vulnerability",
                predicted_behavior={"focus_score": 0.2, "likely_distraction": True},
                confidence=0.7,
                trigger_features={"focus_decline_rate": 0.3, "context_switches": 5},
                historical_pattern_match=None,
                optimal_intervention="micro_focus_prompt",
                intervention_window=(
                    datetime.utcnow() + timedelta(minutes=10),
                    datetime.utcnow() + timedelta(minutes=20)
                ),
                expected_outcome_if_intervene=0.6,
                expected_outcome_if_ignore=0.2
            )
        
        # If sustained focus, predict opportunity for deep work extension
        if focus_trend > 0.7:
            return PredictedState(
                horizon=PredictionHorizon.IMMINENT,
                predicted_time=datetime.utcnow() + timedelta(minutes=20),
                state_type="opportunity",
                predicted_behavior={"focus_score": 0.8, "deep_work_continuation": True},
                confidence=0.6,
                trigger_features={"sustained_attention": 25, "no_interruptions": 5},
                historical_pattern_match=None,
                optimal_intervention="offer_focus_extension",
                intervention_window=(
                    datetime.utcnow() + timedelta(minutes=15),
                    datetime.utcnow() + timedelta(minutes=30)
                ),
                expected_outcome_if_intervene=0.9,
                expected_outcome_if_ignore=0.5
            )
        
        return None
    
    async def _predict_short_term(self, context: Dict) -> List[PredictedState]:
        """
        Predict next 4 hours based on daily patterns.
        """
        predictions = []
        now = datetime.utcnow()
        hour = now.hour
        
        # Get historical pattern for this time of day
        store = await get_behavioral_store()
        if not store:
            return predictions
        
        # Query similar historical periods
        similar_time = store.client.table('behavioral_observations').select('*').eq(
            'user_id', str(self.user_id)
        ).execute()  # Would filter by hour, day of week
        
        # Predict lunch slump (12-14h)
        if 11 <= hour <= 13:
            predictions.append(PredictedState(
                horizon=PredictionHorizon.SHORT,
                predicted_time=now + timedelta(hours=1),
                state_type="risk",
                predicted_behavior={"energy_low": True, "productivity_drop": True},
                confidence=0.6,
                trigger_features={"time_of_day": hour, "pre_lunch": True},
                historical_pattern_match="post_lunch_slump",
                optimal_intervention="suggest_walk_or_nap",
                intervention_window=(now + timedelta(minutes=30), now + timedelta(hours=2)),
                expected_outcome_if_intervene=0.7,
                expected_outcome_if_ignore=0.3
            ))
        
        # Predict evening wind-down need (20-22h)
        if 19 <= hour <= 21:
            predictions.append(PredictedState(
                horizon=PredictionHorizon.SHORT,
                predicted_time=now + timedelta(hours=2),
                state_type="opportunity",
                predicted_behavior={"sleep_prep": True, "screen_time_reduction": True},
                confidence=0.65,
                trigger_features={"evening_time": True, "high_screen_time_today": True},
                historical_pattern_match="evening_ritual",
                optimal_intervention="wind_down_prompt",
                intervention_window=(now + timedelta(hours=1), now + timedelta(hours=3)),
                expected_outcome_if_intervene=0.8,
                expected_outcome_if_ignore=0.4
            ))
        
        return predictions
    
    async def _predict_medium_term(self) -> List[PredictedState]:
        """
        Predict next 24 hours based on weekly patterns and upcoming events.
        """
        predictions = []
        now = datetime.utcnow()
        
        # Check tomorrow morning based on tonight's behavior
        # If high evening usage predicted, predict poor sleep and slow morning
        
        # Would integrate with calendar API here
        
        return predictions
    
    async def _predict_long_term(self) -> List[PredictedState]:
        """
        Predict next 7 days based on trend analysis.
        """
        predictions = []
        
        # Get trend from predictive engine
        engine = PredictiveEngine(self.user_id)
        # Would call trend analysis
        
        return predictions
    
    async def _get_current_context(self) -> Dict:
        """Get user's current behavioral context."""
        store = await get_behavioral_store()
        if not store:
            return {}
        
        # Last hour of observations
        one_hour_ago = (datetime.utcnow() - timedelta(hours=1)).isoformat()
        recent = await store.get_user_observations(
            user_id=self.user_id,
            start_time=one_hour_ago,
            limit=100
        )
        
        return {
            "recent_observations": [o.dict() for o in recent],
            "current_hour": datetime.utcnow().hour,
            "day_of_week": datetime.utcnow().weekday()
        }
    
    async def _execute_intervention(self, prediction: PredictedState):
        """
        Execute the optimal intervention for a predicted state.
        v1.4: Now routes through Ethical Guardrails → MAO Debate → Delivery
        """
        # NEW STEP 1: Ethical Guardrails Check
        ethical_check = await self.ethical_guardrails.approve_intervention(
            user_id=self.user_id,
            intervention_type=prediction.state_type,
            intensity=prediction.confidence
        )
        
        if not ethical_check['approved']:
            print(f"Intervention blocked by ethical guardrails: {ethical_check['reason']}")
            await self._log_blocked_intervention(prediction, ethical_check['reason'])
            return
        
        # NEW STEP 2: Multi-Agent Debate
        # Build user context for debate
        user_context = await self._build_debate_context(prediction)
        
        debate_result = await self.debate_system.debate_intervention(
            user_context=user_context,
            proposed_intervention=prediction.optimal_intervention or "default_intervention",
            predicted_outcome={
                'expected_improvement': prediction.expected_outcome_if_intervene,
                'confidence': prediction.confidence
            }
        )
        
        if not debate_result['proceed']:
            print(f"Intervention blocked by adversary: {debate_result.get('adversary_concerns', 'unknown objection')}")
            await self._log_adversary_objection(prediction, debate_result)
            return
        
        # STEP 3: Execute debated intervention
        handler = self.intervention_handlers.get(prediction.state_type)
        
        if handler:
            # Pass debate result to handler so it can use the refined message
            await handler(prediction, self.user_id, debate_result)
        else:
            # Default: send notification with debated message
            await self._default_intervention(prediction, debate_result)
    
    async def _build_debate_context(self, prediction: PredictedState) -> Dict:
        """Build context for MAO debate."""
        # Get rejection rate for this user
        store = await get_behavioral_store()
        rejection_rate = 0.0
        interventions_today = 0
        
        if store:
            # Query recent interventions
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0).isoformat()
            recent = store.client.table('proactive_interventions').select('*').eq(
                'user_id', str(self.user_id)
            ).gte('executed_at', today_start).execute()
            
            if recent.data:
                interventions_today = len(recent.data)
                # Calculate rejection rate from last 20 interventions
                last_20 = store.client.table('proactive_interventions').select('*').eq(
                    'user_id', str(self.user_id)
                ).order('executed_at', desc=True).limit(20).execute()
                
                if last_20.data:
                    rejected = sum(1 for i in last_20.data if i.get('user_response') == 'dismissed')
                    rejection_rate = rejected / len(last_20.data)
        
        return {
            'current_state': prediction.state_type,
            'recent_pattern': prediction.historical_pattern_match,
            'goal_progress': 0.5,  # Would fetch from goals
            'rejection_rate': rejection_rate,
            'interventions_today': interventions_today,
            'predicted_vulnerability': prediction.state_type == 'vulnerability',
            'confidence': prediction.confidence
        }
    
    async def _default_intervention(self, prediction: PredictedState, debate_result: Optional[Dict] = None):
        """Default intervention: smart notification with debated message."""
        engine = NotificationEngine(self.user_id)
        
        # Use debated message if available, otherwise generate default
        if debate_result and debate_result.get('message'):
            title = "Spirit"
            body = debate_result['message']
        else:
            title = self._get_intervention_title(prediction)
            body = self._get_intervention_body(prediction)
        
        content = {
            "title": title,
            "body": body,
            "data": {
                "prediction_type": prediction.state_type,
                "confidence": prediction.confidence,
                "expected_outcome": prediction.expected_outcome_if_intervene,
                "debate_validated": debate_result is not None,
                "debate_rounds": debate_result.get('debate_rounds', 0) if debate_result else 0
            }
        }
        
        # Determine priority
        priority = NotificationPriority.HIGH if prediction.confidence > 0.7 else NotificationPriority.NORMAL
        
        await engine.send_notification(
            content=content,
            priority=priority,
            notification_type=f"proactive_{prediction.state_type}",
            context={"predicted_vulnerability": prediction.state_type == "vulnerability"}
        )
        
        # Log the proactive intervention
        await self._log_proactive_intervention(prediction, debate_result)
    
    def _get_intervention_title(self, prediction: PredictedState) -> str:
        """Generate contextual title based on prediction."""
        titles = {
            "vulnerability": "Focus fading? Try this",
            "opportunity": "You're in flow—extend it?",
            "risk": "Heads up: energy dip predicted",
            "maintenance": "Quick check-in"
        }
        return titles.get(prediction.state_type, "Spirit here")
    
    def _get_intervention_body(self, prediction: PredictedState) -> str:
        """Generate contextual body text."""
        if prediction.state_type == "vulnerability":
            return "Your focus score is dropping. 2-minute reset available."
        elif prediction.state_type == "opportunity":
            return "You've been deep in work for 25 min. Want to lock in for another 25?"
        elif prediction.state_type == "risk":
            return "You usually struggle after lunch. Pre-positioned support ready."
        return "How are you doing?"
    
    async def _delayed_execution(self, check_id: str, delay_seconds: float, prediction: PredictedState):
        """Execute after delay, if not cancelled."""
        await asyncio.sleep(delay_seconds)
        
        # Check if still scheduled (might have been cancelled)
        if check_id in self.scheduled_checks:
            await self._execute_intervention(prediction)
            del self.scheduled_checks[check_id]
    
    async def _log_proactive_intervention(self, prediction: PredictedState, debate_result: Optional[Dict] = None):
        """Log for learning and causal analysis."""
        store = await get_behavioral_store()
        if store:
            store.client.table('proactive_interventions').insert({
                'intervention_id': str(uuid4()),
                'user_id': str(self.user_id),
                'predicted_state': prediction.state_type,
                'predicted_time': prediction.predicted_time.isoformat(),
                'confidence': prediction.confidence,
                'intervention_type': prediction.optimal_intervention,
                'executed_at': datetime.utcnow().isoformat(),
                'expected_outcome': prediction.expected_outcome_if_intervene,
                'debate_validated': debate_result is not None,
                'debate_rounds': debate_result.get('debate_rounds', 0) if debate_result else 0,
                'consensus_reached': debate_result.get('consensus_reached', False) if debate_result else False
            }).execute()
    
    async def _log_blocked_intervention(self, prediction: PredictedState, reason: str):
        """Log when ethical guardrails block an intervention."""
        store = await get_behavioral_store()
        if store:
            store.client.table('blocked_interventions').insert({
                'user_id': str(self.user_id),
                'predicted_state': prediction.state_type,
                'blocked_reason': reason,
                'blocked_at': datetime.utcnow().isoformat(),
                'confidence': prediction.confidence
            }).execute()
    
    async def _log_adversary_objection(self, prediction: PredictedState, debate_result: Dict):
        """Log adversary objections for learning."""
        store = await get_behavioral_store()
        if store:
            store.client.table('adversary_objections').insert({
                'user_id': str(self.user_id),
                'predicted_state': prediction.state_type,
                'objection': debate_result.get('adversary_concerns', 'unknown'),
                'debate_rounds': debate_result.get('debate_rounds', 0),
                'logged_at': datetime.utcnow().isoformat()
            }).execute()


class AutonomousExperimentRunner:
    """
    Automatically runs micro-experiments to improve predictions.
    Part of the proactive loop: always learning, always testing.
    """
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.experiment_queue: List[Dict] = []
        self.min_hours_between_experiments = 4
    
    async def design_and_queue_experiment(self, context: Dict):
        """
        Design a micro-experiment based on current uncertainty.
        """
        # Check if we should run experiment now
        if not await self._can_run_experiment():
            return
        
        # Find what we're uncertain about
        uncertain_prediction = await self._find_uncertainty(context)
        
        if uncertain_prediction:
            experiment = self._create_experiment(uncertain_prediction)
            self.experiment_queue.append(experiment)
            
            # Execute immediately if high value
            if experiment.get("priority") == "high":
                await self._run_experiment(experiment)
    
    async def _can_run_experiment(self) -> bool:
        """Check if enough time has passed since last experiment."""
        store = await get_behavioral_store()
        if not store:
            return False
        
        last = store.client.table('proactive_experiments').select('*').eq(
            'user_id', str(self.user_id)
        ).order('started_at', desc=True).limit(1).execute()
        
        if not last.data:
            return True
        
        last_time = datetime.fromisoformat(last.data[0]['started_at'])
        hours_since = (datetime.utcnow() - last_time).total_seconds() / 3600
        
        return hours_since > self.min_hours_between_experiments
    
    async def _find_uncertainty(self, context: Dict) -> Optional[Dict]:
        """Find what we're most uncertain about predicting."""
        # Check historical prediction accuracy
        store = await get_behavioral_store()
        if not store:
            return None
        
        # Query past predictions and outcomes
        past = store.client.table('proactive_interventions').select('*').eq(
            'user_id', str(self.user_id)
        ).execute()
        
        # Find state type with lowest accuracy
        accuracy_by_type = {}
        for p in past.data if past.data else []:
            state = p['predicted_state']
            if state not in accuracy_by_type:
                accuracy_by_type[state] = {'correct': 0, 'total': 0}
            accuracy_by_type[state]['total'] += 1
            # Would check if prediction was correct
        
        # Find most uncertain
        most_uncertain = None
        lowest_acc = 1.0
        for state, stats in accuracy_by_type.items():
            if stats['total'] >= 3:
                acc = stats.get('correct', 0) / stats['total']
                if acc < lowest_acc:
                    lowest_acc = acc
                    most_uncertain = state
        
        if most_uncertain and lowest_acc < 0.6:
            return {"state_type": most_uncertain, "current_accuracy": lowest_acc}
        
        return None
    
    def _create_experiment(self, uncertainty: Dict) -> Dict:
        """Create experiment to reduce uncertainty."""
        return {
            "experiment_id": str(uuid4()),
            "hypothesis": f"Alternative intervention improves prediction accuracy for {uncertainty['state_type']}",
            "state_type": uncertainty['state_type'],
            "arms": ["control", "treatment_A", "treatment_B"],
            "sample_size": 21,  # 7 per arm for 1 week
            "priority": "medium",
            "design": "micro_randomized"
        }
    
    async def _run_experiment(self, experiment: Dict):
        """Execute the experiment."""
        # Randomize user into arm
        arm = random.choice(experiment['arms'])
        
        store = await get_behavioral_store()
        if store:
            store.client.table('proactive_experiments').insert({
                'experiment_id': experiment['experiment_id'],
                'user_id': str(self.user_id),
                'hypothesis': experiment['hypothesis'],
                'arm': arm,
                'started_at': datetime.utcnow().isoformat(),
                'status': 'running'
            }).execute()
        
        print(f"Started experiment {experiment['experiment_id']} for user {self.user_id} in arm {arm}")


class GlobalProactiveOrchestrator:
    """
    Manages proactive loops for all active users.
    Singleton that runs in the background.
    """
    
    def __init__(self):
        self.user_loops: Dict[int, ProactiveScheduler] = {}
        self.experiment_runners: Dict[int, AutonomousExperimentRunner] = {}
        self.running = False
    
    async def start_user_loop(self, user_id: int):
        """Start proactive loop for a user."""
        if user_id in self.user_loops:
            return  # Already running
        
        scheduler = ProactiveScheduler(user_id)
        
        # Register intervention handlers
        scheduler.register_handler("vulnerability", self._handle_vulnerability)
        scheduler.register_handler("opportunity", self._handle_opportunity)
        scheduler.register_handler("risk", self._handle_risk)
        
        self.user_loops[user_id] = scheduler
        
        # Start in background
        asyncio.create_task(scheduler.start())
        
        # Also start experiment runner
        runner = AutonomousExperimentRunner(user_id)
        self.experiment_runners[user_id] = runner
        
        print(f"Started proactive loop for user {user_id}")
    
    def stop_user_loop(self, user_id: int):
        """Stop loop for a user."""
        if user_id in self.user_loops:
            self.user_loops[user_id].stop()
            del self.user_loops[user_id]
        
        if user_id in self.experiment_runners:
            del self.experiment_runners[user_id]
    
    async def _handle_vulnerability(self, prediction: PredictedState, user_id: int, debate_result: Optional[Dict] = None):
        """Handle predicted vulnerability with debated message."""
        engine = NotificationEngine(user_id)
        
        # Use debated message if available
        body = debate_result['message'] if debate_result else "Your attention seems scattered. 30-second reset?"
        
        await engine.send_notification(
            content={
                "title": "Focus check",
                "body": body,
                "action": "focus_reset",
                "data": {
                    "debate_validated": debate_result is not None,
                    "debate_rounds": debate_result.get('debate_rounds', 0) if debate_result else 0
                }
            },
            priority=NotificationPriority.HIGH,
            notification_type="proactive_vulnerability"
        )
    
    async def _handle_opportunity(self, prediction: PredictedState, user_id: int, debate_result: Optional[Dict] = None):
        """Handle predicted opportunity with debated message."""
        engine = NotificationEngine(user_id)
        
        # Use debated message if available, otherwise calculate gain
        if debate_result and debate_result.get('message'):
            body = debate_result['message']
        else:
            gain = int(prediction.expected_outcome_if_intervene * 100)
            body = f"You're in flow. Extend this session? Predicted productivity gain: +{gain}%"
        
        await engine.send_notification(
            content={
                "title": "You're in flow",
                "body": body,
                "action": "extend_focus",
                "data": {
                    "debate_validated": debate_result is not None,
                    "debate_rounds": debate_result.get('debate_rounds', 0) if debate_result else 0
                }
            },
            priority=NotificationPriority.NORMAL,
            notification_type="proactive_opportunity"
        )
    
    async def _handle_risk(self, prediction: PredictedState, user_id: int, debate_result: Optional[Dict] = None):
        """Handle predicted risk with debated message."""
        engine = NotificationEngine(user_id)
        
        # Use debated message if available
        body = debate_result['message'] if debate_result else "Your energy typically drops now. Pre-positioned: 5-min walk suggestion ready."
        
        await engine.send_notification(
            content={
                "title": "Heads up",
                "body": body,
                "action": "preventive_break",
                "data": {
                    "debate_validated": debate_result is not None,
                    "debate_rounds": debate_result.get('debate_rounds', 0) if debate_result else 0
                }
            },
            priority=NotificationPriority.NORMAL,
            notification_type="proactive_risk"
        )


# Global singleton
_orchestrator: Optional[GlobalProactiveOrchestrator] = None


def get_orchestrator() -> GlobalProactiveOrchestrator:
    """Get or create global orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = GlobalProactiveOrchestrator()
    return _orchestrator
