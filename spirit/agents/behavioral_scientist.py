"""
Spirit's core intelligence: A LangGraph agent that acts as a behavioral scientist.
It reasons about data, forms hypotheses, predicts outcomes, and decides interventions.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from uuid import UUID, uuid4
import operator

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool

from spirit.config import settings
from spirit.db.supabase_client import get_behavioral_store
from spirit.services.causal_inference import CausalInferenceEngine
from spirit.services.goal_integration import BehavioralGoalBridge


class ScientistState(TypedDict):
    """
    The state of the behavioral scientist agent.
    Accumulates observations, hypotheses, and decisions.
    """
    user_id: str
    current_observation: Optional[Dict]  # Latest behavioral data
    recent_observations: Annotated[List[Dict], operator.add]  # Accumulated
    hypothesis: Optional[str]  # Current working hypothesis
    confidence: float  # 0-1 confidence in hypothesis
    recommended_action: Optional[str]  # Intervention decision
    reasoning: List[str]  # Chain of thought
    next_step: str  # Graph routing


class BehavioralScientistAgent:
    """
    LangGraph agent that continuously analyzes behavioral data
    and makes scientific decisions about interventions.
    """
    
    def __init__(self, user_id: UUID):
        self.user_id = user_id
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=settings.openai_api_key,
            temperature=0.2  # Scientific precision
        )
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph reasoning pipeline."""
        
        # Define nodes
        def observe(state: ScientistState) -> ScientistState:
            """Ingest and contextualize new observation."""
            obs = state["current_observation"]
            
            # Enrich with historical context
            state["reasoning"].append(
                f"Observed at {obs['timestamp']}: {obs.get('behavior', {})}"
            )
            
            # Check if this is anomalous
            if self._is_anomalous(obs):
                state["reasoning"].append("Anomaly detected - deviates from user baseline")
                state["next_step"] = "hypothesize"
            else:
                state["next_step"] = "accumulate"
            
            return state
        
        def hypothesize(state: ScientistState) -> ScientistState:
            """Form causal hypothesis about behavior."""
            # Query recent pattern
            recent = state["recent_observations"][-10:]
            
            # Use LLM to generate hypothesis
            messages = [
                SystemMessage(content="""
                You are a behavioral scientist analyzing digital phenotyping data.
                Form concise causal hypotheses. Use format: "X leads to Y because Z".
                Be specific about mechanisms. Confidence must be justified.
                """),
                HumanMessage(content=f"""
                Recent observations: {recent}
                Current context: {state['current_observation']}
                
                Form a hypothesis about what is causing this behavior pattern
                and what it might lead to. Be specific.
                """)
            ]
            
            response = self.llm.invoke(messages)
            
            state["hypothesis"] = response.content
            state["confidence"] = self._extract_confidence(response.content)
            state["reasoning"].append(f"Hypothesis formed: {response.content}")
            state["next_step"] = "predict"
            
            return state
        
        def predict(state: ScientistState) -> ScientistState:
            """Predict future trajectory if no intervention."""
            hypothesis = state["hypothesis"]
            
            messages = [
                SystemMessage(content="""
                Based on the hypothesis, predict what happens in 24 hours 
                if no intervention occurs. Be specific about outcomes.
                """),
                HumanMessage(content=f"""
                Hypothesis: {hypothesis}
                Current trajectory: {state['recent_observations'][-3:]}
                
                Predict specific outcomes 24 hours from now.
                """)
            ]
            
            response = self.llm.invoke(messages)
            state["reasoning"].append(f"Prediction: {response.content}")
            state["next_step"] = "decide"
            
            return state
        
        def decide(state: ScientistState) -> ScientistState:
            """Decide whether and how to intervene."""
            confidence = state["confidence"]
            
            if confidence < 0.6:
                state["recommended_action"] = "observe_more"
                state["reasoning"].append("Confidence too low - continue observation")
            else:
                # Determine intervention type
                action = self._select_intervention(state)
                state["recommended_action"] = action
                state["reasoning"].append(f"Intervention selected: {action}")
            
            state["next_step"] = "execute"
            return state
        
        def execute(state: ScientistState) -> ScientistState:
            """Execute or queue the intervention."""
            action = state["recommended_action"]
            
            if action == "observe_more":
                state["reasoning"].append("No action taken - insufficient certainty")
            else:
                # Log the decision for causal analysis later
                self._log_intervention_decision(state)
                state["reasoning"].append(f"Intervention {action} queued for delivery")
            
            state["next_step"] = END
            return state
        
        # Build graph
        workflow = StateGraph(ScientistState)
        
        workflow.add_node("observe", observe)
        workflow.add_node("hypothesize", hypothesize)
        workflow.add_node("predict", predict)
        workflow.add_node("decide", decide)
        workflow.add_node("execute", execute)
        
        # Add edges with routing
        workflow.add_edge("observe", "hypothesize")
        workflow.add_edge("hypothesize", "predict")
        workflow.add_edge("predict", "decide")
        workflow.add_edge("decide", "execute")
        
        workflow.set_entry_point("observe")
        
        return workflow.compile()
    
    async def process_observation(self, observation: Dict) -> Dict[str, Any]:
        """
        Process a new behavioral observation through the scientific pipeline.
        """
        initial_state = ScientistState(
            user_id=str(self.user_id),
            current_observation=observation,
            recent_observations=[],
            hypothesis=None,
            confidence=0.0,
            recommended_action=None,
            reasoning=[],
            next_step="observe"
        )
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        return {
            "hypothesis": result["hypothesis"],
            "confidence": result["confidence"],
            "recommended_action": result["recommended_action"],
            "reasoning": result["reasoning"],
            "intervention_queued": result["recommended_action"] not in [None, "observe_more"]
        }
    
    def _is_anomalous(self, observation: Dict) -> bool:
        """Detect if observation deviates from user's baseline."""
        # Simple statistical check - would use proper baseline in production
        behavior = observation.get("behavior", {})
        
        # High app switching is often anomalous
        if behavior.get("app_switches_5min", 0) > 10:
            return True
        
        # Very long session without break
        if behavior.get("session_duration_sec", 0) > 7200:  # 2 hours
            return True
        
        return False
    
    def _extract_confidence(self, hypothesis_text: str) -> float:
        """Extract confidence score from LLM response."""
        # Simple heuristic - would use structured output in production
        if "high confidence" in hypothesis_text.lower():
            return 0.8
        elif "moderate" in hypothesis_text.lower():
            return 0.6
        elif "uncertain" in hypothesis_text.lower():
            return 0.4
        return 0.5
    
    def _select_intervention(self, state: ScientistState) -> str:
        """Select appropriate intervention based on hypothesis."""
        hypothesis = state["hypothesis"].lower()
        
        if "distract" in hypothesis or "scatter" in hypothesis:
            return "focus_mode_prompt"
        elif "fatigue" in hypothesis or "tired" in hypothesis:
            return "rest_suggestion"
        elif "procrastination" in hypothesis:
            return "task_breakdown_ema"
        elif "social media" in hypothesis:
            return "app_block_suggestion"
        
        return "general_check_in"
    
    def _log_intervention_decision(self, state: ScientistState):
        """Log decision for later causal analysis."""
        # Store in Supabase for tracking outcomes
        # This enables "did the intervention work?" analysis
        pass


class PredictiveEngine:
    """
    Predicts future behavioral states and goal outcomes.
    Uses both statistical models and LLM reasoning.
    """
    
    def __init__(self, user_id: UUID):
        self.user_id = user_id
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=settings.openai_api_key,
            temperature=0.3
        )
    
    async def predict_goal_outcome(
        self,
        goal_id: UUID,
        horizon_days: int = 7
    ) -> Dict[str, Any]:
        """
        Predict probability of goal achievement based on current trajectory.
        """
        # Get behavioral history
        store = await get_behavioral_store()
        if not store:
            return {"error": "no_data"}
        
        week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
        observations = await store.get_user_observations(
            user_id=self.user_id,
            start_time=week_ago,
            limit=5000
        )
        
        # Get goal info
        bridge = BehavioralGoalBridge(self.user_id)
        progress = await bridge.compute_goal_progress(goal_id)
        
        # Statistical prediction
        trend = self._calculate_trend(observations)
        
        # LLM reasoning about trajectory
        messages = [
            SystemMessage(content="""
            You are a predictive model analyzing behavioral trajectories.
            Estimate probability of goal achievement and identify key risk factors.
            Be specific and quantitative where possible.
            """),
            HumanMessage(content=f"""
            Goal: {progress.get('goal_id')}
            Current progress score: {progress.get('progress_score')}
            7-day behavioral trend: {trend}
            Recent metrics: {progress.get('metrics')}
            
            Predict:
            1. Probability of goal achievement in {horizon_days} days (0-100%)
            2. Key risk factors that could derail progress
            3. Critical intervention points
            """)
        ]
        
        response = self.llm.invoke(messages)
        
        return {
            "goal_id": str(goal_id),
            "prediction_horizon_days": horizon_days,
            "trajectory_analysis": response.content,
            "statistical_trend": trend,
            "current_progress": progress.get("progress_score"),
            "confidence": "medium" if len(observations) > 50 else "low"
        }
    
    def _calculate_trend(self, observations: List[Dict]) -> str:
        """Calculate simple trend direction from observations."""
        if len(observations) < 10:
            return "insufficient_data"
        
        # Compare first half to second half
        mid = len(observations) // 2
        first_half = observations[:mid]
        second_half = observations[mid:]
        
        # Simple metric: productive time
        first_prod = sum(
            obs.get("behavior", {}).get("session_duration_sec", 0) 
            for obs in first_half
        ) / max(len(first_half), 1)
        
        second_prod = sum(
            obs.get("behavior", {}).get("session_duration_sec", 0)
            for obs in second_half
        ) / max(len(second_half), 1)
        
        if second_prod > first_prod * 1.2:
            return "improving"
        elif second_prod < first_prod * 0.8:
            return "declining"
        else:
            return "stable"


class InterventionOptimizer:
    """
    Optimizes intervention timing and content using multi-armed bandits
    and causal learning.
    """
    
    def __init__(self, user_id: UUID):
        self.user_id = user_id
        
    async def optimize_intervention(
        self,
        context: Dict[str, Any],
        available_interventions: List[str]
    ) -> Dict[str, Any]:
        """
        Select best intervention using Thompson sampling approach.
        Balances exploration (try new things) vs exploitation (use what works).
        """
        store = await get_behavioral_store()
        
        # Get historical effectiveness of each intervention
        effectiveness = {}
        for intervention in available_interventions:
            # Query past outcomes
            past = await self._get_intervention_history(intervention)
            
            if len(past) < 5:
                # Not enough data - high uncertainty, good for exploration
                effectiveness[intervention] = {
                    "success_rate": 0.5,
                    "uncertainty": 0.5,
                    "n_trials": len(past)
                }
            else:
                successes = sum(1 for p in past if p["outcome"] > 0.5)
                effectiveness[intervention] = {
                    "success_rate": successes / len(past),
                    "uncertainty": 1.0 / (len(past) ** 0.5),  # Decreases with more data
                    "n_trials": len(past)
                }
        
        # Thompson sampling: sample from Beta distribution for each
        import random
        best_intervention = None
        best_sample = 0
        
        for intervention, stats in effectiveness.items():
            # Beta(α=successes+1, β=failures+1) approximation
            alpha = stats["success_rate"] * stats["n_trials"] + 1
            beta_param = (1 - stats["success_rate"]) * stats["n_trials"] + 1
            
            # Sample from Beta
            sample = random.betavariate(alpha, beta_param)
            
            if sample > best_sample:
                best_sample = sample
                best_intervention = intervention
        
        return {
            "selected_intervention": best_intervention,
            "selection_method": "thompson_sampling",
            "estimated_success_probability": best_sample,
            "exploration_vs_exploitation": "exploration" if effectiveness[best_intervention]["n_trials"] < 10 else "exploitation",
            "alternatives_considered": list(effectiveness.keys())
        }
    
    async def _get_intervention_history(self, intervention_type: str) -> List[Dict]:
        """Get past outcomes for this intervention type."""
        # Query Supabase for interventions of this type and their outcomes
        return []  # Placeholder
