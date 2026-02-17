"""
Spirit's core intelligence: A LangGraph agent that acts as a behavioral scientist.
It reasons about data, forms hypotheses, predicts outcomes, and decides interventions.
v1.4: Integrated with Multi-Agent Debate (MAO) - now recommends, doesn't decide.
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
# NEW: Import Belief Network for cognitive dissonance detection
from spirit.agents.belief_network import BeliefNetwork, CognitiveDissonanceDetector


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
    recommended_action: Optional[str]  # Intervention recommendation (not final decision)
    reasoning: List[str]  # Chain of thought
    belief_alignment: Optional[Dict]  # NEW: How this aligns with user's beliefs
    dissonance_detected: bool  # NEW: Flag if user beliefs contradict data
    next_step: str  # Graph routing


class BehavioralScientistAgent:
    """
    LangGraph agent that continuously analyzes behavioral data
    and makes scientific recommendations about interventions.
    v1.4: Now integrates with Belief Network to detect cognitive dissonance.
    Recommendations go to MAO for debate before execution.
    """
    
    def __init__(self, user_id: UUID):
        self.user_id = user_id
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=settings.openai_api_key,
            temperature=0.2  # Scientific precision
        )
        self.graph = self._build_graph()
        # NEW: Initialize Belief Network
        self.belief_network = BeliefNetwork(user_id)
        self.dissonance_detector = CognitiveDissonanceDetector(user_id)
        
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
            
            # NEW: Check for cognitive dissonance before forming hypothesis
            dissonance = self.dissonance_detector.check_dissonance(
                observation=state["current_observation"],
                recent_observations=recent
            )
            
            if dissonance['detected']:
                state["dissonance_detected"] = True
                state["belief_alignment"] = {
                    "user_belief": dissonance['user_belief'],
                    "data_reality": dissonance['data_reality'],
                    "gap": dissonance['gap']
                }
                state["reasoning"].append(
                    f"COGNITIVE DISSONANCE: User believes '{dissonance['user_belief']}' "
                    f"but data shows '{dissonance['data_reality']}'"
                )
                # Form hypothesis about the belief gap itself
                hypothesis_prompt = f"""
                User believes: {dissonance['user_belief']}
                Reality shows: {dissonance['data_reality']}
                Gap magnitude: {dissonance['gap']}
                
                Form hypothesis about why this belief-reality gap exists
                and how to address it without causing rebound.
                """
            else:
                state["dissonance_detected"] = False
                hypothesis_prompt = f"""
                Recent observations: {recent}
                Current context: {state['current_observation']}
                
                Form a hypothesis about what is causing this behavior pattern
                and what it might lead to. Be specific.
                """
            
            # Use LLM to generate hypothesis
            messages = [
                SystemMessage(content="""
                You are a behavioral scientist analyzing digital phenotyping data.
                Form concise causal hypotheses. Use format: "X leads to Y because Z".
                Be specific about mechanisms. Confidence must be justified.
                
                If cognitive dissonance is detected, hypothesize about the belief
                formation mechanism and how to gently correct it.
                """),
                HumanMessage(content=hypothesis_prompt)
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
            
            # NEW: If dissonance detected, predict belief change vs behavior change
            if state.get("dissonance_detected"):
                messages = [
                    SystemMessage(content="""
                    Based on the belief-reality gap, predict two scenarios:
                    1. If we challenge the belief directly (risk of rebound/annoyance)
                    2. If we work within the belief frame (slower but safer)
                    
                    Estimate probability of success for each approach.
                    """),
                    HumanMessage(content=f"""
                    Belief: {state['belief_alignment']['user_belief']}
                    Reality: {state['belief_alignment']['data_reality']}
                    Hypothesis: {hypothesis}
                    
                    Predict outcomes for both intervention strategies.
                    """)
                ]
            else:
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
            
            # NEW: Lower threshold if dissonance detected (more urgent)
            threshold = 0.5 if state.get("dissonance_detected") else 0.6
            
            if confidence < threshold:
                state["recommended_action"] = "observe_more"
                state["reasoning"].append(f"Confidence {confidence:.2f} below threshold {threshold} - continue observation")
            else:
                # Determine intervention type
                action = self._select_intervention(state)
                state["recommended_action"] = action
                state["reasoning"].append(f"Intervention recommended: {action}")
                
                # NEW: If dissonance detected, tag for belief-challenge intervention
                if state.get("dissonance_detected"):
                    state["reasoning"].append("Tagged as BELIEF_CHALLENGE - requires careful framing")
                    state["recommended_action"] = f"belief_challenge:{action}"
            
            state["next_step"] = "execute"
            return state
        
        def execute(state: ScientistState) -> ScientistState:
            """Package recommendation for MAO debate (not direct execution)."""
            action = state["recommended_action"]
            
            if action == "observe_more":
                state["reasoning"].append("No action recommended - insufficient certainty")
            else:
                # NEW: Update belief network with this recommendation
                if state.get("dissonance_detected"):
                    self.belief_network.tag_hypothesis(
                        belief=state["belief_alignment"]["user_belief"],
                        hypothesis=state["hypothesis"],
                        intervention_planned=action
                    )
                
                # Log the recommendation for the proactive loop to pick up
                # The proactive loop will run this through MAO debate
                self._queue_for_debate(state)
                state["reasoning"].append(f"Recommendation {action} queued for MAO debate")
            
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
            belief_alignment=None,
            dissonance_detected=False,
            next_step="observe"
        )
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        return {
            "hypothesis": result["hypothesis"],
            "confidence": result["confidence"],
            "recommended_action": result["recommended_action"],
            "reasoning": result["reasoning"],
            "dissonance_detected": result["dissonance_detected"],
            "belief_alignment": result["belief_alignment"],
            "intervention_queued": result["recommended_action"] not in [None, "observe_more"],
            "requires_mao_debate": result["recommended_action"] not in [None, "observe_more"]
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
        
        # NEW: If dissonance detected, use gentler interventions
        if state.get("dissonance_detected"):
            if "distract" in hypothesis or "scatter" in hypothesis:
                return "gentle_focus_awareness"  # Don't challenge directly
            elif "fatigue" in hypothesis:
                return "self_compassion_prompt"  # Acknowledge their belief
        
        if "distract" in hypothesis or "scatter" in hypothesis:
            return "focus_mode_prompt"
        elif "fatigue" in hypothesis or "tired" in hypothesis:
            return "rest_suggestion"
        elif "procrastination" in hypothesis:
            return "task_breakdown_ema"
        elif "social media" in hypothesis:
            return "app_block_suggestion"
        
        return "general_check_in"
    
    def _queue_for_debate(self, state: ScientistState):
        """Queue recommendation for MAO debate in proactive loop."""
        # Store in Supabase for proactive loop to pick up
        # This decouples the scientist from the delivery system
        store = get_behavioral_store()
        if store:
            store.client.table('intervention_recommendations').insert({
                'recommendation_id': str(uuid4()),
                'user_id': str(self.user_id),
                'hypothesis': state["hypothesis"],
                'confidence': state["confidence"],
                'recommended_action': state["recommended_action"],
                'dissonance_detected': state.get("dissonance_detected", False),
                'belief_alignment': state.get("belief_alignment"),
                'reasoning': state["reasoning"],
                'created_at': datetime.utcnow().isoformat(),
                'status': 'pending_debate'
            }).execute()


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
        # NEW: Initialize belief network for belief-aware predictions
        self.belief_network = BeliefNetwork(user_id)
    
    async def predict_goal_outcome(
        self,
        goal_id: UUID,
        horizon_days: int = 7
    ) -> Dict[str, Any]:
        """
        Predict probability of goal achievement based on current trajectory.
        v1.4: Now factors in user's beliefs about their capabilities.
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
        
        # NEW: Get user's beliefs about this goal
        belief_model = await self.belief_network.get_beliefs_about_goal(goal_id)
        
        # Statistical prediction
        trend = self._calculate_trend(observations)
        
        # LLM reasoning about trajectory - now includes beliefs
        belief_context = ""
        if belief_model:
            belief_context = f"""
            User's stated beliefs about this goal:
            - Believes they work best at: {belief_model.get('optimal_time', 'unknown')}
            - Self-efficacy score: {belief_model.get('self_efficacy', 0.5)}
            - Identified barriers: {belief_model.get('barriers', [])}
            
            Note: Adjust prediction if data contradicts beliefs significantly.
            """
        
        messages = [
            SystemMessage(content="""
            You are a predictive model analyzing behavioral trajectories.
            Estimate probability of goal achievement and identify key risk factors.
            Be specific and quantitative where possible.
            
            IMPORTANT: Consider the user's beliefs about themselves. If their beliefs
            are misaligned with reality, this creates either:
            1. Overconfidence (believes they can do more than data shows) -> higher risk
            2. Underconfidence (believes they can do less than data shows) -> hidden potential
            """),
            HumanMessage(content=f"""
            Goal: {progress.get('goal_id')}
            Current progress score: {progress.get('progress_score')}
            7-day behavioral trend: {trend}
            Recent metrics: {progress.get('metrics')}
            
            {belief_context}
            
            Predict:
            1. Probability of goal achievement in {horizon_days} days (0-100%)
            2. Key risk factors that could derail progress
            3. Critical intervention points
            4. Belief-reality gaps that need addressing
            """)
        ]
        
        response = self.llm.invoke(messages)
        
        return {
            "goal_id": str(goal_id),
            "prediction_horizon_days": horizon_days,
            "trajectory_analysis": response.content,
            "statistical_trend": trend,
            "current_progress": progress.get("progress_score"),
            "belief_alignment": belief_model,
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
    v1.4: Now includes belief-aware optimization.
    """
    
    def __init__(self, user_id: UUID):
        self.user_id = user_id
        # NEW: Initialize belief network
        self.belief_network = BeliefNetwork(user_id)
        
    async def optimize_intervention(
        self,
        context: Dict[str, Any],
        available_interventions: List[str]
    ) -> Dict[str, Any]:
        """
        Select best intervention using Thompson sampling approach.
        Balances exploration (try new things) vs exploitation (use what works).
        v1.4: Now filters interventions based on user's belief state.
        """
        store = await get_behavioral_store()
        
        # NEW: Check if user is in a belief-challenged state
        current_beliefs = await self.belief_network.get_current_beliefs()
        
        # Filter interventions that would cause cognitive dissonance
        safe_interventions = self._filter_by_belief_compatibility(
            available_interventions, 
            current_beliefs
        )
        
        # If no safe interventions, default to observation
        if not safe_interventions:
            return {
                "selected_intervention": "observe_more",
                "selection_method": "belief_safety_filter",
                "reason": "All interventions conflict with current user beliefs - would cause rebound"
            }
        
        # Get historical effectiveness of each safe intervention
        effectiveness = {}
        for intervention in safe_interventions:
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
            "selection_method": "thompson_sampling_with_belief_filter",
            "estimated_success_probability": best_sample,
            "exploration_vs_exploitation": "exploration" if effectiveness[best_intervention]["n_trials"] < 10 else "exploitation",
            "alternatives_considered": list(effectiveness.keys()),
            "filtered_out": list(set(available_interventions) - set(safe_interventions)),
            "belief_compatibility_checked": True
        }
    
    def _filter_by_belief_compatibility(
        self, 
        interventions: List[str], 
        beliefs: Dict
    ) -> List[str]:
        """Filter interventions that would clash with user beliefs."""
        safe = []
        
        for intervention in interventions:
            # Check compatibility
            if "focus" in intervention and beliefs.get("believes_multitasking_is_better"):
                # User believes multitasking works - direct focus challenge might rebound
                continue
            if "rest" in intervention and beliefs.get("believes_push_through_fatigue"):
                # User believes in pushing through - rest suggestion might be rejected
                continue
            
            safe.append(intervention)
        
        return safe
    
    async def _get_intervention_history(self, intervention_type: str) -> List[Dict]:
        """Get past outcomes for this intervention type."""
        # Query Supabase for interventions of this type and their outcomes
        return []  # Placeholder
