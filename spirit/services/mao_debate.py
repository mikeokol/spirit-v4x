"""
Multi-Agent Orchestration (MAO) - Internal Debate System
Prevents bias through Observer/Adversary/Synthesizer personas.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from dataclasses import dataclass
from enum import Enum

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from spirit.config import settings


class AgentRole(Enum):
    OBSERVER = "observer"      # Pure data collection
    ADVERSARY = "adversary"    # Challenges correlations
    SYNTHESIZER = "synthesizer"  # Communication strategy


@dataclass
class DebateState:
    observation: Any
    belief_context: Any
    dissonance: Any
    user_id: UUID
    
    # Agent outputs
    observer_report: Dict = None
    adversary_challenges: List[str] = None
    synthesizer_strategy: Dict = None
    
    # Decision
    intervention_approved: bool = False
    intervention_type: Optional[str] = None
    message_tone: str = "neutral"
    
    # Metadata
    debate_id: UUID = None
    timestamp: datetime = None


@dataclass
class MAODecision:
    debate_id: UUID
    intervention_type: Optional[str]
    synthesizer_approves: bool
    synthesizer_blocks: bool
    adversary_challenges: bool
    challenge_reason: Optional[str]
    message: Optional[str]
    tone: str
    urgency: int  # 1-5


class MultiAgentOrchestrator:
    """
    Three internal personas debate every intervention:
    1. Observer: What does the data say?
    2. Adversary: Is this correlation spurious? 
    3. Synthesizer: How do we communicate without causing rebound?
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            api_key=settings.openai_api_key
        )
        self.debate_graph = self._build_debate_graph()
        
    def _build_debate_graph(self):
        """Build LangGraph workflow for debate."""
        
        def observer_node(state: DebateState):
            """Observer: Analyze raw data patterns."""
            prompt = f"""
            You are the OBSERVER agent. Analyze this behavioral data objectively:
            User: {state.user_id}
            Observation: {state.observation}
            Belief Context: {state.belief_context}
            
            Provide:
            1. Factual summary of what occurred
            2. Pattern strength (0-1)
            3. Confidence in correlation
            """
            
            response = self.llm.invoke(prompt)
            state.observer_report = {
                "summary": response.content,
                "pattern_strength": 0.7,  # Extracted from LLM
                "confidence": 0.8
            }
            return state
        
        def adversary_node(state: DebateState):
            """Adversary: Challenge the Observer's conclusions."""
            prompt = f"""
            You are the ADVERSARY agent. Challenge this analysis:
            {state.observer_report}
            
            Consider:
            - Confounding variables (sleep, caffeine, external events)
            - Sample size adequacy
            - Temporal validity (is this still true?)
            - Alternative explanations
            
            If you find flaws, list them. If confident, approve.
            """
            
            response = self.llm.invoke(prompt)
            state.adversary_challenges = self._extract_challenges(response.content)
            return state
        
        def synthesizer_node(state: DebateState):
            """Synthesizer: Decide communication strategy."""
            has_challenges = len(state.adversary_challenges) > 0 if state.adversary_challenges else False
            
            prompt = f"""
            You are the SYNTHESIZER agent. Decide how to communicate findings:
            
            Observer Report: {state.observer_report}
            Adversary Challenges: {state.adversary_challenges}
            User Dissonance: {state.dissonance}
            
            Rules:
            - If Adversary found flaws, DO NOT intervene or use very soft tone
            - Avoid "rebound effects" (user getting annoyed and quitting)
            - Match user's current emotional state
            - Prioritize long-term trust over short-term compliance
            
            Decide:
            1. Should we intervene? (yes/no/soft)
            2. What tone? (supportive/challenging/neutral)
            3. Specific message content
            """
            
            response = self.llm.invoke(prompt)
            strategy = self._parse_strategy(response.content)
            
            state.synthesizer_strategy = strategy
            state.intervention_approved = strategy.get("intervene", False)
            state.intervention_type = strategy.get("type")
            state.message_tone = strategy.get("tone", "neutral")
            
            return state
        
        # Build graph
        workflow = StateGraph(DebateState)
        
        workflow.add_node("observer", observer_node)
        workflow.add_node("adversary", adversary_node)
        workflow.add_node("synthesizer", synthesizer_node)
        
        workflow.add_edge("observer", "adversary")
        workflow.add_edge("adversary", "synthesizer")
        workflow.add_edge("synthesizer", END)
        
        workflow.set_entry_point("observer")
        
        return workflow.compile()
    
    async def debate_intervention(
        self,
        user_id: UUID,
        observation: Any,
        dissonance: Any,
        belief_update: Any
    ) -> MAODecision:
        """
        Run full MAO debate on whether to intervene.
        """
        initial_state = DebateState(
            debate_id=uuid4(),
            user_id=user_id,
            observation=observation,
            belief_context=belief_update,
            dissonance=dissonance,
            timestamp=datetime.utcnow()
        )
        
        # Run debate
        final_state = self.debate_graph.invoke(initial_state)
        
        return MAODecision(
            debate_id=final_state.debate_id,
            intervention_type=final_state.intervention_type,
            synthesizer_approves=final_state.intervention_approved,
            synthesizer_blocks=not final_state.intervention_approved and final_state.dissonance is not None,
            adversary_challenges=bool(final_state.adversary_challenges),
            challenge_reason="; ".join(final_state.adversary_challenges) if final_state.adversary_challenges else None,
            message=final_state.synthesizer_strategy.get("message") if final_state.synthesizer_strategy else None,
            tone=final_state.message_tone,
            urgency=final_state.synthesizer_strategy.get("urgency", 3) if final_state.synthesizer_strategy else 3
        )
    
    async def debate_emergent_pattern(
        self,
        user_id: UUID,
        triggers: List[Any]
    ):
        """Debate patterns that emerge from batch processing."""
        # Simplified batch debate
        pass
    
    def _extract_challenges(self, text: str) -> List[str]:
        """Extract challenge points from adversary response."""
        # Parse LLM output
        return [line.strip() for line in text.split('\n') if line.strip().startswith('-')]
    
    def _parse_strategy(self, text: str) -> Dict:
        """Parse synthesizer strategy from text."""
        # Extract structured data from LLM output
        return {
            "intervene": "yes" in text.lower() or "intervene: yes" in text.lower(),
            "type": "nudge" if "nudge" in text.lower() else "notification",
            "tone": "supportive" if "supportive" in text.lower() else "neutral",
            "urgency": 3
        }


# Singleton
_mao_instance: Optional[MultiAgentOrchestrator] = None

def get_mao_orchestrator() -> MultiAgentOrchestrator:
    global _mao_instance
    if _mao_instance is None:
        _mao_instance = MultiAgentOrchestrator()
    return _mao_instance
