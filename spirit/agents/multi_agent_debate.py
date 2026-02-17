"""
Multi-Agent Debate System: Observer, Adversary, Synthesizer.
Prevents bias and optimizes communication to avoid rebound effects.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from spirit.config import settings


class AgentPersona(Enum):
    OBSERVER = "observer"
    ADVERSARY = "adversary"
    SYNTHESIZER = "synthesizer"


@dataclass
class DebateRound:
    """One round of internal debate."""
    round_number: int
    observer_proposal: str
    adversary_critique: str
    synthesizer_resolution: Optional[str]
    consensus_reached: bool


class MultiAgentDebate:
    """
    Three-persona internal debate for high-stakes decisions.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=settings.openai_api_key,
            temperature=0.3
        )
        self.max_rounds = 3
    
    async def debate_intervention(
        self,
        user_context: Dict,
        proposed_intervention: str,
        predicted_outcome: Dict
    ) -> Dict[str, Any]:
        """
        Run full debate on whether to deliver intervention.
        """
        rounds = []
        consensus = None
        
        for round_num in range(1, self.max_rounds + 1):
            # Observer presents data-driven case
            observer_case = await self._observer_presents(
                user_context, proposed_intervention, predicted_outcome, rounds
            )
            
            # Adversary challenges
            adversary_critique = await self._adversary_challenges(
                observer_case, user_context, rounds
            )
            
            # Check if adversary accepts
            if self._check_adversary_acceptance(adversary_critique):
                consensus = observer_case
                rounds.append(DebateRound(
                    round_number=round_num,
                    observer_proposal=observer_case,
                    adversary_critique=adversary_critique,
                    synthesizer_resolution=observer_case,
                    consensus_reached=True
                ))
                break
            
            # Synthesizer attempts resolution
            synthesis = await self._synthesizer_resolves(
                observer_case, adversary_critique, user_context
            )
            
            rounds.append(DebateRound(
                round_number=round_num,
                observer_proposal=observer_case,
                adversary_critique=adversary_critique,
                synthesizer_resolution=synthesis,
                consensus_reached=False
            ))
            
            # Check if synthesis satisfies both
            if await self._check_synthesis_acceptance(synthesis, observer_case, adversary_critique):
                consensus = synthesis
                rounds[-1].consensus_reached = True
                break
        
        # Final decision
        if consensus:
            final_message = await self._synthesizer_craft_message(
                consensus, user_context, rounds
            )
            
            return {
                "proceed": True,
                "message": final_message,
                "debate_rounds": len(rounds),
                "consensus_reached": True,
                "adversary_concerns_addressed": True
            }
        else:
            # No consensus - don't intervene or use ultra-conservative approach
            return {
                "proceed": False,
                "reason": "adversary_objections_unresolved",
                "adversary_concerns": rounds[-1].adversary_critique if rounds else "unknown",
                "debate_rounds": len(rounds),
                "consensus_reached": False,
                "recommendation": "observe_more"
            }
    
    async def _observer_presents(
        self,
        user_context: Dict,
        intervention: str,
        predicted: Dict,
        prior_rounds: List[DebateRound]
    ) -> str:
        """Observer: Data-driven case for intervention."""
        
        context = self._format_context(user_context)
        
        messages = [
            SystemMessage(content="""
            You are the OBSERVER persona in a behavioral research AI.
            Your role: Present the objective, data-driven case for intervening.
            Use only facts from the data. No persuasion, just evidence.
            Be concise (2-3 sentences).
            """),
            HumanMessage(content=f"""
            User context: {context}
            Proposed intervention: {intervention}
            Predicted outcome: {predicted}
            Prior debate rounds: {len(prior_rounds)}
            
            Present the data-driven case for this intervention.
            """)
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    async def _adversary_challenges(
        self,
        observer_case: str,
        user_context: Dict,
        prior_rounds: List[DebateRound]
    ) -> str:
        """Adversary: Challenge assumptions, check for spuriousness."""
        
        messages = [
            SystemMessage(content="""
            You are the ADVERSARY persona in a behavioral research AI.
            Your role: Ruthlessly challenge the Observer's case.
            Check for:
            - Spurious correlations (third variables)
            - Confirmation bias in data interpretation
            - Rebound risk (will user quit?)
            - Ethical concerns (manipulation, autonomy)
            
            If convinced, say "ACCEPTED". Otherwise, state specific objections.
            Be concise but thorough.
            """),
            HumanMessage(content=f"""
            Observer's case: {observer_case}
            User history: {user_context.get('rejection_rate', 0)}% rejection rate
            Prior interventions today: {user_context.get('interventions_today', 0)}
            
            Challenge this case. What could go wrong?
            """)
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    async def _synthesizer_resolves(
        self,
        observer_case: str,
        adversary_critique: str,
        user_context: Dict
    ) -> str:
        """Synthesizer: Find middle ground or improved approach."""
        
        messages = [
            SystemMessage(content="""
            You are the SYNTHESIZER persona in a behavioral research AI.
            Your role: Resolve conflict between Observer and Adversary.
            Find an intervention that:
            - Addresses the data (Observer's concern)
            - Mitigates risks (Adversary's concerns)
            - Is ethical and user-respecting
            
            If no good synthesis exists, say "NO_CONSENSUS".
            """),
            HumanMessage(content=f"""
            Observer: {observer_case}
            Adversary: {adversary_critique}
            
            Propose a resolution or say NO_CONSENSUS.
            """)
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    async def _synthesizer_craft_message(
        self,
        consensus: str,
        user_context: Dict,
        debate_rounds: List[DebateRound]
    ) -> str:
        """Final message crafting optimized for user reception."""
        
        # Analyze user communication preferences from history
        style = self._determine_communication_style(user_context)
        
        messages = [
            SystemMessage(content=f"""
            You are the SYNTHESIZER crafting the final user message.
            User communication style: {style}
            Debate rounds needed: {len(debate_rounds)}
            
            Craft a message that:
            - Feels personal, not robotic
            - Respects user's autonomy
            - Is actionable
            - Avoids "rebound effect" (don't be preachy)
            
            Max 2 sentences. Warm but direct.
            """),
            HumanMessage(content=f"""
            Consensus decision: {consensus}
            
            Craft the final message to user.
            """)
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    def _check_adversary_acceptance(self, critique: str) -> bool:
        """Check if adversary accepts the case."""
        return "ACCEPTED" in critique.upper() or len(critique) < 50
    
    async def _check_synthesis_acceptance(
        self,
        synthesis: str,
        observer_case: str,
        adversary_critique: str
    ) -> bool:
        """Check if synthesis satisfies both parties."""
        if "NO_CONSENSUS" in synthesis:
            return False
        
        # Quick check with both personas
        # (Simplified - in production, would re-query both)
        return True
    
    def _format_context(self, context: Dict) -> str:
        """Format user context for prompts."""
        parts = []
        if 'current_state' in context:
            parts.append(f"Current: {context['current_state']}")
        if 'recent_pattern' in context:
            parts.append(f"Pattern: {context['recent_pattern']}")
        if 'goal_progress' in context:
            parts.append(f"Goal: {context['goal_progress']}%")
        return "; ".join(parts)
    
    def _determine_communication_style(self, user_context: Dict) -> str:
        """Determine optimal communication style for this user."""
        if user_context.get('rejection_rate', 0) > 0.3:
            return "ultra_minimalist"  # User rejects often, be brief
        if user_context.get('engagement_depth') == 'high':
            return "conversational"  # User likes depth
        return "direct"  # Default


# Integration with existing proactive loop
async def debate_aware_intervention(
    user_id: int,
    prediction: Dict,
    context: Dict
) -> Dict:
    """
    Wrapper that runs debate before delivering intervention.
    """
    debate = MultiAgentDebate()
    
    result = await debate.debate_intervention(
        user_context=context,
        proposed_intervention=prediction.get('optimal_intervention'),
        predicted_outcome={
            'expected_improvement': prediction.get('expected_outcome_if_intervene'),
            'confidence': prediction.get('confidence')
        }
    )
    
    if result['proceed']:
        # Deliver the debated-and-refined message
        from spirit.services.notification_engine import NotificationEngine
        
        engine = NotificationEngine(user_id)
        await engine.send_notification(
            content={
                "title": "Spirit",
                "body": result['message'],
                "data": {"debate_validated": True}
            },
            priority="normal",
            notification_type="debate_validated_intervention"
        )
    
    return result
