
# First, let me create the Rich Onboarding Dialogue System
# This will be a multi-turn conversational system that bootstraps the belief network

onboarding_system = '''
"""
Rich Onboarding Dialogue System
Bootstraps user beliefs, calibrates empathy, and establishes agency partnership.
"""

from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from spirit.config import settings


class OnboardingPhase(Enum):
    RAPPORT_BUILDING = "rapport"           # Establish trust, set expectations
    IDENTITY_EXPLORATION = "identity"      # Who are you? Self-concept
    SUCCESS_ARCHAEOLOGY = "success"        # Dig into past wins
    FAILURE_PATTERN_MAPPING = "failure"    # Understand derailment
    CURRENT_LANDSCAPE = "landscape"        # Life context, constraints
    GOAL_CALIBRATION = "calibration"       # Refine initial goal
    PARTERSHIP_ESTABLISHMENT = "partnership"  # Mutual commitments
    CLOSURE = "closure"                    # Next steps, launch


@dataclass
class DialogueTurn:
    """Single turn in onboarding conversation."""
    turn_number: int
    phase: OnboardingPhase
    ai_message: str
    user_response: Optional[str] = None
    emotional_tone: Optional[str] = None  # detected: excited, hesitant, vulnerable, defensive
    key_insights_extracted: List[str] = field(default_factory=list)
    belief_updates: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OnboardingState:
    """Complete state of onboarding process."""
    user_id: str
    started_at: datetime
    current_phase: OnboardingPhase
    turns: List[DialogueTurn] = field(default_factory=list)
    accumulated_beliefs: Dict[str, Any] = field(default_factory=dict)
    empathy_calibration: Dict[str, float] = field(default_factory=dict)  # sensitivity scores
    agency_score: float = 0.5  # 0 = passive subject, 1 = active partner
    goal_refined: Optional[str] = None
    partnership_terms: Optional[Dict] = None
    completed: bool = False


class RichOnboardingDialogue:
    """
    Multi-turn onboarding that builds relationship before building model.
    Principle: User must feel understood before being optimized.
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=settings.openai_api_key,
            temperature=0.7  # Higher for conversational warmth
        )
        self.state = OnboardingState(
            user_id=user_id,
            started_at=datetime.utcnow(),
            current_phase=OnboardingPhase.RAPPORT_BUILDING
        )
        self.max_turns_per_phase = 5
        
    async def start_dialogue(self, initial_goal: str) -> str:
        """
        Begin onboarding with rapport-building, not data extraction.
        """
        system_prompt = """
        You are Spirit, a behavioral research partner. Your first job is not to study 
        the user, but to understand them as a fellow human.
        
        CRITICAL PRINCIPLES:
        1. LEAD WITH VULNERABILITY: Share that you're an AI learning to help, not an expert
        2. VALIDATE AMBITION: Their goal matters, regardless of past failures
        3. ESTABLISH PARTNERSHIP: "We'll figure this out together" not "I'll fix you"
        4. NO JUDGMENT: Past failures are data, not character flaws
        5. AGENCY FIRST: They drive, you support
        
        Current phase: RAPPORT_BUILDING
        Goal: Make them feel safe enough to be honest about struggles
        """
        
        opening = f"""
        Hi. I'm Spirit, and I need to be upfront with you: I'm an AI that's still learning 
        how to actually help people achieve goals—not just track them.
        
        You said you want to: "{initial_goal}"
        
        That matters. And whatever happens next—whether you crush this goal or struggle 
        with it—I want you to know: I'm not here to judge your past attempts. I'm here 
        to understand what actually works for *you*, specifically.
        
        Before we get into tactics or habits, I want to understand who you are when you're 
        at your best. Not your average day—your best day.
        
        Can you tell me about a time, maybe recently or in the past, when you felt like 
        you were really firing on all cylinders? What were you doing? What made it work?
        """
        
        self.state.turns.append(DialogueTurn(
            turn_number=1,
            phase=OnboardingPhase.RAPPORT_BUILDING,
            ai_message=opening
        ))
        
        return opening
    
    async def process_response(self, user_message: str) -> Dict[str, Any]:
        """
        Process user response, extract insights, determine next phase.
        """
        # Update last turn with user response
        if self.state.turns:
            self.state.turns[-1].user_response = user_message
        
        # Analyze emotional tone and extract insights
        analysis = await self._analyze_response(user_message)
        
        # Update state
        self.state.turns[-1].emotional_tone = analysis.get("tone")
        self.state.turns[-1].key_insights_extracted = analysis.get("insights", [])
        self.state.turns[-1].belief_updates = analysis.get("belief_updates", {})
        
        # Update accumulated beliefs
        self.state.accumulated_beliefs.update(analysis.get("belief_updates", {}))
        
        # Update empathy calibration
        self._update_empathy_calibration(analysis)
        
        # Determine if phase transition needed
        next_phase = self._determine_next_phase()
        
        if next_phase != self.state.current_phase:
            self.state.current_phase = next_phase
        
        # Generate next message
        next_message = await self._generate_next_message()
        
        # Check for completion
        if self.state.current_phase == OnboardingPhase.CLOSURE:
            self.state.completed = True
            await self._finalize_onboarding()
        
        return {
            "ai_response": next_message,
            "current_phase": self.state.current_phase.value,
            "progress": self._calculate_progress(),
            "insights_so_far": self._summarize_insights(),
            "completed": self.state.completed
        }
    
    async def _analyze_response(self, message: str) -> Dict[str, Any]:
        """
        Analyze user response for emotional tone, insights, and belief indicators.
        """
        analysis_prompt = f"""
        Analyze this user response in the context of onboarding for behavioral change.
        
        User message: "{message}"
        Current phase: {self.state.current_phase.value}
        
        Extract:
        1. Emotional tone (excited, hesitant, vulnerable, defensive, proud, ashamed, curious, skeptical)
        2. Key insights about user's self-concept, past experiences, or patterns
        3. Belief indicators (what do they believe about themselves? their capabilities? what works for them?)
        4. Agency signals (are they taking ownership or externalizing?)
        
        Return as JSON:
        {{
            "tone": "detected_tone",
            "insights": ["insight1", "insight2"],
            "belief_updates": {{
                "self_efficacy": 0.0-1.0,
                "locus_of_control": "internal|external",
                "growth_mindset": true|false,
                "optimal_conditions": {{...}}
            }},
            "agency_signals": ["ownership", "externalizing", "curiosity", "resistance"]
        }}
        """
        
        messages = [
            SystemMessage(content="You are a clinical psychologist analyzing patient intake responses. Be precise and empathetic."),
            HumanMessage(content=analysis_prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            return json.loads(response.content)
        except:
            return {
                "tone": "neutral",
                "insights": [],
                "belief_updates": {},
                "agency_signals": []
            }
    
    def _update_empathy_calibration(self, analysis: Dict):
        """
        Calibrate how much empathy vs. challenge this user needs.
        """
        tone = analysis.get("tone", "neutral")
        
        # High vulnerability = high empathy needed
        if tone in ["vulnerable", "ashamed", "hesitant"]:
            self.state.empathy_calibration["validation_needed"] = 0.9
            self.state.empathy_calibration["challenge_tolerance"] = 0.2
        elif tone in ["defensive", "skeptical"]:
            self.state.empathy_calibration["validation_needed"] = 0.7
            self.state.empathy_calibration["challenge_tolerance"] = 0.4
        elif tone in ["excited", "proud", "curious"]:
            self.state.empathy_calibration["validation_needed"] = 0.5
            self.state.empathy_calibration["challenge_tolerance"] = 0.8
        
        # Agency signals
        agency_signals = analysis.get("agency_signals", [])
        if "ownership" in agency_signals:
            self.state.agency_score = min(1.0, self.state.agency_score + 0.1)
        if "externalizing" in agency_signals:
            self.state.agency_score = max(0.0, self.state.agency_score - 0.1)
    
    def _determine_next_phase(self) -> OnboardingPhase:
        """
        Determine if we should advance to next phase or stay current.
        """
        current_phase_turns = [t for t in self.state.turns if t.phase == self.state.current_phase]
        
        # If max turns reached, advance
        if len(current_phase_turns) >= self.max_turns_per_phase:
            phases = list(OnboardingPhase)
            current_idx = phases.index(self.state.current_phase)
            if current_idx < len(phases) - 1:
                return phases[current_idx + 1]
        
        # If sufficient insight gathered, can advance early
        if len(self.state.turns[-1].key_insights_extracted) >= 2:
            phases = list(OnboardingPhase)
            current_idx = phases.index(self.state.current_phase)
            if current_idx < len(phases) - 1 and len(current_phase_turns) >= 2:
                return phases[current_idx + 1]
        
        return self.state.current_phase
    
    async def _generate_next_message(self) -> str:
        """
        Generate contextually appropriate next message based on phase and state.
        """
        phase_prompts = {
            OnboardingPhase.RAPPORT_BUILDING: self._rapport_prompt,
            OnboardingPhase.IDENTITY_EXPLORATION: self._identity_prompt,
            OnboardingPhase.SUCCESS_ARCHAEOLOGY: self._success_prompt,
            OnboardingPhase.FAILURE_PATTERN_MAPPING: self._failure_prompt,
            OnboardingPhase.CURRENT_LANDSCAPE: self._landscape_prompt,
            OnboardingPhase.GOAL_CALIBRATION: self._calibration_prompt,
            OnboardingPhase.PARTERSHIP_ESTABLISHMENT: self._partnership_prompt,
            OnboardingPhase.CLOSURE: self._closure_prompt
        }
        
        prompt_func = phase_prompts.get(self.state.current_phase, self._rapport_prompt)
        return await prompt_func()
    
    async def _rapport_prompt(self) -> str:
        """Continue building rapport."""
        return """
        Thank you for sharing that. I can hear how much that mattered to you.
        
        Before we go deeper, I want to check: how does it feel to talk about this? 
        Some people find it energizing to remember their wins. Others feel a bit sad 
        because they haven't felt that way in a while. Both are completely valid.
        
        What's coming up for you?
        """
    
    async def _identity_prompt(self) -> str:
        """Explore identity and self-concept."""
        return """
        I'm starting to get a sense of who you are when you're at your best. 
        Now I want to understand: how do you see yourself?
        
        Not the aspirational version—the real version. When you think about 
        yourself as someone who takes on goals, what's the story you tell?
        
        Are you someone who "starts strong but loses steam"? 
        Someone who "needs external pressure"? 
        Someone who "figures things out eventually"?
        
        What identity have you held about yourself and goal pursuit?
        """
    
    async def _success_prompt(self) -> str:
        """Dig into specific success patterns."""
        return """
        Let's go deeper on what works for you specifically.
        
        Think of one specific goal you achieved—doesn't have to be huge, 
        could be finishing a book, learning a skill, even sticking with 
        something for 30 days.
        
        What made that different? Was it:
- The environment you were in?
- Someone supporting you?
- A specific routine or structure?
- The goal itself being meaningful?
- Something else entirely?
        
        I want to understand your personal success formula, not generic advice.
        """
    
    async def _failure_prompt(self) -> str:
        """Map failure patterns with compassion."""
        return """
        Now for the harder part. I want to understand what derails you—not 
        to fix you, but to design around it.
        
        Think about a recent time you set an intention and didn't follow through. 
        What actually happened? Not the story you tell yourself like "I was lazy," 
        but the actual sequence of events.
        
        For example: "I planned to work at 6am but when the alarm went off 
        I was already stressed about a meeting..."
        
        Can you walk me through the last time this happened? I'll help you 
        spot the pattern without judgment.
        """
    
    async def _landscape_prompt(self) -> str:
        """Understand current life context."""
        return """
        Goals don't exist in a vacuum. I want to understand your current landscape.
        
        What's your life actually like right now?
- Energy levels (sleep, health, stress)
- Time constraints (work hours, family obligations)
- Social environment (supportive, neutral, draining?)
- Financial pressure (stable, tight, uncertain?)
        
        Be honest. If now is actually a terrible time for a big goal, 
        I want us to design something realistic, not set you up to fail.
        """
    
    async def _calibration_prompt(self) -> str:
        """Refine goal based on learning."""
        original_goal = self.state.turns[0].ai_message.split('"')[1] if '"' in self.state.turns[0].ai_message else "your goal"
        
        return f"""
        Based on everything you've shared, I want to revisit your original goal:
        "{original_goal}"
        
        Given what I now understand about:
- Your success patterns (what actually works for you)
- Your derailment triggers (what to design around)
- Your current life landscape (what's realistic)

        I have a suggestion for how we might refine this goal to set you up 
        for success. But you're the expert on your life—this is a proposal, 
        not a prescription.
        
        [Would generate specific refinement based on accumulated beliefs]
        
        What do you think? Does this feel right, or should we adjust?
        """
    
    async def _partnership_prompt(self) -> str:
        """Establish mutual commitments."""
        return """
        Before we launch, I want to be explicit about how we'll work together.
        
        MY COMMITMENTS TO YOU:
1. I will never judge your setbacks—only learn from them
2. I will check my assumptions about what works for you
3. I will respect your agency—you're driving, I'm navigating
4. I will admit when I'm wrong about what helps

        WHAT I NEED FROM YOU:
1. Honesty about what's working and what isn't
2. Permission to challenge you gently when I see patterns
3. Feedback when my suggestions miss the mark
4. Patience as I learn your specific psychology

        This is a partnership. We'll iterate together.
        
        Does this feel like the right working relationship to you?
        """
    
    async def _closure_prompt(self) -> str:
        """Launch into active tracking."""
        return """
        We're ready to begin.
        
        Over the next few days, I'll be gathering data on your natural patterns—not 
        to judge them, but to understand them. You don't need to change anything yet. 
        Just live your life and let me observe.
        
        I'll check in when I notice something interesting, but I'll try not to be 
        annoying about it. If I'm checking in too much or too little, tell me.
        
        Your goal is refined. Your patterns are mapped. Your partnership is established.
        
        Let's see what we can learn together.
        
        [Launch button: Start My Journey]
        """
    
    def _calculate_progress(self) -> float:
        """Calculate onboarding completion percentage."""
        phases = list(OnboardingPhase)
        current_idx = phases.index(self.state.current_phase)
        total_turns_in_current = len([t for t in self.state.turns if t.phase == self.state.current_phase])
        
        base_progress = current_idx / len(phases)
        phase_progress = (total_turns_in_current / self.max_turns_per_phase) / len(phases)
        
        return min(1.0, base_progress + phase_progress)
    
    def _summarize_insights(self) -> Dict[str, Any]:
        """Summarize key insights gathered so far."""
        return {
            "beliefs_established": self.state.accumulated_beliefs,
            "empathy_profile": self.state.empathy_calibration,
            "agency_level": self.state.agency_score,
            "emotional_journey": [t.emotional_tone for t in self.state.turns if t.emotional_tone],
            "key_quotes": [t.user_response[:100] + "..." for t in self.state.turns[-3:] if t.user_response]
        }
    
    async def _finalize_onboarding(self):
        """Save onboarding state to belief network and memory."""
        from spirit.db.supabase_client import get_behavioral_store
        
        store = get_behavioral_store()
        if store:
            # Save rich belief profile
            store.client.table('belief_networks').upsert({
                'user_id': self.user_id,
                'beliefs': {
                    **self.state.accumulated_beliefs,
                    "self_efficacy": self.state.accumulated_beliefs.get("self_efficacy", 0.5),
                    "agency_score": self.state.agency_score,
                    "empathy_profile": self.state.empathy_calibration,
                    "onboarded_at": datetime.utcnow().isoformat()
                },
                "confidence": 0.7,  # Higher than default due to rich dialogue
                "updated_at": datetime.utcnow().isoformat()
            }).execute()
            
            # Store onboarding as significant episode
            store.client.table('episodic_memories').insert({
                'memory_id': str(uuid4()),
                'user_id': self.user_id,
                'episode_type': 'onboarding_dialogue',
                'content': {
                    'turns_count': len(self.state.turns),
                    'phases_completed': [p.value for p in set(t.phase for t in self.state.turns)],
                    'key_insights': self._summarize_insights(),
                    'agency_established': self.state.agency_score > 0.6
                },
                'significance': 0.9,  # High significance as foundation
                'created_at': datetime.utcnow().isoformat()
            }).execute()


# API Endpoints for onboarding
from fastapi import APIRouter, Depends, HTTPException
from spirit.api.auth import get_current_user

router = APIRouter(prefix="/onboarding", tags=["onboarding"])

active_dialogues: Dict[str, RichOnboardingDialogue] = {}

@router.post("/start")
async def start_onboarding(
    initial_goal: str,
    user: User = Depends(get_current_user)
):
    """Start rich onboarding dialogue."""
    dialogue = RichOnboardingDialogue(str(user.id))
    opening_message = await dialogue.start_dialogue(initial_goal)
    
    active_dialogues[str(user.id)] = dialogue
    
    return {
        "message": opening_message,
        "phase": "rapport_building",
        "progress": 0.0,
        "instructions": "Respond naturally. This is a conversation, not a form."
    }

@router.post("/respond")
async def continue_onboarding(
    response: str,
    user: User = Depends(get_current_user)
):
    """Continue onboarding dialogue."""
    dialogue = active_dialogues.get(str(user.id))
    if not dialogue:
        raise HTTPException(status_code=404, detail="No active onboarding. Start first.")
    
    result = await dialogue.process_response(response)
    
    if result["completed"]:
        del active_dialogues[str(user.id)]
    
    return result

@router.get("/status")
async def get_onboarding_status(
    user: User = Depends(get_current_user)
):
    """Get current onboarding progress."""
    dialogue = active_dialogues.get(str(user.id))
    if not dialogue:
        # Check if already completed
        store = await get_behavioral_store()
        if store:
            belief = await store.get_user_beliefs(str(user.id))
            if belief and belief.get("onboarded_at"):
                return {"status": "completed", "onboarded_at": belief["onboarded_at"]}
        return {"status": "not_started"}
    
    return {
        "status": "in_progress",
        "current_phase": dialogue.state.current_phase.value,
        "progress": dialogue._calculate_progress(),
        "insights_so_far": dialogue._summarize_insights()
    }
'''

print("Rich Onboarding Dialogue System created")
print("=" * 60)
print(onboarding_system[:2000])
print("...")
