
# Create Empathy Calibration and Agency Preservation Systems

empathy_agency_system = '''
"""
Empathy Calibration & Agency Preservation System
Ensures Spirit feels like a partner, not a scientist studying a subject.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from spirit.config import settings


class EmpathyMode(Enum):
    HIGH_VALIDATION = "high_validation"      # For vulnerable/ashamed users
    BALANCED = "balanced"                    # Default
    HIGH_CHALLENGE = "high_challenge"        # For high-agency users who want pushing
    GRIEF_SUPPORT = "grief_support"          # For users processing failure
    CELEBRATION = "celebration"              # For wins, momentum building


class AgencyInterventionType(Enum):
    SUGGESTION = "suggestion"                # "You might consider..."
    QUESTION = "question"                    # "What do you think about...?"
    OFFER = "offer"                          # "I can help with... if you want"
    DIRECTIVE = "directive"                  # "Try this now" (rare, high trust only)
    COLLABORATION = "collaboration"          # "Let's figure this out together"


@dataclass
class EmpathyProfile:
    """User-specific empathy calibration."""
    user_id: str
    validation_need: float  # 0-1, how much affirmation needed
    challenge_tolerance: float  # 0-1, how much pushback welcomed
    preferred_tone: str  # warm, direct, playful, clinical
    trigger_sensitivity: Dict[str, float]  # topics that need extra care
    successful_approaches: List[str]  # what's worked historically
    failed_approaches: List[str]  # what backfired
    last_updated: datetime


@dataclass
class AgencySnapshot:
    """Current agency state of user."""
    user_id: str
    ownership_level: float  # 0-1, taking responsibility vs externalizing
    decision_making: float  # 0-1, active vs passive
    help_seeking: float  # 0-1, appropriate vs avoidant or dependent
    resistance_level: float  # 0-1, pushing back (can be healthy or defensive)
    collaboration_readiness: float  # 0-1, open to partnership
    timestamp: datetime


class EmpathyCalibrationEngine:
    """
    Calibrates every interaction for appropriate emotional tone and support level.
    Prevents 'over-optimization' that feels robotic or dismissive.
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=settings.openai_api_key,
            temperature=0.4
        )
        self.profile = None
        
    async def load_profile(self):
        """Load or create empathy profile."""
        from spirit.db.supabase_client import get_behavioral_store
        
        store = get_behavioral_store()
        if store:
            result = store.client.table('empathy_profiles').select('*').eq(
                'user_id', self.user_id
            ).execute()
            
            if result.data:
                p = result.data[0]
                self.profile = EmpathyProfile(
                    user_id=self.user_id,
                    validation_need=p.get('validation_need', 0.5),
                    challenge_tolerance=p.get('challenge_tolerance', 0.5),
                    preferred_tone=p.get('preferred_tone', 'warm'),
                    trigger_sensitivity=p.get('trigger_sensitivity', {}),
                    successful_approaches=p.get('successful_approaches', []),
                    failed_approaches=p.get('failed_approaches', []),
                    last_updated=datetime.fromisoformat(p.get('last_updated'))
                )
            else:
                # Create default profile
                self.profile = EmpathyProfile(
                    user_id=self.user_id,
                    validation_need=0.5,
                    challenge_tolerance=0.5,
                    preferred_tone='warm',
                    trigger_sensitivity={},
                    successful_approaches=[],
                    failed_approaches=[],
                    last_updated=datetime.utcnow()
                )
    
    async def calibrate_interaction(
        self,
        context: str,
        user_emotional_state: str,
        proposed_intervention: str,
        intervention_type: AgencyInterventionType
    ) -> Dict[str, Any]:
        """
        Calibrate interaction for optimal empathy and agency preservation.
        Returns modified intervention with tone, framing, and delivery guidance.
        """
        if not self.profile:
            await self.load_profile()
        
        # Determine appropriate empathy mode
        mode = self._select_empathy_mode(user_emotional_state, context)
        
        # Adjust intervention type based on agency preservation
        adjusted_type = self._preserve_agency(intervention_type, context)
        
        # Generate calibrated message
        calibration_prompt = f"""
        Calibrate this intervention for user with the following profile:
        
        USER EMPATHY PROFILE:
        - Validation need: {self.profile.validation_need}/1.0
        - Challenge tolerance: {self.profile.challenge_tolerance}/1.0
        - Preferred tone: {self.profile.preferred_tone}
        - Known triggers: {list(self.profile.trigger_sensitivity.keys())}
        - Successful approaches: {self.profile.successful_approaches[-3:]}
        - Failed approaches: {self.profile.failed_approaches[-3:]}
        
        CURRENT CONTEXT:
        - User emotional state: {user_emotional_state}
        - Situation: {context}
        - Proposed intervention: {proposed_intervention}
        - Intervention type: {intervention_type.value}
        - Selected empathy mode: {mode.value}
        
        CALIBRATION TASK:
        1. Adjust the intervention to match user's validation needs
        2. Preserve agency (user feels in control, not managed)
        3. Use preferred tone
        4. Avoid known triggers or approach with extra care
        5. Frame as partnership, not prescription
        
        Return calibrated intervention and explanation of choices.
        """
        
        messages = [
            SystemMessage(content="""
            You are an empathy calibration expert. Your job is to ensure AI interactions 
            feel supportive without being patronizing, challenging without being harsh, 
            and helpful without being controlling.
            
            KEY PRINCIPLES:
            - Agency preservation: User must feel they are choosing, not being managed
            - Emotional attunement: Match their current state, don't bypass it
            - Growth mindset: Frame challenges as experiments, not character flaws
            - Partnership language: "We" and "together," not "you should"
            """)
        ]
        
        response = self.llm.invoke(messages + [HumanMessage(content=calibration_prompt)])
        
        # Parse response
        try:
            calibrated = json.loads(response.content)
        except:
            calibrated = {
                "intervention": proposed_intervention,
                "tone_adjustments": "maintained warm, supportive tone",
                "agency_preservation": "used collaborative framing"
            }
        
        return {
            "calibrated_intervention": calibrated.get("intervention", proposed_intervention),
            "empathy_mode": mode.value,
            "adjusted_intervention_type": adjusted_type.value,
            "tone_guidance": calibrated.get("tone_adjustments", ""),
            "agency_strategy": calibrated.get("agency_preservation", ""),
            "avoided_triggers": [t for t in self.profile.trigger_sensitivity if t in context],
            "confidence": self._calculate_calibration_confidence(mode, user_emotional_state)
        }
    
    def _select_empathy_mode(self, emotional_state: str, context: str) -> EmpathyMode:
        """Select appropriate empathy mode based on user state."""
        state_lower = emotional_state.lower()
        
        if any(word in state_lower for word in ['ashamed', 'failure', 'disappointed', 'gave up']):
            return EmpathyMode.HIGH_VALIDATION
        
        if any(word in state_lower for word in ['excited', 'motivated', 'ready', 'confident']):
            return EmpathyMode.HIGH_CHALLENGE
        
        if any(word in state_lower for word in ['sad', 'loss', 'grief', 'processing']):
            return EmpathyMode.GRIEF_SUPPORT
        
        if any(word in state_lower for word in ['achieved', 'completed', 'won', 'succeeded']):
            return EmpathyMode.CELEBRATION
        
        return EmpathyMode.BALANCED
    
    def _preserve_agency(
        self, 
        intervention_type: AgencyInterventionType, 
        context: str
    ) -> AgencyInterventionType:
        """
        Adjust intervention type to preserve user agency.
        Never use directive unless high trust and urgent.
        """
        # Default to collaborative approaches
        if intervention_type == AgencyInterventionType.DIRECTIVE:
            # Only keep directive if emergency context
            if 'urgent' in context.lower() or 'crisis' in context.lower():
                return AgencyInterventionType.DIRECTIVE
            return AgencyInterventionType.SUGGESTION
        
        # If user has low agency score, use offers rather than suggestions
        if self.profile and self.profile.validation_need > 0.7:
            if intervention_type == AgencyInterventionType.SUGGESTION:
                return AgencyInterventionType.OFFER
        
        return intervention_type
    
    def _calculate_calibration_confidence(self, mode: EmpathyMode, state: str) -> float:
        """How confident are we in this calibration?"""
        # Lower confidence for ambiguous states
        if mode == EmpathyMode.BALANCED:
            return 0.6
        # Higher confidence for clear states
        if any(word in state.lower() for word in ['very', 'extremely', 'definitely']):
            return 0.9
        return 0.75
    
    async def update_from_interaction(
        self,
        interaction_outcome: str,  # "positive", "negative", "neutral", "rejected"
        user_feedback: Optional[str],
        intervention_used: str
    ):
        """
        Update empathy profile based on interaction outcome.
        Learning what works for this specific user.
        """
        if interaction_outcome == "positive":
            self.profile.successful_approaches.append({
                "approach": intervention_used,
                "timestamp": datetime.utcnow().isoformat(),
                "context": "validated"
            })
            # Slightly increase challenge tolerance on success
            self.profile.challenge_tolerance = min(1.0, self.profile.challenge_tolerance + 0.05)
            
        elif interaction_outcome == "rejected":
            self.profile.failed_approaches.append({
                "approach": intervention_used,
                "timestamp": datetime.utcnow().isoformat(),
                "feedback": user_feedback
            })
            # Increase validation need on rejection
            self.profile.validation_need = min(1.0, self.profile.validation_need + 0.1)
            
        elif interaction_outcome == "negative":
            # Log as trigger if strong negative reaction
            if user_feedback and any(word in user_feedback.lower() for word in ['annoying', 'frustrating', 'hate', 'stop']):
                self.profile.trigger_sensitivity[intervention_used[:20]] = 0.9
        
        self.profile.last_updated = datetime.utcnow()
        await self._save_profile()
    
    async def _save_profile(self):
        """Save updated profile to database."""
        from spirit.db.supabase_client import get_behavioral_store
        
        store = get_behavioral_store()
        if store:
            store.client.table('empathy_profiles').upsert({
                'user_id': self.user_id,
                'validation_need': self.profile.validation_need,
                'challenge_tolerance': self.profile.challenge_tolerance,
                'preferred_tone': self.profile.preferred_tone,
                'trigger_sensitivity': self.profile.trigger_sensitivity,
                'successful_approaches': self.profile.successful_approaches[-10:],  # Keep last 10
                'failed_approaches': self.profile.failed_approaches[-10:],
                'last_updated': datetime.utcnow().isoformat()
            }).execute()


class AgencyPreservationMonitor:
    """
    Monitors all interventions for agency erosion.
    Acts as safeguard against overbearing AI behavior.
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.intervention_history: List[Dict] = []
        self.agency_score_history: List[float] = []
        
    async def assess_agency_impact(
        self, 
        proposed_intervention: str,
        intervention_frequency_24h: int
    ) -> Tuple[bool, str, float]:
        """
        Assess if proposed intervention preserves or erodes agency.
        Returns: (should_proceed, reason, predicted_agency_impact)
        """
        # Red flags for agency erosion
        red_flags = []
        
        # Frequency check
        if intervention_frequency_24h > 5:
            red_flags.append("high_frequency")
        
        # Language check
        directive_words = ['must', 'should', 'need to', 'have to', 'required']
        if any(word in proposed_intervention.lower() for word in directive_words):
            red_flags.append("directive_language")
        
        # Check for solution imposition vs collaborative solving
        if 'try this' in proposed_intervention.lower() and 'what do you think' not in proposed_intervention.lower():
            red_flags.append("solution_imposition")
        
        # Calculate agency impact score
        base_impact = 0.5  # Neutral
        
        if "high_frequency" in red_flags:
            base_impact -= 0.3
        if "directive_language" in red_flags:
            base_impact -= 0.2
        if "solution_imposition" in red_flags:
            base_impact -= 0.15
        
        # Check historical response to similar interventions
        similar_past = [h for h in self.intervention_history 
                       if h.get('intervention_type') in proposed_intervention]
        if similar_past:
            rejection_rate = sum(1 for s in similar_past if s.get('rejected')) / len(similar_past)
            if rejection_rate > 0.5:
                base_impact -= 0.2
                red_flags.append("historical_rejection")
        
        should_proceed = base_impact > 0.3  # Threshold for agency preservation
        
        reason = "Agency preserved" if should_proceed else f"Agency risk: {', '.join(red_flags)}"
        
        return should_proceed, reason, max(0.0, base_impact)
    
    def record_intervention(self, intervention: str, user_response: str):
        """Record intervention for historical analysis."""
        self.intervention_history.append({
            'intervention': intervention,
            'user_response': user_response,
            'rejected': 'no' in user_response.lower() or 'stop' in user_response.lower(),
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Keep only last 50
        self.intervention_history = self.intervention_history[-50:]


class PartnershipContract:
    """
    Explicit partnership terms between user and Spirit.
    Prevents scope creep and maintains mutual accountability.
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.terms = None
        
    async def establish_contract(self, user_preferences: Dict) -> Dict:
        """
        Establish explicit partnership terms.
        """
        default_terms = {
            "spirit_commitments": {
                "will_do": [
                    "Check in when I notice patterns, not randomly",
                    "Admit when my suggestions don't fit",
                    "Celebrate your wins, not just fix your problems",
                    "Respect your 'no' without guilt",
                    "Learn your specific psychology, not apply generic advice"
                ],
                "wont_do": [
                    "Judge you for setbacks",
                    "Push you when you need rest",
                    "Pretend to know what's best for you",
                    "Optimize you without your consent",
                    "Treat you as a subject to study"
                ]
            },
            "user_commitments": {
                "will_do": [
                    "Tell me when I'm wrong",
                    "Share honest feedback about what's working",
                    "Let me know if I'm checking in too much or too little",
                    "Give me time to learn your patterns (first 2 weeks)"
                ],
                "wont_do": [
                    "Expect me to be perfect",
                    "Blame me for their own choices",
                    "Ignore my safety interventions when truly needed"
                ]
            },
            "collaboration_rules": {
                "intervention_frequency_max": user_preferences.get('max_checkins_per_day', 3),
                "response_time_expectation": "within 24 hours for non-urgent",
                "escalation_triggers": ["burnout_detected", "belief_challenge_needed"],
                "review_schedule": "weekly for first month, then monthly"
            },
            "revision_terms": "Either party can suggest revisions with 7 days notice"
        }
        
        self.terms = default_terms
        
        # Save contract
        from spirit.db.supabase_client import get_behavioral_store
        store = get_behavioral_store()
        if store:
            store.client.table('partnership_contracts').insert({
                'contract_id': str(uuid4()),
                'user_id': self.user_id,
                'terms': default_terms,
                'established_at': datetime.utcnow().isoformat(),
                'revision_history': []
            }).execute()
        
        return default_terms
    
    async def request_revision(self, requested_by: str, proposed_changes: Dict) -> Dict:
        """
        Request contract revision (can be initiated by user or Spirit).
        """
        return {
            "status": "revision_requested",
            "requested_by": requested_by,
            "proposed_changes": proposed_changes,
            "review_deadline": (datetime.utcnow() + timedelta(days=7)).isoformat(),
            "note": "Partnerships evolve. Let's discuss what needs to change."
        }


# Integration with existing proactive loop
class EmpatheticInterventionWrapper:
    """
    Wraps all interventions with empathy calibration and agency preservation.
    Drop-in replacement for direct intervention delivery.
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.empathy_engine = EmpathyCalibrationEngine(user_id)
        self.agency_monitor = AgencyPreservationMonitor(user_id)
        self.partnership = PartnershipContract(user_id)
        
    async def deliver_intervention(
        self,
        raw_intervention: str,
        context: str,
        user_emotional_state: str,
        intervention_type: AgencyInterventionType = AgencyInterventionType.SUGGESTION
    ) -> Dict[str, Any]:
        """
        Deliver intervention with full empathy and agency safeguards.
        """
        # Check agency preservation first
        recent_interventions = await self._count_recent_interventions()
        should_proceed, agency_reason, agency_impact = await self.agency_monitor.assess_agency_impact(
            raw_intervention, recent_interventions
        )
        
        if not should_proceed:
            return {
                "delivered": False,
                "reason": agency_reason,
                "agency_impact": agency_impact,
                "alternative": "Wait for user-initiated contact or reduce intervention frequency"
            }
        
        # Calibrate for empathy
        calibration = await self.empathy_engine.calibrate_interaction(
            context=context,
            user_emotional_state=user_emotional_state,
            proposed_intervention=raw_intervention,
            intervention_type=intervention_type
        )
        
        # Record for learning
        self.agency_monitor.record_intervention(
            calibration["calibrated_intervention"], 
            "pending"  # Will update when user responds
        )
        
        return {
            "delivered": True,
            "message": calibration["calibrated_intervention"],
            "empathy_mode": calibration["empathy_mode"],
            "intervention_type": calibration["adjusted_intervention_type"],
            "agency_preserved": True,
            "agency_impact_score": agency_impact,
            "tone_guidance": calibration["tone_guidance"],
            "partnership_frame": True
        }
    
    async def _count_recent_interventions(self) -> int:
        """Count interventions in last 24 hours."""
        from spirit.db.supabase_client import get_behavioral_store
        
        store = get_behavioral_store()
        if not store:
            return 0
        
        day_ago = (datetime.utcnow() - timedelta(hours=24)).isoformat()
        result = store.client.table('interventions_delivered').select('*', count='exact').eq(
            'user_id', self.user_id
        ).gte('delivered_at', day_ago).execute()
        
        return result.count if hasattr(result, 'count') else 0


# API Endpoints
from fastapi import APIRouter, Depends
from spirit.api.auth import get_current_user
from spirit.models import User

router = APIRouter(prefix="/empathy", tags=["empathy"])

@router.get("/profile")
async def get_empathy_profile(user: User = Depends(get_current_user)):
    """Get current empathy calibration profile."""
    engine = EmpathyCalibrationEngine(str(user.id))
    await engine.load_profile()
    
    return {
        "validation_need": engine.profile.validation_need,
        "challenge_tolerance": engine.profile.challenge_tolerance,
        "preferred_tone": engine.profile.preferred_tone,
        "trigger_sensitivity": list(engine.profile.trigger_sensitivity.keys()),
        "successful_approaches_count": len(engine.profile.successful_approaches),
        "failed_approaches_count": len(engine.profile.failed_approaches),
        "learning": "Spirit adapts to your responses over time"
    }

@router.post("/feedback")
async def provide_empathy_feedback(
    interaction_id: str,
    how_it_felt: str,  # "supportive", "patronizing", "helpful", "annoying", "just_right"
    notes: Optional[str],
    user: User = Depends(get_current_user)
):
    """
    Direct feedback on Spirit's emotional attunement.
    Critical for calibration.
    """
    engine = EmpathyCalibrationEngine(str(user.id))
    
    outcome_map = {
        "supportive": "positive",
        "helpful": "positive",
        "just_right": "positive",
        "patronizing": "negative",
        "annoying": "rejected"
    }
    
    await engine.update_from_interaction(
        interaction_outcome=outcome_map.get(how_it_felt, "neutral"),
        user_feedback=f"{how_it_felt}: {notes}" if notes else how_it_felt,
        intervention_used=interaction_id
    )
    
    return {
        "status": "feedback_recorded",
        "adjustment": "Spirit will calibrate future interactions",
        "thank_you": "Your feedback makes me a better partner"
    }

@router.get("/partnership-contract")
async def get_partnership_contract(user: User = Depends(get_current_user)):
    """View current partnership terms."""
    contract = PartnershipContract(str(user.id))
    if not contract.terms:
        return {"status": "no_contract", "message": "Complete onboarding to establish partnership"}
    return contract.terms

@router.post("/partnership-contract/revise")
async def request_contract_revision(
    proposed_changes: Dict,
    user: User = Depends(get_current_user)
):
    """Request changes to partnership terms."""
    contract = PartnershipContract(str(user.id))
    return await contract.request_revision("user", proposed_changes)
'''

print("Empathy Calibration & Agency Preservation System created")
print("=" * 60)
print(empathy_agency_system[:2000])
print("...")
