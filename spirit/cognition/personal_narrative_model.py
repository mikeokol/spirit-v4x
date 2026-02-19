
# Create the Personal Narrative Model file
code = '''"""
Personal Narrative Model (PNM): The story the person unconsciously maintains.

HSM explains strategy.
PNM explains meaning.

Two people in identical situations act differently because they believe they are 
living inside different stories.

Spirit should model the story the person unconsciously maintains.
Not philosophically — structurally.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum, auto
from datetime import datetime, timedelta
import json


class NarrativeAxis(Enum):
    """Core narrative axes that predict behavior."""
    CONTROL_DISCOVERY = "control_vs_discovery"
    BUILDER_PROVER = "builder_vs_prover"
    STABILITY_TRANSFORMATION = "stability_vs_transformation"
    RECOGNITION_AUTONOMY = "recognition_vs_autonomy"
    MASTERY_IMPACT = "mastery_vs_impact"


class NarrativePosition(Enum):
    """Position on a narrative axis."""
    STRONG_LEFT = -2   # e.g., strong Control
    MODERATE_LEFT = -1
    CENTER = 0
    MODERATE_RIGHT = 1
    STRONG_RIGHT = 2   # e.g., strong Discovery


@dataclass
class NarrativeAxisProfile:
    """User's position on a narrative axis."""
    axis: NarrativeAxis
    position: NarrativePosition
    confidence: float  # 0-1
    
    # Evidence
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    contradictory_evidence: List[Dict[str, Any]] = field(default_factory=list)
    
    # Dynamics
    stability: float  # How stable is this position (0-1)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    # Predictions
    predicted_behaviors: List[str] = field(default_factory=list)
    predicted_resistance: List[str] = field(default_factory=list)


@dataclass
class NarrativeForce:
    """A force that shapes narrative - what drives meaning."""
    force_type: str  # 'meaning', 'threat', 'avoidance', 'preservation'
    
    # What drives behavior
    what_feels_meaningful: List[str]
    what_feels_threatening: List[str]
    what_future_avoided: List[str]
    what_past_preserved: List[str]
    
    # Observable markers
    behavioral_markers: List[str]
    linguistic_markers: List[str]
    emotional_markers: List[str]
    
    # Strength
    intensity: float  # 0-1
    stability: float  # 0-1, how constant over time


@dataclass
class IdentityEquilibrium:
    """
    Behavior stabilizes around identity equilibrium.
    Current self-concept and its tolerance for deviation.
    """
    core_identity_statements: List[str]  # "I am someone who..."
    
    # Tolerance
    expansion_tolerance: float  # How much growth allowed (0-1)
    threat_sensitivity: float   # How easily identity threatened (0-1)
    
    # Current equilibrium
    current_state: str
    acceptable_range: Tuple[float, float]  # Range of acceptable self-states
    
    # Disruption
    last_disruption: Optional[datetime]
    recovery_pattern: str  # How returns to equilibrium


class PersonalNarrativeModel:
    """
    Models the user's unconscious narrative structure.
    Detects meaning patterns, identity threats, and narrative forces.
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.axis_profiles: Dict[NarrativeAxis, NarrativeAxisProfile] = {}
        self.narrative_forces: List[NarrativeForce] = []
        self.identity_equilibrium: Optional[IdentityEquilibrium] = None
        
        self._initialize_axes()
    
    def _initialize_axes(self):
        """Initialize all narrative axes as unknown."""
        for axis in NarrativeAxis:
            self.axis_profiles[axis] = NarrativeAxisProfile(
                axis=axis,
                position=NarrativePosition.CENTER,
                confidence=0.0,
                supporting_evidence=[],
                contradictory_evidence=[],
                stability=0.5,
                predicted_behaviors=[],
                predicted_resistance=[]
            )
    
    async def update_from_observation(self, observation: Dict[str, Any]):
        """Update narrative model based on behavioral observation."""
        
        # Update each axis based on observation
        for axis in NarrativeAxis:
            evidence = self._extract_axis_evidence(axis, observation)
            if evidence:
                await self._update_axis(axis, evidence)
        
        # Update narrative forces
        await self._detect_narrative_forces(observation)
        
        # Update identity equilibrium
        await self._update_identity_equilibrium(observation)
    
    def _extract_axis_evidence(
        self,
        axis: NarrativeAxis,
        observation: Dict
    ) -> Optional[Dict[str, Any]]:
        """Extract evidence for a specific axis from observation."""
        
        behavior = observation.get('behavior', {})
        ema = observation.get('ema_response', {})
        text = json.dumps(observation).lower()
        
        # AXIS 1: Control vs Discovery
        if axis == NarrativeAxis.CONTROL_DISCOVERY:
            control_markers = [
                'plan', 'schedule', 'organize', 'prepare', 'structure',
                'routine', 'system', 'method', 'control', 'manage'
            ]
            discovery_markers = [
                'explore', 'try', 'experiment', 'discover', 'adventure',
                'spontaneous', 'flexible', 'open', 'curious', 'novel'
            ]
            
            control_score = sum(1 for m in control_markers if m in text)
            discovery_score = sum(1 for m in discovery_markers if m in text)
            
            if control_score > discovery_score + 2:
                return {
                    'direction': 'control',
                    'strength': min(1.0, control_score * 0.2),
                    'markers_found': [m for m in control_markers if m in text][:3]
                }
            elif discovery_score > control_score + 2:
                return {
                    'direction': 'discovery',
                    'strength': min(1.0, discovery_score * 0.2),
                    'markers_found': [m for m in discovery_markers if m in text][:3]
                }
        
        # AXIS 2: Builder vs Prover
        elif axis == NarrativeAxis.BUILDER_PROVER:
            builder_markers = [
                'create', 'build', 'make', 'develop', 'grow', 'craft',
                'produce', 'establish', 'foundation', 'system'
            ]
            prover_markers = [
                'prove', 'demonstrate', 'show', 'achieve', 'succeed',
                'accomplish', 'validate', 'worth', 'capable', 'good enough'
            ]
            
            builder_score = sum(1 for m in builder_markers if m in text)
            prover_score = sum(1 for m in prover_markers if m in text)
            
            if builder_score > prover_score + 2:
                return {
                    'direction': 'builder',
                    'strength': min(1.0, builder_score * 0.2),
                    'markers_found': [m for m in builder_markers if m in text][:3]
                }
            elif prover_score > builder_score + 2:
                return {
                    'direction': 'prover',
                    'strength': min(1.0, prover_score * 0.2),
                    'markers_found': [m for m in prover_markers if m in text][:3]
                }
        
        # AXIS 3: Stability vs Transformation
        elif axis == NarrativeAxis.STABILITY_TRANSFORMATION:
            stability_markers = [
                'maintain', 'preserve', 'keep', 'stable', 'consistent',
                'same', 'continue', 'protect', 'security', 'safe'
            ]
            transformation_markers = [
                'change', 'transform', 'become', 'new', 'different',
                'evolve', 'grow', 'breakthrough', 'reinvent', 'shift'
            ]
            
            stability_score = sum(1 for m in stability_markers if m in text)
            transformation_score = sum(1 for m in transformation_markers if m in text)
            
            if stability_score > transformation_score + 2:
                return {
                    'direction': 'stability',
                    'strength': min(1.0, stability_score * 0.2),
                    'markers_found': [m for m in stability_markers if m in text][:3]
                }
            elif transformation_score > stability_score + 2:
                return {
                    'direction': 'transformation',
                    'strength': min(1.0, transformation_score * 0.2),
                    'markers_found': [m for m in transformation_markers if m in text][:3]
                }
        
        # AXIS 4: Recognition vs Autonomy
        elif axis == NarrativeAxis.RECOGNITION_AUTONOMY:
            recognition_markers = [
                'recognition', 'appreciate', 'acknowledge', 'praise', 'seen',
                'validate', 'credit', 'noticed', 'respected', 'admired'
            ]
            autonomy_markers = [
                'freedom', 'independent', 'own', 'my way', 'choice',
                'control', 'decide', 'self-directed', 'autonomous', 'liberty'
            ]
            
            recognition_score = sum(1 for m in recognition_markers if m in text)
            autonomy_score = sum(1 for m in autonomy_markers if m in text)
            
            if recognition_score > autonomy_score + 2:
                return {
                    'direction': 'recognition',
                    'strength': min(1.0, recognition_score * 0.2),
                    'markers_found': [m for m in recognition_markers if m in text][:3]
                }
            elif autonomy_score > recognition_score + 2:
                return {
                    'direction': 'autonomy',
                    'strength': min(1.0, autonomy_score * 0.2),
                    'markers_found': [m for m in autonomy_markers if m in text][:3]
                }
        
        # AXIS 5: Mastery vs Impact
        elif axis == NarrativeAxis.MASTERY_IMPACT:
            mastery_markers = [
                'master', 'expert', 'deep', 'refine', 'perfect',
                'skill', 'craft', 'excellence', 'depth', 'quality'
            ]
            impact_markers = [
                'impact', 'change', 'difference', 'help', 'affect',
                'outcome', 'result', 'deploy', 'ship', 'deliver'
            ]
            
            mastery_score = sum(1 for m in mastery_markers if m in text)
            impact_score = sum(1 for m in impact_markers if m in text)
            
            if mastery_score > impact_score + 2:
                return {
                    'direction': 'mastery',
                    'strength': min(1.0, mastery_score * 0.2),
                    'markers_found': [m for m in mastery_markers if m in text][:3]
                }
            elif impact_score > mastery_score + 2:
                return {
                    'direction': 'impact',
                    'strength': min(1.0, impact_score * 0.2),
                    'markers_found': [m for m in impact_markers if m in text][:3]
                }
        
        return None
    
    async def _update_axis(self, axis: NarrativeAxis, evidence: Dict):
        """Update axis position based on new evidence."""
        profile = self.axis_profiles[axis]
        
        direction = evidence['direction']
        strength = evidence['strength']
        
        # Map direction to position
        direction_to_position = {
            'control': NarrativePosition.STRONG_LEFT,
            'discovery': NarrativePosition.STRONG_RIGHT,
            'builder': NarrativePosition.STRONG_LEFT,
            'prover': NarrativePosition.STRONG_RIGHT,
            'stability': NarrativePosition.STRONG_LEFT,
            'transformation': NarrativePosition.STRONG_RIGHT,
            'recognition': NarrativePosition.STRONG_LEFT,
            'autonomy': NarrativePosition.STRONG_RIGHT,
            'mastery': NarrativePosition.STRONG_LEFT,
            'impact': NarrativePosition.STRONG_RIGHT
        }
        
        new_position = direction_to_position.get(direction, NarrativePosition.CENTER)
        
        # Update with exponential moving average
        if profile.confidence < 0.3:
            # First strong evidence - set position
            profile.position = new_position
            profile.confidence = strength * 0.5
        else:
            # Update existing position
            current_val = profile.position.value
            new_val = new_position.value
            
            # Weight by confidence
            alpha = 0.3  # Learning rate
            updated_val = current_val * (1 - alpha) + new_val * alpha * strength
            
            # Map back to position
            if updated_val < -1.5:
                profile.position = NarrativePosition.STRONG_LEFT
            elif updated_val < -0.5:
                profile.position = NarrativePosition.MODERATE_LEFT
            elif updated_val > 1.5:
                profile.position = NarrativePosition.STRONG_RIGHT
            elif updated_val > 0.5:
                profile.position = NarrativePosition.MODERATE_RIGHT
            else:
                profile.position = NarrativePosition.CENTER
            
            # Update confidence
            profile.confidence = min(0.95, profile.confidence + strength * 0.1)
        
        # Add to evidence
        profile.supporting_evidence.append({
            'timestamp': datetime.utcnow().isoformat(),
            'evidence': evidence,
            'position_after': profile.position.name
        })
        
        # Update predictions
        profile.predicted_behaviors = self._generate_axis_predictions(axis, profile.position)
        profile.predicted_resistance = self._generate_axis_resistance(axis, profile.position)
        
        profile.last_updated = datetime.utcnow()
    
    def _generate_axis_predictions(
        self,
        axis: NarrativeAxis,
        position: NarrativePosition
    ) -> List[str]:
        """Generate predicted behaviors based on axis position."""
        
        predictions = {
            (NarrativeAxis.CONTROL_DISCOVERY, NarrativePosition.STRONG_LEFT): [
                "Will create detailed plans before acting",
                "Prefers structured environments",
                "Anxiety with uncertainty or spontaneity",
                "Will gather extensive information before decisions"
            ],
            (NarrativeAxis.CONTROL_DISCOVERY, NarrativePosition.STRONG_RIGHT): [
                "Will act spontaneously without full preparation",
                "Prefers flexible, open-ended approaches",
                "Boredom with routine or repetition",
                "Will experiment and iterate rather than plan"
            ],
            (NarrativeAxis.BUILDER_PROVER, NarrativePosition.STRONG_LEFT): [
                "Motivated by creating systems and structures",
                "Long-term focus on building foundations",
                "Satisfaction from process, not just outcome",
                "May neglect to demonstrate or share work"
            ],
            (NarrativeAxis.BUILDER_PROVER, NarrativePosition.STRONG_RIGHT): [
                "Motivated by achievement and validation",
                "Focus on outcomes and accomplishments",
                "Need for external recognition",
                "Risk of burnout from proving cycle"
            ],
            (NarrativeAxis.STABILITY_TRANSFORMATION, NarrativePosition.STRONG_LEFT): [
                "Will maintain current patterns despite suboptimality",
                "Resistance to major life changes",
                "Comfort in familiarity and routine",
                "Risk of stagnation"
            ],
            (NarrativeAxis.STABILITY_TRANSFORMATION, NarrativePosition.STRONG_RIGHT): [
                "Constant drive for change and reinvention",
                "May abandon stable situations prematurely",
                "Identity tied to transformation narrative",
                "Risk of instability from too much change"
            ],
            (NarrativeAxis.RECOGNITION_AUTONOMY, NarrativePosition.STRONG_LEFT): [
                "Will seek validation and feedback frequently",
                "Motivated by praise and acknowledgment",
                "May compromise autonomy for recognition",
                "Performance varies with visibility"
            ],
            (NarrativeAxis.RECOGNITION_AUTONOMY, NarrativePosition.STRONG_RIGHT): [
                "Will resist external direction or control",
                "Motivated by independence and self-direction",
                "May reject help to preserve autonomy",
                "Performance best when self-directed"
            ],
            (NarrativeAxis.MASTERY_IMPACT, NarrativePosition.STRONG_LEFT): [
                "Will refine and perfect before sharing",
                "Deep focus on skill development",
                "May delay deployment indefinitely",
                "Satisfaction from expertise, not application"
            ],
            (NarrativeAxis.MASTERY_IMPACT, NarrativePosition.STRONG_RIGHT): [
                "Will ship quickly and iterate",
                "Focus on outcomes and effects",
                "May sacrifice depth for speed",
                "Satisfaction from making difference"
            ]
        }
        
        return predictions.get((axis, position), ["Behavior depends on context"])
    
    def _generate_axis_resistance(
        self,
        axis: NarrativeAxis,
        position: NarrativePosition
    ) -> List[str]:
        """Generate predicted resistance based on axis position."""
        
        resistance = {
            (NarrativeAxis.CONTROL_DISCOVERY, NarrativePosition.STRONG_LEFT): [
                "Will resist spontaneous or unplanned changes",
                "Anxiety with ambiguous situations",
                "Rejection of 'just try it' approaches"
            ],
            (NarrativeAxis.CONTROL_DISCOVERY, NarrativePosition.STRONG_RIGHT): [
                "Will resist rigid structures or schedules",
                "Boredom with routine tasks",
                "Rejection of detailed planning requirements"
            ],
            (NarrativeAxis.BUILDER_PROVER, NarrativePosition.STRONG_LEFT): [
                "Will resist pressure to demonstrate prematurely",
                "Discomfort with performance evaluation",
                "Rejection of outcome-only focus"
            ],
            (NarrativeAxis.BUILDER_PROVER, NarrativePosition.STRONG_RIGHT): [
                "Will resist invisible or unacknowledged work",
                "Frustration without validation",
                "Rejection of process-for-process-sake"
            ],
            (NarrativeAxis.STABILITY_TRANSFORMATION, NarrativePosition.STRONG_LEFT): [
                "Will resist radical change initiatives",
                "Fear of identity loss with change",
                "Rejection of 'reinvent yourself' framing"
            ],
            (NarrativeAxis.STABILITY_TRANSFORMATION, NarrativePosition.STRONG_RIGHT): [
                "Will resist maintenance or preservation",
                "Boredom with stability",
                "Rejection of 'stay the course' advice"
            ],
            (NarrativeAxis.RECOGNITION_AUTONOMY, NarrativePosition.STRONG_LEFT): [
                "Will resist invisible or private work",
                "Demotivation without feedback",
                "Rejection of 'do it for yourself' framing"
            ],
            (NarrativeAxis.RECOGNITION_AUTONOMY, NarrativePosition.STRONG_RIGHT): [
                "Will resist external accountability",
                "Rejection of surveillance or monitoring",
                "Demotivation with excessive direction"
            ],
            (NarrativeAxis.MASTERY_IMPACT, NarrativePosition.STRONG_LEFT): [
                "Will resist premature shipping",
                "Discomfort with 'good enough'",
                "Rejection of speed over quality"
            ],
            (NarrativeAxis.MASTERY_IMPACT, NarrativePosition.STRONG_RIGHT): [
                "Will resist endless refinement",
                "Impatience with perfectionism",
                "Rejection of 'not ready yet' delays"
            ]
        }
        
        return resistance.get((axis, position), ["Resistance depends on context"])
    
    async def _detect_narrative_forces(self, observation: Dict):
        """Detect narrative forces from observation."""
        
        text = json.dumps(observation).lower()
        
        # Detect meaning forces
        meaning_markers = [
            'purpose', 'meaning', 'matter', 'important', 'significant',
            'why', 'reason', 'point', 'value', 'worth'
        ]
        meaning_score = sum(1 for m in meaning_markers if m in text)
        
        if meaning_score >= 2:
            force = NarrativeForce(
                force_type='meaning',
                what_feels_meaningful=[m for m in meaning_markers if m in text][:3],
                what_feels_threatening=['meaninglessness', 'pointlessness', 'emptiness'],
                what_future_avoided=['meaningless existence', 'wasted life'],
                what_past_preserved=['purposeful moments', 'significant achievements'],
                behavioral_markers=['engagement_with_meaningful_tasks', 'avoidance_of_empty_work'],
                linguistic_markers=['why', 'purpose', 'matter', 'important'],
                emotional_markers=['fulfillment', 'emptiness', 'inspiration'],
                intensity=min(1.0, meaning_score * 0.3),
                stability=0.7
            )
            self.narrative_forces.append(force)
        
        # Detect threat forces
        threat_markers = [
            'afraid', 'scared', 'fear', 'terrified', 'anxious',
            'worried', 'concern', 'dread', 'panic', 'terror'
        ]
        threat_score = sum(1 for m in threat_markers if m in text)
        
        if threat_score >= 2:
            force = NarrativeForce(
                force_type='threat',
                what_feels_meaningful=['safety', 'security', 'protection'],
                what_feels_threatening=[m for m in threat_markers if m in text][:3],
                what_future_avoided=['feared outcomes', 'catastrophes'],
                what_past_preserved=['safe patterns', 'protective habits'],
                behavioral_markers=['avoidance', 'protection_seeking', 'risk_aversion'],
                linguistic_markers=['afraid', 'fear', 'worried', 'scared'],
                emotional_markers=['anxiety', 'fear', 'relief'],
                intensity=min(1.0, threat_score * 0.3),
                stability=0.6
            )
            self.narrative_forces.append(force)
        
        # Keep only recent forces
        self.narrative_forces = self.narrative_forces[-10:]
    
    async def _update_identity_equilibrium(self, observation: Dict):
        """Update identity equilibrium based on observation."""
        
        # Extract identity statements
        text = json.dumps(observation).lower()
        
        identity_patterns = [
            r'i am', r'i\\'m', r'myself', r'who i am', r'identity',
            r'become', r'becoming', r'used to be', r'always been'
        ]
        
        identity_statements = []
        for pattern in identity_patterns:
            if pattern in text:
                # Extract surrounding context
                idx = text.find(pattern)
                start = max(0, idx - 30)
                end = min(len(text), idx + 50)
                identity_statements.append(text[start:end])
        
        if identity_statements:
            if self.identity_equilibrium is None:
                self.identity_equilibrium = IdentityEquilibrium(
                    core_identity_statements=identity_statements[:3],
                    expansion_tolerance=0.5,
                    threat_sensitivity=0.5,
                    current_state='stable',
                    acceptable_range=(0.3, 0.7),
                    last_disruption=None,
                    recovery_pattern='gradual'
                )
            else:
                # Update with new statements
                self.identity_equilibrium.core_identity_statements.extend(identity_statements[:2])
                self.identity_equilibrium.core_identity_statements = \
                    self.identity_equilibrium.core_identity_statements[-5:]
    
    def get_dominant_axes(self, min_confidence: float = 0.5) -> List[NarrativeAxisProfile]:
        """Get axes where position is confidently known."""
        return [
            profile for profile in self.axis_profiles.values()
            if profile.confidence >= min_confidence
        ]
    
    def predict_behavioral_response(
        self,
        intervention_type: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict how user will respond to intervention based on narrative.
        """
        
        # Check alignment with each axis
        alignments = []
        for axis, profile in self.axis_profiles.items():
            if profile.confidence < 0.4:
                continue
            
            alignment = self._check_axis_alignment(axis, profile.position, intervention_type)
            alignments.append({
                'axis': axis.value,
                'alignment': alignment,
                'confidence': profile.confidence
            })
        
        # Calculate overall fit
        if not alignments:
            return {
                'predicted_response': 'uncertain',
                'confidence': 0.3,
                'narrative_fit': 0.5,
                'risks': ['insufficient_narrative_data']
            }
        
        # Weight by confidence
        weighted_alignment = sum(
            a['alignment'] * a['confidence'] for a in alignments
        ) / sum(a['confidence'] for a in alignments)
        
        if weighted_alignment > 0.7:
            predicted = 'receptive'
        elif weighted_alignment > 0.4:
            predicted = 'mixed'
        else:
            predicted = 'resistant'
        
        # Identify risks
        risks = []
        for a in alignments:
            if a['alignment'] < 0.3 and a['confidence'] > 0.6:
                risks.append(f"conflict_with_{a['axis']}")
        
        return {
            'predicted_response': predicted,
            'confidence': sum(a['confidence'] for a in alignments) / len(alignments),
            'narrative_fit': weighted_alignment,
            'axis_alignments': alignments,
            'risks': risks,
            'suggested_modifications': self._suggest_narrative_modifications(
                intervention_type, alignments
            )
        }
    
    def _check_axis_alignment(
        self,
        axis: NarrativeAxis,
        position: NarrativePosition,
        intervention_type: str
    ) -> float:
        """Check how well intervention aligns with axis position."""
        
        # Define alignment for common intervention types
        alignments = {
            (NarrativeAxis.CONTROL_DISCOVERY, 'structured_plan'): {
                NarrativePosition.STRONG_LEFT: 0.9,
                NarrativePosition.STRONG_RIGHT: 0.2
            },
            (NarrativeAxis.CONTROL_DISCOVERY, 'flexible_approach'): {
                NarrativePosition.STRONG_LEFT: 0.2,
                NarrativePosition.STRONG_RIGHT: 0.9
            },
            (NarrativeAxis.BUILDER_PROVER, 'process_focus'): {
                NarrativePosition.STRONG_LEFT: 0.9,
                NarrativePosition.STRONG_RIGHT: 0.3
            },
            (NarrativeAxis.BUILDER_PROVER, 'outcome_focus'): {
                NarrativePosition.STRONG_LEFT: 0.3,
                NarrativePosition.STRONG_RIGHT: 0.9
            },
            (NarrativeAxis.STABILITY_TRANSFORMATION, 'gradual_change'): {
                NarrativePosition.STRONG_LEFT: 0.8,
                NarrativePosition.STRONG_RIGHT: 0.4
            },
            (NarrativeAxis.STABILITY_TRANSFORMATION, 'radical_change'): {
                NarrativePosition.STRONG_LEFT: 0.2,
                NarrativePosition.STRONG_RIGHT: 0.9
            },
            (NarrativeAxis.RECOGNITION_AUTONOMY, 'external_accountability'): {
                NarrativePosition.STRONG_LEFT: 0.9,
                NarrativePosition.STRONG_RIGHT: 0.2
            },
            (NarrativeAxis.RECOGNITION_AUTONOMY, 'self_directed'): {
                NarrativePosition.STRONG_LEFT: 0.2,
                NarrativePosition.STRONG_RIGHT: 0.9
            },
            (NarrativeAxis.MASTERY_IMPACT, 'perfect_before_ship'): {
                NarrativePosition.STRONG_LEFT: 0.9,
                NarrativePosition.STRONG_RIGHT: 0.2
            },
            (NarrativeAxis.MASTERY_IMPACT, 'ship_and_iterate'): {
                NarrativePosition.STRONG_LEFT: 0.2,
                NarrativePosition.STRONG_RIGHT: 0.9
            }
        }
        
        key = (axis, intervention_type)
        if key in alignments:
            return alignments[key].get(position, 0.5)
        
        return 0.5  # Neutral if unknown
    
    def _suggest_narrative_modifications(
        self,
        intervention_type: str,
        alignments: List[Dict]
    ) -> List[str]:
        """Suggest modifications to improve narrative fit."""
        
        suggestions = []
        
        for a in alignments:
            if a['alignment'] < 0.4:
                suggestions.append(
                    f"Consider alternative approach that better aligns with {a['axis']} preference"
                )
        
        return suggestions
    
    def get_identity_threat_assessment(
        self,
        proposed_change: str
    ) -> Dict[str, Any]:
        """
        Assess if proposed change threatens identity equilibrium.
        """
        
        if not self.identity_equilibrium:
            return {
                'threat_level': 'unknown',
                'confidence': 0.3,
                'recommendation': 'insufficient_identity_data'
            }
        
        # Check against core identity statements
        threat_indicators = 0
        for statement in self.identity_equilibrium.core_identity_statements:
            # Check for contradiction
            if self._contradicts_identity(proposed_change, statement):
                threat_indicators += 1
        
        threat_rate = threat_indicators / max(1, len(self.identity_equilibrium.core_identity_statements))
        
        # Factor in threat sensitivity
        adjusted_threat = threat_rate * (0.5 + self.identity_equilibrium.threat_sensitivity)
        
        if adjusted_threat > 0.7:
            level = 'high'
        elif adjusted_threat > 0.4:
            level = 'moderate'
        else:
            level = 'low'
        
        return {
            'threat_level': level,
            'threat_score': adjusted_threat,
            'confidence': 0.6 if self.identity_equilibrium.core_identity_statements else 0.3,
            'threatened_statements': [
                s for s in self.identity_equilibrium.core_identity_statements
                if self._contradicts_identity(proposed_change, s)
            ],
            'recommendation': self._get_threat_mitigation(level)
        }
    
    def _contradicts_identity(self, change: str, identity: str) -> bool:
        """Check if change contradicts identity statement."""
        # Simple keyword matching - would be more sophisticated
        change_words = set(change.lower().split())
        identity_words = set(identity.lower().split())
        
        # Check for antonyms or contradictions
        contradictions = [
            ('change', 'always'), ('new', 'never'), ('different', 'same'),
            ('transform', 'maintain'), ('break', 'preserve')
        ]
        
        for c1, c2 in contradictions:
            if c1 in change_words and c2 in identity_words:
                return True
        
        return False
    
    def _get_threat_mitigation(self, threat_level: str) -> str:
        """Get mitigation strategy for identity threat."""
        
        strategies = {
            'high': 'Frame change as becoming more fully self, not different self. Build narrative bridge.',
            'moderate': 'Connect to existing identity. Show continuity with past self.',
            'low': 'Proceed with standard approach. Monitor for emerging resistance.'
        }
        
        return strategies.get(threat_level, 'assess_and_monitor')
    
    def generate_narrative_bridge(
        self,
        current_state: str,
        desired_state: str
    ) -> Dict[str, Any]:
        """
        Generate narrative bridge from current to desired state.
        Essential for PNM-level interventions.
        """
        
        # Get dominant axes
        dominant = self.get_dominant_axes(min_confidence=0.4)
        
        bridge_elements = []
        
        for profile in dominant:
            element = self._generate_axis_bridge_element(
                profile.axis, profile.position, current_state, desired_state
            )
            if element:
                bridge_elements.append(element)
        
        return {
            'current_state': current_state,
            'desired_state': desired_state,
            'bridge_elements': bridge_elements,
            'narrative_continuity': self._assess_continuity(bridge_elements),
            'suggested_framing': self._generate_bridge_framing(bridge_elements)
        }
    
    def _generate_axis_bridge_element(
        self,
        axis: NarrativeAxis,
        position: NarrativePosition,
        current: str,
        desired: str
    ) -> Optional[Dict]:
        """Generate bridge element for specific axis."""
        
        bridges = {
            (NarrativeAxis.CONTROL_DISCOVERY, NarrativePosition.STRONG_LEFT): {
                'element': 'structured_exploration',
                'framing': f"Explore {desired} through systematic, planned approach"
            },
            (NarrativeAxis.CONTROL_DISCOVERY, NarrativePosition.STRONG_RIGHT): {
                'element': 'disciplined_adventure',
                'framing': f"Bring your exploratory spirit to {desired} with just enough structure"
            },
            (NarrativeAxis.BUILDER_PROVER, NarrativePosition.STRONG_LEFT): {
                'element': 'craft_development',
                'framing': f"Build your capacity for {desired} as a craft"
            },
            (NarrativeAxis.BUILDER_PROVER, NarrativePosition.STRONG_RIGHT): {
                'element': 'achievement_path',
                'framing': f"Demonstrate your capability through {desired}"
            },
            (NarrativeAxis.STABILITY_TRANSFORMATION, NarrativePosition.STRONG_LEFT): {
                'element': 'preservation_through_growth',
                'framing': f"Preserve what matters most while evolving toward {desired}"
            },
            (NarrativeAxis.STABILITY_TRANSFORMATION, NarrativePosition.STRONG_RIGHT): {
                'element': 'transformation_continuity',
                'framing': f"This transformation is the next chapter in your ongoing evolution"
            }
        }
        
        key = (axis, position)
        if key in bridges:
            return {
                'axis': axis.value,
                **bridges[key]
            }
        
        return None
    
    def _assess_continuity(self, elements: List[Dict]) -> float:
        """Assess how much continuity the bridge provides."""
        if not elements:
            return 0.5
        return min(1.0, 0.4 + len(elements) * 0.15)
    
    def _generate_bridge_framing(self, elements: List[Dict]) -> str:
        """Generate unified framing from bridge elements."""
        if not elements:
            return "Approach change as natural evolution"
        
        framings = [e['framing'] for e in elements[:2]]
        return "; ".join(framings)


# Registry for user models
_pnm_registry: Dict[str, PersonalNarrativeModel] = {}

def get_personal_narrative_model(user_id: str) -> PersonalNarrativeModel:
    """Get or create PNM for user."""
    if user_id not in _pnm_registry:
        _pnm_registry[user_id] = PersonalNarrativeModel(user_id)
    return _pnm_registry[user_id]
'''

with open('/mnt/kimi/output/personal_narrative_model.py', 'w') as f:
    f.write(code)

print("Created: personal_narrative_model.py")
print(f"Size: {len(code)} bytes")
