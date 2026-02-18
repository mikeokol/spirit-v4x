"""
Human Operating Model (HOM): Mechanistic regularities for behavior generation.
Not personality theory. Not pop-psych. Observable, testable cognitive mechanisms.

Each mechanism is a candidate explanation for behavior—never a judgment.
Spirit uses these as priors for hypothesis generation, confound detection,
experiment design, and non-judgmental framing.

Sources: Neuroscience, behavioral economics, cognitive science.
Format: Observable signature → Mechanism → Testable prediction.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum, auto
import json


class Subsystem(Enum):
    """The six major subsystems governing behavior."""
    ENERGY_NEUROBIOLOGY = "energy_neurobiology"  # Resource-bounded cognition
    ATTENTION_INFORMATION = "attention_information"  # Processing constraints
    MOTIVATION_REWARD = "motivation_reward"  # Expected relief pursuit
    AVOIDANCE_THREAT = "avoidance_threat"  # Protective regulation
    LEARNING_BELIEF = "learning_belief"  # Belief updating mechanisms
    SOCIAL_IDENTITY = "social_identity"  # Socially regulated behavior


class MechanismCategory(Enum):
    """Confidence levels for mechanism attribution."""
    HIGH_CONFIDENCE = auto()   # Strong empirical support, observable signature
    MODERATE_CONFIDENCE = auto()  # Good support, context-dependent
    EMERGING = auto()          # Preliminary evidence, use with caution


@dataclass
class CognitiveMechanism:
    """
    A single mechanistic regularity from the Human Operating Model.
    
    Each mechanism provides:
    - Observable signatures: What to look for in behavioral data
    - Testable predictions: What should happen if this mechanism is active
    - Intervention levers: How to influence the mechanism
    - Framing language: Non-judgmental way to communicate to user
    """
    mechanism_id: str  # e.g., "ENB-01" for Energy & Neurobiology #1
    subsystem: Subsystem
    
    # Core description
    name: str
    description: str
    empirical_basis: str  # Key studies or findings
    
    # Confidence in this mechanism
    confidence: MechanismCategory
    
    # Observable signatures in behavioral data
    observable_signatures: List[Dict[str, Any]]
    # Each signature: {
    #   "data_type": "app_usage" | "keyboard_dynamics" | "ema_response" | etc.
    #   "pattern": description of what to look for
    #   "confidence_boost": how much this signature increases mechanism confidence
    # }
    
    # Testable predictions
    predictions: List[str]
    # "If this mechanism is active, then X should occur under Y conditions"
    
    # Intervention levers
    intervention_levers: List[Dict[str, Any]]
    # Each lever: {
    #   "action": what to change
    #   "expected_effect": predicted outcome
    #   "evidence_strength": confidence in this intervention
    # }
    
    # Non-judgmental framing for user communication
    user_framing_templates: List[str]
    # Templates with placeholders like {context}, {behavior}, {outcome}
    
    # Related mechanisms (often co-occur or compete)
    related_mechanisms: List[str]  # mechanism_ids
    
    # Confounds that mimic this mechanism
    mimicking_confounds: List[str]  # e.g., "sleep_debt", "caffeine_withdrawal"


class HumanOperatingModel:
    """
    Structured knowledge graph of human cognitive mechanisms.
    Powers Spirit's mechanistic hypothesis generation.
    """
    
    def __init__(self):
        self.mechanisms: Dict[str, CognitiveMechanism] = {}
        self._initialize_mechanisms()
    
    def _initialize_mechanisms(self):
        """Populate the 40 mechanisms from the HOM."""
        
        # ============================================================
        # I. ENERGY & NEUROBIOLOGY (Resource-bounded cognition)
        # ============================================================
        
        self.mechanisms["ENB-01"] = CognitiveMechanism(
            mechanism_id="ENB-01",
            subsystem=Subsystem.ENERGY_NEUROBIOLOGY,
            name="Front-loaded cognitive energy",
            description="Executive control degrades before motivation does. Morning capacity ≠ evening capacity.",
            empirical_basis="Baumeister ego depletion studies; Kahneman cognitive load research; circadian rhythm studies",
            confidence=MechanismCategory.HIGH_CONFIDENCE,
            observable_signatures=[
                {
                    "data_type": "productivity_metrics",
                    "pattern": "High complexity task completion rate drops 30%+ after 2 PM despite continued effort",
                    "confidence_boost": 0.8
                },
                {
                    "data_type": "app_usage",
                    "pattern": "Shift from productivity apps to communication/entertainment after 3 PM",
                    "confidence_boost": 0.6
                },
                {
                    "data_type": "keyboard_dynamics",
                    "pattern": "Typing speed stable but error rate increases; backspace frequency rises",
                    "confidence_boost": 0.7
                },
                {
                    "data_type": "ema_response",
                    "pattern": "Self-reported 'mental clarity' declines while 'motivation' stays stable",
                    "confidence_boost": 0.9
                }
            ],
            predictions=[
                "Complex tasks started after 3 PM will have 40% longer completion time",
                "Interventions requiring executive control (planning, inhibition) fail more often in afternoon",
                "Simple, habitual tasks maintain performance across day"
            ],
            intervention_levers=[
                {
                    "action": "Schedule complex decisions before 11 AM",
                    "expected_effect": "40% improvement in decision quality",
                    "evidence_strength": 0.8
                },
                {
                    "action": "Add structured breaks every 90 min after 2 PM",
                    "expected_effect": "Restore 20-30% of morning performance",
                    "evidence_strength": 0.7
                },
                {
                    "action": "Reduce task complexity after 3 PM (break into subtasks)",
                    "expected_effect": "Maintain completion rate despite energy decline",
                    "evidence_strength": 0.75
                }
            ],
            user_framing_templates=[
                "Your executive control is front-loaded—complex work happens best before {peak_time}.",
                "This isn't motivation dropping; it's cognitive capacity shifting. Let's match task type to energy type.",
                "Your brain prioritizes differently now than at 9 AM. What worked then needs adjustment."
            ],
            related_mechanisms=["ENB-03", "ENB-07", "ATT-11"],
            mimicking_confounds=["sleep_debt", "blood_sugar_crash", "caffeine_tolerance"]
        )
        
        self.mechanisms["ENB-02"] = CognitiveMechanism(
            mechanism_id="ENB-02",
            subsystem=Subsystem.ENERGY_NEUROBIOLOGY,
            name="Sleep debt impairs inhibition first",
            description="Sleep deprivation selectively damages executive control and planning, not knowledge or desire.",
            empirical_basis="Walker sleep research; Harrison & Horne executive function studies",
            confidence=MechanismCategory.HIGH_CONFIDENCE,
            observable_signatures=[
                {
                    "data_type": "app_usage",
                    "pattern": "Increased social media/app switching after 10 PM usage nights",
                    "confidence_boost": 0.7
                },
                {
                    "data_type": "task_completion",
                    "pattern": "Tasks requiring planning (multi-step) fail; single-step tasks succeed",
                    "confidence_boost": 0.8
                },
                {
                    "data_type": "ema_response",
                    "pattern": "Reports 'know what to do but can't start' or 'going in circles'",
                    "confidence_boost": 0.9
                },
                {
                    "data_type": "screen_time",
                    "pattern": "Late night screen time (>11 PM) followed by morning performance drop",
                    "confidence_boost": 0.75
                }
            ],
            predictions=[
                "User will report knowing the solution but being unable to execute",
                "Simple inhibition tasks (don't check phone) fail more than complex reasoning",
                "Morning performance variance correlates with previous night's sleep timing, not duration"
            ],
            intervention_levers=[
                {
                    "action": "Reduce planning load (provide specific next action, not goal)",
                    "expected_effect": "Bypass impaired planning system",
                    "evidence_strength": 0.8
                },
                {
                    "action": "Add external inhibition (app blockers, environment change)",
                    "expected_effect": "Compensate for weakened internal inhibition",
                    "evidence_strength": 0.75
                },
                {
                    "action": "Delay complex decisions 24 hours if sleep debt detected",
                    "expected_effect": "Avoid impulsive choices made under impaired control",
                    "evidence_strength": 0.7
                }
            ],
            user_framing_templates=[
                "Your planning system is running on backup power. Let's reduce the planning load.",
                "This isn't willpower—it's sleep debt affecting how your brain organizes action.",
                "Your knowledge is intact; your ability to sequence steps is temporarily reduced."
            ],
            related_mechanisms=["ENB-01", "ENB-03", "AVD-25"],
            mimicking_confounds=["alcohol_hangover", "medication_side_effects", "depression"]
        )
        
        self.mechanisms["ENB-03"] = CognitiveMechanism(
            mechanism_id="ENB-03",
            subsystem=Subsystem.ENERGY_NEUROBIOLOGY,
            name="Decision fatigue increases default choices",
            description="Repeated decisions deplete executive resources, leading to automatic, safe choices—not laziness.",
            empirical_basis="Baumeister & Tierney willpower research; Iyengar choice overload studies",
            confidence=MechanismCategory.HIGH_CONFIDENCE,
            observable_signatures=[
                {
                    "data_type": "app_usage",
                    "pattern": "After many decisions (meetings, messages), shift to passive consumption",
                    "confidence_boost": 0.7
                },
                {
                    "data_type": "task_selection",
                    "pattern": "Choose familiar, low-stakes tasks over important but novel ones",
                    "confidence_boost": 0.8
                },
                {
                    "data_type": "response_time",
                    "pattern": "Slower responses to open-ended prompts; faster to yes/no",
                    "confidence_boost": 0.6
                }
            ],
            predictions=[
                "User will prefer default options after 3+ decisions in preceding hour",
                "Novel task avoidance increases linearly with prior decision count",
                "Simplified choices (binary) maintain engagement when complex choices fail"
            ],
            intervention_levers=[
                {
                    "action": "Pre-commit to decisions before fatigue accumulates",
                    "expected_effect": "Lock in good choices while capacity is high",
                    "evidence_strength": 0.8
                },
                {
                    "action": "Convert open decisions to binary choices",
                    "expected_effect": "Reduce cognitive load, maintain engagement",
                    "evidence_strength": 0.75
                },
                {
                    "action": "Schedule important decisions before routine ones",
                    "expected_effect": "Preserve capacity for high-value choices",
                    "evidence_strength": 0.8
                }
            ],
            user_framing_templates=[
                "Your decision capacity is depleted—not your motivation. Let's reduce choice complexity.",
                "This is decision fatigue, not avoidance. Binary choices work better right now.",
                "You've used your executive budget on other things. Let's autopilot the small stuff."
            ],
            related_mechanisms=["ENB-01", "ATT-13", "MOT-20"],
            mimicking_confounds=["low_blood_sugar", "boredom", "task_aversion"]
        )
        
        self.mechanisms["ENB-04"] = CognitiveMechanism(
            mechanism_id="ENB-04",
            subsystem=Subsystem.ENERGY_NEUROBIOLOGY,
            name="Glucose stability affects persistence",
            description="Blood glucose fluctuations impair sustained effort more than willingness to exert effort.",
            empirical_basis="Gailliot glucose & self-control research; Inzlicht cognitive effort models",
            confidence=MechanismCategory.MODERATE_CONFIDENCE,
            observable_signatures=[
                {
                    "data_type": "productivity_metrics",
                    "pattern": "Performance drop 2-3 hours post-high-carb meal",
                    "confidence_boost": 0.7
                },
                {
                    "data_type": "app_usage",
                    "pattern": "Energy apps (caffeine tracking, food logging) correlate with productivity",
                    "confidence_boost": 0.5
                },
                {
                    "data_type": "time_of_day",
                    "pattern": "Post-lunch slump (1-3 PM) followed by recovery",
                    "confidence_boost": 0.6
                }
            ],
            predictions=[
                "Persistence on difficult tasks drops before subjective effort reports do",
                "Small, frequent protein intake correlates with stable afternoon performance",
                "High-glycemic lunch predicts 30-40 min productivity drop"
            ],
            intervention_levers=[
                {
                    "action": "Protein-forward breakfast, low-glycemic lunch",
                    "expected_effect": "Stabilize afternoon persistence",
                    "evidence_strength": 0.7
                },
                {
                    "action": "Strategic caffeine 30 min before high-demand period",
                    "expected_effect": "Counter glucose dip, extend persistence",
                    "evidence_strength": 0.75
                }
            ],
            user_framing_templates=[
                "Your energy stability is affecting sustained attention. Let's look at timing.",
                "This isn't willpower—it's blood sugar affecting your brain's persistence system.",
                "Your willingness is there; your metabolic stability is fluctuating."
            ],
            related_mechanisms=["ENB-01", "ENB-05"],
            mimicking_confounds=["sleep_debt", "boredom", "depression"]
        )
        
        self.mechanisms["ENB-05"] = CognitiveMechanism(
            mechanism_id="ENB-05",
            subsystem=Subsystem.ENERGY_NEUROBIOLOGY,
            name="Stress narrows attentional bandwidth",
            description="Threat response prioritizes immediate survival over broad processing.",
            empirical_basis="Sapolsky stress research; Arnsten prefrontal cortex under stress",
            confidence=MechanismCategory.HIGH_CONFIDENCE,
            observable_signatures=[
                {
                    "data_type": "app_usage",
                    "pattern": "Rapid switching between apps, incomplete sessions",
                    "confidence_boost": 0.7
                },
                {
                    "data_type": "communication",
                    "pattern": "Shorter messages, delayed responses to complex requests",
                    "confidence_boost": 0.6
                },
                {
                    "data_type": "ema_response",
                    "pattern": "Reports feeling 'scattered', 'can't focus', 'overwhelmed'",
                    "confidence_boost": 0.85
                },
                {
                    "data_type": "keyboard_dynamics",
                    "pattern": "Faster typing but more errors; less editing",
                    "confidence_boost": 0.65
                }
            ],
            predictions=[
                "User will complete urgent, simple tasks while neglecting important, complex ones",
                "Novel problem-solving capacity drops while routine execution maintains",
                "Social communication becomes more reactive, less reflective"
            ],
            intervention_levers=[
                {
                    "action": "Reduce cognitive load (fewer options, clear next step)",
                    "expected_effect": "Work within narrowed bandwidth",
                    "evidence_strength": 0.8
                },
                {
                    "action": "Brief physical movement or breathing exercise",
                    "expected_effect": "Reset threat response, restore bandwidth",
                    "evidence_strength": 0.7
                },
                {
                    "action": "Delay complex decisions until stress indicator drops",
                    "expected_effect": "Avoid reactive choices made under threat",
                    "evidence_strength": 0.75
                }
            ],
            user_framing_templates=[
                "Your attention has narrowed to threat mode. Let's reduce the load temporarily.",
                "This is your stress response prioritizing immediate survival. It will pass.",
                "Your bandwidth is compressed right now. Simple steps, not complex plans."
            ],
            related_mechanisms=["ENB-06", "AVD-25", "AVD-30"],
            mimicking_confounds=["caffeine_overconsumption", "anxiety_disorder", "time_pressure"]
        )
        
        self.mechanisms["ENB-06"] = CognitiveMechanism(
            mechanism_id="ENB-06",
            subsystem=Subsystem.ENERGY_NEUROBIOLOGY,
            name="High arousal shifts to reaction control",
            description="Elevated physiological arousal moves brain from planning mode to reactive mode.",
            empirical_basis="Yerkes-Dodson law; Beilock choking under pressure",
            confidence=MechanismCategory.HIGH_CONFIDENCE,
            observable_signatures=[
                {
                    "data_type": "task_execution",
                    "pattern": "Abandons planned approach for impulsive action",
                    "confidence_boost": 0.75
                },
                {
                    "data_type": "ema_response",
                    "pattern": "Reports 'racing thoughts', 'can't sit still', 'need to do something'",
                    "confidence_boost": 0.8
                },
                {
                    "data_type": "device_interaction",
                    "pattern": "Rapid, shallow interactions; less time per app",
                    "confidence_boost": 0.7
                }
            ],
            predictions=[
                "Planned, methodical approaches abandoned for quick fixes",
                "User will prefer action over planning, even when planning is optimal",
                "Performance on practiced skills maintains; novel learning fails"
            ],
            intervention_levers=[
                {
                    "action": "Physical movement to discharge arousal",
                    "expected_effect": "Return to planning-capable state",
                    "evidence_strength": 0.75
                },
                {
                    "action": "Structured 'reaction window' (5 min to act on impulse, then reassess)",
                    "expected_effect": "Honor arousal without full abandonment of plan",
                    "evidence_strength": 0.6
                }
            ],
            user_framing_templates=[
                "Your system is in reaction mode right now. Let's channel that energy.",
                "Planning is hard when arousal is high. Let's move first, plan second.",
                "Your brain has shifted gears. Let's work with that, not against it."
            ],
            related_mechanisms=["ENB-05", "MOT-24"],
            mimicking_confounds=["mania", "stimulant_use", "anxiety"]
        )
        
        self.mechanisms["ENB-07"] = CognitiveMechanism(
            mechanism_id="ENB-07",
            subsystem=Subsystem.ENERGY_NEUROBIOLOGY,
            name="Fatigue increases familiar action preference",
            description="Tired brains default to habitual patterns over optimal choices.",
            empirical_basis="Wood & Neal habit research; Payne et al. cognitive miser",
            confidence=MechanismCategory.HIGH_CONFIDENCE,
            observable_signatures=[
                {
                    "data_type": "app_usage",
                    "pattern": "Return to most-used apps despite intending to use new tool",
                    "confidence_boost": 0.8
                },
                {
                    "data_type": "task_execution",
                    "pattern": "Uses old workflow even when new one was planned",
                    "confidence_boost": 0.75
                },
                {
                    "data_type": "time_of_day",
                    "pattern": "Evening reversion to established patterns",
                    "confidence_boost": 0.7
                }
            ],
            predictions=[
                "New habits fail in evening even if successfully practiced in morning",
                "User will report 'autopilot took over' or 'don't know why I did that'",
                "Familiar suboptimal path chosen over unfamiliar optimal path"
            ],
            intervention_levers=[
                {
                    "action": "Make desired behavior the default (environment design)",
                    "expected_effect": "Align default with optimal",
                    "evidence_strength": 0.85
                },
                {
                    "action": "Practice new behavior only during high-energy periods",
                    "expected_effect": "Strengthen habit before relying on it when fatigued",
                    "evidence_strength": 0.8
                }
            ],
            user_framing_templates=[
                "Fatigue makes familiar paths magnetic. Let's make the good path the easy path.",
                "Your autopilot is strong right now. Let's work with it, not against it.",
                "New behaviors need high-energy practice before they'll stick when tired."
            ],
            related_mechanisms=["LRN-38", "ENB-01", "ENB-03"],
            mimicking_confounds=["habit_strength", "lack_of_motivation"]
        )
        
        self.mechanisms["ENB-08"] = CognitiveMechanism(
            mechanism_id="ENB-08",
            subsystem=Subsystem.ENERGY_NEUROBIOLOGY,
            name="Physical discomfort taxes working memory",
            description="Pain, hunger, and thermal discomfort silently consume cognitive resources.",
            empirical_basis="Eccleston pain & attention; Hagger ego depletion meta-analysis",
            confidence=MechanismCategory.HIGH_CONFIDENCE,
            observable_signatures=[
                {
                    "data_type": "ema_response",
                    "pattern": "Reports hunger, headache, cold/hot, discomfort",
                    "confidence_boost": 0.9
                },
                {
                    "data_type": "productivity_metrics",
                    "pattern": "Performance drop not explained by time of day or sleep",
                    "confidence_boost": 0.6
                },
                {
                    "data_type": "app_usage",
                    "pattern": "Food delivery apps, pain research, temperature checks",
                    "confidence_boost": 0.7
                }
            ],
            predictions=[
                "Addressing physical need restores performance faster than motivational intervention",
                "User will report 'couldn't focus' without identifying physical cause",
                "Performance variance correlates with time since last meal/sleep quality"
            ],
            intervention_levers=[
                {
                    "action": "Direct physical need address (snack, temperature, movement)",
                    "expected_effect": "Rapid restoration of cognitive capacity",
                    "evidence_strength": 0.85
                },
                {
                    "action": "Reduce task complexity until physical need addressed",
                    "expected_effect": "Work within reduced capacity",
                    "evidence_strength": 0.7
                }
            ],
            user_framing_templates=[
                "Your body is asking for something basic. Let's address that first.",
                "Physical discomfort is using up your attention budget. Quick fix, then work.",
                "This isn't motivation—it's a physical need reducing your cognitive capacity."
            ],
            related_mechanisms=["ENB-01", "ENB-04"],
            mimicking_confounds=["boredom", "low_motivation", "depression"]
        )
        
        # ============================================================
        # II. ATTENTION & INFORMATION PROCESSING
        # ============================================================
        
        self.mechanisms["ATT-09"] = CognitiveMechanism(
            mechanism_id="ATT-09",
            subsystem=Subsystem.ATTENTION_INFORMATION,
            name="Working memory ~4 unit limit",
            description="Working memory holds approximately 4 meaningful chunks. Overload becomes avoidance.",
            empirical_basis="Cowan working memory research; Miller magical number",
            confidence=MechanismCategory.HIGH_CONFIDENCE,
            observable_signatures=[
                {
                    "data_type": "task_interaction",
                    "pattern": "Abandons tasks with many open tabs/windows",
                    "confidence_boost": 0.8
                },
                {
                    "data_type": "ema_response",
                    "pattern": "Reports 'too much to keep track of', 'overwhelmed by details'",
                    "confidence_boost": 0.9
                },
                {
                    "data_type": "task_completion",
                    "pattern": "Completes simple subtasks but avoids complex integration",
                    "confidence_boost": 0.75
                }
            ],
            predictions=[
                "Tasks requiring tracking >4 variables will be avoided or failed",
                "Chunking information into 3-4 categories restores engagement",
                "User will prefer sequential processing over parallel processing"
            ],
            intervention_levers=[
                {
                    "action": "Externalize working memory (write down, organize visually)",
                    "expected_effect": "Free up capacity for processing",
                    "evidence_strength": 0.85
                },
                {
                    "action": "Chunk information into 3-4 categories maximum",
                    "expected_effect": "Fit within working memory constraints",
                    "evidence_strength": 0.8
                },
                {
                    "action": "Sequential single-tasking vs. parallel multi-tasking",
                    "expected_effect": "Reduce cognitive load",
                    "evidence_strength": 0.75
                }
            ],
            user_framing_templates=[
                "Your working memory is full. Let's externalize some of that.",
                "This task needs more mental slots than available right now. Let's chunk it.",
                "Your brain can hold ~4 things at once. Let's make sure they're the right 4."
            ],
            related_mechanisms=["ATT-11", "ATT-15", "ENB-01"],
            mimicking_confounds=["low_motivation", "complexity_aversion", "perfectionism"]
        )
        
        self.mechanisms["ATT-10"] = CognitiveMechanism(
            mechanism_id="ATT-10",
            subsystem=Subsystem.ATTENTION_INFORMATION,
            name="Ambiguity costs more than difficulty",
            description="Unclear tasks consume more cognitive resources than objectively hard but clear tasks.",
            empirical_basis="Kruglinski need for closure; Wilson & Gilbert affective forecasting",
            confidence=MechanismCategory.HIGH_CONFIDENCE,
            observable_signatures=[
                {
                    "data_type": "task_avoidance",
                    "pattern": "Avoids vague tasks ('work on project') while completing specific ones",
                    "confidence_boost": 0.85
                },
                {
                    "data_type": "ema_response",
                    "pattern": "Reports 'don't know where to start', 'not sure what to do'",
                    "confidence_boost": 0.9
                },
                {
                    "data_type": "search_behavior",
                    "pattern": "Excessive research/planning before starting (procrastination as clarification)",
                    "confidence_boost": 0.7
                }
            ],
            predictions=[
                "Clarifying next specific action will unlock progress where motivation interventions fail",
                "User will choose objectively harder but clearer task over easier but ambiguous one",
                "Time spent clarifying correlates with eventual completion more than time spent working"
            ],
            intervention_levers=[
                {
                    "action": "Convert ambiguous goal to specific next physical action",
                    "expected_effect": "Reduce ambiguity cost, enable initiation",
                    "evidence_strength": 0.9
                },
                {
                    "action": "Provide explicit success criteria",
                    "expected_effect": "Clarify what 'done' looks like",
                    "evidence_strength": 0.8
                },
                {
                    "action": "Time-box clarification (15 min to define, then start)",
                    "expected_effect": "Prevent research as procrastination",
                    "evidence_strength": 0.75
                }
            ],
            user_framing_templates=[
                "This task is ambiguous, not hard. Let's get specific about the next step.",
                "Your brain is spending energy on clarification. Let's do that explicitly.",
                "Uncertainty is more exhausting than effort. Let's remove the ambiguity."
            ],
            related_mechanisms=["AVD-26", "MOT-20", "ATT-14"],
            mimicking_confounds=["perfectionism", "low_self_efficacy", "task_aversion"]
        )
        
        # [Additional mechanisms 11-40 would follow same pattern...]
        # For brevity, I'll include key ones and mark where others go
        
        self.mechanisms["ATT-11"] = CognitiveMechanism(
            mechanism_id="ATT-11",
            subsystem=Subsystem.ATTENTION_INFORMATION,
            name="Task switching destroys momentum non-linearly",
            description="Switching costs are multiplicative, not additive. Deep work requires sustained attention.",
            empirical_basis="Mark task switching research; Newport deep work",
            confidence=MechanismCategory.HIGH_CONFIDENCE,
            observable_signatures=[
                {
                    "data_type": "app_usage",
                    "pattern": "Frequent switches (>10/hour) correlate with low task completion",
                    "confidence_boost": 0.85
                },
                {
                    "data_type": "productivity_metrics",
                    "pattern": "Long sessions (90+ min) produce disproportionately more output",
                    "confidence_boost": 0.8
                }
            ],
            predictions=[
                "Each switch adds 15-23 min recovery time, not just switch duration",
                "Notification interruptions have 10x cost of scheduled breaks",
                "Batching similar tasks reduces total time >50% vs. interleaving"
            ],
            intervention_levers=[
                {
                    "action": "90-120 min protected focus blocks",
                    "expected_effect": "Enable deep work, nonlinear productivity gains",
                    "evidence_strength": 0.85
                },
                {
                    "action": "Notification batching (check every 90 min, not real-time)",
                    "expected_effect": "Eliminate switch costs from interruptions",
                    "evidence_strength": 0.8
                }
            ],
            user_framing_templates=[
                "Each switch costs more than the time it takes. Let's protect your momentum.",
                "Your brain pays a tax every time you switch. Let's batch to reduce taxes.",
                "Deep work happens in sustained blocks, not scattered minutes."
            ],
            related_mechanisms=["ATT-12", "ATT-15", "ENB-01"],
            mimicking_confounds=["boredom", "low_frustration_tolerance"]
        )
        
        self.mechanisms["MOT-17"] = CognitiveMechanism(
            mechanism_id="MOT-17",
            subsystem=Subsystem.MOTIVATION_REWARD,
            name="Behavior follows predicted reward, not declared importance",
            description="Humans pursue expected relief, not stated goals. Predicted reward drives action.",
            empirical_basis="Schultz dopamine prediction error; Berridge wanting vs liking",
            confidence=MechanismCategory.HIGH_CONFIDENCE,
            observable_signatures=[
                {
                    "data_type": "goal_progress",
                    "pattern": "Important goals neglected while pursuing immediately rewarding activities",
                    "confidence_boost": 0.8
                },
                {
                    "data_type": "ema_response",
                    "pattern": "Reports 'know I should but don't feel like it'",
                    "confidence_boost": 0.85
                }
            ],
            predictions=[
                "Increasing anticipated reward (visualization, immediate payoff) unlocks action more than importance reminders",
                "User will choose lower-importance, higher-immediacy task",
                "Delay discounting: value drops 50% per day of delay"
            ],
            intervention_levers=[
                {
                    "action": "Make reward immediate and concrete (visualize completion, add immediate payoff)",
                    "expected_effect": "Increase predicted reward, drive action",
                    "evidence_strength": 0.85
                },
                {
                    "action": "Implementation intentions (if-then planning)",
                    "expected_effect": "Bridge intention-action gap",
                    "evidence_strength": 0.8
                }
            ],
            user_framing_templates=[
                "Your brain chases expected relief, not importance. Let's make the reward visible.",
                "This task needs a more immediate payoff to compete with alternatives.",
                "Importance doesn't drive action—expected reward does. Let's adjust the prediction."
            ],
            related_mechanisms=["MOT-18", "MOT-19", "MOT-24"],
            mimicking_confounds=["laziness", "lack_of_discipline", "procrastination"]
        )
        
        self.mechanisms["AVD-25"] = CognitiveMechanism(
            mechanism_id="AVD-25",
            subsystem=Subsystem.AVOIDANCE_THREAT,
            name="Avoidance reduces short-term distress automatically",
            description="Avoidance provides immediate relief, reinforcing itself. Not laziness—protective regulation.",
            empirical_basis="Barlow anxiety research; Hayes ACT; Mowrer two-factor theory",
            confidence=MechanismCategory.HIGH_CONFIDENCE,
            observable_signatures=[
                {
                    "data_type": "task_avoidance",
                    "pattern": "Task approached then abandoned at first difficulty",
                    "confidence_boost": 0.8
                },
                {
                    "data_type": "ema_response",
                    "pattern": "Reports relief after avoiding, guilt shortly after",
                    "confidence_boost": 0.9
                },
                {
                    "data_type": "app_usage",
                    "pattern": "Rapid shift to comfort apps (social, entertainment) when task opened",
                    "confidence_boost": 0.85
                }
            ],
            predictions=[
                "User will experience immediate relief followed by delayed guilt",
                "Avoidance pattern strengthens with each repetition (negative reinforcement)",
                "Approach attempts increase anxiety before they decrease it"
            ],
            intervention_levers=[
                {
                    "action": "Reduce initial approach cost (2-minute rule, tiny first step)",
                    "expected_effect": "Bypass avoidance trigger",
                    "evidence_strength": 0.85
                },
                {
                    "action": "Add immediate relief to approach (reward first step immediately)",
                    "expected_effect": "Compete with avoidance reinforcement",
                    "evidence_strength": 0.8
                },
                {
                    "action": "Mindfulness of avoidance urge without action",
                    "expected_effect": "Break automatic reinforcement loop",
                    "evidence_strength": 0.75
                }
            ],
            user_framing_templates=[
                "Avoidance gave you immediate relief—that's why it's automatic. Let's add relief to approach.",
                "This isn't laziness; it's your threat system protecting you. Let's reassure it.",
                "The relief from avoiding is temporary. Let's build relief into starting."
            ],
            related_mechanisms=["AVD-26", "AVD-27", "AVD-30", "MOT-17"],
            mimicking_confounds=["laziness", "lack_of_willpower", "procrastination"]
        )
        
        self.mechanisms["LRN-33"] = CognitiveMechanism(
            mechanism_id="LRN-33",
            subsystem=Subsystem.LEARNING_BELIEF,
            name="Experience updates beliefs faster than explanation",
            description="Humans update from lived experience, not told information. Action shapes belief.",
            empirical_basis="Daw reward learning; Friston active inference; James action precedes attitude",
            confidence=MechanismCategory.HIGH_CONFIDENCE,
            observable_signatures=[
                {
                    "data_type": "belief_statements",
                    "pattern": "User states belief but behavior contradicts",
                    "confidence_boost": 0.75
                },
                {
                    "data_type": "intervention_response",
                    "pattern": "Explains why approach failed but continues same pattern",
                    "confidence_boost": 0.8
                }
            ],
            predictions=[
                "Direct experience of success changes behavior more than education about success",
                "User will defend existing belief despite contradictory evidence until they act differently",
                "Small successful actions shift self-concept more than big plans"
            ],
            intervention_levers=[
                {
                    "action": "Design tiny actions that guarantee success (behavioral activation)",
                    "expected_effect": "Create experiential update, not just cognitive",
                    "evidence_strength": 0.9
                },
                {
                    "action": "Reduce action threshold to <2 minutes",
                    "expected_effect": "Enable experience before avoidance kicks in",
                    "evidence_strength": 0.85
                }
            ],
            user_framing_templates=[
                "Beliefs change through experience, not explanation. Let's create an experience.",
                "Your brain learns from what you do, not what you know. Let's start small.",
                "Understanding isn't enough—your nervous system needs direct evidence."
            ],
            related_mechanisms=["LRN-36", "SOC-40"],
            mimicking_confounds=["resistance", "denial", "lack_of_insight"]
        )
        
        self.mechanisms["SOC-40"] = CognitiveMechanism(
            mechanism_id="SOC-40",
            subsystem=Subsystem.SOCIAL_IDENTITY,
            name="Identity consistency overrides optimization",
            description="People protect self-concept even when it harms outcomes. 'That's not me' blocks change.",
            empirical_basis="Steele self-affirmation; Swann self-verification; Dweck mindset",
            confidence=MechanismCategory.HIGH_CONFIDENCE,
            observable_signatures=[
                {
                    "data_type": "ema_response",
                    "pattern": "Reports 'that's not who I am', 'not my style', 'feels fake'",
                    "confidence_boost": 0.9
                },
                {
                    "data_type": "intervention_response",
                    "pattern": "Rejects effective strategy because it doesn't fit self-image",
                    "confidence_boost": 0.85
                },
                {
                    "data_type": "behavioral_pattern",
                    "pattern": "Maintains suboptimal but identity-consistent behavior",
                    "confidence_boost": 0.8
                }
            ],
            predictions=[
                "Interventions framed as 'becoming more yourself' succeed more than 'changing yourself'",
                "Identity threat causes rejection of objectively superior strategies",
                "Self-affirmation before change attempt increases adoption 2-3x"
            ],
            intervention_levers=[
                {
                    "action": "Frame change as 'becoming more fully yourself' not 'changing who you are'",
                    "expected_effect": "Reduce identity threat, enable adoption",
                    "evidence_strength": 0.9
                },
                {
                    "action": "Self-affirmation of core values before change attempt",
                    "expected_effect": "Secure identity, reduce defensive rejection",
                    "evidence_strength": 0.85
                },
                {
                    "action": "Find identity-consistent path to goal (e.g., 'as a curious person...')",
                    "expected_effect": "Align change with existing self-concept",
                    "evidence_strength": 0.8
                }
            ],
            user_framing_templates=[
                "This isn't about changing who you are—it's about becoming more fully yourself.",
                "Your identity is your anchor. Let's use it, not fight it.",
                "Let's find the version of this that feels like *you*.",
                "You're already someone who values {core_value}. This aligns with that."
            ],
            related_mechanisms=["AVD-27", "AVD-32", "LRN-37"],
            mimicking_confounds=["resistance", "stubbornness", "fear_of_success"]
        )
        
        # Additional mechanisms would be initialized here...
        # For complete implementation, all 40 mechanisms would be defined
    
    def get_mechanism(self, mechanism_id: str) -> Optional[CognitiveMechanism]:
        """Retrieve mechanism by ID."""
        return self.mechanisms.get(mechanism_id)
    
    def get_mechanisms_by_subsystem(self, subsystem: Subsystem) -> List[CognitiveMechanism]:
        """Get all mechanisms in a subsystem."""
        return [m for m in self.mechanisms.values() if m.subsystem == subsystem]
    
    def generate_candidate_mechanisms(
        self,
        observation: Dict[str, Any],
        top_n: int = 5
    ) -> List[Tuple[CognitiveMechanism, float]]:
        """
        Generate candidate mechanisms for observed behavior.
        Returns ranked list of (mechanism, confidence_score).
        """
        candidates = []
        
        for mechanism in self.mechanisms.values():
            score = self._score_mechanism_fit(mechanism, observation)
            if score > 0.3:  # Minimum threshold
                candidates.append((mechanism, score))
        
        # Sort by confidence
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_n]
    
    def _score_mechanism_fit(
        self,
        mechanism: CognitiveMechanism,
        observation: Dict[str, Any]
    ) -> float:
        """Score how well a mechanism explains an observation."""
        scores = []
        
        for signature in mechanism.observable_signatures:
            match_score = self._match_signature(signature, observation)
            weighted = match_score * signature.get("confidence_boost", 0.5)
            scores.append(weighted)
        
        if not scores:
            return 0.0
        
        # Average with boost for high-confidence mechanisms
        base_score = sum(scores) / len(scores)
        confidence_multiplier = {
            MechanismCategory.HIGH_CONFIDENCE: 1.2,
            MechanismCategory.MODERATE_CONFIDENCE: 1.0,
            MechanismCategory.EMERGING: 0.8
        }.get(mechanism.confidence, 1.0)
        
        return min(1.0, base_score * confidence_multiplier)
    
    def _match_signature(
        self,
        signature: Dict[str, Any],
        observation: Dict[str, Any]
    ) -> float:
        """Match a single signature against observation."""
        data_type = signature.get("data_type")
        pattern = signature.get("pattern", "").lower()
        
        # Check if observation has this data type
        if data_type == "ema_response" and "ema" not in str(observation.get("observation_type", "")):
            return 0.0
        
        if data_type == "app_usage" and observation.get("behavior", {}).get("app_category") is None:
            return 0.0
        
        # Simple pattern matching (would be more sophisticated in production)
        observation_text = json.dumps(observation).lower()
        
        # Extract key terms from pattern
        key_terms = [term for term in pattern.split() if len(term) > 4]
        matches = sum(1 for term in key_terms if term in observation_text)
        
        return min(1.0, matches / max(1, len(key_terms) * 0.5))
    
    def get_intervention_for_mechanism(
        self,
        mechanism_id: str,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Get best intervention lever for a mechanism given context.
        """
        mechanism = self.mechanisms.get(mechanism_id)
        if not mechanism:
            return None
        
        # Score each intervention lever for this context
        best_intervention = None
        best_score = 0
        
        for lever in mechanism.intervention_levers:
            score = lever.get("evidence_strength", 0.5)
            
            # Boost if lever matches context constraints
            if self._lever_fits_context(lever, context):
                score += 0.2
            
            if score > best_score:
                best_score = score
                best_intervention = lever
        
        if best_intervention:
            return {
                "mechanism": mechanism,
                "intervention": best_intervention,
                "framing": self._select_framing(mechanism, context),
                "confidence": best_score
            }
        
        return None
    
    def _lever_fits_context(self, lever: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if intervention lever fits current context."""
        # Simple checks - would be more sophisticated
        action = lever.get("action", "").lower()
        
        if "morning" in action and context.get("time_of_day") == "evening":
            return False
        if "sleep" in action and context.get("sleep_debt") == False:
            return False
        
        return True
    
    def _select_framing(
        self,
        mechanism: CognitiveMechanism,
        context: Dict[str, Any]
    ) -> str:
        """Select appropriate user-facing framing."""
        templates = mechanism.user_framing_templates
        
        if not templates:
            return f"This appears to be {mechanism.name}."
        
        # Select based on context
        if context.get("user_emotional_state") == "discouraged":
            # Use most supportive framing
            return templates[0]
        
        if context.get("user_knowledge_level") == "high":
            # Use more technical framing
            return templates[-1] if len(templates) > 1 else templates[0]
        
        return templates[0]
    
    def get_related_mechanisms(self, mechanism_id: str) -> List[CognitiveMechanism]:
        """Get mechanisms that often co-occur with this one."""
        mechanism = self.mechanisms.get(mechanism_id)
        if not mechanism:
            return []
        
        related = []
        for related_id in mechanism.related_mechanisms:
            m = self.mechanisms.get(related_id)
            if m:
                related.append(m)
        
        return related
    
    def check_confounds(
        self,
        mechanism_id: str,
        detected_confounds: List[str]
    ) -> List[str]:
        """
        Check if detected confounds might be mimicking this mechanism.
        Returns list of confounds that could explain the pattern instead.
        """
        mechanism = self.mechanisms.get(mechanism_id)
        if not mechanism:
            return []
        
        mimicking = []
        for confound in detected_confounds:
            if confound in mechanism.mimicking_confounds:
                mimicking.append(confound)
        
        return mimicking
    
    def generate_experimental_design(
        self,
        candidate_mechanisms: List[Tuple[CognitiveMechanism, float]],
        ambiguity: str
    ) -> Dict[str, Any]:
        """
        Generate experiment to disambiguate between candidate mechanisms.
        """
        if len(candidate_mechanisms) < 2:
            return {
                "design_type": "single_mechanism_test",
                "target": candidate_mechanisms[0][0].mechanism_id if candidate_mechanisms else None
            }
        
        # Find distinguishing features between top candidates
        mech_a, conf_a = candidate_mechanisms[0]
        mech_b, conf_b = candidate_mechanisms[1]
        
        # Generate conditions that would differentiate
        distinguishing_interventions = []
        
        for lever_a in mech_a.intervention_levers:
            for lever_b in mech_b.intervention_levers:
                if lever_a["action"] != lever_b["action"]:
                    distinguishing_interventions.append({
                        "test_a": lever_a,
                        "test_b": lever_b,
                        "predicted_difference": f"If {mech_a.name}: {lever_a['expected_effect']}. "
                                              f"If {mech_b.name}: {lever_b['expected_effect']}"
                    })
        
        return {
            "design_type": "mechanism_disambiguation",
            "mechanism_a": mech_a.mechanism_id,
            "mechanism_b": mech_b.mechanism_id,
            "confidence_gap": conf_a - conf_b,
            "distinguishing_tests": distinguishing_interventions[:3],
            "experiment_logic": f"Test which intervention produces expected effect. "
                               f"If A works, mechanism is likely {mech_a.name}. "
                               f"If B works, mechanism is likely {mech_b.name}."
        }


# Global singleton instance
_hom_instance: Optional[HumanOperatingModel] = None

def get_human_operating_model() -> HumanOperatingModel:
    """Get or create global HOM instance."""
    global _hom_instance
    if _hom_instance is None:
        _hom_instance = HumanOperatingModel()
    return _hom_instance
