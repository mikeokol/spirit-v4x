
# Create the Human Strategy Model file
code = '''"""
Human Strategy Model (HSM): How agents navigate reality.

HOM = how brains function (biological constraints)
HSM = how agents navigate reality (strategic optimization)

Spirit needs priors like:
- Humans optimize simultaneously for energy, social position, uncertainty reduction, 
  identity stability, and future optionality
- Not happiness. Not productivity.

When Spirit doesn't model this, it gives advice that is locally rational but globally resisted.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum, auto
import json


class OptimizationTarget(Enum):
    """What humans actually optimize for (not what they say)."""
    ENERGY_EFFICIENCY = "energy_efficiency"  # Minimize effort for acceptable outcomes
    SOCIAL_POSITION = "social_position"      # Relative standing, respect, status
    UNCERTAINTY_REDUCTION = "uncertainty_reduction"  # Predictability, safety
    IDENTITY_STABILITY = "identity_stability"  # Self-coherence, narrative continuity
    FUTURE_OPTIONALITY = "future_optionality"  # Keep doors open, preserve choices
    EMOTIONAL_REGULATION = "emotional_regulation"  # Manage affective state
    COGNITIVE_FLUENCY = "cognitive_fluency"  # Mental ease, processing smoothness


class GameType(Enum):
    """Types of 'games' humans play repeatedly."""
    SURVIVAL_UNCERTAINTY = "survival_uncertainty"  # Predictability before achievement
    STATUS_SOCIAL = "status_social"              # Position games
    EFFORT_ENERGY = "effort_energy"              # Resource allocation
    IDENTITY_COHERENCE = "identity_coherence"    # Self-consistency
    TIME_HORIZON = "time_horizon"                # Present vs future negotiation


@dataclass
class StrategicPressure:
    """
    A recurring optimization pressure that shapes decisions.
    Not a belief - a structural tension.
    """
    pressure_id: str
    game_type: GameType
    
    # Description
    name: str
    description: str
    
    # When active
    activation_triggers: List[str]  # Conditions that activate this pressure
    deactivation_triggers: List[str]  # Conditions that release it
    
    # Observable signatures
    behavioral_signatures: List[Dict[str, Any]]
    # e.g., {"pattern": "delay_near_commitment", "confidence": 0.8}
    
    # Strategic logic
    optimization_target: OptimizationTarget
    what_is_protected: str  # What cost is being avoided
    what_is_risked: str     # What is sacrificed
    
    # Testable predictions
    predictions: List[str]
    
    # Intervention levers
    intervention_logic: str  # How to change the game, not just the behavior
    
    # Related pressures (often co-occur)
    related_pressures: List[str]  # pressure_ids
    
    # Conflicts with
    conflicting_pressures: List[str]  # pressure_ids


class HumanStrategyModel:
    """
    Structured knowledge of strategic optimization pressures.
    Powers Spirit's ability to detect hidden games.
    """
    
    def __init__(self):
        self.pressures: Dict[str, StrategicPressure] = {}
        self._initialize_pressures()
    
    def _initialize_pressures(self):
        """Initialize the 40 strategic pressures across 5 game types."""
        
        # ============================================================
        # A. SURVIVAL & UNCERTAINTY GAMES
        # ============================================================
        
        self.pressures["UNC-01"] = StrategicPressure(
            pressure_id="UNC-01",
            game_type=GameType.SURVIVAL_UNCERTAINTY,
            name="Uncertainty minimization beats reward maximization",
            description="Humans first optimize predictability before achievement. Certainty > optimal uncertain outcomes.",
            activation_triggers=["ambiguous_path", "unclear_outcomes", "novel_situation"],
            deactivation_triggers=["established_routine", "clear_causality", "past_success_in_domain"],
            behavioral_signatures=[
                {"pattern": "chooses_known_mediocre_over_unknown_superior", "confidence": 0.9},
                {"pattern": "gathers_excessive_information_before_acting", "confidence": 0.8},
                {"pattern": "prefers_reliable_small_wins_over_volatile_big_wins", "confidence": 0.85}
            ],
            optimization_target=OptimizationTarget.UNCERTAINTY_REDUCTION,
            what_is_protected="Psychological safety, predictability of environment",
            what_is_risked="Optimal outcomes, growth opportunities",
            predictions=[
                "User will choose lower-EV option if it has lower variance",
                "Information gathering increases with outcome uncertainty",
                "Established routines resist change even when suboptimal"
            ],
            intervention_logic="Reduce perceived uncertainty before asking for change. Make new path feel familiar.",
            related_pressures=["UNC-02", "UNC-04", "EFF-17"],
            conflicting_pressures=["STA-09", "IDE-25"]
        )
        
        self.pressures["UNC-02"] = StrategicPressure(
            pressure_id="UNC-02",
            game_type=GameType.SURVIVAL_UNCERTAINTY,
            name="Reliable mediocre preferred over volatile superior",
            description="Consistent mediocre outcomes feel safer than high-variance superior ones.",
            activation_triggers=["past_volatility_trauma", "resource_constraints", "high_stakes"],
            deactivation_triggers=["abundant_resources", "risk_tolerance_developed", "safety_net_established"],
            behavioral_signatures=[
                {"pattern": "maintains_stable_underperforming_routine", "confidence": 0.85},
                {"pattern": "rejects_opportunities_with_uncertain_payoffs", "confidence": 0.8},
                {"pattern": "values_job_security_over_career_growth", "confidence": 0.75}
            ],
            optimization_target=OptimizationTarget.UNCERTAINTY_REDUCTION,
            what_is_protected="Emotional stability, sleep quality, anxiety management",
            what_is_risked="Upside potential, compounding gains",
            predictions=[
                "User will stay in suboptimal stable situation rather than pursue uncertain better",
                "Volatility in any domain triggers avoidance across domains",
                "Preference for employment over entrepreneurship even with lower EV"
            ],
            intervention_logic="Make new path feel more certain than current. Emphasize reliability of change.",
            related_pressures=["UNC-01", "UNC-03"],
            conflicting_pressures=["STA-10", "EFF-19"]
        )
        
        self.pressures["UNC-03"] = StrategicPressure(
            pressure_id="UNC-03",
            game_type=GameType.SURVIVAL_UNCERTAINTY,
            name="Optionality protection over gain pursuit",
            description="People protect future choices more than they pursue current gains. Keeping doors open.",
            activation_triggers=["irreversible_decision", "commitment_required", "path_elimination"],
            deactivation_triggers=["clear_best_path", "sunk_cost_activated", "identity_aligned_commitment"],
            behavioral_signatures=[
                {"pattern": "delay_decision_to_preserve_choices", "confidence": 0.85},
                {"pattern": "maintains_multiple_parallel_paths", "confidence": 0.8},
                {"pattern": "avoids_specialization", "confidence": 0.75}
            ],
            optimization_target=OptimizationTarget.FUTURE_OPTIONALITY,
            what_is_protected="Future flexibility, escape routes, possibility space",
            what_is_risked="Deep expertise, compound returns from commitment",
            predictions=[
                "User will delay specialization even when it's optimal",
                "Commitment devices trigger resistance",
                "Maintains backup plans that reduce primary plan success"
            ],
            intervention_logic="Frame change as increasing optionality, not reducing it. Show how commitment opens new doors.",
            related_pressures=["UNC-01", "TIM-33"],
            conflicting_pressures=["IDE-25", "EFF-22"]
        )
        
        self.pressures["UNC-04"] = StrategicPressure(
            pressure_id="UNC-04",
            game_type=GameType.SURVIVAL_UNCERTAINTY,
            name="Ambiguity feels riskier than difficulty",
            description="Unclear paths feel more dangerous than difficult but clear ones.",
            activation_triggers=["unclear_requirements", "ambiguous_success_criteria", "novel_domain"],
            deactivation_triggers=["clear_roadmap", "proven_process", "expert_guidance_available"],
            behavioral_signatures=[
                {"pattern": "avoids_ambiguous_tasks_despite_capability", "confidence": 0.9},
                {"pattern": "over_researches_before_starting", "confidence": 0.85},
                {"pattern": "prefers_difficult_clear_over_easy_unclear", "confidence": 0.8}
            ],
            optimization_target=OptimizationTarget.COGNITIVE_FLUENCY,
            what_is_protected="Mental ease, confidence in approach, reduced anxiety",
            what_is_risked="Optimal paths, innovation, efficiency",
            predictions=[
                "User will choose harder task with clear steps over easier ambiguous one",
                "Time spent clarifying exceeds time spent executing",
                "Paralysis when next step unclear despite high capability"
            ],
            intervention_logic="Provide clarity before asking for effort. Reduce ambiguity cost.",
            related_pressures=["UNC-01", "EFF-18"],
            conflicting_pressures=[]
        )
        
        self.pressures["UNC-05"] = StrategicPressure(
            pressure_id="UNC-05",
            game_type=GameType.SURVIVAL_UNCERTAINTY,
            name="Commitment anxiety with irreversible consequences",
            description="Anxiety increases as decision irreversibility grows.",
            activation_triggers=["irreversible_choice", "high_stakes_commitment", "public_commitment"],
            deactivation_triggers=["reversible_decision", "trial_period", "exit_options_clear"],
            behavioral_signatures=[
                {"pattern": "delay_near_final_commitment", "confidence": 0.9},
                {"pattern": "seeks_escape_clauses", "confidence": 0.8},
                {"pattern": "cold_feet_before_deadline", "confidence": 0.85}
            ],
            optimization_target=OptimizationTarget.FUTURE_OPTIONALITY,
            what_is_protected="Freedom, escape routes, psychological safety",
            what_is_risked="Seizing opportunities, deep engagement, compound returns",
            predictions=[
                "User will delay final commitment even after extensive preparation",
                "Last-minute requests for modifications/extensions",
                "Backing out at commitment point despite prior enthusiasm"
            ],
            intervention_logic="Build in reversibility or trial periods. Make commitment feel provisional.",
            related_pressures=["UNC-03", "TIM-33"],
            conflicting_pressures=["IDE-25"]
        )
        
        self.pressures["UNC-06"] = StrategicPressure(
            pressure_id="UNC-06",
            game_type=GameType.SURVIVAL_UNCERTAINTY,
            name="Delay decisions to preserve future selves",
            description="Decisions bind future selves, creating obligation to strangers (future us).",
            activation_triggers=["long_term_commitment", "identity_implications", "future_self_conflict"],
            deactivation_triggers=["short_term_horizon", "identity_aligned_choice", "present_focus_activated"],
            behavioral_signatures=[
                {"pattern": "avoids_long_term_planning", "confidence": 0.75},
                {"pattern": "prefers_flexibility_over_optimization", "confidence": 0.8},
                {"pattern": "sabotages_commitments_to_future", "confidence": 0.7}
            ],
            optimization_target=OptimizationTarget.FUTURE_OPTIONALITY,
            what_is_protected="Autonomy of future self, freedom from past obligations",
            what_is_risked="Long-term goals, compound benefits, consistency",
            predictions=[
                "User will underweight future benefits in decisions",
                "Preference for short-term contracts over long-term",
                "Procrastination on decisions with long-term implications"
            ],
            intervention_logic="Make future benefits immediate. Reduce perceived binding of decisions.",
            related_pressures=["UNC-03", "TIM-33", "TIM-34"],
            conflicting_pressures=["IDE-25"]
        )
        
        self.pressures["UNC-07"] = StrategicPressure(
            pressure_id="UNC-07",
            game_type=GameType.SURVIVAL_UNCERTAINTY,
            name="Information seeking replaces action",
            description="Research and planning serve emotional regulation, not preparation.",
            activation_triggers=["anxiety_about_outcome", "unclear_path", "high_stakes"],
            deactivation_triggers=["deadline_imminent", "action_forced", "analysis_paralysis_recognized"],
            behavioral_signatures=[
                {"pattern": "excessive_research_without_action", "confidence": 0.9},
                {"pattern": "continuous_planning_refinement", "confidence": 0.85},
                {"pattern": "information_gathering_increases_anxiety", "confidence": 0.8}
            ],
            optimization_target=OptimizationTarget.EMOTIONAL_REGULATION,
            what_is_protected="Sense of control, anxiety reduction, illusion of preparation",
            what_is_risked="Timeliness, momentum, actual progress",
            predictions=[
                "User will gather information beyond point of diminishing returns",
                "Research continues until deadline forces action",
                "More information increases rather than decreases anxiety"
            ],
            intervention_logic="Limit information gathering. Make action the anxiety regulator.",
            related_pressures=["UNC-01", "UNC-04", "EFF-17"],
            conflicting_pressures=["EFF-19"]
        )
        
        self.pressures["UNC-08"] = StrategicPressure(
            pressure_id="UNC-08",
            game_type=GameType.SURVIVAL_UNCERTAINTY,
            name="Planning as emotional regulation",
            description="Planning reduces anxiety about future, not actually prepares for it.",
            activation_triggers=["future_anxiety", "uncertainty_about_outcome", "low_control_perception"],
            deactivation_triggers=["plan_implemented", "outcome_uncertainty_resolved", "acceptance_reached"],
            behavioral_signatures=[
                {"pattern": "detailed_planning_without_execution", "confidence": 0.85},
                {"pattern": "repeated_planning_for_same_goal", "confidence": 0.8},
                {"pattern": "planning_reduces_anxiety_temporarily", "confidence": 0.9}
            ],
            optimization_target=OptimizationTarget.EMOTIONAL_REGULATION,
            what_is_protected="Emotional stability, sense of control, reduced anxiety",
            what_is_risked="Actual preparation, adaptability, response to reality",
            predictions=[
                "User will replan when anxiety returns, not when plan fails",
                "Detailed plans don't correlate with better outcomes",
                "Planning stops when anxiety reduces, not when preparation complete"
            ],
            intervention_logic="Distinguish planning from preparation. Focus on doing over planning.",
            related_pressures=["UNC-07", "EFF-17"],
            conflicting_pressures=["EFF-19"]
        )
        
        # ============================================================
        # B. STATUS & SOCIAL POSITION GAMES
        # ============================================================
        
        self.pressures["STA-09"] = StrategicPressure(
            pressure_id="STA-09",
            game_type=GameType.STATUS_SOCIAL,
            name="Relative position matters more than absolute outcome",
            description="Social comparison drives satisfaction more than objective results.",
            activation_triggers=["social_comparison_available", "public_performance", "ranking_visible"],
            deactivation_triggers=["isolated_context", "absolute_standards_clear", "intrinsic_motivation_dominant"],
            behavioral_signatures=[
                {"pattern": "satisfaction_correlates_with_ranking_not_absolute", "confidence": 0.9},
                {"pattern": "effort_increases_when_ranking_visible", "confidence": 0.85},
                {"pattern": "quits_when_ranking_low_despite_absolute_success", "confidence": 0.8}
            ],
            optimization_target=OptimizationTarget.SOCIAL_POSITION,
            what_is_protected="Self-esteem, social standing, relative status",
            what_is_risked="Absolute gains, intrinsic satisfaction, collaboration",
            predictions=[
                "User will be dissatisfied with good absolute outcome if ranked low",
                "Effort increases in competitive contexts even when suboptimal",
                "Avoidance of domains where relative performance poor"
            ],
            intervention_logic="Frame progress in relative terms. Make comparison favorable or irrelevant.",
            related_pressures=["STA-10", "STA-12", "IDE-29"],
            conflicting_pressures=["UNC-01", "IDE-26"]
        )
        
        self.pressures["STA-10"] = StrategicPressure(
            pressure_id="STA-10",
            game_type=GameType.STATUS_SOCIAL,
            name="Avoid arenas where rank may be low",
            description="People avoid domains where they might rank poorly, even if absolute gains possible.",
            activation_triggers=["new_domain", "visible_ranking", "high_performers_present"],
            deactivation_triggers=["established_competence", "private_progress", "beginner_context"],
            behavioral_signatures=[
                {"pattern": "avoids_domains_with_experts", "confidence": 0.8},
                {"pattern": "prefers_big_fish_small_pond", "confidence": 0.85},
                {"pattern": "quits_when_comparison_unfavorable", "confidence": 0.8}
            ],
            optimization_target=OptimizationTarget.SOCIAL_POSITION,
            what_is_protected="Self-image, confidence, social standing",
            what_is_risked="Growth, learning, optimal challenge",
            predictions=[
                "User will choose easier domain over harder one if ranking higher",
                "Avoidance of competitive environments despite interest",
                "Preference for contexts where already top performer"
            ],
            intervention_logic="Create private or beginner-friendly contexts. Remove social comparison.",
            related_pressures=["STA-09", "STA-11", "IDE-28"],
            conflicting_pressures=["EFF-19"]
        )
        
        self.pressures["STA-11"] = StrategicPressure(
            pressure_id="STA-11",
            game_type=GameType.STATUS_SOCIAL,
            name="Public failure hurts more than private failure",
            description="Visibility amplifies failure cost. Private practice > public performance.",
            activation_triggers=["public_audience", "visible_failure", "reputation_at_stake"],
            deactivation_triggers=["private_context", "anonymous_attempt", "failure_normed"],
            behavioral_signatures=[
                {"pattern": "performs_better_in_private", "confidence": 0.85},
                {"pattern": "avoids_public_attempts_until_perfect", "confidence": 0.8},
                {"pattern": "anxiety_correlates_with_audience_size", "confidence": 0.9}
            ],
            optimization_target=OptimizationTarget.SOCIAL_POSITION,
            what_is_protected="Reputation, social standing, others' perception",
            what_is_risked="Feedback, improvement, timely shipping",
            predictions=[
                "User will delay public launch until 'perfect'",
                "Performance anxiety proportional to audience status",
                "Preference for private practice over public performance"
            ],
            intervention_logic="Reduce visibility of failure. Create safe practice contexts.",
            related_pressures=["STA-10", "IDE-28", "IDE-32"],
            conflicting_pressures=["STA-12"]
        )
        
        self.pressures["STA-12"] = StrategicPressure(
            pressure_id="STA-12",
            game_type=GameType.STATUS_SOCIAL,
            name="Effort displayed where recognition exists",
            description="Effort follows visibility of recognition. Invisible effort is deprioritized.",
            activation_triggers=["recognition_available", "credit_visible", "audience_engaged"],
            deactivation_triggers=["invisible_work", "no_feedback", "unrecognized_effort"],
            behavioral_signatures=[
                {"pattern": "effort_correlates_with_recognition_likelihood", "confidence": 0.85},
                {"pattern": "neglects_invisible_maintenance_work", "confidence": 0.8},
                {"pattern": "prefers_visible_tasks_over_important_invisible", "confidence": 0.85}
            ],
            optimization_target=OptimizationTarget.SOCIAL_POSITION,
            what_is_protected="Recognition, appreciation, status signals",
            what_is_risked="Important but invisible work, maintenance, sustainability",
            predictions=[
                "User will prioritize visible over important tasks",
                "Effort drops when recognition absent",
                "Neglect of maintenance and relationship work"
            ],
            intervention_logic="Make invisible work visible. Create recognition systems.",
            related_pressures=["STA-09", "IDE-29"],
            conflicting_pressures=["IDE-26"]
        )
        
        self.pressures["STA-13"] = StrategicPressure(
            pressure_id="STA-13",
            game_type=GameType.STATUS_SOCIAL,
            name="Competence signaling competes with competence building",
            description="Showing competence > building competence. Performance goals > learning goals.",
            activation_triggers=["evaluation_context", "performance_pressure", "status_visible"],
            deactivation_triggers=["learning_context", "mastery_focus", "private_improvement"],
            behavioral_signatures=[
                {"pattern": "chooses_easier_tasks_to_ensure_success", "confidence": 0.85},
                {"pattern": "avoids_challenging_learning_opportunities", "confidence": 0.8},
                {"pattern": "prefers_performance_over_improvement", "confidence": 0.85}
            ],
            optimization_target=OptimizationTarget.SOCIAL_POSITION,
            what_is_protected="Image of competence, others' respect, self-esteem",
            what_is_risked="Actual learning, growth, long-term competence",
            predictions=[
                "User will avoid challenging tasks that reveal incompetence",
                "Preference for tasks already good at over growth opportunities",
                "Hiding mistakes rather than learning from them"
            ],
            intervention_logic="Separate learning from evaluation. Create safe practice contexts.",
            related_pressures=["STA-10", "STA-11", "IDE-28"],
            conflicting_pressures=["EFF-19", "IDE-26"]
        )
        
        self.pressures["STA-14"] = StrategicPressure(
            pressure_id="STA-14",
            game_type=GameType.STATUS_SOCIAL,
            name="Association transfers status",
            description="Who you're near matters. Proximity to high-status = status.",
            activation_triggers=["high_status_present", "networking_opportunity", "visible_association"],
            deactivation_triggers=["isolated_context", "status_irrelevant", "intrinsic_focus"],
            behavioral_signatures=[
                {"pattern": "seeks_proximity_to_high_status", "confidence": 0.85},
                {"pattern": "values_connections_over_competence", "confidence": 0.75},
                {"pattern": "distances_from_low_status_associations", "confidence": 0.8}
            ],
            optimization_target=OptimizationTarget.SOCIAL_POSITION,
            what_is_protected="Social standing, perceived status, network value",
            what_is_risked="Authenticity, independent judgment, time",
            predictions=[
                "User will prioritize networking over skill development",
                "Choice of collaborators based on status not fit",
                "Avoidance of stigmatized groups or activities"
            ],
            intervention_logic="Leverage association effects. Connect to high-status reference groups.",
            related_pressures=["STA-09", "IDE-29"],
            conflicting_pressures=["IDE-26"]
        )
        
        self.pressures["STA-15"] = StrategicPressure(
            pressure_id="STA-15",
            game_type=GameType.STATUS_SOCIAL,
            name="Scarcity increases perceived value",
            description="Limited availability increases desirability independent of quality.",
            activation_triggers=["limited_supply", "exclusive_access", "competition_for_resource"],
            deactivation_triggers=["abundant_availability", "unrestricted_access", "commoditized"],
            behavioral_signatures=[
                {"pattern": "desire_increases_with_scarcity", "confidence": 0.9},
                {"pattern": "values_exclusive_over_superior", "confidence": 0.8},
                {"pattern": "competes_for_limited_slots", "confidence": 0.85}
            ],
            optimization_target=OptimizationTarget.SOCIAL_POSITION,
            what_is_protected="Exclusivity, specialness, relative advantage",
            what_is_risked="Optimal fit, rational evaluation, collaboration",
            predictions=[
                "User will pursue scarce opportunities over abundant better ones",
                "Interest increases when told 'limited spots'",
                "Retention of low-value exclusive memberships"
            ],
            intervention_logic="Create scarcity and exclusivity. Limit availability artificially.",
            related_pressures=["STA-09", "UNC-03"],
            conflicting_pressures=[]
        )
        
        self.pressures["STA-16"] = StrategicPressure(
            pressure_id="STA-16",
            game_type=GameType.STATUS_SOCIAL,
            name="Prefer being respected over being correct",
            description="Social approval > accuracy. Agreement > truth.",
            activation_triggers=["social_evaluation", "disagreement_risk", "reputation_visible"],
            deactivation_triggers=["private_decision", "truth_seeking_context", "anonymous_expression"],
            behavioral_signatures=[
                {"pattern": "agrees_with_group_despite_private_disagreement", "confidence": 0.85},
                {"pattern": "avoids_controversial_positions", "confidence": 0.8},
                {"pattern": "changes_opinion_with_social_pressure", "confidence": 0.75}
            ],
            optimization_target=OptimizationTarget.SOCIAL_POSITION,
            what_is_protected="Social harmony, belonging, acceptance",
            what_is_risked="Accuracy, integrity, independent thought",
            predictions=[
                "User will publicly agree with wrong consensus",
                "Avoidance of unpopular but correct positions",
                "Opinion shifts based on social context"
            ],
            intervention_logic="Make truth-seeking socially rewarded. Create safe dissent contexts.",
            related_pressures=["STA-09", "IDE-29", "IDE-32"],
            conflicting_pressures=["IDE-26"]
        )
        
        # ============================================================
        # C. EFFORT & ENERGY ALLOCATION GAMES
        # ============================================================
        
        self.pressures["EFF-17"] = StrategicPressure(
            pressure_id="EFF-17",
            game_type=GameType.EFFORT_ENERGY,
            name="Effort follows perceived return-on-energy",
            description="Energy is allocated like scarce currency based on expected ROI.",
            activation_triggers=["multiple_demands", "energy_depleted", "competing_priorities"],
            deactivation_triggers=["abundant_energy", "single_focus", "high_expected_roi"],
            behavioral_signatures=[
                {"pattern": "effort_correlates_with_expected_outcome", "confidence": 0.9},
                {"pattern": "abandons_low_roi_activities", "confidence": 0.85},
                {"pattern": "energy_budgeted_across_life_roles", "confidence": 0.8}
            ],
            optimization_target=OptimizationTarget.ENERGY_EFFICIENCY,
            what_is_protected="Energy reserves, sustainable effort, burnout prevention",
            what_is_risked="Long-term investments, delayed gratification, compound returns",
            predictions=[
                "User will abandon goals with delayed or uncertain payoff",
                "Effort shifts to domains with visible progress",
                "Energy conserved for identity-relevant domains only"
            ],
            intervention_logic="Increase perceived ROI. Make progress visible and immediate.",
            related_pressures=["EFF-18", "EFF-20", "EFF-21"],
            conflicting_pressures=["TIM-35"]
        )
        
        self.pressures["EFF-18"] = StrategicPressure(
            pressure_id="EFF-18",
            game_type=GameType.EFFORT_ENERGY,
            name="Starting cost dominates total cost",
            description="Initiation feels larger than execution. Getting started > continuing.",
            activation_triggers=["new_task", "initiation_required", "activation_energy_needed"],
            deactivation_triggers=["already_in_motion", "momentum_established", "habit_activated"],
            behavioral_signatures=[
                {"pattern": "procrastination_on_starting_not_continuing", "confidence": 0.9},
                {"pattern": "once_started_completes_easily", "confidence": 0.85},
                {"pattern": "overestimates_task_difficulty_before_start", "confidence": 0.8}
            ],
            optimization_target=OptimizationTarget.ENERGY_EFFICIENCY,
            what_is_protected="Energy, comfort, immediate ease",
            what_is_risked="Initiation, momentum, compound progress",
            predictions=[
                "User will delay starting but continue once begun",
                "Pre-task anxiety exceeds during-task difficulty",
                "Tasks completed easily once initiated"
            ],
            intervention_logic="Reduce starting cost. Make initiation trivial. Leverage momentum.",
            related_pressures=["EFF-17", "EFF-19", "UNC-04"],
            conflicting_pressures=[]
        )
        
        self.pressures["EFF-19"] = StrategicPressure(
            pressure_id="EFF-19",
            game_type=GameType.EFFORT_ENERGY,
            name="Progress visibility increases persistence",
            description="Seeing progress sustains effort. Invisible improvement discourages.",
            activation_triggers=["long_term_goal", "delayed_feedback", "invisible_progress"],
            deactivation_triggers=["visible_milestones", "immediate_feedback", "progress_tracking"],
            behavioral_signatures=[
                {"pattern": "quits_when_progress_invisible", "confidence": 0.85},
                {"pattern": "persistence_correlates_with_progress_visibility", "confidence": 0.9},
                {"pattern": "prefers_tasks_with_clear_progress_indicators", "confidence": 0.8}
            ],
            optimization_target=OptimizationTarget.ENERGY_EFFICIENCY,
            what_is_protected="Motivation, sense of efficacy, continued effort",
            what_is_risked="Deep expertise, mastery, long-term achievements",
            predictions=[
                "User will abandon goals without visible progress markers",
                "Effort increases when progress made visible",
                "Preference for games/activities with clear leveling"
            ],
            intervention_logic="Make progress visible. Create intermediate milestones. Track improvement.",
            related_pressures=["EFF-17", "EFF-20"],
            conflicting_pressures=["UNC-07"]
        )
        
        self.pressures["EFF-20"] = StrategicPressure(
            pressure_id="EFF-20",
            game_type=GameType.EFFORT_ENERGY,
            name="Invisible improvement discourages continuation",
            description="When improvement isn't visible, effort feels wasted.",
            activation_triggers=["skill_plateau", "unconscious_competence", "long_learning_curve"],
            deactivation_triggers=["benchmark_available", "comparison_point", "objective_assessment"],
            behavioral_signatures=[
                {"pattern": "quits_during_invisible_improvement_phase", "confidence": 0.8},
                {"pattern": "underestimates_own_improvement", "confidence": 0.85},
                {"pattern": "needs_external_validation_of_progress", "confidence": 0.8}
            ],
            optimization_target=OptimizationTarget.ENERGY_EFFICIENCY,
            what_is_protected":"Sense of efficacy, motivation, effort justification",
            what_is_risked="Mastery, expertise, breakthrough improvements",
            predictions=[
                "User will plateau prematurely when improvement invisible",
                "Abandonment during 'dip' before breakthrough",
                "Need for external feedback to sustain effort"
            ],
            intervention_logic="Make invisible improvement visible. Provide external benchmarks.",
            related_pressures=["EFF-19", "STA-12"],
            conflicting_pressures=[]
        )
        
        self.pressures["EFF-21"] = StrategicPressure(
            pressure_id="EFF-21",
            game_type=GameType.EFFORT_ENERGY,
            name="Willpower conserved for identity-relevant domains",
            description="Energy budgeted across life roles. Some domains get priority.",
            activation_triggers=["multiple_life_demands", "identity_salient_domain", "role_conflicts"],
            deactivation_triggers=["single_role", "abundant_resources", "aligned_domains"],
            behavioral_signatures=[
                {"pattern": "high_effort_in_identity_domains_low_elsewhere", "confidence": 0.85},
                {"pattern": "work_life_balance_as_energy_allocation", "confidence": 0.8},
                {"pattern": "sacrifices_non_identity_goals", "confidence": 0.8}
            ],
            optimization_target=OptimizationTarget.IDENTITY_STABILITY,
            what_is_protected":"Identity coherence, core self-concept, role integrity",
            what_is_risked="Balanced success, peripheral goals, holistic wellbeing",
            predictions=[
                "User will excel in 'who I am' domains, neglect others",
                "Energy allocation follows identity not importance",
                "Sacrifice of objectively important non-identity goals"
            ],
            intervention_logic="Connect goal to core identity. Frame as expression of self.",
            related_pressures=["IDE-25", "IDE-26", "EFF-17"],
            conflicting_pressures=[]
        )
        
        self.pressures["EFF-22"] = StrategicPressure(
            pressure_id="EFF-22",
            game_type=GameType.EFFORT_ENERGY,
            name="Automation replaces discipline when stable",
            description="Once stable, habits replace willpower. Environment > effort.",
            activation_triggers=["behavior_stabilized", "context_consistent", "repetition_sufficient"],
            deactivation_triggers=["novel_situation", "context_change", "habit_disrupted"],
            behavioral_signatures=[
                {"pattern": "automated_behavior_maintained_without_effort", "confidence": 0.9},
                {"pattern": "discipline_fails_when_habit_broken", "confidence": 0.85},
                {"pattern": "environment_design_more_effective_than_willpower", "confidence": 0.9}
            ],
            optimization_target=OptimizationTarget.ENERGY_EFFICIENCY,
            what_is_protected":"Sustainable behavior, reduced cognitive load, consistency",
            what_is_risked="Flexibility, adaptability, conscious choice",
            predictions=[
                "User will maintain automated behaviors without effort",
                "Relapse when habit disrupted (travel, disruption)",
                "Environment design more effective than motivation"
            ],
            intervention_logic="Build habits and environment. Reduce reliance on willpower.",
            related_pressures=["EFF-23", "IDE-38"],
            conflicting_pressures=["UNC-03"]
        )
        
        self.pressures["EFF-23"] = StrategicPressure(
            pressure_id="EFF-23",
            game_type=GameType.EFFORT_ENERGY,
            name="Friction redirects behavior more reliably than intention",
            description="Small friction changes behavior more than big motivation.",
            activation_triggers=["friction_present", "easy_alternative_available", "default_path_clear"],
            deactivation_triggers=["friction_removed", "path_of_least_resistance_aligned", "high_motivation"],
            behavioral_signatures=[
                {"pattern": "small_friction_prevents_desired_behavior", "confidence": 0.9},
                {"pattern": "behavior_follows_path_of_least_resistance", "confidence": 0.9},
                {"pattern": "intention_fails_against_friction", "confidence": 0.85}
            ],
            optimization_target=OptimizationTarget.ENERGY_EFFICIENCY,
            what_is_protected":"Ease, convenience, minimal effort",
            what_is_risked":"Intentional behavior, optimal choices, long-term goals",
            predictions=[
                "User will follow default path despite contrary intentions",
                "Small obstacles prevent important behaviors",
                "Friction reduction more effective than motivation increase"
            ],
            intervention_logic="Design friction, not motivation. Make desired path easiest.",
            related_pressures=["EFF-22", "EFF-18"],
            conflicting_pressures=[]
        )
        
        self.pressures["EFF-24"] = StrategicPressure(
            pressure_id="EFF-24",
            game_type=GameType.EFFORT_ENERGY,
            name="Energy budgeted across life roles not tasks",
            description="Energy allocated to roles (parent, worker, friend), not individual tasks.",
            activation_triggers=["role_conflicts", "multiple_identity_demands", "life_domain_tradeoffs"],
            deactivation_triggers=["single_role_focus", "aligned_roles", "abundant_energy"],
            behavioral_signatures=[
                {"pattern": "sacrifices_work_for_family_or_vice_versa", "confidence": 0.85},
                {"pattern": "energy_depletion_in_one_domain_affects_others", "confidence": 0.9},
                {"pattern": "role_switching_costs_visible", "confidence": 0.8}
            ],
            optimization_target=OptimizationTarget.IDENTITY_STABILITY,
            what_is_protected":"Role integrity, identity balance, holistic self",
            what_is_risked":"Optimal task allocation, domain excellence, efficiency",
            predictions=[
                "User will sacrifice work goal for family role or vice versa",
                "Energy in one domain depletes available for others",
                "Preference for integrated over segregated roles"
            ],
            intervention_logic="Respect role boundaries. Frame goals in role-consistent ways.",
            related_pressures=["EFF-21", "IDE-25"],
            conflicting_pressures=[]
        )
        
        # ============================================================
        # D. IDENTITY & SELF-COHERENCE GAMES
        # ============================================================
        
        self.pressures["IDE-25"] = StrategicPressure(
            pressure_id="IDE-25",
            game_type=GameType.IDENTITY_COHERENCE,
            name="Act to remain predictable to themselves",
            description="Consistency of self-image outranks optimization.",
            activation_triggers=["identity_relevant_choice", "self_concept_challenged", "narrative_threat"],
            deactivation_triggers=["identity_flexible", "self_concept_secure", "growth_mindset_active"],
            behavioral_signatures=[
                {"pattern": "maintains_suboptimal_consistent_behavior", "confidence": 0.85},
                {"pattern": "rejects_success_that_invalidates_past_self", "confidence": 0.75},
                {"pattern": "preference_for_predictable_over_optimal", "confidence": 0.8}
            ],
            optimization_target=OptimizationTarget.IDENTITY_STABILITY,
            what_is_protected":"Self-coherence, narrative continuity, psychological stability",
            what_is_risked":"Optimal outcomes, growth, adaptation",
            predictions=[
                "User will maintain 'consistent' behavior even when suboptimal",
                "Resistance to success that contradicts self-image",
                "Preference for familiar identity over improved outcomes"
            ],
            intervention_logic="Frame change as becoming more fully self, not different self.",
            related_pressures=["IDE-26", "IDE-27", "IDE-32"],
            conflicting_pressures=["UNC-03", "EFF-19"]
        )
        
        self.pressures["IDE-26"] = StrategicPressure(
            pressure_id="IDE-26",
            game_type=GameType.IDENTITY_COHERENCE,
            name="Contradicting identity feels like loss",
            description="Identity-inconsistent success feels like losing something valuable.",
            activation_triggers=["success_requires_identity_shift", "becoming_different_person", "values_conflict"],
            deactivation_triggers=["identity_aligned_success", "values_coherent_change", "self_expansion"],
            behavioral_signatures=[
                {"pattern": "sabotages_success_near_completion", "confidence": 0.75},
                {"pattern": "discomfort_with_praise_or_recognition", "confidence": 0.7},
                {"pattern": "returns_to_old_patterns_after_success", "confidence": 0.8}
            ],
            optimization_target=OptimizationTarget.IDENTITY_STABILITY,
            what_is_protected":"Familiar self-concept, psychological continuity, known identity",
            what_is_risked="Growth, success, new possibilities",
            predictions=[
                "User will self-sabotage when success threatens identity",
                "Discomfort with outcomes that contradict self-image",
                "Regression after periods of growth"
            ],
            intervention_logic="Create narrative continuity. Bridge old and new self.",
            related_pressures=["IDE-25", "IDE-27", "IDE-30"],
            conflicting_pressures=["STA-09", "EFF-19"]
        )
        
        self.pressures["IDE-27"] = StrategicPressure(
            pressure_id="IDE-27",
            game_type=GameType.IDENTITY_COHERENCE,
            name="Success that invalidates past self is resisted",
            description="Growth can feel like betrayal of who you were.",
            activation_triggers=["radical_improvement", "past_self_devalued", "identity_gap_grows"],
            deactivation_triggers=["incremental_growth", "past_self_honored", "continuity_preserved"],
            behavioral_signatures=[
                {"pattern": "stops_progress_when_gap_from_past_self_large", "confidence": 0.75},
                {"pattern": "nostalgia_for_old_self_sabotages_new_self", "confidence": 0.7},
                {"pattern": "minimizes_or_hides_improvements", "confidence": 0.7}
            ],
            optimization_target=OptimizationTarget.IDENTITY_STABILITY,
            what_is_protected":"Relationship with past self, narrative coherence, self-respect",
            what_is_risked="Transformation, radical improvement, new identity",
            predictions=[
                "User will plateau when improvement distances from past self",
                "Sabotage to maintain connection to former identity",
                "Downplaying achievements that contradict past self"
            ],
            intervention_logic="Honor past self. Frame growth as fulfillment of past potential.",
            related_pressures=["IDE-25", "IDE-26", "IDE-30"],
            conflicting_pressures=["STA-09"]
        )
        
        self.pressures["IDE-28"] = StrategicPressure(
            pressure_id="IDE-28",
            game_type=GameType.IDENTITY_COHERENCE,
            name="Defend narratives more than outcomes",
            description="Story coherence > results. Better to fail consistently than succeed inconsistently.",
            activation_triggers=["narrative_threatened", "explanation_challenged", "meaning_at_stake"],
            deactivation_triggers=["narrative_flexible", "outcome_focused", "pragmatic_mode"],
            behavioral_signatures=[
                {"pattern": "maintains_failing_approach_to_preserve_narrative", "confidence": 0.8},
                {"pattern": "rejects_solutions_that_contradict_story", "confidence": 0.75},
                {"pattern": "explains_failures_in_way_that_preserves_identity", "confidence": 0.85}
            ],
            optimization_target=OptimizationTarget.IDENTITY_STABILITY,
            what_is_protected":"Self-story, meaning system, psychological coherence",
            what_is_risked":"Optimal outcomes, learning, adaptation",
            predictions=[
                "User will persist with failing strategy to maintain narrative",
                "Rejection of solutions that don't fit self-story",
                "Rationalization preserves identity at cost of success"
            ],
            intervention_logic="Work within narrative. Reframe solutions to fit story.",
            related_pressures=["IDE-25", "IDE-32", "STA-16"],
            conflicting_pressures=["UNC-07"]
        )
        
        self.pressures["IDE-29"] = StrategicPressure(
            pressure_id="IDE-29",
            game_type=GameType.IDENTITY_COHERENCE,
            name="Beliefs chosen partly for social belonging",
            description="Identity includes group membership. Beliefs signal affiliation.",
            activation_triggers=["group_identity_salient", "beliefs_tested", "social_belonging_threatened"],
            deactivation_triggers=["isolated_context", "identity_independent", "private_decision"],
            behavioral_signatures=[
                {"pattern": "beliefs_align_with_group_despite_evidence", "confidence": 0.8},
                {"pattern": "defends_group_positions_not_personally_held", "confidence": 0.75},
                {"pattern": "belief_change_follows_group_change", "confidence": 0.8}
            ],
            optimization_target=OptimizationTarget.SOCIAL_POSITION,
            what_is_protected":"Group belonging, social identity, affiliation",
            what_is_risked":"Accuracy, independence, optimal beliefs",
            predictions=[
                "User will adopt beliefs of desired group",
                "Defense of group positions not personally important",
                "Belief rigidity proportional to group identity strength"
            ],
            intervention_logic="Leverage group identity. Frame change as group norm.",
            related_pressures=["STA-09", "STA-16", "IDE-25"],
            conflicting_pressures=["IDE-26"]
        )
        
        self.pressures["IDE-30"] = StrategicPressure(
            pressure_id="IDE-30",
            game_type=GameType.IDENTITY_COHERENCE,
            name="Radical change requires narrative continuity",
            description="Big transformations need story bridge from old to new self.",
            activation_triggers=["transformation_attempted", "identity_gap_large", "radical_change"],
            deactivation_triggers=["incremental_change", "continuity_preserved", "bridge_built"],
            behavioral_signatures=[
                {"pattern": "radical_change_attempts_fail_without_bridge", "confidence": 0.8},
                {"pattern": "successful_transformations_have_clear_origin_story", "confidence": 0.75},
                {"pattern": "reversion_when_narrative_coherence_lost", "confidence": 0.8}
            ],
            optimization_target=OptimizationTarget.IDENTITY_STABILITY,
            what_is_protected":"Narrative coherence, self-understanding, psychological continuity",
            what_is_risked="Transformation, rapid growth, new identity",
            predictions=[
                "User will fail at radical change without narrative bridge",
                "Successful transformations preceded by 'moment of clarity'",
                "Regression when can't explain change to self"
            ],
            intervention_logic="Build narrative bridge. Connect new self to old self meaningfully.",
            related_pressures=["IDE-25", "IDE-26", "IDE-27"],
            conflicting_pressures=["UNC-03"]
        )
        
        self.pressures["IDE-31"] = StrategicPressure(
            pressure_id="IDE-31",
            game_type=GameType.IDENTITY_COHERENCE,
            name="Small self-signals reshape identity gradually",
            description="Identity updates from small consistent signals, not big declarations.",
            activation_triggers=["small_action_taken", "identity_signal_sent", "consistent_pattern"],
            deactivation_triggers=["identity_declaration_only", "inconsistent_signals", "one_off_action"],
            behavioral_signatures=[
                {"pattern": "identity_shifts_from_repeated_small_actions", "confidence": 0.85},
                {"pattern": "declarations_without_action_fail", "confidence": 0.8},
                {"pattern": "gradual_identity_update_from_behavior", "confidence": 0.9}
            ],
            optimization_target=OptimizationTarget.IDENTITY_STABILITY,
            what_is_protected":"Authentic identity, earned self-concept, genuine change",
            what_is_risked="Rapid transformation, declared identity, premature commitment",
            predictions=[
                "User will become what they repeatedly do, not what they declare",
                "Small consistent actions shift identity more than big one-offs",
                "Identity lag behind behavior change"
            ],
            intervention_logic="Design small identity-consistent actions. Let identity follow behavior.",
            related_pressures=["IDE-25", "EFF-22", "IDE-38"],
            conflicting_pressures=["IDE-30"]
        )
        
        self.pressures["IDE-32"] = StrategicPressure(
            pressure_id="IDE-32",
            game_type=GameType.IDENTITY_COHERENCE,
            name="Embarrassment avoidance drives long-term choices",
            description="Fear of social embarrassment shapes major life decisions.",
            activation_triggers=["public_failure_risk", "embarrassment_possible", "social_evaluation"],
            deactivation_triggers=["private_context", "embarrassment_normed", "anonymity"],
            behavioral_signatures=[
                {"pattern": "avoids_attempts_due_to_embarrassment_fear", "confidence": 0.85},
                {"pattern": "choices_constrained_by_potential_shame", "confidence": 0.8},
                {"pattern": "prefers_invisible_failure_to_visible_attempt", "confidence": 0.8}
            ],
            optimization_target=OptimizationTarget.SOCIAL_POSITION,
            what_is_protected":"Social image, dignity, others' respect",
            what_is_risked="Growth, attempts, learning from failure",
            predictions=[
                "User will avoid domains where might look foolish",
                "Preference for safe mediocrity over risky excellence",
                "Life choices constrained by embarrassment avoidance"
            ],
            intervention_logic="Normalize failure. Create safe practice contexts. Reduce visibility.",
            related_pressures=["STA-11", "IDE-28", "STA-16"],
            conflicting_pressures=["IDE-26"]
        )
        
        # ============================================================
        # E. TIME HORIZON GAMES
        # ============================================================
        
        self.pressures["TIM-33"] = StrategicPressure(
            pressure_id="TIM-33",
            game_type=GameType.TIME_HORIZON,
            name="Future self treated as another person",
            description="Discounting of future benefits as if they go to someone else.",
            activation_triggers=["delayed_gratification", "long_term_benefit", "present_cost"],
            deactivation_triggers=["immediate_feedback", "future_self_vivid", "identity_continuity"],
            behavioral_signatures=[
                {"pattern": "hyperbolic_discounting", "confidence": 0.9},
                {"pattern": "sacrifices_future_for_present", "confidence": 0.85},
                {"pattern": "regret_from_past_self_sabotage", "confidence": 0.8}
            ],
            optimization_target=OptimizationTarget.EMOTIONAL_REGULATION,
            what_is_protected":"Present comfort, immediate gratification, current mood",
            what_is_risked":"Future wellbeing, long-term goals, compound benefits",
            predictions=[
                "User will choose smaller immediate over larger delayed reward",
                "Future benefits undervalued relative to present costs",
                "Regret from past decisions that sacrificed future"
            ],
            intervention_logic="Make future self vivid. Create continuity with future. Reduce delay.",
            related_pressures=["TIM-34", "TIM-35", "UNC-06"],
            conflicting_pressures=["IDE-25"]
        )
        
        self.pressures["TIM-34"] = StrategicPressure(
            pressure_id="TIM-34",
            game_type=GameType.TIME_HORIZON,
            name="Short feedback loops dominate motivation",
            description="Motivation requires near-term feedback. Long loops fail.",
            activation_triggers=["long_term_goal", "delayed_feedback", "distant_deadline"],
            deactivation_triggers=["immediate_results", "short_cycles", "visible_progress"],
            behavioral_signatures=[
                {"pattern": "abandons_long_loop_goals", "confidence": 0.85},
                {"pattern": "motivation_correlates_with_feedback_frequency", "confidence": 0.9},
                {"pattern": "prefers_tasks_with_quick_completion", "confidence": 0.8}
            ],
            optimization_target=OptimizationTarget.EMOTIONAL_REGULATION,
            what_is_protected":"Motivation, engagement, sense of progress",
            what_is_risked="Long-term achievements, mastery, compound returns",
            predictions=[
                "User will abandon goals with feedback loops > 1 week",
                "Motivation drops exponentially with delay to feedback",
                "Preference for short-term projects over long-term"
            ],
            intervention_logic="Create short feedback loops. Break long goals into short cycles.",
            related_pressures=["TIM-33", "EFF-19", "EFF-20"],
            conflicting_pressures=[]
        )
        
        self.pressures["TIM-35"] = StrategicPressure(
            pressure_id="TIM-35",
            game_type=GameType.TIME_HORIZON,
            name="Delayed rewards require trust in system",
            description="Future benefits only motivating if system seems reliable.",
            activation_triggers=["delayed_outcome", "system_uncertainty", "trust_required"],
            deactivation_triggers=["reliable_system", "guaranteed_outcome", "trust_established"],
            behavioral_signatures=[
                {"pattern": "discounts_rewards_when_system_unreliable", "confidence": 0.85},
                {"pattern": "prefers_certain_small_over_uncertain_large", "confidence": 0.8},
                {"pattern": "effort_correlates_with_trust_in_system", "confidence": 0.85}
            ],
            optimization_target=OptimizationTarget.UNCERTAINTY_REDUCTION,
            what_is_protected":"Risk mitigation, trust preservation, psychological safety",
            what_is_risked="Optimal expected value, long-term investments, growth",
            predictions=[
                "User will avoid investments in unreliable systems",
                "Effort proportional to perceived system reliability",
                "Preference for certain mediocrity over uncertain excellence"
            ],
            intervention_logic="Build system reliability. Create guarantees. Reduce trust requirements.",
            related_pressures=["UNC-01", "UNC-02", "TIM-33"],
            conflicting_pressures=[]
        )
        
        self.pressures["TIM-36"] = StrategicPressure(
            pressure_id="TIM-36",
            game_type=GameType.TIME_HORIZON,
            name="Abandon goals when causal link unclear",
            description="Need to see how effort leads to outcome. Opaque processes abandoned.",
            activation_triggers=["opaque_process", "unclear_causality", "hidden_mechanism"],
            deactivation_triggers=["clear_causality", "visible_mechanism", "effort_outcome_link"],
            behavioral_signatures=[
                {"pattern": "quits_when_causality_unclear", "confidence": 0.8},
                {"pattern": "effort_correlates_with_causal_clarity", "confidence": 0.85},
                {"pattern": "prefers_simple_clear_over_complex_opaque", "confidence": 0.8}
            ],
            optimization_target=OptimizationTarget.COGNITIVE_FLUENCY,
            what_is_protected":"Sense of control, understanding, predictability",
            what_is_risked":"Complex optimal strategies, expert approaches, nuance",
            predictions=[
                "User will abandon approaches they don't understand",
                "Effort drops when effort-outcome link unclear",
                "Preference for simple clear over complex optimal"
            ],
            intervention_logic="Make causality visible. Show effort-outcome link clearly.",
            related_pressures=["UNC-04", "EFF-19", "TIM-34"],
            conflicting_pressures=[]
        )
        
        self.pressures["TIM-37"] = StrategicPressure(
            pressure_id="TIM-37",
            game_type=GameType.TIME_HORIZON,
            name="Deadlines create value perception",
            description="Urgency creates value. Without deadline, tasks deprioritized.",
            activation_triggers=["no_deadline", "indefinite_timeline", "optional_task"],
            deactivation_triggers=["clear_deadline", "urgency_present", "time_pressure"],
            behavioral_signatures=[
                {"pattern": "tasks_without_deadlines_never_done", "confidence": 0.85},
                {"pattern": "effort_increases_near_deadline", "confidence": 0.9},
                {"pattern": "creates_artificial_deadlines_to_motivate", "confidence": 0.8}
            ],
            optimization_target=OptimizationTarget.EMOTIONAL_REGULATION,
            what_is_protected":"Urgency response, motivation, prioritization clarity",
            what_is_risked="Important but non-urgent goals, prevention, maintenance",
            predictions=[
                "User will neglect important goals without deadlines",
                "Procrastination until urgency triggers action",
                "Creation of artificial urgency to motivate"
            ],
            intervention_logic="Create deadlines. Make time scarcity visible. Add urgency.",
            related_pressures=["TIM-34", "EFF-18", "UNC-04"],
            conflicting_pressures=[]
        )
        
        self.pressures["TIM-38"] = StrategicPressure(
            pressure_id="TIM-38",
            game_type=GameType.TIME_HORIZON,
            name="Urgency substitutes for importance",
            description="Urgent tasks prioritized over important ones. Eisenhower matrix failure.",
            activation_triggers=["urgent_task_present", "important_task_absent_urgency", "competing_demands"],
            deactivation_triggers=["importance_salient", "urgency_managed", "prioritization_clear"],
            behavioral_signatures=[
                {"pattern": "urgent_over_important_consistently", "confidence": 0.9},
                {"pattern": "important_goals_neglected_for_urgent_trivia", "confidence": 0.85},
                {"pattern": "stress_from_urgency_not_importance", "confidence": 0.8}
            ],
            optimization_target=OptimizationTarget.EMOTIONAL_REGULATION,
            what_is_protected":"Immediate relief, stress reduction, crisis management",
            what_is_risked":"Important goals, long-term success, strategic progress",
            predictions=[
                "User will do urgent email over important project",
                "Important goals perpetually delayed by urgent demands",
                "Stress from volume not importance of tasks"
            ],
            intervention_logic="Create urgency for important. Protect from urgent. Prioritize visibly.",
            related_pressures=["TIM-37", "EFF-17", "UNC-04"],
            conflicting_pressures=[]
        )
        
        self.pressures["TIM-39"] = StrategicPressure(
            pressure_id="TIM-39",
            game_type=GameType.TIME_HORIZON,
            name="Progress resets perceived effort tolerance",
            description="After progress, willingness to exert effort renews. Not cumulative fatigue.",
            activation_triggers=["progress_made", "milestone_reached", "completion_celebrated"],
            deactivation_triggers=["stalled_progress", "plateau_reached", "effort_without_result"],
            behavioral_signatures=[
                {"pattern": "effort_resets_after_progress", "confidence": 0.8},
                {"pattern": "renewed_energy_after_milestone", "confidence": 0.85},
                {"pattern": "quits_when_progress_stalls", "confidence": 0.8}
            ],
            optimization_target=OptimizationTarget.EMOTIONAL_REGULATION,
            what_is_protected":"Motivation, sense of progress, continued engagement",
            what_is_risked="Sustained effort through plateaus, grit, persistence",
            predictions=[
                "User will exert more effort after visible progress",
                "Renewed motivation at milestones, not cumulative fatigue",
                "Abandonment when progress invisible despite effort"
            ],
            intervention_logic="Create frequent milestones. Celebrate progress. Make improvement visible.",
            related_pressures=["EFF-19", "EFF-20", "TIM-34"],
            conflicting_pressures=[]
        )
        
        self.pressures["TIM-40"] = StrategicPressure(
            pressure_id="TIM-40",
            game_type=GameType.TIME_HORIZON,
            name="Quit when identity payoff seems distant",
            description="Persistence requires sense that goal connects to who you're becoming.",
            activation_triggers=["identity_payoff_distant", "meaning_unclear", "becoming_uncertain"],
            deactivation_triggers=["identity_payoff_visible", "meaning_clear", "becoming_certain"],
            behavioral_signatures=[
                {"pattern": "quits_when_identity_connection_unclear", "confidence": 0.8},
                {"pattern": "persistence_correlates_with_identity_relevance", "confidence": 0.85},
                {"pattern": "abandons_goals_that_dont_fit_self_story", "confidence": 0.8}
            ],
            optimization_target=OptimizationTarget.IDENTITY_STABILITY,
            what_is_protected":"Sense of meaning, identity coherence, purposeful becoming",
            what_is_risked="Long-term goals, delayed identity payoffs, transformation",
            predictions=[
                "User will quit when can't see who they become",
                "Persistence proportional to identity relevance",
                "Abandonment of objectively valuable but identity-irrelevant goals"
            ],
            intervention_logic="Connect to identity. Show who they become. Make future self vivid.",
            related_pressures=["IDE-25", "TIM-33", "EFF-21"],
            conflicting_pressures=[]
        )
    
    def get_pressure(self, pressure_id: str) -> Optional[StrategicPressure]:
        """Retrieve pressure by ID."""
        return self.pressures.get(pressure_id)
    
    def get_pressures_by_game(self, game_type: GameType) -> List[StrategicPressure]:
        """Get all pressures in a game type."""
        return [p for p in self.pressures.values() if p.game_type == game_type]
    
    def detect_active_pressures(
        self,
        observation: Dict[str, Any],
        user_history: List[Dict],
        top_n: int = 5
    ) -> List[Tuple[StrategicPressure, float]]:
        """
        Detect which strategic pressures are likely active.
        Returns ranked list of (pressure, confidence).
        """
        candidates = []
        
        for pressure in self.pressures.values():
            score = self._score_pressure_activation(pressure, observation, user_history)
            if score > 0.4:  # Minimum threshold
                candidates.append((pressure, score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_n]
    
    def _score_pressure_activation(
        self,
        pressure: StrategicPressure,
        observation: Dict,
        user_history: List[Dict]
    ) -> float:
        """Score how likely this pressure is active."""
        scores = []
        
        # Check behavioral signatures
        for signature in pressure.behavioral_signatures:
            match = self._match_signature(signature["pattern"], observation, user_history)
            weighted = match * signature["confidence"]
            scores.append(weighted)
        
        if not scores:
            return 0.0
        
        # Take top 2 signatures
        scores.sort(reverse=True)
        top_scores = scores[:2]
        
        return sum(top_scores) / len(top_scores) if top_scores else 0.0
    
    def _match_signature(
        self,
        pattern: str,
        observation: Dict,
        user_history: List[Dict]
    ) -> float:
        """Match a behavioral signature against data."""
        # Simplified pattern matching - would be more sophisticated
        pattern_lower = pattern.lower()
        
        # Check observation
        obs_text = json.dumps(observation).lower()
        if pattern_lower in obs_text:
            return 0.8
        
        # Check history
        for hist in user_history[-5:]:
            hist_text = json.dumps(hist).lower()
            if pattern_lower in hist_text:
                return 0.6
        
        return 0.0
    
    def analyze_strategic_conflict(
        self,
        active_pressures: List[Tuple[StrategicPressure, float]]
    ) -> Dict[str, Any]:
        """
        Analyze conflicts between active pressures.
        Returns conflict analysis and resolution strategy.
        """
        if len(active_pressures) < 2:
            return {"conflict_detected": False}
        
        conflicts = []
        
        for i, (p1, s1) in enumerate(active_pressures):
            for p2, s2 in active_pressures[i+1:]:
                if p2.pressure_id in p1.conflicting_pressures:
                    conflicts.append({
                        "pressure_a": p1.pressure_id,
                        "pressure_b": p2.pressure_id,
                        "conflict_type": "direct",
                        "severity": min(s1, s2),
                        "resolution": f"Address {p1.pressure_id if s1 > s2 else p2.pressure_id} first"
                    })
        
        if not conflicts:
            return {"conflict_detected": False}
        
        # Find most severe conflict
        worst = max(conflicts, key=lambda x: x["severity"])
        
        return {
            "conflict_detected": True,
            "n_conflicts": len(conflicts),
            "most_severe": worst,
            "all_conflicts": conflicts,
            "recommendation": worst["resolution"]
        }
    
    def get_intervention_for_pressure(
        self,
        pressure_id: str,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get intervention strategy for a specific pressure."""
        pressure = self.pressures.get(pressure_id)
        if not pressure:
            return None
        
        return {
            "pressure": pressure,
            "intervention_logic": pressure.intervention_logic,
            "what_is_protected": pressure.what_is_protected,
            "what_is_risked": pressure.what_is_risked,
            "target_optimization": pressure.optimization_target.value,
            "related_pressures": [self.pressures.get(pid) for pid in pressure.related_pressures[:3]]
        }


# Global singleton
_hsm_instance: Optional[HumanStrategyModel] = None

def get_human_strategy_model() -> HumanStrategyModel:
    """Get or create global HSM instance."""
    global _hsm_instance
    if _hsm_instance is None:
        _hsm_instance = HumanStrategyModel()
    return _hsm_instance
'''

with open('/mnt/kimi/output/human_strategy_model.py', 'w') as f:
    f.write(code)

print("Created: human_strategy_model.py")
print(f"Size: {len(code)} bytes")
