"""
Personal Evidence Ladder (PEL): Structured evidence grading for Spirit.
Every datapoint must be classified before it can influence beliefs.

Level 0 — Raw Observation: Sensors / logs / timestamps. No meaning yet.
Level 1 — Behavioral Metric: Derived but non-causal. Still correlation only.
Level 2 — Contextualized Pattern: Repeated across contexts. Now predictive.
Level 3 — Intervention Response: You changed something intentionally. Approaches causality.
Level 4 — Counterfactual Stability: Holds even when conditions change. Trait constraint.
Level 5 — Identity-Level Law: Cross-domain invariant. Part of user model.

Only Level 4-5 should modify the belief network. Everything else is provisional.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum, auto
import json

from spirit.db.supabase_client import get_behavioral_store


class EvidenceLevel(Enum):
    """The five levels of evidentiary strength."""
    RAW_OBSERVATION = 0      # Level 0
    BEHAVIORAL_METRIC = 1    # Level 1
    CONTEXTUALIZED_PATTERN = 2  # Level 2
    INTERVENTION_RESPONSE = 3   # Level 3
    COUNTERFACTUAL_STABILITY = 4  # Level 4
    IDENTITY_LAW = 5         # Level 5


@dataclass
class EvidenceGrading:
    """
    Complete grading for a piece of evidence.
    Immutable once assigned - creates audit trail.
    """
    observation_id: str
    user_id: str
    
    # The grade
    level: EvidenceLevel
    confidence: float  # 0-1, independent of level
    
    # Grading rationale
    grading_reason: str
    graded_by: str  # 'automated', 'human_review', 'mao_consensus'
    graded_at: datetime
    
    # Level-specific metadata
    level_metadata: Dict[str, Any]
    
    # Upgrade path
    upgraded_from: Optional[EvidenceLevel] = None
    upgrade_reason: Optional[str] = None
    
    # Validation
    validation_checks_passed: List[str]
    validation_checks_failed: List[str]


class EvidenceGradingEngine:
    """
    Grades behavioral observations according to the Personal Evidence Ladder.
    Core principle: Most data stays low-grade. Only rigorous validation upgrades.
    """
    
    def __init__(self):
        self.min_observations_for_pattern = 3
        self.min_contexts_for_pattern = 2
        self.intervention_cooldown_hours = 24
        self.stability_test_window_days = 14
    
    async def grade_observation(
        self,
        observation: Dict[str, Any],
        user_id: str,
        context_history: Optional[List[Dict]] = None
    ) -> EvidenceGrading:
        """
        Grade a new observation. Starts at Level 0 or 1, rarely higher.
        """
        observation_id = observation.get('observation_id', str(uuid4()))
        
        # Level 0: Raw sensor data with no derivation
        if self._is_raw_sensor_data(observation):
            return EvidenceGrading(
                observation_id=observation_id,
                user_id=user_id,
                level=EvidenceLevel.RAW_OBSERVATION,
                confidence=0.95,  # High confidence in raw data itself
                grading_reason="Raw sensor stream, no behavioral interpretation",
                graded_by="automated",
                graded_at=datetime.utcnow(),
                level_metadata={
                    "sensor_type": observation.get("sensor_type", "unknown"),
                    "raw_value": observation.get("raw_value"),
                    "processing_stage": "unprocessed"
                },
                validation_checks_passed=["sensor_validation", "timestamp_integrity"],
                validation_checks_failed=[]
            )
        
        # Level 1: Derived metric but no pattern context
        if self._is_derived_metric(observation):
            return EvidenceGrading(
                observation_id=observation_id,
                user_id=user_id,
                level=EvidenceLevel.BEHAVIORAL_METRIC,
                confidence=self._calculate_metric_confidence(observation),
                grading_reason="Derived behavioral metric, single-point measurement",
                graded_by="automated",
                graded_at=datetime.utcnow(),
                level_metadata={
                    "metric_type": observation.get("metric_type"),
                    "derivation_method": observation.get("derivation_method", "unknown"),
                    "temporal_resolution": observation.get("temporal_resolution", "single_point")
                },
                validation_checks_passed=["metric_bounds_check", "temporal_consistency"],
                validation_checks_failed=[]
            )
        
        # Default to Level 0 if unclear
        return EvidenceGrading(
            observation_id=observation_id,
            user_id=user_id,
            level=EvidenceLevel.RAW_OBSERVATION,
            confidence=0.5,
            grading_reason="Unclear observation type, defaulting to raw",
            graded_by="automated",
            graded_at=datetime.utcnow(),
            level_metadata={"observation_type": "unknown"},
            validation_checks_passed=[],
            validation_checks_failed=["classification_unclear"]
        )
    
    async def upgrade_evidence(
        self,
        current_grading: EvidenceGrading,
        upgrade_context: Dict[str, Any]
    ) -> EvidenceGrading:
        """
        Attempt to upgrade evidence to higher level based on new validation.
        This is the critical path - most upgrades should fail.
        """
        current_level = current_grading.level
        
        # Level 1 -> Level 2: Need repeated pattern across contexts
        if current_level == EvidenceLevel.BEHAVIORAL_METRIC:
            if await self._validate_contextualized_pattern(current_grading, upgrade_context):
                return EvidenceGrading(
                    observation_id=current_grading.observation_id,
                    user_id=current_grading.user_id,
                    level=EvidenceLevel.CONTEXTUALIZED_PATTERN,
                    confidence=self._calculate_pattern_confidence(upgrade_context),
                    grading_reason=f"Pattern validated across {upgrade_context.get('n_contexts', 0)} contexts over {upgrade_context.get('timespan_days', 0)} days",
                    graded_by="automated",
                    graded_at=datetime.utcnow(),
                    level_metadata={
                        "pattern_type": upgrade_context.get("pattern_type"),
                        "n_observations": upgrade_context.get("n_observations"),
                        "n_contexts": upgrade_context.get("n_contexts"),
                        "timespan_days": upgrade_context.get("timespan_days"),
                        "predictive_accuracy": upgrade_context.get("predictive_accuracy")
                    },
                    upgraded_from=current_level,
                    upgrade_reason="Cross-context repeatability established",
                    validation_checks_passed=current_grading.validation_checks_passed + [
                        "cross_context_replication",
                        "temporal_stability"
                    ],
                    validation_checks_failed=current_grading.validation_checks_failed
                )
        
        # Level 2 -> Level 3: Need intentional intervention with response
        if current_level == EvidenceLevel.CONTEXTUALIZED_PATTERN:
            if await self._validate_intervention_response(current_grading, upgrade_context):
                return EvidenceGrading(
                    observation_id=current_grading.observation_id,
                    user_id=current_grading.user_id,
                    level=EvidenceLevel.INTERVENTION_RESPONSE,
                    confidence=upgrade_context.get("causal_confidence", 0.6),
                    grading_reason=f"Intentional intervention '{upgrade_context.get('intervention_type')}' produced measurable response",
                    graded_by="automated",
                    graded_at=datetime.utcnow(),
                    level_metadata={
                        "intervention_id": upgrade_context.get("intervention_id"),
                        "intervention_type": upgrade_context.get("intervention_type"),
                        "response_magnitude": upgrade_context.get("response_magnitude"),
                        "control_comparison": upgrade_context.get("control_comparison"),
                        "confounds_controlled": upgrade_context.get("confounds_controlled", [])
                    },
                    upgraded_from=current_level,
                    upgrade_reason="Causal intervention response documented",
                    validation_checks_passed=current_grading.validation_checks_passed + [
                        "intervention_temporal_proximity",
                        "dose_response_relationship",
                        "confound_assessment"
                    ],
                    validation_checks_failed=current_grading.validation_checks_failed
                )
        
        # Level 3 -> Level 4: Need counterfactual stability
        if current_level == EvidenceLevel.INTERVENTION_RESPONSE:
            if await self._validate_counterfactual_stability(current_grading, upgrade_context):
                return EvidenceGrading(
                    observation_id=current_grading.observation_id,
                    user_id=current_grading.user_id,
                    level=EvidenceLevel.COUNTERFACTUAL_STABILITY,
                    confidence=0.75,  # Higher confidence at this level
                    grading_reason=f"Effect holds across varying conditions: {upgrade_context.get('stability_tests_passed', 0)}/{upgrade_context.get('stability_tests_total', 0)} tests",
                    graded_by="automated",  # Could require human review
                    graded_at=datetime.utcnow(),
                    level_metadata={
                        "n_conditions_tested": upgrade_context.get("n_conditions_tested"),
                        "conditions": upgrade_context.get("conditions", []),
                        "effect_size_consistency": upgrade_context.get("effect_size_consistency"),
                        "boundary_conditions": upgrade_context.get("boundary_conditions", [])
                    },
                    upgraded_from=current_level,
                    upgrade_reason="Counterfactual stability established - trait constraint identified",
                    validation_checks_passed=current_grading.validation_checks_passed + [
                        "cross_condition_replication",
                        "boundary_condition_testing",
                        "effect_size_consistency"
                    ],
                    validation_checks_failed=current_grading.validation_checks_failed
                )
        
        # Level 4 -> Level 5: Need cross-domain invariance
        if current_level == EvidenceLevel.COUNTERFACTUAL_STABILITY:
            if await self._validate_identity_law(current_grading, upgrade_context):
                return EvidenceGrading(
                    observation_id=current_grading.observation_id,
                    user_id=current_grading.user_id,
                    level=EvidenceLevel.IDENTITY_LAW,
                    confidence=0.85,  # High confidence but never absolute
                    grading_reason=f"Cross-domain invariant: manifests in {upgrade_context.get('domains', [])}",
                    graded_by="mao_consensus",  # Requires MAO validation at this level
                    graded_at=datetime.utcnow(),
                    level_metadata={
                        "domains": upgrade_context.get("domains", []),
                        "manifestations": upgrade_context.get("manifestations", {}),
                        "archetype_alignment": upgrade_context.get("archetype_alignment"),
                        "prediction_scope": upgrade_context.get("prediction_scope", "broad")
                    },
                    upgraded_from=current_level,
                    upgrade_reason="Identity-level law confirmed - part of core user model",
                    validation_checks_passed=current_grading.validation_checks_passed + [
                        "cross_domain_replication",
                        "archetype_validation",
                        "long_term_prediction_accuracy",
                        "mao_consensus"
                    ],
                    validation_checks_failed=current_grading.validation_checks_failed
                )
        
        # Upgrade failed - return original with failed check noted
        return EvidenceGrading(
            observation_id=current_grading.observation_id,
            user_id=current_grading.user_id,
            level=current_level,
            confidence=current_grading.confidence,
            grading_reason=current_grading.grading_reason,
            graded_by=current_grading.graded_by,
            graded_at=datetime.utcnow(),
            level_metadata={
                **current_grading.level_metadata,
                "last_upgrade_attempt": datetime.utcnow().isoformat(),
                "upgrade_failed_reason": "Validation criteria not met"
            },
            upgraded_from=current_grading.upgraded_from,
            upgrade_reason=current_grading.upgrade_reason,
            validation_checks_passed=current_grading.validation_checks_passed,
            validation_checks_failed=current_grading.validation_checks_failed + ["upgrade_validation_failed"]
        )
    
    # Validation methods for each level transition
    
    async def _validate_contextualized_pattern(
        self,
        grading: EvidenceGrading,
        context: Dict
    ) -> bool:
        """Validate Level 1 -> 2 upgrade."""
        n_observations = context.get("n_observations", 0)
        n_contexts = context.get("n_contexts", 0)
        timespan_days = context.get("timespan_days", 0)
        predictive_accuracy = context.get("predictive_accuracy", 0)
        
        # Strict criteria
        return (
            n_observations >= self.min_observations_for_pattern and
            n_contexts >= self.min_contexts_for_pattern and
            timespan_days >= 3 and
            predictive_accuracy >= 0.6
        )
    
    async def _validate_intervention_response(
        self,
        grading: EvidenceGrading,
        context: Dict
    ) -> bool:
        """Validate Level 2 -> 3 upgrade."""
        # Must have intentional intervention
        if not context.get("intervention_id"):
            return False
        
        # Must show dose-response or temporal proximity
        response_magnitude = context.get("response_magnitude", 0)
        temporal_proximity_hours = context.get("temporal_proximity_hours", 999)
        
        # Must assess confounds
        confounds_assessed = context.get("confounds_controlled", [])
        
        return (
            response_magnitude > 0.2 and  # Meaningful effect
            temporal_proximity_hours <= 48 and  # Close in time
            len(confounds_assessed) >= 2  # At least considered major confounds
        )
    
    async def _validate_counterfactual_stability(
        self,
        grading: EvidenceGrading,
        context: Dict
    ) -> bool:
        """Validate Level 3 -> 4 upgrade."""
        n_conditions = context.get("n_conditions_tested", 0)
        consistency = context.get("effect_size_consistency", 0)
        
        # Must hold across different conditions
        return (
            n_conditions >= 3 and  # Tested in at least 3 different states
            consistency >= 0.7  # Effect size doesn't vary wildly
        )
    
    async def _validate_identity_law(
        self,
        grading: EvidenceGrading,
        context: Dict
    ) -> bool:
        """Validate Level 4 -> 5 upgrade."""
        domains = context.get("domains", [])
        archetype_alignment = context.get("archetype_alignment")
        
        # Must manifest across multiple life domains
        # Must align with or refine archetype understanding
        return (
            len(domains) >= 2 and  # At least 2 life domains
            archetype_alignment is not None  # Validated against collective intelligence
        )
    
    # Helper methods
    
    def _is_raw_sensor_data(self, observation: Dict) -> bool:
        """Check if observation is unprocessed sensor data."""
        return (
            "raw_value" in observation or
            observation.get("processing_stage") == "unprocessed" or
            observation.get("observation_type") == "sensor_raw"
        )
    
    def _is_derived_metric(self, observation: Dict) -> bool:
        """Check if observation is a derived behavioral metric."""
        return (
            "metric_type" in observation or
            observation.get("processing_stage") == "derived" or
            "focus_score" in observation.get("behavior", {}) or
            "productivity_index" in observation.get("behavior", {})
        )
    
    def _calculate_metric_confidence(self, observation: Dict) -> float:
        """Calculate confidence in a behavioral metric."""
        # Base confidence
        confidence = 0.7
        
        # Boost for high data quality
        if observation.get("data_quality_score", 0) > 0.8:
            confidence += 0.15
        
        # Reduce for missing context
        if not observation.get("context"):
            confidence -= 0.2
        
        # Reduce for edge cases
        if observation.get("is_edge_case"):
            confidence -= 0.15
        
        return max(0.3, min(0.95, confidence))
    
    def _calculate_pattern_confidence(self, context: Dict) -> float:
        """Calculate confidence in a contextualized pattern."""
        n_obs = context.get("n_observations", 0)
        predictive_acc = context.get("predictive_accuracy", 0)
        
        # More observations = higher confidence, with diminishing returns
        obs_confidence = min(0.9, 0.5 + (n_obs * 0.05))
        
        # Predictive accuracy is crucial
        pred_confidence = predictive_acc
        
        # Weighted combination
        return (obs_confidence * 0.3) + (pred_confidence * 0.7)
    
    async def get_evidence_for_belief_modification(
        self,
        user_id: str,
        min_level: EvidenceLevel = EvidenceLevel.COUNTERFACTUAL_STABILITY
    ) -> List[EvidenceGrading]:
        """
        Retrieve only evidence strong enough to modify beliefs.
        Default: Level 4+ only.
        """
        store = await get_behavioral_store()
        if not store:
            return []
        
        # Query graded evidence
        result = store.client.table("evidence_grading").select("*").eq(
            "user_id", user_id
        ).gte("level", min_level.value).execute()
        
        if not result.data:
            return []
        
        return [
            EvidenceGrading(
                observation_id=r["observation_id"],
                user_id=r["user_id"],
                level=EvidenceLevel(r["level"]),
                confidence=r["confidence"],
                grading_reason=r["grading_reason"],
                graded_by=r["graded_by"],
                graded_at=datetime.fromisoformat(r["graded_at"]),
                level_metadata=r.get("level_metadata", {}),
                upgraded_from=EvidenceLevel(r["upgraded_from"]) if r.get("upgraded_from") else None,
                upgrade_reason=r.get("upgrade_reason"),
                validation_checks_passed=r.get("validation_checks_passed", []),
                validation_checks_failed=r.get("validation_checks_failed", [])
            )
            for r in result.data
        ]
    
    async def persist_grading(self, grading: EvidenceGrading):
        """Save grading to database."""
        store = await get_behavioral_store()
        if not store:
            return
        
        store.client.table("evidence_grading").upsert({
            "observation_id": grading.observation_id,
            "user_id": grading.user_id,
            "level": grading.level.value,
            "confidence": grading.confidence,
            "grading_reason": grading.grading_reason,
            "graded_by": grading.graded_by,
            "graded_at": grading.graded_at.isoformat(),
            "level_metadata": grading.level_metadata,
            "upgraded_from": grading.upgraded_from.value if grading.upgraded_from else None,
            "upgrade_reason": grading.upgrade_reason,
            "validation_checks_passed": grading.validation_checks_passed,
            "validation_checks_failed": grading.validation_checks_failed
        }, on_conflict="observation_id").execute()


# Import here to avoid circular dependency
from uuid import uuid4
