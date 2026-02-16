"""
Causal Inference Engine for Spirit.
Analyzes behavioral observations to discover causal relationships
and estimate intervention effects using n-of-1 trial methodology.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from uuid import UUID, uuid4
import statistics

from spirit.db.supabase_client import get_behavioral_store
from spirit.models.behavioral import (
    BehavioralObservation,
    UserCausalHypothesis,
    ObservationType
)


class CausalInferenceEngine:
    """
    Implements single-subject (n-of-1) causal analysis.
    Discovers: "Does X cause Y for this specific user?"
    
    Methods:
    - Temporal precedence (Granger causality light)
    - Association with time-lagged correlation
  - Interrupted time series for intervention effects
    """
    
    def __init__(self, user_id: UUID):
        self.user_id = user_id
        self.min_observations = 20  # Minimum for statistical power
        self.max_lag_hours = 24  # Look back up to 24 hours
    
    async def analyze_variable_pair(
        self,
        cause_var: str,  # e.g., "late_night_social_media"
        effect_var: str,  # e.g., "next_morning_focus_score"
        lag_hours: int = 8
    ) -> Optional[UserCausalHypothesis]:
        """
        Analyze if cause_var temporally precedes and predicts effect_var.
        """
        store = await get_behavioral_store()
        if not store:
            return None
        
        # Get observations with both variables
        observations = await store.get_user_observations(
            user_id=self.user_id,
            start_time=(datetime.utcnow() - timedelta(days=30)).isoformat(),
            limit=10000
        )
        
        if len(observations) < self.min_observations:
            return None
        
        # Extract time series for both variables
        cause_series = self._extract_series(observations, cause_var)
        effect_series = self._extract_series(observations, effect_var, lag_hours)
        
        if len(cause_series) < self.min_observations or len(effect_series) < self.min_observations:
            return None
        
        # Calculate time-lagged correlation
        correlation = self._lagged_correlation(cause_series, effect_series)
        
        # Estimate effect size using simple linear model
        effect_size = self._estimate_effect_size(cause_series, effect_series)
        
        # Calculate confidence interval via bootstrap
        ci_lower, ci_upper = self._bootstrap_ci(cause_series, effect_series)
        
        # Determine if effect is statistically meaningful
        is_significant = abs(correlation) > 0.3 and ci_lower * ci_upper > 0
        
        hypothesis = UserCausalHypothesis(
            hypothesis_id=uuid4(),
            user_id=self.user_id,
            cause_variable=cause_var,
            effect_variable=effect_var,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            n_observations=len(cause_series),
            model_type="lagged_correlation",
            last_validated_at=datetime.utcnow() if is_significant else None,
            falsified=not is_significant and len(cause_series) > 100
        )
        
        # Store hypothesis
        await store.update_causal_hypothesis(hypothesis)
        
        return hypothesis
    
    async def evaluate_intervention_effect(
        self,
        intervention_id: UUID,
        outcome_var: str,
        window_days: int = 7
    ) -> Dict[str, Any]:
        """
        Evaluate causal effect of an intervention using interrupted time series.
        
        Compares: 
        - Pre-intervention baseline
        - Post-intervention outcomes
        - Counterfactual (what would have happened without intervention)
        """
        store = await get_behavioral_store()
        if not store:
            return {"error": "store_unavailable"}
        
        # Get intervention delivery timestamp
        intervention_obs = await store.get_user_observations(
            user_id=self.user_id,
            observation_type=ObservationType.INTERVENTION_DELIVERED,
            limit=100
        )
        
        intervention_time = None
        for obs in intervention_obs:
            if obs.intervention_id == intervention_id:
                intervention_time = obs.timestamp
                break
        
        if not intervention_time:
            return {"error": "intervention_not_found"}
        
        # Get pre/post observations
        pre_window = intervention_time - timedelta(days=window_days)
        post_window = intervention_time + timedelta(days=window_days)
        
        all_obs = await store.get_user_observations(
            user_id=self.user_id,
            start_time=pre_window.isoformat(),
            end_time=post_window.isoformat(),
            limit=10000
        )
        
        pre_outcomes = []
        post_outcomes = []
        
        for obs in all_obs:
            val = self._extract_value(obs, outcome_var)
            if val is None:
                continue
            
            if obs.timestamp < intervention_time:
                pre_outcomes.append(val)
            else:
                post_outcomes.append(val)
        
        if len(pre_outcomes) < 5 or len(post_outcomes) < 5:
            return {
                "intervention_id": str(intervention_id),
                "effect_detected": False,
                "reason": "insufficient_data",
                "pre_n": len(pre_outcomes),
                "post_n": len(post_outcomes)
            }
        
        # Calculate means and trend
        pre_mean = statistics.mean(pre_outcomes)
        post_mean = statistics.mean(post_outcomes)
        pre_trend = self._calculate_trend(pre_outcomes)
        
        # Counterfactual: what would have happened without intervention
        counterfactual = pre_mean + (pre_trend * len(post_outcomes))
        
        # Actual vs counterfactual
        actual_change = post_mean - pre_mean
        causal_effect = post_mean - counterfactual
        
        # Confidence via simple t-test approximation
        pooled_std = statistics.stdev(pre_outcomes + post_outcomes) if len(pre_outcomes + post_outcomes) > 2 else 1.0
        se = pooled_std / (len(pre_outcomes) ** 0.5)
        
        return {
            "intervention_id": str(intervention_id),
            "outcome_variable": outcome_var,
            "effect_detected": abs(causal_effect) > (1.96 * se),
            "pre_mean": pre_mean,
            "post_mean": post_mean,
            "raw_change": actual_change,
            "causal_effect_estimate": causal_effect,
            "counterfactual_estimate": counterfactual,
            "confidence_interval": (
                causal_effect - 1.96 * se,
                causal_effect + 1.96 * se
            ),
            "n_pre": len(pre_outcomes),
            "n_post": len(post_outcomes),
            "method": "interrupted_time_series"
        }
    
    def _extract_series(
        self,
        observations: List[BehavioralObservation],
        variable: str,
        lag_hours: int = 0
    ) -> List[Tuple[datetime, float]]:
        """Extract time series for a variable from observations."""
        series = []
        
        for obs in observations:
            # Check context
            if variable in obs.context:
                val = obs.context[variable]
                if isinstance(val, (int, float)):
                    time = obs.timestamp + timedelta(hours=lag_hours)
                    series.append((time, float(val)))
                    continue
            
            # Check behavior
            if variable in obs.behavior:
                val = obs.behavior[variable]
                if isinstance(val, (int, float)):
                    time = obs.timestamp + timedelta(hours=lag_hours)
                    series.append((time, float(val)))
                    continue
            
            # Check outcome
            if obs.outcome and variable in obs.outcome:
                val = obs.outcome[variable]
                if isinstance(val, (int, float)):
                    time = obs.timestamp + timedelta(hours=lag_hours)
                    series.append((time, float(val)))
        
        return sorted(series, key=lambda x: x[0])
    
    def _extract_value(self, obs: BehavioralObservation, variable: str) -> Optional[float]:
        """Extract single value from observation."""
        for source in [obs.context, obs.behavior, obs.outcome or {}]:
            if variable in source:
                val = source[variable]
                if isinstance(val, (int, float)):
                    return float(val)
        return None
    
    def _lagged_correlation(
        self,
        series_a: List[Tuple[datetime, float]],
        series_b: List[Tuple[datetime, float]]
    ) -> float:
        """Calculate correlation between series_a and lagged series_b."""
        # Align series by time
        values_a = []
        values_b = []
        
        # Simple alignment: find closest match within 1 hour
        for time_a, val_a in series_a:
            closest_val_b = None
            closest_diff = timedelta(hours=1)
            
            for time_b, val_b in series_b:
                diff = abs((time_b - time_a).total_seconds())
                if diff < closest_diff.total_seconds():
                    closest_diff = timedelta(seconds=diff)
                    closest_val_b = val_b
            
            if closest_val_b is not None:
                values_a.append(val_a)
                values_b.append(closest_val_b)
        
        if len(values_a) < 5:
            return 0.0
        
        # Pearson correlation
        mean_a = statistics.mean(values_a)
        mean_b = statistics.mean(values_b)
        
        numerator = sum((a - mean_a) * (b - mean_b) for a, b in zip(values_a, values_b))
        denom_a = sum((a - mean_a) ** 2 for a in values_a) ** 0.5
        denom_b = sum((b - mean_b) ** 2 for b in values_b) ** 0.5
        
        if denom_a == 0 or denom_b == 0:
            return 0.0
        
        return numerator / (denom_a * denom_b)
    
    def _estimate_effect_size(
        self,
        cause_series: List[Tuple[datetime, float]],
        effect_series: List[Tuple[datetime, float]]
    ) -> float:
        """Estimate standardized effect size (Cohen's d approximation)."""
        cause_vals = [v for _, v in cause_series]
        effect_vals = [v for _, v in effect_series]
        
        if not cause_vals or not effect_vals:
            return 0.0
        
        # Simple regression coefficient as effect size
        mean_cause = statistics.mean(cause_vals)
        mean_effect = statistics.mean(effect_vals)
        
        # Standardized
        try:
            std_cause = statistics.stdev(cause_vals)
            if std_cause == 0:
                return 0.0
            return (mean_effect - statistics.median(effect_vals)) / std_cause
        except:
            return 0.0
    
    def _bootstrap_ci(
        self,
        series_a: List[Tuple[datetime, float]],
        series_b: List[Tuple[datetime, float]],
        n_bootstrap: int = 100
    ) -> Tuple[float, float]:
        """Calculate 95% confidence interval via bootstrap."""
        if len(series_a) < 10:
            return (-1.0, 1.0)  # Wide CI for insufficient data
        
        bootstrapped_effects = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = [hash(f"{i}_{_}") % len(series_a) for i in range(len(series_a))]
            sample_a = [series_a[i] for i in indices if i < len(series_a)]
            sample_b = [series_b[i] for i in indices if i < len(series_b)]
            
            if sample_a and sample_b:
                corr = self._lagged_correlation(sample_a, sample_b)
                bootstrapped_effects.append(corr)
        
        if not bootstrapped_effects:
            return (-1.0, 1.0)
        
        bootstrapped_effects.sort()
        lower_idx = int(0.025 * len(bootstrapped_effects))
        upper_idx = int(0.975 * len(bootstrapped_effects))
        
        return (bootstrapped_effects[lower_idx], bootstrapped_effects[upper_idx])
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate simple linear trend (slope)."""
        if len(values) < 2:
            return 0.0
        
        # Simple difference-based trend
        changes = [values[i+1] - values[i] for i in range(len(values)-1)]
        return statistics.mean(changes) if changes else 0.0


class ExperimentDesigner:
    """
    Designs micro-randomized trials based on causal hypotheses.
    """
    
    def __init__(self, user_id: UUID):
        self.user_id = user_id
    
    async def design_experiment(
        self,
        hypothesis: UserCausalHypothesis,
        intervention_options: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Design A/B test to validate causal hypothesis.
        """
        return {
            "experiment_id": str(uuid4()),
            "user_id": str(self.user_id),
            "hypothesis": {
                "cause": hypothesis.cause_variable,
                "effect": hypothesis.effect_variable
            },
            "design": "micro_randomized",
            "arms": ["control"] + [opt["name"] for opt in intervention_options],
            "randomization_probability": 0.5,
            "stratification": ["time_of_day", "day_of_week"],
            "min_observations_per_arm": 10,
            "max_daily_interventions": 3
        }
