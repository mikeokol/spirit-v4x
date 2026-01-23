from datetime import date
from typing import Dict, Optional
from spirit.models import GoalProfile
from spirit.schemas.goal_profile import GoalProfileCreate

def complexity_score(goal_text: str) -> str:
    text = goal_text.lower()
    if any(k in text for k in ("weight", "habit", "run", "read", "meditate")):
        return "low"
    if any(k in text for k in ("career", "job", "skill", "promotion", "degree")):
        return "medium"
    return "high"  # default for revenue, startup, build, launch, etc.


def build_prompt(profile: Optional[GoalProfileCreate], goal_text: str, bottleneck: str) -> str:
    if profile is None:
        return (
            "The user has not yet completed calibration. "
            "Create ONE objective that helps the user finish the 3-question calibration."
        )

    drivers = {
        "daily_leads": profile.weekly_lead_target / 7 if hasattr(profile, "weekly_lead_target") else 0,
        "daily_conversations": profile.weekly_conversation_target / 7 if hasattr(profile, "weekly_conversation_target") else 0,
        "daily_closes": profile.weekly_close_target / 7 if hasattr(profile, "weekly_close_target") else 0,
    }

    return (
        f"Goal: {goal_text}\n"
        f"Calibration: {profile.time_budget_weekly}h/week, ${profile.money_budget_monthly or 0}/mo, "
        f"constraints: {','.join(profile.constraints)}, starting: {profile.starting_point}, "
        f"success: {profile.success_definition}, confidence: {profile.confidence or 'N/A'}.\n"
        f"Bottleneck: {bottleneck}. "
        f"Create ONE daily objective that respects time ≤ 60 min, difficulty ≤ 4, ≤3 micro-steps."
    )
