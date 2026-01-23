from datetime import date
from typing import Dict
from spirit.models import RealityAnchor
from spirit.schemas.reality_anchor import RealityAnchorSchema

def driver_math(anchor: RealityAnchorSchema) -> Dict[str, float]:
    """
    Convert anchor numbers into daily driver metrics.
    """
    return {
        "daily_leads": anchor.weekly_lead_target / 7,
        "daily_conversations": anchor.weekly_conversation_target / 7,
        "daily_closes": anchor.weekly_close_target / 7,
        "conversion_rate": anchor.weekly_close_target / max(anchor.weekly_conversation_target, 1),
        "avg_order_value": anchor.price,
    }


def bottleneck_pick(recent: list[dict], drivers: Dict[str, float]) -> str:
    """
    Deterministic rule: pick the biggest bottleneck for the next 7 days.
    recent: [{date, status}] for last 7 executions
    """
    misses = sum(1 for e in recent if e.get("status") == "miss")
    if misses >= 3:
        return "stabilize"

    # simple heuristic: lowest ratio vs target
    if drivers["daily_leads"] < drivers["daily_conversations"]:
        return "lead_gen"
    if drivers["daily_conversations"] < drivers["daily_closes"] * 10:
        return "conversation"
    return "close"
