# spirit/strategies/library.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

Domain = Literal["business", "career", "fitness", "creator"]
Decision = Literal["stabilize", "continue", "pivot"]
StrategyKey = str

@dataclass(frozen=True)
class StrategyCard:
    key: StrategyKey
    domain: Domain
    name: str
    hypothesis: str
    signals: List[str]          # 1–3 keys
    review_days: int = 7

# --- Library (3 per domain) ---
STRATEGIES: Dict[StrategyKey, StrategyCard] = {
    # BUSINESS
    "biz_service_first": StrategyCard(
        key="biz_service_first",
        domain="business",
        name="Service-first",
        hypothesis="Direct outreach + clear offer produces paid pilots fastest.",
        signals=["outreach_sent", "replies", "calls_booked"],
    ),
    "biz_distribution_first": StrategyCard(
        key="biz_distribution_first",
        domain="business",
        name="Distribution-first",
        hypothesis="Publishing + deliberate distribution increases inbound demand.",
        signals=["posts_published", "distribution_actions", "subs_gained"],
    ),
    "biz_productize": StrategyCard(
        key="biz_productize",
        domain="business",
        name="Productize after proof",
        hypothesis="Turning a proven service into a product increases scale.",
        signals=["product_ships", "demo_requests", "conversions"],
    ),

    # CAREER
    "car_pipeline_first": StrategyCard(
        key="car_pipeline_first",
        domain="career",
        name="Pipeline-first",
        hypothesis="High application volume + fast iteration yields interviews.",
        signals=["applications_sent", "replies", "interviews"],
    ),
    "car_portfolio_first": StrategyCard(
        key="car_portfolio_first",
        domain="career",
        name="Portfolio-first",
        hypothesis="Shipping proof of work increases interview pull.",
        signals=["portfolio_ships", "applications_sent", "replies"],
    ),
    "car_outreach_first": StrategyCard(
        key="car_outreach_first",
        domain="career",
        name="Outreach-first",
        hypothesis="Warm outreach + referrals beats cold applying.",
        signals=["recruiter_messages_sent", "replies", "interviews"],
    ),

    # FITNESS
    "fit_adherence_first": StrategyCard(
        key="fit_adherence_first",
        domain="fitness",
        name="Adherence-first",
        hypothesis="Consistency beats intensity; adherence creates trend.",
        signals=["workouts_completed", "nutrition_days_on_plan"],
    ),
    "fit_training_first": StrategyCard(
        key="fit_training_first",
        domain="fitness",
        name="Training-first",
        hypothesis="Progressive overload drives body composition over time.",
        signals=["workouts_completed", "sessions_logged"],
    ),
    "fit_nutrition_first": StrategyCard(
        key="fit_nutrition_first",
        domain="fitness",
        name="Nutrition-first",
        hypothesis="A reliable nutrition system drives composition changes.",
        signals=["nutrition_days_on_plan", "protein_days"],
    ),

    # CREATOR
    "cre_publish_first": StrategyCard(
        key="cre_publish_first",
        domain="creator",
        name="Publish-first",
        hypothesis="Higher output increases surface area for winners.",
        signals=["posts_published", "distribution_actions"],
    ),
    "cre_quality_first": StrategyCard(
        key="cre_quality_first",
        domain="creator",
        name="Quality-first",
        hypothesis="Better retention increases recommendation.",
        signals=["avg_view_duration_sec", "subs_gained"],
    ),
    "cre_distribution_first": StrategyCard(
        key="cre_distribution_first",
        domain="creator",
        name="Distribution-first",
        hypothesis="Collaboration + deliberate distribution beats pure posting.",
        signals=["distribution_actions", "subs_gained", "impressions"],
    ),
}

DEFAULT_STRATEGY_BY_DOMAIN: Dict[Domain, StrategyKey] = {
    "business": "biz_service_first",
    "career": "car_pipeline_first",
    "fitness": "fit_adherence_first",
    "creator": "cre_publish_first",
}
