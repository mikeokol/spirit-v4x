"""
Spirit Behavioral Research Agent - Configuration
Merged: Original settings + Data ingestion layer configuration
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    # ==========================================
    # ORIGINAL SETTINGS (Preserved)
    # ==========================================
    env: str = "dev"  # dev | prod
    db_url: str = "sqlite+aiosqlite:///./spirit.db"
    jwt_secret: str
    jwt_expire_minutes: int = 30 * 24 * 60  # 30 days
    cors_origins: list[str] = ["http://localhost:3000"]
    log_level: str = "INFO"
    openai_api_key: str | None = None
    
    # ==========================================
    # NEW: SUPABASE CONFIGURATION (Memory Layer)
    # ==========================================
    supabase_url: Optional[str] = None
    supabase_anon_key: Optional[str] = None
    supabase_service_key: Optional[str] = None  # Backend operations
    
    # ==========================================
    # NEW: DATA INGESTION CONFIGURATION
    # ==========================================
    
    # Stream Processing (Optional - for Kafka/Redis later)
    kafka_bootstrap_servers: str = "localhost:9092"
    redis_url: str = "redis://localhost:6379/0"
    
    # Edge Privacy Controls
    edge_aggregation_window_seconds: int = 60
    edge_max_raw_retention_hours: int = 24
    privacy_budget_daily: float = 1.0  # Epsilon for differential privacy
    
    # Data Quality Thresholds
    min_confidence_score: float = 0.6
    max_ingestion_latency_ms: int = 5000
    
    # Mobile Screen Time Specific
    screen_time_batch_size: int = 100
    app_usage_min_duration_seconds: int = 5  # Filter noise
    
    # JITAI Triggers
    vulnerability_detection_enabled: bool = True
    ema_cooldown_minutes: int = 30  # Prevent survey fatigue

    class Config:
        env_file = ".env"


# Global settings instance (your existing pattern preserved)
settings = Settings()


# ==========================================
# NEW: Ingestion-specific config accessor
# For use in new ingestion modules
# ==========================================

@lru_cache()
def get_ingestion_config():
    """
    Get ingestion-specific configuration.
    Cached for performance.
    """
    return {
        "supabase_url": settings.supabase_url,
        "supabase_key": settings.supabase_anon_key,
        "supabase_service_key": settings.supabase_service_key,
        "kafka_bootstrap_servers": settings.kafka_bootstrap_servers,
        "redis_url": settings.redis_url,
        "edge_aggregation_window_seconds": settings.edge_aggregation_window_seconds,
        "privacy_budget_daily": settings.privacy_budget_daily,
        "min_confidence_score": settings.min_confidence_score,
        "screen_time_batch_size": settings.screen_time_batch_size,
        "ema_cooldown_minutes": settings.ema_cooldown_minutes,
    }


# Feature flags for experimental data sources
class DataSourceFlags:
    """Feature flags for gradual rollout of data sources."""
    ANDROID_USAGE_STATS: bool = True
    IOS_SCREENTIME: bool = False
    KEYBOARD_DYNAMICS: bool = False  # Experimental
    NOTIFICATION_RESPONSE: bool = True


# Mobile data schema constants
class MobileDataSchema:
    """Schema definitions for mobile screen time ingestion."""
    
    # Raw events from mobile OS
    SCREEN_EVENT: str = "screen_state_change"  # ON/OFF/UNLOCK
    APP_TRANSITION: str = "app_session"        # App foreground/background
    INTERACTION: str = "interaction_event"     # Touch, type, scroll
    
    # Aggregated behavioral markers
    FOCUS_SESSION: str = "focus_session"       # Deep work detection
    CONTEXT_SWITCH: str = "context_switch"     # App switching pattern
    SEDENTARY_BOUT: str = "sedentary_usage"    # Extended passive consumption
