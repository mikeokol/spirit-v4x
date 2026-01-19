from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    jwt_secret: str
    jwt_expire_minutes: int = 30 * 24 * 60  # 30 days
    db_url: str = "sqlite+aiosqlite:///./spirit.db"
    cors_origins: list[str] = ["http://localhost:3000"]
    log_level: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()
