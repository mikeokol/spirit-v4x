from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    env: str = "dev"  # dev | prod
    db_url: str = "sqlite+aiosqlite:///./spirit.db"
    jwt_secret: str
    jwt_expire_minutes: int = 30 * 24 * 60  # 30 days
    cors_origins: list[str] = ["http://localhost:3000"]
    log_level: str = "INFO"
    openai_api_key: str | None = None

    class Config:
        env_file = ".env"

settings = Settings()
