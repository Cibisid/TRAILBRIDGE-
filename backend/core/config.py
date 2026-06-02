"""
TrialBridge — Core Configuration
Loads all settings from environment variables with validation.
Uses Pydantic Settings for type safety and automatic .env loading.
"""

from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration for the entire TrialBridge application.
    All values are loaded from environment variables or .env file.
    Pydantic validates types at startup — the app will crash immediately
    if a required variable is missing or has the wrong type. This is intentional.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------
    # App
    # -------------------------
    app_name: str = "TrialBridge"
    app_env: Literal["development", "staging", "production"] = "development"
    app_version: str = "0.1.0"
    debug: bool = False
    secret_key: str = Field(min_length=32)

    # -------------------------
    # API
    # -------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    allowed_origins: list[str] = ["http://localhost:8501"]

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list) -> list[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    # -------------------------
    # Database
    # -------------------------
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "trialbridge"
    postgres_user: str = "trialbridge_user"
    postgres_password: str
    database_url: str  # Full async URL for SQLAlchemy

    # -------------------------
    # Redis
    # -------------------------
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_password: str = ""
    redis_url: str = "redis://redis:6379/0"

    # -------------------------
    # Celery
    # -------------------------
    celery_broker_url: str = "redis://redis:6379/1"
    celery_result_backend: str = "redis://redis:6379/2"

    # -------------------------
    # JWT Auth
    # -------------------------
    jwt_secret_key: str = Field(min_length=32)
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60
    jwt_refresh_token_expire_days: int = 7

    # -------------------------
    # Rate Limiting
    # -------------------------
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 500

    # -------------------------
    # AWS
    # -------------------------
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    aws_s3_bucket: str = "trialbridge-data"

    # -------------------------
    # Anthropic
    # -------------------------
    anthropic_api_key: str = ""

    # -------------------------
    # ClinicalTrials.gov
    # -------------------------
    ctgov_api_base_url: str = "https://clinicaltrials.gov/api/v2"
    ctgov_page_size: int = 100
    ctgov_max_trials: int = 50000

    # -------------------------
    # NLP Models
    # -------------------------
    scispacy_model: str = "en_core_sci_lg"
    biobert_model: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    embedding_dimension: int = 768

    # -------------------------
    # Monitoring
    # -------------------------
    sentry_dsn: str = ""
    prometheus_port: int = 9090

    # -------------------------
    # Logging
    # -------------------------
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"

    # -------------------------
    # Derived properties
    # -------------------------
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    Using lru_cache means we only read from .env once —
    not on every request. This is the standard FastAPI pattern.

    Usage:
        from backend.core.config import get_settings
        settings = get_settings()
    """
    return Settings()
