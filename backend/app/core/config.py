"""Application configuration.

All settings are environment-overridable so the same codebase runs on SQLite
(zero-setup local demo) or PostgreSQL (docker-compose / production-like).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent
ARTIFACT_DIR = BACKEND_DIR / "ml_artifacts"
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- App ----
    APP_NAME: str = "IntelliBank"
    APP_ENV: str = "development"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = True

    # ---- Database ----
    # SQLite by default; set DATABASE_URL=postgresql+psycopg://user:pass@host/db for Postgres.
    DATABASE_URL: str = f"sqlite:///{(BACKEND_DIR / 'smartbank.db').as_posix()}"
    SQL_ECHO: bool = False

    # ---- Security ----
    JWT_SECRET: str = "dev-only-insecure-secret-change-me"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_TTL_MINUTES: int = 30
    REFRESH_TOKEN_TTL_DAYS: int = 7
    BCRYPT_ROUNDS: int = 12
    COOKIE_SECURE: bool = False  # True behind HTTPS
    COOKIE_SAMESITE: str = "lax"

    # ---- Redis (optional; falls back to in-process backend) ----
    REDIS_URL: str | None = None

    # ---- Rate limits: "times/seconds" ----
    RATE_LIMIT_LOGIN: str = "8/300"
    RATE_LIMIT_REGISTER: str = "5/3600"
    RATE_LIMIT_TRANSFER: str = "20/60"
    RATE_LIMIT_DEFAULT: str = "300/60"

    # ---- Banking rules ----
    LARGE_TXN_NOTIFY_THRESHOLD: float = 50_000.0
    DAILY_TRANSFER_LIMIT: float = 500_000.0
    MIN_SAVINGS_BALANCE: float = 500.0
    CURRENT_ACCOUNT_OVERDRAFT: float = 25_000.0

    # ---- ML ----
    ML_ARTIFACT_DIR: str = str(ARTIFACT_DIR)
    FRAUD_BLOCK_THRESHOLD: float = 0.90
    FRAUD_REVIEW_THRESHOLD: float = 0.55
    ANOMALY_ALERT_THRESHOLD: float = 0.62

    # ---- AI assistant ----
    # The assistant degrades to a deterministic intent router when no key is
    # set, so the feature stays demoable without credentials. Both paths run
    # through the identical, user-scoped tool layer.
    GEMINI_API_KEY: str | None = None
    # "gemini-flash-latest" rather than a pinned version: the pinned aliases
    # (gemini-2.0-flash, -lite) returned 429 on the free tier while the rolling
    # alias served fine, so this is the more reliable default.
    GEMINI_MODEL: str = "gemini-flash-latest"
    ASSISTANT_ENABLED: bool = True
    # Hard ceiling on tool-call rounds, so a confused model cannot loop forever
    # billing tokens.
    ASSISTANT_MAX_TOOL_ROUNDS: int = 4
    ASSISTANT_MAX_MESSAGE_CHARS: int = 1000
    # Turns of prior conversation replayed for context.
    ASSISTANT_HISTORY_TURNS: int = 8
    RATE_LIMIT_ASSISTANT: str = "30/300"

    # ---- CORS ----
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _split_origins(cls, v):
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @property
    def is_sqlite(self) -> bool:
        return self.DATABASE_URL.startswith("sqlite")

    @property
    def artifact_path(self) -> Path:
        p = Path(self.ML_ARTIFACT_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
