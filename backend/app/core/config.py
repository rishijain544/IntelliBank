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

    # ---- Demo data ----
    # When true, the app seeds demo users on startup IF the database has no
    # users yet. Deliberately opt-in and never destructive: it is a no-op the
    # moment any account exists, so it cannot wipe real data on a redeploy.
    #
    # This exists because Render's free tier provides no shell, so
    # `python manage.py seed` cannot be run against the deployed database.
    SEED_ON_STARTUP: bool = False
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
    def is_production(self) -> bool:
        return self.APP_ENV.lower() in {"production", "prod"}

    @property
    def artifact_path(self) -> Path:
        p = Path(self.ML_ARTIFACT_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def normalised_database_url(self) -> str:
        """Coerce a provider-supplied URL into the driver form SQLAlchemy needs.

        Render, Heroku and most managed Postgres providers hand out URLs starting
        with ``postgres://``, which SQLAlchemy 2 rejects outright, and
        ``postgresql://`` selects the psycopg2 driver that is not installed here.
        Rewriting to ``postgresql+psycopg://`` avoids a deploy-time crash that is
        otherwise reported as an opaque dialect error.
        """
        url = self.DATABASE_URL
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+psycopg://", 1)
        elif url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
        return url

    def production_warnings(self) -> list[str]:
        """Configuration that is unsafe once the app is publicly reachable.

        Returned rather than raised so the caller decides: the app refuses to
        start on genuinely dangerous settings, but merely logs the rest.
        """
        problems: list[str] = []

        if self.JWT_SECRET.startswith("dev-only") or len(self.JWT_SECRET) < 32:
            problems.append(
                "JWT_SECRET is the insecure default or shorter than 32 bytes. "
                "Anyone who knows it can mint valid tokens for any account. "
                "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(48))\""
            )
        if self.is_sqlite:
            problems.append(
                "DATABASE_URL points at SQLite. On a platform with an ephemeral "
                "filesystem every write is lost on restart. Use managed Postgres."
            )
        if self.DEBUG:
            problems.append("DEBUG is enabled; tracebacks and SQL may leak to clients.")
        if not self.COOKIE_SECURE:
            problems.append("COOKIE_SECURE is false, so HSTS is not sent. Enable it behind TLS.")
        if any(o.startswith("http://") for o in self.CORS_ORIGINS):
            problems.append(f"CORS_ORIGINS contains a plaintext http:// origin: {self.CORS_ORIGINS}")
        if "*" in self.CORS_ORIGINS:
            problems.append(
                "CORS_ORIGINS is a wildcard. Combined with credentialed requests this "
                "permits any site to call the API as a signed-in user."
            )
        return problems

    def fatal_production_errors(self) -> list[str]:
        """Settings so dangerous that booting anyway would be irresponsible."""
        fatal: list[str] = []
        if self.JWT_SECRET.startswith("dev-only") or len(self.JWT_SECRET) < 32:
            fatal.append("JWT_SECRET must be set to a strong unique value in production.")
        if "*" in self.CORS_ORIGINS:
            fatal.append("CORS_ORIGINS must list explicit origins, never '*'.")
        return fatal


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
