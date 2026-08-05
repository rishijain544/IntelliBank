"""FastAPI application assembly.

Notable choices:

* Models are warmed at startup so the first customer request does not pay the
  library initialisation cost that would blow the latency budget.
* Validation and integrity errors are mapped to stable JSON shapes, because the
  SPA needs predictable error payloads rather than framework defaults.
* Security headers are applied globally; ``Strict-Transport-Security`` is only
  emitted when TLS is configured, since sending it over plain HTTP is meaningless.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.api.routes import admin, assistant, auth, banking, loans, profile, risk
from app.core.cache import get_kv
from app.core.config import settings
from app.core.database import engine, init_db
from app.ml.inference import models_status, warm_up
from app.schemas import HealthResponse

logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("intellibank")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s (%s)", settings.APP_NAME, settings.APP_ENV)

    # Fail fast on unsafe production configuration. Refusing to boot is the
    # correct behaviour: a public API signing tokens with a known dev secret is
    # worse than a failed deploy, and the deploy log makes the cause obvious.
    if settings.is_production:
        fatal = settings.fatal_production_errors()
        if fatal:
            for problem in fatal:
                logger.critical("FATAL CONFIG: %s", problem)
            raise RuntimeError(
                "Refusing to start in production with unsafe configuration: "
                + " | ".join(fatal)
            )
        for warning in settings.production_warnings():
            logger.warning("CONFIG: %s", warning)

    init_db()
    logger.info("Database ready: %s", "sqlite" if settings.is_sqlite else "postgresql")
    logger.info("Cache backend: %s", get_kv().name)

    if settings.is_production and get_kv().name == "in-memory":
        # Per-process state silently breaks rate limiting and token revocation
        # once more than one worker or instance is running.
        logger.warning(
            "CONFIG: using the in-memory cache in production. Rate limits and token "
            "revocation are per-process and will not hold across instances. Set REDIS_URL."
        )

    status_map = warm_up()
    for name, loaded in status_map.items():
        logger.info("Model %s: %s", name, "loaded" if loaded else "NOT TRAINED")
    if not any(status_map.values()):
        logger.warning(
            "No ML artifacts found. Risk scoring will fall back to rules. "
            "Train them with: python -m app.ml.train --all"
        )
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="IntelliBank API",
    version="1.0.0",
    description=(
        "A **simulated** full-stack banking platform with three production-wired ML models: "
        "fraud detection (XGBoost), credit scoring (calibrated XGBoost) and anomaly "
        "detection (Isolation Forest).\n\n"
        "> **Educational project.** This system holds no real money and is not a licensed "
        "financial institution. All accounts, transactions and identity documents are simulated."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "X-Process-Time"],
)


@app.middleware("http")
async def security_and_timing(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000

    response.headers["X-Process-Time"] = f"{elapsed_ms:.2f}ms"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    # Only assert HSTS when the deployment is actually served over TLS.
    if settings.COOKIE_SECURE:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Flatten Pydantic errors into a field->message map the UI can render inline."""
    fields: dict[str, str] = {}
    for error in exc.errors():
        location = [str(p) for p in error["loc"] if p not in ("body", "query", "path")]
        fields[".".join(location) or "request"] = error["msg"]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation failed", "fields": fields},
    )


@app.exception_handler(IntegrityError)
async def integrity_handler(request: Request, exc: IntegrityError) -> JSONResponse:
    logger.warning("Integrity error on %s: %s", request.url.path, exc.orig)
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={"detail": "That operation conflicts with existing data"},
    )


@app.exception_handler(SQLAlchemyError)
async def database_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    # Log the real cause, return an opaque message: driver errors leak schema details.
    logger.error("Database error on %s: %s", request.url.path, exc, exc_info=settings.DEBUG)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "A database error occurred"},
    )


api_prefix = settings.API_PREFIX
for router in (
    auth.router,
    profile.router,
    banking.router,
    loans.router,
    risk.router,
    assistant.router,
    admin.router,
):
    app.include_router(router, prefix=api_prefix)


@app.get("/", tags=["meta"], summary="API root")
def root() -> dict[str, str]:
    return {
        "name": settings.APP_NAME,
        "version": "1.0.0",
        "docs": "/docs",
        "health": f"{api_prefix}/health",
        "disclaimer": "Simulated banking platform for educational use. Holds no real funds.",
    }


@app.get(f"{api_prefix}/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_state = "ok"
    except SQLAlchemyError:
        db_state = "unavailable"

    return HealthResponse(
        status="ok" if db_state == "ok" else "degraded",
        app=settings.APP_NAME,
        environment=settings.APP_ENV,
        database=db_state,
        cache_backend=get_kv().name,
        models={k: v["loaded"] for k, v in models_status().items()},
        timestamp=datetime.now(UTC),
    )


@app.get(f"{api_prefix}/ml/status", tags=["meta"], summary="Model metrics and versions")
def ml_status() -> dict:
    """Public model transparency endpoint, used by the marketing pages."""
    return models_status()
