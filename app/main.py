#!/usr/bin/env python3
"""
Lavix Vault - Main API
Encrypted file storage with AI-powered search and chat
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import psutil
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.agent.gateway import router as agent_gateway_router
from app.config import settings
from app.database import DatabaseUnavailableError, close_pool, get_connection
from app.db.readiness import require_schema_head
from app.routers import admin, ai_endpoints, auth, file_upload, files, folders
from app.security.redact import RedactingFilter

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
for _handler in logging.getLogger().handlers:
    _handler.addFilter(RedactingFilter())
logger = logging.getLogger(__name__)


async def startup_event() -> None:
    """Verify configuration, the migrated schema, then init model discovery."""

    settings.validate_config_file()
    settings.validate_runtime_secrets()
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    connection = get_connection()
    try:
        schema_status = require_schema_head(connection)
    finally:
        from app.database import put_connection
        put_connection(connection)
    logger.info("Database schema verified (version %s)", schema_status.current_version)

    from app.services.model_service import init_model_service

    init_model_service(settings.ollama_base_url)
    logger.info("Model service initialized (Ollama: %s)", settings.ollama_base_url)

    import asyncio

    from app.db.embedding_check import check_embedding_deployment

    # Fail closed when the configured embedding model no longer matches the
    # stored vectors (points at `app.cli reembed`); warn-only when Ollama
    # itself is unreachable.
    await asyncio.to_thread(check_embedding_deployment)


async def shutdown_event() -> None:
    close_pool()
    logger.info("Shutting down %s", settings.app_name)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await startup_event()
    try:
        yield
    finally:
        await shutdown_event()


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-Powered Encrypted File Storage with Semantic Search — built on a privacy-first architecture with user-controlled AI access and reversible intelligence.",
    lifespan=lifespan,
)


# Global exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.warning("HTTP %s — %s %s", exc.status_code, request.method, request.url.path)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error": exc.detail, "code": exc.status_code},
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    details = [
        {
            "location": list(error.get("loc", ())),
            "message": error.get("msg", "Invalid value"),
            "type": error.get("type", "value_error"),
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"error": "Validation error", "code": 422, "detail": details},
    )


@app.exception_handler(DatabaseUnavailableError)
async def database_unavailable_handler(request: Request, _exc: DatabaseUnavailableError):
    """Expose transient database loss as a bounded, retryable API failure."""

    logger.warning("Database unavailable — %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Service temporarily unavailable",
            "error": "Service temporarily unavailable",
            "code": 503,
        },
        headers={"Retry-After": "2"},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception(
        "Unhandled %s on %s %s",
        type(exc).__name__,
        request.method,
        request.url.path,
    )
    return JSONResponse(status_code=500, content={"error": "Internal server error", "code": 500})


# Gzip compression — compresses large JSON responses (file list ~950KB → ~80KB)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router imports are fail-fast: a partially registered API is not healthy.
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(file_upload.router, prefix="/api/files", tags=["Files"])
app.include_router(files.router, prefix="/api/files", tags=["Files"])
app.include_router(folders.router, prefix="/api", tags=["Folders"])
app.include_router(ai_endpoints.router, prefix="/api/ai", tags=["AI & Search"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])
app.include_router(
    agent_gateway_router,
    prefix="/api/internal/agent",
    tags=["Internal Agent Gateway"],
    include_in_schema=False,
)


@app.get("/")
@app.get("/api")
def read_root():
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "status": "healthy",
        "doc_url": "/docs",
        "api_root": "/api",
    }


@app.get("/health/live")
@app.get("/api/health/live")
def liveness():
    return {"status": "ok", "version": settings.app_version}


@app.get("/health")
@app.get("/api/health")
@app.get("/health/ready")
@app.get("/api/health/ready")
def readiness():
    vm = psutil.virtual_memory()
    available_percent = (vm.available / vm.total) * 100 if vm.total else 0
    low_memory = available_percent < 7
    payload = {
        "status": "low_memory" if low_memory else "healthy",
        "reason": "low_memory" if low_memory else None,
        "memory": {
            "available_bytes": vm.available,
            "total_bytes": vm.total,
            "available_percent": round(available_percent, 2),
            "low_memory_threshold_percent": 7,
        },
        "reranker": _reranker_health(),
    }
    return JSONResponse(status_code=503 if low_memory else 200, content=payload)


def _reranker_health() -> dict:
    """Active reranker wiring for operators (mode + model, no secrets)."""
    try:
        return {"mode": settings.rerank_mode, "model": settings.rerank_model}
    except Exception as exc:
        return {"mode": "unknown", "model": "", "error": str(exc)[:120]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8080)
