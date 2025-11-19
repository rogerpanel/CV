"""
RobustIDPS.ai - Main FastAPI Application
=========================================

Production-grade web application for AI-powered intrusion detection.

Author: Roger Nick Anaedevha
"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "integrated_ai_ids"))

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.api.v1 import api_router
from app.core.events import startup_handler, shutdown_handler
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.security import SecurityHeadersMiddleware
from app.db.session import engine
from app.db.base import Base

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("🚀 Starting RobustIDPS.ai application...")
    await startup_handler()
    logger.info("✅ Application started successfully")

    yield

    # Shutdown
    logger.info("🛑 Shutting down RobustIDPS.ai application...")
    await shutdown_handler()
    logger.info("✅ Application shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="Advanced AI-Powered Intrusion Detection & Prevention System",
    version=settings.API_VERSION,
    docs_url="/docs" if settings.ENABLE_SWAGGER else None,
    redoc_url="/redoc" if settings.ENABLE_SWAGGER else None,
    openapi_url=f"/api/{settings.API_VERSION}/openapi.json",
    lifespan=lifespan
)

# ==========================================
# Middleware Configuration
# ==========================================

# CORS Middleware
if settings.ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security Headers
app.add_middleware(SecurityHeadersMiddleware)

# Rate Limiting
if settings.ENABLE_RATE_LIMITING:
    app.add_middleware(
        RateLimitMiddleware,
        rate_limit_per_minute=settings.RATE_LIMIT_PER_MINUTE,
        rate_limit_per_hour=settings.RATE_LIMIT_PER_HOUR
    )


# ==========================================
# Exception Handlers
# ==========================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Global exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.DEBUG else "An error occurred",
            "path": str(request.url.path)
        }
    )


# ==========================================
# API Routes
# ==========================================

# Include API router
app.include_router(
    api_router,
    prefix=f"/api/{settings.API_VERSION}"
)


# ==========================================
# Root Endpoints
# ==========================================

@app.get("/", tags=["root"])
async def root():
    """Root endpoint"""
    return {
        "name": settings.APP_NAME,
        "version": settings.API_VERSION,
        "status": "operational",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint for monitoring"""
    from app.services.health import get_system_health

    health_status = await get_system_health()

    status_code = (
        status.HTTP_200_OK
        if health_status["status"] == "healthy"
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )

    return JSONResponse(
        status_code=status_code,
        content=health_status
    )


@app.get("/metrics", tags=["monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    from app.services.metrics import get_prometheus_metrics

    metrics_data = await get_prometheus_metrics()
    return JSONResponse(content=metrics_data)


# ==========================================
# Static Files (Frontend)
# ==========================================

# Serve frontend build
frontend_build_path = Path(__file__).parent.parent.parent / "frontend" / "build"
if frontend_build_path.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(frontend_build_path), html=True),
        name="frontend"
    )


# ==========================================
# Application Entry Point
# ==========================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
        workers=4 if not settings.DEBUG else 1
    )
