"""
Embedding Service - Main Application

A production-ready FastAPI microservice for generating text embeddings
using sentence-transformers with GPU acceleration.

Features:
- GPU-accelerated embeddings
- Dynamic configuration via database
- Hot-reload capability
- Comprehensive metrics and health checks

Author: Embedding Service Team
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from config import settings_manager, load_settings_from_env
from service import model_manager, initialize_model, cleanup_model
from database import db_manager, SettingsRepository
from routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Exception Handlers
# =============================================================================

async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """
    Handle Pydantic validation errors with detailed messages.

    Args:
        request: FastAPI request
        exc: Validation exception

    Returns:
        Formatted error response
    """
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(x) for x in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })

    logger.warning(f"Validation error: {errors}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "detail": "Request validation failed",
            "errors": errors
        }
    )


async def general_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """
    Handle unexpected exceptions gracefully.

    Args:
        request: FastAPI request
        exc: Exception

    Returns:
        Generic error response
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred. Please try again later."
        }
    )


# =============================================================================
# Configuration Loading
# =============================================================================

async def load_initial_settings():
    """
    Load initial settings from environment and optional database.

    Priority: Database > Environment > Defaults

    Returns:
        Validated Settings instance
    """
    # Load base settings from environment
    base_settings = load_settings_from_env()

    # If database settings are disabled, use environment only
    if not base_settings.use_db_settings:
        logger.info("Database settings disabled - using environment configuration")
        return base_settings

    # Attempt to load from database
    try:
        logger.info("Initializing database connection...")
        await db_manager.initialize()

        repo = SettingsRepository(service_name="embedding-service")
        async with db_manager.session() as session:
            merged_settings = await repo.get_typed(session, defaults=base_settings)
            logger.info("✓ Settings loaded from database")
            return merged_settings

    except Exception as e:
        logger.warning(
            f"Database unavailable (use_db_settings=true): {e}\n"
            f"Falling back to environment settings.\n"
            f"Set EMBEDDING_USE_DB_SETTINGS=false to suppress this warning."
        )
        return base_settings


# =============================================================================
# Application Lifecycle Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle events.

    Startup:
    1. Load configuration (environment + optional database)
    2. Initialize embedding model
    3. Verify model is ready

    Shutdown:
    1. Cleanup model resources
    2. Close database connections
    """
    logger.info("=" * 70)
    logger.info("EMBEDDING SERVICE STARTUP")
    logger.info("=" * 70)

    # Load configuration
    try:
        settings = await load_initial_settings()
        settings_manager.update(settings)

        logger.info("Configuration loaded:")
        logger.info(f"  ├─ Model: {settings.model_name}")
        logger.info(f"  ├─ Device: {settings.device}")
        logger.info(f"  ├─ Batch Size: {settings.max_batch_size}")
        logger.info(f"  ├─ Normalize: {settings.normalize_embeddings}")
        logger.info(f"  └─ DB Settings: {'enabled' if settings.use_db_settings else 'disabled'}")

    except Exception as e:
        logger.error(f"✗ Configuration loading failed: {e}")
        sys.exit(1)

    # Initialize embedding model
    try:
        logger.info("Loading embedding model...")
        initialize_model(
            model_name=settings.model_name,
            device=settings.device,
            cache_dir=settings.model_cache_dir,
            normalize_embeddings=settings.normalize_embeddings,
            max_batch_size=settings.max_batch_size
        )
        logger.info("✓ Embedding model ready")

    except Exception as e:
        logger.error(f"✗ Model initialization failed: {e}")
        logger.error("Service cannot start without a valid embedding model")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info(f"SERVICE READY: http://{settings.host}:{settings.port}")
    logger.info(f"API Documentation: http://{settings.host}:{settings.port}/docs")
    logger.info("=" * 70)

    # Application is now running
    yield

    # Shutdown sequence
    logger.info("=" * 70)
    logger.info("EMBEDDING SERVICE SHUTDOWN")
    logger.info("=" * 70)

    # Cleanup model
    try:
        cleanup_model()
        logger.info("✓ Model resources released")
    except Exception as e:
        logger.error(f"Error during model cleanup: {e}")

    # Close database connections
    if db_manager.is_initialized:
        try:
            await db_manager.close()
            logger.info("✓ Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database: {e}")

    logger.info("=" * 70)
    logger.info("SHUTDOWN COMPLETE")
    logger.info("=" * 70)


# =============================================================================
# FastAPI Application
# =============================================================================

# Load initial settings for app metadata
_initial_settings = load_settings_from_env()

app = FastAPI(
    title="Embedding Service",
    description="""
    ## Production-Ready Embedding Microservice
    
    Generate high-quality text embeddings using state-of-the-art transformer models.
    
    ### Key Features
    - 🚀 **GPU Acceleration**: Automatic GPU/CPU offloading for optimal performance
    - 🔄 **Hot Reload**: Update models without service restart
    - ⚙️ **Dynamic Config**: Runtime configuration via database
    - 📊 **Observability**: Built-in metrics and health checks
    - 🔒 **Production Ready**: Thread-safe, validated, enterprise-grade
    
    ### Endpoints
    
    **Embeddings**
    - `POST /embed` - Generate embeddings for multiple texts
    - `POST /embed/single` - Generate embedding for single text
    
    **Monitoring**
    - `GET /health` - Service health status
    - `GET /metrics` - Performance metrics
    - `GET /model/info` - Model information
    
    **Configuration**
    - `GET /settings` - Current settings
    - `POST /settings` - Update setting
    - `POST /settings/bulk` - Bulk update
    - `DELETE /settings/{key}` - Delete setting
    
    ### Support
    For issues or questions, please contact the development team.
    """,
    version=_initial_settings.service_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=_initial_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router)


# =============================================================================
# Root Endpoint
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with service information.

    Returns:
        Service metadata and links
    """
    settings = settings_manager.current
    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "status": "running",
        "documentation": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


# =============================================================================
# Development Server
# =============================================================================

if __name__ == "__main__":
    import os
    import uvicorn

    settings = load_settings_from_env()

    # Determine environment
    is_development = os.getenv("ENVIRONMENT", "development") == "development"

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=is_development,
        workers=1,  # Important: sentence-transformers not fork-safe
        log_level="info",
        access_log=True,
    )