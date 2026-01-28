"""
Chat Service - Main Application with MySQL Database

FastAPI microservice for RAG-based medical document Q&A.
Now with persistent storage for sessions, messages, and citations.

Database Tables:
- sessions: Conversation sessions
- messages: Individual messages
- message_chunks: Retrieved chunks per message (for citation display)

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
import sys
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional, Literal

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings_manager, load_settings_from_env
from app.models import HealthStatus
from app.llm.openai_client import initialize_openai_client
from app.graph.workflow import RAGWorkflow, SearchServiceClient
from app.database.mysql import (
    DatabaseManager,
    SessionRepository,
    MessageRepository,
    set_database_manager,
)
from app.routers import chat

# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# =============================================================================
# Application Lifecycle
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle with database using app.state."""

    # Initialize start time on state
    app.state.start_time = datetime.now(timezone.utc)

    # Default state values (to avoid AttributeErrors if initialization fails)
    app.state.rag_workflow = None
    app.state.search_client = None
    app.state.llm_client = None
    app.state.db_manager = None
    app.state.session_repo = None
    app.state.message_repo = None

    logger.info("=" * 70)
    logger.info("CHAT SERVICE STARTUP")
    logger.info("=" * 70)

    # Load settings
    settings = load_settings_from_env()
    settings_manager.update(settings)

    # =========================================================================
    # Initialize MySQL Database
    # =========================================================================
    mysql_url = os.getenv("MYSQL_URL")

    if mysql_url:
        try:
            logger.info("Initializing MySQL database...")
            # Initialize Manager
            db_manager = DatabaseManager.from_url(mysql_url)
            await db_manager.initialize()
            set_database_manager(db_manager)

            # Attach to state
            app.state.db_manager = db_manager
            app.state.session_repo = SessionRepository(db_manager)
            app.state.message_repo = MessageRepository(db_manager)

            logger.info("✓ MySQL database initialized")

        except Exception as e:
            logger.error(f"✗ Failed to initialize MySQL: {e}")
            logger.warning("Continuing without database - sessions won't persist!")
    else:
        logger.warning("MYSQL_URL not set - using in-memory sessions")

    # =========================================================================
    # Initialize LLM Client
    # =========================================================================
    try:
        if settings.llm_provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY not configured")

            app.state.llm_client = initialize_openai_client(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                temperature=settings.openai_temperature,
                max_tokens=settings.openai_max_tokens,
            )
            logger.info(f"✓ OpenAI client initialized: {settings.openai_model}")
        else:
            raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")

    except Exception as e:
        logger.error(f"✗ Failed to initialize LLM client: {e}")
        raise

    # =========================================================================
    # Initialize Search Client
    # =========================================================================
    try:
        search_client = SearchServiceClient(
            base_url=settings.search_service_url,
            timeout=settings.search_timeout_seconds,
        )
        app.state.search_client = search_client

        is_healthy = await search_client.health_check()
        if is_healthy:
            logger.info(f"✓ Search service available: {settings.search_service_url}")
        else:
            logger.warning(f"⚠ Search service not responding")

    except Exception as e:
        logger.error(f"✗ Failed to initialize search client: {e}")
        raise

    # =========================================================================
    # Initialize RAG Workflow
    # =========================================================================
    app.state.rag_workflow = RAGWorkflow(
        llm=app.state.llm_client,
        search_client=app.state.search_client,
    )
    logger.info("✓ RAG Workflow initialized")

    logger.info("=" * 70)
    logger.info(f"SERVICE READY: http://{settings.host}:{settings.port}")
    logger.info(f"Database: {'MySQL connected' if app.state.db_manager else 'In-memory only'}")
    logger.info("=" * 70)

    yield

    # Shutdown
    logger.info("=" * 70)
    logger.info("CHAT SERVICE SHUTDOWN")

    if app.state.db_manager:
        await app.state.db_manager.close()
        logger.info("✓ MySQL connection closed")

    if app.state.search_client:
        await app.state.search_client.close()

    logger.info("Shutdown complete")


# =============================================================================
# FastAPI Application
# =============================================================================

_initial_settings = load_settings_from_env()

app = FastAPI(
    title="Chat Service",
    description="""
    ## Medical Document Q&A with Verifiable RAG
    
    Now with persistent storage for:
    - Conversation sessions
    - Message history  
    - Retrieved chunks per message (for citation display)
    """,
    version=_initial_settings.service_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_initial_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Routes
# =============================================================================

app.include_router(chat.router)


@app.get("/", tags=["Root"])
async def root(request: Request):
    """Root endpoint with service info."""
    settings = settings_manager.current
    state = request.app.state

    db_status = "connected" if state.db_manager and await state.db_manager.health_check() else "not connected"
    search_status = "healthy" if state.search_client and await state.search_client.health_check() else "unavailable"

    uptime = (datetime.now(timezone.utc) - state.start_time).total_seconds() if hasattr(state, 'start_time') else 0

    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "status": "running",
        "uptime_seconds": round(uptime, 1),
        "database": {
            "type": "mysql",
            "status": db_status,
        },
        "llm": {
            "provider": settings.llm_provider,
            "model": settings.openai_model,
        },
        "search_service": {
            "url": settings.search_service_url,
            "status": search_status,
        },
    }


@app.get(
    "/health",
    response_model=HealthStatus,
    tags=["Health"],
)
async def health_check(request: Request):
    """Check service health."""
    settings = settings_manager.current
    state = request.app.state

    db_healthy = state.db_manager and await state.db_manager.health_check()
    llm_healthy = state.llm_client and await state.llm_client.health_check()
    search_healthy = state.search_client and await state.search_client.health_check()

    overall_health: Literal["healthy", "degraded", "unhealthy"] = "healthy"
    if not (db_healthy and llm_healthy and search_healthy):
        overall_health = "degraded"
    if not llm_healthy:
        overall_health = "unhealthy"

    uptime = (datetime.now(timezone.utc) - state.start_time).total_seconds() if hasattr(state, 'start_time') else 0

    return HealthStatus(
        service=settings.service_name,
        status=overall_health,
        version=settings.service_version,
        uptime_seconds=uptime,
        dependencies={
            "mysql": "healthy" if db_healthy else "unavailable",
            "llm": "healthy" if llm_healthy else "unavailable",
            "search": "healthy" if search_healthy else "unavailable",
        }
    )

if __name__ == "__main__":
    import uvicorn
    settings = load_settings_from_env()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )