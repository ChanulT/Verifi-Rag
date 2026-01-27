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
import uuid
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional, Dict, List, Literal

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings_manager, load_settings_from_env
from app.models import (
    ChatRequest,
    ChatResponse,
    HealthStatus,
)
from app.llm.openai_client import initialize_openai_client, OpenAIClient
from app.graph.workflow import RAGWorkflow, SearchServiceClient
from app.database.mysql import (
    DatabaseManager,
    SessionRepository,
    MessageRepository,
    set_database_manager,
)

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
# Global Instances
# =============================================================================

rag_workflow: Optional[RAGWorkflow] = None
search_client: Optional[SearchServiceClient] = None
llm_client: Optional[OpenAIClient] = None
db_manager: Optional[DatabaseManager] = None
session_repo: Optional[SessionRepository] = None
message_repo: Optional[MessageRepository] = None

# Track service start time
_start_time: Optional[datetime] = None


# =============================================================================
# Application Lifecycle
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle with database."""
    global rag_workflow, search_client, llm_client, db_manager
    global session_repo, message_repo, _start_time

    _start_time = datetime.now(timezone.utc)

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
            db_manager = DatabaseManager.from_url(mysql_url)
            await db_manager.initialize()
            set_database_manager(db_manager)

            # Create repositories
            session_repo = SessionRepository(db_manager)
            message_repo = MessageRepository(db_manager)

            logger.info("✓ MySQL database initialized")

        except Exception as e:
            logger.error(f"✗ Failed to initialize MySQL: {e}")
            logger.warning("Continuing without database - sessions won't persist!")
            db_manager = None
    else:
        logger.warning("MYSQL_URL not set - using in-memory sessions")
        db_manager = None

    # =========================================================================
    # Initialize LLM Client
    # =========================================================================
    try:
        if settings.llm_provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY not configured")

            llm_client = initialize_openai_client(
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
    rag_workflow = RAGWorkflow(
        llm=llm_client,
        search_client=search_client,
    )
    logger.info("✓ RAG Workflow initialized")

    logger.info("=" * 70)
    logger.info(f"SERVICE READY: http://{settings.host}:{settings.port}")
    logger.info(f"Database: {'MySQL connected' if db_manager else 'In-memory only'}")
    logger.info("=" * 70)

    yield

    # Shutdown
    logger.info("=" * 70)
    logger.info("CHAT SERVICE SHUTDOWN")

    if db_manager:
        await db_manager.close()
        logger.info("✓ MySQL connection closed")

    if search_client:
        await search_client.close()

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
    
    ### Citation Flow
    
    Each message stores which chunks were retrieved AND which were cited:
    - `message_chunks`: All chunks retrieved for that message
    - `was_cited`: Whether the chunk was actually used in the answer
    - `citation_number`: The [1], [2] number in the answer
    
    This allows your UI to show:
    - The answer with citation markers
    - "Sources used" section with cited chunks
    - "Show all retrieved" to see all context
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
# Helper Functions
# =============================================================================

async def get_or_create_session(session_id: Optional[str], user_id: str = None) -> str:
    """Get existing session or create new one."""
    if not session_repo:
        # In-memory fallback
        return session_id or str(uuid.uuid4())

    if session_id:
        existing = await session_repo.get_session(session_id)
        if existing:
            await session_repo.extend_session(session_id)
            return session_id

    # Create new session
    new_id = session_id or str(uuid.uuid4())
    await session_repo.create_session(
        session_id=new_id,
        user_id=user_id,
        expires_minutes=settings_manager.current.session_timeout_minutes,
    )
    return new_id


async def save_message_with_chunks(
    message_id: str,
    session_id: str,
    role: str,
    content: str,
    response: ChatResponse = None,
    retrieved_chunks: List[Dict] = None,
) -> None:
    """Save message and its retrieved chunks to database."""
    if not message_repo:
        return

    # Extract stats from response
    confidence = response.confidence if response else None
    status = response.status.value if response else "success"
    chunks_retrieved = len(retrieved_chunks) if retrieved_chunks else 0
    chunks_cited = len(response.citations) if response else 0
    processing_time = response.processing_time_ms if response else 0

    # Save the message
    await message_repo.create_message(
        message_id=message_id,
        session_id=session_id,
        role=role,
        content=content,
        confidence=confidence,
        status=status,
        chunks_retrieved=chunks_retrieved,
        chunks_cited=chunks_cited,
        processing_time_ms=processing_time,
    )

    # Save retrieved chunks (for assistant messages)
    if role == "assistant" and retrieved_chunks:
        cited_numbers = [c.number for c in (response.citations if response else [])]
        await message_repo.save_retrieved_chunks(
            message_id=message_id,
            chunks=retrieved_chunks,
            cited_numbers=cited_numbers,
        )


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with service info."""
    settings = settings_manager.current

    db_status = "connected" if db_manager and await db_manager.health_check() else "not connected"
    search_status = "healthy" if search_client and await search_client.health_check() else "unavailable"

    uptime = (datetime.now(timezone.utc) - _start_time).total_seconds() if _start_time else 0

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


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Send a message and get response with citations",
)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.

    Flow:
    1. Get/create session in database
    2. Save user message
    3. Get conversation history from database
    4. Run RAG workflow (retrieve chunks, generate answer)
    5. Save assistant message with retrieved chunks
    6. Return response with citations

    The retrieved chunks are stored PER MESSAGE, so your UI can
    show exactly which sources were used for each response.
    """
    if not rag_workflow:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service not initialized"
        )

    # Get or create session
    session_id = await get_or_create_session(request.session_id)

    # Generate message IDs
    user_message_id = str(uuid.uuid4())
    assistant_message_id = str(uuid.uuid4())

    # Save user message
    await save_message_with_chunks(
        message_id=user_message_id,
        session_id=session_id,
        role="user",
        content=request.message,
    )

    # Get conversation history from database
    chat_history = []
    if message_repo:
        chat_history = await message_repo.get_conversation_for_llm(
            session_id=session_id,
            max_turns=settings_manager.current.max_history_turns,
        )
        # Exclude current message (we just added it)
        if chat_history and chat_history[-1]["content"] == request.message:
            chat_history = chat_history[:-1]

    # Run RAG workflow
    try:
        response = await rag_workflow.run(
            query=request.message,
            session_id=session_id,
            message_id=assistant_message_id,
            document_filter=request.document_ids,
            chat_history=chat_history,
        )

        # Get the retrieved chunks for storage
        # These come from the workflow state
        retrieved_chunks = []
        if hasattr(rag_workflow, '_last_retrieved_chunks'):
            retrieved_chunks = rag_workflow._last_retrieved_chunks

        # Save assistant message with chunks
        await save_message_with_chunks(
            message_id=assistant_message_id,
            session_id=session_id,
            role="assistant",
            content=response.answer,
            response=response,
            retrieved_chunks=retrieved_chunks,
        )

        # Override session_id in response
        response.session_id = session_id
        response.message_id = assistant_message_id

        return response

    except Exception as e:
        logger.error(f"Chat failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process message: {str(e)}"
        )


@app.get(
    "/chat/sessions/{session_id}",
    tags=["Sessions"],
    summary="Get session details",
)
async def get_session(session_id: str):
    """Get session with message count."""
    if not session_repo:
        raise HTTPException(status_code=501, detail="Database not configured")

    session = await session_repo.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return session


@app.get(
    "/chat/sessions/{session_id}/messages",
    tags=["Sessions"],
    summary="Get all messages in a session",
)
async def get_session_messages(
    session_id: str,
    include_chunks: bool = False,
):
    """
    Get conversation history with optional chunk data.

    If include_chunks=True, includes cited chunks for each assistant message.
    Use this when you need to show sources for past messages.
    """
    if not message_repo:
        raise HTTPException(status_code=501, detail="Database not configured")

    messages = await message_repo.get_session_messages(
        session_id=session_id,
        include_chunks=include_chunks,
    )

    return {
        "session_id": session_id,
        "message_count": len(messages),
        "messages": messages,
    }


@app.get(
    "/chat/messages/{message_id}",
    tags=["Messages"],
    summary="Get message with full citation details",
)
async def get_message_details(message_id: str):
    """
    Get a single message with all its citation data.

    Returns:
    - The message content
    - citations: Chunks that were cited (with [1], [2] numbers)
    - all_retrieved_chunks: All chunks that were retrieved

    Use this when user clicks on a message to see full source details.
    """
    if not message_repo:
        raise HTTPException(status_code=501, detail="Database not configured")

    result = await message_repo.get_message_with_citations(message_id)
    if not result:
        raise HTTPException(status_code=404, detail="Message not found")

    return result


@app.get(
    "/chat/messages/{message_id}/chunks",
    tags=["Messages"],
    summary="Get chunks for a message",
)
async def get_message_chunks(
    message_id: str,
    cited_only: bool = True,
):
    """
    Get chunks retrieved for a specific message.

    Args:
        message_id: Message ID
        cited_only: If True, only returns chunks that were cited [1], [2]
                   If False, returns all retrieved chunks

    Use cited_only=True for the "Sources" section under an answer.
    Use cited_only=False for a "Show all context" feature.
    """
    if not message_repo:
        raise HTTPException(status_code=501, detail="Database not configured")

    chunks = await message_repo.get_message_chunks(
        message_id=message_id,
        cited_only=cited_only,
    )

    return {
        "message_id": message_id,
        "cited_only": cited_only,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


@app.post(
    "/chat/messages/{message_id}/feedback",
    tags=["Messages"],
    summary="Submit feedback for a message",
)
async def submit_feedback(
    message_id: str,
    rating: int = None,
    feedback_type: str = None,
    feedback_text: str = None,
):
    """
    Submit feedback for an assistant message.

    Args:
        rating: 1-5 star rating
        feedback_type: helpful, not_helpful, incorrect, incomplete
        feedback_text: Free-form feedback
    """
    if not message_repo:
        raise HTTPException(status_code=501, detail="Database not configured")

    await message_repo.add_feedback(
        message_id=message_id,
        rating=rating,
        feedback_type=feedback_type,
        feedback_text=feedback_text,
    )

    return {"status": "success", "message": "Feedback recorded"}


@app.delete(
    "/chat/sessions/{session_id}",
    tags=["Sessions"],
    summary="Delete a session",
)
async def delete_session(session_id: str):
    """Delete a session and all its messages."""
    if not session_repo:
        raise HTTPException(status_code=501, detail="Database not configured")

    await session_repo.delete_session(session_id)
    return {"status": "success", "message": "Session deleted"}


@app.get(
    "/health",
    response_model=HealthStatus,
    tags=["Health"],
)
async def health_check():
    """Check service health."""
    settings = settings_manager.current

    db_healthy = db_manager and await db_manager.health_check()
    llm_healthy = llm_client and await llm_client.health_check()
    search_healthy = search_client and await search_client.health_check()

    overall_health: Literal["healthy", "degraded", "unhealthy"] = "healthy"
    if not (db_healthy and llm_healthy and search_healthy):
        overall_health = "degraded"
    if not llm_healthy:
        overall_health = "unhealthy"

    uptime = (datetime.now(timezone.utc) - _start_time).total_seconds() if _start_time else 0

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


# =============================================================================
# Development Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    settings = load_settings_from_env()

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )