import uuid
import logging
from typing import List, Dict, Optional
from fastapi import APIRouter, HTTPException, status, Request

from app.models import (
    ChatRequest,
    ChatResponse,
    CitationDisplay,
    HealthStatus
)
from app.config import settings_manager
# We import AppState only for type checking if needed, but we use request.app.state dynamically
from app.state import AppState

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["Chat"],
    # You can add a prefix here if you want all URLs to start with /chat
    # But since your original app had mixed paths (some starting with /chat, some root),
    # we will keep the paths explicit in the decorators below to match your original URL structure.
)


# =============================================================================
# Helper Functions (Refactored to use Request state)
# =============================================================================

async def get_or_create_session(request: Request, session_id: Optional[str], user_id: str = None) -> str:
    """Get existing session or create new one using app state."""
    session_repo = request.app.state.session_repo

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
        request: Request,
        message_id: str,
        session_id: str,
        role: str,
        content: str,
        response: ChatResponse = None,
        retrieved_chunks: List[Dict] = None,
) -> None:
    """Save message and its retrieved chunks to database using app state."""
    message_repo = request.app.state.message_repo

    if not message_repo:
        return

    # Extract stats from response
    confidence = response.confidence if response else None
    status_val = response.status.value if response else "success"
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
        status=status_val,
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

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message and get response with citations",
)
async def chat(request: Request, chat_request: ChatRequest):
    """
    Main chat endpoint.
    """
    rag_workflow = request.app.state.rag_workflow
    message_repo = request.app.state.message_repo

    if not rag_workflow:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service not initialized"
        )

    # Get or create session
    session_id = await get_or_create_session(request, chat_request.session_id)

    # Generate message IDs
    user_message_id = str(uuid.uuid4())
    assistant_message_id = str(uuid.uuid4())

    # Save user message
    await save_message_with_chunks(
        request=request,
        message_id=user_message_id,
        session_id=session_id,
        role="user",
        content=chat_request.message,
    )

    # Get conversation history from database
    chat_history = []
    if message_repo:
        chat_history = await message_repo.get_conversation_for_llm(
            session_id=session_id,
            max_turns=settings_manager.current.max_history_turns,
        )
        # Exclude current message (we just added it)
        if chat_history and chat_history[-1]["content"] == chat_request.message:
            chat_history = chat_history[:-1]

    # Run RAG workflow
    try:
        response = await rag_workflow.run(
            query=chat_request.message,
            session_id=session_id,
            message_id=assistant_message_id,
            document_filter=chat_request.document_ids,
            chat_history=chat_history,
        )

        # Get the retrieved chunks for storage
        retrieved_chunks = []
        if hasattr(rag_workflow, '_last_retrieved_chunks'):
            retrieved_chunks = rag_workflow._last_retrieved_chunks

        # Save assistant message with chunks
        await save_message_with_chunks(
            request=request,
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


@router.get(
    "/chat/sessions/{session_id}",
    tags=["Sessions"],
    summary="Get session details",
)
async def get_session(request: Request, session_id: str):
    """Get session with message count."""
    session_repo = request.app.state.session_repo

    if not session_repo:
        raise HTTPException(status_code=501, detail="Database not configured")

    session = await session_repo.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return session


@router.get(
    "/chat/sessions/{session_id}/messages",
    tags=["Sessions"],
    summary="Get all messages in a session",
)
async def get_session_messages(
        request: Request,
        session_id: str,
        include_chunks: bool = False,
):
    """Get conversation history with optional chunk data."""
    message_repo = request.app.state.message_repo

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


@router.get(
    "/chat/messages/{message_id}",
    tags=["Messages"],
    summary="Get message with full citation details",
)
async def get_message_details(request: Request, message_id: str):
    """Get a single message with all its citation data."""
    message_repo = request.app.state.message_repo

    if not message_repo:
        raise HTTPException(status_code=501, detail="Database not configured")

    result = await message_repo.get_message_with_citations(message_id)
    if not result:
        raise HTTPException(status_code=404, detail="Message not found")

    return result


@router.get(
    "/chat/messages/{message_id}/chunks",
    tags=["Messages"],
    summary="Get chunks for a message",
)
async def get_message_chunks(
        request: Request,
        message_id: str,
        cited_only: bool = True,
):
    """Get chunks retrieved for a specific message."""
    message_repo = request.app.state.message_repo

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


@router.post(
    "/chat/messages/{message_id}/feedback",
    tags=["Messages"],
    summary="Submit feedback for a message",
)
async def submit_feedback(
        request: Request,
        message_id: str,
        rating: int = None,
        feedback_type: str = None,
        feedback_text: str = None,
):
    """Submit feedback for an assistant message."""
    message_repo = request.app.state.message_repo

    if not message_repo:
        raise HTTPException(status_code=501, detail="Database not configured")

    await message_repo.add_feedback(
        message_id=message_id,
        rating=rating,
        feedback_type=feedback_type,
        feedback_text=feedback_text,
    )

    return {"status": "success", "message": "Feedback recorded"}


@router.delete(
    "/chat/sessions/{session_id}",
    tags=["Sessions"],
    summary="Delete a session",
)
async def delete_session(request: Request, session_id: str):
    """Delete a session and all its messages."""
    session_repo = request.app.state.session_repo

    if not session_repo:
        raise HTTPException(status_code=501, detail="Database not configured")

    await session_repo.delete_session(session_id)
    return {"status": "success", "message": "Session deleted"}