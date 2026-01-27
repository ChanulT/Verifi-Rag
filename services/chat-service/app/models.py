"""
Pydantic models for Chat Service.

Contains:
- Request/Response schemas
- Citation models for UI display
- Conversation history models
- Internal domain models
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Literal
from enum import Enum
from pydantic import BaseModel, Field
from dataclasses import dataclass, field

__all__ = [
    # Enums
    "MessageRole",
    "ResponseStatus",

    # Citation models
    "Citation",
    "CitationDisplay",

    # Message models
    "ChatMessage",
    "ConversationHistory",

    # Request/Response
    "ChatRequest",
    "ChatResponse",
    "StreamChunk",

    # Health
    "HealthStatus",
]


# =============================================================================
# Enums
# =============================================================================

class MessageRole(str, Enum):
    """Chat message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ResponseStatus(str, Enum):
    """Response generation status."""
    SUCCESS = "success"
    NO_RELEVANT_CONTEXT = "no_relevant_context"
    LOW_CONFIDENCE = "low_confidence"
    ERROR = "error"


# =============================================================================
# Citation Models - Critical for UI Display
# =============================================================================

class Citation(BaseModel):
    """
    Citation linking answer to source chunk.

    This is what enables your UI to show:
    "📄 Source: report.pdf, Page 3, Section: Lab Results"
    """

    # Identifiers
    citation_id: str = Field(..., description="Unique citation ID (e.g., [1])")
    chunk_id: str = Field(..., description="Source chunk ID")
    document_id: str = Field(..., description="Source document ID")

    # Display info for UI
    source_file: str = Field(..., description="Filename for display")
    page_number: Optional[int] = Field(None, description="Page number if available")
    section_title: Optional[str] = Field(None, description="Section heading if available")

    # Content
    text_snippet: str = Field(..., description="Relevant text snippet (for tooltip)")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="How relevant this citation is")

    # Position in answer (for highlighting)
    start_index: Optional[int] = Field(None, description="Start position in answer text")
    end_index: Optional[int] = Field(None, description="End position in answer text")


class CitationDisplay(BaseModel):
    """
    Simplified citation for frontend display.

    Use this when you just need to show sources at the bottom of an answer.
    """

    number: int = Field(..., description="Citation number [1], [2], etc.")
    source: str = Field(..., description="Source filename")
    page: Optional[int] = Field(None, description="Page number")
    section: Optional[str] = Field(None, description="Section title")
    snippet: str = Field(..., description="Brief text preview")

    def to_display_string(self) -> str:
        """Format for UI display."""
        parts = [f"[{self.number}] {self.source}"]
        if self.page:
            parts.append(f"Page {self.page}")
        if self.section:
            parts.append(self.section)
        return ", ".join(parts)


# =============================================================================
# Message Models
# =============================================================================

class ChatMessage(BaseModel):
    """Single chat message."""

    role: MessageRole = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # For assistant messages
    citations: Optional[List[Citation]] = Field(None, description="Citations if assistant message")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")


class ConversationHistory(BaseModel):
    """Conversation history for a session."""

    session_id: str = Field(..., description="Session identifier")
    messages: List[ChatMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Metadata
    documents_in_scope: List[str] = Field(
        default_factory=list,
        description="Document IDs the user has uploaded/selected"
    )

    def add_message(self, role: MessageRole, content: str, **kwargs) -> ChatMessage:
        """Add a message to history."""
        message = ChatMessage(role=role, content=content, **kwargs)
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
        return message

    def get_recent_messages(self, n: int) -> List[ChatMessage]:
        """Get last n messages."""
        return self.messages[-n:] if n > 0 else []

    def to_llm_messages(self, max_turns: int = 10) -> List[Dict[str, str]]:
        """
        Convert to format suitable for LLM API.

        Returns list of {"role": "user/assistant", "content": "..."}
        """
        recent = self.get_recent_messages(max_turns * 2)  # 2 messages per turn
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in recent
            if msg.role in [MessageRole.USER, MessageRole.ASSISTANT]
        ]


# =============================================================================
# Request/Response Models
# =============================================================================

class ChatRequest(BaseModel):
    """
    Chat request from frontend.

    Example:
    ```json
    {
        "message": "What is my hemoglobin level?",
        "session_id": "abc123",
        "document_ids": ["doc-uuid-1", "doc-uuid-2"]
    }
    ```
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="User's question"
    )
    session_id: Optional[str] = Field(
        None,
        description="Session ID for conversation continuity"
    )
    document_ids: Optional[List[str]] = Field(
        None,
        description="Limit search to specific documents"
    )

    # Optional overrides
    top_k: Optional[int] = Field(
        None,
        ge=1,
        le=20,
        description="Override number of chunks to retrieve"
    )
    include_sources: bool = Field(
        default=True,
        description="Include source citations in response"
    )
    stream: bool = Field(
        default=False,
        description="Enable streaming response"
    )


class ChatResponse(BaseModel):
    """
    Chat response to frontend.

    Contains the answer, citations for UI display, and metadata.

    Example:
    ```json
    {
        "answer": "Based on your blood test report, your hemoglobin level is 14.2 g/dL [1], which is within the normal range for adults [2].",
        "citations": [
            {
                "number": 1,
                "source": "blood_test_2024.pdf",
                "page": 2,
                "section": "Complete Blood Count",
                "snippet": "Hemoglobin: 14.2 g/dL"
            }
        ],
        "status": "success",
        "confidence": 0.92
    }
    ```
    """

    # Main response
    answer: str = Field(..., description="Generated answer with inline citations like [1], [2]")

    # Citations for UI
    citations: List[CitationDisplay] = Field(
        default_factory=list,
        description="Citations for display below answer"
    )

    # Full citation details (for advanced UI features)
    citation_details: Optional[List[Citation]] = Field(
        None,
        description="Detailed citation info (chunk IDs, exact positions)"
    )

    # Status and confidence
    status: ResponseStatus = Field(..., description="Response status")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in answer")

    # Metadata
    session_id: str = Field(..., description="Session ID")
    message_id: str = Field(..., description="Unique message ID")

    # Debug/observability (optional)
    chunks_retrieved: int = Field(default=0, description="Number of chunks retrieved")
    chunks_used: int = Field(default=0, description="Number of chunks used in answer")
    processing_time_ms: float = Field(default=0.0, description="Total processing time")

    # For "no context" scenarios
    fallback_message: Optional[str] = Field(
        None,
        description="Message when no relevant context found"
    )


class StreamChunk(BaseModel):
    """
    Streaming response chunk.

    For real-time UI updates during generation.
    """

    type: Literal["token", "citation", "done", "error"] = Field(
        ...,
        description="Chunk type"
    )
    content: Optional[str] = Field(None, description="Token content")
    citation: Optional[CitationDisplay] = Field(None, description="Citation if type=citation")

    # Final data (when type=done)
    final_response: Optional[ChatResponse] = Field(None)
    error_message: Optional[str] = Field(None)


# =============================================================================
# Health Models
# =============================================================================

class HealthStatus(BaseModel):
    """Service health status."""

    service: str
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    uptime_seconds: float
    dependencies: Dict[str, str] = Field(default_factory=dict)


# =============================================================================
# Internal Models (Dataclasses for performance)
# =============================================================================

@dataclass
class RetrievedChunk:
    """
    Chunk retrieved from search service.

    Used internally during RAG processing.
    """
    chunk_id: str
    document_id: str
    content: str
    score: float

    # Metadata for citations
    filename: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    content_preview: str = ""

    def to_citation(self, citation_number: int) -> Citation:
        """Convert to Citation object."""
        return Citation(
            citation_id=f"[{citation_number}]",
            chunk_id=self.chunk_id,
            document_id=self.document_id,
            source_file=self.filename,
            page_number=self.page_number,
            section_title=self.section_title,
            text_snippet=self.content_preview,
            relevance_score=self.score,
        )

    def to_citation_display(self, citation_number: int) -> CitationDisplay:
        """Convert to CitationDisplay for UI."""
        return CitationDisplay(
            number=citation_number,
            source=self.filename,
            page=self.page_number,
            section=self.section_title,
            snippet=self.content_preview ,
        )


@dataclass
class GenerationResult:
    """Result from LLM generation."""

    answer: str
    raw_response: str
    tokens_used: int = 0
    finish_reason: str = "stop"

    # Extracted citations (numbers found in answer)
    cited_numbers: List[int] = field(default_factory=list)