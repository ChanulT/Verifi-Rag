"""
LangGraph State Definition for RAG Workflow - WITH TEMPORAL SUPPORT

NEW: Added TemporalFilter for year-based queries
"""

from typing import List, Optional, Dict, Any, Annotated, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from app.models import RetrievedChunk, Citation, CitationDisplay

__all__ = [
    "RAGState",
    "WorkflowStatus",
    "QueryType",
    "TemporalFilter",  # NEW!
]


class WorkflowStatus(str, Enum):
    """Current status of the RAG workflow."""
    PENDING = "pending"
    ANALYZING = "analyzing"
    RETRIEVING = "retrieving"
    GENERATING = "generating"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    NO_CONTEXT = "no_context"


class QueryType(str, Enum):
    """Type of user query (determined by analysis)."""
    FACTUAL = "factual"
    COMPARISON = "comparison"
    EXPLANATION = "explanation"
    SUMMARY = "summary"
    CLARIFICATION = "clarification"
    OUT_OF_SCOPE = "out_of_scope"


# =============================================================================
# NEW: Temporal Filter
# =============================================================================

@dataclass
class TemporalFilter:
    """
    Temporal filtering for time-based queries.

    Examples:
    - "last 3 years" → years=[2023, 2024, 2025]
    - "in 2024" → years=[2024]
    - "current WBC" → prefer_latest=True
    """
    years: List[int] = field(default_factory=list)
    prefer_latest: bool = False

    def has_filters(self) -> bool:
        """Check if any temporal filters are active."""
        return bool(self.years) or self.prefer_latest


@dataclass
class RAGState:
    """
    State object for the RAG workflow graph.

    NEW: Added temporal_filter for time-based queries.
    """

    # =========================================================================
    # Input (from request)
    # =========================================================================
    user_query: str = ""
    session_id: str = ""
    message_id: str = ""
    document_filter: List[str] = field(default_factory=list)

    # Conversation history (for context)
    chat_history: List[Dict[str, str]] = field(default_factory=list)

    # =========================================================================
    # Query Analysis Results
    # =========================================================================
    query_type: QueryType = QueryType.FACTUAL
    search_query: str = ""
    needs_retrieval: bool = True
    is_followup: bool = False

    # NEW: Temporal filtering
    temporal_filter: TemporalFilter = field(default_factory=TemporalFilter)

    # =========================================================================
    # Retrieval Results
    # =========================================================================
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)
    context_string: str = ""
    retrieval_scores: List[float] = field(default_factory=list)

    # =========================================================================
    # Generation Results
    # =========================================================================
    answer: str = ""
    raw_llm_response: str = ""
    cited_chunk_indices: List[int] = field(default_factory=list)

    # =========================================================================
    # Validation Results
    # =========================================================================
    is_grounded: bool = False
    confidence: float = 0.0
    validation_notes: List[str] = field(default_factory=list)

    # =========================================================================
    # Final Output
    # =========================================================================
    citations: List[Citation] = field(default_factory=list)
    citations_display: List[CitationDisplay] = field(default_factory=list)

    # =========================================================================
    # Workflow Metadata
    # =========================================================================
    status: WorkflowStatus = WorkflowStatus.PENDING
    error_message: Optional[str] = None

    # Timing
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    step_timings: Dict[str, float] = field(default_factory=dict)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def get_processing_time_ms(self) -> float:
        """Get total processing time in milliseconds."""
        if self.end_time:
            delta = self.end_time - self.start_time
            return delta.total_seconds() * 1000
        return 0.0

    def has_relevant_context(self) -> bool:
        """Check if we have relevant chunks."""
        return len(self.retrieved_chunks) > 0 and any(
            c.score > 0.3 for c in self.retrieved_chunks
        )

    def get_top_chunks(self, n: int = 5) -> List[RetrievedChunk]:
        """Get top n chunks by score."""
        sorted_chunks = sorted(
            self.retrieved_chunks,
            key=lambda c: c.score,
            reverse=True
        )
        return sorted_chunks[:n]

    def build_context_string(self, max_chunks: int = 5) -> str:
        """Build formatted context string for LLM prompt."""
        top_chunks = self.get_top_chunks(max_chunks)

        if not top_chunks:
            return "No relevant information found in the documents."

        context_parts = []

        for i, chunk in enumerate(top_chunks, 1):
            source_parts = [f"Source {i}: {chunk.filename}"]

            if chunk.page_number:
                source_parts.append(f"Page {chunk.page_number}")

            if chunk.section_title:
                source_parts.append(f"Section: {chunk.section_title}")

            source_ref = ", ".join(source_parts)
            context_parts.append(f"[{source_ref}]\n{chunk.content}")

        self.context_string = "\n\n---\n\n".join(context_parts)
        return self.context_string

    def build_citations_from_answer(self) -> None:
        """Extract citations from the generated answer."""
        import re

        citation_pattern = r'\[(\d+)\]'
        matches = re.findall(citation_pattern, self.answer)
        cited_numbers = sorted(set(int(m) for m in matches))

        self.cited_chunk_indices = cited_numbers
        self.citations = []
        self.citations_display = []

        top_chunks = self.get_top_chunks(len(self.retrieved_chunks))

        for num in cited_numbers:
            if 1 <= num <= len(top_chunks):
                chunk = top_chunks[num - 1]
                self.citations.append(chunk.to_citation(num))
                self.citations_display.append(chunk.to_citation_display(num))

    def calculate_confidence(self) -> float:
        """Calculate confidence score based on multiple factors."""
        if not self.retrieved_chunks:
            self.confidence = 0.0
            return 0.0

        top_scores = sorted(
            [c.score for c in self.retrieved_chunks],
            reverse=True
        )[:3]
        avg_score = sum(top_scores) / len(top_scores) if top_scores else 0

        chunk_boost = min(0.1 * len(self.retrieved_chunks), 0.2)
        citation_boost = 0.1 if self.cited_chunk_indices else 0
        grounding_factor = 1.0 if self.is_grounded else 0.7

        self.confidence = min(
            (avg_score + chunk_boost + citation_boost) * grounding_factor,
            1.0
        )

        return self.confidence

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/debugging."""
        return {
            "session_id": self.session_id,
            "message_id": self.message_id,
            "user_query": self.user_query,
            "query_type": self.query_type.value,
            "status": self.status.value,
            "chunks_retrieved": len(self.retrieved_chunks),
            "chunks_cited": len(self.cited_chunk_indices),
            "confidence": self.confidence,
            "is_grounded": self.is_grounded,
            "processing_time_ms": self.get_processing_time_ms(),
            "temporal_years": self.temporal_filter.years,  # NEW!
            "prefer_latest": self.temporal_filter.prefer_latest,  # NEW!
        }