"""
LangGraph State Definition for RAG Workflow.

Defines the state that flows through the RAG graph:
1. Query Analysis → 2. Retrieval → 3. Generation → 4. Validation

The state carries all information needed at each step and
accumulates results for the final response.
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
    FACTUAL = "factual"  # "What is my hemoglobin level?"
    COMPARISON = "comparison"  # "How has my cholesterol changed?"
    EXPLANATION = "explanation"  # "What does elevated WBC mean?"
    SUMMARY = "summary"  # "Summarize my blood test results"
    CLARIFICATION = "clarification"  # "What do you mean by that?"
    OUT_OF_SCOPE = "out_of_scope"  # "What's the weather today?"


@dataclass
class RAGState:
    """
    State object for the RAG workflow graph.

    This flows through each node in the LangGraph workflow,
    accumulating information at each step.

    Flow:
    1. Input: user_query, session_id, history
    2. Query Analysis: query_type, search_query, needs_retrieval
    3. Retrieval: retrieved_chunks, context_string
    4. Generation: answer, raw_llm_response
    5. Validation: is_grounded, confidence, final_citations
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
    search_query: str = ""  # Optimized query for retrieval
    needs_retrieval: bool = True
    is_followup: bool = False

    # =========================================================================
    # Retrieval Results
    # =========================================================================
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)
    context_string: str = ""  # Formatted context for LLM
    retrieval_scores: List[float] = field(default_factory=list)

    # =========================================================================
    # Generation Results
    # =========================================================================
    answer: str = ""
    raw_llm_response: str = ""
    cited_chunk_indices: List[int] = field(default_factory=list)  # Which chunks were cited

    # =========================================================================
    # Validation Results
    # =========================================================================
    is_grounded: bool = False  # Is the answer supported by context?
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

    # Step timings (for observability)
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
        """
        Build formatted context string for LLM prompt.

        Format:
        [Source 1: report.pdf, Page 3, Section: Lab Results]
        <chunk content>

        [Source 2: ...]
        """
        top_chunks = self.get_top_chunks(max_chunks)

        if not top_chunks:
            return "No relevant information found in the documents."

        context_parts = []

        for i, chunk in enumerate(top_chunks, 1):
            # Build source reference
            source_parts = [f"Source {i}: {chunk.filename}"]

            if chunk.page_number:
                source_parts.append(f"Page {chunk.page_number}")

            if chunk.section_title:
                source_parts.append(f"Section: {chunk.section_title}")

            source_ref = ", ".join(source_parts)

            # Format chunk
            context_parts.append(f"[{source_ref}]\n{chunk.content}")

        self.context_string = "\n\n---\n\n".join(context_parts)
        return self.context_string

    def build_citations_from_answer(self) -> None:
        """
        Extract citations from the generated answer.

        Looks for patterns like [1], [2] in the answer and
        maps them to the corresponding chunks.
        """
        import re

        # Find all citation numbers in answer
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

                # Full citation
                self.citations.append(chunk.to_citation(num))

                # Display citation
                self.citations_display.append(chunk.to_citation_display(num))

    def calculate_confidence(self) -> float:
        """
        Calculate confidence score based on multiple factors.

        Factors:
        - Retrieval scores
        - Number of supporting chunks
        - Whether citations were found
        - Grounding validation
        """
        if not self.retrieved_chunks:
            self.confidence = 0.0
            return 0.0

        # Base confidence from retrieval scores
        top_scores = sorted(
            [c.score for c in self.retrieved_chunks],
            reverse=True
        )[:3]
        avg_score = sum(top_scores) / len(top_scores) if top_scores else 0

        # Boost if multiple supporting chunks
        chunk_boost = min(0.1 * len(self.retrieved_chunks), 0.2)

        # Boost if citations found
        citation_boost = 0.1 if self.cited_chunk_indices else 0

        # Penalty if not grounded
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
        }


# =============================================================================
# LangGraph State Type (for type hints in graph)
# =============================================================================

# For LangGraph, we need a TypedDict-style state
# But we use the dataclass above for actual implementation
# This is a bridge for LangGraph compatibility

from typing import TypedDict


class GraphState(TypedDict):
    """
    LangGraph-compatible state definition.

    This mirrors RAGState but as a TypedDict for LangGraph.
    The actual workflow uses RAGState internally.
    """
    user_query: str
    session_id: str
    message_id: str
    document_filter: List[str]
    chat_history: List[Dict[str, str]]

    query_type: str
    search_query: str
    needs_retrieval: bool

    retrieved_chunks: List[Dict[str, Any]]
    context_string: str

    answer: str
    cited_chunk_indices: List[int]

    is_grounded: bool
    confidence: float

    citations: List[Dict[str, Any]]
    status: str
    error_message: Optional[str]