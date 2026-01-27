"""
LangGraph RAG Workflow - WITH TEMPORAL QUERY SUPPORT

NEW: Extracts temporal information and filters search by years
"""

import logging
import time
import uuid
import re
from datetime import datetime
from typing import Dict, Any, Optional, List
import httpx

from app.config import settings_manager
from app.graph.state import RAGState, WorkflowStatus, QueryType, TemporalFilter
from app.models import RetrievedChunk, ChatResponse, CitationDisplay, ResponseStatus
from app.llm.openai_client import OpenAIClient, get_openai_client

logger = logging.getLogger(__name__)

__all__ = ["RAGWorkflow", "create_rag_workflow", "SearchServiceClient"]


# =============================================================================
# NEW: Temporal Pattern Extraction (Fallback)
# =============================================================================

def extract_temporal_patterns(query: str) -> TemporalFilter:
    """
    Extract temporal information from query using regex patterns.

    This is a FALLBACK if LLM doesn't extract temporal info.

    Examples:
        "last 3 years" → years=[2023, 2024, 2025]
        "in 2024" → years=[2024]
        "current WBC" → prefer_latest=True
    """
    current_year = datetime.now().year
    query_lower = query.lower()
    years = []
    prefer_latest = False

    # Pattern 1: Specific year mentions ("in 2024", "from 2023")
    year_pattern = r'\b(20\d{2})\b'
    year_matches = re.findall(year_pattern, query)
    if year_matches:
        years.extend([int(y) for y in year_matches])

    # Pattern 2: "last N years" / "past N years"
    last_years_pattern = r'\b(?:last|past)\s+(\d+)\s+years?\b'
    last_years_match = re.search(last_years_pattern, query_lower)
    if last_years_match:
        n = int(last_years_match.group(1))
        years.extend(range(current_year - n, current_year))

    # Pattern 3: "between YEAR1 and YEAR2"
    between_pattern = r'\bbetween\s+(20\d{2})\s+and\s+(20\d{2})\b'
    between_match = re.search(between_pattern, query_lower)
    if between_match:
        year1, year2 = int(between_match.group(1)), int(between_match.group(2))
        years.extend(range(min(year1, year2), max(year1, year2) + 1))

    # Pattern 4: Recency indicators
    recency_keywords = [
        'current', 'latest', 'most recent', 'now', 'today',
        'this year', 'recent'
    ]
    if any(keyword in query_lower for keyword in recency_keywords):
        prefer_latest = True

    # Remove duplicates and sort
    years = sorted(set(years))

    logger.info(f"Temporal extraction: years={years}, prefer_latest={prefer_latest}")

    return TemporalFilter(
        years=years,
        prefer_latest=prefer_latest
    )


def filter_to_latest_year(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """
    Filter chunks to only the most recent year.

    Used when user asks for "current" or "latest" values.
    """
    if not chunks:
        return chunks

    # Extract years from chunks
    chunks_with_years = []
    for chunk in chunks:
        year = None
        # Try to get year from chunk (you may need to add this field)
        # For now, assume it might be in metadata or content
        # You'll need to modify based on your actual chunk structure
        if hasattr(chunk, 'year'):
            year = chunk.year
        elif hasattr(chunk, 'metadata') and isinstance(chunk.metadata, dict):
            year = chunk.metadata.get('year')

        if year:
            chunks_with_years.append((chunk, year))

    if not chunks_with_years:
        # No year info, return all chunks
        return chunks

    # Find latest year
    latest_year = max(year for _, year in chunks_with_years)

    # Keep only chunks from latest year
    filtered = [chunk for chunk, year in chunks_with_years if year == latest_year]

    logger.info(
        f"Filtered to latest year {latest_year}: "
        f"{len(filtered)}/{len(chunks)} chunks"
    )

    return filtered or chunks  # Fallback to all if filtering fails


# =============================================================================
# Search Service Client
# =============================================================================

class SearchServiceClient:
    """Client for the search/ingestion service."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

    async def search(
        self,
        query: str,
        top_k: int = 7,
        score_threshold: float = 0.3,
        document_ids: Optional[List[str]] = None,
        filter_years: Optional[List[int]] = None,  # NEW!
    ) -> tuple[List[RetrievedChunk], List[Dict]]:
        """
        Search for relevant chunks.

        NEW: Supports year filtering!

        Args:
            query: Search query
            top_k: Number of results
            score_threshold: Minimum score
            document_ids: Document filter
            filter_years: NEW - Filter by years [2023, 2024, 2025]

        Returns:
            Tuple of (RetrievedChunk list, raw dict list for storage)
        """
        try:
            payload = {
                "query": query,
                "top_k": top_k,
                "score_threshold": score_threshold,
            }

            if document_ids:
                payload["filter_document_ids"] = document_ids

            # NEW: Add year filtering
            if filter_years:
                payload["filter_years"] = filter_years
                logger.info(f"Filtering search to years: {filter_years}")

            response = await self.client.post(
                f"{self.base_url}/search",
                json=payload,
            )
            response.raise_for_status()

            data = response.json()

            chunks = []
            raw_chunks = []

            for result in data.get("results", []):
                chunk = RetrievedChunk(
                    chunk_id=result.get("chunk_id", ""),
                    document_id=result.get("document_id", ""),
                    content=result.get("content", ""),
                    score=result.get("score", 0.0),
                    filename=result.get("filename", "unknown"),
                    page_number=result.get("page_number"),
                    section_title=result.get("section_title"),
                    content_preview=result.get("content", ""),
                )
                chunks.append(chunk)
                raw_chunks.append(result)

            logger.info(
                f"Retrieved {len(chunks)} chunks "
                f"(years={filter_years if filter_years else 'all'})"
            )
            return chunks, raw_chunks

        except httpx.HTTPError as e:
            logger.error(f"Search service error: {e}")
            return [], []
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return [], []

    async def health_check(self) -> bool:
        """Check if search service is available."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# =============================================================================
# Workflow Nodes
# =============================================================================

async def analyze_query_node(state: RAGState, llm: OpenAIClient) -> RAGState:
    """
    Analyze the user query.

    NEW: Extracts temporal information from the query.
    """
    start = time.time()
    state.status = WorkflowStatus.ANALYZING

    try:
        result = await llm.analyze_query(
            query=state.user_query,
            chat_history=state.chat_history,
        )

        state.query_type = result.query_type
        state.search_query = result.search_query or state.user_query
        state.needs_retrieval = result.needs_retrieval
        state.is_followup = result.is_followup

        if state.query_type == QueryType.OUT_OF_SCOPE:
            state.needs_retrieval = False

        # NEW: Extract temporal information
        # Use regex fallback (you can enhance this with LLM later)
        state.temporal_filter = extract_temporal_patterns(state.user_query)

        state.step_timings["analyze"] = (time.time() - start) * 1000

        logger.info(
            f"Query analysis: type={state.query_type}, "
            f"years={state.temporal_filter.years}, "
            f"prefer_latest={state.temporal_filter.prefer_latest}"
        )

    except Exception as e:
        logger.error(f"Query analysis failed: {e}")
        state.search_query = state.user_query
        state.needs_retrieval = True
        # Fallback temporal extraction
        state.temporal_filter = extract_temporal_patterns(state.user_query)

    return state


async def retrieve_context_node(
    state: RAGState,
    search_client: SearchServiceClient,
) -> tuple[RAGState, List[Dict]]:
    """
    Retrieve relevant chunks.

    NEW: Passes temporal filters to search service.

    Returns:
        Tuple of (state, raw_chunks for database storage)
    """
    start = time.time()
    state.status = WorkflowStatus.RETRIEVING
    raw_chunks = []

    if not state.needs_retrieval:
        state.step_timings["retrieve"] = 0
        return state, raw_chunks

    settings = settings_manager.current

    try:
        # NEW: Pass temporal filters to search
        chunks, raw_chunks = await search_client.search(
            query=state.search_query,
            top_k=settings.retrieval_top_k,
            score_threshold=settings.retrieval_score_threshold,
            document_ids=[],  # ALWAYS search ALL documents
            filter_years=state.temporal_filter.years if state.temporal_filter.years else None,  # NEW!
        )

        # NEW: If prefer_latest, filter to most recent year
        if state.temporal_filter.prefer_latest and chunks:
            chunks = filter_to_latest_year(chunks)
            logger.info(f"Applied recency filter (prefer_latest=True)")

        state.retrieved_chunks = chunks
        state.retrieval_scores = [c.score for c in chunks]

        if chunks:
            state.build_context_string(max_chunks=settings.max_context_chunks)
        else:
            state.context_string = ""

        state.step_timings["retrieve"] = (time.time() - start) * 1000

    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        state.retrieved_chunks = []
        state.context_string = ""

    return state, raw_chunks


async def generate_answer_node(state: RAGState, llm: OpenAIClient) -> RAGState:
    """Generate answer with citations."""
    start = time.time()
    state.status = WorkflowStatus.GENERATING

    settings = settings_manager.current

    try:
        if not state.has_relevant_context():
            if state.query_type == QueryType.OUT_OF_SCOPE:
                state.answer = await llm.generate_no_context_response(
                    state.user_query,
                    reason="this question appears to be outside the scope of your medical documents"
                )
            else:
                state.answer = await llm.generate_no_context_response(
                    state.user_query,
                    reason="no relevant information was found in the uploaded documents"
                )

            state.status = WorkflowStatus.NO_CONTEXT
            state.step_timings["generate"] = (time.time() - start) * 1000
            return state

        answer, tokens_used = await llm.generate_answer(
            query=state.user_query,
            context=state.context_string,
            chat_history=state.chat_history,
        )

        state.answer = answer
        state.raw_llm_response = answer
        state.build_citations_from_answer()

        state.step_timings["generate"] = (time.time() - start) * 1000

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        state.error_message = str(e)
        state.status = WorkflowStatus.FAILED
        state.answer = "I encountered an error generating your answer. Please try again."

    return state


async def validate_answer_node(state: RAGState, llm: OpenAIClient) -> RAGState:
    """Validate answer grounding."""
    start = time.time()
    state.status = WorkflowStatus.VALIDATING

    if not state.has_relevant_context() or state.status == WorkflowStatus.FAILED:
        state.is_grounded = True
        state.calculate_confidence()
        state.step_timings["validate"] = 0
        return state

    try:
        result = await llm.validate_answer(
            answer=state.answer,
            context=state.context_string,
        )

        state.is_grounded = result.is_grounded
        state.validation_notes = result.issues

        if result.confidence > 0:
            state.confidence = result.confidence
        else:
            state.calculate_confidence()

        state.step_timings["validate"] = (time.time() - start) * 1000

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        state.is_grounded = True
        state.calculate_confidence()

    return state


def finalize_response_node(state: RAGState) -> RAGState:
    """Finalize response."""
    state.end_time = datetime.utcnow()

    if state.status not in [WorkflowStatus.FAILED, WorkflowStatus.NO_CONTEXT]:
        state.status = WorkflowStatus.COMPLETED

    return state


# =============================================================================
# RAG Workflow
# =============================================================================

class RAGWorkflow:
    """
    RAG Workflow with temporal query support.

    NEW: Automatically detects and applies temporal filters.
    """

    def __init__(
        self,
        llm: OpenAIClient,
        search_client: SearchServiceClient,
    ):
        self.llm = llm
        self.search_client = search_client
        self._last_retrieved_chunks: List[Dict] = []

        logger.info("RAG Workflow initialized with temporal support")

    async def run(
        self,
        query: str,
        session_id: Optional[str] = None,
        message_id: Optional[str] = None,
        document_filter: Optional[List[str]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> ChatResponse:
        """
        Run the RAG workflow.

        NEW: Automatically handles temporal queries!
        """
        state = RAGState(
            user_query=query,
            session_id=session_id or str(uuid.uuid4()),
            message_id=message_id or str(uuid.uuid4())[:8],
            document_filter=document_filter or [],
            chat_history=chat_history or [],
        )

        self._last_retrieved_chunks = []

        try:
            # Step 1: Analyze query (now extracts temporal info)
            state = await analyze_query_node(state, self.llm)

            # Step 2: Retrieve context (now uses temporal filters)
            if state.needs_retrieval:
                state, raw_chunks = await retrieve_context_node(state, self.search_client)
                self._last_retrieved_chunks = raw_chunks

            # Step 3: Generate answer
            state = await generate_answer_node(state, self.llm)

            # Step 4: Validate
            if state.has_relevant_context() and state.status != WorkflowStatus.FAILED:
                state = await validate_answer_node(state, self.llm)
            else:
                state.calculate_confidence()

            # Finalize
            state = finalize_response_node(state)

        except Exception as e:
            logger.error(f"Workflow failed: {e}", exc_info=True)
            state.status = WorkflowStatus.FAILED
            state.error_message = str(e)
            state.answer = "I'm sorry, I encountered an error. Please try again."
            state.end_time = datetime.utcnow()

        return self._build_response(state)

    def _build_response(self, state: RAGState) -> ChatResponse:
        """Convert workflow state to ChatResponse."""
        if state.status == WorkflowStatus.FAILED:
            status = ResponseStatus.ERROR
        elif state.status == WorkflowStatus.NO_CONTEXT:
            status = ResponseStatus.NO_RELEVANT_CONTEXT
        elif state.confidence < settings_manager.current.confidence_threshold:
            status = ResponseStatus.LOW_CONFIDENCE
        else:
            status = ResponseStatus.SUCCESS

        citations_display = []
        for i, chunk in enumerate(state.get_top_chunks(len(state.cited_chunk_indices or []))):
            if (i + 1) in (state.cited_chunk_indices or []):
                citations_display.append(chunk.to_citation_display(i + 1))

        if not citations_display and state.retrieved_chunks:
            for i, chunk in enumerate(state.get_top_chunks(3), 1):
                citations_display.append(chunk.to_citation_display(i))

        return ChatResponse(
            answer=state.answer,
            citations=citations_display,
            citation_details=state.citations if state.citations else None,
            status=status,
            confidence=state.confidence,
            session_id=state.session_id,
            message_id=state.message_id,
            chunks_retrieved=len(state.retrieved_chunks),
            chunks_used=len(state.cited_chunk_indices) if state.cited_chunk_indices else 0,
            processing_time_ms=state.get_processing_time_ms(),
            fallback_message=state.error_message if status == ResponseStatus.ERROR else None,
        )

    def get_last_retrieved_chunks(self) -> List[Dict]:
        """Get the raw chunks from the last run."""
        return self._last_retrieved_chunks


def create_rag_workflow(
    llm: Optional[OpenAIClient] = None,
    search_url: Optional[str] = None,
) -> RAGWorkflow:
    """Create a RAG workflow instance."""
    settings = settings_manager.current

    if llm is None:
        llm = get_openai_client()

    search_client = SearchServiceClient(
        base_url=search_url or settings.search_service_url,
        timeout=settings.search_timeout_seconds,
    )

    return RAGWorkflow(llm=llm, search_client=search_client)