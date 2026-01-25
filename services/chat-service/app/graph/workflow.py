"""
LangGraph RAG Workflow with Chunk Storage.

Enhanced to expose retrieved chunks for database storage,
enabling per-message citation tracking.
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
import httpx

from app.config import settings_manager
from app.graph.state import RAGState, WorkflowStatus, QueryType
from app.models import RetrievedChunk, ChatResponse, CitationDisplay, ResponseStatus
from app.llm.openai_client import OpenAIClient, get_openai_client

logger = logging.getLogger(__name__)

__all__ = ["RAGWorkflow", "create_rag_workflow", "SearchServiceClient"]


class SearchServiceClient:
    """Client for the search/ingestion service."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.3,
        document_ids: Optional[List[str]] = None,
    ) -> tuple[List[RetrievedChunk], List[Dict]]:
        """
        Search for relevant chunks.

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

            response = await self.client.post(
                f"{self.base_url}/search",
                json=payload,
            )
            response.raise_for_status()

            data = response.json()

            chunks = []
            raw_chunks = []

            for result in data.get("results", []):
                # Create RetrievedChunk for processing
                chunk = RetrievedChunk(
                    chunk_id=result.get("chunk_id", ""),
                    document_id=result.get("document_id", ""),
                    content=result.get("content", ""),
                    score=result.get("score", 0.0),
                    filename=result.get("filename", "unknown"),
                    page_number=result.get("page_number"),
                    section_title=result.get("section_title"),
                    content_preview=result.get("content_preview", ""),
                )
                chunks.append(chunk)

                # Keep raw dict for database storage
                raw_chunks.append(result)

            logger.info(f"Retrieved {len(chunks)} chunks")
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


async def analyze_query_node(state: RAGState, llm: OpenAIClient) -> RAGState:
    """Analyze the user query."""
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

        state.step_timings["analyze"] = (time.time() - start) * 1000

    except Exception as e:
        logger.error(f"Query analysis failed: {e}")
        state.search_query = state.user_query
        state.needs_retrieval = True

    return state


async def retrieve_context_node(
    state: RAGState,
    search_client: SearchServiceClient,
) -> tuple[RAGState, List[Dict]]:
    """
    Retrieve relevant chunks.

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
        chunks, raw_chunks = await search_client.search(
            query=state.search_query,
            top_k=settings.retrieval_top_k,
            score_threshold=settings.retrieval_score_threshold,
            document_ids=state.document_filter if state.document_filter else None,
        )

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


class RAGWorkflow:
    """
    RAG Workflow with chunk storage support.

    Exposes retrieved chunks for database storage.
    """

    def __init__(
        self,
        llm: OpenAIClient,
        search_client: SearchServiceClient,
    ):
        self.llm = llm
        self.search_client = search_client

        # Store last retrieved chunks for database persistence
        self._last_retrieved_chunks: List[Dict] = []

        logger.info("RAG Workflow initialized")

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

        After running, access self._last_retrieved_chunks to get
        the raw chunks for database storage.
        """
        state = RAGState(
            user_query=query,
            session_id=session_id or str(uuid.uuid4()),
            message_id=message_id or str(uuid.uuid4())[:8],
            document_filter=document_filter or [],
            chat_history=chat_history or [],
        )

        # Clear previous chunks
        self._last_retrieved_chunks = []

        try:
            # Step 1: Analyze query
            state = await analyze_query_node(state, self.llm)

            # Step 2: Retrieve context
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

        # Build citation displays
        citations_display = []
        for i, chunk in enumerate(state.get_top_chunks(len(state.cited_chunk_indices or []))):
            if (i + 1) in (state.cited_chunk_indices or []):
                citations_display.append(chunk.to_citation_display(i + 1))

        # Include top chunks as citations if none explicit
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
        """
        Get the raw chunks from the last run.

        Use this to save chunks to the database.
        """
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