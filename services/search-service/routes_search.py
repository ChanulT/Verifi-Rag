"""
Search Routes - WITH YEAR FILTERING SUPPORT

NEW: Accepts filter_years parameter for temporal queries
"""

import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, status, Request
from vector import SearchQuery


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["Search"])


# =============================================================================
# Request/Response Models
# =============================================================================

class SearchRequest(BaseModel):
    """
    Search request.

    NEW: Added filter_years for temporal queries!
    """
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    score_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    filter_document_ids: Optional[List[str]] = Field(
        None,
        description="Filter to specific documents. None = search all."
    )

    # NEW: Year filtering for temporal queries!
    filter_years: Optional[List[int]] = Field(
        None,
        description="Filter to specific years (e.g., [2023, 2024, 2025])"
    )


class SearchResult(BaseModel):
    """Single search result with citation metadata."""
    chunk_id: str
    document_id: str
    content: str
    score: float

    # Citation metadata
    filename: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    content_preview: str = ""

    # NEW: Year for temporal queries
    year: Optional[int] = Field(None, description="Year from chunk metadata")

    chunk_index: int = 0


class CitationInfo(BaseModel):
    """Citation info for UI display."""
    number: int
    source: str
    page: Optional[int] = None
    section: Optional[str] = None
    snippet: str
    year: Optional[int] = None  # NEW!


class SearchResponse(BaseModel):
    """Search response with results and citations."""
    results: List[SearchResult]
    citations: List[CitationInfo]
    context_string: str = Field(description="Pre-formatted context for LLM")
    query_metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Endpoints
# =============================================================================

@router.post("", response_model=SearchResponse)
async def search(request: SearchRequest, fastapi_req: Request):
    """
    Search with optional year filtering.

    NEW: Supports filter_years parameter for temporal queries!

    Example:
    ```json
    {
        "query": "What is my WBC count?",
        "top_k": 5,
        "filter_years": [2024, 2025]  // Only search 2024-2025 chunks!
    }
    ```
    """
    # Access the service from app.state
    vector_service = getattr(fastapi_req.app.state, "vector_service", None)

    if not vector_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search service not initialized"
        )

    try:
        # Create SearchQuery with year filter
        query_params = SearchQuery(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            filter_document_ids=request.filter_document_ids,
            filter_years=request.filter_years  # NEW!
        )

        # Execute search
        search_resp = await vector_service.search(query_params)

        # Build response
        results = []
        citations = []

        for i, r in enumerate(search_resp.results, 1):
            results.append(SearchResult(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                content=r.content,
                score=r.score,
                filename=r.filename,
                page_number=r.page_number,
                section_title=r.section_title,
                content_preview=r.content_preview,
                chunk_index=r.chunk_index,
                year=r.document_year  # NEW!
            ))

            citations.append(CitationInfo(
                number=i,
                source=r.filename,
                page=r.page_number,
                section=r.section_title,
                snippet=r.content_preview,
                year=r.document_year  # NEW!
            ))

        logger.info(
            f"Search completed: {len(results)} results "
            f"(years={request.filter_years if request.filter_years else 'all'})"
        )

        return SearchResponse(
            results=results,
            citations=citations,
            context_string=search_resp.to_context_string(),
            query_metadata=search_resp.query_metadata
        )

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.post("/simple")
async def search_simple(
    query: str,
    top_k: int = 5,
    fastapi_req: Request = None
):
    """
    Simple search endpoint.

    Returns just the results without formatting.
    """
    request = SearchRequest(query=query, top_k=top_k)
    response = await search(request, fastapi_req)

    return {
        "query": query,
        "results": [r.dict() for r in response.results],
    }


@router.get("/stats")
async def get_search_stats(fastapi_req: Request):
    """Get vector database statistics."""
    vector_service = getattr(fastapi_req.app.state, "vector_service", None)

    if not vector_service:
        return {"error": "Search service not initialized"}

    stats = await vector_service.get_stats()
    return stats