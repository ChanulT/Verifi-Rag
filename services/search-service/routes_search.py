"""
Search Routes - Updated to Include Date/Year

Changes:
- SearchResult model includes document_date and document_year
- Search endpoint returns date/year for temporal queries
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
    """Search request."""
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    score_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    filter_document_ids: Optional[List[str]] = Field(
        None,
        description="Filter to specific documents. None = search all."
    )
    year_filter: Optional[int] = Field(
        None,
        description="Filter to specific year (e.g., 2023)"
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

    # Date/year for temporal queries
    document_date: Optional[str] = Field(None, description="ISO date: 2023-05-15")
    document_year: Optional[int] = Field(None, description="Year: 2023")

    chunk_index: int = 0


class CitationInfo(BaseModel):
    """Citation info for UI display."""
    number: int
    source: str
    page: Optional[int] = None
    section: Optional[str] = None
    snippet: str
    year: Optional[int] = None  # NEW
    date: Optional[str] = None  # NEW


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
@router.post("", response_model=SearchResponse)
async def search(request: SearchRequest, fastapi_req: Request):  # Add fastapi_req

    # Access the service from app.state (initialized in main.py)
    vector_service = getattr(fastapi_req.app.state, "vector_service", None)

    if not vector_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search service not initialized"
        )

    try:
        # Create the SearchQuery object expected by vector_service.search
        query_params = SearchQuery(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            filter_document_ids=request.filter_document_ids
        )

        # Use the high-level service (it handles embedding and Qdrant calls)
        search_resp = await vector_service.search(query_params)

        # Build response
        results = []
        citations = []

        for i, r in enumerate(search_resp.results, 1):
            # Map VectorSearchResult to your SearchResult model
            results.append(SearchResult(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                content=r.content,
                score=r.score,
                filename=r.filename,
                page_number=r.page_number,
                section_title=r.section_title,
                content_preview=r.content_preview,
                chunk_index=r.chunk_index
            ))

            citations.append(CitationInfo(
                number=i,
                source=r.filename,
                page=r.page_number,
                section=r.section_title,
                snippet=r.content_preview,
            ))

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
):
    """
    Simple search endpoint.

    Returns just the results without formatting.
    """
    request = SearchRequest(query=query, top_k=top_k)
    response = await search(request)

    return {
        "query": query,
        "results": [r.dict() for r in response.results],
    }


@router.get("/stats")
async def get_search_stats():
    """Get vector database statistics."""
    from app.main import qdrant_repo

    if not qdrant_repo:
        return {"error": "Qdrant not configured"}

    stats = await qdrant_repo.get_collection_stats()
    return stats