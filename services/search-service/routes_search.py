"""
Search API Routes
"""
import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, status, Depends, Request

# UPDATED IMPORTS
from vector import VectorSearchService, SearchQuery

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["Search"])

# =============================================================================
# Dependencies
# =============================================================================

# This allows us to inject the service into the routes
def get_vector_service(request: Request) -> VectorSearchService:
    if not hasattr(request.app.state, "vector_service"):
         raise HTTPException(status_code=503, detail="Vector service not initialized")
    return request.app.state.vector_service

# =============================================================================
# Schemas (Kept same as original)
# =============================================================================

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results")
    score_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum score")
    filter_document_ids: Optional[List[str]] = Field(None, description="Filter by document IDs")
    filter_filenames: Optional[List[str]] = Field(None, description="Filter by filenames")

class CitationInfo(BaseModel):
    chunk_id: str
    document_id: str
    source: str
    page: Optional[int]
    section: Optional[str]
    text: str
    preview: str
    relevance_score: float

class SearchResultItem(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    score: float
    filename: str
    page_number: Optional[int]
    section_title: Optional[str]
    chunk_index: int
    content_preview: str

class SearchResponse(BaseModel):
    results: List[SearchResultItem]
    total_found: int
    query: str
    citations: List[CitationInfo]
    context_string: str

class SearchStatsResponse(BaseModel):
    qdrant: dict
    embedding_provider: dict
    health_status: str

# =============================================================================
# Routes
# =============================================================================

@router.post("", response_model=SearchResponse)
async def search(
    search_data: SearchRequest,  # 3. Renamed 'request' to 'search_data'
    service: VectorSearchService = Depends(get_vector_service)
):
    try:
        search_query = SearchQuery(
            query=search_data.query, # 4. Use 'search_data' here
            top_k=search_data.top_k,
            score_threshold=search_data.score_threshold,
            filter_document_ids=search_data.filter_document_ids,
            filter_filenames=search_data.filter_filenames,
        )

        result = await service.search(search_query)

        result_items = [
            SearchResultItem(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                content=r.content,
                score=r.score,
                filename=r.filename,
                page_number=r.page_number,
                section_title=r.section_title,
                chunk_index=r.chunk_index,
                content_preview=r.content_preview,
            )
            for r in result.results
        ]

        citation_items = [
            CitationInfo(
                chunk_id=c["chunk_id"],
                document_id=c["document_id"],
                source=c["source"],
                page=c["page"],
                section=c["section"],
                text=c["text"],
                preview=c["preview"],
                relevance_score=c["relevance_score"],
            )
            for c in result.citations
        ]

        return SearchResponse(
            results=result_items,
            total_found=result.total_found,
            query=result.query,
            citations=citation_items,
            context_string=result.to_context_string(),
        )

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@router.post("/simple", response_model=SearchResponse)
async def search_simple(
    query: str,
    top_k: int = 5,
    filenames: Optional[str] = None,
    service: VectorSearchService = Depends(get_vector_service)
):
    filter_filenames = [f.strip() for f in filenames.split(",")] if filenames else None

    # Delegate to the main search function logic (or recreate request)
    req = SearchRequest(query=query, top_k=top_k, filter_filenames=filter_filenames)
    return await search(req, service)

@router.get("/stats", response_model=SearchStatsResponse)
async def get_search_stats(service: VectorSearchService = Depends(get_vector_service)):
    try:
        stats = await service.get_stats()
        health = await service.health_check()

        return SearchStatsResponse(
            qdrant=stats.get("qdrant", {}),
            embedding_provider=stats.get("embedding_provider", {}),
            health_status=health.get("status", "unknown"),
        )
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/documents/{document_id}")
async def delete_document_vectors(
    document_id: str,
    service: VectorSearchService = Depends(get_vector_service)
):
    try:
        # Access repo via service
        await service.qdrant_repo.delete_document(document_id)
        return {
            "status": "success",
            "message": f"Deleted vectors for document {document_id}"
        }
    except Exception as e:
        logger.error(f"Failed to delete document vectors: {e}")
        raise HTTPException(status_code=500, detail=str(e))