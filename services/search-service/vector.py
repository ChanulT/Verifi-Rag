"""
Vector Search Service - WITH YEAR FILTERING SUPPORT

NEW: Passes year filters to Qdrant for temporal queries
"""

import logging
import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from openai import AsyncOpenAI
from qdrantc import QdrantRepository, VectorSearchResult

logger = logging.getLogger(__name__)

@dataclass
class SearchQuery:
    """
    Search query parameters.

    NEW: Added filter_years for temporal queries.
    """
    query: str
    top_k: int = 5
    score_threshold: float = 0.3
    filter_document_ids: Optional[List[str]] = None
    filter_filenames: Optional[List[str]] = None
    filter_years: Optional[List[int]] = None  # NEW!
    rerank: bool = False

@dataclass
class SearchResponse:
    """Search response with results and metadata."""
    results: List[VectorSearchResult]
    total_found: int
    query: str
    citations: List[Dict[str, Any]] = field(default_factory=list)
    query_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_context_string(self, max_chunks: int = 5) -> str:
        if not self.results:
            return "No relevant information found in the documents."

        context_parts = []
        for i, result in enumerate(self.results[:max_chunks], 1):
            source_parts = [f"Source {i}: {result.filename}"]
            if result.page_number:
                source_parts.append(f"Page {result.page_number}")
            if result.section_title:
                source_parts.append(f"Section: {result.section_title}")

            source_ref = ", ".join(source_parts)
            context_parts.append(f"[{source_ref}]\n{result.content}")

        return "\n\n---\n\n".join(context_parts)

class VectorSearchService:
    """
    High-level vector search service for RAG.

    NEW: Supports year filtering for temporal queries.
    """

    def __init__(self, qdrant_repo: QdrantRepository):
        self.qdrant_repo = qdrant_repo

        self.api_key = os.getenv("OPENAI_API_KEY")
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set in environment variables.")

        self.client = AsyncOpenAI(api_key=self.api_key)
        logger.info(f"VectorSearchService initialized with temporal support")

    async def search(self, query: SearchQuery) -> SearchResponse:
        """
        Search with optional year filtering.

        NEW: Supports filter_years parameter!
        """
        logger.info(
            f"Searching: '{query.query[:50]}...' "
            f"(top_k={query.top_k}, years={query.filter_years})"
        )

        # Step 1: Generate query embedding
        try:
            response = await self.client.embeddings.create(
                input=query.query,
                model=self.embedding_model
            )
            query_embedding = response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate query embedding via OpenAI: {e}")
            raise

        # Step 2: Search Qdrant with year filter
        try:
            results = await self.qdrant_repo.search(
                query_embedding=query_embedding,
                top_k=query.top_k,
                score_threshold=query.score_threshold,
                filter_document_ids=query.filter_document_ids,
                filter_filenames=query.filter_filenames,
                filter_years=query.filter_years,  # NEW!
            )
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            raise

        # Step 3: Format citations
        citations = [result.to_citation_dict() for result in results]

        # Step 4: Build response
        return SearchResponse(
            results=results,
            total_found=len(results),
            query=query.query,
            citations=citations,
            query_metadata={
                "top_k": query.top_k,
                "score_threshold": query.score_threshold,
                "embedding_model": self.embedding_model,
                "filter_years": query.filter_years  # NEW!
            }
        )

    async def get_stats(self) -> Dict[str, Any]:
        qdrant_stats = await self.qdrant_repo.get_collection_stats()
        return {
            "qdrant": qdrant_stats,
            "embedding_provider": {
                "name": "openai",
                "model": self.embedding_model
            },
        }

    async def health_check(self) -> Dict[str, Any]:
        qdrant_healthy = await self.qdrant_repo.health_check()
        openai_healthy = bool(self.api_key)
        return {
            "status": "healthy" if (qdrant_healthy and openai_healthy) else "degraded"
        }