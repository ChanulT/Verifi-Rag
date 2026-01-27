"""
Qdrant Repository - WITH YEAR FILTERING SUPPORT

NEW: Supports filtering chunks by year metadata
"""
import os
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from qdrant_client import QdrantClient, models
from qdrant_client.http import models as rest
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """Standardized search result object."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    filename: str
    chunk_index: int
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    total_pages: Optional[int] = None
    document_year: Optional[int] = None  # NEW!

    @property
    def content_preview(self) -> str:
        """Generate a short preview of the content."""
        return self.content[:200] + "..." if len(self.content) > 200 else self.content

    def to_citation_dict(self) -> Dict[str, Any]:
        """Convert to citation format for UI."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source": self.filename,
            "page": self.page_number,
            "section": self.section_title,
            "text": self.content,
            "preview": self.content_preview,
            "relevance_score": self.score,
            "year": self.document_year  # NEW!
        }


class QdrantRepository:
    def __init__(self):
        self.url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = os.getenv("QDRANT_API_KEY", None)
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "medical_documents")

        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
        )
        logger.info(f"Connected to Qdrant at {self.url} (Collection: {self.collection_name})")

    async def search(
            self,
            query_embedding: List[float],
            top_k: int = 5,
            score_threshold: float = 0.0,
            filter_document_ids: Optional[List[str]] = None,
            filter_filenames: Optional[List[str]] = None,
            filter_years: Optional[List[int]] = None,  # NEW!
    ) -> List[VectorSearchResult]:
        """
        Search for vectors similar to the query embedding.

        NEW: Supports year filtering for temporal queries!

        Args:
            query_embedding: Query vector
            top_k: Number of results
            score_threshold: Minimum score
            filter_document_ids: Filter by document IDs
            filter_filenames: Filter by filenames
            filter_years: NEW - Filter by years [2023, 2024, 2025]
        """

        # Build filters
        must_filters = []

        if filter_document_ids:
            must_filters.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=filter_document_ids)
                )
            )

        if filter_filenames:
            must_filters.append(
                models.FieldCondition(
                    key="filename",
                    match=models.MatchAny(any=filter_filenames)
                )
            )

        # NEW: Year filtering
        if filter_years:
            must_filters.append(
                models.FieldCondition(
                    key="document_year",
                    match=models.MatchAny(any=filter_years)
                )
            )
            logger.info(f"Applying year filter: {filter_years}")

        # Create the filter object
        q_filter = models.Filter(must=must_filters) if must_filters else None

        # Execute search
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=q_filter,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )

        # Map results
        results = [self._map_scored_point(point) for point in response.points]

        logger.info(
            f"Search completed: {len(results)} results "
            f"(years={filter_years if filter_years else 'all'})"
        )

        return results

    def _map_scored_point(self, point: models.ScoredPoint) -> VectorSearchResult:
        """
        Map Qdrant ScoredPoint to VectorSearchResult.

        NEW: Extracts year from payload.
        """
        payload = point.payload or {}
        return VectorSearchResult(
            chunk_id=point.id,
            document_id=payload.get("document_id", "unknown"),
            content=payload.get("content", ""),
            score=point.score,
            filename=payload.get("filename", "unknown"),
            chunk_index=payload.get("chunk_index", 0),
            page_number=payload.get("page_number"),
            section_title=payload.get("section_title"),
            total_pages=payload.get("total_pages"),
            document_year=payload.get("document_year")  # NEW!
        )

    async def delete_document(self, document_id: str):
        """Delete all vectors matching a document_id."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id)
                        )
                    ]
                )
            )
        )

    async def health_check(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    async def get_collection_stats(self) -> Dict:
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "status": str(info.status),
                "vectors_count": info.vectors_count,
                "points_count": info.points_count
            }
        except Exception as e:
            return {"error": str(e)}