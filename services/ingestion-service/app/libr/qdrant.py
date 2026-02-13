"""
Qdrant Repository - Updated with Date/Year Metadata

Changes:
- Added document_date and document_year to vector payload
- Search results now include date/year for temporal queries
- Added index on document_year for filtering
"""

import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
)

logger = logging.getLogger(__name__)

__all__ = ["QdrantRepository", "QdrantConfig", "VectorSearchResult"]


@dataclass
class QdrantConfig:
    """Qdrant connection configuration."""
    url: str = "http://localhost:6333"
    api_key: Optional[str] = None
    collection_name: str = "medical_documents"
    vector_size: int = 384
    distance: str = "Cosine"
    timeout: float = 30.0
    prefer_grpc: bool = False


@dataclass
class VectorSearchResult:
    """
    Search result with metadata for citations.

    Includes date/year for temporal queries.
    """
    chunk_id: str
    document_id: str
    content: str
    score: float

    # Citation metadata
    filename: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    content_preview: str = ""

    # NEW: Date/year for temporal queries
    document_date: Optional[str] = None  # ISO format: "2023-05-15"
    document_year: Optional[int] = None  # e.g., 2023

    # Additional metadata
    chunk_index: int = 0
    total_pages: Optional[int] = None
    has_table: bool = False
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "score": self.score,
            "filename": self.filename,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "content_preview": self.content_preview,
            "document_date": self.document_date,
            "document_year": self.document_year,
            "chunk_index": self.chunk_index,
        }


class QdrantRepository:
    """
    Qdrant vector database repository.

    Handles:
    - Vector storage with rich metadata
    - Semantic search
    - Document filtering
    - Year-based filtering for temporal queries
    """

    def __init__(self, config: QdrantConfig):
        self.config = config
        self.client: Optional[AsyncQdrantClient] = None

        # Map distance string to Qdrant Distance enum
        self._distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT,
        }

    async def initialize(self) -> None:
        """Initialize Qdrant client and ensure collection exists."""
        try:
            self.client = AsyncQdrantClient(
                url=self.config.url,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                prefer_grpc=self.config.prefer_grpc,
            )

            # Check if collection exists
            collections = await self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.config.collection_name not in collection_names:
                await self._create_collection()
            else:
                logger.info(f"Collection '{self.config.collection_name}' already exists")

            # Ensure indexes exist
            await self._create_indexes()

            logger.info(f"Qdrant repository initialized: {self.config.url}")

        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            raise

    async def _create_collection(self) -> None:
        """Create the vector collection."""
        distance = self._distance_map.get(
            self.config.distance.lower(),
            Distance.COSINE
        )

        await self.client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config=VectorParams(
                size=self.config.vector_size,
                distance=distance,
            ),
        )

        logger.info(
            f"Created collection '{self.config.collection_name}' "
            f"(size={self.config.vector_size}, distance={distance})"
        )

    async def _create_indexes(self) -> None:
        """Create payload indexes for efficient filtering."""
        try:
            # Index on document_id (for document-level queries)
            await self.client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="document_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )

            # Index on filename
            await self.client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="filename",
                field_schema=PayloadSchemaType.KEYWORD,
            )

            # NEW: Index on document_year for temporal filtering
            await self.client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="document_year",
                field_schema=PayloadSchemaType.INTEGER,
            )

            logger.info("Payload indexes created")

        except Exception as e:
            # Indexes might already exist
            logger.debug(f"Index creation note: {e}")

    async def upsert_chunks(
        self,
        document_id: str,
        chunks: List,
        filename: str,
        document_date: Optional[str] = None,
        document_year: Optional[int] = None,
        batch_size: int = 100,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Upsert chunks with embeddings to Qdrant.

        Args:
            document_id: Unique document identifier
            chunks: List of chunk objects with content and embedding
            filename: Source filename
            document_date: ISO date string (e.g., "2023-05-15")
            document_year: Year as integer (e.g., 2023)
            batch_size: Batch size for upsert

        Returns:
            Number of vectors upserted
        """
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")

        points = []

        for chunk in chunks:
            if not chunk.embedding:
                logger.warning(f"Chunk {chunk.index} has no embedding, skipping")
                continue

            # Generate deterministic UUID for idempotent upserts
            chunk_id = str(uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"{document_id}-{chunk.index}"
            ))

            # Build rich payload with date/year
            payload = {
                # Identifiers
                "document_id": document_id,
                "chunk_id": chunk_id,
                "chunk_index": chunk.index,

                # Source info for citations
                "filename": filename,
                "source_file": filename,
                "page_number": getattr(chunk, 'page_number', None),
                "section_title": getattr(chunk, 'section_title', None),

                # Content
                "content": chunk.content,
                "content_preview": chunk.content[:200] if chunk.content else "",
                "content_length": len(chunk.content) if chunk.content else 0,

                # NEW: Date/year for temporal queries
                "document_date": document_date,
                "document_year": document_year,

                # Metadata
                "total_pages": getattr(chunk, 'total_pages', None),
                "has_table": getattr(chunk, 'has_table', False),
                "created_at": datetime.utcnow().isoformat(),
            }

            if extra_metadata:
                payload.update(extra_metadata)

            # Add any additional metadata from chunk
            if hasattr(chunk, 'metadata') and chunk.metadata:
                for key, value in chunk.metadata.items():
                    if key not in payload:
                        payload[key] = value

            points.append(PointStruct(
                id=chunk_id,
                vector=chunk.embedding,
                payload=payload,
            ))

        if not points:
            logger.warning("No valid points to upsert")
            return 0

        # Upsert in batches
        total_upserted = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            await self.client.upsert(
                collection_name=self.config.collection_name,
                points=batch,
                wait=True,
            )
            total_upserted += len(batch)
            logger.debug(f"Upserted batch: {total_upserted}/{len(points)}")

        logger.info(
            f"Upserted {total_upserted} vectors for document {document_id} "
            f"(year={document_year})"
        )

        return total_upserted

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: float = 0.0,
        document_ids: Optional[List[str]] = None,
        year_filter: Optional[int] = None,
        year_range: Optional[tuple] = None,
    ) -> List[VectorSearchResult]:
        """
        Search for similar vectors.

        Args:
            query_embedding: Query vector
            top_k: Number of results
            score_threshold: Minimum similarity score
            document_ids: Filter to specific documents (None = all)
            year_filter: Filter to specific year
            year_range: Filter to year range (min_year, max_year)

        Returns:
            List of VectorSearchResult with date/year metadata
        """
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")

        # Build filter
        filter_conditions = []

        # Document filter
        if document_ids:
            for doc_id in document_ids:
                filter_conditions.append(
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=doc_id),
                    )
                )

        # Year filter
        if year_filter:
            filter_conditions.append(
                FieldCondition(
                    key="document_year",
                    match=MatchValue(value=year_filter),
                )
            )

        # Build final filter
        search_filter = None
        if filter_conditions:
            if len(filter_conditions) == 1:
                search_filter = Filter(must=filter_conditions)
            else:
                # OR for document_ids, AND for other filters
                if document_ids and len(document_ids) > 1:
                    search_filter = Filter(should=filter_conditions)
                else:
                    search_filter = Filter(must=filter_conditions)

        # Execute search
        results = await self.client.search(
            collection_name=self.config.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=search_filter,
            with_payload=True,
        )

        # Convert to VectorSearchResult
        search_results = []
        for result in results:
            payload = result.payload or {}

            search_results.append(VectorSearchResult(
                chunk_id=payload.get("chunk_id", str(result.id)),
                document_id=payload.get("document_id", ""),
                content=payload.get("content", ""),
                score=result.score,
                filename=payload.get("filename", "unknown"),
                page_number=payload.get("page_number"),
                section_title=payload.get("section_title"),
                content_preview=payload.get("content_preview", ""),
                document_date=payload.get("document_date"),
                document_year=payload.get("document_year"),
                chunk_index=payload.get("chunk_index", 0),
                total_pages=payload.get("total_pages"),
                has_table=payload.get("has_table", False),
                created_at=payload.get("created_at"),
            ))

        logger.debug(f"Search returned {len(search_results)} results")
        return search_results

    async def delete_document(self, document_id: str) -> int:
        """Delete all vectors for a document."""
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")

        result = await self.client.delete(
            collection_name=self.config.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        )

        logger.info(f"Deleted vectors for document: {document_id}")
        return 1

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self.client:
            return {"error": "Client not initialized"}

        try:
            info = await self.client.get_collection(self.config.collection_name)
            return {
                "collection_name": self.config.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.name,
            }
        except Exception as e:
            return {"error": str(e)}

    async def health_check(self) -> bool:
        """Check Qdrant health."""
        try:
            if not self.client:
                return False
            await self.client.get_collections()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close the client connection."""
        if self.client:
            await self.client.close()
            logger.info("Qdrant client closed")