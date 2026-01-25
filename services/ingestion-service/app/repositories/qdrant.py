"""
Qdrant Vector Database Repository

Handles all Qdrant operations:
- Collection management
- Document/chunk upsert with rich metadata
- Vector search with filtering

Rich metadata strategy enables chatbot UI to display:
- Source document name
- Page numbers
- Text snippets for citations
"""

import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)

__all__ = [
    "QdrantRepository",
    "VectorSearchResult",
    "QdrantConfig",
]


@dataclass
class QdrantConfig:
    """Qdrant connection configuration."""
    url: str = "http://localhost:6333"
    api_key: Optional[str] = None
    collection_name: str = "medical_documents"
    vector_size: int = 1536  # BGE-small default
    distance: str = "Cosine"
    # Connection settings
    timeout: float = 30.0
    prefer_grpc: bool = False
    # Schema management
    recreate_if_dimension_mismatch: bool = False


@dataclass
class VectorSearchResult:
    """
    Search result with full metadata for citation display.

    Contains all information needed by chatbot UI:
    - chunk_id: Unique identifier for deduplication
    - content: The actual text to display/use
    - score: Relevance score (0-1 for cosine)
    - metadata: Rich context for citations
    """
    chunk_id: str
    document_id: str
    content: str
    score: float

    # Citation metadata
    filename: str
    page_number: Optional[int]
    section_title: Optional[str]
    chunk_index: int
    content_preview: str

    # Additional context
    total_pages: Optional[int]
    source_file: Optional[str]
    created_at: Optional[str]

    def to_citation_dict(self) -> Dict[str, Any]:
        """
        Format for chatbot citation display.

        Returns dict suitable for UI rendering like:
        "Source: report.pdf, Page 3, Section: Lab Results"
        """
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source": self.filename,
            "page": self.page_number,
            "section": self.section_title,
            "text": self.content,
            "preview": self.content_preview,
            "relevance_score": round(self.score, 4),
        }


class QdrantRepository:
    """
    Repository for Qdrant vector database operations.

    Handles:
    - Collection creation with optimal settings
    - Document ingestion with rich metadata
    - Similarity search with filtering
    - Batch operations for efficiency

    Metadata Schema (stored with each vector):
    {
        "document_id": "uuid",
        "chunk_id": "uuid",
        "chunk_index": 0,
        "filename": "report.pdf",
        "source_file": "/path/to/report.pdf",
        "page_number": 3,
        "section_title": "Lab Results",
        "content": "Full chunk text...",
        "content_preview": "First 200 chars...",
        "total_pages": 10,
        "created_at": "2024-01-15T10:30:00Z",
        "content_length": 850,
        "has_table": false
    }

    Example:
        repo = QdrantRepository(config)
        await repo.initialize()

        # Ingest chunks
        await repo.upsert_chunks(document_id, chunks, filename, metadata)

        # Search
        results = await repo.search(query_embedding, top_k=5)
    """

    def __init__(self, config: QdrantConfig):
        """
        Initialize Qdrant repository.

        Args:
            config: Qdrant connection configuration
        """
        self.config = config
        self._client: Optional[AsyncQdrantClient] = None
        self._sync_client: Optional[QdrantClient] = None
        self._initialized = False

        logger.info(f"QdrantRepository created: {config.url}")

    async def initialize(self) -> None:
        """
        Initialize Qdrant connection and ensure collection exists.

        Creates collection with optimal settings for RAG:
        - Cosine distance (normalized embeddings)
        - Payload indexing on document_id and filename
        """
        if self._initialized:
            return

        try:
            # Create async client
            self._client = AsyncQdrantClient(
                url=self.config.url,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                prefer_grpc=self.config.prefer_grpc,
            )

            # Also create sync client for operations that need it
            self._sync_client = QdrantClient(
                url=self.config.url,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                prefer_grpc=self.config.prefer_grpc,
            )

            # Check connection
            await self._client.get_collections()
            logger.info(f"✓ Connected to Qdrant: {self.config.url}")

            # Ensure collection exists
            await self._ensure_collection()

            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            raise

    async def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist (and optionally recreate if dim mismatches)."""
        try:
            # Check if collection exists
            collections = await self._client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.config.collection_name in collection_names:
                # Validate existing collection vector size matches expected
                info = await self._client.get_collection(self.config.collection_name)

                # Qdrant returns vectors config in a couple of shapes; handle the common single-vector case.
                existing_size = None
                try:
                    if getattr(info.config.params, "vectors", None) is not None:
                        existing_size = info.config.params.vectors.size
                except Exception:
                    existing_size = None

                if existing_size is not None and existing_size != self.config.vector_size:
                    logger.warning(
                        f"Collection '{self.config.collection_name}' exists with dim={existing_size} "
                        f"but configured dim={self.config.vector_size}."
                    )

                    # Safety: only recreate if explicitly allowed
                    if getattr(self.config, "recreate_if_dimension_mismatch", False):
                        logger.warning(
                            f"Recreating collection '{self.config.collection_name}' to apply new dimension..."
                        )
                        await self._client.delete_collection(self.config.collection_name)
                    else:
                        logger.warning(
                            "Not recreating collection (recreate_if_dimension_mismatch=false). "
                            "Upserts will fail until you recreate the collection or change EMBEDDING_DIMENSION."
                        )
                        return
                else:
                    logger.info(f"Collection '{self.config.collection_name}' already exists")
                    return

            # Create collection with optimal settings
            distance_map = {
                "Cosine": qdrant_models.Distance.COSINE,
                "Euclidean": qdrant_models.Distance.EUCLID,
                "Dot": qdrant_models.Distance.DOT,
            }

            await self._client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=self.config.vector_size,
                    distance=distance_map.get(self.config.distance, qdrant_models.Distance.COSINE),
                ),
                # Optimize for RAG workloads
                optimizers_config=qdrant_models.OptimizersConfigDiff(
                    indexing_threshold=10000,  # Build index after 10k vectors
                ),
            )

            # Create payload indexes for efficient filtering
            await self._client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="document_id",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )

            await self._client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="filename",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )

            logger.info(
                f"✓ Created collection '{self.config.collection_name}' "
                f"(dim={self.config.vector_size}, dist={self.config.distance})"
            )

        except UnexpectedResponse as e:
            if "already exists" in str(e).lower():
                logger.info(f"Collection '{self.config.collection_name}' already exists")
            else:
                raise

    async def upsert_chunks(
            self,
            document_id: str,
            chunks: List[Any],  # List of Chunk dataclass
            filename: str,
            source_file: str,
            total_pages: int,
            extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Upsert document chunks with rich metadata.

        Each chunk is stored with comprehensive metadata enabling:
        - Citation display in chatbot UI
        - Filtering by document/filename
        - Page number references

        Args:
            document_id: Unique document identifier
            chunks: List of Chunk objects (must have .embedding)
            filename: Original filename for display
            source_file: Full path to source
            total_pages: Total pages in document
            extra_metadata: Additional metadata to include

        Returns:
            Number of chunks successfully upserted
        """
        if not self._initialized:
            await self.initialize()

        if not chunks:
            logger.warning("No chunks to upsert")
            return 0

        # Filter chunks with embeddings
        chunks_with_embeddings = [c for c in chunks if c.embedding is not None]

        if not chunks_with_embeddings:
            logger.warning("No chunks have embeddings")
            return 0

        logger.info(
            f"Upserting {len(chunks_with_embeddings)} chunks "
            f"for document '{filename}' ({document_id})"
        )

        # Prepare points for upsert
        points = []
        timestamp = datetime.utcnow().isoformat()

        for chunk in chunks_with_embeddings:
            # Generate unique chunk ID
            chunk_id = f"{document_id}-{chunk.index}"

            # Extract page number from metadata if available
            page_number = None
            section_title = None
            has_table = False

            if chunk.metadata:
                page_number = chunk.metadata.get("page_number")
                section_title = chunk.metadata.get("section_title")
                has_table = chunk.metadata.get("has_table", False)

            # If page_number still None, try chunk's direct attribute
            if page_number is None and hasattr(chunk, 'page_number'):
                page_number = chunk.page_number

            if section_title is None and hasattr(chunk, 'section_title'):
                section_title = chunk.section_title

            # Build rich metadata payload
            payload = {
                # Identifiers
                "document_id": document_id,
                "chunk_id": chunk_id,
                "chunk_index": chunk.index,

                # Source info (for citations)
                "filename": filename,
                "source_file": source_file,
                "page_number": page_number,
                "section_title": section_title,

                # Content
                "content": chunk.content,
                "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                "content_length": len(chunk.content),

                # Document context
                "total_pages": total_pages,
                "total_chunks": len(chunks),

                # Metadata
                "created_at": timestamp,
                "has_table": has_table,
            }

            # Add extra metadata
            if extra_metadata:
                payload["extra"] = extra_metadata

            # Create point
            point = qdrant_models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)),  # Deterministic UUID
                vector=chunk.embedding,
                payload=payload,
            )
            points.append(point)

        # Batch upsert
        try:
            await self._client.upsert(
                collection_name=self.config.collection_name,
                points=points,
                wait=True,  # Wait for indexing
            )

            logger.info(
                f"✓ Upserted {len(points)} chunks for '{filename}'"
            )
            return len(points)

        except Exception as e:
            logger.error(f"Failed to upsert chunks: {e}")
            raise

    async def search(
            self,
            query_embedding: List[float],
            top_k: int = 5,
            score_threshold: float = 0.0,
            filter_document_ids: Optional[List[str]] = None,
            filter_filenames: Optional[List[str]] = None,
    ) -> List[VectorSearchResult]:
        """
        Search for similar chunks.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1 for cosine)
            filter_document_ids: Only search these documents
            filter_filenames: Only search these files

        Returns:
            List of VectorSearchResult with full metadata
        """
        if not self._initialized:
            await self.initialize()

        # Build filter
        filter_conditions = []

        if filter_document_ids:
            filter_conditions.append(
                qdrant_models.FieldCondition(
                    key="document_id",
                    match=qdrant_models.MatchAny(any=filter_document_ids),
                )
            )

        if filter_filenames:
            filter_conditions.append(
                qdrant_models.FieldCondition(
                    key="filename",
                    match=qdrant_models.MatchAny(any=filter_filenames),
                )
            )

        query_filter = None
        if filter_conditions:
            query_filter = qdrant_models.Filter(
                must=filter_conditions
            )

        # Execute search
        try:
            results = await self._client.search(
                collection_name=self.config.collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )

            # Convert to VectorSearchResult
            search_results = []

            for hit in results:
                payload = hit.payload or {}

                result = VectorSearchResult(
                    chunk_id=payload.get("chunk_id", ""),
                    document_id=payload.get("document_id", ""),
                    content=payload.get("content", ""),
                    score=hit.score,
                    filename=payload.get("filename", "unknown"),
                    page_number=payload.get("page_number"),
                    section_title=payload.get("section_title"),
                    chunk_index=payload.get("chunk_index", 0),
                    content_preview=payload.get("content_preview", ""),
                    total_pages=payload.get("total_pages"),
                    source_file=payload.get("source_file"),
                    created_at=payload.get("created_at"),
                )
                search_results.append(result)

            logger.debug(f"Search returned {len(search_results)} results")
            return search_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    async def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks for a document.

        Args:
            document_id: Document to delete

        Returns:
            Number of points deleted
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Delete by filter
            result = await self._client.delete(
                collection_name=self.config.collection_name,
                points_selector=qdrant_models.FilterSelector(
                    filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="document_id",
                                match=qdrant_models.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
            )

            logger.info(f"Deleted document {document_id} from vector store")
            return 1  # Qdrant doesn't return count

        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            raise

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self._initialized:
            await self.initialize()

        try:
            info = await self._client.get_collection(self.config.collection_name)

            return {
                "collection_name": self.config.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status.value,
            }

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}

    async def health_check(self) -> bool:
        """Check Qdrant connection health."""
        try:
            if not self._client:
                return False
            await self._client.get_collections()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close Qdrant connections."""
        if self._client:
            await self._client.close()
            self._client = None

        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None

        self._initialized = False
        logger.info("Qdrant connections closed")