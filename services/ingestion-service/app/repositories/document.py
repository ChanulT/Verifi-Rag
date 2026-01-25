"""
Document Repository

Handles document and chunk persistence.
Updated to accept document_id for consistency with Qdrant.
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
import uuid

logger = logging.getLogger(__name__)

__all__ = ["DocumentRepository", "InMemoryDocumentStore"]


class DocumentRepository:
    """
    Abstract document repository interface.

    Implementations:
    - InMemoryDocumentStore: For development/testing
    - PostgresDocumentStore: For production (implement separately)
    """

    async def create_document(
        self,
        title: str,
        source: str,
        content: str,
        metadata: Dict[str, Any],
        document_id: Optional[str] = None,  # NEW: Accept specific ID
    ) -> str:
        """Create a document, returning its ID."""
        raise NotImplementedError

    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        raise NotImplementedError

    async def save_chunks(self, document_id: str, chunks: List) -> int:
        """Save chunks for a document."""
        raise NotImplementedError

    async def get_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document."""
        raise NotImplementedError

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks."""
        raise NotImplementedError


# =============================================================================
# In-Memory Implementation
# =============================================================================

# Global in-memory stores
_documents_store: Dict[str, Dict[str, Any]] = {}
_chunks_store: Dict[str, List[Dict[str, Any]]] = {}


class InMemoryDocumentStore(DocumentRepository):
    """
    In-memory document storage for development/demo.

    Replace with actual database in production.
    """

    async def create_document(
        self,
        title: str,
        source: str,
        content: str,
        metadata: Dict[str, Any],
        document_id: Optional[str] = None,
    ) -> str:
        """Create a document with optional specific ID."""
        # Use provided ID or generate new one
        doc_id = document_id or str(uuid.uuid4())

        _documents_store[doc_id] = {
            "id": doc_id,
            "title": title,
            "source": source,
            "content": content,
            "content_length": len(content),
            "metadata": metadata,
            "created_at": datetime.utcnow().isoformat(),
        }

        logger.debug(f"Created document: {doc_id} ({title})")
        return doc_id

    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        return _documents_store.get(document_id)

    async def save_chunks(self, document_id: str, chunks: List) -> int:
        """Save chunks for a document."""
        if document_id not in _chunks_store:
            _chunks_store[document_id] = []

        saved_count = 0

        for chunk in chunks:
            chunk_data = {
                "chunk_id": f"{document_id}-{chunk.index}",
                "document_id": document_id,
                "content": chunk.content,
                "chunk_index": chunk.index,
                "size": len(chunk.content),
                "token_count": chunk.token_count if hasattr(chunk, 'token_count') else len(chunk.content) // 4,
                "page_number": chunk.page_number if hasattr(chunk, 'page_number') else None,
                "section_title": chunk.section_title if hasattr(chunk, 'section_title') else None,
                "metadata": chunk.metadata or {},
                "has_embedding": chunk.embedding is not None,
                "created_at": datetime.utcnow().isoformat(),
            }
            _chunks_store[document_id].append(chunk_data)
            saved_count += 1

        logger.debug(f"Saved {saved_count} chunks for document {document_id}")
        return saved_count

    async def get_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document."""
        chunks = _chunks_store.get(document_id, [])
        # Sort by index
        return sorted(chunks, key=lambda x: x.get("chunk_index", 0))

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks."""
        deleted = False

        if document_id in _documents_store:
            del _documents_store[document_id]
            deleted = True

        if document_id in _chunks_store:
            del _chunks_store[document_id]

        if deleted:
            logger.debug(f"Deleted document: {document_id}")

        return deleted

    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents (summary)."""
        return [
            {
                "id": doc["id"],
                "title": doc["title"],
                "content_length": doc["content_length"],
                "created_at": doc["created_at"],
            }
            for doc in _documents_store.values()
        ]

    async def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_chunks = sum(len(chunks) for chunks in _chunks_store.values())

        return {
            "documents_count": len(_documents_store),
            "chunks_count": total_chunks,
            "storage_type": "in_memory",
        }