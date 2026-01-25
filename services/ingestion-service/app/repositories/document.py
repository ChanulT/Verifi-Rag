"""
Document Repository

Handles document and chunk storage in PostgreSQL.
Follows the repository pattern from embedding-service/database.py

Provides:
- Clean data access layer
- Separation from business logic
- Testability
"""

import json
import logging
from typing import List, Optional, Dict, Any
from uuid import UUID

from app.models import Chunk

logger = logging.getLogger(__name__)

__all__ = ["DocumentRepository"]


class DocumentRepository:
    """
    Repository for document and chunk operations.

    Follows the repository pattern from embedding-service to
    separate data access from business logic.

    Example:
        repo = DocumentRepository(db_pool)
        doc_id = await repo.create_document(title, source, content, metadata)
        await repo.save_chunks(doc_id, chunks)
    """

    def __init__(self, db_pool):
        """
        Initialize repository.

        Args:
            db_pool: AsyncPG connection pool
        """
        self.db_pool = db_pool
        logger.info("DocumentRepository initialized")

    async def create_document(
            self,
            title: str,
            source: str,
            content: str,
            metadata: Dict[str, Any]
    ) -> str:
        """
        Create a new document.

        Args:
            title: Document title
            source: Source path/URL
            content: Full document content
            metadata: Document metadata

        Returns:
            Document ID (UUID as string)
        """
        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                result = await conn.fetchrow(
                    """
                    INSERT INTO documents (title, source, content, metadata)
                    VALUES ($1, $2, $3, $4) RETURNING id::text
                    """,
                    title,
                    source,
                    content,
                    json.dumps(metadata)
                )

                document_id = result["id"]
                logger.debug(f"Created document: {document_id}")

                return document_id

    async def get_document(
            self,
            document_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get document by ID.

        Args:
            document_id: Document UUID

        Returns:
            Document data or None if not found
        """
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id::text, title,
                       source,
                       content,
                       metadata,
                       created_at
                FROM documents
                WHERE id = $1::uuid
                """,
                document_id
            )

            if not row:
                return None

            return dict(row)

    async def save_chunks(
            self,
            document_id: str,
            chunks: List[Chunk]
    ) -> int:
        """
        Save chunks for a document.

        Args:
            document_id: Document UUID
            chunks: List of Chunk objects

        Returns:
            Number of chunks saved
        """
        if not chunks:
            return 0

        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                saved = 0

                for chunk in chunks:
                    # Prepare embedding data for PostgreSQL vector type
                    embedding_data = None
                    if chunk.embedding:
                        # Format: '[1.0,2.0,3.0]' (no spaces after commas)
                        embedding_data = '[' + ','.join(
                            str(x) for x in chunk.embedding
                        ) + ']'

                    # Build chunk metadata
                    chunk_metadata = chunk.metadata or {}
                    chunk_metadata.update({
                        "chunk_index": chunk.index,
                        "chunk_size": chunk.size,
                        "token_count": chunk.token_count,
                        "page_number": chunk.page_number,
                        "section_title": chunk.section_title,
                    })

                    # Insert chunk
                    await conn.execute(
                        """
                        INSERT INTO chunks (document_id,
                                            content,
                                            embedding,
                                            chunk_index,
                                            metadata,
                                            token_count)
                        VALUES ($1::uuid, $2, $3::vector, $4, $5, $6)
                        """,
                        document_id,
                        chunk.content,
                        embedding_data,
                        chunk.index,
                        json.dumps(chunk_metadata),
                        chunk.token_count
                    )

                    saved += 1

                logger.debug(f"Saved {saved} chunks for document {document_id}")

                return saved

    async def get_chunks(
            self,
            document_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get all chunks for a document.

        Args:
            document_id: Document UUID

        Returns:
            List of chunk dictionaries
        """
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id::text as chunk_id, content,
                       chunk_index,
                       metadata,
                       token_count,
                       embedding IS NOT NULL as has_embedding,
                       created_at
                FROM chunks
                WHERE document_id = $1::uuid
                ORDER BY chunk_index
                """,
                document_id
            )

            return [dict(row) for row in rows]

    async def delete_document(
            self,
            document_id: str
    ) -> bool:
        """
        Delete a document and its chunks.

        Args:
            document_id: Document UUID

        Returns:
            True if deleted, False if not found
        """
        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                # Delete chunks first (foreign key)
                await conn.execute(
                    "DELETE FROM chunks WHERE document_id = $1::uuid",
                    document_id
                )

                # Delete document
                result = await conn.execute(
                    "DELETE FROM documents WHERE id = $1::uuid",
                    document_id
                )

                # Check if any rows were deleted
                deleted = result.split()[-1] != "0"

                if deleted:
                    logger.debug(f"Deleted document: {document_id}")

                return deleted

    async def list_documents(
            self,
            limit: int = 100,
            offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List documents with pagination.

        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip

        Returns:
            List of document dictionaries
        """
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT d.id::text, d.title,
                       d.source,
                       d.metadata,
                       d.created_at,
                       COUNT(c.id) as chunk_count
                FROM documents d
                         LEFT JOIN chunks c ON d.id = c.document_id
                GROUP BY d.id
                ORDER BY d.created_at DESC
                    LIMIT $1
                OFFSET $2
                """,
                limit,
                offset
            )

            return [dict(row) for row in rows]