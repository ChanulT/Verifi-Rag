"""
Chunking Service

Handles text chunking with multiple strategies.
Follows the async pattern from embedding-service.

Strategies:
- Semantic: Groups similar content (requires embedding service)
- Recursive: Respects natural boundaries
"""

import asyncio
import logging
import re
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from app.models import Chunk
from app.providers.embedding import EmbeddingProvider

logger = logging.getLogger(__name__)

__all__ = ["ChunkingService"]


class ChunkingService:
    """
    Service for text chunking.

    Provides multiple chunking strategies:
    - Semantic: Groups semantically similar content (uses embedding service)
    - Recursive: Respects sentence/paragraph boundaries

    Follows async pattern from embedding-service to prevent blocking.

    Example:
        service = ChunkingService(
            chunk_size=1000,
            chunk_overlap=200,
            use_semantic=True,
            embedding_provider=local_provider
        )
        chunks = await service.chunk(content)
    """

    def __init__(
            self,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            min_chunk_size: int = 100,
            max_chunk_size: int = 2000,
            use_semantic: bool = True,
            embedding_provider: Optional[EmbeddingProvider] = None
    ):
        """
        Initialize chunking service.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            min_chunk_size: Minimum chunk size
            max_chunk_size: Maximum chunk size
            use_semantic: Use semantic chunking
            embedding_provider: Provider for semantic chunking
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.use_semantic = use_semantic
        self.embedding_provider = embedding_provider

        # Initialize splitters
        self._setup_splitters()

        logger.info(
            f"ChunkingService initialized: "
            f"size={chunk_size}, overlap={chunk_overlap}, "
            f"semantic={use_semantic}"
        )

    def _setup_splitters(self):
        """Setup chunking splitters."""
        # Recursive splitter (always available)
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Semantic splitter (if embedding provider available)
        self.semantic_splitter = None
        if self.use_semantic and self.embedding_provider:
            try:
                # Create a wrapper for the embedding provider
                # that LangChain can use
                embeddings_wrapper = LangChainEmbeddingsWrapper(
                    self.embedding_provider
                )

                self.semantic_splitter = SemanticChunker(
                    embeddings=embeddings_wrapper,
                    breakpoint_threshold_type="percentile"
                )
                logger.info("✓ Semantic chunker initialized")

            except Exception as e:
                logger.warning(f"Failed to initialize semantic chunker: {e}")
                logger.info("Falling back to recursive chunking")

        if self.use_semantic and not self.semantic_splitter:
            logger.warning(
                "Semantic chunking requested but not available. "
                "Using recursive chunking instead."
            )

    async def chunk(
            self,
            content: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Chunk text content (async, non-blocking).

        Uses run_in_executor() to prevent blocking the event loop,
        following the same pattern as embedding-service.

        Args:
            content: Text content to chunk
            metadata: Optional metadata to attach to chunks

        Returns:
            List of Chunk objects
        """
        if "=== Page" in content:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.chunk_by_page, content, metadata)

            # Fallback to standard recursive chunking
        return await super().chunk(content, metadata)

    def chunk_by_page(self, content: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Splits content strictly by page delimiters added by the extractor.
        """
        # Split by the "=== Page X ===" delimiter used in ocr_extraction.py
        # We capture the page number in the regex
        page_splits = re.split(r'(=== Page \d+ ===\n)', content)

        chunks = []
        current_chunk_index = 0

        # The split result looks like: ['', '=== Page 1 ===\n', 'Content...', '=== Page 2 ===\n', 'Content...']
        # We iterate in steps of 2 to pair the Header with the Content
        for i in range(1, len(page_splits), 2):
            page_header = page_splits[i]
            page_content = page_splits[i + 1] if i + 1 < len(page_splits) else ""

            full_text = page_header + page_content

            # Extract page number for metadata
            page_num_match = re.search(r'Page (\d+)', page_header)
            page_num = int(page_num_match.group(1)) if page_num_match else None

            # Create the Chunk
            chunk = Chunk(
                content=full_text.strip(),
                index=current_chunk_index,
                page_number=page_num,
                metadata={
                    **metadata,
                    "chunk_strategy": "page_based",
                    "page_number": page_num
                }
            )
            chunks.append(chunk)
            current_chunk_index += 1

        logger.info(f"✓ Created {len(chunks)} page-based chunks")
        return chunks

    def _chunk_sync(
            self,
            content: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Synchronous chunking implementation.

        Called by chunk() via run_in_executor().

        Args:
            content: Text content to chunk
            metadata: Optional metadata

        Returns:
            List of Chunk objects
        """
        base_metadata = metadata or {}

        # Create LangChain document
        doc = Document(
            page_content=content,
            metadata=base_metadata
        )

        # Try semantic chunking first (if available)
        if self.semantic_splitter and len(content) > self.chunk_size:
            try:
                langchain_chunks = self.semantic_splitter.split_documents([doc])
                chunk_method = "semantic"

            except Exception as e:
                logger.warning(f"Semantic chunking failed: {e}, using recursive")
                langchain_chunks = self.recursive_splitter.split_documents([doc])
                chunk_method = "recursive_fallback"
        else:
            # Use recursive chunking
            langchain_chunks = self.recursive_splitter.split_documents([doc])
            chunk_method = "recursive"

        # Convert to our Chunk model
        chunks = []
        for i, lc_chunk in enumerate(langchain_chunks):
            text = lc_chunk.page_content.strip()

            # Filter by minimum size
            if len(text) < self.min_chunk_size:
                continue

            # Build chunk metadata
            chunk_metadata = {
                **lc_chunk.metadata,
                "chunk_index": i,
                "total_chunks": len(langchain_chunks),
                "chunk_size": len(text),
                "chunk_method": chunk_method,
            }

            # Create Chunk object
            chunk = Chunk(
                content=text,
                index=i,
                start_char=0,  # Can be enhanced if needed
                end_char=len(text),
                metadata=chunk_metadata
            )

            chunks.append(chunk)

        return chunks

    def update_config(
            self,
            chunk_size: Optional[int] = None,
            chunk_overlap: Optional[int] = None,
            min_chunk_size: Optional[int] = None,
            use_semantic: Optional[bool] = None
    ):
        """
        Update chunking configuration.

        Reinitializes splitters with new settings.

        Args:
            chunk_size: Target chunk size
            chunk_overlap: Chunk overlap
            min_chunk_size: Minimum chunk size
            use_semantic: Use semantic chunking
        """
        if chunk_size is not None:
            self.chunk_size = chunk_size
        if chunk_overlap is not None:
            self.chunk_overlap = chunk_overlap
        if min_chunk_size is not None:
            self.min_chunk_size = min_chunk_size
        if use_semantic is not None:
            self.use_semantic = use_semantic

        # Reinitialize splitters
        self._setup_splitters()

        logger.info("Chunking configuration updated")


# =============================================================================
# LangChain Embeddings Wrapper
# =============================================================================

class LangChainEmbeddingsWrapper:
    """
    Wrapper to make our EmbeddingProvider compatible with LangChain.

    LangChain's SemanticChunker expects embeddings with specific methods.
    This wrapper adapts our async provider to LangChain's interface.
    """

    def __init__(self, provider: EmbeddingProvider):
        """
        Initialize wrapper.

        Args:
            provider: Our EmbeddingProvider instance
        """
        self.provider = provider

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed documents (synchronous wrapper for async provider).

        LangChain's SemanticChunker calls this synchronously,
        so we use asyncio.run() to execute our async provider.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        import asyncio

        # Run async provider in event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.provider.embed(texts))

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.

        Args:
            text: Query text

        Returns:
            Embedding vector
        """
        embeddings = self.embed_documents([text])
        return embeddings[0] if embeddings else []