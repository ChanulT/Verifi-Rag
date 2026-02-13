"""
Embedding Service

Orchestrates embedding generation using configured provider.
Handles batching, error recovery, and provider health checks.
"""

import logging
from typing import List

from app.models import Chunk
from app.providers.embedding import EmbeddingProvider

logger = logging.getLogger(__name__)

__all__ = ["EmbeddingService"]


class EmbeddingService:
    """
    Service for generating embeddings.

    Orchestrates the embedding provider and adds:
    - Batch processing
    - Error handling
    - Health monitoring
    - Progress tracking

    Example:
        service = EmbeddingService(provider=local_provider)
        chunks_with_embeddings = await service.generate_embeddings(chunks)
    """

    def __init__(
            self,
            provider: EmbeddingProvider,
            batch_size: int = 32
    ):
        """
        Initialize embedding service.

        Args:
            provider: Embedding provider (local or openai)
            batch_size: Maximum batch size for embedding requests
        """
        self.provider = provider
        self.batch_size = batch_size

        provider_info = provider.get_info()
        logger.info(
            f"EmbeddingService initialized: "
            f"provider={provider_info['provider']}, "
            f"batch_size={batch_size}"
        )

    async def generate_embeddings(
            self,
            chunks: List[Chunk]
    ) -> List[Chunk]:
        """
        Generate embeddings for all chunks.

        Processes chunks in batches and attaches embeddings
        to each chunk object.

        Args:
            chunks: List of Chunk objects

        Returns:
            List of Chunk objects with embeddings attached
        """
        if not chunks:
            return []

        logger.info(f"Generating embeddings for {len(chunks)} chunks...")

        # Extract texts
        texts = [chunk.content for chunk in chunks]

        # Generate embeddings in batches
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

            logger.debug(
                f"Processing batch {batch_num}/{total_batches} "
                f"({len(batch)} chunks)"
            )

            try:
                embeddings = await self.provider.embed(batch)
                all_embeddings.extend(embeddings)

            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch {batch_num}: {e}")
                # Add None for failed embeddings
                all_embeddings.extend([None] * len(batch))

        # Attach embeddings to chunks
        embedded_chunks = []
        successful = 0

        for chunk, embedding in zip(chunks, all_embeddings):
            if embedding is not None:
                chunk.embedding = embedding
                successful += 1

            embedded_chunks.append(chunk)

        logger.info(
            f"✓ Generated {successful}/{len(chunks)} embeddings "
            f"({(successful / len(chunks) * 100):.1f}% success rate)"
        )

        return embedded_chunks

    async def health_check(self) -> bool:
        """
        Check if embedding provider is available.

        Returns:
            True if healthy, False otherwise
        """
        return await self.provider.health_check()

    def get_provider_info(self) -> dict:
        """
        Get embedding provider information.

        Returns:
            Dictionary with provider metadata
        """
        return self.provider.get_info()