"""
Embedding providers for the Ingestion Service.

Provides abstraction over different embedding backends:
- LocalEmbeddingProvider: Uses your local embedding microservice
- OpenAIEmbeddingProvider: Uses OpenAI API (fallback)

The LocalEmbeddingProvider integrates with your existing embedding service
running on port 8001.
"""

import logging
from abc import ABC, abstractmethod
from typing import List
import httpx
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

__all__ = [
    "EmbeddingProvider",
    "LocalEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "get_embedding_provider",
]


# =============================================================================
# Abstract Base
# =============================================================================

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the embedding provider is available.

        Returns:
            True if healthy, False otherwise
        """
        pass

    @abstractmethod
    def get_info(self) -> dict:
        """
        Get provider information.

        Returns:
            Dictionary with provider metadata
        """
        pass


# =============================================================================
# Local Embedding Provider (Your Service!)
# =============================================================================

class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider using your local embedding microservice.

    Integrates with the embedding service running on port 8001.
    This follows the same async pattern as your embedding service.

    Example:
        provider = LocalEmbeddingProvider("http://localhost:8001")
        embeddings = await provider.embed(["text1", "text2"])
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        timeout: float = 600.0,  # 10 minutes - enough for CPU embeddings
        max_retries: int = 3
    ):
        """
        Initialize local embedding provider.

        Args:
            base_url: Base URL of your embedding service
            timeout: Request timeout in seconds (default: 600s = 10 min for CPU)
            max_retries: Maximum number of retry attempts
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

        # Create client with custom timeout
        # None means no timeout on individual operations
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                timeout=timeout,  # Overall timeout
                connect=10.0,     # Connection timeout
                read=timeout,     # Read timeout (most important for long requests)
                write=10.0        # Write timeout
            )
        )

        logger.info(f"Initialized LocalEmbeddingProvider: {self.base_url} (timeout: {timeout}s)")

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using your local embedding service.

        Calls POST /embed endpoint on your embedding service.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors

        Raises:
            httpx.HTTPError: If the request fails
            ValueError: If response format is invalid
        """
        if not texts:
            return []

        logger.info(f"Requesting embeddings for {len(texts)} texts from {self.base_url} (timeout: {self.timeout}s)")

        try:
            response = await self.client.post(
                f"{self.base_url}/embed",
                json={"texts": texts}
            )
            response.raise_for_status()

            data = response.json()
            embeddings = data.get("embeddings", [])

            if not embeddings:
                raise ValueError("No embeddings returned from service")

            if len(embeddings) != len(texts):
                raise ValueError(
                    f"Expected {len(texts)} embeddings, got {len(embeddings)}"
                )

            logger.info(
                f"✓ Generated {len(embeddings)} embeddings "
                f"(dimension: {len(embeddings[0]) if embeddings else 0})"
            )

            return embeddings

        except httpx.TimeoutException as e:
            logger.error(
                f"Timeout calling embedding service after {self.timeout}s: {e}\n"
                f"  Your embedding service may need more time for {len(texts)} texts.\n"
                f"  Consider increasing EMBEDDING_TIMEOUT in .env"
            )
            raise
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP {e.response.status_code} error from embedding service: {e}\n"
                f"  Response: {e.response.text[:200]}"
            )
            raise
        except httpx.HTTPError as e:
            logger.error(f"HTTP error calling embedding service: {type(e).__name__}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error generating embeddings: {type(e).__name__}: {str(e)}")
            raise

    async def health_check(self) -> bool:
        """
        Check if your embedding service is available.

        Calls GET /health endpoint on your embedding service.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            response = await self.client.get(
                f"{self.base_url}/health",
                timeout=5.0  # Short timeout for health checks
            )

            if response.status_code != 200:
                logger.warning(f"Embedding service returned status {response.status_code}")
                return False

            data = response.json()
            status = data.get("status", "unknown")

            is_healthy = status == "healthy"

            if is_healthy:
                logger.debug("✓ Embedding service is healthy")
            else:
                logger.warning(f"✗ Embedding service status: {status}")

            return is_healthy

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_info(self) -> dict:
        """
        Get provider information.

        Returns:
            Dictionary with provider metadata
        """
        return {
            "provider": "local",
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "description": "Local embedding microservice",
        }

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# =============================================================================
# OpenAI Embedding Provider (Fallback)
# =============================================================================

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider using OpenAI API.

    This is a fallback option if local service is unavailable.
    Note: This requires OPENAI_API_KEY environment variable.

    Example:
        provider = OpenAIEmbeddingProvider(
            api_key="sk-...",
            model="text-embedding-3-small"
        )
        embeddings = await provider.embed(["text1", "text2"])
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small"
    ):
        """
        Initialize OpenAI embedding provider.

        Args:
            api_key: OpenAI API key
            model: Model name (e.g., "text-embedding-3-small")
        """
        self.api_key = api_key
        self.model = model
        self.client = OpenAIEmbeddings(
            api_key=api_key,
            model=model
        )

        logger.info(f"Initialized OpenAIEmbeddingProvider: {model}")

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using OpenAI API.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        logger.debug(f"Requesting embeddings for {len(texts)} texts from OpenAI")

        try:
            # LangChain's OpenAIEmbeddings has async support
            embeddings = await self.client.aembed_documents(texts)

            logger.info(f"✓ Generated {len(embeddings)} embeddings from OpenAI")

            return embeddings

        except Exception as e:
            logger.error(f"Error generating OpenAI embeddings: {e}")
            raise

    async def health_check(self) -> bool:
        """
        Check if OpenAI API is available.

        Returns:
            True if API key is set, False otherwise
        """
        # Simple check: API key is set
        return bool(self.api_key)

    def get_info(self) -> dict:
        """Get provider information."""
        return {
            "provider": "openai",
            "model": self.model,
            "api_key_set": bool(self.api_key),
            "description": "OpenAI Embeddings API",
        }


# =============================================================================
# Provider Factory
# =============================================================================

def get_embedding_provider(
    provider_type: str,
    **kwargs
) -> EmbeddingProvider:
    """
    Factory function to create embedding providers.

    Args:
        provider_type: Type of provider ("local" or "openai")
        **kwargs: Provider-specific arguments

    Returns:
        EmbeddingProvider instance

    Raises:
        ValueError: If provider_type is unknown

    Example:
        # Use your local service
        provider = get_embedding_provider(
            "local",
            base_url="http://localhost:8001"
        )

        # Use OpenAI (fallback)
        provider = get_embedding_provider(
            "openai",
            api_key="sk-...",
            model="text-embedding-3-small"
        )
    """
    if provider_type == "local":
        return LocalEmbeddingProvider(**kwargs)
    elif provider_type == "openai":
        return OpenAIEmbeddingProvider(**kwargs)
    else:
        raise ValueError(
            f"Unknown embedding provider: {provider_type}. "
            f"Supported: 'local', 'openai'"
        )