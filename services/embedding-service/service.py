"""
Embedding Model Service Layer

Provides:
- Embedding model lifecycle management
- GPU-CPU offloading support
- Thread-safe operations
- Request metrics and monitoring
- Automatic resource cleanup

Author: Embedding Service Team
"""

import asyncio
import logging
import torch
from dataclasses import dataclass, field
from time import time
from typing import List, Optional, Dict, Any

from sentence_transformers import SentenceTransformer

__all__ = [
    "EmbeddingModel",
    "ModelManager",
    "Metrics",
    "model_manager",
    "metrics",
    "get_embedding_model",
    "initialize_model",
    "cleanup_model",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================

class ModelNotLoadedError(RuntimeError):
    """Raised when attempting to use an unloaded model."""

    def __init__(self, message: str = "Embedding model is not loaded"):
        super().__init__(message)


class ModelLoadError(RuntimeError):
    """Raised when model fails to load."""

    def __init__(self, model_name: str, reason: str):
        self.model_name = model_name
        self.reason = reason
        super().__init__(f"Failed to load model '{model_name}': {reason}")


# =============================================================================
# Metrics
# =============================================================================

@dataclass
class Metrics:
    """
    Thread-safe service metrics.

    Tracks:
    - Request counts (success/failure)
    - Embedding generation counts
    - Processing times
    - Model reload events
    - Service uptime
    """

    requests_total: int = 0
    requests_failed: int = 0
    embeddings_generated: int = 0
    total_processing_ms: float = 0.0
    model_reloads: int = 0
    model_reload_failures: int = 0
    _start_time: float = field(default_factory=time)

    @property
    def uptime_seconds(self) -> float:
        """Get service uptime in seconds."""
        return time() - self._start_time

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.requests_total == 0:
            return 100.0
        return ((self.requests_total - self.requests_failed) / self.requests_total) * 100

    def record_request(
        self,
        success: bool,
        latency_ms: float,
        embedding_count: int = 0
    ) -> None:
        """
        Record a request completion.

        Args:
            success: Whether the request succeeded
            latency_ms: Processing time in milliseconds
            embedding_count: Number of embeddings generated
        """
        self.requests_total += 1
        if not success:
            self.requests_failed += 1
        else:
            self.embeddings_generated += embedding_count
        self.total_processing_ms += latency_ms

    def record_model_reload(self, success: bool) -> None:
        """
        Record a model reload event.

        Args:
            success: Whether the reload succeeded
        """
        self.model_reloads += 1
        if not success:
            self.model_reload_failures += 1

    def to_dict(self) -> Dict[str, Any]:
        """
        Export metrics as dictionary.

        Returns:
            Dictionary with all metrics
        """
        return {
            "requests_total": self.requests_total,
            "requests_failed": self.requests_failed,
            "success_rate": round(self.success_rate, 2),
            "embeddings_generated": self.embeddings_generated,
            "avg_latency_ms": round(
                self.total_processing_ms / max(1, self.requests_total), 2
            ),
            "model_reloads": self.model_reloads,
            "model_reload_failures": self.model_reload_failures,
            "uptime_seconds": round(self.uptime_seconds, 1),
        }


# Global metrics instance
metrics = Metrics()


# =============================================================================
# Embedding Model
# =============================================================================

class EmbeddingModel:
    """
    Wrapper for sentence-transformer embedding models.

    Features:
    - GPU-CPU offloading via device_map='auto'
    - Thread-safe encoding
    - Automatic batching
    - Resource cleanup

    Usage:
        >>> model = EmbeddingModel(model_name="BAAI/bge-small-en-v1.5")
        >>> model.load()
        >>> embeddings = await model.encode(["hello", "world"])
        >>> model.unload()
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cuda",
        cache_dir: str = "./model_cache",
        normalize_embeddings: bool = True,
        max_batch_size: int = 32,
    ):
        """
        Initialize embedding model configuration.

        Args:
            model_name: HuggingFace model identifier or local path
            device: Target device ('cpu', 'cuda', or 'auto' for offloading)
            cache_dir: Directory for caching downloaded models
            normalize_embeddings: Apply L2 normalization to outputs
            max_batch_size: Maximum batch size for encoding
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.normalize_embeddings = normalize_embeddings
        self.max_batch_size = max_batch_size

        self._model: Optional[SentenceTransformer] = None
        self._dimension: Optional[int] = None
        self._lock = asyncio.Lock()
        self._load_time: Optional[float] = None

    def load(self) -> None:
        """
        Load the embedding model into memory.

        Raises:
            ModelLoadError: If model fails to load
        """
        logger.info(f"Loading embedding model: {self.model_name}")
        start = time()

        try:
            # Configure device placement
            model_kwargs = {}
            target_device = self.device

            if self.device == "auto":
                logger.info("Using 'auto' device: enabling GPU-CPU offloading")
                model_kwargs["device_map"] = "auto"
                target_device = None  # Let accelerate handle placement
            elif self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but unavailable, falling back to CPU")
                target_device = "cpu"

            # Load model
            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_dir,
                device=target_device,
                model_kwargs=model_kwargs,
            )

            # Determine embedding dimension via test encoding
            test_embedding = self._model.encode(
                ["dimensionality test"],
                convert_to_numpy=True
            )
            self._dimension = test_embedding.shape[1]
            self._load_time = time() - start

            # Log placement information
            actual_device = getattr(self._model, "device", target_device)
            logger.info(
                f"Model loaded successfully | "
                f"Time: {self._load_time:.2f}s | "
                f"Device: {actual_device} | "
                f"Dimension: {self._dimension}"
            )

        except Exception as e:
            logger.error(f"Failed to load model '{self.model_name}': {e}")
            raise ModelLoadError(self.model_name, str(e)) from e

    def unload(self) -> None:
        """
        Unload model and free resources.

        Clears GPU memory if CUDA is available.
        """
        if self._model is not None:
            del self._model
            self._model = None
            self._dimension = None

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"Model unloaded: {self.model_name}")

    @property
    def dimension(self) -> int:
        """
        Get embedding dimension.

        Returns:
            Embedding vector dimension

        Raises:
            ModelNotLoadedError: If model is not loaded
        """
        if self._dimension is None:
            raise ModelNotLoadedError()
        return self._dimension

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._model is not None

    async def encode(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Encode texts into embeddings asynchronously.

        Args:
            texts: List of texts to encode
            batch_size: Override default batch size (optional)

        Returns:
            List of embedding vectors

        Raises:
            ModelNotLoadedError: If model is not loaded
        """
        if not self.is_loaded:
            raise ModelNotLoadedError()

        async with self._lock:
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                self._encode_sync,
                texts,
                batch_size
            )
            return embeddings

    def _encode_sync(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Synchronous encoding implementation.

        Args:
            texts: Texts to encode
            batch_size: Batch size override

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        batch_size = batch_size or self.max_batch_size

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed model information.

        Returns:
            Dictionary with model metadata
        """
        info = {
            "model_name": self.model_name,
            "device_config": self.device,
            "is_loaded": self.is_loaded,
            "normalize_embeddings": self.normalize_embeddings,
            "max_batch_size": self.max_batch_size,
        }

        if self.is_loaded:
            info["dimension"] = self._dimension
            info["load_time_seconds"] = round(self._load_time or 0.0, 2)

            # Add GPU memory info if available
            if torch.cuda.is_available():
                info["gpu_memory_allocated_mb"] = round(
                    torch.cuda.memory_allocated() / (1024 ** 2), 2
                )
                info["gpu_memory_reserved_mb"] = round(
                    torch.cuda.memory_reserved() / (1024 ** 2), 2
                )

        return info


# =============================================================================
# Model Manager
# =============================================================================

class ModelManager:
    """
    Singleton manager for embedding model lifecycle.

    Provides:
    - Single point of access to embedding model
    - Thread-safe model reloading
    - Metrics tracking

    Usage:
        >>> manager = ModelManager()
        >>> manager.initialize(model_name="...")
        >>> model = manager.model
        >>> await manager.reload(model_name="new_model")
    """

    def __init__(self):
        self._model: Optional[EmbeddingModel] = None
        self._lock = asyncio.Lock()

    @property
    def model(self) -> EmbeddingModel:
        """
        Get the current embedding model.

        Returns:
            Current EmbeddingModel instance

        Raises:
            ModelNotLoadedError: If model is not initialized
        """
        if self._model is None:
            raise ModelNotLoadedError("Model manager not initialized")
        return self._model

    def initialize(self, **kwargs) -> EmbeddingModel:
        """
        Initialize the embedding model.

        Args:
            **kwargs: Arguments passed to EmbeddingModel constructor

        Returns:
            Initialized EmbeddingModel instance

        Raises:
            ModelLoadError: If initialization fails
        """
        self._model = EmbeddingModel(**kwargs)
        self._model.load()
        return self._model

    async def reload(self, **kwargs) -> bool:
        """
        Hot-reload the embedding model with new configuration.

        Args:
            **kwargs: New model configuration

        Returns:
            True if reload succeeded, False otherwise
        """
        async with self._lock:
            new_model = EmbeddingModel(**kwargs)

            try:
                # Load new model in executor to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, new_model.load)

                # Swap models atomically
                old_model = self._model
                self._model = new_model

                # Clean up old model
                if old_model:
                    old_model.unload()

                metrics.record_model_reload(success=True)
                logger.info(f"Model reloaded successfully: {kwargs.get('model_name')}")
                return True

            except Exception as e:
                metrics.record_model_reload(success=False)
                logger.error(f"Model reload failed: {e}")
                return False


# =============================================================================
# Global Instances
# =============================================================================

# Singleton model manager
model_manager = ModelManager()


# =============================================================================
# Convenience Functions
# =============================================================================

def get_embedding_model() -> EmbeddingModel:
    """
    Get the current embedding model.

    Returns:
        Current EmbeddingModel instance
    """
    return model_manager.model


def initialize_model(**kwargs) -> EmbeddingModel:
    """
    Initialize the embedding model.

    Args:
        **kwargs: Model configuration parameters

    Returns:
        Initialized model
    """
    return model_manager.initialize(**kwargs)


def cleanup_model() -> None:
    """Cleanup and unload the current model."""
    try:
        model_manager.model.unload()
    except ModelNotLoadedError:
        pass