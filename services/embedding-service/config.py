"""
Configuration Management Module

Provides thread-safe configuration management with support for:
- Environment variable configuration (default)
- Database-backed configuration (optional, for runtime changes)
- Immutable settings with validation
- Change detection and version tracking

Author: Embedding Service Team
"""

import os
import threading
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

__all__ = ["Settings", "SettingsManager", "settings_manager", "load_settings_from_env"]


# =============================================================================
# Constants
# =============================================================================

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8001
DEFAULT_DEVICE = "cuda"
DEFAULT_MAX_BATCH_SIZE = 32

VALID_DEVICES = {"cpu", "cuda", "auto"}


# =============================================================================
# Settings Model
# =============================================================================

class Settings(BaseModel):
    """
    Immutable service settings.

    All settings are validated on creation to ensure system consistency.
    Changes require creating a new Settings instance.
    """

    # Service Identity
    service_name: str = Field(
        default="embedding-service",
        description="Service identifier for multi-service deployments"
    )
    service_version: str = Field(
        default="1.0.0",
        description="Semantic version for API compatibility tracking"
    )

    # Network Configuration
    host: str = Field(
        default=DEFAULT_HOST,
        description="Network interface to bind (0.0.0.0 for all interfaces)"
    )
    port: int = Field(
        default=DEFAULT_PORT,
        ge=1024,
        le=65535,
        description="TCP port for HTTP service"
    )

    # Model Configuration
    model_name: str = Field(
        default=DEFAULT_MODEL_NAME,
        min_length=1,
        description="HuggingFace model identifier or local path"
    )
    model_cache_dir: str = Field(
        default="./model_cache",
        description="Directory for caching downloaded models"
    )
    device: str = Field(
        default=DEFAULT_DEVICE,
        description="Compute device: 'cpu', 'cuda', or 'auto' for GPU-CPU offloading"
    )
    normalize_embeddings: bool = Field(
        default=True,
        description="Apply L2 normalization to embeddings (recommended for cosine similarity)"
    )
    max_batch_size: int = Field(
        default=DEFAULT_MAX_BATCH_SIZE,
        ge=1,
        le=1024,
        description="Maximum number of texts to process in a single batch"
    )

    # API Configuration
    cors_origins: List[str] = Field(
        default=["*"],
        description="CORS allowed origins (use ['*'] for development only)"
    )

    # Feature Flags
    use_db_settings: bool = Field(
        default=False,
        description="Load and persist settings from MySQL database"
    )

    # Validators
    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Ensure device is a recognized value."""
        if v not in VALID_DEVICES:
            raise ValueError(
                f"Invalid device '{v}'. Must be one of: {', '.join(VALID_DEVICES)}"
            )
        return v

    @field_validator("cors_origins")
    @classmethod
    def validate_cors(cls, v: List[str]) -> List[str]:
        """Ensure CORS origins are non-empty."""
        if not v:
            raise ValueError("cors_origins cannot be empty")
        return v

    class Config:
        frozen = True  # Immutable after creation
        str_strip_whitespace = True


# =============================================================================
# Settings Loader
# =============================================================================

def load_settings_from_env() -> Settings:
    """
    Load settings from environment variables.

    Returns:
        Validated Settings instance

    Environment Variables:
        EMBEDDING_HOST: Network host (default: 0.0.0.0)
        EMBEDDING_PORT: Service port (default: 8001)
        EMBEDDING_MODEL_NAME: HuggingFace model (default: BAAI/bge-small-en-v1.5)
        EMBEDDING_MODEL_CACHE_DIR: Model cache path (default: ./model_cache)
        EMBEDDING_DEVICE: Device type (default: cuda)
        EMBEDDING_NORMALIZE: Normalize embeddings (default: true)
        EMBEDDING_MAX_BATCH_SIZE: Batch size (default: 32)
        CORS_ORIGINS: Comma-separated origins (default: *)
        EMBEDDING_USE_DB_SETTINGS: Enable DB config (default: false)
    """
    return Settings(
        host=os.getenv("EMBEDDING_HOST", DEFAULT_HOST),
        port=int(os.getenv("EMBEDDING_PORT", str(DEFAULT_PORT))),
        model_name=os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_MODEL_NAME),
        model_cache_dir=os.getenv("EMBEDDING_MODEL_CACHE_DIR", "./model_cache"),
        device=os.getenv("EMBEDDING_DEVICE", DEFAULT_DEVICE),
        normalize_embeddings=os.getenv("EMBEDDING_NORMALIZE", "true").lower() == "true",
        max_batch_size=int(os.getenv("EMBEDDING_MAX_BATCH_SIZE", str(DEFAULT_MAX_BATCH_SIZE))),
        cors_origins=[
            origin.strip()
            for origin in os.getenv("CORS_ORIGINS", "*").split(",")
            if origin.strip()
        ],
        use_db_settings=os.getenv("EMBEDDING_USE_DB_SETTINGS", "false").lower() == "true",
    )


# =============================================================================
# Settings Manager
# =============================================================================

class SettingsManager:
    """
    Thread-safe settings manager with atomic updates.

    Provides:
    - Safe concurrent read access via property
    - Atomic updates with version tracking
    - Change detection for model reload decisions

    Usage:
        >>> manager = SettingsManager()
        >>> current = manager.current  # Thread-safe read
        >>> manager.update(new_settings)  # Atomic write
    """

    # Keys that require model reload when changed
    _MODEL_RELOAD_KEYS = frozenset({
        "model_name",
        "model_cache_dir",
        "device",
        "normalize_embeddings",
        "max_batch_size"
    })

    def __init__(self, initial: Optional[Settings] = None):
        """
        Initialize the settings manager.

        Args:
            initial: Initial settings (defaults to environment-based settings)
        """
        self._settings: Settings = initial or load_settings_from_env()
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        self._version: int = 0

    @property
    def current(self) -> Settings:
        """
        Get current settings in a thread-safe manner.

        Returns:
            Current Settings instance (immutable)
        """
        with self._lock:
            return self._settings

    @property
    def version(self) -> int:
        """
        Get current settings version number.

        Version increments on each update, useful for cache invalidation.

        Returns:
            Current version number
        """
        with self._lock:
            return self._version

    def update(self, new_settings: Settings) -> Settings:
        """
        Atomically update settings.

        Args:
            new_settings: New Settings instance to apply

        Returns:
            Previous Settings instance (for rollback if needed)
        """
        with self._lock:
            old_settings = self._settings
            self._settings = new_settings
            self._version += 1
            return old_settings

    def requires_model_reload(self, new_settings: Settings) -> bool:
        """
        Determine if settings change requires embedding model reload.

        Args:
            new_settings: Proposed new settings

        Returns:
            True if any model-related setting has changed
        """
        current = self.current
        return any(
            getattr(current, key) != getattr(new_settings, key)
            for key in self._MODEL_RELOAD_KEYS
        )


# =============================================================================
# Global Singleton Instance
# =============================================================================

# Thread-safe singleton manager
settings_manager = SettingsManager()


# =============================================================================
# Convenience Proxy (Backward Compatibility)
# =============================================================================

class _SettingsProxy:
    """
    Proxy object for convenient settings access.

    Usage:
        >>> from config import settings
        >>> print(settings.model_name)  # Instead of settings_manager.current.model_name
    """

    def __getattr__(self, name: str):
        """Delegate attribute access to current settings."""
        return getattr(settings_manager.current, name)

    def model_dump(self):
        """Get settings as dictionary."""
        return settings_manager.current.model_dump()


# Singleton proxy instance for backward compatibility
settings = _SettingsProxy()