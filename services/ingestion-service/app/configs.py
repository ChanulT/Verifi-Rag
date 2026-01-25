"""
Configuration management for the Ingestion Service.

Supports:
- Environment variable configuration (default)
- Database-backed configuration (optional, for runtime changes)
- Thread-safe settings access

Mirrors the pattern used in embedding-service/config.py
"""

import os
import threading
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
load_dotenv = True

__all__ = ["Settings", "SettingsManager", "settings_manager", "load_settings_from_env"]


class Settings(BaseModel):
    """Immutable service settings."""

    # Service identity
    service_name: str = "ingestion-service"
    service_version: str = "1.0.0"

    # Network
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8002, ge=1024, le=65535)

    # PDF Extraction (Docling)
    extraction_service: str = Field(
        default="docling",
        description="Extraction service: 'docling' or 'lighton_ocr'"
    )

    # Docling configuration
    enable_ocr: bool = Field(default=False, description="Enable OCR for scanned PDFs")
    images_scale: float = Field(default=1.0, ge=0.5, le=4.0)
    include_images: bool = Field(default=True)
    include_tables: bool = Field(default=True)

    # LightOnOCR configuration
    ocr_model_name: str = Field(
        default="lightonai/LightOnOCR-1B-1025",
        description="HuggingFace model for OCR"
    )
    ocr_device: str = Field(
        default="cpu",
        description="Device for OCR model: 'cpu' or 'cuda'"
    )
    ocr_dpi: int = Field(
        default=200,
        ge=100,
        le=600,
        description="DPI for PDF to image conversion"
    )

    # Chunking configuration
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    min_chunk_size: int = Field(default=100, ge=50)
    max_chunk_size: int = Field(default=2000, ge=500)
    use_semantic_chunking: bool = Field(default=True)

    # Embedding configuration
    embedding_provider: str = Field(
        default="local",
        description="Embedding provider: 'local' (your service) or 'openai'"
    )
    embedding_service_url: str = Field(
        default="http://localhost:8001",
        description="URL of local embedding service"
    )
    embedding_timeout_seconds: int = Field(
        default=600,
        ge=30,
        le=3600,
        description="Timeout for embedding requests (600s = 10min for CPU)"
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key (only if using openai provider)"
    )
    openai_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI model name"
    )

    # Storage
    upload_dir: str = Field(default="./uploads")
    cache_dir: str = Field(default="./cache")
    max_file_size_mb: int = Field(default=50, ge=1, le=500)

    # Processing
    max_concurrent_jobs: int = Field(default=10, ge=1, le=100)
    job_timeout_seconds: int = Field(default=300, ge=30)

    # API configuration
    cors_origins: List[str] = Field(default=["*"])

    # Feature flags
    use_db_settings: bool = Field(
        default=False,
        description="Load/save settings from database when true"
    )
    save_intermediate: bool = Field(
        default=True,
        description="Save intermediate JSON files for inspection"
    )
    enable_metrics: bool = Field(default=True)

    @field_validator("chunk_overlap")
    def validate_overlap(cls, v, info):
        """Ensure overlap is less than chunk_size."""
        chunk_size = info.data.get("chunk_size", 1000)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

    class Config:
        frozen = True  # Make settings immutable after creation


def load_settings_from_env() -> Settings:
    """Load settings from environment variables."""
    return Settings(
        # Service
        host=os.getenv("INGESTION_HOST", "0.0.0.0"),
        port=int(os.getenv("INGESTION_PORT", "8002")),

        # Extraction
        extraction_service=os.getenv("EXTRACTION_SERVICE", "docling"),

        # PDF Extraction - Docling
        enable_ocr=os.getenv("ENABLE_OCR", "false").lower() == "true",
        images_scale=float(os.getenv("IMAGES_SCALE", "1.0")),
        include_images=os.getenv("INCLUDE_IMAGES", "true").lower() == "true",
        include_tables=os.getenv("INCLUDE_TABLES", "true").lower() == "true",

        # PDF Extraction - LightOnOCR
        ocr_model_name=os.getenv("OCR_MODEL_NAME", "lightonai/LightOnOCR-1B-1025"),
        ocr_device=os.getenv("OCR_DEVICE", "cpu"),
        ocr_dpi=int(os.getenv("OCR_DPI", "200")),

        # Chunking
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        min_chunk_size=int(os.getenv("MIN_CHUNK_SIZE", "100")),
        max_chunk_size=int(os.getenv("MAX_CHUNK_SIZE", "2000")),
        use_semantic_chunking=os.getenv("USE_SEMANTIC_CHUNKING", "true").lower() == "true",

        # Embedding
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "local"),
        embedding_service_url=os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8001"),
        embedding_timeout_seconds=int(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "600")),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "text-embedding-3-small"),

        # Storage
        upload_dir=os.getenv("UPLOAD_DIR", "./uploads"),
        cache_dir=os.getenv("CACHE_DIR", "./cache"),
        max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "50")),

        # Processing
        max_concurrent_jobs=int(os.getenv("MAX_CONCURRENT_JOBS", "10")),
        job_timeout_seconds=int(os.getenv("JOB_TIMEOUT_SECONDS", "300")),

        # API
        cors_origins=[
            x.strip()
            for x in os.getenv("CORS_ORIGINS", "*").split(",")
            if x.strip()
        ],

        # Features
        use_db_settings=os.getenv("INGESTION_USE_DB_SETTINGS", "false").lower() == "true",
        save_intermediate=os.getenv("SAVE_INTERMEDIATE", "true").lower() == "true",
        enable_metrics=os.getenv("ENABLE_METRICS", "true").lower() == "true",
    )


class SettingsManager:
    """
    Thread-safe settings manager.

    Provides atomic access to settings, preventing race conditions
    when settings are updated while requests are in flight.

    Mirrors the pattern from embedding-service/config.py
    """

    def __init__(self, initial: Optional[Settings] = None):
        self._settings: Settings = initial or load_settings_from_env()
        self._lock = threading.RLock()
        self._version: int = 0  # Track changes for cache invalidation

    @property
    def current(self) -> Settings:
        """Get current settings (thread-safe read)."""
        with self._lock:
            return self._settings

    @property
    def version(self) -> int:
        """Get settings version for change detection."""
        with self._lock:
            return self._version

    def update(self, new_settings: Settings) -> Settings:
        """
        Update settings atomically.

        Returns the old settings after update.
        """
        with self._lock:
            old = self._settings
            self._settings = new_settings
            self._version += 1
            return old

    def requires_restart(self, new_settings: Settings) -> bool:
        """Check if settings change requires service restart."""
        current = self.current
        restart_keys = ["host", "port", "max_concurrent_jobs"]
        return any(
            getattr(current, k) != getattr(new_settings, k)
            for k in restart_keys
        )

    def ensure_directories(self):
        """Create required directories if they don't exist."""
        import pathlib
        current = self.current
        pathlib.Path(current.upload_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(current.cache_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(current.cache_dir, "intermediate").mkdir(parents=True, exist_ok=True)


# Global singleton - thread-safe by design
settings_manager = SettingsManager()


# Convenience proxy for backward compatibility
class _SettingsProxy:
    """Proxy that delegates to settings_manager.current."""

    def __getattr__(self, name: str):
        return getattr(settings_manager.current, name)

    def model_dump(self):
        return settings_manager.current.model_dump()


settings = _SettingsProxy()