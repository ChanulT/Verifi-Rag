"""
API Routes for Embedding Service

Provides RESTful endpoints for:
- Embedding generation
- Health monitoring
- Metrics collection
- Dynamic configuration management

Author: Embedding Service Team
"""

import logging
import time
import uuid
from contextvars import ContextVar
from typing import List, Any, Dict, Optional

from fastapi import APIRouter, HTTPException, status, Depends, Body
from pydantic import BaseModel, Field, field_validator

from config import settings_manager, load_settings_from_env, Settings
from service import model_manager, metrics, get_embedding_model, ModelNotLoadedError
from database import db_manager, SettingsRepository, SUPPORTED_KEYS

__all__ = ["router"]

logger = logging.getLogger(__name__)


# =============================================================================
# Request Context
# =============================================================================

# Request ID for distributed tracing
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    """
    Get or generate a unique request ID for tracing.

    Returns:
        8-character request ID
    """
    rid = request_id_var.get()
    if not rid:
        rid = str(uuid.uuid4())[:8]
        request_id_var.set(rid)
    return rid


# =============================================================================
# Request/Response Schemas
# =============================================================================

class EmbeddingRequest(BaseModel):
    """Request schema for embedding generation."""

    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of texts to embed (1-1000 texts)"
    )

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        """Ensure texts are non-empty strings."""
        if not v:
            raise ValueError("texts list cannot be empty")

        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f"Text at index {i} is empty or whitespace only")

        return v


class EmbeddingResponse(BaseModel):
    """Response schema for embedding generation."""

    embeddings: List[List[float]] = Field(
        ...,
        description="Generated embedding vectors"
    )
    model_name: str = Field(..., description="Model used for generation")
    dimension: int = Field(..., description="Embedding vector dimension")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    request_id: str = Field(..., description="Unique request identifier")


class SingleEmbeddingResponse(BaseModel):
    """Response schema for single text embedding."""

    embedding: List[float] = Field(..., description="Generated embedding vector")
    dimension: int = Field(..., description="Embedding vector dimension")
    model_name: str = Field(..., description="Model used for generation")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class HealthStatus(BaseModel):
    """Service health status response."""

    service: str = Field(..., description="Service name")
    status: str = Field(..., description="Overall health status")
    version: str = Field(..., description="Service version")
    uptime_seconds: float = Field(..., description="Service uptime")
    dependencies: Dict[str, str] = Field(..., description="Dependency status")


class MetricsResponse(BaseModel):
    """Service metrics response."""

    requests_total: int = Field(..., description="Total requests processed")
    requests_failed: int = Field(..., description="Failed requests")
    success_rate: float = Field(..., description="Success rate percentage")
    embeddings_generated: int = Field(..., description="Total embeddings generated")
    avg_latency_ms: float = Field(..., description="Average processing latency")
    model_reloads: int = Field(..., description="Model reload count")
    model_reload_failures: int = Field(..., description="Failed reload count")
    uptime_seconds: float = Field(..., description="Service uptime")


class SettingRequest(BaseModel):
    """Request to update a single setting."""

    key: str = Field(
        ...,
        min_length=1,
        description="Setting key (must be in SUPPORTED_KEYS)"
    )
    value: Any = Field(..., description="Setting value")

    @field_validator("key")
    @classmethod
    def validate_key(cls, v: str) -> str:
        """Ensure key is supported."""
        if v not in SUPPORTED_KEYS:
            raise ValueError(
                f"Unsupported key '{v}'. "
                f"Supported keys: {', '.join(sorted(SUPPORTED_KEYS))}"
            )
        return v


class BulkSettingsRequest(BaseModel):
    """Request to update multiple settings."""

    settings: Dict[str, Any] = Field(
        ...,
        min_length=1,
        description="Dictionary of setting key-value pairs"
    )

    @field_validator("settings")
    @classmethod
    def validate_settings(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all keys are supported."""
        unsupported = set(v.keys()) - SUPPORTED_KEYS
        if unsupported:
            raise ValueError(
                f"Unsupported keys: {', '.join(unsupported)}. "
                f"Supported: {', '.join(sorted(SUPPORTED_KEYS))}"
            )
        return v


class SettingsSnapshot(BaseModel):
    """Complete settings snapshot."""

    settings: Dict[str, Any] = Field(..., description="Current settings")
    supported_keys: List[str] = Field(..., description="All supported setting keys")
    source: str = Field(..., description="Configuration source (env or database)")


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")
    request_id: Optional[str] = Field(None, description="Request ID if available")


# =============================================================================
# Dependencies
# =============================================================================

settings_repo = SettingsRepository(service_name="embedding-service")


async def get_db_session():
    """
    Dependency for database session access.

    Yields:
        AsyncSession instance

    Raises:
        HTTPException: If database is not initialized
    """
    if not db_manager.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized. Check EMBEDDING_DB_URL configuration."
        )

    async with db_manager.session() as session:
        yield session


async def require_db_settings():
    """
    Dependency that enforces database settings are enabled.

    Raises:
        HTTPException: If database settings are disabled
    """
    if not settings_manager.current.use_db_settings:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Database-backed settings are disabled. "
                "Set EMBEDDING_USE_DB_SETTINGS=true in environment to enable."
            )
        )


# =============================================================================
# Router Setup
# =============================================================================

router = APIRouter()


# =============================================================================
# Embedding Endpoints
# =============================================================================

@router.post(
    "/embed",
    response_model=EmbeddingResponse,
    status_code=status.HTTP_200_OK,
    tags=["Embeddings"],
    summary="Generate embeddings for multiple texts",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Server error"},
    }
)
async def generate_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """
    Generate embeddings for a batch of texts.

    - Automatically batches texts for efficient processing
    - Uses GPU if available for acceleration
    - Returns L2-normalized embeddings ready for cosine similarity
    - Maximum 1000 texts per request

    Args:
        request: Embedding request with list of texts

    Returns:
        Embeddings with metadata

    Raises:
        HTTPException: On validation or processing errors
    """
    rid = get_request_id()
    start = time.perf_counter()

    try:
        # Get model
        try:
            model = get_embedding_model()
        except ModelNotLoadedError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Embedding model is not loaded. Service may be starting up."
            )

        logger.info(f"[{rid}] Embedding request: {len(request.texts)} texts")

        # Generate embeddings
        embeddings = await model.encode(request.texts)
        processing_time = (time.perf_counter() - start) * 1000

        # Record metrics
        metrics.record_request(
            success=True,
            latency_ms=processing_time,
            embedding_count=len(embeddings)
        )

        logger.info(
            f"[{rid}] Generated {len(embeddings)} embeddings "
            f"in {processing_time:.2f}ms"
        )

        return EmbeddingResponse(
            embeddings=embeddings,
            model_name=model.model_name,
            dimension=model.dimension,
            processing_time_ms=round(processing_time, 2),
            request_id=rid
        )

    except HTTPException:
        raise
    except Exception as e:
        processing_time = (time.perf_counter() - start) * 1000
        metrics.record_request(success=False, latency_ms=processing_time)
        logger.error(f"[{rid}] Embedding generation failed: {e}", exc_info=True)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {str(e)}"
        )


@router.post(
    "/embed/single",
    response_model=SingleEmbeddingResponse,
    status_code=status.HTTP_200_OK,
    tags=["Embeddings"],
    summary="Generate embedding for a single text"
)
async def generate_single_embedding(
    text: str = Body(..., min_length=1, description="Text to embed")
) -> SingleEmbeddingResponse:
    """
    Generate embedding for a single text.

    Convenience endpoint for single-text embedding.
    For multiple texts, use POST /embed instead.

    Args:
        text: Text to embed

    Returns:
        Single embedding with metadata
    """
    request = EmbeddingRequest(texts=[text])
    response = await generate_embeddings(request)

    return SingleEmbeddingResponse(
        embedding=response.embeddings[0],
        dimension=response.dimension,
        model_name=response.model_name,
        processing_time_ms=response.processing_time_ms
    )


# =============================================================================
# Health & Monitoring Endpoints
# =============================================================================

@router.get(
    "/health",
    response_model=HealthStatus,
    status_code=status.HTTP_200_OK,
    tags=["Health"],
    summary="Service health check"
)
async def health_check() -> HealthStatus:
    """
    Check service health and dependency status.

    Returns:
        - Service status (healthy/unhealthy)
        - Model information
        - Database status
        - Uptime metrics
    """
    current_settings = settings_manager.current

    try:
        model = get_embedding_model()
        model_info = model.get_model_info()

        # Check database if enabled
        db_status = "disabled"
        if current_settings.use_db_settings:
            db_health = await db_manager.health_check()
            db_status = db_health.get("status", "unknown")

        return HealthStatus(
            service=current_settings.service_name,
            status="healthy" if model.is_loaded else "degraded",
            version=current_settings.service_version,
            uptime_seconds=round(metrics.uptime_seconds, 2),
            dependencies={
                "model_loaded": str(model.is_loaded),
                "model_name": model_info.get("model_name", "unknown"),
                "device": model_info.get("device_config", "unknown"),
                "dimension": str(model_info.get("dimension", 0)),
                "database": db_status,
            }
        )

    except ModelNotLoadedError:
        return HealthStatus(
            service=current_settings.service_name,
            status="unhealthy",
            version=current_settings.service_version,
            uptime_seconds=round(metrics.uptime_seconds, 2),
            dependencies={"model": "not_loaded"}
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return HealthStatus(
            service=current_settings.service_name,
            status="unhealthy",
            version=current_settings.service_version,
            uptime_seconds=round(metrics.uptime_seconds, 2),
            dependencies={"error": str(e)}
        )


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    status_code=status.HTTP_200_OK,
    tags=["Observability"],
    summary="Get service metrics"
)
async def get_service_metrics() -> MetricsResponse:
    """
    Get detailed service metrics.

    Returns:
        - Request counts and success rate
        - Processing latency statistics
        - Model reload events
        - Uptime information
    """
    return MetricsResponse(**metrics.to_dict())


@router.get(
    "/model/info",
    status_code=status.HTTP_200_OK,
    tags=["Model"],
    summary="Get model information"
)
async def get_model_information() -> Dict[str, Any]:
    """
    Get detailed embedding model information.

    Returns:
        Model metadata including:
        - Model name and configuration
        - Device placement
        - Embedding dimension
        - Memory usage (if GPU)
    """
    try:
        model = get_embedding_model()
        return model.get_model_info()
    except ModelNotLoadedError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    except Exception as e:
        logger.error(f"Failed to get model info: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model information: {str(e)}"
        )


# =============================================================================
# Settings Management Endpoints
# =============================================================================

@router.get(
    "/settings",
    response_model=SettingsSnapshot,
    status_code=status.HTTP_200_OK,
    tags=["Settings"],
    summary="Get current settings"
)
async def get_current_settings() -> SettingsSnapshot:
    """
    Get the current effective settings.

    Shows merged result of environment variables and database overrides.
    """
    current = settings_manager.current
    return SettingsSnapshot(
        settings=current.model_dump(),
        supported_keys=sorted(SUPPORTED_KEYS),
        source="database" if current.use_db_settings else "environment"
    )


@router.post(
    "/settings",
    response_model=SettingsSnapshot,
    status_code=status.HTTP_200_OK,
    tags=["Settings"],
    summary="Update a setting",
    dependencies=[Depends(require_db_settings)]
)
async def update_setting(
    payload: SettingRequest,
    session=Depends(get_db_session)
) -> SettingsSnapshot:
    """
    Create or update a configuration setting.

    - Requires EMBEDDING_USE_DB_SETTINGS=true
    - Changes persist to database
    - Model-related changes trigger automatic hot-reload

    Args:
        payload: Setting key-value pair
        session: Database session (injected)

    Returns:
        Updated settings snapshot
    """
    try:
        # Save to database
        await settings_repo.upsert(session, payload.key, payload.value)

        # Load merged settings
        new_settings = await settings_repo.get_typed(
            session,
            defaults=load_settings_from_env()
        )

        # Check if model reload is needed
        if settings_manager.requires_model_reload(new_settings):
            logger.info(f"Setting '{payload.key}' requires model reload")

            reload_success = await model_manager.reload(
                model_name=new_settings.model_name,
                device=new_settings.device,
                cache_dir=new_settings.model_cache_dir,
                normalize_embeddings=new_settings.normalize_embeddings,
                max_batch_size=new_settings.max_batch_size,
            )

            if not reload_success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=(
                        "Setting saved but model reload failed. "
                        "Previous model remains active."
                    )
                )

        # Update global settings
        settings_manager.update(new_settings)

        return SettingsSnapshot(
            settings=new_settings.model_dump(),
            supported_keys=sorted(SUPPORTED_KEYS),
            source="database"
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to update setting: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Setting update failed: {str(e)}"
        )


@router.post(
    "/settings/bulk",
    response_model=SettingsSnapshot,
    status_code=status.HTTP_200_OK,
    tags=["Settings"],
    summary="Bulk update settings",
    dependencies=[Depends(require_db_settings)]
)
async def bulk_update_settings(
    payload: BulkSettingsRequest,
    session=Depends(get_db_session)
) -> SettingsSnapshot:
    """
    Update multiple settings in a single transaction.

    - Atomic operation (all or nothing)
    - Triggers single model reload if needed
    - More efficient than individual updates

    Args:
        payload: Dictionary of settings to update
        session: Database session (injected)

    Returns:
        Updated settings snapshot
    """
    try:
        # Atomic bulk update
        await settings_repo.bulk_upsert(session, payload.settings)

        # Load merged settings
        new_settings = await settings_repo.get_typed(
            session,
            defaults=load_settings_from_env()
        )

        # Single reload if needed
        if settings_manager.requires_model_reload(new_settings):
            logger.info("Bulk update requires model reload")
            await model_manager.reload(
                model_name=new_settings.model_name,
                device=new_settings.device,
                cache_dir=new_settings.model_cache_dir,
                normalize_embeddings=new_settings.normalize_embeddings,
                max_batch_size=new_settings.max_batch_size,
            )

        settings_manager.update(new_settings)

        return SettingsSnapshot(
            settings=new_settings.model_dump(),
            supported_keys=sorted(SUPPORTED_KEYS),
            source="database"
        )

    except Exception as e:
        logger.error(f"Bulk update failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete(
    "/settings/{key}",
    response_model=SettingsSnapshot,
    status_code=status.HTTP_200_OK,
    tags=["Settings"],
    summary="Delete a setting",
    dependencies=[Depends(require_db_settings)]
)
async def delete_setting(
    key: str,
    session=Depends(get_db_session)
) -> SettingsSnapshot:
    """
    Delete a setting from the database.

    The setting reverts to its environment variable default.

    Args:
        key: Setting key to delete
        session: Database session (injected)

    Returns:
        Updated settings snapshot
    """
    try:
        # Validate key
        if key not in SUPPORTED_KEYS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unsupported key: {key}"
            )

        # Delete from database
        deleted = await settings_repo.delete(session, key)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Setting '{key}' not found in database"
            )

        # Reload settings
        new_settings = await settings_repo.get_typed(
            session,
            defaults=load_settings_from_env()
        )

        # Reload model if necessary
        if settings_manager.requires_model_reload(new_settings):
            await model_manager.reload(
                model_name=new_settings.model_name,
                device=new_settings.device,
                cache_dir=new_settings.model_cache_dir,
                normalize_embeddings=new_settings.normalize_embeddings,
                max_batch_size=new_settings.max_batch_size,
            )

        settings_manager.update(new_settings)

        return SettingsSnapshot(
            settings=new_settings.model_dump(),
            supported_keys=sorted(SUPPORTED_KEYS),
            source="database"
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to delete setting: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )