"""
API Routes for Ingestion Service

Endpoints:
- POST /ingest - Upload and process document
- GET /jobs/{job_id} - Check job status
- GET /jobs/{job_id}/chunks - Get processed chunks
- GET /jobs/{job_id}/intermediate - Download intermediate JSON
- GET /health - Service health
- GET /metrics - Service metrics
"""

import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import (
    APIRouter,
    File,
    UploadFile,
    HTTPException,
    status,
    Form,
)
from fastapi.responses import FileResponse

from app.configs import settings_manager
from app.models import (
    IngestRequest,
    IngestResponse,
    JobStatus,
    JobStatusResponse,
    ChunkResponse,
    ChunksListResponse,
    HealthStatus,
    MetricsResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# Global Metrics (simple in-memory tracking)
# =============================================================================

class ServiceMetrics:
    """Simple metrics tracking."""

    def __init__(self):
        self.start_time = datetime.utcnow()
        self.jobs_total = 0
        self.jobs_completed = 0
        self.jobs_failed = 0

    @property
    def uptime_seconds(self) -> float:
        return (datetime.utcnow() - self.start_time).total_seconds()

    @property
    def success_rate(self) -> float:
        if self.jobs_total == 0:
            return 100.0
        return (self.jobs_completed / self.jobs_total) * 100


metrics = ServiceMetrics()


# =============================================================================
# Ingestion Endpoints
# =============================================================================

@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Ingestion"],
    summary="Upload and process document"
)
async def ingest_document(
        file: UploadFile = File(..., description="PDF document to process"),
        use_semantic_chunking: Optional[bool] = Form(None),
        chunk_size: Optional[int] = Form(None),
        chunk_overlap: Optional[int] = Form(None),
        enable_ocr: Optional[bool] = Form(None),
        include_tables: Optional[bool] = Form(None),
        include_images: Optional[bool] = Form(None),
        extraction_service: Optional[str] = Form(None),
        save_intermediate: Optional[bool] = Form(None),
        generate_embeddings: bool = Form(True),
        save_to_database: bool = Form(True),
):
    """
    Upload and process a PDF document.

    **Workflow:**
    1. Upload file → Returns job_id
    2. Document is extracted (Docling)
    3. Text is chunked (semantic/recursive)
    4. Embeddings generated (using your local service!)
    5. Intermediate JSON saved for inspection
    6. Results stored in database

    **Integration with Your Embedding Service:**
    - If `EMBEDDING_PROVIDER=local`, this uses your service at port 8001
    - Calls `POST http://localhost:8001/embed` with chunk texts
    - Receives embeddings and attaches to chunks

    **Returns:**
    - job_id: Use this to poll for status
    - intermediate_file: Path to JSON (available after processing)
    """
    settings = settings_manager.current

    # Validate file type
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Only PDF files are supported. Got: {file.content_type}"
        )

    # Validate file size (rough estimate)
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset

    max_size_bytes = settings.max_file_size_mb * 1024 * 1024
    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum of {settings.max_file_size_mb}MB"
        )

    # Save uploaded file
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_id = str(uuid.uuid4())
    file_path = upload_dir / f"{file_id}_{file.filename}"

    try:
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}"
        )

    # Get orchestrator from app state
    from app.main import orchestrator

    if not orchestrator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not fully initialized"
        )

    # Start ingestion
    try:
        job_id = await orchestrator.ingest(
            file_path=str(file_path),
            filename=file.filename,
            file_size_bytes=file_size,
            generate_embeddings=generate_embeddings,
            save_intermediate=save_intermediate if save_intermediate is not None else settings.save_intermediate,
            save_to_database=save_to_database,
            metadata={
                "use_semantic_chunking": use_semantic_chunking,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "enable_ocr": enable_ocr,
                "include_tables": include_tables,
                "include_images": include_images,
                "extraction_service": extraction_service,
            }
        )

        metrics.jobs_total += 1

        # Estimate processing time
        estimated_time = (file_size / (1024 * 1024)) * 10  # ~10s per MB

        return IngestResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            filename=file.filename,
            file_size_mb=round(file_size / (1024 * 1024), 2),
            estimated_time_seconds=round(estimated_time, 1),
            message="Document queued for processing"
        )

    except Exception as e:
        logger.error(f"Failed to start ingestion: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start processing: {str(e)}"
        )


# =============================================================================
# Job Status Endpoints
# =============================================================================

@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    tags=["Jobs"],
    summary="Get job status"
)
async def get_job_status(job_id: str):
    """
    Get the current status of a processing job.

    **Poll this endpoint** to track progress (progress field: 0-100%)
    """
    from app.main import orchestrator

    job = await orchestrator.get_job_status(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    return JobStatusResponse(
        job_id=job["job_id"],
        status=JobStatus(job["status"]),
        progress=job["progress"],
        document_id=job.get("document_id"),
        chunks_created=job.get("chunks_created", 0),
        tables_extracted=job.get("tables_extracted", 0),
        pages_processed=job.get("pages_processed", 0),
        embeddings_generated=job.get("embeddings_generated", 0),
        created_at=job["created_at"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        processing_time_seconds=job.get("processing_time_seconds"),
        error_message=job.get("error_message"),
        error_type=job.get("error_type"),
        filename=job["filename"],
        file_size_mb=round(job["file_size_bytes"] / (1024 * 1024), 2),
        intermediate_file=job.get("intermediate_file"),
        metadata=job.get("metadata")
    )


@router.get(
    "/jobs/{job_id}/intermediate",
    tags=["Jobs"],
    summary="Download intermediate JSON"
)
async def get_intermediate_file(job_id: str):
    """
    Download intermediate processing results as JSON.

    **Contains:**
    - Full extraction results
    - All chunks with metadata
    - Processing statistics
    - Table/image counts

    **Use this to:**
    - Inspect extraction quality
    - Verify chunking before vector DB
    - Debug processing issues
    """
    settings = settings_manager.current

    # Find intermediate file
    intermediate_path = Path(settings.cache_dir) / "intermediate" / f"{job_id}.json"

    if not intermediate_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Intermediate file not found. Job may not have save_intermediate enabled."
        )

    return FileResponse(
        path=intermediate_path,
        filename=f"{job_id}_results.json",
        media_type="application/json"
    )


@router.get(
    "/jobs/{job_id}/chunks",
    response_model=ChunksListResponse,
    tags=["Jobs"],
    summary="Get processed chunks"
)
async def get_job_chunks(job_id: str):
    """
    Retrieve all processed chunks for a job.

    **Use this data to:**
    1. Send to vector database
    2. Verify chunk quality
    3. Export for other systems
    """
    from app.main import orchestrator

    # Check job exists and is completed
    job = await orchestrator.get_job_status(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    if job["status"] != JobStatus.COMPLETED.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job is not completed yet. Current status: {job['status']}"
        )

    # Get chunks
    document_id = job.get("document_id")
    if not document_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No document ID found for this job"
        )

    chunks = await orchestrator.get_document_chunks(document_id)

    if not chunks:
        return ChunksListResponse(
            job_id=job_id,
            document_id=document_id,
            total_chunks=0,
            chunks=[],
            avg_chunk_size=0.0,
            total_embeddings=0
        )

    # Convert to response schema
    chunk_responses = []
    total_size = 0
    total_embeddings = 0

    for chunk in chunks:
        metadata_dict = chunk.get("metadata", {})

        chunk_responses.append(
            ChunkResponse(
                chunk_id=chunk["chunk_id"],
                chunk_index=chunk["chunk_index"],
                content=chunk["content"],
                size=len(chunk["content"]),
                token_count=chunk.get("token_count", 0),
                page_number=metadata_dict.get("page_number"),
                section_title=metadata_dict.get("section_title"),
                has_embedding=chunk.get("has_embedding", False),
                metadata=metadata_dict
            )
        )

        total_size += len(chunk["content"])
        if chunk.get("has_embedding"):
            total_embeddings += 1

    avg_size = total_size / len(chunks) if chunks else 0

    return ChunksListResponse(
        job_id=job_id,
        document_id=document_id,
        total_chunks=len(chunks),
        chunks=chunk_responses,
        avg_chunk_size=round(avg_size, 1),
        total_embeddings=total_embeddings
    )


# =============================================================================
# Health & Metrics
# =============================================================================

@router.get(
    "/health",
    response_model=HealthStatus,
    tags=["Health"],
    summary="Service health check"
)
async def health_check():
    """
    Check service health and dependency status.
    """
    settings = settings_manager.current
    from app.main import orchestrator

    # Check embedding service
    embedding_status = "unknown"
    if orchestrator and orchestrator.embedding_service:
        is_healthy = await orchestrator.embedding_service.health_check()
        embedding_status = "healthy" if is_healthy else "unavailable"

    return HealthStatus(
        service=settings.service_name,
        status="healthy",
        version=settings.service_version,
        uptime_seconds=round(metrics.uptime_seconds, 1),
        dependencies={
            "embedding_service": f"{settings.embedding_service_url} ({embedding_status})",
            "embedding_provider": settings.embedding_provider,
            "database": "not_configured",
        }
    )


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["Observability"],
    summary="Service metrics"
)
async def get_metrics():
    """
    Get detailed service metrics.
    """
    settings = settings_manager.current
    from app.main import orchestrator

    # Check embedding service status
    embedding_status = "unknown"
    if orchestrator and orchestrator.embedding_service:
        is_healthy = await orchestrator.embedding_service.health_check()
        embedding_status = "healthy" if is_healthy else "unavailable"

    return MetricsResponse(
        jobs_total=metrics.jobs_total,
        jobs_completed=metrics.jobs_completed,
        jobs_failed=metrics.jobs_failed,
        jobs_pending=0,  # Would come from database
        jobs_processing=0,  # Would come from database
        success_rate=round(metrics.success_rate, 2),
        chunks_total=0,  # Would come from database
        embeddings_total=0,  # Would come from database
        pages_total=0,  # Would come from database
        tables_total=0,  # Would come from database
        avg_processing_time_seconds=0.0,  # Would calculate from database
        avg_chunks_per_document=0.0,  # Would calculate from database
        uptime_seconds=round(metrics.uptime_seconds, 1),
        embedding_service_status=embedding_status
    )

