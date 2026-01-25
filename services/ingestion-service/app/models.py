"""
Pydantic models for the Ingestion Service.

Contains:
- Request/Response schemas
- Domain models (Chunk, ExtractionResult, etc.)
- Job tracking models
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
from dataclasses import dataclass

__all__ = [
    "JobStatus",
    "Chunk",
    "ExtractionResult",
    "IngestRequest",
    "IngestResponse",
    "JobStatusResponse",
    "ChunkResponse",
    "ChunksListResponse",
]


# =============================================================================
# Enums
# =============================================================================

class JobStatus(str, Enum):
    """Job processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# =============================================================================
# Domain Models (Dataclasses for performance)
# =============================================================================

@dataclass
class Chunk:
    """
    A document chunk with metadata.

    Uses dataclass instead of Pydantic for better performance
    in tight loops (chunking can create 1000s of these).
    """
    content: str
    index: int
    start_char: int = 0
    end_char: int = 0
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None

    @property
    def size(self) -> int:
        """Get chunk size in characters."""
        return len(self.content)

    @property
    def token_count(self) -> int:
        """Approximate token count (more accurate counting can be added)."""
        # Simple approximation: 1 token ≈ 4 characters
        return len(self.content) // 4

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "content": self.content,
            "index": self.index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "size": self.size,
            "token_count": self.token_count,
            "metadata": self.metadata,
            "has_embedding": self.embedding is not None,
        }


@dataclass
class ExtractionResult:
    """Result of PDF extraction."""
    content: str
    metadata: Dict[str, Any]
    pages: int
    tables: int
    images: int
    processing_time_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content_preview": self.content[:500] + "..." if len(self.content) > 500 else self.content,
            "content_length": len(self.content),
            "metadata": self.metadata,
            "pages": self.pages,
            "tables": self.tables,
            "images": self.images,
            "processing_time_seconds": self.processing_time_seconds,
        }


# =============================================================================
# Request Schemas
# =============================================================================

class IngestRequest(BaseModel):
    """Request schema for document ingestion."""

    # Chunking options
    use_semantic_chunking: Optional[bool] = Field(
        default=None,
        description="Use semantic chunking (requires embedding service)"
    )
    chunk_size: Optional[int] = Field(
        default=None,
        ge=100,
        le=10000,
        description="Override default chunk size"
    )
    chunk_overlap: Optional[int] = Field(
        default=None,
        ge=0,
        le=1000,
        description="Override default chunk overlap"
    )

    # Extraction options
    enable_ocr: Optional[bool] = Field(
        default=None,
        description="Enable OCR for scanned PDFs"
    )
    include_tables: Optional[bool] = Field(
        default=None,
        description="Extract tables"
    )
    include_images: Optional[bool] = Field(
        default=None,
        description="Extract images"
    )

    # Processing options
    save_intermediate: Optional[bool] = Field(
        default=None,
        description="Save intermediate JSON for inspection"
    )
    generate_embeddings: bool = Field(
        default=True,
        description="Generate embeddings using embedding service"
    )

    # Storage options
    save_to_database: bool = Field(
        default=True,
        description="Save to database (disable for inspection only)"
    )


# =============================================================================
# Response Schemas
# =============================================================================

class IngestResponse(BaseModel):
    """Response schema for document ingestion."""

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Initial job status")
    filename: str = Field(..., description="Uploaded filename")
    file_size_mb: float = Field(..., description="File size in megabytes")
    estimated_time_seconds: Optional[float] = Field(
        None,
        description="Estimated processing time"
    )
    message: str = Field(..., description="Status message")


class JobStatusResponse(BaseModel):
    """Response schema for job status check."""

    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Current status")
    progress: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Processing progress percentage"
    )

    # Processing results
    document_id: Optional[str] = Field(None, description="Document ID (when completed)")
    chunks_created: int = Field(default=0, description="Number of chunks created")
    tables_extracted: int = Field(default=0, description="Number of tables extracted")
    pages_processed: int = Field(default=0, description="Number of pages processed")
    embeddings_generated: int = Field(default=0, description="Number of embeddings generated")

    # Timing
    created_at: datetime = Field(..., description="Job creation time")
    started_at: Optional[datetime] = Field(None, description="Processing start time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    processing_time_seconds: Optional[float] = Field(
        None,
        description="Total processing time"
    )

    # Error info (if failed)
    error_message: Optional[str] = Field(None, description="Error message if failed")
    error_type: Optional[str] = Field(None, description="Error type")

    # Files
    filename: str = Field(..., description="Original filename")
    file_size_mb: float = Field(..., description="File size in MB")
    intermediate_file: Optional[str] = Field(
        None,
        description="Path to intermediate JSON file"
    )

    # Metadata
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ChunkResponse(BaseModel):
    """A processed document chunk."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    chunk_index: int = Field(..., description="Chunk sequence number")
    content: str = Field(..., description="Chunk text content")
    size: int = Field(..., description="Character count")
    token_count: int = Field(..., description="Approximate token count")

    # Metadata
    page_number: Optional[int] = Field(None, description="Source page number")
    section_title: Optional[str] = Field(None, description="Section heading")
    has_embedding: bool = Field(default=False, description="Embedding generated?")

    # Additional metadata
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ChunksListResponse(BaseModel):
    """Response schema for chunk retrieval."""

    job_id: str = Field(..., description="Job identifier")
    document_id: str = Field(..., description="Document identifier")
    total_chunks: int = Field(..., description="Total number of chunks")
    chunks: List[ChunkResponse] = Field(..., description="List of chunks")

    # Summary statistics
    avg_chunk_size: float = Field(..., description="Average chunk size")
    total_embeddings: int = Field(..., description="Number of chunks with embeddings")


class HealthStatus(BaseModel):
    """Service health status."""

    service: str = Field(..., description="Service name")
    status: str = Field(..., description="Overall health status")
    version: str = Field(..., description="Service version")
    uptime_seconds: float = Field(..., description="Service uptime")
    dependencies: Dict[str, str] = Field(..., description="Dependency status")


class MetricsResponse(BaseModel):
    """Service metrics."""

    jobs_total: int = Field(..., description="Total jobs processed")
    jobs_completed: int = Field(..., description="Successfully completed jobs")
    jobs_failed: int = Field(..., description="Failed jobs")
    jobs_pending: int = Field(..., description="Pending jobs")
    jobs_processing: int = Field(..., description="Currently processing jobs")
    success_rate: float = Field(..., description="Success rate percentage")

    chunks_total: int = Field(..., description="Total chunks created")
    embeddings_total: int = Field(..., description="Total embeddings generated")
    pages_total: int = Field(..., description="Total pages processed")
    tables_total: int = Field(..., description="Total tables extracted")

    avg_processing_time_seconds: float = Field(
        ...,
        description="Average job processing time"
    )
    avg_chunks_per_document: float = Field(
        ...,
        description="Average chunks per document"
    )

    uptime_seconds: float = Field(..., description="Service uptime")
    embedding_service_status: str = Field(..., description="Embedding service health")