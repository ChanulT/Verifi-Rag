"""
In-Memory Storage for Development/Demo

Simple in-memory storage to replace database during development.
Replace with actual PostgreSQL in production.
"""

import logging
from datetime import datetime
from typing import Dict, Optional, List, Any
from app.models import JobStatus

logger = logging.getLogger(__name__)

# Global in-memory stores
_jobs_store: Dict[str, Dict[str, Any]] = {}
_documents_store: Dict[str, Dict[str, Any]] = {}
_chunks_store: Dict[str, List[Dict[str, Any]]] = {}


class InMemoryJobStore:
    """In-memory job storage."""

    async def create_job(
            self,
            job_id: str,
            filename: str,
            file_size_bytes: int,
            metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a job."""
        _jobs_store[job_id] = {
            "job_id": job_id,
            "filename": filename,
            "file_size_bytes": file_size_bytes,
            "status": JobStatus.PENDING.value,
            "progress": 0.0,
            "metadata": metadata or {},
            "created_at": datetime.utcnow(),
            "started_at": None,
            "completed_at": None,
            "document_id": None,
            "chunks_created": 0,
            "tables_extracted": 0,
            "pages_processed": 0,
            "embeddings_generated": 0,
            "processing_time_seconds": None,
            "error_message": None,
            "error_type": None,
            "intermediate_file": None,
        }
        logger.debug(f"Created job: {job_id}")
        return job_id

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a job."""
        return _jobs_store.get(job_id)

    async def update_status(
            self,
            job_id: str,
            status: JobStatus,
            progress: Optional[float] = None,
            started_at: Optional[datetime] = None,
            completed_at: Optional[datetime] = None,
            error_message: Optional[str] = None,
            error_type: Optional[str] = None
    ):
        """Update job status."""
        if job_id not in _jobs_store:
            return

        job = _jobs_store[job_id]
        job["status"] = status.value

        if progress is not None:
            job["progress"] = progress
        if started_at is not None:
            job["started_at"] = started_at
        if completed_at is not None:
            job["completed_at"] = completed_at
        if error_message is not None:
            job["error_message"] = error_message
        if error_type is not None:
            job["error_type"] = error_type

        logger.debug(f"Updated job {job_id}: status={status.value}")

    async def update_progress(
            self,
            job_id: str,
            progress: float,
            chunks_created: Optional[int] = None,
            tables_extracted: Optional[int] = None,
            pages_processed: Optional[int] = None,
            embeddings_generated: Optional[int] = None
    ):
        """Update job progress."""
        if job_id not in _jobs_store:
            return

        job = _jobs_store[job_id]
        job["progress"] = progress

        if chunks_created is not None:
            job["chunks_created"] = chunks_created
        if tables_extracted is not None:
            job["tables_extracted"] = tables_extracted
        if pages_processed is not None:
            job["pages_processed"] = pages_processed
        if embeddings_generated is not None:
            job["embeddings_generated"] = embeddings_generated

    async def mark_completed(
            self,
            job_id: str,
            document_id: str,
            chunks_created: int,
            tables_extracted: int,
            pages_processed: int,
            embeddings_generated: int,
            processing_time_seconds: float,
            intermediate_file: Optional[str] = None
    ):
        """Mark job as completed."""
        if job_id not in _jobs_store:
            return

        job = _jobs_store[job_id]
        job["status"] = JobStatus.COMPLETED.value
        job["progress"] = 100.0
        job["document_id"] = document_id
        job["chunks_created"] = chunks_created
        job["tables_extracted"] = tables_extracted
        job["pages_processed"] = pages_processed
        job["embeddings_generated"] = embeddings_generated
        job["processing_time_seconds"] = processing_time_seconds
        job["intermediate_file"] = intermediate_file
        job["completed_at"] = datetime.utcnow()

        logger.info(f"Job {job_id} completed successfully")

    async def mark_failed(
            self,
            job_id: str,
            error_message: str,
            error_type: str
    ):
        """Mark job as failed."""
        if job_id not in _jobs_store:
            return

        job = _jobs_store[job_id]
        job["status"] = JobStatus.FAILED.value
        job["error_message"] = error_message
        job["error_type"] = error_type
        job["completed_at"] = datetime.utcnow()

        logger.error(f"Job {job_id} failed: {error_message}")


class InMemoryDocumentStore:
    """In-memory document storage."""

    async def create_document(
            self,
            title: str,
            source: str,
            content: str,
            metadata: Dict[str, Any]
    ) -> str:
        """Create a document."""
        import uuid
        document_id = str(uuid.uuid4())

        _documents_store[document_id] = {
            "id": document_id,
            "title": title,
            "source": source,
            "content": content,
            "metadata": metadata,
            "created_at": datetime.utcnow(),
        }

        logger.debug(f"Created document: {document_id}")
        return document_id

    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document."""
        return _documents_store.get(document_id)

    async def save_chunks(
            self,
            document_id: str,
            chunks: List
    ) -> int:
        """Save chunks."""
        if document_id not in _chunks_store:
            _chunks_store[document_id] = []

        for chunk in chunks:
            chunk_data = {
                "chunk_id": f"{document_id}-{chunk.index}",
                "document_id": document_id,
                "content": chunk.content,
                "chunk_index": chunk.index,
                "metadata": chunk.metadata or {},
                "token_count": chunk.token_count,
                "has_embedding": chunk.embedding is not None,
                "created_at": datetime.utcnow(),
            }
            _chunks_store[document_id].append(chunk_data)

        logger.debug(f"Saved {len(chunks)} chunks for document {document_id}")
        return len(chunks)

    async def get_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get chunks for a document."""
        return _chunks_store.get(document_id, [])