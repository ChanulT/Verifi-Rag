# app/repositories/postgres.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging
import json

from app.sql import JobModel, DocumentModel, ChunkModel
from app.models import JobStatus

logger = logging.getLogger(__name__)

class PostgresJobRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    async def create_job(self, job_id: str, filename: str, file_size_bytes: int, metadata: dict = None) -> str:
        async with self.session_factory() as session:
            job = JobModel(
                job_id=job_id,
                filename=filename,
                file_size_bytes=file_size_bytes,
                metadata_=metadata or {}
            )
            session.add(job)
            await session.commit()
            return job_id

    async def get_job(self, job_id: str) -> Optional[Dict]:
        async with self.session_factory() as session:
            result = await session.execute(select(JobModel).where(JobModel.job_id == job_id))
            job = result.scalars().first()
            if not job: return None

            # Map SQLAlchemy model to dictionary
            return {
                "job_id": job.job_id,
                "status": job.status,
                "progress": job.progress,
                "filename": job.filename,
                "file_size_bytes": job.file_size_bytes,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
                "document_id": None,  # You might want to store this in JobModel if needed
                "chunks_created": job.chunks_created,
                "processing_time_seconds": job.processing_time_seconds,
                "error_message": job.error_message,
                "error_type": job.error_type,
                "intermediate_file": job.intermediate_file,
                "metadata": job.metadata_
            }

    async def update_status(self, job_id: str, status: JobStatus, **kwargs):
        """Generic status update."""
        async with self.session_factory() as session:
            # Filter kwargs to only include valid columns to prevent CompileError
            valid_columns = {c.name for c in JobModel.__table__.columns}
            updates = {k: v for k, v in kwargs.items() if k in valid_columns}

            updates['status'] = status

            stmt = update(JobModel).where(JobModel.job_id == job_id).values(**updates)
            await session.execute(stmt)
            await session.commit()

    async def update_progress(self, job_id: str, progress: float, **kwargs):
        """Update progress and metrics."""
        async with self.session_factory() as session:
            valid_columns = {c.name for c in JobModel.__table__.columns}
            updates = {k: v for k, v in kwargs.items() if k in valid_columns}

            updates['progress'] = progress

            stmt = update(JobModel).where(JobModel.job_id == job_id).values(**updates)
            await session.execute(stmt)
            await session.commit()

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
        """Mark job as successfully completed."""
        async with self.session_factory() as session:
            stmt = update(JobModel).where(JobModel.job_id == job_id).values(
                status=JobStatus.COMPLETED,
                progress=100.0,
                chunks_created=chunks_created,
                tables_extracted=tables_extracted,
                pages_processed=pages_processed,
                embeddings_generated=embeddings_generated,
                processing_time_seconds=processing_time_seconds,
                intermediate_file=intermediate_file,
                completed_at=datetime.utcnow()
            )
            await session.execute(stmt)
            await session.commit()
            logger.info(f"Job {job_id} marked as COMPLETED")

    async def mark_failed(self, job_id: str, error_message: str, error_type: str):
        """Mark job as failed."""
        async with self.session_factory() as session:
            stmt = update(JobModel).where(JobModel.job_id == job_id).values(
                status=JobStatus.FAILED,
                error_message=error_message,
                error_type=error_type,
                completed_at=datetime.utcnow()
            )
            await session.execute(stmt)
            await session.commit()
            logger.error(f"Job {job_id} marked as FAILED: {error_message}")


class PostgresDocumentRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    async def create_document(self, document_id: str, title: str, content: str, metadata: dict, source: str = None):
        async with self.session_factory() as session:
            # Ensure metadata is a dict
            safe_metadata = metadata or {}

            # Save 'source' into metadata since we don't have a dedicated column for it
            if source:
                safe_metadata["source"] = source

            doc = DocumentModel(
                id=document_id,
                title=title,
                content=content,
                metadata_=safe_metadata
            )
            session.add(doc)
            await session.commit()

    async def save_chunks(self, document_id: str, chunks: List):
        async with self.session_factory() as session:
            chunk_models = []
            for chunk in chunks:
                chunk_models.append(ChunkModel(
                    id=f"{document_id}-{chunk.index}",
                    document_id=document_id,
                    content=chunk.content,
                    chunk_index=chunk.index,
                    page_number=chunk.page_number
                ))
            session.add_all(chunk_models)
            await session.commit()
            return len(chunk_models)