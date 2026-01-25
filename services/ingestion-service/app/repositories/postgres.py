# app/repositories/postgres.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update
from typing import Optional, Dict, Any, List
from datetime import datetime
import json

from app.models.sql import JobModel, DocumentModel, ChunkModel
from app.models import JobStatus
from app.storage import InMemoryJobStore  # Import interface if you have an abstract base class


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

    async def update_status(self, job_id: str, status: JobStatus, **kwargs):
        async with self.session_factory() as session:
            stmt = update(JobModel).where(JobModel.job_id == job_id).values(status=status, **kwargs)
            await session.execute(stmt)
            await session.commit()

    async def get_job(self, job_id: str) -> Optional[Dict]:
        async with self.session_factory() as session:
            result = await session.execute(select(JobModel).where(JobModel.job_id == job_id))
            job = result.scalars().first()
            if not job: return None

            # Convert SQLAlchemy model to dict for compatibility with your existing app
            return {
                "job_id": job.job_id,
                "status": job.status,
                "progress": job.progress,
                "created_at": job.created_at,
                "metadata": job.metadata_,
                # ... map other fields
            }


class PostgresDocumentRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    async def create_document(self, document_id: str, title: str, content: str, metadata: dict):
        async with self.session_factory() as session:
            doc = DocumentModel(
                id=document_id,
                title=title,
                content=content,
                metadata_=metadata
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