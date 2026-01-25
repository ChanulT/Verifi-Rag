"""
Job Repository

Handles ingestion job tracking in database.
Provides status management, progress tracking, and error logging.
"""

import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

from app.models import JobStatus

logger = logging.getLogger(__name__)

__all__ = ["JobRepository"]


class JobRepository:
    """
    Repository for job tracking operations.

    Tracks ingestion job lifecycle:
    - Creation
    - Status updates
    - Progress tracking
    - Error logging
    - Completion metrics

    Example:
        repo = JobRepository(db_pool)
        await repo.create_job(job_id, filename, file_size)
        await repo.update_progress(job_id, 50.0)
        await repo.mark_completed(job_id, results)
    """

    def __init__(self, db_pool):
        """
        Initialize repository.

        Args:
            db_pool: AsyncPG connection pool
        """
        self.db_pool = db_pool
        logger.info("JobRepository initialized")

    async def create_job(
            self,
            job_id: str,
            filename: str,
            file_size_bytes: int,
            metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new ingestion job.

        Args:
            job_id: Unique job identifier
            filename: Original filename
            file_size_bytes: File size in bytes
            metadata: Optional job metadata

        Returns:
            Job ID
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ingestion_jobs (job_id,
                                            filename,
                                            file_size_bytes,
                                            status,
                                            progress,
                                            metadata,
                                            created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                job_id,
                filename,
                file_size_bytes,
                JobStatus.PENDING.value,
                0.0,
                json.dumps(metadata or {}),
                datetime.utcnow()
            )

            logger.debug(f"Created job: {job_id}")
            return job_id

    async def get_job(
            self,
            job_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get job by ID.

        Args:
            job_id: Job identifier

        Returns:
            Job data or None if not found
        """
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT job_id,
                       filename,
                       file_size_bytes,
                       status,
                       progress,
                       document_id,
                       chunks_created,
                       tables_extracted,
                       pages_processed,
                       embeddings_generated,
                       created_at,
                       started_at,
                       completed_at,
                       processing_time_seconds,
                       error_message,
                       error_type,
                       metadata,
                       intermediate_file
                FROM ingestion_jobs
                WHERE job_id = $1
                """,
                job_id
            )

            if not row:
                return None

            return dict(row)

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
        """
        Update job status.

        Args:
            job_id: Job identifier
            status: New status
            progress: Progress percentage (0-100)
            started_at: Start timestamp
            completed_at: Completion timestamp
            error_message: Error message if failed
            error_type: Error type if failed
        """
        async with self.db_pool.acquire() as conn:
            # Build update query dynamically
            updates = ["status = $2"]
            values = [job_id, status.value]
            param_idx = 3

            if progress is not None:
                updates.append(f"progress = ${param_idx}")
                values.append(progress)
                param_idx += 1

            if started_at is not None:
                updates.append(f"started_at = ${param_idx}")
                values.append(started_at)
                param_idx += 1

            if completed_at is not None:
                updates.append(f"completed_at = ${param_idx}")
                values.append(completed_at)
                param_idx += 1

            if error_message is not None:
                updates.append(f"error_message = ${param_idx}")
                values.append(error_message)
                param_idx += 1

            if error_type is not None:
                updates.append(f"error_type = ${param_idx}")
                values.append(error_type)
                param_idx += 1

            query = f"""
                UPDATE ingestion_jobs
                SET {', '.join(updates)}
                WHERE job_id = $1
            """

            await conn.execute(query, *values)

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
        """
        Update job progress and metrics.

        Args:
            job_id: Job identifier
            progress: Progress percentage (0-100)
            chunks_created: Number of chunks created
            tables_extracted: Number of tables extracted
            pages_processed: Number of pages processed
            embeddings_generated: Number of embeddings generated
        """
        async with self.db_pool.acquire() as conn:
            updates = ["progress = $2"]
            values = [job_id, progress]
            param_idx = 3

            if chunks_created is not None:
                updates.append(f"chunks_created = ${param_idx}")
                values.append(chunks_created)
                param_idx += 1

            if tables_extracted is not None:
                updates.append(f"tables_extracted = ${param_idx}")
                values.append(tables_extracted)
                param_idx += 1

            if pages_processed is not None:
                updates.append(f"pages_processed = ${param_idx}")
                values.append(pages_processed)
                param_idx += 1

            if embeddings_generated is not None:
                updates.append(f"embeddings_generated = ${param_idx}")
                values.append(embeddings_generated)
                param_idx += 1

            query = f"""
                UPDATE ingestion_jobs
                SET {', '.join(updates)}
                WHERE job_id = $1
            """

            await conn.execute(query, *values)

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
        """
        Mark job as completed.

        Args:
            job_id: Job identifier
            document_id: Created document ID
            chunks_created: Number of chunks
            tables_extracted: Number of tables
            pages_processed: Number of pages
            embeddings_generated: Number of embeddings
            processing_time_seconds: Total processing time
            intermediate_file: Path to intermediate JSON
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE ingestion_jobs
                SET status                  = $2,
                    progress                = 100.0,
                    document_id             = $3,
                    chunks_created          = $4,
                    tables_extracted        = $5,
                    pages_processed         = $6,
                    embeddings_generated    = $7,
                    processing_time_seconds = $8,
                    intermediate_file       = $9,
                    completed_at            = $10
                WHERE job_id = $1
                """,
                job_id,
                JobStatus.COMPLETED.value,
                document_id,
                chunks_created,
                tables_extracted,
                pages_processed,
                embeddings_generated,
                processing_time_seconds,
                intermediate_file,
                datetime.utcnow()
            )

            logger.info(f"Job {job_id} completed successfully")

    async def mark_failed(
            self,
            job_id: str,
            error_message: str,
            error_type: str
    ):
        """
        Mark job as failed.

        Args:
            job_id: Job identifier
            error_message: Error message
            error_type: Error type/class
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE ingestion_jobs
                SET status        = $2,
                    error_message = $3,
                    error_type    = $4,
                    completed_at  = $5
                WHERE job_id = $1
                """,
                job_id,
                JobStatus.FAILED.value,
                error_message,
                error_type,
                datetime.utcnow()
            )

            logger.error(f"Job {job_id} failed: {error_message}")

    async def list_jobs(
            self,
            status: Optional[JobStatus] = None,
            limit: int = 100,
            offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List jobs with optional filtering.

        Args:
            status: Filter by status
            limit: Maximum results
            offset: Offset for pagination

        Returns:
            List of job dictionaries
        """
        async with self.db_pool.acquire() as conn:
            if status:
                rows = await conn.fetch(
                    """
                    SELECT *
                    FROM ingestion_jobs
                    WHERE status = $1
                    ORDER BY created_at DESC
                        LIMIT $2
                    OFFSET $3
                    """,
                    status.value,
                    limit,
                    offset
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT *
                    FROM ingestion_jobs
                    ORDER BY created_at DESC
                        LIMIT $1
                    OFFSET $2
                    """,
                    limit,
                    offset
                )

            return [dict(row) for row in rows]