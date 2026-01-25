"""
Ingestion Orchestrator

Coordinates the document ingestion pipeline:
1. PDF Extraction (ExtractionService)
2. Text Chunking (ChunkingService)
3. Embedding Generation (EmbeddingService)
4. Database Storage (DocumentRepository)
5. Job Tracking (JobRepository)

Follows the separation of concerns principle - each service
has ONE job, and the orchestrator coordinates them.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import uuid

from app.models import Chunk, JobStatus
from app.services.extraction import ExtractionService
from app.services.chunking import ChunkingService
from app.services.embedding import EmbeddingService
from app.repositories.document import DocumentRepository
from app.repositories.job import JobRepository

logger = logging.getLogger(__name__)

__all__ = ["IngestionOrchestrator"]


class IngestionOrchestrator:
    """
    Orchestrates the document ingestion pipeline.

    Coordinates multiple services to process documents:
    - ExtractionService: Extracts PDF content
    - ChunkingService: Chunks text
    - EmbeddingService: Generates embeddings
    - DocumentRepository: Stores documents
    - JobRepository: Tracks job status

    This follows the orchestration pattern - business logic is here,
    but actual work is delegated to specialized services.

    Example:
        orchestrator = IngestionOrchestrator(
            extraction_service=extraction_svc,
            chunking_service=chunking_svc,
            embedding_service=embedding_svc,
            document_repo=doc_repo,
            job_repo=job_repo
        )

        job_id = await orchestrator.ingest(
            file_path="document.pdf",
            filename="document.pdf"
        )
    """

    def __init__(
            self,
            extraction_service: ExtractionService,
            chunking_service: ChunkingService,
            embedding_service: EmbeddingService,
            document_repo: DocumentRepository,
            job_repo: JobRepository,
            cache_dir: str = "./cache"
    ):
        """
        Initialize orchestrator with all required services.

        Args:
            extraction_service: PDF extraction service
            chunking_service: Text chunking service
            embedding_service: Embedding generation service
            document_repo: Document repository
            job_repo: Job repository
            cache_dir: Directory for intermediate files
        """
        self.extraction_service = extraction_service
        self.chunking_service = chunking_service
        self.embedding_service = embedding_service
        self.document_repo = document_repo
        self.job_repo = job_repo
        self.cache_dir = Path(cache_dir)

        # Ensure intermediate directory exists
        self.intermediate_dir = self.cache_dir / "intermediate"
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)

        logger.info("IngestionOrchestrator initialized")

    async def ingest(
            self,
            file_path: str,
            filename: str,
            file_size_bytes: int,
            generate_embeddings: bool = True,
            save_intermediate: bool = True,
            save_to_database: bool = True,
            metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Ingest a document through the full pipeline.

        This method orchestrates all services to:
        1. Create job tracker
        2. Extract PDF content
        3. Chunk text
        4. Generate embeddings (optional)
        5. Save to database (optional)
        6. Save intermediate JSON (optional)

        Args:
            file_path: Path to PDF file
            filename: Original filename
            file_size_bytes: File size
            generate_embeddings: Generate embeddings using embedding service
            save_intermediate: Save intermediate JSON file
            save_to_database: Save to database
            metadata: Additional metadata

        Returns:
            Job ID for tracking progress
        """
        # Generate job ID
        job_id = str(uuid.uuid4())

        # Create job tracker
        await self.job_repo.create_job(
            job_id=job_id,
            filename=filename,
            file_size_bytes=file_size_bytes,
            metadata=metadata
        )

        # Start processing in background
        asyncio.create_task(
            self._process_document(
                job_id=job_id,
                file_path=file_path,
                filename=filename,
                generate_embeddings=generate_embeddings,
                save_intermediate=save_intermediate,
                save_to_database=save_to_database,
                metadata=metadata or {}
            )
        )

        logger.info(f"Started ingestion job: {job_id}")

        return job_id

    async def _process_document(
            self,
            job_id: str,
            file_path: str,
            filename: str,
            generate_embeddings: bool,
            save_intermediate: bool,
            save_to_database: bool,
            metadata: Dict[str, Any]
    ):
        """
        Process document asynchronously.

        This is the main processing logic, separated from the
        public ingest() method to allow background execution.
        """
        start_time = datetime.utcnow()

        try:
            # Update status to processing
            await self.job_repo.update_status(
                job_id=job_id,
                status=JobStatus.PROCESSING,
                progress=10.0,
                started_at=start_time
            )

            # Step 1: Extract PDF content (20% progress)
            logger.info(f"[{job_id}] Step 1/5: Extracting PDF content...")

            # Ensure extraction implementation matches settings (allow runtime override)
            # Determine desired extractor: per-job metadata overrides global settings
            try:
                from app.configs import settings_manager
                settings = settings_manager.current

                desired_extractor = (metadata or {}).get("extraction_service") or settings.extraction_service
                job_enable_ocr = (metadata or {}).get("enable_ocr")
                # If metadata explicitly provides enable_ocr use it, else fall back to settings
                if job_enable_ocr is None:
                    job_enable_ocr = settings.enable_ocr

                if desired_extractor == "lighton_ocr":
                    # If current extractor is not LightOnOCR, replace it at runtime
                    if self.extraction_service.__class__.__name__ != "LightOnOCRService":
                        logger.info(f"[{job_id}] Switching to LightOnOCRService per configuration")
                        from app.ocr_extraction import LightOnOCRService

                        ocr_service = LightOnOCRService(
                            model_name=settings.ocr_model_name,
                            device=settings.ocr_device,
                            dpi=settings.ocr_dpi
                        )

                        # Initialize model (async)
                        await ocr_service.initialize()

                        # Swap in the new service instance
                        self.extraction_service = ocr_service
                        logger.info(f"[{job_id}] ✓ LightOnOCRService is now active")

                elif desired_extractor == "unstructured":
                    # If current extractor is not Unstructured, replace it at runtime
                    if self.extraction_service.__class__.__name__ not in ("UnstructuredService", "UnstructuredServiceOptimized"):
                        logger.info(f"[{job_id}] Switching to UnstructuredService per configuration")
                        from app.services.unstructured_extraction import UnstructuredService, UnstructuredServiceOptimized

                        if job_enable_ocr:
                            # Use OCR-only strategy for scanned PDFs
                            unstructured = UnstructuredService(
                                strategy="ocr_only",
                                extract_tables=True,
                                extract_images=False,
                                languages=["eng"],
                                ocr_languages=["eng"],
                                use_chunking=False,
                            )
                        else:
                            unstructured = UnstructuredServiceOptimized()

                        # Swap in the new service instance
                        self.extraction_service = unstructured
                        logger.info(f"[{job_id}] ✓ UnstructuredService is now active")
            except Exception as e:
                logger.warning(f"[{job_id}] Could not switch extraction service at runtime: {e}")

            extraction_result = await self.extraction_service.extract(file_path)

            await self.job_repo.update_progress(
                job_id=job_id,
                progress=30.0,
                pages_processed=extraction_result.pages,
                tables_extracted=extraction_result.tables
            )

            # Step 2: Chunk text (40% progress)
            logger.info(f"[{job_id}] Step 2/5: Chunking text...")
            chunks = await self.chunking_service.chunk(
                content=extraction_result.content,
                metadata={
                    **metadata,
                    **extraction_result.metadata
                }
            )

            await self.job_repo.update_progress(
                job_id=job_id,
                progress=50.0,
                chunks_created=len(chunks)
            )

            # Step 3: Generate embeddings (60% progress)
            if generate_embeddings:
                logger.info(f"[{job_id}] Step 3/5: Generating embeddings...")
                chunks = await self.embedding_service.generate_embeddings(chunks)
                embeddings_count = sum(1 for c in chunks if c.embedding is not None)

                await self.job_repo.update_progress(
                    job_id=job_id,
                    progress=70.0,
                    embeddings_generated=embeddings_count
                )
            else:
                logger.info(f"[{job_id}] Step 3/5: Skipping embeddings (disabled)")

            # Step 4: Save intermediate JSON (80% progress)
            intermediate_file_path = None
            if save_intermediate:
                logger.info(f"[{job_id}] Step 4/5: Saving intermediate results...")
                intermediate_file_path = await self._save_intermediate(
                    job_id=job_id,
                    filename=filename,
                    extraction_result=extraction_result,
                    chunks=chunks
                )
                logger.info(f"[{job_id}] ✓ Intermediate file: {intermediate_file_path}")
            else:
                logger.info(f"[{job_id}] Step 4/5: Skipping intermediate file")

            # Step 5: Save to database (90% progress)
            document_id = None
            if save_to_database:
                logger.info(f"[{job_id}] Step 5/5: Saving to database...")
                document_id = await self._save_to_database(
                    title=filename,
                    source=file_path,
                    content=extraction_result.content,
                    chunks=chunks,
                    metadata={
                        **metadata,
                        **extraction_result.metadata
                    }
                )
                logger.info(f"[{job_id}] ✓ Document ID: {document_id}")
            else:
                logger.info(f"[{job_id}] Step 5/5: Skipping database save")
                document_id = "inspection_mode"

            # Calculate processing time
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()

            # Mark as completed
            await self.job_repo.mark_completed(
                job_id=job_id,
                document_id=document_id,
                chunks_created=len(chunks),
                tables_extracted=extraction_result.tables,
                pages_processed=extraction_result.pages,
                embeddings_generated=sum(1 for c in chunks if c.embedding is not None),
                processing_time_seconds=processing_time,
                intermediate_file=str(intermediate_file_path) if intermediate_file_path else None
            )

            logger.info(
                f"[{job_id}] ✓ Ingestion completed in {processing_time:.2f}s: "
                f"{len(chunks)} chunks, "
                f"{extraction_result.pages} pages, "
                f"{extraction_result.tables} tables"
            )

        except Exception as e:
            logger.error(f"[{job_id}] ✗ Ingestion failed: {e}", exc_info=True)

            # Mark as failed
            await self.job_repo.mark_failed(
                job_id=job_id,
                error_message=str(e),
                error_type=type(e).__name__
            )

    async def _save_intermediate(
            self,
            job_id: str,
            filename: str,
            extraction_result,
            chunks: list
    ) -> Path:
        """
        Save intermediate processing results to JSON.

        This allows inspection of OCR quality and chunking
        before committing to vector database.
        """
        intermediate_data = {
            "job_id": job_id,
            "filename": filename,
            "timestamp": datetime.utcnow().isoformat(),
            "extraction": extraction_result.to_dict(),
            "chunks": [chunk.to_dict() for chunk in chunks],
            "stats": {
                "total_chunks": len(chunks),
                "avg_chunk_size": sum(c.size for c in chunks) / len(chunks) if chunks else 0,
                "chunks_with_embeddings": sum(1 for c in chunks if c.embedding is not None),
                "total_pages": extraction_result.pages,
                "total_tables": extraction_result.tables,
                "total_images": extraction_result.images,
            }
        }

        # Save to file
        file_path = self.intermediate_dir / f"{job_id}.json"

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(intermediate_data, f, indent=2, ensure_ascii=False)

        return file_path

    async def _save_to_database(
            self,
            title: str,
            source: str,
            content: str,
            chunks: list,
            metadata: dict
    ) -> str:
        """
        Save document and chunks to database.

        Uses repository pattern for clean data access.
        """
        # Create document
        document_id = await self.document_repo.create_document(
            title=title,
            source=source,
            content=content,
            metadata=metadata
        )

        # Save chunks
        await self.document_repo.save_chunks(
            document_id=document_id,
            chunks=chunks
        )

        return document_id

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job status.

        Args:
            job_id: Job identifier

        Returns:
            Job data or None if not found
        """
        return await self.job_repo.get_job(job_id)

    async def get_document_chunks(self, document_id: str) -> list:
        """
        Get all chunks for a document.

        Args:
            document_id: Document identifier

        Returns:
            List of chunk dictionaries
        """
        return await self.document_repo.get_chunks(document_id)