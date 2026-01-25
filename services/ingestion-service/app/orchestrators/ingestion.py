"""
Ingestion Orchestrator

Coordinates the document ingestion pipeline:
1. PDF Extraction (ExtractionService)
2. Text Chunking (ChunkingService)
3. Embedding Generation (EmbeddingService)
4. Vector Storage (QdrantRepository)
5. Database Storage (DocumentRepository)
6. Job Tracking (JobRepository)
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import uuid

from app.models import Chunk, JobStatus
from app.services.chunking import ChunkingService
from app.services.embedding import EmbeddingService
from app.repositories.document import DocumentRepository
from app.repositories.qdrant import QdrantRepository
from app.configs import settings_manager
# Assuming this utility exists in your app.utils or similar
from app.utils import extract_document_date
from app.repositories.postgres import PostgresJobRepository

logger = logging.getLogger(__name__)

__all__ = ["IngestionOrchestrator"]


class IngestionOrchestrator:
    def __init__(
            self,
            extraction_service: Any,
            chunking_service: ChunkingService,
            embedding_service: EmbeddingService,
            document_repo: DocumentRepository,
            job_repo: PostgresJobRepository,
            qdrant_repo: Optional[QdrantRepository] = None,
            cache_dir: str = "./cache"
    ):
        self.extraction_service = extraction_service
        self.chunking_service = chunking_service
        self.embedding_service = embedding_service
        self.document_repo = document_repo
        self.job_repo = job_repo
        self.qdrant_repo = qdrant_repo
        self.cache_dir = Path(cache_dir)

        self.intermediate_dir = self.cache_dir / "intermediate"
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)

        qdrant_status = "enabled" if qdrant_repo else "disabled"
        logger.info(f"IngestionOrchestrator initialized (Qdrant: {qdrant_status})")

    async def ingest(
            self,
            file_path: str,
            filename: str,
            file_size_bytes: int,
            generate_embeddings: bool = True,
            save_intermediate: bool = True,
            save_to_database: bool = True,
            save_to_qdrant: bool = True,
            metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        job_id = str(uuid.uuid4())

        await self.job_repo.create_job(
            job_id=job_id,
            filename=filename,
            file_size_bytes=file_size_bytes,
            metadata=metadata
        )

        asyncio.create_task(
            self._process_document(
                job_id=job_id,
                file_path=file_path,
                filename=filename,
                generate_embeddings=generate_embeddings,
                save_intermediate=save_intermediate,
                save_to_database=save_to_database,
                save_to_qdrant=save_to_qdrant,
                metadata=metadata or {}
            )
        )

        logger.info(f"Started ingestion job: {job_id}")
        return job_id

    async def _enrich_with_summary(self, content: str) -> str:
        settings = settings_manager.current
        api_key = settings.openai_api_key
        if not api_key:
            logger.warning("Skipping summary enrichment: No OpenAI API Key found.")
            return content

        try:
            logger.info("Enriching content with OpenAI summary (model: gpt-4o-mini)...")
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=api_key)

            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful medical data assistant."},
                    {"role": "user", "content": (
                        f"Analyze the following text extracted from a lab report. "
                        f"Identify key results and abnormal values. "
                        f"Write a concise summary.\n\n"
                        f"--- TEXT START ---\n{content[:15000]}\n--- TEXT END ---"
                    )}
                ],
                temperature=0.0
            )

            summary = response.choices[0].message.content
            return f"{content}\n\n=== SUMMARY ===\n{summary}\n"

        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return content

    async def _process_document(
            self,
            job_id: str,
            file_path: str,
            filename: str,
            generate_embeddings: bool,
            save_intermediate: bool,
            save_to_database: bool,
            save_to_qdrant: bool,
            metadata: Dict[str, Any]
    ):
        start_time = datetime.utcnow()
        document_id = str(uuid.uuid4())

        try:
            await self.job_repo.update_status(
                job_id=job_id,
                status=JobStatus.PROCESSING,
                progress=5.0,
                started_at=start_time
            )

            # Step 1: Extract PDF content
            logger.info(f"[{job_id}] Step 1/6: Extracting PDF content...")
            await self._configure_extraction_service(job_id, metadata)
            extraction_result = await self.extraction_service.extract(file_path)

            await self.job_repo.update_progress(
                job_id=job_id,
                progress=20.0,
                pages_processed=extraction_result.pages,
                tables_extracted=extraction_result.tables
            )

            # Step 1.5: Enrich with Summary
            if generate_embeddings:
                logger.info(f"[{job_id}] Step 1.5/6: Generating AI Summary...")
                extraction_result.content = await self._enrich_with_summary(extraction_result.content)

            document_date, document_year = extract_document_date(
                filename=filename,
                content=extraction_result.content[:5000]
            )
            logger.info(f"[{job_id}] Metadata found - Date: {document_date}, Year: {document_year}")

            # Step 2: Chunk text (Modified to include date metadata)
            logger.info(f"[{job_id}] Step 2/6: Chunking text...")
            chunks = await self.chunking_service.chunk(
                content=extraction_result.content,
                metadata={
                    **metadata,
                    **extraction_result.metadata,
                    "document_id": document_id,
                    "filename": filename,
                    "document_date": document_date, # NEW
                    "document_year": document_year, # NEW
                }
            )

            await self.job_repo.update_progress(
                job_id=job_id,
                progress=40.0,
                chunks_created=len(chunks)
            )

            # Step 3: Generate embeddings
            embeddings_count = 0
            if generate_embeddings:
                logger.info(f"[{job_id}] Step 3/6: Generating embeddings...")
                chunks = await self.embedding_service.generate_embeddings(chunks)
                embeddings_count = sum(1 for c in chunks if c.embedding is not None)
                await self.job_repo.update_progress(job_id=job_id, progress=60.0, embeddings_generated=embeddings_count)
            else:
                await self.job_repo.update_progress(job_id=job_id, progress=60.0)

            # Step 4: Save to Qdrant (Updated with date info)
            qdrant_count = 0
            if save_to_qdrant and self.qdrant_repo and embeddings_count > 0:
                logger.info(f"[{job_id}] Step 4/6: Saving to Qdrant...")
                try:
                    # We move source_file, date, and year into extra_metadata
                    # and REMOVE them from the main arguments list
                    qdrant_count = await self.qdrant_repo.upsert_chunks(
                        document_id=document_id,
                        chunks=chunks,
                        filename=filename,
                        extra_metadata={
                            "job_id": job_id,
                            "source_file": file_path,  # Moved here
                            "document_date": document_date,  # Moved here
                            "document_year": document_year,  # Moved here
                            "tables_extracted": extraction_result.tables,
                            "has_summary": True
                        }
                    )
                except Exception as e:
                    logger.error(f"[{job_id}] Qdrant upsert failed: {e}")

            await self.job_repo.update_progress(job_id=job_id, progress=75.0)

            # Step 5: Save intermediate JSON (Updated with date info)
            intermediate_file_path = None
            if save_intermediate:
                logger.info(f"[{job_id}] Step 5/6: Saving intermediate results...")
                intermediate_file_path = await self._save_intermediate(
                    job_id=job_id,
                    document_id=document_id,
                    filename=filename,
                    extraction_result=extraction_result,
                    chunks=chunks,
                    qdrant_count=qdrant_count,
                    document_date=document_date, # NEW
                    document_year=document_year  # NEW
                )

            await self.job_repo.update_progress(job_id=job_id, progress=85.0)

            # Step 6: Save to database (Updated with date info)
            if save_to_database:
                logger.info(f"[{job_id}] Step 6/6: Saving to database...")
                await self._save_to_database(
                    document_id=document_id,
                    title=filename,
                    source=file_path,
                    content=extraction_result.content,
                    chunks=chunks,
                    metadata={
                        **metadata,
                        "qdrant_vectors": qdrant_count,
                        "document_date": document_date, # NEW
                        "document_year": document_year  # NEW
                    }
                )

            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()

            await self.job_repo.mark_completed(
                job_id=job_id,
                document_id=document_id,
                chunks_created=len(chunks),
                tables_extracted=extraction_result.tables,
                pages_processed=extraction_result.pages,
                embeddings_generated=embeddings_count,
                processing_time_seconds=processing_time,
                intermediate_file=str(intermediate_file_path) if intermediate_file_path else None
            )

        except Exception as e:
            logger.error(f"[{job_id}] Ingestion failed: {e}", exc_info=True)
            await self.job_repo.mark_failed(job_id=job_id, error_message=str(e), error_type=type(e).__name__)

    async def _save_intermediate(
            self,
            job_id: str,
            document_id: str,
            filename: str,
            extraction_result,
            chunks: List[Chunk],
            qdrant_count: int = 0,
            document_date: str = None, # NEW
            document_year: int = None  # NEW
    ) -> Path:
        """Saves enriched intermediate results to JSON."""
        intermediate_data = {
            "job_id": job_id,
            "document_id": document_id,
            "filename": filename,
            "document_date": document_date, # NEW
            "document_year": document_year, # NEW
            "timestamp": datetime.utcnow().isoformat(),
            "extraction": extraction_result.to_dict(),
            "chunks": [
                {
                    "index": chunk.index,
                    "content": chunk.content,
                    "metadata": {
                        **chunk.metadata,
                        "document_date": document_date,
                        "document_year": document_year
                    },
                }
                for chunk in chunks
            ],
            "stats": {
                "total_chunks": len(chunks),
                "document_year": document_year, # NEW
                "qdrant_vectors_stored": qdrant_count,
                "total_pages": extraction_result.pages,
            }
        }

        file_path = self.intermediate_dir / f"{job_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(intermediate_data, f, indent=2, ensure_ascii=False)
        return file_path

    async def _save_to_database(
            self,
            document_id: str,
            title: str,
            source: str,
            content: str,
            chunks: List[Chunk],
            metadata: dict
    ) -> str:
        await self.document_repo.create_document(
            document_id=document_id,
            title=title,
            source=source,
            content=content,
            metadata=metadata
        )
        await self.document_repo.save_chunks(document_id=document_id, chunks=chunks)
        return document_id

    async def _configure_extraction_service(
            self,
            job_id: str,
            metadata: Dict[str, Any]
    ) -> None:
        """Configure extraction service based on job metadata."""
        try:
            settings = settings_manager.current

            desired_extractor = metadata.get("extraction_service") or settings.extraction_service
            job_enable_ocr = metadata.get("enable_ocr")

            if job_enable_ocr is None:
                job_enable_ocr = settings.enable_ocr

            # Handle LightOnOCR
            if desired_extractor == "lighton_ocr":
                if self.extraction_service.__class__.__name__ != "LightOnOCRService":
                    logger.info(f"[{job_id}] Switching to LightOnOCRService")
                    from app.ocr_extraction import LightOnOCRService
                    import os

                    ocr_service = LightOnOCRService(
                        endpoint_url=os.getenv(
                            "OCR_ENDPOINT_URL",
                            "https://zuquhaomdlqs0z-8000.proxy.runpod.net/v1/chat/completions"
                        ),
                        dpi=settings.ocr_dpi
                    )
                    await ocr_service.initialize()
                    self.extraction_service = ocr_service
                    logger.info(f"[{job_id}] ✓ LightOnOCRService active")

            # Handle Unstructured
            elif desired_extractor == "unstructured":
                if self.extraction_service.__class__.__name__ not in ["UnstructuredService", "UnstructuredServiceOptimized"]:
                    logger.info(f"[{job_id}] Switching to UnstructuredService")
                    from app.services.unstructured_extraction import UnstructuredService, UnstructuredServiceOptimized

                    if job_enable_ocr:
                        self.extraction_service = UnstructuredService(
                            strategy="ocr_only",
                            extract_tables=True,
                            extract_images=False,
                            languages=["eng"],
                            ocr_languages=["eng"],
                            use_chunking=False,
                        )
                    else:
                        self.extraction_service = UnstructuredServiceOptimized()

                    logger.info(f"[{job_id}] ✓ UnstructuredService active")

        except Exception as e:
            logger.warning(f"[{job_id}] Could not switch extraction service: {e}")

    async def _save_to_database(
            self,
            document_id: str,
            title: str,
            source: str,
            content: str,
            chunks: List[Chunk],
            metadata: dict
    ) -> str:
        """
        Save document and chunks to database.

        Uses the provided document_id (same as Qdrant) for consistency.
        """
        # Create document with specific ID
        await self.document_repo.create_document(
            document_id=document_id,
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
        """Get job status."""
        return await self.job_repo.get_job(job_id)

    async def get_document_chunks(self, document_id: str) -> list:
        """Get all chunks for a document from database."""
        return await self.document_repo.get_chunks(document_id)

    async def search_similar(
            self,
            query_embedding: List[float],
            top_k: int = 5,
            filter_document_ids: Optional[List[str]] = None,
    ) -> list:
        """
        Search for similar chunks in Qdrant.

        Convenience method that wraps Qdrant repository search.

        Args:
            query_embedding: Query vector
            top_k: Number of results
            filter_document_ids: Optional document filter

        Returns:
            List of VectorSearchResult
        """
        if not self.qdrant_repo:
            logger.warning("Qdrant not configured, cannot search")
            return []

        return await self.qdrant_repo.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_document_ids=filter_document_ids,
        )