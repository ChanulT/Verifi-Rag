"""
Ingestion Orchestrator

Coordinates the document ingestion pipeline:
1. PDF Extraction (ExtractionService)
2. Text Chunking (ChunkingService)
3. Embedding Generation (EmbeddingService)
4. Vector Storage (QdrantRepository)  <-- NEW
5. Database Storage (DocumentRepository)
6. Job Tracking (JobRepository)

Follows the separation of concerns principle - each service
has ONE job, and the orchestrator coordinates them.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import uuid

from app.models import Chunk, JobStatus
from app.services.extraction import ExtractionService
from app.services.chunking import ChunkingService
from app.services.embedding import EmbeddingService
from app.repositories.document import DocumentRepository
from app.repositories.job import JobRepository
from app.repositories.qdrant import QdrantRepository  # NEW
from app.configs import settings_manager

logger = logging.getLogger(__name__)

__all__ = ["IngestionOrchestrator"]


class IngestionOrchestrator:
    """
    Orchestrates the document ingestion pipeline.

    Coordinates multiple services to process documents:
    - ExtractionService: Extracts PDF content
    - ChunkingService: Chunks text
    - EmbeddingService: Generates embeddings
    - QdrantRepository: Stores vectors with metadata (NEW)
    - DocumentRepository: Stores documents
    - JobRepository: Tracks job status

    The key addition is Qdrant integration which stores chunks
    with rich metadata enabling chatbot citation display.

    Example:
        orchestrator = IngestionOrchestrator(
            extraction_service=extraction_svc,
            chunking_service=chunking_svc,
            embedding_service=embedding_svc,
            qdrant_repo=qdrant_repo,  # NEW
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
            qdrant_repo: Optional[QdrantRepository] = None,  # NEW - optional for backward compat
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
            qdrant_repo: Qdrant repository for vector storage (NEW)
            cache_dir: Directory for intermediate files
        """
        self.extraction_service = extraction_service
        self.chunking_service = chunking_service
        self.embedding_service = embedding_service
        self.document_repo = document_repo
        self.job_repo = job_repo
        self.qdrant_repo = qdrant_repo  # NEW
        self.cache_dir = Path(cache_dir)

        # Ensure intermediate directory exists
        self.intermediate_dir = self.cache_dir / "intermediate"
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)

        # Log initialization status
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
            save_to_qdrant: bool = True,  # NEW
            metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Ingest a document through the full pipeline.

        This method orchestrates all services to:
        1. Create job tracker
        2. Extract PDF content
        3. Chunk text
        4. Generate embeddings (optional)
        5. Save to Qdrant vector DB (NEW, optional)
        6. Save to database (optional)
        7. Save intermediate JSON (optional)

        Args:
            file_path: Path to PDF file
            filename: Original filename
            file_size_bytes: File size
            generate_embeddings: Generate embeddings using embedding service
            save_intermediate: Save intermediate JSON file
            save_to_database: Save to database
            save_to_qdrant: Save to Qdrant vector DB (NEW)
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
                save_to_qdrant=save_to_qdrant,  # NEW
                metadata=metadata or {}
            )
        )

        logger.info(f"Started ingestion job: {job_id}")

        return job_id

    async def _enrich_with_summary(self, content: str) -> str:
        """
        Uses OpenAI to generate a natural language summary of the extracted tables.
        Appends this summary to the content.
        """
        settings = settings_manager.current

        # 1. Check if we have an API Key
        api_key = settings.openai_api_key
        if not api_key:
            logger.warning("Skipping summary enrichment: No OpenAI API Key found.")
            return content

        try:
            logger.info("Enriching content with OpenAI summary (model: gpt-4o-mini)...")
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=api_key)

            # 2. Call OpenAI (Cheap Model)
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful medical data assistant."
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Analyze the following text extracted from a lab report (which contains Markdown tables). "
                            f"Identify the key results, specifically focusing on any abnormal values (High, Low, Positive). "
                            f"Write a concise summary in plain English sentences.\n\n"
                            f"--- TEXT START ---\n{content[:15000]}\n--- TEXT END ---"
                            # Limit context to avoid token limits if file is huge
                        )
                    }
                ],
                temperature=0.0
            )

            summary = response.choices[0].message.content

            # 3. Format the new content
            # We append the summary with a clear header so the chunker sees it
            enriched_content = f"{content}\n\n=== SUMMARY ===\n{summary}\n"
            return enriched_content

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

            # ================================================================
            # Step 1: Extract PDF content
            # ================================================================
            logger.info(f"[{job_id}] Step 1/6: Extracting PDF content...")
            await self._configure_extraction_service(job_id, metadata)
            extraction_result = await self.extraction_service.extract(file_path)

            await self.job_repo.update_progress(
                job_id=job_id,
                progress=20.0,
                pages_processed=extraction_result.pages,
                tables_extracted=extraction_result.tables
            )

            # ================================================================
            # Step 1.5 (NEW): Enrich with Summary
            # ================================================================
            # We only do this if we are planning to use embeddings/search
            if generate_embeddings:
                logger.info(f"[{job_id}] Step 1.5/6: Generating AI Summary...")

                # Overwrite content with the summarized version
                enriched_content = await self._enrich_with_summary(extraction_result.content)
                extraction_result.content = enriched_content

                logger.info(f"[{job_id}] ✓ Content enriched with summary")

            # ================================================================
            # Step 2: Chunk text
            # ================================================================
            logger.info(f"[{job_id}] Step 2/6: Chunking text...")

            chunks = await self.chunking_service.chunk(
                content=extraction_result.content,
                metadata={
                    **metadata,
                    **extraction_result.metadata,
                    "document_id": document_id,
                    "filename": filename,
                }
            )

            await self.job_repo.update_progress(
                job_id=job_id,
                progress=40.0,
                chunks_created=len(chunks)
            )

            # ================================================================
            # Step 3: Generate embeddings
            # ================================================================
            embeddings_count = 0
            if generate_embeddings:
                logger.info(f"[{job_id}] Step 3/6: Generating embeddings...")
                chunks = await self.embedding_service.generate_embeddings(chunks)
                embeddings_count = sum(1 for c in chunks if c.embedding is not None)
                await self.job_repo.update_progress(
                    job_id=job_id,
                    progress=60.0,
                    embeddings_generated=embeddings_count
                )
            else:
                await self.job_repo.update_progress(job_id=job_id, progress=60.0)

            # ================================================================
            # Step 4: Save to Qdrant
            # ================================================================
            qdrant_count = 0
            if save_to_qdrant and self.qdrant_repo and embeddings_count > 0:
                logger.info(f"[{job_id}] Step 4/6: Saving to Qdrant...")
                try:
                    qdrant_count = await self.qdrant_repo.upsert_chunks(
                        document_id=document_id,
                        chunks=chunks,
                        filename=filename,
                        source_file=file_path,
                        total_pages=extraction_result.pages,
                        extra_metadata={
                            "job_id": job_id,
                            "tables_extracted": extraction_result.tables,
                            "has_summary": True  # Mark that this has a summary
                        }
                    )
                except Exception as e:
                    logger.error(f"[{job_id}] ⚠ Qdrant upsert failed: {e}")

            await self.job_repo.update_progress(job_id=job_id, progress=75.0)

            # ================================================================
            # Step 5: Save intermediate JSON
            # ================================================================
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
                )

            await self.job_repo.update_progress(job_id=job_id, progress=85.0)

            # ================================================================
            # Step 6: Save to database
            # ================================================================
            if save_to_database:
                logger.info(f"[{job_id}] Step 6/6: Saving to database...")
                await self._save_to_database(
                    document_id=document_id,
                    title=filename,
                    source=file_path,
                    content=extraction_result.content,
                    chunks=chunks,
                    metadata={**metadata, "qdrant_vectors": qdrant_count}
                )

            # ================================================================
            # Complete!
            # ================================================================
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

            logger.info(
                f"[{job_id}] ✓ Ingestion completed in {processing_time:.2f}s:\n"
                f"    - Document ID: {document_id}\n"
                f"    - Chunks: {len(chunks)}\n"
                f"    - Embeddings: {embeddings_count}\n"
                f"    - Qdrant vectors: {qdrant_count}\n"
                f"    - Pages: {extraction_result.pages}\n"
                f"    - Tables: {extraction_result.tables}"
            )

        except Exception as e:
            logger.error(f"[{job_id}] ✗ Ingestion failed: {e}", exc_info=True)

            await self.job_repo.mark_failed(
                job_id=job_id,
                error_message=str(e),
                error_type=type(e).__name__
            )

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
                            "https://mqrph4kl9d2186-8000.proxy.runpod.net/v1/chat/completions"
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

    async def _save_intermediate(
            self,
            job_id: str,
            document_id: str,
            filename: str,
            extraction_result,
            chunks: List[Chunk],
            qdrant_count: int = 0,
    ) -> Path:
        """
        Save intermediate processing results to JSON.

        Enhanced to include Qdrant integration info and
        richer metadata for debugging.
        """
        intermediate_data = {
            "job_id": job_id,
            "document_id": document_id,
            "filename": filename,
            "timestamp": datetime.utcnow().isoformat(),

            # Extraction results
            "extraction": extraction_result.to_dict(),

            # Chunks (without embeddings to save space)
            "chunks": [
                {
                    "index": chunk.index,
                    "content": chunk.content,
                    "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                    "size": len(chunk.content),
                    "page_number": chunk.page_number if hasattr(chunk, 'page_number') else None,
                    "section_title": chunk.section_title if hasattr(chunk, 'section_title') else None,
                    "has_embedding": chunk.embedding is not None,
                    "embedding_dim": len(chunk.embedding) if chunk.embedding else None,
                    "metadata": chunk.metadata,
                }
                for chunk in chunks
            ],

            # Statistics
            "stats": {
                "total_chunks": len(chunks),
                "avg_chunk_size": sum(len(c.content) for c in chunks) / len(chunks) if chunks else 0,
                "chunks_with_embeddings": sum(1 for c in chunks if c.embedding is not None),
                "qdrant_vectors_stored": qdrant_count,
                "total_pages": extraction_result.pages,
                "total_tables": extraction_result.tables,
                "total_images": extraction_result.images,
            },

            # For chatbot citation verification
            "citation_preview": [
                {
                    "chunk_index": i,
                    "source": filename,
                    "page": chunks[i].page_number if hasattr(chunks[i], 'page_number') else None,
                    "preview": chunks[i].content[:100] + "...",
                }
                for i in range(min(5, len(chunks)))
            ]
        }

        # Save to file
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