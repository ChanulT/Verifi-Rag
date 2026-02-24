"""
Ingestion Service - Main Application
"""
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.shared.configs import settings_manager, load_settings_from_env
from app.libr.chunking import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.libr.providers.embedding import get_embedding_provider
from app.libr.qdrant import QdrantRepository, QdrantConfig
from app.ingestion.service import IngestionOrchestrator
from app.database import AsyncSessionLocal, engine, Base
from app.repositories.postgres import PostgresJobRepository, PostgresDocumentRepository
from app.ingestion.router import router as ingestion_router
from app.libr.providers.embedding import EmbeddingProvider
from app.libr.ocr_extraction import LightOnOCRService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

orchestrator: Optional[IngestionOrchestrator] = None
qdrant_repo: Optional[QdrantRepository] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator, qdrant_repo

    settings = load_settings_from_env()
    settings_manager.update(settings)
    settings_manager.ensure_directories()

    # Embedding provider
    if settings.embedding_provider == "local":
        embedding_provider = get_embedding_provider(
            "local", base_url=settings.embedding_service_url, timeout=settings.embedding_timeout_seconds
        )
    elif settings.embedding_provider == "OpenAI":
        embedding_provider = get_embedding_provider(
            "openai", api_key=settings.openai_api_key, model=settings.openai_model
        )
    else:
        raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")

    # Qdrant
    qdrant_repo = None
    if settings.enable_qdrant:
        try:
            qdrant_repo = QdrantRepository(QdrantConfig(
                url=settings.qdrant_url, api_key=settings.qdrant_api_key,
                collection_name=settings.qdrant_collection_name,
                vector_size=settings.embedding_dimension,
                distance=settings.qdrant_distance, timeout=settings.qdrant_timeout,
            ))
            await qdrant_repo.initialize()
            logger.info("✓ Qdrant initialized")
        except Exception as e:
            logger.warning(f"Qdrant unavailable: {e}")
            qdrant_repo = None

    # Extraction service
    if settings.extraction_service == "lighton_ocr":
        extraction_service = LightOnOCRService(
            endpoint_url=os.getenv("OCR_ENDPOINT_URL", ""),
            dpi=settings.ocr_dpi,
        )
        await extraction_service.initialize()

    # Services
    chunking_service = ChunkingService(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap,
        min_chunk_size=settings.min_chunk_size, max_chunk_size=settings.max_chunk_size,
        use_semantic=settings.use_semantic_chunking,
        embedding_provider=embedding_provider if settings.use_semantic_chunking else None
    )
    embedding_service = EmbeddingService(provider=embedding_provider, batch_size=32)


    job_repo = PostgresJobRepository(AsyncSessionLocal)
    document_repo = PostgresDocumentRepository(AsyncSessionLocal)

    # Orchestrator
    orchestrator = IngestionOrchestrator(
        extraction_service=extraction_service,
        chunking_service=chunking_service,
        embedding_service=embedding_service,
        document_repo=document_repo,
        job_repo=job_repo,
        qdrant_repo=qdrant_repo,
        cache_dir=settings.cache_dir
    )
    logger.info("✓ Service ready")

    yield

    # Shutdown
    await engine.dispose()
    if qdrant_repo:
        await qdrant_repo.close()
    if hasattr(embedding_provider, 'close'):
        await embedding_provider.close()


_settings = load_settings_from_env()

app = FastAPI(
    title="Ingestion Service",
    version=_settings.service_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingestion_router)


@app.get("/", tags=["Root"])
async def root():
    settings = settings_manager.current
    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "status": "running",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    settings = load_settings_from_env()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port,
                reload=os.getenv("ENVIRONMENT", "development") == "development")