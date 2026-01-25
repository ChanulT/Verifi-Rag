"""
Ingestion Service - Main Application

A FastAPI microservice for document ingestion with:
- PDF extraction (Docling/LightOnOCR/Unstructured)
- Intelligent chunking (semantic/recursive)
- Embedding generation using local embedding service
- Vector storage in Qdrant (NEW!)
- Semantic search API (NEW!)
- Job tracking and progress monitoring
- Intermediate JSON file generation

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Local imports
from app.configs import settings_manager, load_settings_from_env
from app.services.chunking import ChunkingService
from app.services.embedding import EmbeddingService
from app.providers.embedding import get_embedding_provider
from app.repositories.qdrant import QdrantRepository, QdrantConfig  # NEW
from app.orchestrators.ingestion import IngestionOrchestrator
from app.database import AsyncSessionLocal, engine, Base
from app.repositories.postgres import PostgresJobRepository, PostgresDocumentRepository

# Import routes
from app import routes

# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# =============================================================================
# Global Service Instances
# =============================================================================

orchestrator: Optional[IngestionOrchestrator] = None
qdrant_repo: Optional[QdrantRepository] = None


# =============================================================================
# Application Lifecycle
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.

    Startup:
    1. Load configuration
    2. Initialize embedding provider
    3. Initialize Qdrant connection (NEW)
    4. Initialize services
    5. Initialize search service (NEW)
    6. Setup storage

    Shutdown:
    1. Close Qdrant connections
    2. Close embedding provider
    3. Cleanup resources
    """
    global orchestrator, vector_search_service, qdrant_repo

    logger.info("=" * 70)
    logger.info("INGESTION SERVICE STARTUP")
    logger.info("=" * 70)

    # Load settings
    settings = load_settings_from_env()
    settings_manager.update(settings)

    logger.info("Configuration loaded:")
    logger.info(f"  ├─ Embedding Provider: {settings.embedding_provider}")
    logger.info(f"  ├─ Embedding Service: {settings.embedding_service_url}")
    logger.info(f"  ├─ Qdrant: {settings.qdrant_url} (enabled={settings.enable_qdrant})")
    logger.info(f"  ├─ Collection: {settings.qdrant_collection_name}")
    logger.info(f"  ├─ Chunk Size: {settings.chunk_size}")
    logger.info(f"  ├─ Semantic Chunking: {'enabled' if settings.use_semantic_chunking else 'disabled'}")
    logger.info(f"  └─ Max File Size: {settings.max_file_size_mb}MB")

    # Create required directories
    settings_manager.ensure_directories()
    logger.info(f"✓ Directories created")

    # =========================================================================
    # Initialize Embedding Provider
    # =========================================================================
    try:
        logger.info(f"Initializing embedding provider: {settings.embedding_provider}")

        if settings.embedding_provider == "local":
            embedding_provider = get_embedding_provider(
                "local",
                base_url=settings.embedding_service_url,
                timeout=settings.embedding_timeout_seconds
            )

            is_healthy = await embedding_provider.health_check()
            if is_healthy:
                logger.info(f"✓ Embedding service available: {settings.embedding_service_url}")
            else:
                logger.warning(
                    f"⚠ Embedding service not responding: {settings.embedding_service_url}\n"
                    f"  Please ensure your embedding service is running."
                )

        elif settings.embedding_provider == "OpenAI":
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key not configured")

            embedding_provider = get_embedding_provider(
                "openai",
                api_key=settings.openai_api_key,
                model=settings.openai_model
            )
            logger.info(f"✓ OpenAI embeddings configured: {settings.openai_model}")

        else:
            raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")

    except Exception as e:
        logger.error(f"✗ Failed to initialize embedding provider: {e}")
        raise

    # =========================================================================
    # Initialize Qdrant (NEW)
    # =========================================================================
    if settings.enable_qdrant:
        try:
            logger.info(f"Initializing Qdrant: {settings.qdrant_url}")

            qdrant_config = QdrantConfig(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                collection_name=settings.qdrant_collection_name,
                vector_size=settings.embedding_dimension,
                distance=settings.qdrant_distance,
                timeout=settings.qdrant_timeout,
            )

            qdrant_repo = QdrantRepository(qdrant_config)
            await qdrant_repo.initialize()

            # Get collection stats
            stats = await qdrant_repo.get_collection_stats()
            logger.info(
                f"✓ Qdrant initialized: {stats.get('vectors_count', 0)} vectors "
                f"in collection '{settings.qdrant_collection_name}'"
            )

        except Exception as e:
            logger.error(f"✗ Failed to initialize Qdrant: {e}")
            logger.warning("Continuing without Qdrant - vector search will be disabled")
            qdrant_repo = None
    else:
        logger.info("Qdrant disabled by configuration")
        qdrant_repo = None

    # =========================================================================
    # Initialize Services
    # =========================================================================
    try:
        logger.info("Initializing services...")

        # Extraction service
        if settings.extraction_service == "lighton_ocr":
            logger.info("Using LightOnOCR for PDF extraction")
            from app.ocr_extraction import LightOnOCRService

            endpoint_url = os.getenv(
                "OCR_ENDPOINT_URL",
                "https://zuquhaomdlqs0z-8000.proxy.runpod.net//v1/chat/completions",
            )

            extraction_service = LightOnOCRService(
                endpoint_url=endpoint_url,
                dpi=settings.ocr_dpi,
            )
            await extraction_service.initialize()
            logger.info("✓ LightOnOCRService initialized")

        elif settings.extraction_service == "unstructured":
            logger.info("Using UnstructuredService for PDF extraction")
            from app.services.unstructured_extraction import UnstructuredService, UnstructuredServiceOptimized

            if settings.enable_ocr:
                extraction_service = UnstructuredService(
                    strategy="ocr_only",
                    extract_tables=True,
                    extract_images=False,
                    languages=["eng"],
                    ocr_languages=["eng"],
                    use_chunking=False,
                )
            else:
                extraction_service = UnstructuredServiceOptimized()
            logger.info("✓ UnstructuredService initialized")

        else:
            # Default: Docling
            logger.info("Using Docling for PDF extraction")
            extraction_service = ExtractionService(
                enable_ocr=settings.enable_ocr,
                images_scale=settings.images_scale,
                include_images=settings.include_images,
                include_tables=settings.include_tables
            )
            logger.info("✓ ExtractionService (Docling) initialized")

        # Chunking service
        chunking_service = ChunkingService(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            min_chunk_size=settings.min_chunk_size,
            max_chunk_size=settings.max_chunk_size,
            use_semantic=settings.use_semantic_chunking,
            embedding_provider=embedding_provider if settings.use_semantic_chunking else None
        )
        logger.info("✓ ChunkingService initialized")

        # Embedding service
        embedding_service = EmbeddingService(
            provider=embedding_provider,
            batch_size=32
        )
        logger.info("✓ EmbeddingService initialized")

        # Vector search service (NEW)

    except Exception as e:
        logger.error(f"✗ Failed to initialize services: {e}")
        raise

    # =========================================================================
    # Initialize Storage Repositories
    # =========================================================================
    # Use in-memory for development (replace with PostgreSQL for production)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # 2. Initialize Repositories with the session factory
    global job_repo, document_repo
    job_repo = PostgresJobRepository(AsyncSessionLocal)
    document_repo = PostgresDocumentRepository(AsyncSessionLocal)

    # =========================================================================
    # Initialize Orchestrator
    # =========================================================================
    orchestrator = IngestionOrchestrator(
        extraction_service=extraction_service,
        chunking_service=chunking_service,
        embedding_service=embedding_service,
        document_repo=document_repo,
        job_repo=job_repo,
        qdrant_repo=qdrant_repo,  # NEW
        cache_dir=settings.cache_dir
    )
    logger.info("✓ IngestionOrchestrator initialized")

    logger.info("=" * 70)
    logger.info(f"SERVICE READY: http://{settings.host}:{settings.port}")
    logger.info(f"API Documentation: http://{settings.host}:{settings.port}/docs")
    logger.info(f"Health Check: http://{settings.host}:{settings.port}/health")
    if qdrant_repo:
        logger.info(f"Search API: http://{settings.host}:{settings.port}/search")
    logger.info("=" * 70)

    # Application runs here
    yield

    await engine.dispose()

    # =========================================================================
    # Shutdown
    # =========================================================================
    logger.info("=" * 70)
    logger.info("INGESTION SERVICE SHUTDOWN")
    logger.info("=" * 70)

    # Close Qdrant
    if qdrant_repo:
        await qdrant_repo.close()
        logger.info("✓ Qdrant connections closed")

    # Close embedding provider
    if hasattr(embedding_provider, 'close'):
        await embedding_provider.close()
        logger.info("✓ Embedding provider closed")

    logger.info("=" * 70)
    logger.info("SHUTDOWN COMPLETE")
    logger.info("=" * 70)


# =============================================================================
# FastAPI Application
# =============================================================================

_initial_settings = load_settings_from_env()

app = FastAPI(
    title="Ingestion Service",
    description="""
    ## Document Ingestion Microservice with Vector Search
    
    Process documents with advanced extraction, chunking, and vector storage.
    
    ### Key Features
    - 🔍 **PDF Extraction**: Docling, LightOnOCR, or Unstructured
    - 📊 **Table Preservation**: Maintains table structure
    - 🧩 **Smart Chunking**: Semantic and recursive strategies
    - 🔗 **Local Embeddings**: Integrates with your embedding microservice
    - 🗄️ **Vector Storage**: Qdrant with rich metadata (NEW!)
    - 🔎 **Semantic Search**: Search API for RAG (NEW!)
    - 📁 **Intermediate Files**: Inspect results before committing
    
    ### Endpoints
    
    **Document Processing**
    - `POST /ingest` - Upload and process document
    - `GET /jobs/{job_id}` - Check processing status
    - `GET /jobs/{job_id}/chunks` - Retrieve processed chunks
    - `GET /jobs/{job_id}/intermediate` - Download intermediate JSON
    
    **Search (NEW)**
    - `POST /search` - Semantic search across documents
    - `GET /search/stats` - Search service statistics
    
    **Monitoring**
    - `GET /health` - Service health status
    - `GET /metrics` - Processing metrics
    
    ### Integration with Chat Service
    
    Your Chat Service can call the search endpoint:
    ```python
    response = await httpx.post(
        "http://localhost:8002/search",
        json={"query": "patient lab results", "top_k": 5}
    )
    context = response.json()["context_string"]
    citations = response.json()["citations"]
    ```
    """,
    version=_initial_settings.service_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=_initial_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(routes.router)  # NEW


# =============================================================================
# Root Endpoint
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with service information."""
    settings = settings_manager.current

    # Get service status
    embedding_status = "unknown"
    qdrant_status = "disabled"
    vectors_count = 0

    if orchestrator and orchestrator.embedding_service:
        is_healthy = await orchestrator.embedding_service.health_check()
        embedding_status = "healthy" if is_healthy else "unavailable"

    if qdrant_repo:
        is_healthy = await qdrant_repo.health_check()
        qdrant_status = "healthy" if is_healthy else "unavailable"
        if is_healthy:
            stats = await qdrant_repo.get_collection_stats()
            vectors_count = stats.get("vectors_count", 0)

    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "status": "running",
        "documentation": "/docs",
        "health": "/health",
        "search": "/search" if qdrant_repo else None,

        # Integration info
        "embedding_provider": settings.embedding_provider,
        "embedding_service": settings.embedding_service_url,
        "embedding_status": embedding_status,

        # Qdrant info
        "qdrant_enabled": settings.enable_qdrant,
        "qdrant_url": settings.qdrant_url if settings.enable_qdrant else None,
        "qdrant_status": qdrant_status,
        "qdrant_collection": settings.qdrant_collection_name,
        "vectors_indexed": vectors_count,

        # Processing info
        "chunking_strategy": "semantic" if settings.use_semantic_chunking else "recursive",
    }


# =============================================================================
# Development Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    settings = load_settings_from_env()
    is_development = os.getenv("ENVIRONMENT", "development") == "development"

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=is_development,
        workers=1,
        log_level="info",
        access_log=True,
    )