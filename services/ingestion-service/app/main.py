"""
Ingestion Service - Main Application

A FastAPI microservice for document ingestion with:
- PDF extraction using Docling
- Intelligent chunking (semantic/recursive)
- Embedding generation using local embedding service
- Job tracking and progress monitoring
- Intermediate JSON file generation

Follows the same architectural patterns as embedding-service/main.py

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Local imports (clean imports with proper package structure)
from app.configs import settings_manager, load_settings_from_env
from app.services.extraction import ExtractionService
from app.services.chunking import ChunkingService
from app.services.embedding import EmbeddingService
from app.providers.embedding import get_embedding_provider
from app.orchestrators.ingestion import IngestionOrchestrator

# Import routes (will be created next)
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

# These will be initialized during startup
orchestrator: IngestionOrchestrator = None


# =============================================================================
# Application Lifecycle
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.

    Startup:
    1. Load configuration
    2. Initialize services
    3. Setup database (if configured)
    4. Verify embedding service connection

    Shutdown:
    1. Close database connections
    2. Cleanup resources

    Mirrors the pattern from embedding-service/main.py
    """
    global orchestrator, db_pool

    logger.info("=" * 70)
    logger.info("INGESTION SERVICE STARTUP")
    logger.info("=" * 70)

    # Load settings
    settings = load_settings_from_env()
    settings_manager.update(settings)

    logger.info("Configuration loaded:")
    logger.info(f"  ├─ Embedding Provider: {settings.embedding_provider}")
    logger.info(f"  ├─ Embedding Service: {settings.embedding_service_url}")
    logger.info(f"  ├─ Chunk Size: {settings.chunk_size}")
    logger.info(f"  ├─ Semantic Chunking: {'enabled' if settings.use_semantic_chunking else 'disabled'}")
    logger.info(f"  ├─ OCR: {'enabled' if settings.enable_ocr else 'disabled'}")
    logger.info(f"  ├─ Tables: {'enabled' if settings.include_tables else 'disabled'}")
    logger.info(f"  └─ Max File Size: {settings.max_file_size_mb}MB")

    # Create required directories
    settings_manager.ensure_directories()
    logger.info(f"✓ Directories created:")
    logger.info(f"  ├─ Upload: {settings.upload_dir}")
    logger.info(f"  ├─ Cache: {settings.cache_dir}")
    logger.info(f"  └─ Intermediate: {settings.cache_dir}/intermediate")

    # Initialize embedding provider
    try:
        logger.info(f"Initializing embedding provider: {settings.embedding_provider}")

        if settings.embedding_provider == "local":
            embedding_provider = get_embedding_provider(
                "local",
                base_url=settings.embedding_service_url,
                timeout=settings.embedding_timeout_seconds
            )

            # Check if embedding service is available
            is_healthy = await embedding_provider.health_check()
            if is_healthy:
                logger.info(f"✓ Embedding service is available: {settings.embedding_service_url}")
            else:
                logger.warning(
                    f"⚠ Embedding service is not responding: {settings.embedding_service_url}\n"
                    f"  Semantic chunking may fail. Please ensure your embedding service is running."
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

    # Initialize services
    try:
        logger.info("Initializing services...")

        # Extraction service (choose between Docling, LightOnOCR, and Unstructured)
        if settings.extraction_service == "lighton_ocr":
            logger.info("Using LightOnOCR for PDF extraction")
            from app.ocr_extraction import LightOnOCRService

            # Remote RunPod/vLLM endpoint (expects OpenAI-compatible /v1/chat/completions)
            endpoint_url = os.getenv(
                "OCR_ENDPOINT_URL",
                "https://9ldkyfwmuf18is-8000.proxy.runpod.net/v1/chat/completions",
            )

            extraction_service = LightOnOCRService(
                endpoint_url=endpoint_url,
                dpi=settings.ocr_dpi,
            )

            # Initialize client (health/reachability)
            logger.info("Initializing remote OCR endpoint...")
            await extraction_service.initialize()
            logger.info("✓ LightOnOCRService initialized")

        elif settings.extraction_service == "unstructured":
            logger.info("Using UnstructuredService for PDF extraction")
            from app.services.unstructured_extraction import UnstructuredService, UnstructuredServiceOptimized

            if settings.enable_ocr:
                # Use Unstructured's OCR path for scanned PDFs
                logger.info("Unstructured configured for OCR (ocr_only strategy)")
                extraction_service = UnstructuredService(
                    strategy="ocr_only",
                    extract_tables=True,
                    extract_images=False,
                    languages=["eng"],
                    ocr_languages=["eng"],
                    use_chunking=False,
                )
            else:
                # Default optimized extractor for native PDFs
                extraction_service = UnstructuredServiceOptimized()

            logger.info("✓ UnstructuredService initialized")

        else:
            # Default: Docling
            logger.info("Using Docling for PDF extraction")
            from app.services.extraction import ExtractionService

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
            batch_size=32  # Can be made configurable
        )
        logger.info("✓ EmbeddingService initialized")

    except Exception as e:
        logger.error(f"✗ Failed to initialize services: {e}")
        raise

    # Initialize database
    logger.info("=" * 70)
    logger.info("DATABASE INITIALIZATION")
    logger.info("=" * 70)


    from app.repositories.document import DocumentRepository
    from app.repositories.job import JobRepository

    # Check if database is configured
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        logger.warning(
            "⚠ DATABASE_URL not configured!\n"
            "  Using in-memory storage (data will be lost on restart)\n"
            "  For production, set DATABASE_URL in .env file\n"
            "  Example: DATABASE_URL=postgresql://user:pass@localhost/ingestion_db"
        )

        # Use in-memory storage
        from app.storage import InMemoryJobStore, InMemoryDocumentStore
        job_repo = InMemoryJobStore()
        document_repo = InMemoryDocumentStore()
        logger.info("✓ Using in-memory storage (demo mode)")

    else:
        # Use PostgreSQL
        logger.info(f"Database URL: {database_url.split('@')[0]}@***")

        db_manager = DatabaseManager(
            database_url=database_url,
            min_size=int(os.getenv("DB_POOL_MIN_SIZE", "10")),
            max_size=int(os.getenv("DB_POOL_MAX_SIZE", "20"))
        )

        try:
            # Initialize connection pool
            await db_manager.initialize()

            # Run schema migrations if requested
            if os.getenv("AUTO_MIGRATE", "false").lower() == "true":
                logger.info("Running database migrations...")
                schema_path = Path("sql/schema.sql")
                if schema_path.exists():
                    with open(schema_path, 'r') as f:
                        schema_sql = f.read()
                    await db_manager.execute_script(schema_sql)
                    logger.info("✓ Database schema initialized")
                else:
                    logger.warning(f"Schema file not found: {schema_path}")

            # Set global database manager
            set_database_manager(db_manager)

            # Create repositories
            job_repo = JobRepository(db_manager)
            document_repo = DocumentRepository(db_manager)

            logger.info("✓ PostgreSQL initialized successfully")

        except Exception as e:
            logger.error(f"✗ Failed to initialize PostgreSQL: {e}")
            logger.warning("Falling back to in-memory storage")

            # Fallback to in-memory
            from app.storage import InMemoryJobStore, InMemoryDocumentStore
            job_repo = InMemoryJobStore()
            document_repo = InMemoryDocumentStore()

    logger.info("=" * 70)

    # Initialize orchestrator
    orchestrator = IngestionOrchestrator(
        extraction_service=extraction_service,
        chunking_service=chunking_service,
        embedding_service=embedding_service,
        document_repo=document_repo,
        job_repo=job_repo,
        cache_dir=settings.cache_dir
    )
    logger.info("✓ IngestionOrchestrator initialized")

    logger.info("=" * 70)
    logger.info(f"SERVICE READY: http://{settings.host}:{settings.port}")
    logger.info(f"API Documentation: http://{settings.host}:{settings.port}/docs")
    logger.info(f"Health Check: http://{settings.host}:{settings.port}/health")
    logger.info("=" * 70)

    # Application is now running
    yield

    # Shutdown
    logger.info("=" * 70)
    logger.info("INGESTION SERVICE SHUTDOWN")
    logger.info("=" * 70)

    # Close embedding provider
    if hasattr(embedding_provider, 'close'):
        await embedding_provider.close()
        logger.info("✓ Embedding provider closed")

    database_url = os.getenv("DATABASE_URL")
    if database_url:
        try:
            from app.database import get_database_manager
            db_manager = get_database_manager()
            if db_manager:
                await db_manager.close()
                logger.info("✓ Database connections closed")
        except ImportError:
            logger.warning("Database module not found, skipping DB shutdown.")
    else:
        logger.info("✓ In-memory storage used; no database connections to close.")

    logger.info("=" * 70)
    logger.info("SHUTDOWN COMPLETE")
    logger.info("=" * 70)


# =============================================================================
# FastAPI Application
# =============================================================================

# Load initial settings for app metadata
_initial_settings = load_settings_from_env()

app = FastAPI(
    title="Ingestion Service",
    description="""
    ## Production-Ready Document Ingestion Microservice
    
    Process documents with advanced extraction and chunking for RAG systems.
    
    ### Key Features
    - 🔍 **PDF Extraction**: High-quality extraction using Docling
    - 📊 **Table Preservation**: Maintains table structure
    - 🧩 **Smart Chunking**: Semantic and recursive strategies
    - 🔗 **Local Embeddings**: Integrates with your embedding microservice
    - 📁 **Intermediate Files**: Inspect results before vector DB
    - 📈 **Job Tracking**: Real-time progress monitoring
    
    ### Endpoints
    
    **Document Processing**
    - `POST /ingest` - Upload and process document
    - `GET /jobs/{job_id}` - Check processing status
    - `GET /jobs/{job_id}/chunks` - Retrieve processed chunks
    - `GET /jobs/{job_id}/intermediate` - Download intermediate JSON
    
    **Monitoring**
    - `GET /health` - Service health status
    - `GET /metrics` - Processing metrics
    
    ### Integration with Embedding Service
    
    This service integrates seamlessly with your embedding microservice
    running on port 8001. Set `EMBEDDING_PROVIDER=local` in your environment
    to use your local service instead of OpenAI.
    
    ```bash
    # Use your local embedding service (recommended!)
    export EMBEDDING_PROVIDER=local
    export EMBEDDING_SERVICE_URL=http://localhost:8001
    
    # Or use OpenAI (fallback)
    export EMBEDDING_PROVIDER=openai
    export OPENAI_API_KEY=sk-...
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
app.include_router(routes.router)


# =============================================================================
# Root Endpoint
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with service information.

    Returns:
        Service metadata and useful links
    """
    settings = settings_manager.current

    # Get embedding service status
    embedding_status = "unknown"
    if orchestrator and orchestrator.embedding_service:
        is_healthy = await orchestrator.embedding_service.health_check()
        embedding_status = "healthy" if is_healthy else "unavailable"

    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "status": "running",
        "documentation": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "embedding_provider": settings.embedding_provider,
        "embedding_service": settings.embedding_service_url,
        "embedding_service_status": embedding_status,
        "chunking_strategy": "semantic" if settings.use_semantic_chunking else "recursive",
    }


# =============================================================================
# Development Server
# =============================================================================

if __name__ == "__main__":
    import os
    import uvicorn

    settings = load_settings_from_env()

    # Determine environment
    is_development = os.getenv("ENVIRONMENT", "development") == "development"

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=is_development,
        workers=1,  # Important: Docling is not fork-safe
        log_level="info",
        access_log=True,
    )