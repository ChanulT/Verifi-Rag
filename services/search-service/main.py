import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv

# Import our modules
from qdrantc import QdrantRepository
from vector import VectorSearchService
from routes_search import router as search_router

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load env vars
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to initialize services on startup.
    This replaces the 'app.main' dependency.
    """
    logger.info("Starting up Search Service...")

    # 1. Initialize Qdrant Repo
    qdrant_repo = QdrantRepository()

    # 2. Initialize Vector Service
    vector_service = VectorSearchService(qdrant_repo=qdrant_repo)

    # 3. Inject into app state so routes can access it
    app.state.vector_service = vector_service

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="Vector Search Service API",
    description="Standalone API for semantic search",
    version="1.0.0",
    lifespan=lifespan
)

# Register Routes
app.include_router(search_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)