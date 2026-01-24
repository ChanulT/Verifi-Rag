"""
Shared Pydantic schemas for inter-service communication.
These schemas define the contract between microservices.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import uuid


# =============================================================================
# Enums
# =============================================================================

class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    IMAGE = "image"
    TEXT = "text"


class ProcessingStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""
    SEMANTIC = "semantic"
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    HYBRID = "hybrid"


# =============================================================================
# Embedding Service Schemas
# =============================================================================

class EmbeddingRequest(BaseModel):
    """Request schema for embedding generation."""
    texts: List[str] = Field(..., min_length=1, description="List of texts to embed")
    model_name: Optional[str] = Field(None, description="Optional model override")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "texts": ["LDL cholesterol: 120 mg/dL", "HDL cholesterol: 55 mg/dL"],
            "model_name": None
        }
    })


class EmbeddingResponse(BaseModel):
    """Response schema for embedding generation."""
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    model_name: str = Field(..., description="Model used for embedding")
    dimension: int = Field(..., description="Embedding dimension")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


# =============================================================================
# Document & Chunk Schemas
# =============================================================================

class ChunkMetadata(BaseModel):
    """Metadata for a document chunk."""
    document_id: str = Field(..., description="Parent document ID")
    document_name: str = Field(..., description="Original document name")
    page_number: int = Field(..., ge=1, description="Page number (1-indexed)")
    chunk_index: int = Field(..., ge=0, description="Chunk index within page")
    total_chunks_in_page: int = Field(..., ge=1, description="Total chunks in this page")
    start_char: Optional[int] = Field(None, description="Start character position")
    end_char: Optional[int] = Field(None, description="End character position")
    contains_table: bool = Field(False, description="Whether chunk contains table data")
    contains_image_text: bool = Field(False, description="Whether chunk is from OCR'd image")
    extraction_method: str = Field(..., description="Method used for text extraction")
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="OCR confidence")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentChunk(BaseModel):
    """A processed document chunk with metadata."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(..., min_length=1, description="Chunk text content")
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = Field(None, description="Embedding vector")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "id": "chunk_abc123",
            "content": "LDL Cholesterol: 142 mg/dL (High)",
            "metadata": {
                "document_id": "doc_123",
                "document_name": "blood_test_2023.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "total_chunks_in_page": 5,
                "contains_table": True,
                "contains_image_text": False,
                "extraction_method": "pdfplumber",
                "confidence_score": 0.95
            }
        }
    })


class ProcessedDocument(BaseModel):
    """A fully processed document with all chunks."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Original file name")
    document_type: DocumentType
    total_pages: int = Field(..., ge=1)
    chunks: List[DocumentChunk] = Field(default_factory=list)
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    processing_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    session_id: str = Field(..., description="User session ID")


# =============================================================================
# Ingestion Service Schemas
# =============================================================================

class IngestionRequest(BaseModel):
    """Request to ingest a document."""
    session_id: str = Field(..., description="User session identifier")
    file_name: str = Field(..., description="Original file name")
    file_content_base64: str = Field(..., description="Base64 encoded file content")
    chunking_strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.HYBRID,
        description="Strategy for chunking the document"
    )
    chunk_size: int = Field(default=512, ge=100, le=2000, description="Target chunk size")
    chunk_overlap: int = Field(default=50, ge=0, le=500, description="Overlap between chunks")


class IngestionResponse(BaseModel):
    """Response from document ingestion."""
    document_id: str
    status: ProcessingStatus
    total_chunks: int
    total_pages: int
    processing_time_ms: float
    message: str


class IngestionStatusResponse(BaseModel):
    """Status of an ingestion job."""
    document_id: str
    status: ProcessingStatus
    progress_percent: float = Field(ge=0, le=100)
    chunks_processed: int
    error_message: Optional[str] = None


# =============================================================================
# Search Service Schemas
# =============================================================================

class SearchRequest(BaseModel):
    """Request to search documents."""
    query: str = Field(..., min_length=1, description="Search query")
    session_id: str = Field(..., description="Session ID to scope search")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")
    min_score: float = Field(default=0.5, ge=0, le=1, description="Minimum similarity score")
    use_hybrid: bool = Field(default=True, description="Use hybrid search (vector + BM25)")
    filter_document_ids: Optional[List[str]] = Field(None, description="Filter by document IDs")


class SearchResult(BaseModel):
    """A single search result."""
    chunk_id: str
    content: str
    score: float = Field(ge=0, le=1, description="Relevance score")
    metadata: ChunkMetadata
    highlight: Optional[str] = Field(None, description="Highlighted matching text")


class SearchResponse(BaseModel):
    """Response from search service."""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: float
    search_type: str = Field(description="vector, bm25, or hybrid")


# =============================================================================
# Chat Service Schemas
# =============================================================================

class Citation(BaseModel):
    """A citation for an answer."""
    document_name: str
    page_number: int
    chunk_id: str
    source_text: str = Field(..., description="Original text from the document")
    relevance_score: float


class ChatMessage(BaseModel):
    """A message in the chat history."""
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str
    citations: Optional[List[Citation]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    """Request to the chat service."""
    session_id: str = Field(..., description="User session identifier")
    message: str = Field(..., min_length=1, description="User message")
    llm_provider: LLMProvider = Field(default=LLMProvider.OPENAI)
    model_name: Optional[str] = Field(None, description="Specific model to use")
    temperature: float = Field(default=0.1, ge=0, le=1, description="LLM temperature")
    max_tokens: int = Field(default=1024, ge=100, le=4096)
    chat_history: Optional[List[ChatMessage]] = Field(default=None)


class ChatResponse(BaseModel):
    """Response from the chat service."""
    session_id: str
    answer: str = Field(..., description="Generated answer")
    citations: List[Citation] = Field(default_factory=list, description="Source citations")
    confidence: str = Field(
        ...,
        description="Confidence level: high, medium, low, or insufficient_evidence"
    )
    retrieved_chunks: List[SearchResult] = Field(
        default_factory=list,
        description="Chunks used for answer generation"
    )
    processing_time_ms: float
    llm_provider: LLMProvider
    model_used: str


class SessionDocuments(BaseModel):
    """Documents in a user session."""
    session_id: str
    documents: List[ProcessedDocument]
    total_chunks: int
    created_at: datetime


# =============================================================================
# Health Check Schemas
# =============================================================================

class HealthStatus(BaseModel):
    """Health check response."""
    service: str
    status: str = Field(pattern="^(healthy|unhealthy|degraded)$")
    version: str
    uptime_seconds: float
    dependencies: Dict[str, str] = Field(default_factory=dict)


# =============================================================================
# Error Schemas
# =============================================================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
