"""
Custom exceptions for the Verifiable RAG system.
Provides consistent error handling across all microservices.
"""

from typing import Optional, Dict, Any


class BaseRAGException(Exception):
    """Base exception for all RAG system errors."""

    def __init__(
            self,
            message: str,
            error_code: str,
            details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details
        }


# =============================================================================
# Embedding Service Exceptions
# =============================================================================

class EmbeddingException(BaseRAGException):
    """Base exception for embedding service errors."""
    pass


class ModelLoadError(EmbeddingException):
    """Raised when the embedding model fails to load."""

    def __init__(self, model_name: str, reason: str):
        super().__init__(
            message=f"Failed to load embedding model '{model_name}': {reason}",
            error_code="EMBEDDING_MODEL_LOAD_ERROR",
            details={"model_name": model_name, "reason": reason}
        )


class EmbeddingGenerationError(EmbeddingException):
    """Raised when embedding generation fails."""

    def __init__(self, reason: str, text_count: int):
        super().__init__(
            message=f"Failed to generate embeddings for {text_count} texts: {reason}",
            error_code="EMBEDDING_GENERATION_ERROR",
            details={"text_count": text_count, "reason": reason}
        )


# =============================================================================
# Ingestion Service Exceptions
# =============================================================================

class IngestionException(BaseRAGException):
    """Base exception for ingestion service errors."""
    pass


class DocumentParsingError(IngestionException):
    """Raised when document parsing fails."""

    def __init__(self, document_name: str, reason: str):
        super().__init__(
            message=f"Failed to parse document '{document_name}': {reason}",
            error_code="DOCUMENT_PARSING_ERROR",
            details={"document_name": document_name, "reason": reason}
        )


class OCRError(IngestionException):
    """Raised when OCR processing fails."""

    def __init__(self, page_number: int, reason: str):
        super().__init__(
            message=f"OCR failed on page {page_number}: {reason}",
            error_code="OCR_ERROR",
            details={"page_number": page_number, "reason": reason}
        )


class UnsupportedDocumentError(IngestionException):
    """Raised for unsupported document types."""

    def __init__(self, document_type: str):
        super().__init__(
            message=f"Unsupported document type: {document_type}",
            error_code="UNSUPPORTED_DOCUMENT_TYPE",
            details={"document_type": document_type}
        )


class ChunkingError(IngestionException):
    """Raised when document chunking fails."""

    def __init__(self, document_id: str, reason: str):
        super().__init__(
            message=f"Failed to chunk document '{document_id}': {reason}",
            error_code="CHUNKING_ERROR",
            details={"document_id": document_id, "reason": reason}
        )


# =============================================================================
# Search Service Exceptions
# =============================================================================

class SearchException(BaseRAGException):
    """Base exception for search service errors."""
    pass


class VectorDBConnectionError(SearchException):
    """Raised when connection to vector DB fails."""

    def __init__(self, db_type: str, reason: str):
        super().__init__(
            message=f"Failed to connect to {db_type}: {reason}",
            error_code="VECTOR_DB_CONNECTION_ERROR",
            details={"db_type": db_type, "reason": reason}
        )


class IndexNotFoundError(SearchException):
    """Raised when the search index is not found."""

    def __init__(self, index_name: str):
        super().__init__(
            message=f"Search index '{index_name}' not found",
            error_code="INDEX_NOT_FOUND",
            details={"index_name": index_name}
        )


class SearchQueryError(SearchException):
    """Raised when a search query is invalid."""

    def __init__(self, query: str, reason: str):
        super().__init__(
            message=f"Invalid search query: {reason}",
            error_code="SEARCH_QUERY_ERROR",
            details={"query": query, "reason": reason}
        )


# =============================================================================
# Chat Service Exceptions
# =============================================================================

class ChatException(BaseRAGException):
    """Base exception for chat service errors."""
    pass


class LLMProviderError(ChatException):
    """Raised when LLM provider fails."""

    def __init__(self, provider: str, reason: str):
        super().__init__(
            message=f"LLM provider '{provider}' error: {reason}",
            error_code="LLM_PROVIDER_ERROR",
            details={"provider": provider, "reason": reason}
        )


class InsufficientContextError(ChatException):
    """Raised when there's not enough context to answer."""

    def __init__(self, query: str, chunks_retrieved: int):
        super().__init__(
            message="Insufficient context to answer the query",
            error_code="INSUFFICIENT_CONTEXT",
            details={"query": query, "chunks_retrieved": chunks_retrieved}
        )


class SessionNotFoundError(ChatException):
    """Raised when a session is not found."""

    def __init__(self, session_id: str):
        super().__init__(
            message=f"Session '{session_id}' not found",
            error_code="SESSION_NOT_FOUND",
            details={"session_id": session_id}
        )


class RateLimitError(ChatException):
    """Raised when rate limit is exceeded."""

    def __init__(self, provider: str, retry_after: Optional[int] = None):
        super().__init__(
            message=f"Rate limit exceeded for {provider}",
            error_code="RATE_LIMIT_EXCEEDED",
            details={"provider": provider, "retry_after": retry_after}
        )


# =============================================================================
# Service Communication Exceptions
# =============================================================================

class ServiceCommunicationError(BaseRAGException):
    """Raised when inter-service communication fails."""

    def __init__(self, source_service: str, target_service: str, reason: str):
        super().__init__(
            message=f"Communication from {source_service} to {target_service} failed: {reason}",
            error_code="SERVICE_COMMUNICATION_ERROR",
            details={
                "source_service": source_service,
                "target_service": target_service,
                "reason": reason
            }
        )


class ServiceUnavailableError(BaseRAGException):
    """Raised when a required service is unavailable."""

    def __init__(self, service_name: str):
        super().__init__(
            message=f"Service '{service_name}' is currently unavailable",
            error_code="SERVICE_UNAVAILABLE",
            details={"service_name": service_name}
        )