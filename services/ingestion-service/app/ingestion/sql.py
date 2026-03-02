# app/models/sql.py
from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, ForeignKey, Text, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
from app.models import JobStatus
from app.database import Base


class JobModel(Base):
    __tablename__ = "ingestion_jobs"

    job_id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    status = Column(Enum(JobStatus), default=JobStatus.PENDING)
    progress = Column(Float, default=0.0)

    # Metrics
    chunks_created = Column(Integer, default=0)
    tables_extracted = Column(Integer, default=0)  # <--- ADDED
    pages_processed = Column(Integer, default=0)  # <--- ADDED
    embeddings_generated = Column(Integer, default=0)
    processing_time_seconds = Column(Float, nullable=True)

    # Error tracking
    error_message = Column(Text, nullable=True)
    error_type = Column(String, nullable=True)  # <--- ADDED

    # File Paths
    intermediate_file = Column(String, nullable=True)  # <--- ADDED

    # JSON Metadata
    metadata_ = Column("metadata", JSON, default={})

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)  # <--- ADDED (Fixes CompileError)
    completed_at = Column(DateTime, nullable=True)


class DocumentModel(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    title = Column(String, index=True)
    content = Column(Text, nullable=False)  # The full extracted text
    created_at = Column(DateTime, default=datetime.utcnow)

    # Metadata stored as JSON
    metadata_ = Column("metadata", JSON, default={})

    # Relationship to chunks
    chunks = relationship("ChunkModel", back_populates="document", cascade="all, delete")


class ChunkModel(Base):
    __tablename__ = "chunks"

    id = Column(String, primary_key=True)  # usually "{doc_id}-{index}"
    document_id = Column(String, ForeignKey("documents.id"))
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, index=True)
    token_count = Column(Integer)
    page_number = Column(Integer, nullable=True)

    document = relationship("DocumentModel", back_populates="chunks")