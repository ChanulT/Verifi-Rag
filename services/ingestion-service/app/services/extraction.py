"""
PDF Extraction Service

Handles PDF content extraction using Docling.
Follows the async pattern from embedding-service/service.py

Key features:
- Async extraction (non-blocking)
- Table and image extraction
- Structured results
- Error handling
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

# # Docling imports
# from docling.document_converter import DocumentConverter, PdfFormatOption
# from docling.datamodel.base_models import InputFormat
# from docling.datamodel.pipeline_options import PdfPipelineOptions

from app.models import ExtractionResult

logger = logging.getLogger(__name__)

__all__ = ["ExtractionService"]


class ExtractionService:
    """
    Service for PDF content extraction.

    Uses Docling for high-quality extraction with table and image support.
    Follows the async pattern from embedding-service to prevent blocking.

    Example:
        service = ExtractionService(
            enable_ocr=False,
            include_tables=True,
            include_images=True
        )
        result = await service.extract(pdf_path)
    """

    def __init__(
            self,
            enable_ocr: bool = False,
            images_scale: float = 1.0,
            include_images: bool = True,
            include_tables: bool = True
    ):
        """
        Initialize extraction service.

        Args:
            enable_ocr: Enable OCR for scanned PDFs
            images_scale: Image scaling factor (0.5-4.0)
            include_images: Extract images
            include_tables: Extract tables
        """
        self.enable_ocr = enable_ocr
        self.images_scale = images_scale
        self.include_images = include_images
        self.include_tables = include_tables

        # Initialize converter
        self._setup_converter()

        logger.info(
            f"ExtractionService initialized: "
            f"ocr={enable_ocr}, tables={include_tables}, images={include_images}"
        )

    def _setup_converter(self):
        """Setup Docling document converter with options."""
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = self.enable_ocr
        pipeline_options.do_picture_description = self.include_images
        pipeline_options.do_table_structure = self.include_tables
        pipeline_options.images_scale = self.images_scale

        try:
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options
                    )
                }
            )
            logger.info("✓ Docling converter initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Docling converter: {e}")
            raise

    async def extract(self, pdf_path: str) -> ExtractionResult:
        """
        Extract content from PDF file (async, non-blocking).

        This method uses asyncio.run_in_executor() to run the blocking
        Docling conversion in a separate thread, preventing it from
        blocking the event loop.

        This is the same pattern used in embedding-service/service.py
        for the model encoding.

        Args:
            pdf_path: Path to PDF file

        Returns:
            ExtractionResult with content and metadata

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If extraction fails
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Extracting content from: {pdf_path.name}")

        # Run blocking extraction in executor (like embedding service does)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._extract_sync,
            str(pdf_path)
        )

        logger.info(
            f"✓ Extraction complete: {result.pages} pages, "
            f"{result.tables} tables, {result.images} images "
            f"({result.processing_time_seconds:.2f}s)"
        )

        return result

    def _extract_sync(self, pdf_path: str) -> ExtractionResult:
        """
        Synchronous extraction implementation.

        This is called by extract() via run_in_executor().
        Separated to keep async/sync code clear.

        Args:
            pdf_path: Path to PDF file

        Returns:
            ExtractionResult
        """
        start_time = time.time()

        # Convert PDF using Docling
        result = self.converter.convert(pdf_path)
        doc = result.document

        # Extract content as Markdown
        content_text = doc.export_to_markdown()

        # Build metadata
        metadata = {
            "source": pdf_path,
            "title": Path(pdf_path).stem,
            "pages": len(doc.pages),
            "texts": len(doc.texts),
            "pictures": len(doc.pictures),
            "tables": len(doc.tables),
            "extraction_method": "docling",
            "content_type": "pdf",
            "ocr_enabled": self.enable_ocr,
        }

        processing_time = time.time() - start_time

        return ExtractionResult(
            content=content_text,
            metadata=metadata,
            pages=len(doc.pages),
            tables=len(doc.tables),
            images=len(doc.pictures),
            processing_time_seconds=processing_time
        )

    def update_config(
            self,
            enable_ocr: Optional[bool] = None,
            images_scale: Optional[float] = None,
            include_images: Optional[bool] = None,
            include_tables: Optional[bool] = None
    ):
        """
        Update extraction configuration.

        Reinitializes the converter with new settings.

        Args:
            enable_ocr: Enable OCR
            images_scale: Image scaling factor
            include_images: Extract images
            include_tables: Extract tables
        """
        if enable_ocr is not None:
            self.enable_ocr = enable_ocr
        if images_scale is not None:
            self.images_scale = images_scale
        if include_images is not None:
            self.include_images = include_images
        if include_tables is not None:
            self.include_tables = include_tables

        # Reinitialize converter
        self._setup_converter()

        logger.info("Extraction configuration updated")